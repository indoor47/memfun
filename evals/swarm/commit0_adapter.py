#!/usr/bin/env python3
"""Build Commit0 libraries into run_smoke bundles (``--suite commit0``).

Commit0 (https://github.com/commit-0/commit0, MIT) hands an agent a Python library whose
function bodies were replaced by ``pass`` plus its documentation, and grades the rebuilt
library by the library's own unit tests.  This adapter does the setup part without
Docker or Modal: it clones ``commit-0/<lib>`` from GitHub, reads the pinned
``base_commit`` / ``reference_commit`` from the public dataset
(``wentingzhao/commit0_combined``), and writes a bundle in the format ``run_smoke.py``
already grades::

    <cache>/bundles/c0-<lib>/
        task.toml     one control unit that owns every source file, hidden_test = "hidden"
        repo/         the stubbed tree at base_commit + SPEC.txt (the spec PDF as text)
        hidden/       the reference tests, plus ``.deselect`` (tests that do not pass
                      against the reference twice in the grader: failing, skipped, flaky)
        solution/     the reference source files (the grader's oracle)

Metric: % of the reference-passing unit tests that pass (``hidden_passed`` over
``hidden_passed + hidden_failed`` in run_smoke's results), like Commit0's own.

Usage::

    python evals/swarm/commit0_adapter.py build cachetools wcwidth --cache ~/.cache/c0
    python evals/swarm/run_smoke.py --suite commit0 --commit0-dir ~/.cache/c0 ...

Run it with a Python that has ``pytest``, ``pypdf`` and the libraries' test dependencies
installed: the grader runs the tests with the same interpreter.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:  # also runnable as a plain script
    sys.path.insert(0, str(_REPO_ROOT))

from evals.swarm import validate_bundles as vb  # noqa: E402

DATASET = "wentingzhao/commit0_combined"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=" + DATASET.replace("/", "%2F")
    + "&config=default&split=test&offset={offset}&length=100"
)  # fmt: skip
GIT_BASE = "https://github.com/commit-0/"
DEFAULT_CACHE = Path.home() / ".cache" / "memfun-commit0"
#: Commit0's own "lite" split (commit0/harness/constants.py SPLIT_LITE).
LITE = [
    "tinydb",
    "simpy",
    "deprecated",
    "wcwidth",
    "voluptuous",
    "cachetools",
    "imapclient",
    "marshmallow",
    "jinja",
    "cookiecutter",
    "portalocker",
    "parsel",
    "pyjwt",
    "chardet",
    "babel",
    "minitorch",
]
LIB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SPEC_NAME = "SPEC.txt"
MAX_SPEC_CHARS = 2_000_000
REFERENCE_RUNS = 2
GRADER_TIMEOUT = 900.0

#: Pytest plugin the adapter loads (from the grader's _netguard dir) to learn per-test
#: outcomes with exact node ids; ``vb.run_grader`` only reports counts.
RECORDER = """
import json, os
_phases = {}
_collect_errors = []

def pytest_runtest_logreport(report):
    _phases.setdefault(report.nodeid, {})[report.when] = report.outcome

def pytest_collectreport(report):
    if report.failed:
        _collect_errors.append(report.nodeid)

def pytest_sessionfinish(session, exitstatus):
    tests = {}
    for nodeid, ph in _phases.items():
        ok = ph.get("setup") == "passed" and ph.get("call") == "passed" and (
            ph.get("teardown", "passed") == "passed")
        tests[nodeid] = ok
    with open(os.environ["C0_RECORD"], "w") as fh:
        json.dump({"tests": tests, "collect_errors": _collect_errors}, fh)
"""


class AdapterError(Exception):
    """A library cannot be turned into a bundle (reason in the message)."""


@dataclasses.dataclass(frozen=True)
class Row:
    name: str
    original_repo: str
    base_commit: str
    reference_commit: str
    src_dir: str
    test_dir: str
    test_cmd: str
    specification: str


def _s(value: Any) -> str:
    return "" if value is None else str(value)


def row_from_dataset(raw: dict[str, Any]) -> Row:
    test = cast("dict[str, Any]", raw.get("test") or {})
    setup = cast("dict[str, Any]", raw.get("setup") or {})
    return Row(
        name=_s(raw["repo"]).split("/")[-1],
        original_repo=_s(raw.get("original_repo")),
        base_commit=_s(raw["base_commit"]),
        reference_commit=_s(raw["reference_commit"]),
        src_dir=_s(raw["src_dir"]),
        test_dir=_s(test.get("test_dir")),
        test_cmd=_s(test.get("test_cmd")),
        specification=_s(setup.get("specification")),
    )


def fetch_rows(cache: Path, refresh: bool = False) -> dict[str, Row]:
    """The dataset rows by library name (cached under ``cache/rows.json``)."""
    path = cache / "rows.json"
    if path.is_file() and not refresh:
        raw_rows = json.loads(path.read_text("utf-8"))
    else:
        raw_rows = []
        offset = 0
        while True:
            with urllib.request.urlopen(ROWS_URL.format(offset=offset), timeout=60) as resp:
                page = json.load(resp)
            rows = [r["row"] for r in page["rows"]]
            raw_rows += rows
            offset += len(rows)
            if not rows or offset >= page.get("num_rows_total", 0):
                break
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw_rows), "utf-8")
    out = {}
    for raw in raw_rows:
        row = row_from_dataset(raw)
        out[row.name] = row
    return out


def _git(args: list[str], cwd: Path | None = None, timeout: float = 600) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, timeout=timeout
    )
    return proc.stdout


def clone(name: str, cache: Path, git_base: str) -> Path:
    dest = cache / "clones" / name
    if not (dest / ".git").exists() and not (dest / "HEAD").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", "--bare", f"{git_base}{name}.git", str(dest)])
    return dest


def check_rel_dir(value: str, what: str) -> str:
    """A dataset-supplied repo-relative directory: no option-like, absolute or ``..`` path."""
    parts = PurePosixPath(value).parts
    if (
        not value
        or value.startswith("-")
        or PurePosixPath(value).is_absolute()
        or ".." in parts
        or "\\" in value
        or "\0" in value
    ):
        raise AdapterError(f"unsafe {what} {value!r}")
    return value


def export_tree(clone_dir: Path, commit: str, dest: Path, subdir: str = "") -> None:
    """``git archive`` of ``commit`` (optionally one subdirectory) into ``dest``.

    Symlinks, hard links, devices and escaping paths are refused: a bundle is data the
    grader copies around, so it must be plain files.
    """
    if not COMMIT_RE.match(commit):
        raise AdapterError(f"commit {commit!r} is not a 40-hex object id")
    args = ["archive", "--format=tar", commit]
    if subdir and subdir != ".":
        args += ["--", check_rel_dir(subdir, "subdirectory").rstrip("/") + "/"]
    data = _git(args, cwd=clone_dir)
    dest.mkdir(parents=True, exist_ok=True)
    strip = len(PurePosixPath(subdir).parts) if subdir and subdir != "." else 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.name in ("", ".") or member.name.startswith("pax_global_header"):
                continue
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                continue  # dropped, not followed (e.g. a docs symlink)
            if ".." in parts or PurePosixPath(member.name).is_absolute():
                raise AdapterError(f"unsafe path in archive: {member.name!r}")
            rel = PurePosixPath(*parts[strip:])
            if not rel.parts:
                continue
            target = dest / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                fobj = tar.extractfile(member)
                if fobj is None:  # pragma: no cover - isfile() guarantees a file object
                    continue
                target.write_bytes(fobj.read())
                target.chmod(0o644)


def source_files(tree: Path, src_dir: str, test_dir: str) -> list[str]:
    """``.py`` files under ``src_dir`` (repo-relative, posix), minus the tests."""
    root = tree / src_dir
    out: list[str] = []
    if not root.is_dir():
        return out
    tests = PurePosixPath(test_dir.rstrip("/")) if test_dir not in ("", ".") else None
    for path in sorted(root.rglob("*.py")):
        rel = PurePosixPath(path.relative_to(tree).as_posix())
        if "__pycache__" in rel.parts:
            continue
        if tests is not None and (rel == tests or tests in rel.parents):
            continue
        out.append(rel.as_posix())
    return out


def stub_count(tree: Path, files: Iterable[str]) -> int:
    """Functions whose body is only ``pass`` / a docstring / ``...``: the work to do."""
    n = 0
    for rel in files:
        try:
            module = ast.parse((tree / rel).read_text("utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    body = body[1:]
                if all(
                    isinstance(b, ast.Pass)
                    or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
                    for b in body
                ):
                    n += 1
    return n


def spec_text(bz2_pdf: Path) -> str:
    """The spec PDF as text (pypdf)."""
    import bz2

    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - environment
        raise AdapterError("pypdf is needed to turn spec.pdf.bz2 into text") from exc
    data = bz2.decompress(bz2_pdf.read_bytes())
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text[:MAX_SPEC_CHARS]


def task_spec(row: Row, src_dir: str, test_dir: str, n_stubs: int, spec_chars: int) -> str:
    return (
        f"Rebuild the Python library `{row.name}` ({row.original_repo}) from its "
        f"documentation. The source under `{src_dir}` has had the bodies of about "
        f"{n_stubs} functions and methods replaced by `pass`. Read `{SPEC_NAME}` (the "
        f"library's documentation as text, about {spec_chars // 1000}k characters: read "
        f"it in parts) and the unit tests under `{test_dir}`, then implement every stub "
        "so that the library behaves as documented and the unit tests pass. Keep the "
        "existing module layout, names and signatures. Run the tests with "
        f"`python -m pytest {test_dir} -x -q` and run only the test files you need while "
        "you work. The library has no network access in the grader."
    )


def _run_reference(bundle: Path, owned: list[str], work: Path, timeout: float) -> dict[str, Any]:
    """Run hidden/ against solution/ once, recording per-test outcomes."""
    tree = vb.graded_tree(bundle, owned, bundle / "solution", work / "tree")
    guard = work / "_netguard"
    guard.mkdir(parents=True, exist_ok=True)
    (guard / "c0rec.py").write_text(RECORDER, "utf-8")
    record = work / "record.json"
    record.unlink(missing_ok=True)
    res = vb.run_grader(
        tree,
        bundle / "hidden",
        work,
        timeout,
        extra_args=["-p", "c0rec"],
        extra_env={"C0_RECORD": str(record)},
    )
    if not record.is_file():
        raise AdapterError(
            f"reference run produced no record (rc={res.returncode}):\n{res.output_tail}"
        )
    data = cast("dict[str, Any]", json.loads(record.read_text("utf-8")))
    # Node ids are rootdir-relative or not depending on how pytest was started (plain
    # vs under another pytest/CI); store them without the grader's directory prefix.
    prefix = "graded_tests/"
    data["tests"] = {k.removeprefix(prefix): v for k, v in data["tests"].items()}
    data["collect_errors"] = [e.removeprefix(prefix) for e in data["collect_errors"]]
    data["rc"] = res.returncode
    return data


def reference_selection(
    bundle: Path, owned: list[str], work: Path, timeout: float, runs: int = REFERENCE_RUNS
) -> tuple[list[str], list[str], int]:
    """(deselect ids, files dropped for collection errors, tests kept).

    A test is kept only if it passed in every run; collection errors drop the file.
    """
    passing: set[str] | None = None
    seen: set[str] = set()
    broken: set[str] = set()
    for i in range(runs):
        data = _run_reference(bundle, owned, work / f"ref{i}", timeout)
        tests = cast("dict[str, bool]", data["tests"])
        seen |= set(tests)
        ok = {k for k, v in tests.items() if v}
        passing = ok if passing is None else passing & ok
        broken |= {e.split("::")[0] for e in cast("list[str]", data["collect_errors"])}
    keep = passing or set()
    broken_files = sorted(broken)
    deselect = sorted(n for n in seen - keep if n.split("::")[0] not in broken)
    return deselect, broken_files, len(keep)


def build_bundle(
    row: Row,
    cache: Path,
    git_base: str = GIT_BASE,
    force: bool = False,
    timeout: float = GRADER_TIMEOUT,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create ``<cache>/bundles/c0-<name>`` and return a summary dict."""
    if not LIB_RE.match(row.name):
        raise AdapterError(f"bad library name {row.name!r}")
    for commit in (row.base_commit, row.reference_commit):
        if not COMMIT_RE.match(commit):
            raise AdapterError(f"{row.name}: commit {commit!r} is not a 40-hex object id")
    src, tests = row.src_dir.rstrip("/"), row.test_dir.rstrip("/")
    if src:
        check_rel_dir(src, "src_dir")
    if tests and tests != ".":
        check_rel_dir(tests, "test_dir")
    if not src or not tests or tests == ".":
        raise AdapterError(f"{row.name}: unsupported layout (src {src!r}, tests {tests!r})")
    if tests == src or tests.startswith(src + "/"):
        raise AdapterError(f"{row.name}: tests live inside the package ({tests!r} in {src!r})")
    bundle = cache / "bundles" / f"c0-{row.name}"
    if bundle.exists():
        if not force:
            raise AdapterError(f"{bundle} exists (use --force)")
        shutil.rmtree(bundle)
    git_dir = clone(row.name, cache, git_base)
    try:
        log(
            f"[{row.name}] exporting base {row.base_commit[:8]}"
            f" / reference {row.reference_commit[:8]}"
        )
        export_tree(git_dir, row.base_commit, bundle / "repo")
        with tempfile.TemporaryDirectory(prefix=f"c0-ref-{row.name}-") as tmp_s:
            tmp = Path(tmp_s)
            export_tree(git_dir, row.reference_commit, tmp / "ref")
            export_tree(git_dir, row.reference_commit, bundle / "hidden", subdir=tests)
            base_files = source_files(bundle / "repo", src, tests)
            ref_files = source_files(tmp / "ref", src, tests)
            owned = sorted(set(base_files) | set(ref_files))
            if not owned:
                raise AdapterError(f"{row.name}: no .py files under {src!r}")
            for rel in ref_files:
                target = bundle / "solution" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp / "ref" / rel, target)
            n_stubs = stub_count(bundle / "repo", base_files)
            pdf = bundle / "repo" / "spec.pdf.bz2"
            text = spec_text(pdf) if pdf.is_file() else ""
            if text:
                (bundle / "repo" / SPEC_NAME).write_text(text, "utf-8")
            else:
                log(f"[{row.name}] WARNING: no spec.pdf.bz2; the agent gets tests only")
            if (bundle / "repo" / "src").is_dir() and not (
                bundle / "repo" / "conftest.py"
            ).exists():
                # src layout: let the agents' own pytest runs import the package.
                (bundle / "repo" / "conftest.py").write_text(
                    "import os\nimport sys\n\n"
                    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))\n",
                    "utf-8",
                )
            deselect, dropped, kept = reference_selection(bundle, owned, tmp / "refruns", timeout)
        for rel in dropped:
            path = bundle / "hidden" / rel
            if path.is_file():
                path.unlink()
        if deselect:
            (bundle / "hidden" / vb.DESELECT_FILE).write_text("\n".join(deselect) + "\n", "utf-8")
        if kept == 0:
            raise AdapterError(f"{row.name}: no test passes against the reference")
        spec = task_spec(row, src, tests, n_stubs, len(text))
        owns = ", ".join(json.dumps(o) for o in owned)
        (bundle / "task.toml").write_text(
            f'id = "c0-{row.name}"\nkind = "control"\nspec = {json.dumps(spec)}\n\n'
            f'[[units]]\nid = "lib"\nowns = [{owns}]\nhidden_test = "{vb.SUITE_UNIT}"\n\n'
            f"[source]\ndataset = {json.dumps(DATASET)}\n"
            f"repo = {json.dumps(row.original_repo)}\n"
            f"base_commit = {json.dumps(row.base_commit)}\n"
            f"reference_commit = {json.dumps(row.reference_commit)}\n"
            f"stubs = {n_stubs}\ntests_kept = {kept}\n"
            f"tests_deselected = {len(deselect)}\n"
            f"files_dropped = {json.dumps(dropped)}\n",
            "utf-8",
        )
        report = vb.validate_bundle(bundle, timeout)
        summary: dict[str, Any] = {
            "lib": row.name,
            "owned_files": len(owned),
            "stubs": n_stubs,
            "tests_kept": kept,
            "tests_deselected": len(deselect),
            "files_dropped": dropped,
            "spec_chars": len(text),
            "null_passed": (report.null or {}).get("passed"),
            "oracle": report.oracle,
            "ok": report.ok,
            "problems": report.problems,
        }
        if not report.ok:
            raise AdapterError(f"{row.name}: bundle fails validation: {report.problems}")
        # sanity: task.toml parses and says what we think
        parsed = tomllib.loads((bundle / "task.toml").read_text("utf-8"))
        summary["id"] = parsed["id"]
        return summary
    except BaseException:
        shutil.rmtree(bundle, ignore_errors=True)
        raise


# ── Function-level tasks (``split``) ─────────────────────────────────────────────
#
# One bundle per stubbed function: the rest of the library is the reference
# implementation, only the target function is ``pass``, and the unit is graded by the tests
# that actually execute that function (found with coverage contexts on the reference run).
# Small independent tasks, no planner: each ends in one code edit.

FUNC_PREFIX = "c0s"
#: A function task graded by fewer tests than this is flagged ``weak``: a wrong
#: implementation may satisfy so few assertions (the tests are the library's own, and a
#: helper is only exercised through its callers).
MAX_LISTED_TESTS = 8
WEAK_TESTS = 3


@dataclasses.dataclass(frozen=True)
class FuncSpan:
    qualname: str
    start: int  # first line, decorators included (1-based)
    end: int
    body_lines: tuple[int, ...]  # first line of every body statement (docstring excluded)
    is_stub: bool


def _is_stub_body(body: list[ast.stmt]) -> bool:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return all(
        isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
        for b in body
    )


_NESTED_BODIES = ("body", "orelse", "handlers", "finalbody")


def function_spans(source: str, duplicates: list[str] | None = None) -> dict[str, FuncSpan]:
    """Functions and methods (``Class.method``) of *source*, by qualname.

    Looks inside ``if``/``try``/``with``/``for``/``while`` blocks at module and class level.
    A qualname defined more than once (a property getter + setter, ``try``/``except``
    alternatives, ``typing.overload``) is ambiguous: it is left out and listed in
    *duplicates*.
    """
    out: dict[str, FuncSpan] = {}
    dups: set[str] = set()

    def add(qual: str, span: FuncSpan) -> None:
        if qual in out or qual in dups:
            out.pop(qual, None)
            dups.add(qual)
        else:
            out[qual] = span

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                body = child.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    body = body[1:]
                add(
                    prefix + child.name,
                    FuncSpan(
                        prefix + child.name,
                        start,
                        child.end_lineno or child.lineno,
                        tuple(b.lineno for b in body) or (child.lineno,),
                        _is_stub_body(child.body),
                    ),
                )
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(
                child, ast.If | ast.Try | ast.With | ast.For | ast.While | ast.ExceptHandler
            ):
                visit(child, prefix)

    visit(ast.parse(source), "")
    if duplicates is not None:
        duplicates.extend(sorted(dups))
    return out


def stub_one(ref_text: str, base_text: str, qualname: str) -> str:
    """*ref_text* with only ``qualname`` replaced by its stubbed version from *base_text*."""
    ref, base = function_spans(ref_text)[qualname], function_spans(base_text)[qualname]
    ref_lines = ref_text.splitlines(keepends=True)
    base_lines = base_text.splitlines(keepends=True)
    seg = base_lines[base.start - 1 : base.end]
    if seg and not seg[-1].endswith("\n"):
        seg[-1] += "\n"
    return "".join(ref_lines[: ref.start - 1] + seg + ref_lines[ref.end :])


def stub_functions(
    base_tree: Path, ref_tree: Path, owned: list[str]
) -> list[tuple[str, FuncSpan]]:
    """``(file, span in the reference)`` for every function that is a stub in the base tree
    and implemented (differs) in the reference."""
    found: list[tuple[str, FuncSpan]] = []
    for rel in owned:
        b, r = base_tree / rel, ref_tree / rel
        if not (b.is_file() and r.is_file()):
            continue
        try:
            base_f = function_spans(b.read_text("utf-8"))
            ref_f = function_spans(r.read_text("utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        base_lines = b.read_text("utf-8").splitlines()
        ref_lines = r.read_text("utf-8").splitlines()
        for q, bs in base_f.items():
            rs_ = ref_f.get(q)
            if rs_ is None or not bs.is_stub or rs_.is_stub:
                continue
            if base_lines[bs.start - 1 : bs.end] == ref_lines[rs_.start - 1 : rs_.end]:
                continue
            found.append((rel, rs_))
    return found


def _context_test_id(ctx: str) -> str:
    """The test id of a pytest-cov context (``<id>|run`` / ``|setup`` / ``|teardown``).

    Only a known phase suffix is cut, so a parametrized id such as ``test_x[a|b]`` stays whole.
    """
    head, sep, phase = ctx.rpartition("|")
    return head if sep and phase in ("run", "setup", "teardown") else ctx


def coverage_tests(
    bundle: Path, owned: list[str], work: Path, timeout: float
) -> tuple[dict[tuple[str, int], set[str]], set[str]]:
    """Per-line test contexts of the reference run: ``{(file, line): {test ids}}`` and all ids.

    Needs ``pytest-cov`` / ``coverage`` in the interpreter that runs the grader.
    """
    try:
        import coverage
    except ImportError as exc:  # pragma: no cover - environment
        raise AdapterError("coverage (pytest-cov) is needed for `split`") from exc
    tree = vb.graded_tree(bundle, owned, bundle / "solution", work / "tree")
    guard = work / "_netguard"
    guard.mkdir(parents=True, exist_ok=True)
    (guard / "c0rec.py").write_text(RECORDER, "utf-8")
    record, covfile = work / "record.json", work / ".coverage"
    for f in (record, covfile):
        f.unlink(missing_ok=True)
    res = vb.run_grader(
        tree,
        bundle / "hidden",
        work,
        timeout,
        extra_args=["-p", "c0rec", f"--cov={tree}", "--cov-context=test", "--cov-report="],
        extra_env={"C0_RECORD": str(record), "COVERAGE_FILE": str(covfile)},
    )
    if not record.is_file() or not covfile.is_file():
        raise AdapterError(
            f"coverage run produced no data (rc={res.returncode}):\n{res.output_tail}"
        )
    prefix = "graded_tests/"
    ids = {k.removeprefix(prefix) for k in json.loads(record.read_text("utf-8"))["tests"]}
    data = coverage.CoverageData(basename=str(covfile))
    data.read()
    root = tree.resolve()
    lines: dict[tuple[str, int], set[str]] = {}
    for fname in data.measured_files():
        try:
            rel = Path(fname).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        for lineno, ctxs in data.contexts_by_lineno(fname).items():
            tests = {_context_test_id(c).removeprefix(prefix) for c in ctxs if _context_test_id(c)}
            if tests:
                lines[(rel, lineno)] = tests
    return lines, ids


def function_task_id(name: str, rel: str, qualname: str, used: set[str]) -> str:
    """A unique bundle id (<= 64 chars) for one function task."""
    tid = f"{FUNC_PREFIX}-{name}-{qualname.replace('.', '_')}"
    if len(tid) > 64 or tid in used:  # truncation, or `A.run` vs a top-level `A_run`
        digest = hashlib.sha1(f"{rel}:{qualname}".encode()).hexdigest()[:6]
        tid = f"{tid[:57].rstrip('-_')}-{digest}"
    return tid


def func_task_spec(row_name: str, rel: str, qualname: str, tests: list[str]) -> str:
    listed = ", ".join(f"`{t}`" for t in tests[:MAX_LISTED_TESTS])
    more = f" (and {len(tests) - MAX_LISTED_TESTS} more)" if len(tests) > MAX_LISTED_TESTS else ""
    return (
        f"Implement the function `{qualname}` in `{rel}` of the Python library `{row_name}`. "
        "Its body is currently `pass`; everything else in the library is already implemented, "
        f"so change ONLY `{rel}` and keep the function's name and signature. The docstring and "
        f"signature are in the file; the library's documentation is `{SPEC_NAME}` (search it, "
        f"e.g. grep for `{qualname.split('.')[-1]}`; do not read it all). The unit tests that "
        f"exercise this function: {listed}{more}. Run them with "
        "`python -m pytest <test file> -q -k <name>`; finish when they pass. No network access."
    )


def split_lib(
    name: str,
    cache: Path,
    timeout: float = GRADER_TIMEOUT,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Build one bundle per stubbed function from the built ``c0-<name>`` bundle."""
    if not LIB_RE.match(name):
        raise AdapterError(f"bad library name {name!r}")
    full = cache / "bundles" / f"c0-{name}"
    if not (full / "task.toml").is_file():
        raise AdapterError(f"{full} not built yet: run `build {name}` first")
    task = tomllib.loads((full / "task.toml").read_text("utf-8"))
    owned = [str(p) for p in task["units"][0]["owns"]]
    with tempfile.TemporaryDirectory(prefix=f"c0-split-{name}-") as tmp_s:
        tmp = Path(tmp_s)
        funcs = stub_functions(full / "repo", full / "solution", owned)
        if not funcs:
            raise AdapterError(f"{name}: no stubbed function found")
        summaries: list[dict[str, Any]] = []
        for rel in owned:  # ambiguous definitions cannot be stubbed one at a time
            base_file = full / "repo" / rel
            if base_file.is_file():
                dups: list[str] = []
                try:
                    function_spans(base_file.read_text("utf-8"), dups)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for q in dups:
                    log(f"[{name}] {rel}: {q} is defined more than once: skipped")
                    summaries.append({"id": f"{rel}:{q}", "skipped": "defined more than once"})
        lines, all_ids = coverage_tests(full, owned, tmp / "cov", timeout)
        base_deselect = []
        dfile = full / "hidden" / vb.DESELECT_FILE
        if dfile.is_file():
            base_deselect = [x for x in dfile.read_text("utf-8").splitlines() if x.strip()]
        used_ids: set[str] = set()
        for rel, span in funcs:
            relevant = sorted(
                {t for ln in span.body_lines for t in lines.get((rel, ln), set())} & all_ids
            )
            tid = function_task_id(name, rel, span.qualname, used_ids)
            used_ids.add(tid)
            if not relevant:
                log(f"[{tid}] skipped: no test executes it")
                summaries.append({"id": tid, "skipped": "no test executes it"})
                continue
            dest = cache / "bundles" / tid
            if dest.exists():
                if not force:
                    raise AdapterError(f"{dest} exists (use --force)")
                shutil.rmtree(dest)
            shutil.copytree(full / "repo", dest / "repo")
            shutil.copytree(full / "hidden", dest / "hidden")
            for o in owned:  # everything else is the reference implementation
                src = full / "solution" / o
                if src.is_file():
                    shutil.copy2(src, dest / "repo" / o)
            ref_text = (full / "solution" / rel).read_text("utf-8")
            base_text = (full / "repo" / rel).read_text("utf-8")
            (dest / "repo" / rel).write_text(stub_one(ref_text, base_text, span.qualname), "utf-8")
            (dest / "solution" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full / "solution" / rel, dest / "solution" / rel)
            deselect = sorted({*base_deselect, *(all_ids - set(relevant))})
            (dest / "hidden" / vb.DESELECT_FILE).write_text(
                "\n".join(deselect) + ("\n" if deselect else ""), "utf-8"
            )
            spec = func_task_spec(name, rel, span.qualname, relevant)
            (dest / "task.toml").write_text(
                f'id = "{tid}"\nkind = "control"\nspec = {json.dumps(spec)}\n\n'
                f'[[units]]\nid = "fn"\nowns = [{json.dumps(rel)}]\n'
                f'hidden_test = "{vb.SUITE_UNIT}"\n\n'
                f"[source]\nlibrary = {json.dumps(name)}\nfunction = {json.dumps(span.qualname)}\n"
                f"file = {json.dumps(rel)}\ntests_kept = {len(relevant)}\n"
                f"parent = {json.dumps(f'c0-{name}')}\n"
                f"weak = {'true' if len(relevant) < WEAK_TESTS else 'false'}\n",
                "utf-8",
            )
            report = vb.validate_bundle(dest, timeout)
            if not report.ok:
                shutil.rmtree(dest, ignore_errors=True)
                raise AdapterError(f"{tid}: bundle fails validation: {report.problems}")
            summaries.append(
                {
                    "id": tid,
                    "file": rel,
                    "function": span.qualname,
                    "lines": span.end - span.start + 1,
                    "tests": len(relevant),
                    "weak": len(relevant) < WEAK_TESTS,
                    "null_passed": (report.null or {}).get("passed"),
                }
            )
            log(f"[{tid}] {len(relevant)} tests, {span.end - span.start + 1} lines")
        return summaries


def format_table(rows: list[dict[str, Any]]) -> str:
    head = (
        f"{'lib':<14}{'files':>6}{'stubs':>7}{'tests':>7}{'desel':>7}"
        f"{'null-pass':>10}{'spec-k':>8}"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['lib']:<14}{r['owned_files']:>6}{r['stubs']:>7}{r['tests_kept']:>7}"
            f"{r['tests_deselected']:>7}{r['null_passed']!s:>10}{r['spec_chars'] // 1000:>8}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build one or more libraries into bundles")
    b.add_argument("libs", nargs="+", help="library names, or 'lite' for Commit0's lite split")
    b.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    b.add_argument("--force", action="store_true", help="rebuild an existing bundle")
    b.add_argument("--refresh", action="store_true", help="re-download the dataset rows")
    b.add_argument("--timeout", type=float, default=GRADER_TIMEOUT)
    b.add_argument(
        "--split",
        action="store_true",
        help="after building, also split each library into one bundle per stubbed function",
    )
    sp = sub.add_parser("split", help="split built libraries into one bundle per stubbed function")
    sp.add_argument("libs", nargs="+")
    sp.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--timeout", type=float, default=GRADER_TIMEOUT)
    return ap.parse_args(argv)


def _print_split(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{name}: {sum(1 for r in rows if 'skipped' not in r)} function tasks")
    for r in rows:
        if "skipped" in r:
            print(f"  {r['id']:<40} skipped ({r['skipped']})")
        else:
            weak = "  WEAK (<3 tests)" if r.get("weak") else ""
            print(
                f"  {r['id']:<40} {r['tests']:>4} tests  {r['lines']:>4} lines  {r['file']}{weak}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = args.cache.expanduser().resolve()
    names = [n for lib in args.libs for n in (LITE if lib == "lite" else [lib])]
    if args.cmd == "split":
        failed = 0
        for name in names:
            try:
                _print_split(name, split_lib(name, cache, args.timeout, args.force))
            except (AdapterError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"[{name}] SPLIT FAILED: {exc}", file=sys.stderr)
                failed += 1
        print(f"bundles: {cache / 'bundles'}")
        return 1 if failed else 0
    rows = fetch_rows(cache, args.refresh)
    summaries: list[dict[str, Any]] = []
    failed = 0
    for name in names:
        row = rows.get(name)
        if row is None:
            print(f"[{name}] not in the dataset", file=sys.stderr)
            failed += 1
            continue
        try:
            summaries.append(build_bundle(row, cache, force=args.force, timeout=args.timeout))
            if args.split:
                _print_split(name, split_lib(name, cache, args.timeout, args.force))
        except (AdapterError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[{name}] SKIPPED: {exc}", file=sys.stderr)
            failed += 1
    if summaries:
        print(format_table(summaries))
        print(f"bundles: {cache / 'bundles'}")
    return 1 if failed and not summaries else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
