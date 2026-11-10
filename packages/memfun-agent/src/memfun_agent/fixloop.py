"""Progress-aware verify-and-fix helpers for the context-first solver.

The plain loop ran N verify rounds, always kept the LAST edit, and showed the model only
the first 3000 characters of the verifier output.  On the B300 harness benchmark that
ended runs worse than they had been (a late edit broke the module's import: 0 of 23 tests
ran) and starved the model of what it needed (the failing tests' source, the earlier
attempts).  These helpers add:

* :func:`summarize` / :func:`score` -- read a pytest run into ``(broken, failing, -passing)``
  so rounds can be compared (lower is better);
* :class:`BestSoFar` -- snapshot the owned files after every verified round and restore the
  best one at the end, so the loop can never finish worse than its best state;
* :func:`failing_tests_source` / :func:`compact_errors` -- what the fix step should read;
* :func:`clip` -- head + tail clipping of verifier output (pytest's summary is at the END).

Pure functions and small classes: no LLM, no network, hermetic tests.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$", re.M)
_COUNT = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_BROKEN_HINTS = (
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
    "NameError: name",
    "no tests ran",
    "collected 0 items",
    "errors during collection",
    "Interrupted:",
)


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@dataclass(frozen=True)
class TestSummary:
    """A parsed pytest run (``found`` is False when the output is not pytest's)."""

    __test__ = False  # not a pytest test class

    found: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failed_ids: tuple[str, ...] = ()
    broken: bool = False


def summarize(output: str) -> TestSummary:
    """Parse a pytest run's output: counts, failing test ids, and whether the code is broken
    (does not import, does not collect) as opposed to merely failing some tests."""
    text = strip_ansi(output)
    counts = {"passed": 0, "failed": 0, "errors": 0}
    last = ""
    for line in text.splitlines():
        if re.search(r"\b(passed|failed|error|errors)\b.*\bin [\d.]+s", line):
            last = line
    for n, word in _COUNT.findall(last):
        key = "errors" if word.startswith("error") else word
        if key in counts:
            counts[key] = int(n)
    ids = tuple(dict.fromkeys(m.group(1) for m in _FAILED_LINE.finditer(text)))
    found = bool(last) or bool(ids)
    # Broken = the code does not import / collect: nothing ran green at all.
    broken = counts["passed"] == 0 and (
        counts["errors"] > 0 or any(h in text for h in _BROKEN_HINTS)
    )
    return TestSummary(
        found=found,
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["errors"],
        failed_ids=ids,
        broken=broken,
    )


def score(output: str, passed: bool) -> tuple[int, int, int]:
    """Comparable progress score, lower is better: ``(broken, failing, -passing)``.

    A run that does not import/collect is the worst; otherwise fewer failing tests win,
    then more passing tests.  A non-pytest verifier (a linter) scores by its error lines."""
    if passed:
        return (0, 0, 0)
    s = summarize(output)
    if s.found:
        return (1 if s.broken else 0, s.failed + s.errors, -s.passed)
    lint = len(re.findall(r"^\S+:\d+:\d+:", strip_ansi(output), re.M))
    return (0, lint or 1, 0)


def clip(text: str, limit: int = 10_000) -> str:
    """Head + tail of *text* (pytest prints its failure summary last)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters clipped] ...\n" + text[-half:]


def _test_function_source(path: Path, qualname: list[str]) -> str:
    try:
        src = path.read_text("utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""
    nodes: list[ast.stmt] = list(tree.body)
    found: ast.AST | None = None
    for name in qualname:
        found = next(
            (
                n
                for n in nodes
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and n.name == name
            ),
            None,
        )
        if found is None:
            return ""
        nodes = list(getattr(found, "body", []))
    return (ast.get_source_segment(src, found) or "") if found is not None else ""


def failing_tests_source(
    project_root: Path, failed_ids: tuple[str, ...] | list[str], max_chars: int = 9_000
) -> str:
    """The source of each failing test (``path::Class::test``), read-only context for the fix."""
    parts: list[str] = []
    total = 0
    for tid in failed_ids:
        path_s, *rest = tid.split("::")
        base = re.sub(r"\[.*\]$", "", rest[-1]) if rest else ""
        qual = [*rest[:-1], base] if rest else []
        p = project_root / path_s
        body = _test_function_source(p, qual) if p.is_file() and qual else ""
        if not body:
            continue
        chunk = f"# {tid}\n{body}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def compact_errors(output: str, per_failure: int = 900, max_failures: int = 12) -> str:
    """The failure summary lines first, then a short excerpt per failing test section."""
    text = strip_ansi(output)
    s = summarize(text)
    head = [f"{s.passed} passed, {s.failed} failed, {s.errors} errors"] if s.found else []
    summary_lines = [ln for ln in text.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    head += summary_lines[: max_failures * 2]
    sections = re.split(r"\n_{3,}\s+(.+?)\s+_{3,}\n", text)
    excerpts: list[str] = []
    for i in range(1, len(sections) - 1, 2):
        if len(excerpts) >= max_failures:
            break
        excerpts.append(f"--- {sections[i]} ---\n{sections[i + 1].strip()[:per_failure]}")
    body = "\n\n".join(excerpts) if excerpts else clip(text, 6_000)
    return ("\n".join(head) + "\n\n" + body).strip()


def compact_blocks(blocks: list[str], other_limit: int = 3_000) -> str:
    """:func:`compact_errors` per failing command: a pytest run is compacted as usual, the
    output of any other command (a syntax or import check, a reproduction script) is kept,
    clipped.  Compacting the joined text would keep only the pytest part."""
    return "\n\n".join(
        compact_errors(b) if summarize(b).found else clip(b, other_limit) for b in blocks
    )


@dataclass
class BestSoFar:
    """Snapshot the owned files whenever a round beats the best score; restore at the end."""

    paths: list[Path]
    best_score: tuple[int, int, int] | None = None
    best_files: dict[Path, str] = field(default_factory=lambda: {})
    best_round: int = 0

    def observe(self, round_no: int, sc: tuple[int, int, int]) -> bool:
        """Record round *round_no*'s score; True when it is a new best (files snapshotted)."""
        if self.best_score is not None and sc >= self.best_score:
            return False
        self.best_score, self.best_round = sc, round_no
        self.best_files = {}
        for p in self.paths:
            try:
                self.best_files[p] = p.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        return True

    def restore(self) -> int:
        """Write the best snapshot back; returns how many files changed."""
        changed = 0
        for p, text in self.best_files.items():
            try:
                if p.read_text("utf-8") != text:
                    p.write_text(text, "utf-8")
                    changed += 1
            except (OSError, UnicodeDecodeError):
                continue
        return changed


# ── Probes: let the model look at real behaviour when patching stalls ─────────

MAX_PROBES = 2
MAX_PROBE_CHARS = 2_000
PROBE_TIMEOUT_S = 30.0
PROBE_OUTPUT_CHARS = 3_000
#: A friendly early refusal of the obvious (the audit hook in PROBE_PREAMBLE is what enforces).
_PROBE_FORBIDDEN = re.compile(
    r"\b(os\.(remove|unlink|rmdir|system|popen|rename|replace)|shutil\.|subprocess|socket|"
    r"urllib|requests|httpx|http\.client|write_text|write_bytes)"
)

#: Runs before the probe in its own interpreter.  The probe imports the project's own packages
#: from the project (project root and ``src/`` first on sys.path); an audit hook makes writes,
#: child processes, network and ctypes fail, and reads outside the project / the interpreter's
#: own directories fail.  Two modes (argv[3]):
#:
#: * ``strict`` (function-level benchmarks): started as ``python -I -S -B`` so no site-packages
#:   exist at all; any path under ``site-packages`` is unreadable.  Standard library + project.
#: * ``project`` (real repositories): the project's own interpreter with its installed
#:   dependencies importable; reads of an INSTALLED COPY of a project-owned package (argv[4], a
#:   comma separated list of top-level names) by path are denied, other site-packages reads work.
#:
#: * ``off`` (disposable sandboxes only, e.g. a throwaway task container): the project's
#:   interpreter, project packages first, and NO audit hook: temp files, child processes, local
#:   sockets and import-time caches all work.  Nothing stops the script; the caller's sandbox must.
#:
#: Defence in depth against accidents and curiosity, NOT a sandbox: native code, or a determined
#: script, can get around an audit hook.
PROBE_PREAMBLE = r"""
import os, sys
_root = os.path.realpath(sys.argv[1])
_snippet, _mode = sys.argv[2], sys.argv[3]
_owned = set(filter(None, sys.argv[4].split(",")))
sys.path[:0] = [_root, os.path.join(_root, "src")]
_allowed = {_root, "/dev"}
for _p in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
    _allowed.add(os.path.realpath(_p))
if _mode != "strict":
    for _p in sys.path:
        if _p and os.path.isdir(_p):
            _allowed.add(os.path.realpath(_p))
_BLOCKED = (
    "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod", "os.chown", "os.truncate",
    "os.symlink", "os.link", "os.utime", "os.mkfifo", "os.mknod", "os.system", "os.exec",
    "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.putenv", "os.unsetenv",
    "shutil.", "subprocess.", "socket.", "ctypes.", "tempfile.", "pty.", "webbrowser.",
    "fcntl.", "mmap.", "multiprocessing.",
)
def _under(real, base):
    return real == base or real.startswith(base + os.sep)
def _inside(path):
    try:
        real = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return True  # a file descriptor or None
    parts = real.split(os.sep)
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            if _mode == "strict":
                return False
            rest = parts[parts.index(marker) + 1:]
            top = rest[0].split(".")[0] if rest else ""
            if top in _owned:
                return False  # an installed copy of a package this project owns
    if _under(real, _root):
        return True
    return any(_under(real, a) for a in _allowed)
def _hook(event, args):
    if event.startswith(_BLOCKED):
        raise PermissionError("probe: %s is not allowed (read-only experiments)" % event)
    if event == "open":
        path, mode, flags = args
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        if write:
            raise PermissionError("probe: writing files is not allowed (read-only experiments)")
        if not isinstance(path, int) and path is not None and not _inside(path):
            raise PermissionError("probe: no reading outside the project: %s" % (path,))
    elif event in ("os.listdir", "os.scandir", "glob.glob", "os.chdir") and args:
        if args[0] is not None and not isinstance(args[0], int) and not _inside(args[0]):
            raise PermissionError("probe: no listing outside the project: %s" % (args[0],))
if _mode != "off":
    sys.addaudithook(_hook)
os.chdir(_root)
exec(compile(_snippet, "<probe>", "exec"), {"__name__": "__main__"})
"""


def owned_packages(project_root: Path) -> list[str]:
    """Top-level importable names the project itself defines (root and ``src/``)."""
    names: set[str] = set()
    for base in (project_root, project_root / "src"):
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.name.startswith((".", "_")) or e.name in {
                "tests",
                "test",
                "docs",
                "build",
                "dist",
            }:
                continue
            if e.is_dir() and (e / "__init__.py").exists():
                names.add(e.name)
            elif e.is_file() and e.suffix == ".py":
                names.add(e.stem)
    return sorted(names)


def _not_memfuns_own(path: str | None) -> str | None:
    """*path* unless it is the interpreter memfun itself runs in (which has other dependencies
    and may be another Python version)."""
    if path and os.path.realpath(path) != os.path.realpath(sys.executable):
        return path
    return None


def _shebang_python(script: str) -> str | None:
    try:
        with open(script, "rb") as fh:
            first = fh.readline(300).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if parts and parts[0].endswith("/env") and len(parts) > 1:
        parts = parts[1:]
    return parts[0] if parts and re.fullmatch(r".*python(\d+(\.\d+)?)?", parts[0]) else None


def probe_python(verify_cmds: list[str], project_root: Path) -> str | None:
    """The interpreter the project's own tests run with, or None when it cannot be found.

    From the verify command: an absolute ``python`` / ``.../pytest`` is taken as given (a
    ``pytest`` script gives its sibling ``python`` or its shebang interpreter); a bare name is
    resolved through ``PATH`` and accepted only if it is not memfun's own interpreter.  Then
    ``<project>/.venv/bin/python``, then ``python3`` / ``python`` on ``PATH`` (again never memfun's
    own).  Never memfun's own interpreter: it has other dependencies and may be another Python.
    """
    uses_python = False
    for cmd in verify_cmds:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue
        for t in tokens[:3]:
            name = Path(t).name
            is_python = re.fullmatch(r"python(\d+(\.\d+)?)?", name) is not None
            if not (is_python or name == "pytest"):
                continue
            uses_python = True
            explicit = "/" in t
            found = t if explicit else shutil.which(t)
            if not found:
                continue
            if is_python and explicit and not Path(found).exists():
                continue
            if is_python:
                got = found if explicit else _not_memfuns_own(found)
            else:
                sibling = Path(found).parent / "python"
                got = str(sibling) if sibling.exists() else _shebang_python(found)
                if got and not explicit:
                    got = _not_memfuns_own(got)
            if got:
                return got
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    if uses_python:
        for name in ("python3", "python"):
            got = _not_memfuns_own(shutil.which(name))
            if got:
                return got
    return None


def parse_probes(raw: str) -> list[str]:
    """The snippets of a model's ``probes`` answer: a JSON array of strings, tolerant of fences
    and of a single bare snippet.  At most :data:`MAX_PROBES`, each at most
    :data:`MAX_PROBE_CHARS`."""
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
    items: list[object]
    try:
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
    except ValueError:
        items = [text] if text and "print" in text else []
    out = [
        x.strip() for x in items if isinstance(x, str) and x.strip() and len(x) <= MAX_PROBE_CHARS
    ]
    return out[:MAX_PROBES]


def probe_refusal(snippet: str) -> str:
    """Why *snippet* may not run ("" when it may)."""
    m = _PROBE_FORBIDDEN.search(snippet)
    return f"refused (uses {m.group(0)!r})" if m else ""


PROBE_GUARDS = ("strict", "project", "off")
#: Environment a guarded probe inherits (everything else is dropped: no keys, no tokens).
PROBE_ENV_ALLOW = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "LD_LIBRARY_PATH",
)


def probe_env(guard: str) -> dict[str, str]:
    """The environment of a probe process.  ``off`` (a disposable sandbox) inherits everything;
    the guarded modes get ``PATH`` plus :data:`PROBE_ENV_ALLOW` only."""
    if guard == "off":
        return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    if guard == "project":
        env["PATH"] = os.environ.get("PATH", env["PATH"])
        env.update({k: os.environ[k] for k in PROBE_ENV_ALLOW if k in os.environ})
    return env


def _run_in_group(
    cmd: list[str], cwd: str, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` in its own process group; the whole group is killed on timeout, so a
    child the probe started cannot outlive it."""
    import signal

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    finally:
        with contextlib.suppress(OSError):  # stragglers of a probe that exited early
            os.killpg(proc.pid, signal.SIGKILL)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


async def run_probes(
    project_root: Path,
    snippets: list[str],
    *,
    python: str | None = None,
    guard: str = "project",
) -> str:
    """Run each snippet in a guarded interpreter (see :data:`PROBE_PREAMBLE`) from
    *project_root* (timeout, clipped output); return a text block of each snippet and its real
    output for the next fix round.

    *guard* (see :data:`PROBE_PREAMBLE`): ``strict`` is ``python -I -S -B`` (standard library +
    the project only; the function-level benchmarks); ``project`` is *python* (the project's
    interpreter, see :func:`probe_python`) with its dependencies, read-only; ``off`` is the same
    interpreter with no audit hook, ONLY for disposable sandboxes.  Without an interpreter
    (None) the non-strict modes cannot run and each probe says so.
    """
    if guard not in PROBE_GUARDS:
        raise ValueError(f"probe guard must be one of {PROBE_GUARDS}, got {guard!r}")
    interpreter = python or (sys.executable if guard == "strict" else None)
    flags = {"strict": ["-I", "-S", "-B"], "project": ["-I", "-B"], "off": ["-B"]}[guard]
    mode = guard
    owned = ",".join(owned_packages(project_root))
    blocks: list[str] = []
    for i, snippet in enumerate(snippets[:MAX_PROBES], 1):
        why = probe_refusal(snippet) if guard != "off" else ""
        if why:
            blocks.append(f"--- experiment {i} ---\n{snippet}\n=> {why}")
            continue
        if interpreter is None:
            blocks.append(
                f"--- experiment {i} ---\n{snippet}\n=> (could not run: no project interpreter "
                "found; the verify command names none)"
            )
            continue
        try:
            res = await asyncio.to_thread(
                _run_in_group,
                [
                    interpreter,
                    *flags,
                    "-c",
                    PROBE_PREAMBLE,
                    str(project_root),
                    snippet,
                    mode,
                    owned,
                ],
                str(project_root),
                probe_env(guard),
                PROBE_TIMEOUT_S,
            )
            out = (res.stdout + (("\n" + res.stderr) if res.stderr else "")).strip()
            body = clip(out or "(no output)", PROBE_OUTPUT_CHARS)
        except subprocess.TimeoutExpired:
            body = f"(timed out after {PROBE_TIMEOUT_S:.0f}s)"
        except (OSError, ValueError) as exc:  # ValueError: a NUL byte in the snippet
            body = f"(could not run: {exc})"
        blocks.append(f"--- experiment {i} ---\n{snippet}\n=> {body}")
    return "\n\n".join(blocks)


# ── Round memory: what each round changed, and whether we have been here before ──


def read_texts(paths: list[Path]) -> dict[Path, str]:
    """Current text of every readable file in *paths*."""
    out: dict[Path, str] = {}
    for p in paths:
        try:
            out[p] = p.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def state_key(texts: dict[Path, str]) -> str:
    """A hash of the file contents: two rounds with the same key left the code identical."""
    h = hashlib.sha256()
    for p in sorted(texts, key=str):
        h.update(str(p).encode())
        h.update(b"\0")
        h.update(texts[p].encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def short_diff(before: dict[Path, str], after: dict[Path, str], limit: int = 700) -> str:
    """A compact unified diff of what a round changed (changed lines only), clipped."""
    lines: list[str] = []
    for p in sorted(after, key=str):
        a, b = before.get(p, ""), after[p]
        if a == b:
            continue
        for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
            if ln.startswith(("---", "+++", "@@")):
                continue
            lines.append(ln)
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n... [diff clipped]"


# ── The LOOK step: untruncated failures and differential probes ────────────────────────────


def pytest_detail_command(
    verify_cmds: list[str], failed_ids: list[str] | tuple[str, ...]
) -> list[str] | None:
    """A pytest command that re-runs only *failed_ids* with full assertion detail, or None.

    pytest prints an assertion on a list or dict as one truncated line
    (``assert [{'a...'}, ...] == []``) unless verbosity is 2 or more.  This takes the project's own
    pytest verify command, drops its test selectors and ``-q``, adds ``-vv`` and the failing ids.
    Other verify commands (linters) give None.
    """
    for cmd in verify_cmds:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue
        if "pytest" not in tokens and not any(t.endswith("pytest") for t in tokens):
            continue
        idx = next(i for i, t in enumerate(tokens) if t == "pytest" or t.endswith("/pytest"))
        head, rest = tokens[: idx + 1], tokens[idx + 1 :]
        opts: list[str] = []
        skip_next = False
        for i, t in enumerate(rest):
            if skip_next:
                opts.append(t)
                skip_next = False
                continue
            if "::" in t or t.endswith(".py") or t in {"tests", "test"}:
                continue  # a selector: replaced by the failing ids
            if t in {"-q", "--quiet", "-qq"}:
                continue
            if t in {"-p", "-k", "-m", "-o", "--tb", "-c", "--rootdir"} and i + 1 < len(rest):
                opts.append(t)
                skip_next = True
                continue
            opts.append(t)
        if not any(o.startswith("--tb") for o in opts):
            opts.append("--tb=short")
        return [*head, "-vv", *opts, *failed_ids]
    return None


def extract_failure_detail(output: str, per_test: int = 2_500, total: int = 9_000) -> str:
    """The assertion detail of each failing test: its traceback head and its ``E`` lines.

    Per-test and total byte budgets; the head and the first mismatching items are kept.
    """
    text = strip_ansi(output)
    sections = re.split(r"\n_{3,}\s+(.+?)\s+_{3,}\n", text)
    parts: list[str] = []
    used = 0
    for i in range(1, len(sections) - 1, 2):
        name, body = sections[i], re.split(r"\n={3,}", sections[i + 1], maxsplit=1)[0]
        lines = body.splitlines()
        e_lines = [ln for ln in lines if ln.startswith("E ")]
        head = [ln for ln in lines if not ln.startswith("E ")][:14]
        keep = head + (["..."] if e_lines else []) + e_lines
        chunk = clip(f"--- {name} ---\n" + "\n".join(keep), per_test)
        if used + len(chunk) > total:
            parts.append(clip(chunk, max(400, total - used)))
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)


async def run_detail(
    project_root: Path,
    verify_cmds: list[str],
    failed_ids: list[str] | tuple[str, ...],
    timeout: float = 120.0,
) -> str:
    """Re-run the failing tests with full assertion output and return the extracted detail."""
    cmd = pytest_detail_command(verify_cmds, list(failed_ids)[:12])
    if cmd is None:
        return ""
    try:
        res = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return extract_failure_detail(res.stdout + "\n" + res.stderr)


def probe_fingerprint(output: str) -> str:
    """A short key for the OUTPUT of a probe run (not the script that produced it), empty when it
    holds no information (nothing printed, only refusals / timeouts).  Two runs with the same key
    told the model the same thing."""
    bodies = re.findall(r"\n=> (.*?)(?=\n\n--- experiment|\Z)", output, re.S)
    useful: list[str] = []
    for body in bodies:
        b = body.strip()
        if (
            not b
            or b == "(no output)"
            or b.startswith(("refused (", "(timed out", "(could not run"))
        ):
            continue
        useful.append(b)
    if not useful:
        return ""
    return hashlib.sha256("\n".join(useful).encode()).hexdigest()[:16]


PROBE_LOG = ".memfun/probes.jsonl"
CHECK_LOG = ".memfun/checks.jsonl"


def log_check(project_root: Path, record: dict[str, object]) -> None:
    """Append one record of what the solver's own checks did (a reproduction attempt, the
    tests it selected, a verify round) to ``.memfun/checks.jsonl``: the evidence of which real
    tests ran and what they said.  Best effort: a logging failure never affects the solve."""
    try:
        path = project_root / CHECK_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def init_probe_log(project_root: Path) -> None:
    """Create the (empty) probe log when the look step is enabled, so that an audit can tell
    "probes were on, none was needed" (an empty file) from "the log is gone" (no file)."""
    try:
        path = project_root / PROBE_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError:
        pass


def log_probe(project_root: Path, snippets: list[str], output: str) -> None:
    """Append one probe run (the scripts and their output) to ``.memfun/probes.jsonl``.

    A record of what the model ran, for audits (e.g. did a script look outside the project).
    Best effort: a logging failure never affects the fix loop.
    """
    try:
        path = project_root / PROBE_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "scripts": snippets, "output": output}) + "\n")
    except OSError:
        pass
