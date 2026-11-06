#!/usr/bin/env python3
"""Validate swarm task bundles (SWARM-EVAL-DESIGN.md section 2 format; oracle/null arms).

For every ``<root>/<id>/task.toml`` this checks:

* **schema**: id matches the directory, kind is fanout|control, units have unique ids,
  every ``hidden_test`` exists, fan-out units own disjoint paths, every owned path exists
  in ``repo/`` or ``solution/``, and every file in ``solution/`` is owned by some unit;
* **oracle**: grading the reference solution passes ALL hidden tests and the visible tests;
* **null**: grading the untouched ``repo/`` FAILS every unit's hidden test file (and
  ``test_integration.py`` when present), so no unit is solved by doing nothing;
* **independence** (fan-out only): for each unit, grading the reference solution with
  THAT unit's owned files replaced by a syntax error fails only that unit's hidden tests;
  every other unit's hidden tests still pass (the integration test is exempt).

Grading models how agent output must be graded (see evals/swarm/smoke/README.md):

1. the graded tree is a clean copy of ``repo/`` plus ONLY the files the units own, taken
   from the candidate tree (here: ``solution/``); nothing else from the candidate is used;
2. hidden tests run from a separate grader directory, never from inside the graded tree:
   ``--rootdir`` and ``-c`` point at the grader's own ini, ``--confcutdir`` stops conftest
   discovery there, ``-p no:cacheprovider``, ``python -P``; the graded tree is appended to
   ``sys.path`` (after site-packages) by the grader's own conftest, so a planted
   ``pytest.py``/``conftest.py``/``sitecustomize.py`` in the tree cannot take over;
3. a ``sitecustomize`` guard disables outbound sockets, and each run has a timeout.

Usage: python evals/swarm/validate_bundles.py [ROOT ...] [--json] [--timeout 120]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "smoke"
INTEGRATION = "test_integration"
#: A unit whose ``hidden_test`` is exactly this names the whole ``hidden/`` directory as
#: its test suite (Commit0 libraries ship many test files plus conftest/helpers).  Such
#: a unit is graded over every collected test, not one module.
SUITE_UNIT = "hidden"
ALL_MODULES = "*"
#: Optional file in a test directory: pytest node ids as the grader's own pytest
#: reports them, to deselect, one per line.
DESELECT_FILE = ".deselect"
BROKEN_SOURCE = "def broken(:  # deliberately not Python (independence check)\n"

NETGUARD = """
import socket as _s

def _blocked(*_a, **_k):
    raise OSError("network disabled by validate_bundles")

_orig_connect = _s.socket.connect
def _connect(self, address):
    if self.family in (_s.AF_INET, _s.AF_INET6):
        _blocked()
    return _orig_connect(self, address)

_s.socket.connect = _connect
_s.socket.connect_ex = lambda self, address: _blocked() if self.family in (
    _s.AF_INET, _s.AF_INET6) else 0
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
"""

GRADER_CONFTEST = """
import os
import sys

# The graded tree goes AFTER site-packages/stdlib: it can provide the task's packages but
# cannot shadow pytest, the stdlib or this grader.
sys.path.append(os.environ["BUNDLE_GRADED_TREE"])
_src = os.path.join(os.environ["BUNDLE_GRADED_TREE"], "src")  # src-layout libraries
if os.path.isdir(_src):
    sys.path.append(_src)
"""

GRADER_INI = "[pytest]\naddopts =\n"


@dataclass
class RunResult:
    returncode: int
    passed: int
    failed: int
    by_file: dict[str, dict[str, int]]
    output_tail: str


@dataclass
class BundleReport:
    id: str
    ok: bool
    problems: list[str] = field(default_factory=lambda: list[str]())
    oracle: dict[str, int] | None = None
    null: dict[str, int] | None = None
    independence: dict[str, str] | None = None


def _copytree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _parse_junit(path: Path) -> tuple[int, int, dict[str, dict[str, int]]]:
    passed = failed = 0
    by_file: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return 0, 0, {}
    for case in ET.parse(path).getroot().iter("testcase"):
        # classname looks like "hidden_tests.test_x[.TestCls]"; a collection error has an
        # empty classname and the module path in name.
        label = case.get("classname") or case.get("name", "")
        module = next((p for p in label.split(".") if p.startswith("test")), label)
        bad = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        slot = by_file.setdefault(module, {"passed": 0, "failed": 0})
        if bad or skipped:  # a skipped hidden test does not count as solved
            failed += 1
            slot["failed"] += 1
        else:
            passed += 1
            slot["passed"] += 1
    return passed, failed, by_file


def graded_tree(bundle: Path, owned: list[str], candidate: Path | None, dest: Path) -> Path:
    """Clean repo/ plus ONLY the owned paths taken from ``candidate``."""
    _copytree(bundle / "repo", dest)
    dest.mkdir(parents=True, exist_ok=True)
    if candidate is not None:
        for rel in owned:
            src = candidate / rel
            target = dest / rel
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            elif src.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    return dest


def unit_module(hidden_test: str) -> str:
    """Module name a unit is graded by: the test file's stem, or ALL_MODULES for a suite."""
    if hidden_test.rstrip("/") == SUITE_UNIT:
        return ALL_MODULES
    return Path(hidden_test).stem


def run_grader(
    tree: Path,
    tests_src: Path,
    scratch: Path,
    timeout: float,
    extra_args: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
) -> RunResult:
    """Run the tests in ``tests_src`` against ``tree`` from an isolated grader dir.

    ``extra_args`` / ``extra_env`` are for tooling (e.g. a recording plugin importable
    from ``scratch/_netguard``); grading itself passes neither.
    """
    grader = scratch / "grader"
    shutil.rmtree(grader, ignore_errors=True)
    grader.mkdir(parents=True)
    _copytree(tests_src, grader / "graded_tests")
    (grader / "conftest.py").write_text(GRADER_CONFTEST, encoding="utf-8")
    ini = grader / "pytest.ini"
    ini.write_text(GRADER_INI, encoding="utf-8")
    guard = scratch / "_netguard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(NETGUARD, encoding="utf-8")
    report = scratch / "junit.xml"
    report.unlink(missing_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(guard),
        "BUNDLE_GRADED_TREE": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HOME": str(scratch),
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
    }
    deselect: list[str] = []
    listed = tests_src / DESELECT_FILE
    if tests_src.is_dir() and listed.is_file():
        for line in listed.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                # Node ids are rootdir-relative or not depending on how pytest resolved
                # the rootdir; passing both forms deselects the same test either way.
                deselect += ["--deselect", line]
                if not line.startswith("graded_tests/"):
                    deselect += ["--deselect", f"graded_tests/{line}"]
    env.update(extra_env or {})
    cmd = [
        sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "-c", str(ini), "--rootdir", str(grader), "--confcutdir", str(grader),
        "--continue-on-collection-errors",
        f"--junitxml={report}", *deselect, *extra_args, "graded_tests",
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, cwd=grader, env=env, capture_output=True, text=True, timeout=timeout
        )
        rc, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out = -9, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}"
    passed, failed, by_file = _parse_junit(report)
    return RunResult(rc, passed, failed, by_file, str(out)[-1500:])


def schema_problems(bundle: Path, task: dict[str, object]) -> list[str]:
    """Schema problems of one bundle (empty list: the bundle is well formed)."""
    problems: list[str] = []
    if task.get("id") != bundle.name:
        problems.append(f"id {task.get('id')!r} != directory name {bundle.name!r}")
    if task.get("kind") not in ("fanout", "control"):
        problems.append(f"kind must be fanout|control, got {task.get('kind')!r}")
    if not isinstance(task.get("spec"), str) or not str(task.get("spec")).strip():
        problems.append("spec missing")
    units = task.get("units")
    if not isinstance(units, list) or not units:
        return [*problems, "units missing"]
    seen_ids: set[str] = set()
    owners: dict[str, str] = {}
    for raw in units:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(raw, dict):
            problems.append("unit is not a table")
            continue
        unit: dict[str, object] = raw  # pyright: ignore[reportUnknownVariableType]
        uid = str(unit.get("id"))
        if uid in seen_ids:
            problems.append(f"duplicate unit id {uid}")
        seen_ids.add(uid)
        hidden = unit.get("hidden_test")
        if not isinstance(hidden, str):
            problems.append(f"unit {uid}: hidden_test {hidden!r} not found")
        elif hidden.rstrip("/") == SUITE_UNIT:
            if not any((bundle / SUITE_UNIT).rglob("test*.py")):
                problems.append(f"unit {uid}: hidden/ holds no test*.py")
        elif not (bundle / hidden).is_file():
            problems.append(f"unit {uid}: hidden_test {hidden!r} not found")
        owns = unit.get("owns")
        if not isinstance(owns, list) or not owns:
            problems.append(f"unit {uid}: owns missing")
            continue
        for path in owns:  # pyright: ignore[reportUnknownVariableType]
            p = str(path)  # pyright: ignore[reportUnknownArgumentType]
            if not ((bundle / "repo" / p).exists() or (bundle / "solution" / p).exists()):
                problems.append(f"unit {uid}: owned path {p} not in repo/ or solution/")
            if task.get("kind") == "fanout" and p in owners:
                problems.append(f"path {p} owned by both {owners[p]} and {uid}")
            owners[p] = uid
    if task.get("kind") == "fanout" and len(seen_ids) < 2:
        problems.append("a fanout task needs >= 2 units")
    for d in ("repo", "hidden", "solution"):
        if not (bundle / d).is_dir():
            problems.append(f"{d}/ missing")
    sol = bundle / "solution"
    if sol.is_dir():
        for f in sorted(sol.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                rel = f.relative_to(sol).as_posix()
                if not any(rel == o or rel.startswith(o.rstrip("/") + "/") for o in owners):
                    problems.append(
                        f"solution/{rel} is not owned by any unit (grading ignores it)"
                    )
    return problems


def unit_fails(res: RunResult, module: str) -> bool:
    """True unless ``module`` collected tests and all of them passed.

    ``ALL_MODULES`` means the whole run: it must have collected tests, none failed.
    """
    if module == ALL_MODULES:
        return res.passed == 0 or res.failed > 0
    stats = res.by_file.get(module)
    # No testcases under that module name (collection error) counts as failing.
    return stats is None or stats["failed"] > 0 or stats["passed"] == 0


def validate_bundle(bundle: Path, timeout: float) -> BundleReport:
    report = BundleReport(id=bundle.name, ok=False)
    try:
        task = tomllib.loads((bundle / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        report.problems.append(f"task.toml unreadable: {exc}")
        return report
    report.problems.extend(schema_problems(bundle, task))
    if report.problems:
        return report
    units: list[dict[str, object]] = task["units"]
    unit_modules = {str(u["id"]): unit_module(str(u["hidden_test"])) for u in units}
    unit_owns = {str(u["id"]): [str(p) for p in u["owns"]] for u in units}  # type: ignore[union-attr]
    all_owned = [p for owns in unit_owns.values() for p in owns]
    hidden = bundle / "hidden"

    with tempfile.TemporaryDirectory(prefix=f"bundle-{bundle.name}-") as tmp_s:
        tmp = Path(tmp_s)

        # --- oracle ----------------------------------------------------------------------
        tree = graded_tree(bundle, all_owned, bundle / "solution", tmp / "oracle" / "tree")
        res = run_grader(tree, hidden, tmp / "oracle", timeout)
        report.oracle = {"passed": res.passed, "failed": res.failed, "rc": res.returncode}
        if res.returncode != 0 or res.failed or not res.passed:
            report.problems.append(f"oracle does not pass all hidden tests:\n{res.output_tail}")
        missing = [m for m in unit_modules.values() if unit_fails(res, m)]
        if missing:
            report.problems.append(f"oracle: no passing tests collected from {missing}")
        # Suite (Commit0) bundles keep the library's own tests visible, and some of those
        # fail against the reference on purpose (deselected in hidden/), so skip this.
        suite = ALL_MODULES in unit_modules.values()
        if (bundle / "repo" / "tests").is_dir() and not suite:
            vis = run_grader(tree, bundle / "repo" / "tests", tmp / "oracle-visible", timeout)
            if vis.returncode != 0:
                report.problems.append(f"oracle fails visible tests:\n{vis.output_tail}")

        # --- null ------------------------------------------------------------------------
        tree = graded_tree(bundle, all_owned, None, tmp / "null" / "tree")
        res = run_grader(tree, hidden, tmp / "null", timeout)
        report.null = {"passed": res.passed, "failed": res.failed, "rc": res.returncode}
        if res.returncode == 0:
            report.problems.append("null (untouched repo/) PASSES the hidden tests")
        must_fail = dict(unit_modules)
        if (hidden / f"{INTEGRATION}.py").is_file():
            must_fail["<integration>"] = INTEGRATION
        for uid, module in must_fail.items():
            if not unit_fails(res, module):
                report.problems.append(f"null: unit {uid} ({module}) passes without any work")

        # --- independence: break one unit, only its tests may fail -----------------------
        if task.get("kind") == "fanout":
            report.independence = {}
            for uid, owns in unit_owns.items():
                cand = tmp / f"mut-{uid}" / "candidate"
                _copytree(bundle / "solution", cand)
                for rel in owns:
                    target = cand / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(BROKEN_SOURCE, encoding="utf-8")
                tree = graded_tree(bundle, all_owned, cand, tmp / f"mut-{uid}" / "tree")
                res = run_grader(tree, hidden, tmp / f"mut-{uid}", timeout)
                collateral = [
                    other
                    for other, module in unit_modules.items()
                    if other != uid and unit_fails(res, module)
                ]
                own_fails = unit_fails(res, unit_modules[uid])
                if own_fails and not collateral:
                    report.independence[uid] = "ok"
                else:
                    detail: list[str] = []
                    if not own_fails:
                        detail.append("its own tests still pass")
                    if collateral:
                        detail.append(f"also fails units {collateral}")
                    report.independence[uid] = "; ".join(detail)
                    report.problems.append(f"independence: breaking {uid}: {'; '.join(detail)}")
    report.ok = not report.problems
    return report


def find_bundles(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if (root / "task.toml").is_file():
            out.append(root)
        else:
            out.extend(sorted(p.parent for p in root.glob("*/task.toml")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("roots", nargs="*", type=Path, default=[DEFAULT_ROOT])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0, help="per pytest run, seconds")
    args = ap.parse_args(argv)
    bundles = find_bundles(args.roots)
    if not bundles:
        print("no bundles found", file=sys.stderr)
        return 2
    reports = [validate_bundle(b, args.timeout) for b in bundles]
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        for r in reports:
            status = "OK  " if r.ok else "FAIL"
            print(f"{status} {r.id:<16} oracle={r.oracle} null={r.null}")
            if r.independence is not None:
                print(f"     independence: {r.independence}")
            for p in r.problems:
                print("     - " + p.replace("\n", "\n       "))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
