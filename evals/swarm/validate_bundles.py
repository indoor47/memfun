#!/usr/bin/env python3
"""Validate swarm task bundles (SWARM-EVAL-DESIGN.md section 2 format; oracle/null arms).

For every ``<root>/<id>/task.toml`` this checks:

* schema: id matches the directory, kind is fanout|control, units have unique ids, every
  ``hidden_test`` exists, fan-out units own disjoint paths, and every owned path exists in
  ``repo/`` or ``solution/``;
* **oracle**: ``repo/`` + ``solution/`` overlay passes ALL hidden tests and the visible tests;
* **null**: the untouched ``repo/`` FAILS, and fails every unit's hidden test file (and
  ``test_integration.py`` when present), so no unit is graded as solved by doing nothing.

Each run happens in a fresh temp copy, in a subprocess with a timeout, with networking
disabled by a ``sitecustomize`` guard that makes ``socket.connect`` raise.

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

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "smoke"

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


def _copytree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _parse_junit(path: Path) -> tuple[int, int, dict[str, dict[str, int]]]:
    passed = failed = 0
    by_file: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return 0, 0, {}
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        # classname looks like "hidden_tests.test_x" or "hidden_tests.test_x.TestCls".
        module = next((p for p in classname.split(".") if p.startswith("test")), classname)
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


def run_pytest(workdir: Path, test_dir: str, timeout: float) -> RunResult:
    guard = workdir.parent / "_netguard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(NETGUARD, encoding="utf-8")
    report = workdir.parent / f"junit-{test_dir.replace('/', '_')}.xml"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join([str(guard), str(workdir)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HOME": str(workdir.parent),
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
    }
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--rootdir", str(workdir), f"--junitxml={report}", test_dir,
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout
        )
        rc, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out = -9, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}"
    passed, failed, by_file = _parse_junit(report)
    return RunResult(rc, passed, failed, by_file, str(out)[-1500:])


def _schema(bundle: Path, task: dict[str, object]) -> list[str]:
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
        if not isinstance(hidden, str) or not (bundle / hidden).is_file():
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
    return problems


def validate_bundle(bundle: Path, timeout: float) -> BundleReport:
    report = BundleReport(id=bundle.name, ok=False)
    try:
        task = tomllib.loads((bundle / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        report.problems.append(f"task.toml unreadable: {exc}")
        return report
    report.problems.extend(_schema(bundle, task))
    if report.problems:
        return report
    units: list[dict[str, object]] = task["units"]
    unit_modules = {str(u["id"]): Path(str(u["hidden_test"])).stem for u in units}

    with tempfile.TemporaryDirectory(prefix=f"bundle-{bundle.name}-") as tmp:
        # --- oracle: start tree + reference solution + hidden tests ------------------
        oracle = Path(tmp) / "oracle" / "work"
        _copytree(bundle / "repo", oracle)
        _copytree(bundle / "solution", oracle)
        _copytree(bundle / "hidden", oracle / "hidden_tests")
        res = run_pytest(oracle, "hidden_tests", timeout)
        report.oracle = {"passed": res.passed, "failed": res.failed, "rc": res.returncode}
        if res.returncode != 0 or res.failed or not res.passed:
            report.problems.append(f"oracle does not pass all hidden tests:\n{res.output_tail}")
        missing = [
            m for m in unit_modules.values() if res.by_file.get(m, {}).get("passed", 0) == 0
        ]
        if missing:
            report.problems.append(f"oracle: no passing tests collected from {missing}")
        if (oracle / "tests").is_dir():
            vis = run_pytest(oracle, "tests", timeout)
            if vis.returncode != 0:
                report.problems.append(f"oracle fails visible tests:\n{vis.output_tail}")

        # --- null: untouched start tree + hidden tests -------------------------------
        null = Path(tmp) / "null" / "work"
        _copytree(bundle / "repo", null)
        _copytree(bundle / "hidden", null / "hidden_tests")
        res = run_pytest(null, "hidden_tests", timeout)
        report.null = {"passed": res.passed, "failed": res.failed, "rc": res.returncode}
        if res.returncode == 0:
            report.problems.append("null (untouched repo/) PASSES the hidden tests")
        must_fail = dict(unit_modules)
        if (bundle / "hidden" / "test_integration.py").is_file():
            must_fail["<integration>"] = "test_integration"
        for uid, module in must_fail.items():
            stats = res.by_file.get(module)
            # A module that errors at collection has no testcases with its name but
            # still counts as failing (pytest reports a collection error, rc != 0).
            if stats is not None and stats["failed"] == 0:
                report.problems.append(f"null: unit {uid} ({module}) passes without any work")
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
            for p in r.problems:
                print("     - " + p.replace("\n", "\n       "))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
