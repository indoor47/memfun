"""The smoke task bundles are valid: oracle passes all hidden tests, null fails them.

Runs ``evals/swarm/validate_bundles.py`` in a subprocess (each bundle run is itself a
pytest subprocess with a generous timeout).  No wall-clock assertions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "evals" / "swarm" / "validate_bundles.py"
SMOKE = REPO_ROOT / "evals" / "swarm" / "smoke"


def _validate(*roots: Path) -> tuple[int, list[dict[str, Any]]]:
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), *map(str, roots), "--json", "--timeout", "300"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_all_smoke_bundles_valid() -> None:
    rc, reports = _validate(SMOKE)
    bad = {r["id"]: r["problems"] for r in reports if not r["ok"]}
    assert rc == 0, bad
    ids = {r["id"] for r in reports}
    assert len(ids) >= 6
    tasks = [p.parent.name for p in SMOKE.glob("*/task.toml")]
    assert sorted(ids) == sorted(tasks)


def test_smoke_set_has_fanout_and_control() -> None:
    import tomllib

    kinds: dict[str, int] = {}
    for toml in SMOKE.glob("*/task.toml"):
        task = tomllib.loads(toml.read_text())
        kinds[task["kind"]] = kinds.get(task["kind"], 0) + 1
        if task["kind"] == "fanout":
            assert 2 <= len(task["units"]) <= 4
    assert kinds.get("fanout", 0) >= 5 and kinds.get("control", 0) >= 1


def test_validator_flags_broken_oracle_and_trivial_null(tmp_path: Path) -> None:
    broken = tmp_path / "formats-2"
    shutil.copytree(SMOKE / "formats-2", broken)
    (broken / "solution" / "formats" / "duration.py").unlink()  # oracle loses one unit

    trivial = tmp_path / "textkit-3"
    shutil.copytree(SMOKE / "textkit-3", trivial)
    shutil.copy(  # start tree already contains the roman solution: null passes that unit
        trivial / "solution" / "textkit" / "roman.py", trivial / "repo" / "textkit" / "roman.py"
    )

    rc, reports = _validate(broken, trivial)
    assert rc == 1
    by_id = {r["id"]: r for r in reports}
    assert not by_id["formats-2"]["ok"]
    assert any("oracle" in p for p in by_id["formats-2"]["problems"])
    assert not by_id["textkit-3"]["ok"]
    assert any("unit roman" in p for p in by_id["textkit-3"]["problems"])


def _load_validator() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_bundles", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_bundles"] = module
    spec.loader.exec_module(module)
    return module


EAGER_MAIN = """
import argparse, importlib, sys

COMMANDS = ("stats", "convert", "hashsum")


def build_parser():
    parser = argparse.ArgumentParser(prog="minicli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        importlib.import_module(f"minicli.commands.{name}").register(sub)
    return parser


def main(argv=None, out=None):
    args = build_parser().parse_args(argv)
    return int(args.func(args, out or sys.stdout))
"""


def test_validator_flags_units_that_are_not_independent(tmp_path: Path) -> None:
    eager = tmp_path / "minicli-3"
    shutil.copytree(SMOKE / "minicli-3", eager)
    (eager / "repo" / "minicli" / "main.py").write_text(EAGER_MAIN)  # imports every command
    rc, reports = _validate(eager)
    assert rc == 1
    problems = reports[0]["problems"]
    assert any(p.startswith("independence: breaking stats") for p in problems), problems
    assert reports[0]["independence"]["stats"] != "ok"


def test_grader_ignores_planted_conftest_and_ini(tmp_path: Path) -> None:
    vb = _load_validator()
    bundle = SMOKE / "textkit-3"
    tree = vb.graded_tree(bundle, [], None, tmp_path / "tree")  # null: untouched repo
    # A cheating agent plants files that would make every test pass if they were loaded.
    (tree / "conftest.py").write_text(
        "import pytest\n"
        "@pytest.hookimpl(hookwrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n"
        "    outcome.get_result().outcome = 'passed'\n"
    )
    (tree / "pytest.ini").write_text("[pytest]\naddopts = -p no:terminal --co\n")
    (tree / "sitecustomize.py").write_text("raise SystemExit(0)\n")
    res = vb.run_grader(tree, bundle / "hidden", tmp_path / "scratch", 300)
    assert res.returncode != 0
    assert res.failed > 0 and res.passed == 0


def test_graded_tree_takes_only_owned_files(tmp_path: Path) -> None:
    vb = _load_validator()
    bundle = SMOKE / "textkit-3"
    cand = tmp_path / "cand"
    shutil.copytree(bundle / "solution", cand)
    (cand / "conftest.py").write_text("raise RuntimeError('planted')\n")
    (cand / "textkit" / "extra.py").write_text("x = 1\n")
    tree = vb.graded_tree(bundle, ["textkit/roman.py"], cand, tmp_path / "tree")
    assert not (tree / "conftest.py").exists() and not (tree / "textkit" / "extra.py").exists()
    assert (tree / "textkit" / "roman.py").read_text() == (
        cand / "textkit" / "roman.py"
    ).read_text()
    assert "NotImplementedError" in (tree / "textkit" / "wrap.py").read_text()  # not owned here
