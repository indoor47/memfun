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
