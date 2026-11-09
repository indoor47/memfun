"""The grading wrapper (``evals/swebench_live/grade.py``): hermetic, the evaluator is faked."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evals.swebench_live import grade as gr


def fake_evaluator(monkeypatch, *, results: dict[str, Any] | None, exit_code: int) -> list[Any]:
    """Replace ``subprocess.run``: the clone is at the pinned commit; the evaluator writes
    *results* (when given) to its ``--output_dir`` and exits with *exit_code*."""
    calls: list[Any] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=gr.EVALUATOR_COMMIT + "\n")
        out = cmd[cmd.index("--output_dir") + 1]
        if results is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "results.json").write_text(json.dumps(results), "utf-8")
        if exit_code and kw.get("check"):
            raise subprocess.CalledProcessError(exit_code, cmd)
        return subprocess.CompletedProcess(cmd, exit_code)

    monkeypatch.setattr(gr.subprocess, "run", run)
    return calls


def test_gold_results_written_before_the_evaluators_known_crash_are_used(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    good = {"success_ids": ["a"], "failure_ids": ["b"], "success": 1, "failure": 1}
    calls = fake_evaluator(monkeypatch, results=good, exit_code=1)
    res = gr.evaluate(tmp_path, "gold", tmp_path / "out", "full", 1, ["a", "b"], False)
    assert json.loads(res.read_text("utf-8")) == good
    assert "known gold-mode defect" in capsys.readouterr().out
    assert "--instance_ids" in calls[-1] and "--remove-image" not in calls[-1]


def test_any_other_evaluator_failure_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    good = {"success_ids": ["a"], "failure_ids": []}
    fake_evaluator(monkeypatch, results=good, exit_code=1)
    with pytest.raises(subprocess.CalledProcessError):  # not gold mode: no excuse
        gr.evaluate(
            tmp_path, str(tmp_path / "preds.json"), tmp_path / "o1", "full", 1, ["a"], True
        )
    with pytest.raises(subprocess.CalledProcessError):  # gold, but one instance has no verdict
        gr.evaluate(tmp_path, "gold", tmp_path / "o2", "full", 1, ["a", "b"], True)
    fake_evaluator(monkeypatch, results=None, exit_code=1)
    with pytest.raises(subprocess.CalledProcessError):  # gold, nothing written
        gr.evaluate(tmp_path, "gold", tmp_path / "o3", "full", 1, ["a"], True)


def test_stale_results_are_not_taken_for_this_run(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"success_ids": ["a"]}), "utf-8")
    now = path.stat().st_mtime
    assert gr.results_complete(path, ["a"], now)
    assert not gr.results_complete(path, ["a"], now + 60)  # written before this run started
    assert not gr.results_complete(path, [], now) and not gr.results_complete(
        tmp_path / "x", ["a"], 0
    )


def test_the_clone_must_be_at_the_pinned_commit(tmp_path: Path, monkeypatch) -> None:
    def run(cmd: list[str], **_kw: Any) -> Any:
        return subprocess.CompletedProcess(cmd, 0, stdout="0" * 40 + "\n")

    monkeypatch.setattr(gr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="not the pinned"):
        gr.evaluate(tmp_path, "gold", tmp_path / "out", "full", 1, ["a"], True)


def test_a_run_that_is_not_finalized_is_never_graded(tmp_path: Path, monkeypatch) -> None:
    calls = fake_evaluator(monkeypatch, results={"success_ids": ["a"]}, exit_code=0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"finalized": False, "instance_ids": ["a"]}))
    with pytest.raises(SystemExit, match="not finalized"):
        gr.main(["grade", "--dir", str(tmp_path), "--run-dir", str(run_dir), "--split", "full"])
    assert calls == []  # the evaluator was never started
    (run_dir / "run.json").write_text(json.dumps({"finalized": True, "instance_ids": ["a"]}))
    assert (
        gr.main(["grade", "--dir", str(tmp_path), "--run-dir", str(run_dir), "--split", "full"])
        == 0
    )
    assert json.loads((run_dir / "results.json").read_text("utf-8")) == {"success_ids": ["a"]}
