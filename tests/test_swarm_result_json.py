"""``MEMFUN_SWARM_RESULT_JSON``: the opt-in machine-readable result of ``memfun swarm``.

Hermetic: the WorkflowEngine is mocked the same way as in ``test_swarm_command.py``.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.workflow import SubTaskStatus, WorkflowResult
from memfun_cli.commands import swarm as swarm_mod
from memfun_core.types import TaskResult
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _app() -> typer.Typer:
    app = typer.Typer()
    app.command("swarm")(swarm_mod.swarm_command)
    return app


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(("git", "init", "-q", "-b", "main"), cwd=tmp_path, check=True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _result() -> WorkflowResult:
    ops_t1: list[Any] = [
        {"type": "grep", "target": "def slugify", "detail": "3"},
        {"type": "read", "target": "textkit/slugify.py", "detail": "120"},
        {"type": "edit", "target": "textkit/slugify.py", "detail": "x" * 1000},
    ]
    ops_t2: list[Any] = [("find_files", "*.py", 4), ("write", "textkit/wrap.py", 50)]
    decomposition = DecompositionResult(
        sub_tasks=[SubTask("T1", "slugify", "coder"), SubTask("T2", "wrap", "coder")],
        shared_spec="",
        parallelism_groups=[["T1", "T2"]],
        is_single_task=False,
    )
    return WorkflowResult(
        workflow_id="wf-json-1",
        success=False,
        answer="",
        sub_task_results={
            "T1": TaskResult(
                "wf_T1",
                "coder-agent",
                True,
                {"ops": ops_t1, "iterations": 4, "files_created": ["textkit/slugify.py"]},
            ),
            "T2": TaskResult("wf_T2", "coder-agent", False, {"ops": ops_t2}, error="boom"),
        },
        review_rounds=1,
        files_created=["textkit/slugify.py"],
        decomposition=decomposition,
        unaccepted_tasks=["T2"],
        review_errors=1,
    )


def _no_dspy(_cfg: object) -> None:
    return None


def _invoke(raise_exc: bool = False) -> Any:
    """Run ``memfun swarm`` with a fake engine that reports T1 / T2 status transitions."""
    captured: dict[str, Any] = {}

    def engine_factory(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        inst = MagicMock()

        async def execute_workflow(**_kw: Any) -> WorkflowResult:
            cb = captured["on_sub_task_status"]
            for tid in ("T1", "T2"):
                st = SubTaskStatus(task_id=f"wf_{tid}", sub_task=SubTask(tid, tid, "coder"))
                st.agent_name, st.status, st.started_at, st.completed_at = (
                    "coder-agent",
                    "completed",
                    10.0,
                    20.0,
                )
                cb(tid, st)
            if raise_exc:
                raise RuntimeError("decomposer exploded")
            return _result()

        inst.execute_workflow = execute_workflow
        return inst

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()
    with (
        patch("memfun_agent.workflow.WorkflowEngine", side_effect=engine_factory),
        patch("memfun_runtime.builder.RuntimeBuilder", builder),
        patch("memfun_runtime.lifecycle.AgentManager", return_value=manager),
        patch("memfun_runtime.orchestrator.AgentOrchestrator", return_value=MagicMock()),
        patch("memfun_cli.commands.agent_commands._configure_dspy", _no_dspy),
        patch(
            "memfun_cli.commands.chat._start_dashboard_background",
            AsyncMock(return_value=(None, False)),
        ),
    ):
        return CliRunner().invoke(_app(), ["build it", "--no-use-worktrees", "--no-commit"])


def test_result_json_content_and_schema(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = project / "out" / "result.json"
    out.parent.mkdir()
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    res = _invoke()
    assert res.exit_code == 1, res.output  # the workflow reported failure
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == 1
    assert doc["workflow_id"] == "wf-json-1" and doc["success"] is False
    assert doc["unaccepted_tasks"] == ["T2"] and doc["review_errors"] == 1
    assert doc["decomposition"] == {"n_sub_tasks": 2, "is_single_task": False,
                                    "groups": [["T1", "T2"]]}  # fmt: skip
    t1, t2 = doc["sub_tasks"]["T1"], doc["sub_tasks"]["T2"]
    assert t1["agent"] == "coder-agent" and t1["success"] is True and t1["iterations"] == 4
    assert [o["type"] for o in t1["ops"]] == ["grep", "read", "edit"]
    assert len(t1["ops"][2]["detail"]) == 200  # truncated detail
    assert t1["started_at"] == 10.0 and t1["completed_at"] == 20.0
    assert t2["success"] is False and t2["error"] == "boom"
    assert t2["ops"][0] == {"type": "find_files", "target": "*.py", "detail": "4"}
    assert doc["ops_total"] == {"grep": 1, "read": 1, "edit": 1, "find_files": 1, "write": 1}
    assert "prompt" not in json.dumps(doc)
    assert sorted(p.name for p in out.parent.iterdir()) == ["result.json"]  # no temp left


def test_result_json_written_when_the_workflow_raises(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = project / "result.json"
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    res = _invoke(raise_exc=True)
    assert res.exit_code == 1
    doc = json.loads(out.read_text())
    assert doc["success"] is False and "decomposer exploded" in doc["error"]
    assert set(doc["sub_tasks"]) == {"T1", "T2"}


def test_env_unset_writes_nothing_and_output_is_unchanged(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(swarm_mod.RESULT_JSON_ENV, raising=False)
    before = sorted(p.name for p in project.iterdir())
    res = _invoke()
    assert res.exit_code == 1
    assert sorted(p.name for p in project.iterdir()) == before
    assert "Swarm result" in res.output


def test_planted_symlink_is_refused_and_never_breaks_the_swarm(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    victim = project / "victim.txt"
    victim.write_text("keep")
    out = project / "result.json"
    out.symlink_to(victim)
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    res = _invoke()
    assert res.exit_code == 1  # same exit code as without the hook
    assert victim.read_text() == "keep" and out.is_symlink()
    assert any(swarm_mod.RESULT_JSON_ENV in r.getMessage() for r in caplog.records)


def test_unwritable_target_logs_a_warning_only(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(project / "no-such-dir" / "r.json"))
    res = _invoke()
    assert res.exit_code == 1 and "Swarm result" in res.output
    assert any(swarm_mod.RESULT_JSON_ENV in r.getMessage() for r in caplog.records)


def test_ops_are_capped_per_sub_task() -> None:
    result = _result()
    result.sub_task_results["T1"].result["ops"] = [("read", "f", 1)] * 2500
    doc = swarm_mod._swarm_result_payload(result, 1.0, {})  # pyright: ignore[reportPrivateUsage]
    t1 = doc["sub_tasks"]["T1"]
    assert len(t1["ops"]) == 2000 and t1["truncated"] is True and t1["ops_count"] == 2500
    assert doc["ops_total"]["read"] == 2500
