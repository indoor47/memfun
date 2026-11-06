"""Fail-closed: a code-writing run that changed no file is a failure, never a success.

Hermetic: no network, no GPU, no model.  Regression for the live B300 run that ended
``success True`` / ``iterations 0`` / ``files_created (none)`` after 36 minutes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.workflow import (
    NO_EDITS_PREFIX,
    ReviewResult,
    SubTaskStatus,
    WorkflowEngine,
    WorkflowResult,
    WorkflowState,
    no_edit_reason,
)
from memfun_cli.commands import swarm as swarm_mod
from memfun_core.types import TaskResult
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

# ── no_edit_reason ───────────────────────────────────────────────────────────────


def test_zero_op_coder_and_rlm_coder_are_no_edits() -> None:
    for agent in ("coder-agent", "test-agent", "rlm-coder"):
        reason = no_edit_reason(agent, {"iterations": 0, "ops": [], "files_created": []})
        assert reason is not None and reason.startswith(NO_EDITS_PREFIX)
    assert no_edit_reason("rlm-coder", None) is not None


def test_edit_only_and_write_runs_count_as_changes() -> None:
    edit_only = {"ops": [{"type": "edit", "target": "pkg/a.py", "detail": 10}], "iterations": 3}
    assert no_edit_reason("coder-agent", edit_only) is None
    tuples = {"ops": [("read", "a.py", 1), ("edit", "a.py", 9)]}
    assert no_edit_reason("rlm-coder", tuples) is None
    assert no_edit_reason("coder-agent", {"files_created": ["x.py"]}) is None


def test_reads_and_searches_alone_are_no_edits() -> None:
    data = {"ops": [{"type": "read", "target": "a.py"}, {"type": "grep", "target": "x"}]}
    assert no_edit_reason("coder-agent", data) is not None


def test_read_only_specialists_are_exempt() -> None:
    for agent in ("file-agent", "review-agent", "web-search-agent", "planner-agent",
                  "security-agent", "debug-agent"):  # fmt: skip
        assert no_edit_reason(agent, {"ops": [], "iterations": 0}) is None


def test_cmd_only_is_judged_by_git_status(tmp_path: Path) -> None:
    data = {"ops": [{"type": "cmd", "target": "ruff format .", "detail": 0}]}
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    assert no_edit_reason("coder-agent", data, tmp_path) is not None  # clean tree
    (tmp_path / "a.py").write_text("x = 1\n")
    assert no_edit_reason("coder-agent", data, tmp_path) is None  # dirty tree
    # git cannot judge (no root): benefit of the doubt, no false failure.
    assert no_edit_reason("coder-agent", data, None) is None


# ── _build_result ────────────────────────────────────────────────────────────────


def _state(agent: str, atype: str, data: dict[str, Any], success: bool = True) -> WorkflowState:
    st = WorkflowState(workflow_id="wf")
    sub = SubTaskStatus(task_id="wf_T1", sub_task=SubTask("T1", "do it", atype))
    sub.agent_name = agent
    sub.result = TaskResult("wf_T1", agent, success, data)
    st.sub_task_statuses["T1"] = sub
    return st


def _build(state: WorkflowState) -> WorkflowResult:
    return WorkflowEngine._build_result(
        state, ReviewResult(approved=True, issues=[], summary=""), 1.0
    )


def test_cmd_only_subtask_is_judged_in_its_own_worktree(tmp_path: Path) -> None:
    """Audit finding: with worktrees a cmd-only sub-task edits its worktree, not the root."""
    root, wt = tmp_path / "root", tmp_path / "wt"
    for d in (root, wt):
        d.mkdir()
        subprocess.run(("git", "init", "-q"), cwd=d, check=True)
    cmd_only = {"ops": [{"type": "cmd", "target": "sed -i s/a/b/ f.py", "detail": 0}]}
    (wt / "f.py").write_text("b\n")  # the sub-task's change lives in its worktree
    state = _state("coder-agent", "coder", cmd_only)
    kw: dict[str, Any] = {"require_per_task_review": False, "project_root": root}
    # clean root, changed worktree: accepted (was a false failure when judged in the root)
    assert WorkflowEngine._unaccepted_reason(state, "T1", worktrees={"T1": wt}, **kw) is None
    res = WorkflowEngine._build_result(
        state, ReviewResult(approved=True, issues=[], summary=""), 1.0,
        project_root=root, worktrees={"T1": wt},
    )  # fmt: skip
    assert res.success is True and res.no_edit_subtasks == 0
    # dirty root, untouched worktree: still a no-edit failure (was a false success)
    (root / "other.py").write_text("x\n")
    (wt / "f.py").unlink()
    assert WorkflowEngine._unaccepted_reason(state, "T1", worktrees={"T1": wt}, **kw) is not None
    # no worktree for the task: falls back to the root
    assert WorkflowEngine._unaccepted_reason(state, "T1", **kw) is None


def test_zero_op_coder_subtask_fails_workflow() -> None:
    res = _build(_state("coder-agent", "coder", {"answer": "done", "ops": [], "iterations": 5}))
    assert res.success is False and res.no_edits is True and res.no_edit_subtasks == 1
    assert res.unaccepted_tasks == ["T1"]
    assert res.error is not None and NO_EDITS_PREFIX in res.error


def test_edit_only_subtask_succeeds() -> None:
    ops = [{"type": "edit", "target": "stub.py", "detail": 3}]
    res = _build(_state("coder-agent", "coder", {"answer": "ok", "ops": ops, "iterations": 4}))
    assert res.success is True and res.no_edits is False and res.no_edit_subtasks == 0


def test_read_only_specialist_with_zero_edits_succeeds() -> None:
    res = _build(_state("file-agent", "file", {"answer": "listing", "ops": [], "iterations": 2}))
    assert res.success is True and res.no_edit_subtasks == 0


# ── single-task rlm-coder path ───────────────────────────────────────────────────


def _engine(result: TaskResult) -> WorkflowEngine:
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng._manager = MagicMock()
    eng._manager.is_running.return_value = True
    eng._orchestrator = MagicMock()
    eng._orchestrator.dispatch = AsyncMock(return_value=result)
    eng._project_root = None
    eng._emit_status = lambda _s: None  # type: ignore[method-assign]
    eng._run_auto_commit = lambda **_k: None  # type: ignore[method-assign]
    return eng


async def _single(result: TaskResult) -> WorkflowResult:
    import time

    eng = _engine(result)
    state = WorkflowState(workflow_id="wf")
    return await eng._execute_single_task(
        state, "implement every stub", "", None, time.perf_counter()
    )


@pytest.mark.asyncio
async def test_iterations_zero_rlm_coder_is_never_success() -> None:
    # The exact live-run shape: success True, iterations 0, no ops, escalate marker.
    data = {"answer": "", "iterations": 0, "ops": [], "files_created": [], "_escalate": True}
    res = await _single(TaskResult("wf_single", "rlm-coder", True, data))
    assert res.success is False and res.no_edits is True and res.no_edit_subtasks == 1
    assert res.error is not None and NO_EDITS_PREFIX in res.error
    assert res.sub_task_results["T1"].success is False


@pytest.mark.asyncio
async def test_iterations_zero_without_escalate_marker_still_fails() -> None:
    data = {"answer": "x", "iterations": 0, "ops": [], "files_created": []}
    res = await _single(TaskResult("wf_single", "rlm-coder", True, data))
    assert res.success is False and res.no_edits is True


@pytest.mark.asyncio
async def test_single_task_edit_only_succeeds() -> None:
    data = {"answer": "x", "iterations": 7, "ops": [{"type": "edit", "target": "a.py"}]}
    res = await _single(TaskResult("wf_single", "rlm-coder", True, data))
    assert res.success is True and res.no_edits is False


# ── swarm CLI: result JSON + exit code + panel ───────────────────────────────────


def _app() -> typer.Typer:
    app = typer.Typer()
    app.command("swarm")(swarm_mod.swarm_command)
    return app


def _invoke(result: WorkflowResult) -> Any:
    async def execute_workflow(**_kw: Any) -> WorkflowResult:
        return result

    inst = MagicMock()
    inst.execute_workflow = execute_workflow
    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()
    with (
        patch("memfun_agent.workflow.WorkflowEngine", return_value=inst),
        patch("memfun_runtime.builder.RuntimeBuilder", builder),
        patch("memfun_runtime.lifecycle.AgentManager", return_value=manager),
        patch("memfun_runtime.orchestrator.AgentOrchestrator", return_value=MagicMock()),
        patch("memfun_cli.commands.agent_commands._configure_dspy", lambda _c: None),
        patch(
            "memfun_cli.commands.chat._start_dashboard_background",
            AsyncMock(return_value=(None, False)),
        ),
    ):
        return CliRunner().invoke(_app(), ["build it", "--no-use-worktrees", "--no-commit"])


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(("git", "init", "-q", "-b", "main"), cwd=tmp_path, check=True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _no_edit_result() -> WorkflowResult:
    decomposition = DecompositionResult(
        sub_tasks=[SubTask("T1", "all", "coder")], shared_spec="",
        parallelism_groups=[["T1"]], is_single_task=True,
    )  # fmt: skip
    return WorkflowResult(
        workflow_id="wf-noedit",
        success=False,
        answer="",
        sub_task_results={
            "T1": TaskResult("t", "rlm-coder", False, {"iterations": 0, "ops": []}, error="e")
        },
        error=f"{NO_EDITS_PREFIX} (agent=rlm-coder, iterations=0, ops=0)",
        decomposition=decomposition,
        no_edit_subtasks=1,
        no_edits=True,
    )


def test_swarm_cli_reports_no_edits(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = project / "out" / "result.json"
    out.parent.mkdir()
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    res = _invoke(_no_edit_result())
    assert res.exit_code == 1, res.output
    assert "no_edits" in res.output and NO_EDITS_PREFIX in res.output
    doc = json.loads(out.read_text())
    assert doc["success"] is False and doc["no_edits"] is True and doc["no_edit_subtasks"] == 1
    assert NO_EDITS_PREFIX in doc["error"]


def test_swarm_cli_success_has_no_no_edits(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = project / "out" / "result.json"
    out.parent.mkdir()
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    ok = WorkflowResult(workflow_id="wf-ok", success=True, answer="", files_created=["a.py"])
    res = _invoke(ok)
    assert res.exit_code == 0, res.output
    doc = json.loads(out.read_text())
    assert doc["no_edits"] is False and doc["no_edit_subtasks"] == 0


# ── run_smoke runner ─────────────────────────────────────────────────────────────


def test_classify_no_edits_is_model_failure() -> None:
    assert rs.classify(no_edits=True, hidden_passed=0) == (
        "model_failure",
        "no file changes produced",
    )
    # A real harness fault still wins; and without the flag nothing changes.
    assert rs.classify(fatal="boom", no_edits=True)[0] == "harness_error"
    assert rs.classify(all_pass=True)[0] == "pass"


def test_aggregate_counts_no_edit_episodes() -> None:
    def ep(n: int, no_edits: bool) -> dict[str, Any]:
        return {"episode": f"b/{n}", "bundle": "b", "outcome": "model_failure",
                "no_edits": no_edits, "flags": []}  # fmt: skip

    agg = rs.aggregate([ep(1, True), ep(2, False), ep(3, True)], {})
    assert agg["overall"]["no_edit_episodes"] == 2
    assert agg["bundles"]["b"]["no_edit_episodes"] == 2
    assert "no_edit_episodes=2" in rs.summary_table(agg)
