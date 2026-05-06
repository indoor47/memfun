"""Integration tests: :class:`WorkflowEngine` + :class:`WorktreeManager`.

We don't spin up real specialist agents — that's covered by the
end-to-end suite — but we exercise the full ``_execute_group`` path
to verify each fan-out task is provisioned its own isolated git
worktree and the path is injected as ``payload["cwd"]``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from memfun_agent.decomposer import (
    DecompositionResult,
    SubTask,
)
from memfun_agent.workflow import WorkflowEngine, WorkflowState
from memfun_core.types import TaskMessage, TaskResult
from memfun_runtime.worktree import WorktreeManager

# ── Helpers ────────────────────────────────────────────────────


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialise a one-commit git repo."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git("init", "-b", "main", cwd=repo_dir)
    _git("config", "user.email", "test@example.com", cwd=repo_dir)
    _git("config", "user.name", "Test", cwd=repo_dir)
    _git("config", "commit.gpgsign", "false", cwd=repo_dir)
    (repo_dir / "README.md").write_text("hi\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo_dir)
    _git("commit", "-m", "init", cwd=repo_dir)
    return repo_dir


def _make_engine(
    repo: Path,
    *,
    worktree_manager: WorktreeManager | None,
) -> tuple[WorkflowEngine, list[TaskMessage]]:
    """Build a WorkflowEngine with a captured-payload mock orchestrator."""
    captured: list[TaskMessage] = []

    async def fake_fan_out(
        tasks: list[TaskMessage],
        agent_names: list[str],
        *,
        timeout: float = 600.0,
    ) -> list[TaskResult]:
        # Save the actual TaskMessages so the test can inspect them.
        captured.extend(tasks)
        # Each "specialist" simulates writing one file inside its
        # provided cwd, so we can later assert isolation.
        results: list[TaskResult] = []
        for t in tasks:
            cwd = t.payload.get("cwd")
            if cwd:
                target = Path(cwd) / f"out_{t.payload.get('sub_task_id', 't')}.txt"
                target.write_text(
                    f"from-{t.payload.get('sub_task_id', 't')}",
                    encoding="utf-8",
                )
            results.append(
                TaskResult(
                    task_id=t.task_id,
                    agent_id=t.agent_id,
                    success=True,
                    result={"answer": "ok", "files_created": [], "ops": []},
                    duration_ms=0.0,
                )
            )
        return results

    orchestrator = MagicMock()
    orchestrator.fan_out = AsyncMock(side_effect=fake_fan_out)
    orchestrator.dispatch = AsyncMock(
        return_value=TaskResult(
            task_id="x",
            agent_id="x",
            success=True,
            result={"answer": "ok"},
            duration_ms=0.0,
        ),
    )

    manager = MagicMock()
    manager.is_running = MagicMock(return_value=True)
    manager.start_agent = AsyncMock()

    context = MagicMock()
    context.state_store = MagicMock()
    context.event_bus = MagicMock()
    context.event_bus.publish = AsyncMock()

    engine = WorkflowEngine(
        context=context,
        orchestrator=orchestrator,
        manager=manager,
        worktree_manager=worktree_manager,
    )
    return engine, captured


# ── Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_group_provisions_worktree_per_task(
    repo: Path,
) -> None:
    """Each fan-out task gets a distinct cwd and writes only there."""
    wt_manager = WorktreeManager(project_root=repo)
    engine, captured = _make_engine(repo, worktree_manager=wt_manager)

    workflow_id = "wfit1"
    state = WorkflowState(workflow_id=workflow_id)

    sub_tasks = [
        SubTask(
            id=f"T{i}",
            description=f"task {i}",
            agent_type="coder",
            inputs=[],
            outputs=[],
            depends_on=[],
        )
        for i in range(3)
    ]
    decomposition = DecompositionResult(
        sub_tasks=sub_tasks,
        parallelism_groups=[[st.id for st in sub_tasks]],
        shared_spec="",
        is_single_task=False,
    )
    state.decomposition = decomposition
    for st in sub_tasks:
        from memfun_agent.workflow import SubTaskStatus

        state.sub_task_statuses[st.id] = SubTaskStatus(
            task_id=f"{workflow_id}_{st.id}",
            sub_task=st,
            agent_name="coder-agent",
        )

    await engine._execute_group(
        state,
        [st.id for st in sub_tasks],
        "ctx",
        history=None,
    )

    # Each task's payload must carry a distinct cwd.
    cwds = [t.payload.get("cwd") for t in captured]
    assert len(cwds) == 3
    assert all(c is not None for c in cwds)
    assert len(set(cwds)) == 3, f"expected unique cwds, got {cwds}"

    # Files must land inside their respective worktrees only.
    for st in sub_tasks:
        wt_path = (repo / ".memfun" / "worktrees" / workflow_id / st.id).resolve()
        out = wt_path / f"out_{st.id}.txt"
        assert out.is_file(), f"missing {out}"
        assert out.read_text(encoding="utf-8") == f"from-{st.id}"

    # No file leaks into the parent repo.
    for st in sub_tasks:
        assert not (repo / f"out_{st.id}.txt").exists()

    # Cleanup empties everything.
    engine.cleanup_worktrees(workflow_id)
    for st in sub_tasks:
        wt_path = (repo / ".memfun" / "worktrees" / workflow_id / st.id).resolve()
        assert not wt_path.exists()


@pytest.mark.asyncio
async def test_execute_group_without_manager_omits_cwd(repo: Path) -> None:
    """When no WorktreeManager is configured, ``cwd`` is not injected."""
    engine, captured = _make_engine(repo, worktree_manager=None)

    workflow_id = "wfit2"
    state = WorkflowState(workflow_id=workflow_id)
    sub_task = SubTask(
        id="T1",
        description="lone task",
        agent_type="coder",
        inputs=[],
        outputs=[],
        depends_on=[],
    )
    state.decomposition = DecompositionResult(
        sub_tasks=[sub_task],
        parallelism_groups=[["T1"]],
        shared_spec="",
        is_single_task=False,
    )
    from memfun_agent.workflow import SubTaskStatus

    state.sub_task_statuses["T1"] = SubTaskStatus(
        task_id=f"{workflow_id}_T1",
        sub_task=sub_task,
        agent_name="coder-agent",
    )

    await engine._execute_group(
        state,
        ["T1"],
        "ctx",
        history=None,
    )

    assert len(captured) == 1
    payload: dict[str, Any] = dict(captured[0].payload)
    assert "cwd" not in payload, "cwd must not be injected when worktree_manager is None"

    # And no .memfun/worktrees directory should appear.
    assert not (repo / ".memfun" / "worktrees").exists()
