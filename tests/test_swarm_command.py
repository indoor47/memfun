"""Tests for the ``memfun swarm`` command.

These tests do **not** spin up an LLM or a real runtime — they mock
``WorkflowEngine.execute_workflow`` and assert that the swarm command:

1. Invokes the WorkflowEngine path directly (bypassing triage / the
   context-first solver).
2. Constructs a :class:`WorktreeManager` when ``--use-worktrees`` is
   set (and **does not** when ``--no-use-worktrees`` is set).
3. Plumbs ``--max-pool`` into the orchestrator config.
4. Errors cleanly when ``--use-worktrees`` is requested on a non-git
   project.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from memfun_cli.commands.swarm import swarm_command
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _build_app() -> typer.Typer:
    """Build a tiny Typer app exposing only the swarm command."""
    app = typer.Typer()
    app.command("swarm")(swarm_command)
    return app


@pytest.fixture
def git_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmpdir that is a real git working tree.

    The swarm CLI uses ``git rev-parse --is-inside-work-tree`` to test
    for a repo (so it works for worktrees + submodules), so a bare
    ``.git/`` mkdir is not enough — we actually initialise one.
    """
    import subprocess

    subprocess.run(
        ("git", "init", "-q", "-b", "main"), cwd=str(tmp_path), check=True,
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Avoid leaking the developer's real ``~/.memfun/`` into the test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _make_engine_mock() -> MagicMock:
    """Build a mock WorkflowEngine class.

    Returns the class mock; the instance is accessed via
    ``engine_cls.return_value``.
    """
    from memfun_agent.workflow import WorkflowResult

    instance = MagicMock()
    instance.execute_workflow = AsyncMock(
        return_value=WorkflowResult(
            workflow_id="wf-test-123",
            success=True,
            answer="Swarm completed.",
            sub_task_results={},
            total_duration_ms=1234.0,
        )
    )
    cls = MagicMock(return_value=instance)
    return cls


# ── Core behaviour ─────────────────────────────────────────────────


def test_swarm_invokes_workflow_engine_directly(
    git_project: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``memfun swarm "..."`` calls ``WorkflowEngine.execute_workflow``
    without going through QueryTriage or the context-first solver.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    engine_cls = _make_engine_mock()

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()

    with (
        patch("memfun_agent.workflow.WorkflowEngine", engine_cls),
        patch(
            "memfun_runtime.builder.RuntimeBuilder", builder,
        ),
        patch(
            "memfun_runtime.lifecycle.AgentManager",
            return_value=manager,
        ),
        patch(
            "memfun_runtime.orchestrator.AgentOrchestrator",
            return_value=MagicMock(),
        ),
        patch(
            "memfun_cli.commands.agent_commands._configure_dspy",
            lambda _cfg: None,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            ["build a thing", "--no-use-worktrees"],
        )

    assert result.exit_code == 0, result.output
    assert engine_cls.return_value.execute_workflow.await_count == 1
    # Verify the spec was passed through verbatim.
    call_kwargs = engine_cls.return_value.execute_workflow.await_args.kwargs
    assert call_kwargs["task_description"] == "build a thing"


def test_swarm_constructs_worktree_manager_when_enabled(
    git_project: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--use-worktrees`` causes a WorktreeManager to be created and
    threaded into the WorkflowEngine.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    engine_cls = _make_engine_mock()
    worktree_cls = MagicMock()
    worktree_cls.return_value = MagicMock()

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()

    with (
        patch("memfun_agent.workflow.WorkflowEngine", engine_cls),
        patch(
            "memfun_runtime.worktree.WorktreeManager", worktree_cls,
        ),
        patch(
            "memfun_runtime.builder.RuntimeBuilder", builder,
        ),
        patch(
            "memfun_runtime.lifecycle.AgentManager",
            return_value=manager,
        ),
        patch(
            "memfun_runtime.orchestrator.AgentOrchestrator",
            return_value=MagicMock(),
        ),
        patch(
            "memfun_cli.commands.agent_commands._configure_dspy",
            lambda _cfg: None,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            ["build a thing"],
        )

    assert result.exit_code == 0, result.output
    # Default is --use-worktrees, so WorktreeManager should have been
    # constructed with project_root pointing at our git_project.
    assert worktree_cls.call_count == 1
    kwargs = worktree_cls.call_args.kwargs
    args = worktree_cls.call_args.args
    project_root_arg = kwargs.get("project_root") or (
        args[0] if args else None
    )
    assert project_root_arg is not None
    # The engine receives the same worktree manager instance.
    engine_kwargs = engine_cls.call_args.kwargs
    assert "worktree_manager" in engine_kwargs
    assert engine_kwargs["worktree_manager"] is worktree_cls.return_value


def test_swarm_no_worktrees_when_disabled(
    git_project: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-use-worktrees`` skips WorktreeManager construction
    entirely and passes ``worktree_manager=None`` to the engine.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    engine_cls = _make_engine_mock()
    worktree_cls = MagicMock()

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()

    with (
        patch("memfun_agent.workflow.WorkflowEngine", engine_cls),
        patch(
            "memfun_runtime.worktree.WorktreeManager", worktree_cls,
        ),
        patch(
            "memfun_runtime.builder.RuntimeBuilder", builder,
        ),
        patch(
            "memfun_runtime.lifecycle.AgentManager",
            return_value=manager,
        ),
        patch(
            "memfun_runtime.orchestrator.AgentOrchestrator",
            return_value=MagicMock(),
        ),
        patch(
            "memfun_cli.commands.agent_commands._configure_dspy",
            lambda _cfg: None,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            ["build a thing", "--no-use-worktrees"],
        )

    assert result.exit_code == 0, result.output
    assert worktree_cls.call_count == 0
    engine_kwargs = engine_cls.call_args.kwargs
    assert engine_kwargs.get("worktree_manager") is None


def test_swarm_plumbs_max_pool_into_orchestrator(
    git_project: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--max-pool=3`` reaches the OrchestratorConfig and the engine's
    ``_max_pool`` hint that ``_execute_group`` reads.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    engine_cls = _make_engine_mock()

    captured: dict[str, Any] = {}

    def orch_factory(*args: Any, **kwargs: Any) -> MagicMock:
        captured["config"] = kwargs.get("config")
        return MagicMock()

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.stop_all = AsyncMock()

    with (
        patch("memfun_agent.workflow.WorkflowEngine", engine_cls),
        patch(
            "memfun_runtime.builder.RuntimeBuilder", builder,
        ),
        patch(
            "memfun_runtime.lifecycle.AgentManager",
            return_value=manager,
        ),
        patch(
            "memfun_runtime.orchestrator.AgentOrchestrator",
            side_effect=orch_factory,
        ),
        patch(
            "memfun_cli.commands.agent_commands._configure_dspy",
            lambda _cfg: None,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            [
                "build a thing",
                "--no-use-worktrees", "--max-pool=3",
            ],
        )

    assert result.exit_code == 0, result.output
    orch_config = captured.get("config")
    assert orch_config is not None
    # max_concurrency should be at least max_pool (the orchestrator
    # semaphore cap).
    assert orch_config.max_concurrency >= 3
    # Engine should have received the per-pool hint.
    instance = engine_cls.return_value
    assert getattr(instance, "_max_pool", None) == 3


def test_swarm_rejects_use_worktrees_outside_git_repo(
    tmp_path: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the project isn't a git repo, ``--use-worktrees`` errors
    cleanly (exit 2) without calling the WorkflowEngine.
    """
    monkeypatch.chdir(tmp_path)  # no .git here
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    engine_cls = _make_engine_mock()

    builder = MagicMock()
    builder.return_value.build = AsyncMock(return_value=MagicMock())

    with (
        patch("memfun_agent.workflow.WorkflowEngine", engine_cls),
        patch(
            "memfun_runtime.builder.RuntimeBuilder", builder,
        ),
        patch(
            "memfun_cli.commands.agent_commands._configure_dspy",
            lambda _cfg: None,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            ["build a thing"],  # --use-worktrees is default
        )

    assert result.exit_code == 2, result.output
    assert "requires a git repository" in result.output.lower()
    # The engine was never constructed.
    assert engine_cls.call_count == 0
