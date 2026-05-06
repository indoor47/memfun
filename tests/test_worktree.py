"""Unit tests for :class:`memfun_runtime.worktree.WorktreeManager`.

Each test creates an isolated git repository under ``tmp_path`` so
the suite never touches the project's own working tree.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import TYPE_CHECKING

import pytest
from memfun_runtime.worktree import (
    WorktreeError,
    WorktreeInfo,
    WorktreeManager,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ───────────────────────────────────────────────────


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
    """Initialise a one-commit git repo in ``tmp_path``."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git("init", "-b", "main", cwd=repo_dir)
    _git("config", "user.email", "test@example.com", cwd=repo_dir)
    _git("config", "user.name", "Test", cwd=repo_dir)
    _git("config", "commit.gpgsign", "false", cwd=repo_dir)
    (repo_dir / "README.md").write_text("# initial\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo_dir)
    _git("commit", "-m", "init", cwd=repo_dir)
    return repo_dir


@pytest.fixture
def manager(repo: Path) -> WorktreeManager:
    return WorktreeManager(project_root=repo)


# ── make_worktree ──────────────────────────────────────────────


def test_make_worktree_creates_path(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """Creates a worktree at the expected location."""
    path = manager.make_worktree("wf1", "T1")

    assert path == (repo / ".memfun" / "worktrees" / "wf1" / "T1").resolve()
    assert path.is_dir()
    # README.md from the parent commit should be checked out.
    assert (path / "README.md").is_file()


def test_make_worktree_idempotent(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """Re-issuing ``make_worktree`` with the same args returns the same path."""
    first = manager.make_worktree("wf1", "T1")
    second = manager.make_worktree("wf1", "T1")
    assert first == second
    assert first.is_dir()


def test_make_worktree_at_specific_sha(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """Checking out a worktree at a specific SHA pins HEAD to that SHA."""
    # First commit SHA.
    sha1 = _git("rev-parse", "HEAD", cwd=repo).strip()

    # Add a second commit so HEAD moves forward.
    (repo / "two.txt").write_text("two\n", encoding="utf-8")
    _git("add", "two.txt", cwd=repo)
    _git("commit", "-m", "two", cwd=repo)
    sha2 = _git("rev-parse", "HEAD", cwd=repo).strip()
    assert sha1 != sha2

    # Worktree pinned to sha1 must report sha1 in its own HEAD.
    path = manager.make_worktree("wf1", "T1", base_sha=sha1)
    actual = _git("rev-parse", "HEAD", cwd=path).strip()
    assert actual == sha1


def test_make_worktree_rejects_invalid_ids(manager: WorktreeManager) -> None:
    """Invalid workflow / task / base_sha values raise :class:`WorktreeError`."""
    with pytest.raises(WorktreeError):
        manager.make_worktree("../escape", "T1")
    with pytest.raises(WorktreeError):
        manager.make_worktree("wf1", "../escape")
    with pytest.raises(WorktreeError):
        manager.make_worktree("wf1", "T1", base_sha="-rf")


# ── cleanup_worktree ───────────────────────────────────────────


def test_cleanup_worktree_removes_path_and_branch(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """``cleanup_worktree`` removes both the directory and the auto branch."""
    path = manager.make_worktree("wf1", "T1")
    assert path.is_dir()

    # Branch should exist beforehand.
    branches_before = _git("branch", "--list", cwd=repo)
    assert "memfun/wf1/T1" in branches_before

    manager.cleanup_worktree(path)

    assert not path.exists()
    branches_after = _git("branch", "--list", cwd=repo)
    assert "memfun/wf1/T1" not in branches_after

    # Idempotent — second call must not raise.
    manager.cleanup_worktree(path)


def test_cleanup_worktree_tolerates_missing_path(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """Cleaning up a non-existent worktree is a silent no-op."""
    bogus = repo / ".memfun" / "worktrees" / "wf-x" / "T-x"
    # Should not raise.
    manager.cleanup_worktree(bogus)


# ── 5-way fan-out ──────────────────────────────────────────────


def test_five_way_fan_out_isolated(
    manager: WorktreeManager,
    repo: Path,
) -> None:
    """Five worktrees coexist; each writes its own file; cleanup removes all."""
    paths: list[Path] = []
    for i in range(5):
        p = manager.make_worktree("wfA", f"T{i}")
        # Write a unique file in each worktree.
        (p / f"out_{i}.txt").write_text(f"hello-{i}", encoding="utf-8")
        paths.append(p)

    # All five exist as distinct directories.
    assert len(set(paths)) == 5
    for i, p in enumerate(paths):
        assert p.is_dir()
        assert (p / f"out_{i}.txt").read_text(encoding="utf-8") == f"hello-{i}"

    # list_worktrees reports all five (plus possibly the parent
    # repo, which is filtered out by base_dir scoping).
    listed = manager.list_worktrees()
    assert len({info.path for info in listed}) == 5
    for info in listed:
        assert isinstance(info, WorktreeInfo)
        assert info.path in set(paths)

    # Cleanup removes them all.
    for p in paths:
        manager.cleanup_worktree(p)
    for p in paths:
        assert not p.exists()
    assert manager.list_worktrees() == []


# ── list_worktrees scope ───────────────────────────────────────


def test_list_worktrees_ignores_unmanaged(
    manager: WorktreeManager,
    repo: Path,
    tmp_path: Path,
) -> None:
    """Worktrees outside ``.memfun/worktrees/`` are not reported."""
    foreign = tmp_path / "foreign-wt"
    _git(
        "worktree",
        "add",
        "-B",
        "foreign-branch",
        str(foreign),
        "HEAD",
        cwd=repo,
    )

    try:
        listed = manager.list_worktrees()
        # The foreign worktree must not appear.
        assert foreign.resolve() not in {info.path for info in listed}
    finally:
        _git("worktree", "remove", "--force", str(foreign), cwd=repo)
        _git("branch", "-D", "foreign-branch", cwd=repo)


# ── Specialist payload["cwd"] override ─────────────────────────


def test_specialist_handle_uses_payload_cwd(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_SpecialistBase.handle`` resolves output reads against ``payload["cwd"]``."""
    from unittest.mock import MagicMock

    from memfun_agent.specialists import FileAgent
    from memfun_core.types import TaskMessage

    # Run the specialist's handle() but stub the heavy dependencies
    # so we can assert the cwd plumbing in isolation.
    captured: dict[str, str] = {}

    async def fake_read_affected_files(files, project_root):  # type: ignore[no-untyped-def]
        captured["project_root"] = str(project_root)
        return ""

    async def fake_aforward(**kwargs):  # type: ignore[no-untyped-def]
        from memfun_agent.rlm import RLMResult

        return RLMResult(
            answer="ok",
            iterations=1,
            ops=[],
            files_created=[],
            success=True,
            trajectory=[],
            final_reasoning="",
        )

    monkeypatch.setattr(
        "memfun_agent.context_first.read_affected_files",
        fake_read_affected_files,
    )

    # Stub the RLMModule constructor so handle() doesn't try to
    # spin up DSPy.
    class _FakeRLM:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def aforward(self, **kwargs: object):  # type: ignore[no-untyped-def]
            return await fake_aforward(**kwargs)

    monkeypatch.setattr("memfun_agent.specialists.RLMModule", _FakeRLM)

    context = MagicMock()
    context.state_store = MagicMock()
    agent = FileAgent(context)
    # Skip the heavy on_start: just stub the bits handle() touches.
    agent._extra_tools = {}
    agent._spec_store = None  # type: ignore[assignment]

    # Build a TaskMessage where payload["cwd"] points at our isolated worktree.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    (isolated / "stub.txt").write_text("hi", encoding="utf-8")

    task = TaskMessage(
        task_id="T1",
        agent_id="file-agent",
        payload={
            "type": "ask",
            "query": "test",
            "context": "ctx",
            "outputs": ["stub.txt"],
            "cwd": str(isolated),
            "sub_task_id": "T1",
        },
    )

    asyncio.run(agent.handle(task))

    # The handle() override must have used payload["cwd"], not getcwd().
    assert captured.get("project_root") == str(isolated)


# ── OperationExecutor base_dir confinement ────────────────────


def test_operation_executor_base_dir_writes_inside(
    tmp_path: Path,
) -> None:
    """Writing a relative path inside ``base_dir`` succeeds."""
    from memfun_agent.context_first import OperationExecutor

    base = tmp_path / "wt"
    base.mkdir()
    executor = OperationExecutor(
        project_root=tmp_path,
        base_dir=base,
    )

    asyncio.run(
        executor.execute(
            [
                {"op": "write_file", "path": "subdir/foo.py", "content": "x = 1\n"},
            ]
        )
    )

    target = base / "subdir" / "foo.py"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "x = 1\n"
    assert executor.failed == 0


def test_operation_executor_base_dir_blocks_escape(
    tmp_path: Path,
) -> None:
    """Writing through ``..`` outside ``base_dir`` is rejected."""
    from memfun_agent.context_first import OperationExecutor

    base = tmp_path / "wt"
    base.mkdir()
    outside = tmp_path / "escape.txt"
    executor = OperationExecutor(
        project_root=tmp_path,
        base_dir=base,
    )

    asyncio.run(
        executor.execute(
            [
                {"op": "write_file", "path": "../escape.txt", "content": "boom"},
            ]
        )
    )

    assert not outside.exists()
    assert executor.failed == 1
    assert executor.files_created == []


def test_operation_executor_no_base_dir_unchanged(
    tmp_path: Path,
) -> None:
    """When ``base_dir`` is unset, behaviour matches legacy ``Path.abspath``."""
    from memfun_agent.context_first import OperationExecutor

    target = tmp_path / "out.py"
    executor = OperationExecutor(project_root=tmp_path)

    asyncio.run(
        executor.execute(
            [
                {"op": "write_file", "path": str(target), "content": "y = 2\n"},
            ]
        )
    )

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "y = 2\n"
    assert executor.failed == 0
