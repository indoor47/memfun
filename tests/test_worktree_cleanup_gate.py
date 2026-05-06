"""Regression tests for the ``cleanup_worktree`` managed-root gate.

Issue #18 (HIGH foot-gun finding from the PR #17 audit):
``WorktreeManager.cleanup_worktree(path)`` previously removed
*whatever* path the caller passed in, even one outside
``<project_root>/.memfun/worktrees/``.  Branch deletion was already
correctly gated by ``_branch_from_path`` (which refuses branches
outside the ``memfun/<workflow>/<task>`` namespace), but the
worktree directory itself was not.

These tests pin the new behaviour:

* ``cleanup_worktree(<managed>)`` paths still work (back-compat).
* Any path outside the managed root — including symlink-targets and
  ``..``-escapes — is rejected with :class:`WorktreeError`, BEFORE
  any filesystem or git side-effect.
* ``cleanup_worktrees(workflow_id)`` rejects ``workflow_id`` values
  that would escape the managed root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from memfun_runtime.worktree import WorktreeError, WorktreeManager

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


# ── Allowed: managed paths still work ──────────────────────────


def test_cleanup_managed_path_succeeds(
    manager: WorktreeManager, repo: Path,
) -> None:
    """Back-compat: cleanup of a real managed worktree still works."""
    path = manager.make_worktree("wf1", "T1")
    assert path.is_dir()

    # Must not raise — the path lives inside the managed root.
    manager.cleanup_worktree(path)
    assert not path.exists()


# ── Reject: paths outside the managed root ─────────────────────


def test_reject_external_dir(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """A path under ``/tmp`` (or any sibling of the repo) is rejected.

    The directory must NOT be removed even though it exists and the
    caller is asking for cleanup.
    """
    external = tmp_path / "some-other-dir"
    external.mkdir()
    (external / "user_file.txt").write_text("important", encoding="utf-8")

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(external)

    # Critical assertion: the directory survived the rejection.
    assert external.is_dir()
    assert (external / "user_file.txt").read_text(encoding="utf-8") == "important"


def test_reject_parent_of_managed(
    manager: WorktreeManager, repo: Path,
) -> None:
    """``<project_root>/.memfun`` is the *parent* of the managed root.

    It must be rejected — destroying it would wipe every workflow's
    worktrees plus other ``.memfun/`` subdirectories.
    """
    parent = repo / ".memfun"
    parent.mkdir(exist_ok=True)
    sentinel = parent / "credentials.json"
    sentinel.write_text("{}", encoding="utf-8")

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(parent)

    assert parent.is_dir()
    assert sentinel.is_file()


def test_reject_sibling_of_managed(
    manager: WorktreeManager, repo: Path,
) -> None:
    """A directory at ``<project_root>/sibling`` is outside the gate."""
    sibling = repo / "sibling"
    sibling.mkdir()
    (sibling / "user_code.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(sibling)

    assert sibling.is_dir()
    assert (sibling / "user_code.py").is_file()


def test_reject_relative_dotdot_escape(
    manager: WorktreeManager, repo: Path, tmp_path: Path,
) -> None:
    """Relative ``..`` paths that resolve outside the managed root are rejected."""
    escape = tmp_path / "escape"
    escape.mkdir()
    (escape / "evidence.txt").write_text("alive", encoding="utf-8")

    # Construct a path that LITERALLY contains ".." but resolves outside
    # the managed root after resolution.
    bogus = manager.base_dir / "wf1" / ".." / ".." / ".." / "escape"

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(bogus)

    assert escape.is_dir()
    assert (escape / "evidence.txt").read_text(encoding="utf-8") == "alive"


# ── Reject: symlink whose target is outside ────────────────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlink creation requires admin on Windows",
)
def test_reject_symlink_to_outside(
    manager: WorktreeManager, repo: Path, tmp_path: Path,
) -> None:
    """A symlink inside the managed root pointing outside is rejected.

    The gate calls ``Path.resolve()`` BEFORE checking containment,
    which follows the link.  A naive prefix check on the unresolved
    path would let this through; ``resolve()`` puts the destination
    where it actually lives — outside the managed root — and the
    gate refuses.
    """
    # Real external directory we want to protect.
    external = tmp_path / "external"
    external.mkdir()
    (external / "victim.txt").write_text("must survive", encoding="utf-8")

    # Make sure the managed root exists before symlinking inside it.
    wf_dir = manager.base_dir / "wf-symlink"
    wf_dir.mkdir(parents=True, exist_ok=True)
    link = wf_dir / "symlink_to_outside"
    os.symlink(external, link)

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(link)

    # Both the link's target and the contents must survive.
    assert external.is_dir()
    assert (external / "victim.txt").read_text(encoding="utf-8") == "must survive"


# ── cleanup_worktrees: workflow_id injection ───────────────────


def test_cleanup_worktrees_rejects_traversal_workflow_id(
    manager: WorktreeManager, repo: Path, tmp_path: Path,
) -> None:
    """``cleanup_worktrees("../../etc")`` raises BEFORE any I/O happens."""
    # Sentinel directory at the path the bogus workflow_id would resolve to.
    # ``base_dir / "../../etc"`` resolves to ``<tmp_path>/../etc`` which
    # we cannot safely create.  Instead, target ``../sentinel`` so the
    # resolved location is predictable and writeable.
    sentinel = repo / "sentinel"
    sentinel.mkdir()
    (sentinel / "data").write_text("preserve", encoding="utf-8")

    # ``base_dir / "../../sentinel"`` resolves to ``<repo>/sentinel``.
    bogus_id = "../../sentinel"

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktrees(bogus_id)

    assert sentinel.is_dir()
    assert (sentinel / "data").read_text(encoding="utf-8") == "preserve"


def test_cleanup_worktrees_managed_workflow_id_succeeds(
    manager: WorktreeManager, repo: Path,
) -> None:
    """Back-compat: a normal ``workflow_id`` cleans up its worktrees."""
    p1 = manager.make_worktree("wf-good", "T1")
    p2 = manager.make_worktree("wf-good", "T2")
    assert p1.is_dir()
    assert p2.is_dir()

    manager.cleanup_worktrees("wf-good")

    assert not p1.exists()
    assert not p2.exists()


def test_cleanup_worktrees_missing_workflow_id_is_noop(
    manager: WorktreeManager,
) -> None:
    """Cleaning a never-created workflow is a silent no-op (not a raise)."""
    # Should not raise — directory just doesn't exist.
    manager.cleanup_worktrees("never-created")


# ── No side-effects on rejection ───────────────────────────────


def test_rejection_runs_no_git_command(
    manager: WorktreeManager, repo: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm the gate fires BEFORE ``subprocess.run``.

    This locks in the order-of-operations: a buggy caller passing an
    external path must never invoke ``git worktree remove`` against
    it, even if the call would have failed.
    """
    external = tmp_path / "external-bare"
    external.mkdir()

    calls: list[tuple[str, ...]] = []

    real_run = subprocess.run

    def spy_run(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        cmd = args[0]
        if isinstance(cmd, (list, tuple)):
            calls.append(tuple(str(a) for a in cmd))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)

    with pytest.raises(WorktreeError, match="outside managed root"):
        manager.cleanup_worktree(external)

    # No ``git worktree remove`` ever ran.
    for cmd in calls:
        assert cmd[:3] != ("git", "worktree", "remove"), (
            f"unexpected git worktree remove against external path: {cmd}"
        )
    assert external.is_dir()
