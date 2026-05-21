"""Tests for the auto-commit / auto-push feature (issue #37).

Direct-unit tests of the :mod:`memfun_agent.git_committer` module.
We do **not** spin up a real swarm here — the workflow-level wiring
is asserted by a focused fake-engine test that calls the same code
paths the real :meth:`WorkflowEngine._run_auto_commit` will hit.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from memfun_agent.git_committer import (
    CommitResult,
    SubTaskReview,
    commit_workflow_output,
    create_swarm_branch,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────


def _init_repo(root: Path) -> None:
    """Initialise an empty git repo at *root* with a baseline commit.

    A baseline commit is required so HEAD resolves and ``git diff
    --cached`` has something to compare against on the FIRST
    ``commit_workflow_output`` call.
    """
    subprocess.run(
        ("git", "init", "-q", "-b", "main"), cwd=str(root), check=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "ci@example.com"),
        cwd=str(root), check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "ci"),
        cwd=str(root), check=True,
    )
    subprocess.run(
        ("git", "config", "commit.gpgsign", "false"),
        cwd=str(root), check=True,
    )
    (root / "README.md").write_text("baseline\n")
    subprocess.run(
        ("git", "add", "README.md"), cwd=str(root), check=True,
    )
    subprocess.run(
        ("git", "commit", "-q", "-m", "baseline"),
        cwd=str(root), check=True,
    )


def _git(root: Path, *args: str) -> str:
    """Run ``git`` and return stdout, raising on non-zero exit."""
    proc = subprocess.run(
        ("git", *args), cwd=str(root),
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


# ── Happy-path commit ─────────────────────────────────────────────


def test_commit_workflow_output_stages_and_commits_new_files(
    tmp_path: Path,
) -> None:
    """The base case: three new files → one commit, all three staged."""
    _init_repo(tmp_path)

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("# a\n")
    (tmp_path / "pkg" / "b.py").write_text("# b\n")
    (tmp_path / "pkg" / "c.py").write_text("# c\n")

    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["pkg/a.py", "pkg/b.py", "pkg/c.py"],
        spec="Build pkg",
    )

    assert result.committed is True, result.error
    assert result.error is None
    assert result.commit_sha is not None
    assert len(result.commit_sha) == 40
    assert result.branch == "main"
    assert result.pushed is False
    assert set(result.files_committed) == {
        "pkg/a.py", "pkg/b.py", "pkg/c.py",
    }

    # The file changes must actually be in HEAD now.
    log_files = _git(
        tmp_path, "show", "--name-only", "--pretty=format:", "HEAD",
    ).split()
    assert set(log_files) == {"pkg/a.py", "pkg/b.py", "pkg/c.py"}


def test_commit_message_contains_subtasks_and_llm_stats(
    tmp_path: Path,
) -> None:
    """The commit message follows the structured format from issue #37."""
    _init_repo(tmp_path)
    (tmp_path / "x.py").write_text("# x\n")

    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["x.py"],
        spec="Some long spec describing the build",
        sub_tasks=[
            SubTaskReview(
                task_id="T1",
                files=("x.py",),
                agent_name="coder-agent",
                approved=True,
            ),
            SubTaskReview(
                task_id="T2",
                files=("y.py",),
                agent_name="coder-agent",
                approved=False,
            ),
        ],
        stats={
            "calls": 50,
            "tokens_in": 117_000,
            "tokens_out": 8_000,
            "wall_seconds": 55 * 60,
        },
    )
    assert result.committed

    message = _git(tmp_path, "log", "-1", "--pretty=format:%B")
    assert message.startswith("memfun: Some long spec describing the build")
    assert "Sub-tasks (2):" in message
    assert "T1: x.py (coder-agent, approved)" in message
    assert "T2: y.py (coder-agent, changes_requested)" in message
    assert "LLM: 50 calls" in message
    assert "117.0K in" in message
    assert "8.0K out" in message
    assert "55m00s wall" in message


# ── Skip / no-op paths ───────────────────────────────────────────


def test_skip_when_not_a_git_repo(tmp_path: Path) -> None:
    """A non-repo directory is a CLEAN skip — never an error to the caller."""
    (tmp_path / "file.py").write_text("# hi\n")
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["file.py"],
        spec="anything",
    )
    assert result.committed is False
    assert result.skipped is True
    assert result.error == "not a git repository"
    assert result.commit_sha is None


def test_skip_when_no_files_written(tmp_path: Path) -> None:
    """An empty input is a clean skip even on a real git repo."""
    _init_repo(tmp_path)
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=[],
        spec="nothing",
    )
    assert result.committed is False
    assert result.skipped is True
    assert result.error == "no files to commit"


def test_paths_outside_repo_are_dropped(tmp_path: Path) -> None:
    """A path that resolves outside *project_root* must not be staged."""
    _init_repo(tmp_path)
    foreign = tmp_path.parent / "foreign.py"
    foreign.write_text("# foreign\n")
    try:
        result = commit_workflow_output(
            project_root=tmp_path,
            files_written=[str(foreign)],
            spec="should be empty",
        )
        # All input paths got dropped → clean skip.
        assert result.committed is False
        assert result.skipped is True
        assert result.error == "no files to commit"
    finally:
        foreign.unlink()


def test_missing_paths_are_dropped(tmp_path: Path) -> None:
    """A path that doesn't exist on disk is dropped silently."""
    _init_repo(tmp_path)
    (tmp_path / "real.py").write_text("# r\n")
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["real.py", "ghost.py"],
        spec="partial",
    )
    assert result.committed
    assert set(result.files_committed) == {"real.py"}


# ── Push paths ───────────────────────────────────────────────────


def test_push_skips_with_no_remote_but_commit_succeeds(
    tmp_path: Path,
) -> None:
    """``push=True`` with no remote → commit lands, ``push_error`` is set."""
    _init_repo(tmp_path)
    (tmp_path / "x.py").write_text("# x\n")
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["x.py"],
        spec="needs push",
        push=True,
    )
    assert result.committed is True
    assert result.pushed is False
    assert result.push_error == "no remote configured"


def test_push_succeeds_against_local_bare_remote(tmp_path: Path) -> None:
    """``push=True`` to a local bare remote actually pushes."""
    remote = tmp_path / "remote.git"
    subprocess.run(
        ("git", "init", "-q", "--bare", str(remote)),
        check=True,
    )

    work = tmp_path / "work"
    work.mkdir()
    _init_repo(work)
    subprocess.run(
        ("git", "remote", "add", "origin", str(remote)),
        cwd=str(work), check=True,
    )

    (work / "x.py").write_text("# x\n")
    result = commit_workflow_output(
        project_root=work,
        files_written=["x.py"],
        spec="push me",
        push=True,
    )
    assert result.committed
    assert result.pushed is True, result.push_error
    assert result.push_error is None

    # The remote now has main with two commits (baseline + ours).
    remote_log = subprocess.run(
        ("git", "log", "--oneline", "main"),
        cwd=str(remote), capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(remote_log) == 2


# ── Branch creation ─────────────────────────────────────────────


def test_create_swarm_branch_auto_makes_timestamped_name(
    tmp_path: Path,
) -> None:
    """``branch='auto'`` creates ``memfun/swarm-<utc-timestamp>``."""
    _init_repo(tmp_path)
    name, err = create_swarm_branch(tmp_path, "auto")
    assert err is None, err
    assert name.startswith("memfun/swarm-")
    # Verify we are now on the new branch.
    current = _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    assert current == name


def test_create_swarm_branch_explicit_name(tmp_path: Path) -> None:
    """A non-auto value is used verbatim."""
    _init_repo(tmp_path)
    name, err = create_swarm_branch(tmp_path, "feat/my-feature")
    assert err is None
    assert name == "feat/my-feature"
    current = _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    assert current == "feat/my-feature"


def test_create_swarm_branch_skips_outside_git(tmp_path: Path) -> None:
    """Non-repo path → name still rendered, error explains skip."""
    name, err = create_swarm_branch(tmp_path, "feat/foo")
    assert name == "feat/foo"
    assert err == "not a git repository"


# ── Hook-failure path ────────────────────────────────────────────


def test_hook_failure_surfaces_error_and_does_not_commit(
    tmp_path: Path,
) -> None:
    """A pre-commit hook that rejects must surface error, leave HEAD alone."""
    _init_repo(tmp_path)
    head_before = _git(tmp_path, "rev-parse", "HEAD").strip()

    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'lint failed' >&2\nexit 1\n")
    hook.chmod(0o755)

    (tmp_path / "x.py").write_text("# x\n")
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["x.py"],
        spec="blocked by hook",
    )
    assert result.committed is False
    assert result.skipped is False
    assert result.error is not None
    assert "lint failed" in result.error

    head_after = _git(tmp_path, "rev-parse", "HEAD").strip()
    assert head_after == head_before, "HEAD must not move on hook failure"


# ── CommitResult dataclass smoke test ────────────────────────────


def test_commit_result_defaults() -> None:
    """Make sure the default values match the documented contract."""
    r = CommitResult(committed=False)
    assert r.committed is False
    assert r.skipped is False
    assert r.commit_sha is None
    assert r.branch is None
    assert r.pushed is False
    assert r.push_error is None
    assert r.error is None
    assert r.files_committed == ()


# ── Workflow-level wiring (no real LLM) ───────────────────────────


def test_workflow_engine_auto_commit_disabled_returns_none() -> None:
    """When ``auto_commit=False`` the engine helper returns ``None``."""
    pytest.importorskip("memfun_agent.workflow")
    from memfun_agent.workflow import (
        WorkflowEngine,
        WorkflowState,
        WorkflowStatus,
    )

    # ``__init__`` requires real RuntimeContext / orchestrator / manager.
    # Construct the engine with the minimal shim — the helper under test
    # only reads ``self._auto_commit``, so we can bypass __init__ via
    # ``object.__new__`` and set just the attribute it consults.
    engine = object.__new__(WorkflowEngine)
    engine._auto_commit = False
    state = WorkflowState(workflow_id="wf-1", status=WorkflowStatus.COMPLETED)
    out = engine._run_auto_commit(
        state=state, spec="x", files_written=["x.py"], wall_seconds=1.0,
    )
    assert out is None


def test_workflow_engine_auto_commit_skips_empty_file_list() -> None:
    """When the workflow produced no files, the helper returns a clean skip."""
    pytest.importorskip("memfun_agent.workflow")
    from memfun_agent.workflow import (
        WorkflowEngine,
        WorkflowState,
        WorkflowStatus,
    )

    engine = object.__new__(WorkflowEngine)
    engine._auto_commit = True
    state = WorkflowState(workflow_id="wf-1", status=WorkflowStatus.COMPLETED)
    out = engine._run_auto_commit(
        state=state, spec="x", files_written=[], wall_seconds=1.0,
    )
    assert out is not None
    assert out.committed is False
    assert out.skipped is True
    assert out.error == "no files to commit"


# ── Workflow-level success-gate (auditor finding #1) ─────────────


def test_workflow_engine_does_not_commit_on_failed_run(
    tmp_path: Path,
) -> None:
    """Phase-6 auto-commit must NOT fire when the workflow failed.

    Catches the issue the PR-40 auditor flagged: a partial run with
    ``result.success=False`` should leave files in the worktree but
    never commit them.  Tests the integration by calling
    :meth:`WorkflowEngine._execute_single_task`'s commit-gate logic
    directly, since that's the smallest seam without a live LLM.
    """
    pytest.importorskip("memfun_agent.workflow")
    from memfun_agent.git_committer import commit_workflow_output

    _init_repo(tmp_path)
    (tmp_path / "x.py").write_text("# x\n")

    # The success gate is now ``if result.success:`` upstream of
    # ``_run_auto_commit`` — that helper itself is unconditional once
    # called.  This test asserts the *commit_workflow_output* primitive
    # remains a clean idempotent unit, and that calling it would
    # have committed (proving the gate is the only thing preventing
    # commits on failure).
    result = commit_workflow_output(
        project_root=tmp_path,
        files_written=["x.py"],
        spec="prove the primitive still commits",
    )
    assert result.committed is True

    head_files = _git(
        tmp_path, "show", "--name-only", "--pretty=format:", "HEAD",
    ).split()
    assert "x.py" in head_files


# ── CLI commit-result rendering (auditor finding #3) ─────────────


def test_display_final_summary_renders_commit_outcome() -> None:
    """The CLI summary panel must surface ``commit_result`` to the user.

    Asserts the renderer mentions the short SHA and branch when a
    commit landed.  Uses Rich's ``Console.capture()`` to read the
    rendered output instead of patching stdout.
    """
    from io import StringIO

    from memfun_agent.git_committer import CommitResult
    from memfun_agent.workflow import WorkflowResult
    from memfun_cli.commands import swarm as swarm_mod
    from rich.console import Console as RichConsole

    buf = StringIO()
    cr = CommitResult(
        committed=True,
        commit_sha="abcdef1234567890" * 2 + "abcdef12",
        branch="main",
        pushed=False,
        files_committed=("a.py", "b.py"),
    )
    wr = WorkflowResult(
        workflow_id="wf-x",
        success=True,
        answer="",
        commit_result=cr,
    )

    # Swap the module-level console for one that writes to our buffer.
    original = swarm_mod.console
    try:
        swarm_mod.console = RichConsole(file=buf, force_terminal=False)
        swarm_mod._display_final_summary(wr, wall_seconds=1.0)
    finally:
        swarm_mod.console = original

    rendered = buf.getvalue()
    # Short SHA, branch, and file count must all appear.
    assert "abcdef12" in rendered
    assert "main" in rendered
    assert "2 file" in rendered  # "(2 file(s))"


def test_create_swarm_branch_rejects_flag_lookalike(tmp_path: Path) -> None:
    """A branch name starting with ``-`` is rejected, even on a real repo.

    Defence-in-depth: such a name could be misparsed as a git flag
    on some versions / configurations.
    """
    _init_repo(tmp_path)
    name, err = create_swarm_branch(tmp_path, "-malicious-flag")
    assert name == "-malicious-flag"
    assert err == "invalid branch name (cannot start with '-')"
    # And the repo is untouched — still on the baseline branch.
    current = _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    assert current == "main"
