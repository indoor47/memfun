"""Unit tests for :mod:`memfun_agent.gh_pr_flow` (issue #38).

These tests monkeypatch :func:`subprocess.run` so they never actually
shell out to ``gh`` / ``git`` — they capture the argv each helper
would have passed and assert on it.  This matches the strategy used by
the auto-commit tests in :mod:`tests.test_swarm_autocommit` and keeps
the suite hermetic: it runs in CI on machines without ``gh`` installed
and without GitHub auth.

We cover:

* ``push_subtask_branch`` — happy path + push failure.
* ``open_subtask_pr`` — happy path + auth-missing handling +
  PR-number parsing from gh stdout.
* ``merge_pr`` — happy path + branch-protection rejection.
* ``current_repo_slug`` — gh-tracked vs. not gh-tracked.

The tests use a small ``_capture`` helper that records every
subprocess invocation into a shared list so we can assert both the
captured argv and the order of calls.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from memfun_agent.gh_pr_flow import (
    MergeResult,
    OpenPrResult,
    PushResult,
    add_pr_review_comment,
    close_pr,
    current_repo_slug,
    mark_pr_ready,
    merge_pr,
    open_subtask_pr,
    push_subtask_branch,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

# ── Helpers ────────────────────────────────────────────────────


def _completed(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    args: tuple[str, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess with the given fields."""
    return subprocess.CompletedProcess(
        args=list(args or ("gh",)),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _make_runner(
    responses: list[subprocess.CompletedProcess[str]],
    capture: list[dict[str, Any]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Build a fake :func:`subprocess.run` that pops *responses* in order.

    Captures the argv and kwargs of each call into *capture* so the
    test can assert on them after the helper returns.  When more
    calls happen than responses are supplied, returns a zero-exit
    empty completed process — that keeps the helper for over-call
    tests trivial.
    """

    def _run(
        cmd: tuple[str, ...] | list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        capture.append({"cmd": tuple(cmd), "kwargs": kwargs})
        if responses:
            return responses.pop(0)
        return _completed()

    return _run


# ── push_subtask_branch ────────────────────────────────────────


def test_push_subtask_branch_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful push returns ``PushResult(pushed=True)`` and runs
    exactly the expected git command from the worktree path.
    """
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="To origin\n", returncode=0),
                _completed(stdout="0123abcd" * 5 + "\n", returncode=0),
            ],
            capture,
        ),
    )

    result = push_subtask_branch(tmp_path, "memfun/wf/T1")

    assert isinstance(result, PushResult)
    assert result.pushed is True
    assert result.branch == "memfun/wf/T1"
    assert result.error is None
    # The pushed HEAD SHA is recorded so the merge can be pinned to it.
    assert result.head_sha == "0123abcd" * 5
    # Two git calls: the push (right argv + cwd), then rev-parse HEAD.
    assert len(capture) == 2
    assert capture[1]["cmd"] == ("git", "rev-parse", "HEAD")
    assert capture[0]["cmd"] == (
        "git",
        "push",
        "-u",
        "origin",
        "memfun/wf/T1",
    )
    assert capture[0]["kwargs"]["cwd"] == str(tmp_path)


def test_push_subtask_branch_failure_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero git exit produces ``PushResult.error`` populated
    and ``pushed=False`` — never an exception."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(
                    stderr="fatal: 'origin' does not appear to be a git repository\n",
                    returncode=128,
                )
            ],
            capture,
        ),
    )

    result = push_subtask_branch(tmp_path, "feat-branch")

    assert result.pushed is False
    assert result.branch == "feat-branch"
    assert result.error is not None
    assert "origin" in result.error


def test_push_subtask_branch_missing_worktree_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pushing from a non-existent worktree fails BEFORE shelling out."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner([], capture),
    )

    bad_path = tmp_path / "does-not-exist"
    result = push_subtask_branch(bad_path, "feat-branch")

    assert result.pushed is False
    assert result.error is not None
    assert "does not exist" in result.error
    # No subprocess invocations.
    assert capture == []


def test_push_subtask_branch_rejects_flag_branch_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch name starting with '-' is refused without invoking git."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner([], capture),
    )

    result = push_subtask_branch(tmp_path, "--upload-pack=evil")
    assert result.pushed is False
    assert result.error is not None
    assert "invalid branch name" in result.error
    assert capture == []


# ── open_subtask_pr ────────────────────────────────────────────


def test_open_subtask_pr_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful ``gh pr create`` returns the parsed PR number."""
    capture: list[dict[str, Any]] = []
    # First subprocess call is ``gh --version`` (availability probe);
    # second is the real ``gh pr create``.
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(
                    stdout="https://github.com/owner/name/pull/142\n",
                    returncode=0,
                ),
            ],
            capture,
        ),
    )

    result = open_subtask_pr(
        repo="owner/name",
        base="main",
        head="memfun/wf/T1",
        title="Implement areas.py",
        body="## Spec\nDo the thing.\n",
        draft=True,
    )

    assert isinstance(result, OpenPrResult)
    assert result.pr_number == 142
    assert result.url == "https://github.com/owner/name/pull/142"
    assert result.draft is True
    assert result.error is None

    # Two calls: probe + create.
    assert len(capture) == 2
    create_cmd = capture[1]["cmd"]
    assert create_cmd[0] == "gh"
    assert create_cmd[1:3] == ("pr", "create")
    # Verify the important flags are present.
    assert "--repo" in create_cmd
    assert "owner/name" in create_cmd
    assert "--base" in create_cmd
    assert "main" in create_cmd
    assert "--head" in create_cmd
    assert "memfun/wf/T1" in create_cmd
    assert "--title" in create_cmd
    assert "Implement areas.py" in create_cmd
    assert "--body-file" in create_cmd
    assert "--draft" in create_cmd


def test_open_subtask_pr_auth_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``gh`` reports an auth error we surface a friendly message."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(
                    stderr=(
                        "error: not logged into any GitHub hosts. "
                        "Use `gh auth login` to authenticate.\n"
                    ),
                    returncode=4,
                ),
            ],
            capture,
        ),
    )

    result = open_subtask_pr(
        repo="owner/name",
        base="main",
        head="feat",
        title="t",
        body="b",
    )
    assert result.pr_number is None
    assert result.error is not None
    assert "not authenticated" in result.error
    assert "gh auth login" in result.error


def test_open_subtask_pr_no_gh_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``gh`` binary returns a structured error."""

    def _no_gh(
        cmd: tuple[str, ...] | list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # Simulate ``gh --version`` raising FileNotFoundError.
        if tuple(cmd) == ("gh", "--version"):
            raise FileNotFoundError("gh not installed")
        return _completed()

    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _no_gh,
    )

    result = open_subtask_pr(
        repo="owner/name",
        base="main",
        head="b",
        title="t",
        body="b",
    )
    assert result.pr_number is None
    assert result.error is not None
    assert "gh CLI not available" in result.error


def test_open_subtask_pr_rejects_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty title / flag-like base/head get caught before shelling out."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [_completed(stdout="gh version 2.50.0\n", returncode=0)] * 5,
            capture,
        ),
    )

    # Empty title.
    res = open_subtask_pr(
        repo="o/n",
        base="main",
        head="b",
        title="   ",
        body="b",
    )
    assert res.pr_number is None
    assert res.error is not None
    assert "title" in res.error.lower()

    # Flag-like head.
    res = open_subtask_pr(
        repo="o/n",
        base="main",
        head="--bad",
        title="t",
        body="b",
    )
    assert res.pr_number is None
    assert res.error is not None
    assert "invalid head" in res.error


# ── merge_pr ───────────────────────────────────────────────────


def test_merge_pr_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean merge returns ``MergeResult(merged=True)`` and uses the
    requested strategy flag plus ``--delete-branch``."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(
                    stdout="Squashed and merged pull request #142\n",
                    returncode=0,
                ),
                # Post-merge verification: state must be MERGED.
                _completed(stdout='{"state": "MERGED"}', returncode=0),
            ],
            capture,
        ),
    )

    result = merge_pr("owner/name", 142, strategy="squash")
    assert isinstance(result, MergeResult)
    assert result.merged is True
    assert result.pr_number == 142
    assert result.strategy == "squash"
    assert result.error is None

    merge_cmd = capture[1]["cmd"]
    assert merge_cmd[:4] == ("gh", "pr", "merge", "142")
    assert "--squash" in merge_cmd
    assert "--delete-branch" in merge_cmd
    assert "--admin" not in merge_cmd  # never use --admin
    assert "--repo" in merge_cmd
    assert "owner/name" in merge_cmd


def test_merge_pr_branch_protection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branch-protection failures surface in ``MergeResult.error``
    without ever raising or retrying with ``--admin``."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(
                    stderr=(
                        "X required checks have not passed; merge blocked by branch protection.\n"
                    ),
                    returncode=1,
                ),
            ],
            capture,
        ),
    )

    result = merge_pr("owner/name", 99, strategy="squash")
    assert result.merged is False
    assert result.pr_number == 99
    assert result.error is not None
    assert "branch protection" in result.error


def test_merge_pr_unknown_strategy_defaults_to_squash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown merge strategies are silently rewritten to ``squash``."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stdout="Merged.\n", returncode=0),
                _completed(stdout='{"state": "MERGED"}', returncode=0),
            ],
            capture,
        ),
    )

    result = merge_pr("o/n", 7, strategy="trapdoor")
    assert result.merged is True
    assert result.strategy == "squash"
    assert "--squash" in capture[1]["cmd"]


# ── current_repo_slug ──────────────────────────────────────────


def test_current_repo_slug_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns the owner/name string when gh repo view succeeds."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stdout="indoor47/memfun\n", returncode=0),
            ],
            capture,
        ),
    )

    slug = current_repo_slug(tmp_path)
    assert slug == "indoor47/memfun"
    repo_cmd = capture[1]["cmd"]
    assert repo_cmd[:3] == ("gh", "repo", "view")


def test_current_repo_slug_no_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-gh-tracked directory returns ``None`` without raising."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(
                    stderr="no remotes configured\n",
                    returncode=1,
                ),
            ],
            capture,
        ),
    )

    slug = current_repo_slug(tmp_path)
    assert slug is None


def test_current_repo_slug_gh_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No gh binary on PATH → ``None``, no shell-out beyond the probe."""

    def _no_gh(
        cmd: tuple[str, ...] | list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if tuple(cmd) == ("gh", "--version"):
            raise FileNotFoundError("gh missing")
        return _completed()

    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _no_gh,
    )
    assert current_repo_slug(tmp_path) is None


# ── mark_pr_ready / add_pr_review_comment / close_pr ────────────


def test_mark_pr_ready_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stdout="", returncode=0),
            ],
            capture,
        ),
    )
    assert mark_pr_ready("o/n", 42) is True
    ready_cmd = capture[1]["cmd"]
    assert ready_cmd[:4] == ("gh", "pr", "ready", "42")
    assert "--repo" in ready_cmd
    assert "o/n" in ready_cmd


def test_mark_pr_ready_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stderr="perm denied\n", returncode=1),
            ],
            capture,
        ),
    )
    assert mark_pr_ready("o/n", 42) is False


def test_add_pr_review_comment_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stdout="", returncode=0),
            ],
            capture,
        ),
    )
    assert add_pr_review_comment("o/n", 7, "great PR") is True
    cmt_cmd = capture[1]["cmd"]
    assert cmt_cmd[:4] == ("gh", "pr", "comment", "7")
    assert "--body-file" in cmt_cmd


def test_add_pr_review_comment_rejects_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty bodies are refused without shelling out."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [_completed(stdout="gh version 2.50.0\n", returncode=0)],
            capture,
        ),
    )
    # Only the version probe runs; no comment call.
    assert add_pr_review_comment("o/n", 7, "   ") is False
    assert len(capture) == 1


def test_close_pr_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(stdout="closed\n", returncode=0),
            ],
            capture,
        ),
    )
    assert close_pr("o/n", 11, "could not review") is True
    close_cmd = capture[1]["cmd"]
    assert close_cmd[:4] == ("gh", "pr", "close", "11")
    assert "--comment" in close_cmd
    assert "could not review" in close_cmd


def test_close_pr_without_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty reasons skip the ``--comment`` flag."""
    capture: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _make_runner(
            [
                _completed(stdout="gh version 2.50.0\n", returncode=0),
                _completed(returncode=0),
            ],
            capture,
        ),
    )
    assert close_pr("o/n", 11, "") is True
    close_cmd = capture[1]["cmd"]
    assert "--comment" not in close_cmd
