"""GitHub PR flow helpers for ``memfun swarm --gh-pr`` (issue #38).

This module wraps the ``gh`` CLI so :class:`WorkflowEngine` can,
for every sub-task that succeeds, push the worktree branch, open
a draft pull request, route the PR through the ``pr-reviewer``
agent, and finally merge approved PRs in dependency order.

Public surface (all functions are subprocess-based and return
structured result dataclasses; expected failure modes — missing
``gh``, no auth, branch protection, etc. — are encoded as
``error`` fields rather than raised exceptions):

* :func:`current_repo_slug` — discover ``owner/name`` via
  ``gh repo view``.
* :func:`push_subtask_branch` — ``git push -u origin <branch>``
  from inside a worktree.
* :func:`open_subtask_pr` — ``gh pr create --draft`` with a
  multi-line body delivered through a tempfile (avoids shell-quote
  hell and oversized argv on long specs).
* :func:`mark_pr_ready` — ``gh pr ready <n>``.
* :func:`add_pr_review_comment` — ``gh pr comment <n>
  --body-file <tmp>``.
* :func:`check_pr_mergeable` — ``gh pr view <n> --json
  mergeable,mergeStateStatus,headRefOid`` with fail-closed handling
  of ``CONFLICTING``/``DIRTY``/``UNKNOWN`` and a head-SHA pin.
* :func:`merge_pr` — ``gh pr merge <n> --<strategy>
  --match-head-commit <sha>``, then verifies the PR state is
  ``MERGED``.
* :func:`close_pr` — ``gh pr close <n>`` with reason as the
  closing comment.

Why subprocess + gh CLI rather than a Python GitHub client?
The user already has ``gh`` authenticated for their repo; the
PR-reviewer AGENT.md already speaks ``gh``; and we already
shell out to ``git``. Keeping the surface uniform makes the
flow easy to audit and trivially debuggable from a terminal.

Why frozen dataclasses for results?
Same pattern as :mod:`memfun_agent.git_committer` — callers
can pattern-match on the result, the structure is immutable,
and ``slots=True`` keeps memory tight when a swarm produces
many PRs.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("agent.gh_pr_flow")


# ── Result dataclasses ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PushResult:
    """Outcome of a ``git push -u origin <branch>`` attempt.

    Attributes:
        pushed: ``True`` when the push command exited zero.
        branch: Branch we attempted to push.
        error: Human-readable failure message (``None`` on success).
            Set for any non-zero exit — missing remote, auth error,
            non-fast-forward, etc.
    """

    pushed: bool
    branch: str
    error: str | None = None
    # Local ``HEAD`` SHA right after a successful push (``None`` if it
    # could not be read).  Callers pin merges to it.
    head_sha: str | None = None


@dataclass(frozen=True, slots=True)
class OpenPrResult:
    """Outcome of ``gh pr create``.

    Attributes:
        pr_number: Integer PR number when the open succeeded.
        url: PR URL ``gh`` printed on stdout (handy for log lines).
        draft: ``True`` if the PR was opened as a draft.
        error: Failure reason (``gh`` not installed, no auth, repo
            mismatch, …).  Mutually exclusive with ``pr_number``.
    """

    pr_number: int | None
    url: str | None = None
    draft: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Outcome of ``gh pr merge``.

    Attributes:
        merged: ``True`` when the merge command exited zero.
        pr_number: PR we attempted to merge.
        strategy: The merge strategy that was used (``squash`` /
            ``merge`` / ``rebase``).
        error: Failure reason — typically branch-protection rules,
            failing checks, or insufficient permissions.
    """

    merged: bool
    pr_number: int
    strategy: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MergeabilityResult:
    """Outcome of :func:`check_pr_mergeable`.

    Attributes:
        mergeable: ``True`` only when GitHub reported ``MERGEABLE``,
            the merge state is not ``DIRTY`` and (when pinned) the
            head SHA matches.  Everything else fails closed.
        mergeable_state: Raw ``mergeable`` value (``MERGEABLE`` /
            ``CONFLICTING`` / ``UNKNOWN`` / ``""`` when unread).
        merge_state_status: Raw ``mergeStateStatus`` value.
        head_sha: ``headRefOid`` reported by GitHub.
        error: Why the PR is not mergeable (``None`` when it is).
    """

    mergeable: bool
    mergeable_state: str = ""
    merge_state_status: str = ""
    head_sha: str | None = None
    error: str | None = None


# ── Internals ─────────────────────────────────────────────────────


def _gh_available() -> bool:
    """Return ``True`` when the ``gh`` CLI is on PATH.

    We probe via ``gh --version`` rather than ``shutil.which`` so a
    broken-but-present ``gh`` (eg. quarantined macOS binary that
    fails ``exec``) is also treated as missing.
    """
    try:
        proc = subprocess.run(
            ("gh", "--version"),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    return proc.returncode == 0


def _run_gh(
    *args: str,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh <args>`` capturing stdout/stderr as text.

    Centralised so tests can monkeypatch ``subprocess.run`` once and
    every helper picks it up.  We pass ``check=False`` because every
    caller wants to inspect ``returncode`` to populate its result
    dataclass.
    """
    cmd = ("gh", *args)
    logger.debug(
        "gh_pr_flow: %s (cwd=%s)",
        " ".join(cmd),
        cwd or "<none>",
    )
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
        input=input_text,
    )


def _run_git(
    *args: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` rooted at *cwd* (a worktree path).

    Same shape as :func:`_run_gh` so the test seam is identical.
    """
    cmd = ("git", *args)
    logger.debug("gh_pr_flow: %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _json_dict(text: str | None) -> dict[str, object] | None:
    """Parse *text* as a JSON object; ``None`` on any failure."""
    try:
        data: object = json.loads(text or "")
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return cast("dict[str, object]", data)


def _extract_pr_number(stdout: str) -> int | None:
    """Parse the PR number out of ``gh pr create`` stdout.

    ``gh`` prints the PR URL on the last non-empty line, eg::

        https://github.com/owner/name/pull/142

    We pull the integer after ``/pull/`` rather than trusting
    line order — ``gh`` occasionally prepends warnings on stderr
    that bleed into stdout depending on terminal mode.
    """
    if not stdout:
        return None
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if "/pull/" not in line:
            continue
        try:
            tail = line.rsplit("/pull/", 1)[1]
            # Strip any trailing path component or query string.
            num_str = tail.split("/", 1)[0].split("?", 1)[0]
            return int(num_str)
        except (ValueError, IndexError):
            continue
    return None


# ── Public API ────────────────────────────────────────────────────


def current_repo_slug(project_root: Path) -> str | None:
    """Return the ``owner/name`` slug for the current gh-tracked repo.

    Returns ``None`` when:

    * ``gh`` is not installed,
    * ``gh`` is not authenticated for this directory,
    * the directory is not a git repo with a github remote.

    Callers treat ``None`` as "this project isn't gh-configured —
    skip the entire PR flow gracefully".
    """
    if not _gh_available():
        return None
    proc = _run_gh(
        "repo",
        "view",
        "--json",
        "nameWithOwner",
        "-q",
        ".nameWithOwner",
        cwd=project_root,
    )
    if proc.returncode != 0:
        logger.debug(
            "gh repo view failed (rc=%d): %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return None
    slug = proc.stdout.strip()
    if not slug or "/" not in slug:
        return None
    return slug


def push_subtask_branch(
    worktree_path: Path,
    branch: str,
) -> PushResult:
    """Push *branch* from *worktree_path* to ``origin``.

    Uses ``git push -u origin <branch>`` so subsequent pushes from
    the same worktree have an upstream set — the reviewer's
    re-dispatch loop in the orchestrator may push follow-up commits
    to the same branch.

    Never raises.  A missing worktree, missing remote, or auth
    failure becomes ``PushResult(pushed=False, error=...)``.
    """
    if not worktree_path.exists():
        return PushResult(
            pushed=False,
            branch=branch,
            error=f"worktree path does not exist: {worktree_path}",
        )
    if branch.startswith("-"):
        # Defence in depth: refuse anything that could be parsed
        # as a git flag.  Real branch names never start with '-'.
        return PushResult(
            pushed=False,
            branch=branch,
            error="invalid branch name (cannot start with '-')",
        )

    proc = _run_git(
        "push",
        "-u",
        "origin",
        branch,
        cwd=worktree_path,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        logger.warning(
            "git push of %s failed: %s",
            branch,
            err[:200],
        )
        return PushResult(
            pushed=False,
            branch=branch,
            error=err or "git push failed",
        )
    # Record the SHA we just pushed so the merge can be pinned to it
    # (``gh pr merge --match-head-commit``).  A failure here leaves
    # ``head_sha=None``, which downstream refuses to merge.
    head_sha: str | None = None
    rev = _run_git("rev-parse", "HEAD", cwd=worktree_path)
    if rev.returncode == 0:
        candidate = (rev.stdout or "").strip()
        if candidate and all(c in "0123456789abcdef" for c in candidate.lower()):
            head_sha = candidate
    else:
        logger.warning(
            "git rev-parse HEAD after pushing %s failed: %s",
            branch,
            (rev.stderr or rev.stdout or "").strip()[:200],
        )
    return PushResult(pushed=True, branch=branch, head_sha=head_sha)


def open_subtask_pr(
    *,
    repo: str,
    base: str,
    head: str,
    title: str,
    body: str,
    draft: bool = True,
) -> OpenPrResult:
    """Open a draft PR via ``gh pr create`` and return its number.

    The body is written to a tempfile and passed via ``--body-file``
    so multi-paragraph specs containing backticks / quotes / newlines
    travel through without shell-quote hell, and the argv stays
    short enough to never trip the OS argv limit on long swarms.

    Returns ``OpenPrResult(pr_number=None, error=...)`` on any
    failure path (gh missing, auth missing, head branch not pushed,
    target repo wrong, …).  Never raises.
    """
    if not _gh_available():
        return OpenPrResult(
            pr_number=None,
            error="gh CLI not available on PATH",
        )

    # Defence in depth: reject obvious junk before we shell out.
    for label, value in (("base", base), ("head", head), ("repo", repo)):
        if not value or value.startswith("-"):
            return OpenPrResult(
                pr_number=None,
                error=f"invalid {label}: {value!r}",
            )
    if not title.strip():
        return OpenPrResult(
            pr_number=None,
            error="PR title is empty",
        )

    # Use NamedTemporaryFile in write-then-pass mode so the tempfile
    # is automatically cleaned up after the gh call.  Suffix .md so
    # any editor that picks it up renders nicely.
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="memfun-pr-",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(body)
        body_path = Path(fh.name)

    try:
        args = [
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            base,
            "--head",
            head,
            "--title",
            title,
            "--body-file",
            str(body_path),
        ]
        if draft:
            args.append("--draft")
        proc = _run_gh(*args)
    finally:
        # Best-effort cleanup: a stale tempfile is harmless but noisy.
        with contextlib.suppress(OSError):
            body_path.unlink()

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        # Surface a friendly hint for the common "no auth" case so
        # the orchestrator's log line says something actionable.
        lowered = err.lower()
        if "authentication" in lowered or "not logged into" in lowered:
            err = "gh CLI is not authenticated (run `gh auth login`)"
        logger.warning(
            "gh pr create for %s failed: %s",
            head,
            err[:300],
        )
        return OpenPrResult(
            pr_number=None,
            draft=draft,
            error=err or "gh pr create failed",
        )

    stdout = (proc.stdout or "").strip()
    pr_number = _extract_pr_number(stdout)
    if pr_number is None:
        return OpenPrResult(
            pr_number=None,
            url=stdout or None,
            draft=draft,
            error=f"could not parse PR number from gh output: {stdout[:200]!r}",
        )
    return OpenPrResult(
        pr_number=pr_number,
        url=stdout.splitlines()[-1] if stdout else None,
        draft=draft,
    )


def mark_pr_ready(repo: str, pr_number: int) -> bool:
    """Flip a draft PR to ready-for-review via ``gh pr ready``.

    Returns ``True`` on success.  Failures (network error,
    permission denied, PR already ready) log a warning and
    return ``False`` — callers may treat this as soft-failure
    (a draft PR is still useful) rather than abort.
    """
    if not _gh_available():
        return False
    proc = _run_gh(
        "pr",
        "ready",
        str(pr_number),
        "--repo",
        repo,
    )
    if proc.returncode != 0:
        logger.warning(
            "gh pr ready %s failed: %s",
            pr_number,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return False
    return True


def add_pr_review_comment(
    repo: str,
    pr_number: int,
    body: str,
) -> bool:
    """Post a top-level comment on a PR via ``gh pr comment``.

    Body is delivered via tempfile so long auditor reports
    (with markdown headings, code fences, …) survive intact.

    Returns ``True`` on success.  We don't raise on failure —
    a missed comment shouldn't crash the workflow; the verdict
    in the orchestrator's logs is the source of truth.
    """
    if not _gh_available():
        return False
    if not body.strip():
        logger.debug(
            "Refusing to post empty comment on PR %d",
            pr_number,
        )
        return False

    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="memfun-comment-",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(body)
        body_path = Path(fh.name)

    try:
        proc = _run_gh(
            "pr",
            "comment",
            str(pr_number),
            "--repo",
            repo,
            "--body-file",
            str(body_path),
        )
    finally:
        with contextlib.suppress(OSError):
            body_path.unlink()

    if proc.returncode != 0:
        logger.warning(
            "gh pr comment %s failed: %s",
            pr_number,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return False
    return True


def check_pr_mergeable(
    repo: str,
    pr_number: int,
    *,
    expected_head_sha: str | None = None,
    unknown_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> MergeabilityResult:
    """Check that a PR can be merged, failing closed.

    Runs ``gh pr view <n> --json mergeable,mergeStateStatus,headRefOid``.
    An unmergeable PR gets no CI runs, so "no checks" must never be
    read as passing: callers check this BEFORE marking ready/merging.

    * ``CONFLICTING`` or ``mergeStateStatus == DIRTY`` -> not mergeable.
    * ``UNKNOWN`` (GitHub still computing) -> re-check up to
      *unknown_retries* more times, sleeping *retry_delay_seconds*
      between attempts; still unknown -> not mergeable.
    * *expected_head_sha* given and ``headRefOid`` differs -> not
      mergeable (the head moved after we pushed/reviewed it).
    * Non-zero ``gh`` exit or unparseable JSON -> not mergeable.

    Never raises.
    """
    if not _gh_available():
        return MergeabilityResult(mergeable=False, error="gh CLI not available on PATH")

    attempts = max(0, unknown_retries) + 1
    last = MergeabilityResult(mergeable=False, error="mergeability not checked")
    for attempt in range(attempts):
        if attempt:
            sleep_fn(retry_delay_seconds)
        proc = _run_gh(
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "mergeable,mergeStateStatus,headRefOid",
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return MergeabilityResult(
                mergeable=False,
                error=f"gh pr view failed (rc={proc.returncode}): {err[:200]}",
            )
        data = _json_dict(proc.stdout)
        if data is None:
            return MergeabilityResult(
                mergeable=False,
                error=f"unparseable gh pr view output: {(proc.stdout or '')[:200]!r}",
            )
        mergeable_state = str(data.get("mergeable") or "").upper()
        merge_state = str(data.get("mergeStateStatus") or "").upper()
        head_raw = data.get("headRefOid")
        head = head_raw if isinstance(head_raw, str) and head_raw else None

        base = MergeabilityResult(
            mergeable=False,
            mergeable_state=mergeable_state,
            merge_state_status=merge_state,
            head_sha=head,
        )

        if mergeable_state == "CONFLICTING" or merge_state == "DIRTY":
            return replace(
                base, error=f"merge conflict (mergeable={mergeable_state}, state={merge_state})"
            )
        if mergeable_state == "UNKNOWN" or not mergeable_state or merge_state == "UNKNOWN":
            last = replace(
                base,
                error=f"mergeability still UNKNOWN after {attempt + 1} check(s)",
            )
            continue
        if mergeable_state != "MERGEABLE":
            return replace(base, error=f"unexpected mergeable state {mergeable_state!r}")
        if expected_head_sha and head != expected_head_sha:
            return replace(
                base,
                error=(
                    f"head changed: expected {expected_head_sha[:12]},"
                    f" PR head is {(head or '?')[:12]}"
                ),
            )
        return replace(base, mergeable=True)
    return last


def _pr_state(repo: str, pr_number: int) -> str:
    """Return the PR ``state`` (``OPEN``/``MERGED``/``CLOSED``) or ``""``."""
    proc = _run_gh("pr", "view", str(pr_number), "--repo", repo, "--json", "state")
    if proc.returncode != 0:
        return ""
    data = _json_dict(proc.stdout)
    if data is None:
        return ""
    return str(data.get("state") or "").upper()


def merge_pr(
    repo: str,
    pr_number: int,
    strategy: str = "squash",
    *,
    match_head_commit: str | None = None,
) -> MergeResult:
    """Merge a PR with the requested strategy.

    Valid strategies match ``gh pr merge``'s flags:
    ``squash`` / ``merge`` / ``rebase``.  Unknown values fall
    back to ``squash`` (the issue spec's default).

    When *match_head_commit* is given it is passed as
    ``--match-head-commit`` so GitHub refuses the merge if the PR head
    moved since that SHA was recorded.  Success is gated on the real
    exit code AND a follow-up ``gh pr view --json state`` reporting
    ``MERGED``; anything else is ``merged=False``.

    Returns :class:`MergeResult` regardless of outcome.  Common
    failure: ``gh pr merge`` blocked by branch protection — the
    error text from gh is preserved verbatim so the operator
    can act on it.  We DO NOT pass ``--admin`` to override.
    """
    if not _gh_available():
        return MergeResult(
            merged=False,
            pr_number=pr_number,
            strategy=strategy,
            error="gh CLI not available on PATH",
        )
    if strategy not in ("squash", "merge", "rebase"):
        logger.warning(
            "Unknown merge strategy %r; defaulting to 'squash'",
            strategy,
        )
        strategy = "squash"

    if match_head_commit is not None and (
        not match_head_commit or match_head_commit.startswith("-")
    ):
        return MergeResult(
            merged=False,
            pr_number=pr_number,
            strategy=strategy,
            error=f"invalid head SHA: {match_head_commit!r}",
        )

    strategy_flag = f"--{strategy}"
    args = [
        "pr",
        "merge",
        str(pr_number),
        "--repo",
        repo,
        strategy_flag,
        "--delete-branch",
    ]
    if match_head_commit:
        args.extend(["--match-head-commit", match_head_commit])
    proc = _run_gh(*args)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        logger.warning(
            "gh pr merge %s (%s) failed: %s",
            pr_number,
            strategy,
            err[:300],
        )
        return MergeResult(
            merged=False,
            pr_number=pr_number,
            strategy=strategy,
            error=err or "gh pr merge failed",
        )
    # Exit code 0 is necessary but not sufficient: confirm GitHub now
    # reports the PR as MERGED (e.g. not merely queued).
    state = _pr_state(repo, pr_number)
    if state != "MERGED":
        logger.warning(
            "gh pr merge %s exited 0 but PR state is %r",
            pr_number,
            state or "unknown",
        )
        return MergeResult(
            merged=False,
            pr_number=pr_number,
            strategy=strategy,
            error=f"merge not confirmed: PR state is {state or 'unknown'!s}",
        )
    return MergeResult(
        merged=True,
        pr_number=pr_number,
        strategy=strategy,
    )


def close_pr(repo: str, pr_number: int, reason: str) -> bool:
    """Close a PR without merging, leaving the branch alive.

    *reason* is posted as a closing comment via ``--comment`` so
    the human reviewer landing on the PR later sees why the
    orchestrator gave up on it.  We deliberately do NOT pass
    ``--delete-branch`` — the branch is the artifact a human
    will review.

    Returns ``True`` when the PR is now closed.
    """
    if not _gh_available():
        return False
    args = ["pr", "close", str(pr_number), "--repo", repo]
    reason_text = reason.strip()
    if reason_text:
        args.extend(["--comment", reason_text])
    proc = _run_gh(*args)
    if proc.returncode != 0:
        logger.warning(
            "gh pr close %s failed: %s",
            pr_number,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return False
    return True
