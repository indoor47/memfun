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
* :func:`merge_pr` — ``gh pr merge <n> --<strategy>``.
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
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from memfun_core.logging import get_logger

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
    return PushResult(pushed=True, branch=branch)


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


def merge_pr(
    repo: str,
    pr_number: int,
    strategy: str = "squash",
) -> MergeResult:
    """Merge a PR with the requested strategy.

    Valid strategies match ``gh pr merge``'s flags:
    ``squash`` / ``merge`` / ``rebase``.  Unknown values fall
    back to ``squash`` (the issue spec's default).

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

    strategy_flag = f"--{strategy}"
    proc = _run_gh(
        "pr",
        "merge",
        str(pr_number),
        "--repo",
        repo,
        strategy_flag,
        "--delete-branch",
    )
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
