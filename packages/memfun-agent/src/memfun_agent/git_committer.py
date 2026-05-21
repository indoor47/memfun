"""Auto-commit produced files after a successful workflow run (issue #37).

When :class:`WorkflowEngine` finishes, this module is responsible for
turning the heap of files the swarm wrote into a real git commit so the
user can ``git push`` immediately rather than reaching for ``git add -A``.

The contract is deliberately small:

* :func:`commit_workflow_output` — stage *files_written*, build a
  structured commit message, run ``git commit``, optionally
  ``git push``, and return a :class:`CommitResult` so callers can show
  the user what happened.
* :func:`create_swarm_branch` — pre-workflow branch creation for the
  ``--branch=auto`` / ``--branch=<name>`` flow.

Both helpers are **graceful by design**:

* Operating on a directory that is not a git working tree is a silent
  skip — the workflow output still landed on disk, the user just gets
  no commit.
* Pre-commit hooks that modify staged files are detected via
  ``git diff --cached --quiet`` after the failed commit; we re-add the
  same set of paths and retry once.
* Pre-commit hook hard failures (lint errors, etc.) are surfaced to
  the caller via :attr:`CommitResult.error` so the swarm command can
  exit non-zero with the hook output preserved.
* Push errors (no remote, auth failure, non-fast-forward, …) never
  fail the commit step — they only set :attr:`CommitResult.push_error`.

The structured commit message format follows the spec from issue #37::

    memfun: <spec>

    Sub-tasks (N):
      T1: areas.py (coder-agent, approved)
      T2: angles.py (coder-agent, approved)
      ...

    LLM: <calls> calls · <in>K in · <out>K out · <wall>min wall
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = get_logger("agent.git_committer")


# ── Public dataclasses ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SubTaskReview:
    """Per-sub-task review outcome for the commit message body.

    Populated by :class:`WorkflowEngine` from its
    ``per_task_reviews`` map.  Used purely for human-readable rendering
    inside the commit message body.
    """

    task_id: str
    files: tuple[str, ...]
    agent_name: str
    approved: bool


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of an auto-commit attempt.

    Attributes:
        committed: Whether the commit actually landed.  ``False`` when
            we skipped (not a git repo, no files to stage, --no-commit
            flag, etc.) or when the commit itself failed.
        skipped: When ``True``, ``committed`` will also be ``False``
            and ``error`` will explain *why* we skipped (eg. "not a
            git repository").  Skips never raise.
        commit_sha: 40-character commit SHA when ``committed`` is
            ``True``.  ``None`` otherwise.
        branch: Name of the branch the commit landed on.  Set even
            when ``committed`` is ``False`` if we managed to resolve
            HEAD — useful for "no files to commit" reporting.
        pushed: ``True`` when ``git push`` succeeded.  Always
            ``False`` when ``push`` was not requested.
        push_error: Human-readable push failure (eg. missing remote)
            when push was requested but did not succeed.  ``None``
            otherwise.
        error: Commit failure reason — typically a pre-commit hook's
            stderr.  ``None`` on success or clean skip.
        files_committed: Paths actually present in the commit (a
            subset of the input list — paths the user already removed
            or that were outside the repo are filtered out).
    """

    committed: bool
    skipped: bool = False
    commit_sha: str | None = None
    branch: str | None = None
    pushed: bool = False
    push_error: str | None = None
    error: str | None = None
    files_committed: tuple[str, ...] = field(default_factory=tuple)


# ── Internals ─────────────────────────────────────────────────────


class _GitRunner:
    """Tiny subprocess wrapper for the ``git`` CLI rooted at a project.

    Kept as a private class rather than free functions so tests can
    swap it out and so the cwd / encoding handling stays in one place.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def run(
        self,
        *args: str,
        check: bool = False,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``git <args>`` from the project root.

        Captures stdout + stderr and returns the
        :class:`subprocess.CompletedProcess`.  When ``check`` is
        ``True``, raises :class:`subprocess.CalledProcessError` on a
        non-zero exit; otherwise the caller inspects ``returncode``.
        """
        cmd = ("git", *args)
        logger.debug("git_committer: %s (cwd=%s)", " ".join(cmd), self._root)
        return subprocess.run(
            cmd,
            cwd=str(self._root),
            capture_output=True,
            text=True,
            check=check,
            input=input_text,
        )


def _is_git_repo(project_root: Path) -> bool:
    """Return ``True`` if *project_root* is inside a git working tree.

    Uses ``git rev-parse --is-inside-work-tree`` rather than just
    checking for ``.git/`` because the latter is wrong for both
    submodules (``.git`` is a file) and worktrees (``.git`` is a
    file pointing at the real ``gitdir``).
    """
    if not project_root.exists():
        return False
    try:
        proc = subprocess.run(
            ("git", "rev-parse", "--is-inside-work-tree"),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _current_branch(runner: _GitRunner) -> str | None:
    """Return the current branch name, or ``None`` on detached HEAD."""
    proc = runner.run("rev-parse", "--abbrev-ref", "HEAD")
    if proc.returncode != 0:
        return None
    name = proc.stdout.strip()
    if not name or name == "HEAD":
        return None
    return name


def _normalise_paths(
    project_root: Path, files_written: Iterable[Path | str]
) -> list[str]:
    """Convert mixed absolute/relative paths into repo-relative strings.

    Drops:
      * paths that don't exist on disk (the LLM may have hallucinated
        a write, or a later step rolled the file back),
      * paths that resolve outside the project root (a worktree-leaked
        absolute path shouldn't be staged into the main repo).

    Returns a de-duplicated list preserving the input order.
    """
    root = project_root.resolve()
    seen: set[str] = set()
    out: list[str] = []
    for raw in files_written:
        if raw is None:
            continue
        p = Path(raw)
        candidate = p if p.is_absolute() else (root / p)
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.exists():
            logger.debug("git_committer: skip missing %s", resolved)
            continue
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            logger.debug(
                "git_committer: skip path outside repo: %s", resolved
            )
            continue
        rel_str = rel.as_posix()
        if rel_str in seen:
            continue
        seen.add(rel_str)
        out.append(rel_str)
    return out


def _build_commit_message(
    spec: str,
    sub_tasks: list[SubTaskReview],
    stats: Mapping[str, object] | None,
) -> str:
    """Render the structured commit message body.

    Format (from issue #37)::

        memfun: <spec one-liner>

        Sub-tasks (N):
          T1: files (agent_name, approved|changes_requested)
          ...

        LLM: <calls> calls · <in>K in · <out>K out · <wall>min wall

    The "LLM:" line is omitted if no stats are passed in.  Sub-tasks
    section is omitted when *sub_tasks* is empty (single-task or
    non-pairing workflows).
    """
    # First line: keep it under 100 chars — the user's editor / GitHub
    # PR title display will thank us.
    one_liner = " ".join(spec.split())  # collapse whitespace
    if len(one_liner) > 88:
        one_liner = one_liner[:85] + "..."
    title = f"memfun: {one_liner}"

    body_lines: list[str] = []

    if sub_tasks:
        body_lines.append(f"Sub-tasks ({len(sub_tasks)}):")
        for st in sub_tasks:
            files_label = ", ".join(st.files) if st.files else "(no files)"
            status_label = "approved" if st.approved else "changes_requested"
            body_lines.append(
                f"  {st.task_id}: {files_label} "
                f"({st.agent_name}, {status_label})"
            )

    if stats:
        # Render LLM stats in the same shape the issue showed.
        # We accept missing keys silently — the auto-commit is best
        # effort, we don't want to crash because telemetry was off.
        calls = stats.get("calls")
        tokens_in = stats.get("tokens_in")
        tokens_out = stats.get("tokens_out")
        wall_seconds = stats.get("wall_seconds")
        wall_label: str | None
        if isinstance(wall_seconds, int | float) and wall_seconds > 0:
            mins = int(wall_seconds // 60)
            secs = int(wall_seconds % 60)
            wall_label = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        else:
            wall_label = None

        bits: list[str] = []
        if isinstance(calls, int) and calls >= 0:
            bits.append(f"{calls} calls")
        if isinstance(tokens_in, int) and tokens_in >= 0:
            bits.append(f"{tokens_in / 1000:.1f}K in")
        if isinstance(tokens_out, int) and tokens_out >= 0:
            bits.append(f"{tokens_out / 1000:.1f}K out")
        if wall_label is not None:
            bits.append(f"{wall_label} wall")
        if bits:
            if body_lines:
                body_lines.append("")
            body_lines.append("LLM: " + " · ".join(bits))

    if body_lines:
        return f"{title}\n\n" + "\n".join(body_lines) + "\n"
    return f"{title}\n"


# ── Public API ────────────────────────────────────────────────────


def create_swarm_branch(
    project_root: Path,
    branch: str,
) -> tuple[str, str | None]:
    """Create (and check out) *branch* off the current HEAD.

    Used by the ``--branch=<name>`` / ``--branch=auto`` CLI flow.  The
    special value ``"auto"`` is expanded to
    ``memfun/swarm-<UTC-timestamp>``.

    Returns a ``(branch_name, error)`` tuple.  When ``error`` is
    ``None`` the checkout succeeded; otherwise the branch name was
    still rendered (caller decides what to do).

    Skips silently with an explanatory error when *project_root* is
    not a git repository.
    """
    if branch == "auto":
        ts = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%d-%H%M%S")
        branch_name = f"memfun/swarm-{ts}"
    else:
        branch_name = branch

    if not _is_git_repo(project_root):
        return branch_name, "not a git repository"

    runner = _GitRunner(project_root)
    # ``git checkout -b <name>`` creates and switches in one step.
    # The branch is forked from the current HEAD, which is exactly
    # what we want for the "fresh branch off main" use case.
    proc = runner.run("checkout", "-b", branch_name)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return branch_name, err or "git checkout failed"
    return branch_name, None


def commit_workflow_output(
    project_root: Path,
    files_written: Iterable[Path | str],
    spec: str,
    *,
    sub_tasks: list[SubTaskReview] | None = None,
    stats: Mapping[str, object] | None = None,
    push: bool = False,
) -> CommitResult:
    """Stage and commit *files_written*; optionally push.

    Args:
        project_root: Top-level directory of the user's project.
        files_written: Iterable of paths (absolute or repo-relative)
            the workflow produced.  Paths that don't exist, aren't
            files, or live outside *project_root* are filtered out.
        spec: The user's original spec — first line of the commit
            message.
        sub_tasks: Optional per-sub-task summaries for the message
            body.  Omitted from the message when empty / ``None``.
        stats: Optional LLM telemetry summary
            (``{"calls": int, "tokens_in": int, "tokens_out": int,
            "wall_seconds": float}``).  Missing keys are silently
            dropped from the rendered line.
        push: When ``True`` and the commit lands, run
            ``git push origin <branch>``.  Push failure does not fail
            the commit — it is surfaced via
            :attr:`CommitResult.push_error`.

    Returns:
        A :class:`CommitResult` describing what happened.

    Skip semantics:
        * *project_root* not a git repository → ``CommitResult(
          committed=False, skipped=True, error="not a git repository")``
          — the caller treats this as success.
        * No paths survive normalisation → ``CommitResult(
          committed=False, skipped=True, error="no files to commit")``
          — same treatment.
        * Nothing staged after ``git add`` (eg. all files already
          committed, or all .gitignored) → skipped with the appropriate
          message.

    Hard failure semantics:
        * Pre-commit hook returns non-zero → ``CommitResult(
          committed=False, error="<hook stderr>")``.  Caller MUST
          surface this and exit non-zero.

    Push semantics:
        * No remote configured → ``CommitResult(committed=True,
          pushed=False, push_error="no remote configured")``.  Caller
          shows a warning, exits 0.
    """
    root = project_root.resolve() if project_root else Path.cwd()

    if not _is_git_repo(root):
        logger.info("git_committer: %s is not a git repo — skipping", root)
        return CommitResult(
            committed=False,
            skipped=True,
            error="not a git repository",
        )

    paths = _normalise_paths(root, files_written)
    if not paths:
        logger.info("git_committer: no files to commit in %s", root)
        runner = _GitRunner(root)
        return CommitResult(
            committed=False,
            skipped=True,
            branch=_current_branch(runner),
            error="no files to commit",
        )

    runner = _GitRunner(root)
    branch = _current_branch(runner)

    # ── Stage ────────────────────────────────────────────────
    # Use ``--`` separator so a path starting with ``-`` is never
    # mistaken for a git flag.  ``git add`` on already-tracked /
    # already-staged files is a harmless no-op.
    add_proc = runner.run("add", "--", *paths)
    if add_proc.returncode != 0:
        err = (add_proc.stderr or add_proc.stdout or "").strip()
        logger.warning("git_committer: git add failed: %s", err)
        return CommitResult(
            committed=False,
            branch=branch,
            error=err or "git add failed",
        )

    # If nothing was actually staged (eg. every file was .gitignored
    # or already up-to-date), bail with a clean skip rather than
    # creating an empty commit.
    diff_proc = runner.run(
        "diff", "--cached", "--quiet", "--", *paths,
    )
    if diff_proc.returncode == 0:
        logger.info(
            "git_committer: nothing to commit (paths were already "
            "tracked or ignored): %s", paths,
        )
        return CommitResult(
            committed=False,
            skipped=True,
            branch=branch,
            files_committed=(),
            error="nothing staged",
        )

    # ── Commit ───────────────────────────────────────────────
    message = _build_commit_message(spec, sub_tasks or [], stats)
    commit_proc = runner.run("commit", "-m", message)

    if commit_proc.returncode != 0:
        # A common failure mode: a pre-commit hook reformatted files,
        # which leaves the worktree dirty but with nothing staged.
        # If that's the case, re-stage the same paths and try once
        # more before giving up.
        post_diff = runner.run("diff", "--quiet", "--", *paths)
        if post_diff.returncode != 0:
            # Files were modified by the hook — re-stage + retry.
            logger.info(
                "git_committer: pre-commit hook modified files; "
                "re-staging and retrying commit",
            )
            runner.run("add", "--", *paths)
            commit_proc = runner.run("commit", "-m", message)

    if commit_proc.returncode != 0:
        err = (
            (commit_proc.stderr or "")
            + ("\n" if commit_proc.stderr and commit_proc.stdout else "")
            + (commit_proc.stdout or "")
        ).strip()
        logger.warning("git_committer: git commit failed: %s", err)
        return CommitResult(
            committed=False,
            branch=branch,
            error=err or "git commit failed",
            files_committed=tuple(paths),
        )

    # Resolve the SHA we just produced.
    sha_proc = runner.run("rev-parse", "HEAD")
    sha = sha_proc.stdout.strip() if sha_proc.returncode == 0 else None

    pushed = False
    push_error: str | None = None
    if push:
        pushed, push_error = _try_push(runner, branch)

    return CommitResult(
        committed=True,
        commit_sha=sha,
        branch=branch,
        pushed=pushed,
        push_error=push_error,
        files_committed=tuple(paths),
    )


def _try_push(
    runner: _GitRunner, branch: str | None,
) -> tuple[bool, str | None]:
    """Attempt to push *branch* to ``origin``.

    Returns ``(pushed, error)``.  When ``error`` is set we did NOT
    push — caller logs the warning and continues.  We never raise.
    """
    # Refuse to push from a detached HEAD — there's no sensible branch
    # to push.  Same logic as ``git push`` itself, but we want a
    # cleaner error message than git's default.
    if branch is None:
        return False, "cannot push from detached HEAD"

    # Check that ``origin`` actually exists.  ``git remote get-url
    # origin`` returns 0 + URL on success, non-zero on missing remote.
    remote_proc = runner.run("remote", "get-url", "origin")
    if remote_proc.returncode != 0:
        return False, "no remote configured"

    push_proc = runner.run("push", "origin", branch)
    if push_proc.returncode != 0:
        err = (push_proc.stderr or push_proc.stdout or "").strip()
        return False, err or "git push failed"
    return True, None
