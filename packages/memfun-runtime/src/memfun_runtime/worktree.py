"""Per-agent isolated git worktree management.

When the :class:`WorkflowEngine` fans out parallel sub-tasks, every
specialist agent must edit files in isolation — otherwise concurrent
``write_file`` / ``edit_file`` operations clobber each other in the
single shared working tree.

:class:`WorktreeManager` provisions and reaps ``git worktree``
directories under ``<project_root>/.memfun/worktrees/<workflow_id>/<task_id>``.
Each call to :func:`WorktreeManager.make_worktree` returns an
isolated absolute path that the workflow engine pushes through
``TaskMessage.payload["cwd"]``; specialists then resolve all file
operations against that path.

The implementation is deliberately small: thin wrappers over
``git worktree add`` / ``git worktree remove`` / ``git branch -D``.
It does *not* manage commit graphs, pull requests, or merging —
that is the caller's job.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from memfun_core.logging import get_logger

logger = get_logger("runtime.worktree")


# ── Errors ─────────────────────────────────────────────────────


class WorktreeError(Exception):
    """Raised when a git worktree operation fails."""


# ── Validation ─────────────────────────────────────────────────


# Worktree directory components must be safe filesystem names.
# We reuse the same conservative ref pattern as ``code/git.py``: no
# leading dash (would be parsed as a git flag), and only an
# alphanumeric / dot / dash / underscore alphabet.  This prevents
# both shell injection and path traversal via ``..`` segments.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")
_SAFE_REF = re.compile(r"^[a-zA-Z0-9_./@^~:][a-zA-Z0-9_./@^~:\-]{0,254}$")


def _validate_id(label: str, value: str) -> str:
    """Validate a workflow_id or task_id segment."""
    if not _SAFE_ID.match(value):
        raise WorktreeError(f"Invalid {label}: {value!r} (must match {_SAFE_ID.pattern})")
    return value


def _validate_base_sha(value: str) -> str:
    """Validate a base SHA / ref / branch name."""
    if value.startswith("-"):
        raise WorktreeError(f"Invalid base_sha: {value!r} (must not start with '-')")
    if not _SAFE_REF.match(value):
        raise WorktreeError(f"Invalid base_sha: {value!r}")
    return value


# ── Result type ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WorktreeInfo:
    """Information about a git worktree.

    Returned by :func:`WorktreeManager.list_worktrees` for callers
    that want both the path and the auto-generated branch name.
    """

    path: Path
    branch: str | None
    head: str | None


# ── Manager ────────────────────────────────────────────────────


class WorktreeManager:
    """Provision and reap per-task git worktrees.

    Worktrees live under ``<project_root>/.memfun/worktrees/<workflow_id>/<task_id>``.
    Each one is created with an auto-generated branch named
    ``memfun/<workflow_id>/<task_id>`` so the parent repository can
    keep track of work without polluting the user's branch list.

    The manager is *idempotent*: ``make_worktree`` re-issued with
    the same arguments returns the existing path; ``cleanup_worktree``
    tolerates already-removed worktrees.

    Example::

        manager = WorktreeManager(project_root=Path.cwd())
        path = manager.make_worktree("wf-123", "T1")
        # ... agent runs with cwd=path ...
        manager.cleanup_worktree(path)
    """

    BASE_REL: Path = Path(".memfun") / "worktrees"
    BRANCH_PREFIX: str = "memfun"

    def __init__(self, project_root: Path | str | None = None) -> None:
        self._project_root = Path(project_root or Path.cwd()).resolve()

    # ── Properties ────────────────────────────────────────────

    @property
    def project_root(self) -> Path:
        """Absolute path of the parent git repository."""
        return self._project_root

    @property
    def base_dir(self) -> Path:
        """Absolute path of the directory containing all worktrees."""
        return self._project_root / self.BASE_REL

    # ── Public API ────────────────────────────────────────────

    def make_worktree(
        self,
        workflow_id: str,
        task_id: str,
        base_sha: str | None = None,
    ) -> Path:
        """Create (or return existing) worktree for a sub-task.

        Args:
            workflow_id: Outer workflow identifier — used as the first
                path segment under ``.memfun/worktrees/``.
            task_id: Sub-task identifier — used as the leaf segment.
            base_sha: Optional commit / branch / tag to check out.
                When ``None``, defaults to ``HEAD`` of the parent
                repository.

        Returns:
            Absolute path of the worktree directory.

        Raises:
            WorktreeError: If the inputs are invalid or git refuses
                to create the worktree.
        """
        wf = _validate_id("workflow_id", workflow_id)
        tid = _validate_id("task_id", task_id)
        ref = _validate_base_sha(base_sha) if base_sha is not None else "HEAD"

        target = (self.base_dir / wf / tid).resolve()

        # Idempotent: return existing path if git already tracks it.
        if target.is_dir() and self._is_registered_worktree(target):
            logger.debug(
                "worktree already present: %s",
                target,
            )
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        branch = self._branch_name(wf, tid)

        # ``git worktree add -B <branch> <path> <ref>`` creates or
        # resets the branch and checks it out in the new worktree.
        # -B is safe because we validated branch above.
        self._run_git(
            "worktree",
            "add",
            "-B",
            branch,
            str(target),
            ref,
        )
        logger.info(
            "Created worktree %s (branch=%s, ref=%s)",
            target,
            branch,
            ref,
        )
        return target

    def cleanup_worktree(self, path: Path | str) -> None:
        """Remove a worktree and its auto-generated branch.

        Tolerates already-removed worktrees: any combination of
        "directory missing", "git doesn't know about it", and
        "branch already deleted" succeeds silently.

        Args:
            path: Absolute or relative path of the worktree.
        """
        target = Path(path).resolve()

        # Worktree remove (best-effort).
        if target.exists() or self._is_registered_worktree(target):
            try:
                self._run_git(
                    "worktree",
                    "remove",
                    "--force",
                    str(target),
                )
            except WorktreeError as exc:
                # If git doesn't know about this path, fall back to
                # plain ``rmtree`` — keeps cleanup idempotent.
                logger.debug(
                    "worktree remove failed (%s); falling back to rmtree",
                    exc,
                )
                self._rmtree_quiet(target)
        else:
            logger.debug("worktree %s already gone", target)

        # Branch delete (best-effort).  We can only guess the branch
        # name from the path layout — anything else is the caller's
        # responsibility.
        branch = self._branch_from_path(target)
        if branch is not None:
            try:
                self._run_git("branch", "-D", branch)
                logger.debug("deleted branch %s", branch)
            except WorktreeError:
                # Branch may already be gone, or may have been
                # claimed by another worktree.  Either way, not our
                # problem.
                logger.debug("branch %s already gone", branch)

        # Always prune stale entries from git's bookkeeping.
        with contextlib.suppress(WorktreeError):
            self._run_git("worktree", "prune")

    def list_worktrees(self) -> list[WorktreeInfo]:
        """Return memfun-managed worktrees currently registered with git.

        Worktrees outside ``.memfun/worktrees/`` are ignored — the
        manager only owns its own subtree, and we don't want to
        accidentally report (or worse, destroy) a worktree the user
        created themselves.
        """
        try:
            out = self._run_git("worktree", "list", "--porcelain")
        except WorktreeError:
            return []

        results: list[WorktreeInfo] = []
        base = self.base_dir.resolve()
        current_path: Path | None = None
        current_head: str | None = None
        current_branch: str | None = None

        for line in out.splitlines():
            if line.startswith("worktree "):
                # Flush previous record.
                if current_path is not None and self._is_under(
                    current_path,
                    base,
                ):
                    results.append(
                        WorktreeInfo(
                            path=current_path,
                            branch=current_branch,
                            head=current_head,
                        )
                    )
                current_path = Path(line[len("worktree ") :]).resolve()
                current_head = None
                current_branch = None
            elif line.startswith("HEAD "):
                current_head = line[len("HEAD ") :].strip()
            elif line.startswith("branch "):
                ref = line[len("branch ") :].strip()
                # ``refs/heads/<name>`` -> <name>
                current_branch = ref.removeprefix("refs/heads/")

        if current_path is not None and self._is_under(current_path, base):
            results.append(
                WorktreeInfo(
                    path=current_path,
                    branch=current_branch,
                    head=current_head,
                )
            )

        return results

    # ── Internals ─────────────────────────────────────────────

    def _branch_name(self, workflow_id: str, task_id: str) -> str:
        return f"{self.BRANCH_PREFIX}/{workflow_id}/{task_id}"

    def _branch_from_path(self, path: Path) -> str | None:
        """Recover the auto-generated branch name from a worktree path.

        Returns ``None`` if *path* is not located under our
        managed ``base_dir`` (in which case we have no claim to
        delete its branch).
        """
        try:
            rel = path.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) != 2:
            return None
        workflow_id, task_id = parts
        # Defensive: make sure the inferred IDs would have validated.
        if not (_SAFE_ID.match(workflow_id) and _SAFE_ID.match(task_id)):
            return None
        return self._branch_name(workflow_id, task_id)

    def _is_registered_worktree(self, path: Path) -> bool:
        """Return ``True`` if git already tracks *path* as a worktree."""
        try:
            out = self._run_git("worktree", "list", "--porcelain")
        except WorktreeError:
            return False
        target = str(path.resolve())
        return any(line == f"worktree {target}" for line in out.splitlines())

    @staticmethod
    def _is_under(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def _rmtree_quiet(path: Path) -> None:
        """Best-effort recursive remove (no-op if path missing)."""
        import shutil

        if not path.exists():
            return
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.debug("rmtree(%s) failed: %s", path, exc)

    def _run_git(self, *args: str) -> str:
        """Run ``git <args>`` from the project root.

        Returns ``stdout`` on success.  Raises :class:`WorktreeError`
        on non-zero exit, ``FileNotFoundError`` (no git installed),
        or any other ``OSError``.
        """
        cmd = ("git", *args)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise WorktreeError("git executable not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise WorktreeError(
                f"git {' '.join(args)} failed (exit {exc.returncode}): {stderr}"
            ) from exc
        return proc.stdout
