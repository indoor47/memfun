from __future__ import annotations

import asyncio
import re

from fastmcp import FastMCP

git_server = FastMCP("memfun-git-tools")

# Pattern for safe git ref names (SHA hex, branch names, HEAD~N,
# tags).  Rejects anything that could inject flags or shell
# metacharacters.
_SAFE_REF = re.compile(
    r"^[a-zA-Z0-9_./@^~:][a-zA-Z0-9_./@^~:\-]{0,254}$"
)
_MAX_LOG_COUNT = 500


def _validate_ref(ref: str) -> str:
    """Validate a git ref (commit/branch/tag) is safe.

    Rejects refs starting with ``-`` (flag injection) and
    refs containing suspicious characters.
    """
    if ref.startswith("-"):
        raise ValueError(
            f"Invalid git ref: {ref!r} (must not start"
            " with '-')"
        )
    if not _SAFE_REF.match(ref):
        raise ValueError(f"Invalid git ref: {ref!r}")
    return ref


def _validate_file_arg(file_path: str) -> str:
    """Validate a file path argument for git commands.

    Rejects paths starting with ``-`` to prevent flag
    injection.
    """
    if file_path.startswith("-"):
        raise ValueError(
            f"Invalid file path: {file_path!r}"
            " (must not start with '-')"
        )
    return file_path


async def _git(args: list[str], cwd: str = ".") -> str:
    """Run a git command and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:"
            f" {stderr.decode()}"
        )
    return stdout.decode("utf-8", errors="replace")


@git_server.tool()
async def git_status(path: str = ".") -> str:
    """Show the working tree status.

    Args:
        path: Repository path.
    """
    return await _git(["status", "--short"], cwd=path)


@git_server.tool()
async def git_diff(
    path: str = ".",
    staged: bool = False,
    file: str | None = None,
) -> str:
    """Show changes in the working tree or staging area.

    Args:
        path: Repository path.
        staged: If true, show staged changes.
        file: Specific file to diff.
    """
    cmd = ["diff"]
    if staged:
        cmd.append("--cached")
    if file:
        # Use -- separator to prevent flag injection
        cmd.extend(["--", _validate_file_arg(file)])
    return await _git(cmd, cwd=path)


@git_server.tool()
async def git_log(
    path: str = ".",
    count: int = 10,
    oneline: bool = True,
) -> str:
    """Show recent commit history.

    Args:
        path: Repository path.
        count: Number of commits to show (max 500).
        oneline: If true, show one line per commit.
    """
    count = max(1, min(count, _MAX_LOG_COUNT))
    cmd = ["log", f"-{count}"]
    if oneline:
        cmd.append("--oneline")
    return await _git(cmd, cwd=path)


@git_server.tool()
async def git_show(
    commit: str = "HEAD", path: str = "."
) -> str:
    """Show details of a specific commit.

    Args:
        commit: Commit hash or reference.
        path: Repository path.
    """
    _validate_ref(commit)
    return await _git(
        ["show", "--stat", commit], cwd=path
    )


@git_server.tool()
async def git_blame(
    file: str, path: str = "."
) -> str:
    """Show who last modified each line of a file.

    Args:
        file: File path relative to repo root.
        path: Repository path.
    """
    _validate_file_arg(file)
    # Use -- separator to prevent flag injection
    return await _git(
        ["blame", "--line-porcelain", "--", file], cwd=path
    )
