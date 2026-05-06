from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

fs_server = FastMCP("memfun-fs-tools")

# Paths that must never be read or written by the agent tools.
_BLOCKED_PATH_PREFIXES: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
)

_BLOCKED_PATH_NAMES: frozenset[str] = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
})


def _project_root_guard() -> Path | None:
    """Return the configured project root, or ``None`` if unscoped.

    When the ``MEMFUN_PROJECT_ROOT`` environment variable is set, all
    filesystem tool calls must resolve to a path strictly inside that
    directory.  This is used by per-task worktrees (issue #6/#8) to
    confine specialists to their isolated checkout.
    """
    raw = os.environ.get("MEMFUN_PROJECT_ROOT")
    if not raw:
        return None
    return Path(raw).resolve()


def _check_in_root(path: Path) -> Path:
    """Resolve *path* and confirm it lies inside ``MEMFUN_PROJECT_ROOT``.

    Symlinks are followed via ``Path.resolve()`` *before* the
    containment check, so a symlink planted inside the project root
    that points outside cannot bypass the guard.

    When ``MEMFUN_PROJECT_ROOT`` is unset, only the blocked-path /
    blocked-name guards apply (backward compatible with pre-#13
    callers).

    Returns:
        The fully resolved path.

    Raises:
        ValueError: If the resolved path is outside the project root.
        PermissionError: If the path matches a blocked prefix or name.
    """
    resolved = path.resolve()

    # Sensitive system / secret paths are always rejected, regardless
    # of MEMFUN_PROJECT_ROOT.  We compare against both the original
    # textual path and the resolved path because some sensitive
    # locations sit behind a platform symlink (``/etc`` ->
    # ``/private/etc`` on macOS) and an attacker could equally use
    # either form to reach them.
    candidates = (str(path), str(resolved))
    for prefix in _BLOCKED_PATH_PREFIXES:
        if any(c.startswith(prefix) for c in candidates):
            raise PermissionError(
                f"Access denied: {resolved} is a protected"
                " system file"
            )
    if resolved.name in _BLOCKED_PATH_NAMES or path.name in _BLOCKED_PATH_NAMES:
        raise PermissionError(
            f"Access denied: {resolved.name} may contain secrets"
        )

    root = _project_root_guard()
    if root is not None and not resolved.is_relative_to(root):
        raise ValueError(
            f"path outside project root: {resolved}"
            f" (MEMFUN_PROJECT_ROOT={root})"
        )
    return resolved


@fs_server.tool()
async def read_file(
    path: str, offset: int = 0, limit: int = 2000
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        path: Absolute path to the file.
        offset: Line number to start from (0-indexed).
        limit: Maximum number of lines to return.
    """
    p = _check_in_root(Path(path))
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    lines = p.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    selected = lines[offset : offset + limit]
    numbered = [
        f"{i + offset + 1:>6}\t{line}"
        for i, line in enumerate(selected)
    ]
    return "\n".join(numbered)


@fs_server.tool()
async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating dirs if needed.

    Args:
        path: Absolute path to the file.
        content: Content to write.
    """
    p = _check_in_root(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


@fs_server.tool()
async def edit_file(path: str, old: str, new: str) -> str:
    """Replace the first occurrence of *old* with *new* in *path*.

    Mirrors the in-process REPL helper used by the RLM agent.  The
    file must already exist and *old* must appear exactly once
    (otherwise the edit is ambiguous).  ``MEMFUN_PROJECT_ROOT`` is
    enforced before any read or write happens.

    Args:
        path: Absolute path to the file to edit.
        old: Exact substring to replace.
        new: Replacement text.

    Returns:
        Human-readable confirmation message.

    Raises:
        ValueError: If *old* is empty, missing from the file, or
            appears more than once; or if *path* falls outside
            ``MEMFUN_PROJECT_ROOT``.
        FileNotFoundError: If *path* does not exist.
    """
    if not old:
        raise ValueError("edit_file: 'old' must be non-empty")

    p = _check_in_root(Path(path))
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")
    occurrences = text.count(old)
    if occurrences == 0:
        raise ValueError(
            f"edit_file: 'old' not found in {path}"
        )
    if occurrences > 1:
        raise ValueError(
            f"edit_file: 'old' is ambiguous in {path}"
            f" ({occurrences} matches); add more context"
        )

    new_text = text.replace(old, new, 1)
    p.write_text(new_text, encoding="utf-8")
    return (
        f"Edited {path}: replaced {len(old)} chars"
        f" with {len(new)} chars"
    )


@fs_server.tool()
async def list_directory(
    path: str, recursive: bool = False
) -> str:
    """List files and directories at a path.

    Args:
        path: Directory path.
        recursive: If true, list recursively (max 1000).
    """
    p = _check_in_root(Path(path))
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []
    if recursive:
        for item in sorted(p.rglob("*"))[:1000]:
            rel = item.relative_to(p)
            prefix = "d " if item.is_dir() else "f "
            entries.append(f"{prefix}{rel}")
    else:
        for item in sorted(p.iterdir()):
            prefix = "d " if item.is_dir() else "f "
            if item.is_file():
                size = item.stat().st_size
                entries.append(
                    f"{prefix}{item.name}  ({size} bytes)"
                )
            else:
                entries.append(f"{prefix}{item.name}/")

    return (
        "\n".join(entries) if entries else "(empty directory)"
    )


@fs_server.tool()
async def glob_files(
    pattern: str, path: str = "."
) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '**/*.py').
        path: Base directory to search from.
    """
    base = _check_in_root(Path(path))
    root = _project_root_guard()

    raw_matches = sorted(base.glob(pattern))
    safe: list[Path] = []
    for m in raw_matches:
        # Pattern segments like ``../`` and symlinks pointing outside
        # the root must not leak past the guard, so each match is
        # resolved and re-checked.
        try:
            resolved = m.resolve()
        except OSError:
            continue
        if root is not None and not resolved.is_relative_to(root):
            continue
        safe.append(m)
        if len(safe) >= 500:
            break

    if not safe:
        return "(no matches)"
    return "\n".join(str(m.relative_to(base)) for m in safe)
