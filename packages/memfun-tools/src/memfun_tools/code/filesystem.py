from __future__ import annotations

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


def _assert_safe_path(p: Path) -> None:
    """Raise if *p* resolves to a known-sensitive location.

    This is a defence-in-depth measure: the agent should
    already be scoped to the project working directory, but
    we block obvious sensitive paths as an extra safeguard.
    """
    resolved = str(p)
    for prefix in _BLOCKED_PATH_PREFIXES:
        if resolved.startswith(prefix):
            raise PermissionError(
                f"Access denied: {resolved} is a protected"
                " system file"
            )
    if p.name in _BLOCKED_PATH_NAMES:
        raise PermissionError(
            f"Access denied: {p.name} may contain secrets"
        )


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
    p = Path(path).resolve()
    _assert_safe_path(p)
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
    p = Path(path).resolve()
    _assert_safe_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


@fs_server.tool()
async def list_directory(
    path: str, recursive: bool = False
) -> str:
    """List files and directories at a path.

    Args:
        path: Directory path.
        recursive: If true, list recursively (max 1000).
    """
    p = Path(path).resolve()
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
    base = Path(path).resolve()
    matches = sorted(base.glob(pattern))[:500]
    return (
        "\n".join(str(m.relative_to(base)) for m in matches)
        if matches
        else "(no matches)"
    )
