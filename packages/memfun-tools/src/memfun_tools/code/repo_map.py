from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

repo_map_server = FastMCP("memfun-repo-map")

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".eggs", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

IGNORE_EXTS = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".class", ".jar"}


def _build_tree(root: Path, max_depth: int = 4, _depth: int = 0) -> list[str]:
    """Build an indented tree representation."""
    if _depth > max_depth:
        return ["  " * _depth + "..."]

    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return []

    for entry in entries:
        if entry.name.startswith(".") and entry.name not in {".github"}:
            continue
        if entry.is_dir():
            if entry.name in IGNORE_DIRS:
                continue
            lines.append("  " * _depth + f"{entry.name}/")
            lines.extend(_build_tree(entry, max_depth, _depth + 1))
        elif entry.is_file():
            if entry.suffix in IGNORE_EXTS:
                continue
            size = entry.stat().st_size
            lines.append("  " * _depth + f"{entry.name} ({_fmt_size(size)})")

    return lines


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


@repo_map_server.tool()
async def repo_map(path: str = ".", max_depth: int = 4) -> str:
    """Generate a tree-structured map of the repository.

    Args:
        path: Root directory path.
        max_depth: Maximum directory depth to explore.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    lines = [f"{root.name}/"]
    lines.extend(_build_tree(root, max_depth))
    return "\n".join(lines)


@repo_map_server.tool()
async def file_summary(path: str = ".") -> str:
    """Summarize file types and counts in the repository.

    Args:
        path: Root directory path.
    """
    root = Path(path).resolve()
    ext_counts: dict[str, int] = {}
    ext_sizes: dict[str, int] = {}
    total_files = 0

    for fpath in root.rglob("*"):
        if not fpath.is_file():
            continue
        if any(part in IGNORE_DIRS for part in fpath.parts):
            continue
        if fpath.suffix in IGNORE_EXTS:
            continue

        ext = fpath.suffix or "(no ext)"
        size = fpath.stat().st_size
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        ext_sizes[ext] = ext_sizes.get(ext, 0) + size
        total_files += 1

    lines = [f"Total files: {total_files}", ""]
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"  {ext:>10}: {count:>5} files  ({_fmt_size(ext_sizes[ext])})")

    return "\n".join(lines)
