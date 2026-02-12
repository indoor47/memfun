"""File-based persistent memory for Memfun (MEMORY.md).

Implements the MEMORY.md pattern (inspired by Claude Code's CLAUDE.md).
Two files provide layered memory:

- ``~/.memfun/MEMORY.md``  -- global preferences (loaded first)
- ``.memfun/MEMORY.md``    -- project-specific (loaded second, overrides)

Both are always loaded every turn as highest-priority context.
User-editable, human-readable markdown.
"""
from __future__ import annotations

import contextlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("cli.memory")

# ── Constants ────────────────────────────────────────────────

_MAX_MEMORY_LINES = 50
_MAX_MEMORY_CHARS = 2000
_SIMILARITY_THRESHOLD = 0.75

_STARTER_MEMORY = """\
# Memfun Memory

## Preferences
<!-- Your preferences (ports, frameworks, coding style) -->

## Patterns
<!-- Project patterns (directory structure, naming, tools) -->

## Technical
<!-- Technical details (API endpoints, versions, configs) -->

## Workflow
<!-- Workflow preferences (test strategy, deploy process) -->
"""


# ── Path helpers ─────────────────────────────────────────────


def global_memory_path() -> Path:
    """Path to the global MEMORY.md file."""
    return Path.home() / ".memfun" / "MEMORY.md"


def project_memory_path() -> Path:
    """Path to the project-local MEMORY.md file."""
    return Path.cwd() / ".memfun" / "MEMORY.md"


# ── Reading ──────────────────────────────────────────────────


def load_memory_context() -> str:
    """Load both global and project MEMORY.md for context injection.

    Returns a formatted string ready to prepend to LLM context,
    or empty string if no memory files exist.
    """
    parts: list[str] = []

    # Global first
    global_content = _read_memory_file(global_memory_path())
    if global_content:
        parts.append(
            "=== GLOBAL MEMORY (always apply) ===\n"
            + global_content
        )

    # Project overrides
    project_content = _read_memory_file(project_memory_path())
    if project_content:
        parts.append(
            "=== PROJECT MEMORY (always apply,"
            " overrides global) ===\n"
            + project_content
        )

    if not parts:
        return ""

    header = (
        "=== AGENT MEMORY — ALWAYS FOLLOW THESE"
        " RULES ===\n"
        "The following are learned preferences and"
        " facts. ALWAYS apply them to your responses."
        " Project memory overrides global memory.\n"
    )
    return header + "\n\n".join(parts)


def _read_memory_file(path: Path) -> str:
    """Read a MEMORY.md, stripping comments and enforcing limits."""
    if not path.exists():
        return ""
    try:
        content = path.read_text().strip()
        lines = [
            line
            for line in content.split("\n")
            if line.strip()
            and not line.strip().startswith("<!--")
        ]
        lines = lines[:_MAX_MEMORY_LINES]
        result = "\n".join(lines)
        if len(result) > _MAX_MEMORY_CHARS:
            result = result[:_MAX_MEMORY_CHARS] + "\n..."
        return result
    except Exception:
        logger.debug(
            "Failed to read memory %s", path, exc_info=True
        )
        return ""


def get_memory_display() -> str:
    """Format memory for the ``/memory`` command.

    Shows both files with full contents (including comments)
    for user readability.
    """
    parts: list[str] = []

    gpath = global_memory_path()
    if gpath.exists():
        parts.append(
            f"[bold]Global memory[/bold] ({gpath}):\n"
        )
        try:
            parts.append(gpath.read_text())
        except Exception:
            parts.append("[red]Error reading file[/red]")
    else:
        parts.append(
            "[dim]No global memory"
            " (~/.memfun/MEMORY.md)[/dim]"
        )

    parts.append("")

    ppath = project_memory_path()
    if ppath.exists():
        parts.append(
            f"[bold]Project memory[/bold] ({ppath}):\n"
        )
        try:
            parts.append(ppath.read_text())
        except Exception:
            parts.append("[red]Error reading file[/red]")
    else:
        parts.append(
            "[dim]No project memory"
            " (.memfun/MEMORY.md)[/dim]"
        )

    return "\n".join(parts)


# ── Writing ──────────────────────────────────────────────────


def remember(
    text: str, *, project: bool = True
) -> str:
    """Add a line to MEMORY.md under the appropriate section.

    Args:
        text: The text to remember.
        project: Write to project MEMORY.md if True, global if False.

    Returns:
        Confirmation message string.
    """
    path = project_memory_path() if project else global_memory_path()
    _ensure_memory_file(path)

    section = _classify_section(text)
    content = path.read_text()
    lines = content.split("\n")

    if _is_duplicate(text, lines):
        return (
            f"Already remembered (similar entry exists):"
            f" {text}"
        )

    non_empty = [
        line
        for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("<!--")
    ]
    if len(non_empty) >= _MAX_MEMORY_LINES:
        return (
            f"Memory is full ({_MAX_MEMORY_LINES} entries)."
            f" Use /forget to remove old entries first."
        )

    new_line = f"- {text}"
    inserted = _insert_under_section(lines, section, new_line)

    if not inserted:
        lines.append(f"\n## {section}")
        lines.append(new_line)

    path.write_text("\n".join(lines))
    scope = "project" if project else "global"
    return f"Remembered ({scope}, {section}): {text}"


def forget(
    target: str, *, project: bool = True
) -> str:
    """Remove a line from MEMORY.md.

    Args:
        target: Line number (1-based) or text substring.
        project: Operate on project MEMORY.md if True, global if False.

    Returns:
        Confirmation message string.
    """
    path = project_memory_path() if project else global_memory_path()
    if not path.exists():
        return "No memory file found."

    content = path.read_text()
    lines = content.split("\n")

    # Try as line number first
    with contextlib.suppress(ValueError):
        line_num = int(target)
        content_lines = [
            (i, line)
            for i, line in enumerate(lines)
            if line.strip().startswith("- ")
        ]
        if 1 <= line_num <= len(content_lines):
            idx, removed_line = content_lines[line_num - 1]
            lines.pop(idx)
            path.write_text("\n".join(lines))
            return f"Forgot: {removed_line.strip()}"
        return (
            f"Line number {line_num} out of range"
            f" (1-{len(content_lines)})."
        )

    # Try as text substring match
    target_lower = target.lower()
    for i, line in enumerate(lines):
        if (
            line.strip().startswith("- ")
            and target_lower in line.lower()
        ):
            removed = lines.pop(i)
            path.write_text("\n".join(lines))
            return f"Forgot: {removed.strip()}"

    return f"No matching memory entry found for: {target}"


def append_learning(
    text: str, *, project: bool = True
) -> bool:
    """Append a learning extracted by the LLM to MEMORY.md.

    Called by LearningManager after each turn. Includes
    deduplication.

    Returns:
        True if appended, False if skipped.
    """
    path = project_memory_path() if project else global_memory_path()
    _ensure_memory_file(path)

    content = path.read_text()
    lines = content.split("\n")

    if _is_duplicate(text, lines):
        logger.debug("Skipping duplicate: %s", text[:60])
        return False

    non_empty = [
        line
        for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("<!--")
    ]
    if len(non_empty) >= _MAX_MEMORY_LINES:
        logger.debug("Memory full, skipping: %s", text[:60])
        return False

    section = _classify_section(text)
    new_line = f"- {text}"
    inserted = _insert_under_section(lines, section, new_line)

    if not inserted:
        lines.append(f"\n## {section}")
        lines.append(new_line)

    path.write_text("\n".join(lines))
    logger.info("Appended to MEMORY.md: %s", text[:80])
    return True


def create_starter_memory(path: Path) -> None:
    """Create a starter MEMORY.md file.

    Idempotent — does nothing if the file already exists.
    Called by ``memfun init``.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    path.write_text(_STARTER_MEMORY)


# ── Internal helpers ─────────────────────────────────────────


def _ensure_memory_file(path: Path) -> None:
    """Create MEMORY.md with starter content if missing."""
    if path.exists():
        return
    create_starter_memory(path)


def _insert_under_section(
    lines: list[str],
    section: str,
    new_line: str,
) -> bool:
    """Insert *new_line* at the end of ``## {section}``.

    Finds the section header, then scans forward to the end of
    the section (next ``##`` header or end of file) and inserts
    the new line there. This preserves insertion order.

    Modifies *lines* in place.

    Returns:
        True if insertion succeeded.
    """
    for i, line in enumerate(lines):
        if line.strip().startswith(f"## {section}"):
            # Find end of this section
            insert_idx = i + 1
            while insert_idx < len(lines):
                next_line = lines[insert_idx].strip()
                if next_line.startswith("## "):
                    break
                insert_idx += 1
            # Insert before the next section header
            # (or at end of file)
            lines.insert(insert_idx, new_line)
            return True
    return False


_PREFERENCE_WORDS: frozenset[str] = frozenset({
    "prefer", "use", "always", "never", "port",
    "default", "style", "like", "want", "should",
})
_PATTERN_WORDS: frozenset[str] = frozenset({
    "structure", "directory", "naming", "convention",
    "organize", "layout", "component", "module",
})
_TECHNICAL_WORDS: frozenset[str] = frozenset({
    "api", "endpoint", "version", "database", "config",
    "url", "key", "library", "framework", "dependency",
})
_WORKFLOW_WORDS: frozenset[str] = frozenset({
    "test", "deploy", "build", "run", "ci",
    "commit", "review", "process", "step",
})


def _classify_section(text: str) -> str:
    """Classify text into a MEMORY.md section name."""
    words = set(text.lower().split())
    scores: dict[str, Any] = {
        "Preferences": len(words & _PREFERENCE_WORDS),
        "Patterns": len(words & _PATTERN_WORDS),
        "Technical": len(words & _TECHNICAL_WORDS),
        "Workflow": len(words & _WORKFLOW_WORDS),
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "Preferences"


def _is_duplicate(
    new_text: str, existing_lines: list[str]
) -> bool:
    """Check if *new_text* is too similar to any existing line."""
    new_lower = new_text.lower().strip()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        existing = stripped[2:].lower().strip()
        if new_lower == existing:
            return True
        ratio = SequenceMatcher(
            None, new_lower, existing
        ).ratio()
        if ratio >= _SIMILARITY_THRESHOLD:
            return True
    return False
