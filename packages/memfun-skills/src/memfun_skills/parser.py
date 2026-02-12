"""SKILL.md parser: extracts YAML frontmatter and markdown instructions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from memfun_core.errors import SkillValidationError

from memfun_skills.types import SkillDefinition

if TYPE_CHECKING:
    from pathlib import Path


def parse_skill_md(path: Path) -> SkillDefinition:
    """Parse a SKILL.md file into a SkillDefinition.

    The file format is YAML frontmatter delimited by ``---`` lines, followed
    by a markdown body containing the skill instructions.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        A fully populated SkillDefinition.

    Raises:
        SkillValidationError: If the file cannot be parsed or is missing
            required fields (``name``, ``description``).
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        msg = f"SKILL.md not found: {path}"
        raise FileNotFoundError(msg)

    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, path)
    meta = _parse_yaml(frontmatter, path)

    # Validate required fields
    if "name" not in meta or not meta["name"]:
        msg = f"SKILL.md missing required field 'name': {path}"
        raise SkillValidationError(msg)
    if "description" not in meta or not meta["description"]:
        msg = f"SKILL.md missing required field 'description': {path}"
        raise SkillValidationError(msg)

    skill_dir = path.parent

    # Resolve optional directories
    scripts_dir = skill_dir / "scripts" if (skill_dir / "scripts").is_dir() else None
    references_dir = (
        skill_dir / "references" if (skill_dir / "references").is_dir() else None
    )

    return SkillDefinition(
        name=str(meta["name"]),
        description=str(meta["description"]),
        version=str(meta.get("version", "0.1.0")),
        user_invocable=bool(meta.get("user-invocable", True)),
        model_invocable=bool(meta.get("model-invocable", True)),
        allowed_tools=_as_str_list(meta.get("allowed-tools")),
        tags=_as_str_list(meta.get("tags")),
        instructions=body.strip(),
        source_path=skill_dir,
        scripts_dir=scripts_dir,
        references_dir=references_dir,
    )


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """Split text into YAML frontmatter and markdown body.

    The frontmatter is expected to be enclosed between two ``---`` lines
    at the start of the file.

    Returns:
        A (frontmatter, body) tuple.
    """
    stripped = text.lstrip("\n")
    if not stripped.startswith("---"):
        msg = f"SKILL.md missing YAML frontmatter (no opening '---'): {path}"
        raise SkillValidationError(msg)

    # Find the closing ---
    first_newline = stripped.index("\n")
    rest = stripped[first_newline + 1 :]
    closing_idx = rest.find("\n---")
    if closing_idx == -1:
        msg = f"SKILL.md missing closing '---' for frontmatter: {path}"
        raise SkillValidationError(msg)

    frontmatter = rest[:closing_idx]
    body = rest[closing_idx + 4 :]  # skip past "\n---"
    return frontmatter, body


def _parse_yaml(frontmatter: str, path: Path) -> dict[str, Any]:
    """Parse the YAML frontmatter string using safe_load.

    Returns:
        Parsed dictionary.

    Raises:
        SkillValidationError: If the YAML is malformed.
    """
    try:
        result = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML frontmatter in {path}: {exc}"
        raise SkillValidationError(msg) from exc

    if not isinstance(result, dict):
        msg = f"YAML frontmatter must be a mapping, got {type(result).__name__}: {path}"
        raise SkillValidationError(msg)

    return result


def _as_str_list(value: Any) -> list[str]:
    """Coerce a value to a list of strings, or return empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
