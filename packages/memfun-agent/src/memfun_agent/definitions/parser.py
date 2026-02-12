"""AGENT.md parser: extracts YAML frontmatter and markdown instructions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from memfun_core.errors import AgentValidationError

from memfun_agent.definitions.types import AgentDefinition

if TYPE_CHECKING:
    from pathlib import Path


def parse_agent_md(path: Path) -> AgentDefinition:
    """Parse an AGENT.md file into an AgentDefinition.

    The file format is YAML frontmatter delimited by ``---`` lines, followed
    by a markdown body containing the agent instructions.

    Args:
        path: Path to the AGENT.md file.

    Returns:
        A fully populated AgentDefinition.

    Raises:
        AgentValidationError: If the file cannot be parsed or is missing
            required fields (``name``, ``description``).
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        msg = f"AGENT.md not found: {path}"
        raise FileNotFoundError(msg)

    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, path)
    meta = _parse_yaml(frontmatter, path)

    # Validate required fields
    if "name" not in meta or not meta["name"]:
        msg = f"AGENT.md missing required field 'name': {path}"
        raise AgentValidationError(msg)
    if "description" not in meta or not meta["description"]:
        msg = f"AGENT.md missing required field 'description': {path}"
        raise AgentValidationError(msg)

    agent_dir = path.parent

    # Resolve optional directories
    scripts_dir = agent_dir / "scripts" if (agent_dir / "scripts").is_dir() else None
    references_dir = (
        agent_dir / "references" if (agent_dir / "references").is_dir() else None
    )

    return AgentDefinition(
        name=str(meta["name"]),
        description=str(meta["description"]),
        version=str(meta.get("version", "0.1.0")),
        capabilities=_as_str_list(meta.get("capabilities")),
        allowed_tools=_as_str_list(meta.get("allowed-tools")),
        delegates_to=_as_str_list(meta.get("delegates-to")),
        model=str(meta["model"]) if meta.get("model") else None,
        max_turns=int(meta.get("max-turns", 10)),
        tags=_as_str_list(meta.get("tags")),
        instructions=body.strip(),
        source_path=agent_dir,
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
        msg = f"AGENT.md missing YAML frontmatter (no opening '---'): {path}"
        raise AgentValidationError(msg)

    # Find the closing ---
    first_newline = stripped.index("\n")
    rest = stripped[first_newline + 1 :]
    closing_idx = rest.find("\n---")
    if closing_idx == -1:
        msg = f"AGENT.md missing closing '---' for frontmatter: {path}"
        raise AgentValidationError(msg)

    frontmatter = rest[:closing_idx]
    body = rest[closing_idx + 4 :]  # skip past "\n---"
    return frontmatter, body


def _parse_yaml(frontmatter: str, path: Path) -> dict[str, Any]:
    """Parse the YAML frontmatter string using safe_load.

    Returns:
        Parsed dictionary.

    Raises:
        AgentValidationError: If the YAML is malformed.
    """
    try:
        result = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML frontmatter in {path}: {exc}"
        raise AgentValidationError(msg) from exc

    if not isinstance(result, dict):
        msg = f"YAML frontmatter must be a mapping, got {type(result).__name__}: {path}"
        raise AgentValidationError(msg)

    return result


def _as_str_list(value: Any) -> list[str]:
    """Coerce a value to a list of strings, or return empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
