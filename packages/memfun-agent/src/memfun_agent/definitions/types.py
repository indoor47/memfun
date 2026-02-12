"""Agent definition types for the AGENT.md system."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """A fully parsed agent definition from an AGENT.md file.

    Represents everything needed to register and execute a definition-based
    agent: the metadata from the YAML frontmatter, the instruction body,
    and resolved filesystem paths.
    """

    name: str
    description: str
    version: str = "0.1.0"
    capabilities: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    delegates_to: list[str] = field(default_factory=list)
    model: str | None = None
    max_turns: int = 10
    tags: list[str] = field(default_factory=list)
    instructions: str = ""
    source_path: Path | None = None
    scripts_dir: Path | None = None
    references_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """A snapshot of all discovered agent definitions and their discovery metadata."""

    agents: list[AgentDefinition] = field(default_factory=list)
    discovery_paths: list[Path] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)
