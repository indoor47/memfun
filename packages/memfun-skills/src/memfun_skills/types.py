"""Skill types for the memfun-skills package."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """A fully parsed skill definition from a SKILL.md file.

    Represents everything needed to register and execute a skill:
    the metadata from the YAML frontmatter, the instruction body,
    and resolved filesystem paths.
    """

    name: str
    description: str
    version: str = "0.1.0"
    user_invocable: bool = True
    model_invocable: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    instructions: str = ""
    source_path: Path = field(default_factory=lambda: Path("."))
    scripts_dir: Path | None = None
    references_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """A snapshot of all discovered skills and their discovery metadata."""

    skills: list[SkillDefinition] = field(default_factory=list)
    discovery_paths: list[Path] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SkillExecutionContext:
    """Context passed to a skill when it is activated/executed."""

    skill: SkillDefinition
    arguments: dict[str, str] = field(default_factory=dict)
    working_dir: Path = field(default_factory=lambda: Path.cwd())
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActivatedSkill:
    """A skill that has been resolved and is ready for execution.

    Contains the original definition plus all resolved runtime
    artifacts: reference file contents, mapped MCP tool names,
    and the fully assembled instruction text.
    """

    skill: SkillDefinition
    resolved_instructions: str
    mapped_tools: list[str] = field(default_factory=list)
    reference_contents: dict[str, str] = field(default_factory=dict)
    working_dir: Path = field(default_factory=lambda: Path.cwd())


@dataclass(frozen=True, slots=True)
class SkillResult:
    """Result of executing a skill (or a dry-run prompt assembly).

    Captures the output text, any tool calls that were made (or
    would be made), wall-clock duration, and success/error status.
    """

    output: str
    tool_calls_made: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
