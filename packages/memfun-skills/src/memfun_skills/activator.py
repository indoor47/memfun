"""Skill activation: resolves references, maps tools, and prepares execution."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memfun_skills.tool_mapping import ToolNameMapper
from memfun_skills.types import ActivatedSkill

if TYPE_CHECKING:
    from memfun_skills.types import SkillDefinition, SkillExecutionContext

logger = logging.getLogger("memfun.skills.activator")


class SkillActivator:
    """Prepares a skill definition for execution.

    Activation resolves all runtime artifacts:
    - Reads reference files from the skill's ``references/`` directory.
    - Maps skill-level tool names to MCP gateway tool names.
    - Assembles the final working directory and instruction text.
    """

    def __init__(
        self,
        tool_mapper: ToolNameMapper | None = None,
    ) -> None:
        self._tool_mapper = tool_mapper or ToolNameMapper()

    async def activate(
        self,
        skill: SkillDefinition,
        context: SkillExecutionContext,
    ) -> ActivatedSkill:
        """Activate a skill by resolving all its runtime dependencies.

        Args:
            skill: The parsed skill definition.
            context: Execution context (working directory, arguments, etc.).

        Returns:
            An ActivatedSkill ready for prompt assembly and execution.
        """
        logger.info("Activating skill: %s", skill.name)

        # 1. Resolve reference files
        reference_contents = self._load_references(skill)

        # 2. Map allowed tools to MCP gateway names
        mapped_tools = self._tool_mapper.map_tools(skill.allowed_tools)
        logger.debug(
            "Mapped tools for %s: %s -> %s",
            skill.name,
            skill.allowed_tools,
            mapped_tools,
        )

        # 3. Assemble resolved instructions
        resolved_instructions = self._build_instructions(
            skill, reference_contents, context
        )

        activated = ActivatedSkill(
            skill=skill,
            resolved_instructions=resolved_instructions,
            mapped_tools=mapped_tools,
            reference_contents=reference_contents,
            working_dir=context.working_dir,
        )

        logger.info(
            "Skill %s activated (tools=%d, refs=%d)",
            skill.name,
            len(mapped_tools),
            len(reference_contents),
        )
        return activated

    @staticmethod
    def _load_references(skill: SkillDefinition) -> dict[str, str]:
        """Read all files in the skill's references/ directory.

        Returns:
            A dict mapping relative filename to file content.
            Empty dict if no references directory exists.
        """
        if skill.references_dir is None or not skill.references_dir.is_dir():
            return {}

        contents: dict[str, str] = {}
        for ref_file in sorted(skill.references_dir.iterdir()):
            if not ref_file.is_file():
                continue
            try:
                text = ref_file.read_text(encoding="utf-8")
                contents[ref_file.name] = text
                logger.debug(
                    "Loaded reference %s (%d chars)",
                    ref_file.name,
                    len(text),
                )
            except Exception:
                logger.warning(
                    "Failed to read reference file: %s",
                    ref_file,
                    exc_info=True,
                )
        return contents

    @staticmethod
    def _build_instructions(
        skill: SkillDefinition,
        reference_contents: dict[str, str],
        context: SkillExecutionContext,
    ) -> str:
        """Assemble the final instruction text from skill parts.

        The instruction string is structured as:

        1. Skill header (name, version, description)
        2. Reference content blocks (if any)
        3. Main instructions body
        4. Context information (working directory, arguments)

        Returns:
            The fully assembled instruction string.
        """
        parts: list[str] = []

        # Header
        parts.append(f"# Skill: {skill.name} (v{skill.version})")
        parts.append(f"\n{skill.description}\n")

        # Reference content
        if reference_contents:
            parts.append("## Reference Material\n")
            for filename, content in reference_contents.items():
                parts.append(f"### {filename}\n")
                parts.append(content.strip())
                parts.append("")  # blank line separator

        # Main instructions
        if skill.instructions:
            parts.append("## Instructions\n")
            parts.append(skill.instructions)
            parts.append("")

        # Context
        parts.append("## Context\n")
        parts.append(f"Working directory: {context.working_dir}")
        if context.arguments:
            parts.append("\nArguments:")
            for key, value in context.arguments.items():
                parts.append(f"  {key}: {value}")

        return "\n".join(parts)
