"""Skill executor: assembles prompts and delegates to the LLM layer."""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from memfun_skills.types import SkillResult

if TYPE_CHECKING:
    from memfun_skills.types import ActivatedSkill

logger = logging.getLogger("memfun.skills.executor")

# Type alias for the optional LLM callback.
# The callback receives a prompt string and returns the LLM output string.
LLMCallback = Callable[[str], Awaitable[str]]


class SkillExecutor:
    """Executes an activated skill by building a prompt and delegating.

    The executor is responsible for:
    1. Assembling a final prompt from the activated skill and the task.
    2. Delegating to an LLM callback (if provided) or returning a
       dry-run result with the assembled prompt.
    3. Measuring execution duration and capturing the result.

    When no ``llm_callback`` is provided, the executor operates in
    **dry-run mode**: it builds the full prompt but does not invoke
    any model, returning the prompt text as the output.
    """

    def __init__(
        self,
        llm_callback: LLMCallback | None = None,
    ) -> None:
        self._llm_callback = llm_callback

    async def execute(
        self,
        activated: ActivatedSkill,
        task: dict[str, Any],
    ) -> SkillResult:
        """Execute a skill against a task payload.

        Args:
            activated: The resolved, ready-to-execute skill.
            task: A dict describing the task.  Expected keys:
                - ``"query"`` or ``"input"``: The user's request text.
                - Any additional keys are included as task context.

        Returns:
            A SkillResult with the output (or the assembled prompt in
            dry-run mode), duration, and success status.
        """
        skill_name = activated.skill.name
        logger.info("Executing skill: %s", skill_name)

        start = time.monotonic()

        try:
            prompt = self._build_prompt(activated, task)

            if self._llm_callback is not None:
                logger.debug(
                    "Invoking LLM callback for skill %s", skill_name
                )
                output = await self._llm_callback(prompt)
            else:
                logger.debug(
                    "Dry-run mode for skill %s (no LLM callback)",
                    skill_name,
                )
                output = prompt

            elapsed_ms = (time.monotonic() - start) * 1000.0

            return SkillResult(
                output=output,
                tool_calls_made=list(activated.mapped_tools),
                duration_ms=elapsed_ms,
                success=True,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.error(
                "Skill %s execution failed: %s",
                skill_name,
                exc,
                exc_info=True,
            )
            return SkillResult(
                output="",
                tool_calls_made=[],
                duration_ms=elapsed_ms,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _build_prompt(
        activated: ActivatedSkill,
        task: dict[str, Any],
    ) -> str:
        """Build the full prompt from activated skill and task.

        The prompt structure is:

        1. System preamble (skill instructions with resolved refs)
        2. Available tools listing
        3. Task payload

        Returns:
            The assembled prompt string.
        """
        sections: list[str] = []

        # Section 1: Skill instructions (already resolved by activator)
        sections.append(activated.resolved_instructions)

        # Section 2: Available tools
        if activated.mapped_tools:
            sections.append("\n## Available Tools\n")
            for tool in activated.mapped_tools:
                sections.append(f"- {tool}")
            sections.append("")

        # Section 3: Task
        sections.append("\n## Task\n")
        task_input = task.get("query") or task.get("input") or ""
        if task_input:
            sections.append(str(task_input))

        # Include additional task context (excluding query/input)
        extra_keys = {k: v for k, v in task.items() if k not in ("query", "input")}
        if extra_keys:
            sections.append("\n### Additional Context\n")
            for key, value in extra_keys.items():
                sections.append(f"- **{key}**: {value}")

        return "\n".join(sections)
