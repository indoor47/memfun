"""Skill synthesis from execution traces.

Analyzes patterns in agent execution traces to identify recurring
strategies and automatically generate new SKILL.md files that
codify those strategies as reusable skills.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from memfun_agent.traces import ExecutionTrace

logger = get_logger("optimizer.skill_synthesis")

# Skill name must be lowercase alphanumeric with hyphens (matching validator.py)
_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_SKILL_NAME_MAX_LENGTH = 64


def _sanitize_yaml_value(value: str) -> str:
    """Escape a string for safe inclusion in YAML frontmatter.

    Prevents YAML injection by removing control characters and quoting
    values that contain YAML-special characters.
    """
    # Strip control characters except newline/tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    # Quote values containing YAML metacharacters
    yaml_special = frozenset(":#{}[],&*?|->!%@`")
    if any(ch in yaml_special for ch in cleaned):
        cleaned = cleaned.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{cleaned}"'
    return cleaned


def _sanitize_skill_name(name: str) -> str:
    """Sanitize a skill name: lowercase, strip unsafe chars, validate.

    Prevents path traversal by stripping path separators, dots, and
    other unsafe characters. Returns a valid skill name.
    """
    # Strip path separators and common traversal patterns
    name = name.replace("/", "-").replace("\\", "-").replace("..", "")
    # Lowercase and remove any characters not matching allowed pattern
    name = re.sub(r"[^a-z0-9-]", "", name.lower())
    # Collapse multiple hyphens
    name = re.sub(r"-{2,}", "-", name).strip("-")
    # Truncate to max length
    name = name[:_SKILL_NAME_MAX_LENGTH]
    # Ensure it starts with a letter
    if not name or not name[0].isalpha():
        name = "skill-" + name
    return name[:_SKILL_NAME_MAX_LENGTH]


@dataclass(frozen=True, slots=True)
class SynthesizedSkill:
    """A skill synthesized from execution trace patterns."""

    name: str
    task_type: str
    skill_md_content: str
    source_trace_ids: list[str]
    confidence: float  # 0.0-1.0, based on pattern frequency and consistency


class SkillSynthesizer:
    """Synthesizes new skills from execution trace patterns.

    Analyzes a collection of traces for a specific task type and
    identifies the dominant strategy pattern. If a consistent pattern
    is found, generates a SKILL.md file that codifies the strategy.

    Example usage::

        synthesizer = SkillSynthesizer(min_pattern_frequency=3)
        traces = [trace1, trace2, trace3, ...]
        skill = synthesizer.synthesize(traces, task_type="code-review")
        if skill:
            print(skill.skill_md_content)
    """

    def __init__(self, *, min_pattern_frequency: int = 3) -> None:
        """Initialize the skill synthesizer.

        Args:
            min_pattern_frequency: Minimum number of traces exhibiting
                the same pattern to consider it for synthesis.
        """
        self._min_pattern_frequency = min_pattern_frequency

    def synthesize(
        self,
        traces: list[ExecutionTrace],
        task_type: str,
    ) -> SynthesizedSkill | None:
        """Synthesize a skill from traces for a given task type.

        Args:
            traces: List of execution traces to analyze.
            task_type: The type of task these traces represent.

        Returns:
            A SynthesizedSkill if a consistent pattern is found,
            otherwise None.
        """
        if len(traces) < self._min_pattern_frequency:
            logger.info(
                "Insufficient traces for synthesis: %d < %d",
                len(traces),
                self._min_pattern_frequency,
            )
            return None

        # Extract metadata from traces
        metadata = self._extract_skill_metadata(traces)

        # Determine if there's a dominant pattern
        if not self._has_dominant_pattern(metadata):
            logger.info("No dominant pattern found in traces for task_type=%s", task_type)
            return None

        # Generate skill name and description
        skill_name = self._generate_skill_name(task_type, metadata)
        description = self._generate_description(task_type, metadata)
        instructions = self._generate_instructions(metadata, traces)

        # Determine allowed tools from trace patterns
        allowed_tools = metadata.get("common_tools", [])
        tags = [task_type, "synthesized", "auto-generated"]

        # Generate SKILL.md content
        skill_md = self._generate_skill_md(
            name=skill_name,
            description=description,
            instructions=instructions,
            version="0.1.0",
            tags=tags,
            allowed_tools=allowed_tools,
        )

        # Calculate confidence based on pattern consistency
        confidence = self._calculate_confidence(metadata, len(traces))

        source_ids = [trace.trace_id for trace in traces]

        logger.info(
            "Synthesized skill '%s' from %d traces (confidence=%.2f)",
            skill_name,
            len(traces),
            confidence,
        )

        return SynthesizedSkill(
            name=skill_name,
            task_type=task_type,
            skill_md_content=skill_md,
            source_trace_ids=source_ids,
            confidence=confidence,
        )

    def _extract_skill_metadata(
        self, traces: list[ExecutionTrace]
    ) -> dict[str, Any]:
        """Extract common patterns from traces.

        Analyzes query patterns, tool usage, trajectory structure,
        and successful strategies.
        """
        # Query pattern analysis
        query_keywords = []
        for trace in traces:
            # Extract keywords from query (simple tokenization)
            words = trace.query.lower().split()
            query_keywords.extend(words)

        keyword_freq = Counter(query_keywords)
        common_keywords = [word for word, count in keyword_freq.most_common(5)]

        # Tool usage analysis
        all_tools: list[str] = []
        for trace in traces:
            # Extract tool patterns from trajectory
            for step in trace.trajectory:
                # Simple heuristic: look for common tool names in reasoning/code
                reasoning = step.reasoning.lower()
                code = step.code.lower()
                combined = reasoning + " " + code

                for tool_name in ["read", "write", "grep", "glob", "bash", "edit"]:
                    if tool_name in combined:
                        all_tools.append(tool_name.capitalize())

        tool_freq = Counter(all_tools)
        common_tools = [tool for tool, count in tool_freq.most_common(10)]

        # Strategy pattern analysis
        avg_steps = sum(len(trace.trajectory) for trace in traces) / len(traces)
        successful_traces = [t for t in traces if t.success]
        success_rate = len(successful_traces) / len(traces) if traces else 0.0

        # Context pattern analysis
        context_lengths = [trace.context_length for trace in traces]
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0

        return {
            "common_keywords": common_keywords,
            "common_tools": common_tools,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "avg_context_length": avg_context_length,
            "total_traces": len(traces),
            "successful_traces": len(successful_traces),
        }

    def _has_dominant_pattern(self, metadata: dict[str, Any]) -> bool:
        """Determine if metadata indicates a consistent, learnable pattern."""
        # Require reasonable success rate
        if metadata["success_rate"] < 0.6:
            return False

        # Require common tools (evidence of strategy)
        if not metadata["common_tools"]:
            return False

        # Require at least some common keywords
        return len(metadata["common_keywords"]) >= 2

    def _generate_skill_name(
        self, task_type: str, metadata: dict[str, Any]
    ) -> str:
        """Generate a descriptive skill name from task type and patterns."""
        # Sanitize task_type for use as skill name
        base_name = task_type.lower().replace(" ", "-").replace("_", "-")

        # Add descriptor based on common keywords if available
        if metadata["common_keywords"]:
            top_keyword = metadata["common_keywords"][0]
            if top_keyword not in base_name:
                base_name = f"{top_keyword}-{base_name}"

        # Sanitize the final name to prevent path traversal and ensure
        # it matches the allowed skill name pattern
        return _sanitize_skill_name(base_name)

    def _generate_description(
        self, task_type: str, metadata: dict[str, Any]
    ) -> str:
        """Generate a description based on task type and patterns."""
        tool_summary = (
            ", ".join(metadata["common_tools"][:3])
            if metadata["common_tools"]
            else "various tools"
        )

        return (
            f"Automatically synthesized skill for {task_type} tasks. "
            f"This skill was generated from {metadata['total_traces']} successful "
            f"execution traces with a {metadata['success_rate']:.0%} success rate. "
            f"Commonly uses: {tool_summary}."
        )

    def _generate_instructions(
        self, metadata: dict[str, Any], traces: list[ExecutionTrace]
    ) -> str:
        """Generate skill instructions based on observed patterns."""
        # Extract a representative successful trace for strategy template
        successful = [t for t in traces if t.success]
        if not successful:
            successful = traces  # fallback to all traces

        # Use the trace with median step count as representative
        successful.sort(key=lambda t: len(t.trajectory))
        representative = successful[len(successful) // 2]

        # Build instructions from the representative trace
        steps_summary = []
        for i, step in enumerate(representative.trajectory, 1):
            # Extract concise reasoning
            reasoning_preview = (
                step.reasoning[:200] + "..."
                if len(step.reasoning) > 200
                else step.reasoning
            )
            steps_summary.append(f"{i}. {reasoning_preview}")

        steps_text = "\n".join(steps_summary)

        instructions = f"""# Synthesized Strategy for Task

This skill was automatically generated from successful execution patterns.

## Observed Strategy

The following approach was extracted from {metadata['total_traces']} successful executions:

{steps_text}

## Key Patterns

- **Common tools**: {', '.join(metadata['common_tools'][:5])}
- **Average steps**: {metadata['avg_steps']:.1f}
- **Success rate**: {metadata['success_rate']:.0%}

## Execution Approach

When invoked, follow these guidelines based on successful patterns:

1. Analyze the user's request to identify the core objective
2. Use the tools listed above in a similar sequence to the observed pattern
3. Iterate until the objective is met or constraints are reached
4. Provide clear output summarizing the results

## Important Notes

- This is a synthesized skill and may require refinement based on real-world usage
- Monitor effectiveness and adjust the strategy as needed
- Report edge cases or failures for further optimization
"""

        return instructions

    def _generate_skill_md(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        version: str = "0.1.0",
        tags: list[str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> str:
        """Generate a valid SKILL.md file from components.

        Produces YAML frontmatter + markdown body matching the
        format used by built-in skills.
        """
        # Sanitize all values for safe YAML embedding
        safe_name = _sanitize_yaml_value(name)
        safe_version = _sanitize_yaml_value(version)

        # Build YAML frontmatter
        frontmatter_lines = [
            "---",
            f"name: {safe_name}",
            "description: >",
        ]

        # Indent multi-line description (sanitize each line)
        for line in description.split("\n"):
            # Strip YAML-breaking characters from folded block content
            safe_line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", line)
            frontmatter_lines.append(f"  {safe_line}")

        frontmatter_lines.append(f"version: {safe_version}")
        frontmatter_lines.append("user-invocable: true")
        frontmatter_lines.append("model-invocable: true")

        if allowed_tools:
            frontmatter_lines.append("allowed-tools:")
            for tool in allowed_tools:
                frontmatter_lines.append(f"  - {_sanitize_yaml_value(tool)}")

        if tags:
            frontmatter_lines.append("tags:")
            for tag in tags:
                frontmatter_lines.append(f"  - {_sanitize_yaml_value(tag)}")

        frontmatter_lines.append("---")

        frontmatter = "\n".join(frontmatter_lines)

        # Combine frontmatter and instructions
        return f"{frontmatter}\n\n{instructions}\n"

    def _calculate_confidence(
        self, metadata: dict[str, Any], trace_count: int
    ) -> float:
        """Calculate confidence score for the synthesized skill.

        Based on:
        - Number of traces
        - Success rate
        - Pattern consistency (common tools, keywords)
        """
        # Base confidence on success rate
        confidence = metadata["success_rate"]

        # Boost confidence for higher trace counts (up to 20 traces)
        trace_factor = min(trace_count / 20.0, 1.0)
        confidence *= (0.5 + 0.5 * trace_factor)

        # Boost confidence for strong tool patterns
        tool_count = len(metadata["common_tools"])
        tool_factor = min(tool_count / 5.0, 1.0)
        confidence *= (0.7 + 0.3 * tool_factor)

        return min(confidence, 1.0)
