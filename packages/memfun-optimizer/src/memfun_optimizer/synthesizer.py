"""Agent code generation from discovered patterns.

Converts DiscoveredAgent specifications into executable DSPy modules and
AGENT.md definition files. Each discovered pattern becomes a multi-stage
module where each stage corresponds to one step in the strategy pattern.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    # DiscoveredAgent is being created by another agent
    # Define minimal structure here for type checking
    from typing import Protocol

    class StrategyPattern(Protocol):
        pattern_id: str
        step_sequence: list[str]  # StepKind enum values
        frequency: int
        confidence: float
        avg_duration_ms: float
        avg_token_cost: float
        success_rate: float
        example_trace_ids: list[str]

    class DiscoveredAgent(Protocol):
        name: str
        description: str
        task_type: str
        source_pattern: StrategyPattern
        stages: list[dict[str, Any]]
        estimated_speedup: float
        estimated_cost_reduction: float

logger = get_logger("optimizer.synthesizer")

# Validation patterns
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_ALLOWED_STAGE_KINDS = frozenset({
    "PEEK", "GREP", "QUERY", "ANALYZE",
    "TRANSFORM", "VALIDATE", "FORMAT",
})


def _sanitize_for_string_literal(value: str) -> str:
    """Escape a value so it is safe inside a Python string literal or YAML field.

    Prevents code injection via pattern data flowing into generated code
    templates. Escapes backslashes, quotes, newlines, and triple-quote
    sequences.
    """
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace('"""', '\\"\\"\\"')
        .replace("'''", "\\'\\'\\'")
    )


def _sanitize_for_yaml(value: str) -> str:
    """Escape a value for safe inclusion in YAML frontmatter.

    Prevents YAML injection by quoting special characters that could
    break out of the YAML value context.
    """
    # Remove control characters except newline/tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    # If value contains YAML-special characters, it needs quoting
    yaml_special = frozenset(":#{}[],&*?|->!%@`")
    if any(ch in yaml_special for ch in cleaned):
        # Double-quote and escape internal quotes
        cleaned = cleaned.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{cleaned}"'
    return cleaned


def _validate_agent_name(name: str) -> str:
    """Validate and return a safe agent name.

    Raises:
        ValueError: If the name is invalid.
    """
    if not name or len(name) > _MAX_NAME_LENGTH:
        msg = f"Agent name must be 1-{_MAX_NAME_LENGTH} characters, got {len(name) if name else 0}"
        raise ValueError(msg)
    if not _SAFE_IDENTIFIER_RE.match(name):
        msg = f"Agent name contains invalid characters: {name!r}"
        raise ValueError(msg)
    return name


# Step kind to code mapping
_STEP_TO_METHOD = {
    "PEEK": "file_peek",
    "GREP": "regex_search",
    "QUERY": "llm_query",
    "ANALYZE": "analyze_snippet",
    "TRANSFORM": "transform_data",
    "VALIDATE": "validate_result",
    "FORMAT": "format_output",
}


# ── Data Structures ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SynthesizedAgent:
    """Result of synthesizing an agent from a discovered pattern.

    Attributes:
        name: Agent name (hyphenated identifier).
        task_type: Task type this agent handles (analyze, fix, review, etc.).
        source_code: Python source code implementing a DSPy module.
        agent_md: Complete AGENT.md content with YAML frontmatter.
        discovered_from: The DiscoveredAgent spec this was synthesized from.
    """

    name: str
    task_type: str
    source_code: str
    agent_md: str
    discovered_from: Any  # DiscoveredAgent when types.py exists


# ── Agent Synthesizer ──────────────────────────────────────────


class AgentSynthesizer:
    """Converts DiscoveredAgent specifications into executable code.

    Takes a pattern-based agent specification and generates:
    1. A DSPy module with multi-stage forward pass
    2. An AGENT.md definition file for deployment

    Usage::

        synthesizer = AgentSynthesizer()
        result = synthesizer.synthesize(discovered_agent)

        # Deploy the synthesized agent
        agent_dir.mkdir(parents=True)
        (agent_dir / "module.py").write_text(result.source_code)
        (agent_dir / "AGENT.md").write_text(result.agent_md)
    """

    def __init__(self) -> None:
        """Initialize the synthesizer."""
        pass

    def synthesize(self, agent: Any) -> SynthesizedAgent:
        """Convert a DiscoveredAgent into executable code and definition.

        Args:
            agent: A DiscoveredAgent specification with strategy pattern.

        Returns:
            SynthesizedAgent with generated source code and AGENT.md content.

        Raises:
            ValueError: If the agent specification is incomplete or invalid.
        """
        if not agent.name or not agent.task_type:
            msg = "Agent must have name and task_type"
            raise ValueError(msg)

        # Validate agent name to prevent injection into class names / filenames
        _validate_agent_name(agent.name)

        if not agent.stages:
            msg = f"Agent {agent.name} has no stages to synthesize"
            raise ValueError(msg)

        logger.info(
            "Synthesizing agent %s (%d stages, speedup=%.1fx)",
            agent.name,
            len(agent.stages),
            agent.estimated_speedup,
        )

        # Generate the DSPy module source code
        source_code = self._generate_module_source(agent)

        # Generate the AGENT.md definition
        agent_md = self._generate_agent_md(agent)

        return SynthesizedAgent(
            name=agent.name,
            task_type=agent.task_type,
            source_code=source_code,
            agent_md=agent_md,
            discovered_from=agent,
        )

    # ── Code generation ────────────────────────────────────────

    def _generate_module_source(self, agent: Any) -> str:
        """Generate Python source code for a DSPy module.

        Each stage in the agent's strategy becomes a method call in the
        forward pass. The module uses dspy.ChainOfThought for reasoning
        stages and custom tool calls for execution stages.

        Args:
            agent: DiscoveredAgent specification.

        Returns:
            Complete Python module source code.
        """
        stages_code = "\n\n".join(
            self._stage_to_code(stage, idx)
            for idx, stage in enumerate(agent.stages, start=1)
        )

        forward_calls = "\n        ".join(
            f"stage_{idx}_out = self._stage_{idx}(context, query, prev_output=stage_{idx-1}_out)"
            if idx > 1
            else f"stage_{idx}_out = self._stage_{idx}(context, query)"
            for idx in range(1, len(agent.stages) + 1)
        )

        final_stage_var = f"stage_{len(agent.stages)}_out"

        # Sanitize all values that will be interpolated into Python source
        safe_name = _sanitize_for_string_literal(agent.name)
        safe_task_type = _sanitize_for_string_literal(agent.task_type)
        safe_pattern_id = _sanitize_for_string_literal(agent.source_pattern.pattern_id)
        safe_description = _sanitize_for_string_literal(agent.description)
        class_name = self._class_name(agent.name)

        module_source = f'''"""
Auto-generated DSPy module for {safe_name}.

Task type: {safe_task_type}
Generated from pattern: {safe_pattern_id}
Estimated speedup: {agent.estimated_speedup:.1f}x
Estimated cost reduction: {agent.estimated_cost_reduction:.1%}

DO NOT EDIT: This file was synthesized by AgentSynthesizer.
To modify, update the source pattern and re-synthesize.
"""
from __future__ import annotations

from typing import Any

import dspy


class {class_name}(dspy.Module):
    """DSPy module implementing {safe_description}

    Multi-stage pipeline with {len(agent.stages)} stages optimized from
    execution traces. Each stage corresponds to a step in the discovered
    pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize predictors for reasoning stages
        self._setup_predictors()

    def _setup_predictors(self) -> None:
        """Set up DSPy predictors for reasoning stages."""
        # ChainOfThought predictors will be configured here
        # Can be overridden after instantiation for fine-tuning
        pass

    def forward(self, context: str, query: str) -> dspy.Prediction:
        """Execute the multi-stage pipeline.

        Args:
            context: Input context (code, documents, etc.).
            query: User query or task description.

        Returns:
            dspy.Prediction with the final answer.
        """
        # Execute stages in sequence
        {forward_calls}

        # Return final result
        return dspy.Prediction(
            answer={final_stage_var}.get("answer", ""),
            reasoning={final_stage_var}.get("reasoning", ""),
            metadata={{
                "num_stages": {len(agent.stages)},
                "pattern_id": "{safe_pattern_id}",
            }},
        )

{stages_code}
'''

        return textwrap.dedent(module_source).strip() + "\n"

    def _stage_to_code(self, stage: dict[str, Any], index: int) -> str:
        """Convert a single stage dict to a Python method.

        Args:
            stage: Stage specification with kind, description, etc.
            index: Stage number (1-indexed).

        Returns:
            Python method source code.
        """
        stage_kind = stage.get("kind", "QUERY")
        description = stage.get("description", f"Stage {index}")

        # Sanitize values that will be interpolated into generated code
        description = _sanitize_for_string_literal(description)
        # Restrict stage_kind to known values to prevent injection
        if stage_kind not in _ALLOWED_STAGE_KINDS:
            stage_kind = "QUERY"

        # Map stage kind to implementation
        stage_templates = {
            "QUERY": self._generate_query_stage,
            "GREP": self._generate_grep_stage,
            "PEEK": self._generate_peek_stage,
        }

        generator = stage_templates.get(stage_kind)
        if generator:
            return generator(index, description)

        # Default: generic processing stage
        return self._generate_generic_stage(index, description, stage_kind)

    def _generate_query_stage(self, index: int, description: str) -> str:
        """Generate a QUERY stage (DSPy ChainOfThought)."""
        return f'''    def _stage_{index}(
        self,
        context: str,
        query: str,
        prev_output: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """{description}"""
        # Build prompt incorporating previous stage output
        enhanced_query = query
        if prev_output:
            enhanced_query = f"{{query}}\\n\\nPrevious: {{prev_output.get('answer', '')}}"

        # Use ChainOfThought for reasoning
        predictor = dspy.ChainOfThought("context, query -> answer, reasoning")
        result = predictor(context=context, query=enhanced_query)

        return {{
            "answer": str(result.answer),
            "reasoning": str(result.reasoning),
            "stage": {index},
        }}'''

    def _generate_grep_stage(self, index: int, description: str) -> str:
        """Generate a GREP stage (pattern matching)."""
        return f'''    def _stage_{index}(
        self,
        context: str,
        query: str,
        prev_output: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """{description}"""
        import re

        # Extract search pattern from query or previous output
        pattern = prev_output.get("pattern", r"\\w+") if prev_output else r"\\w+"

        matches = re.findall(pattern, context, re.MULTILINE)

        return {{
            "answer": f"Found {{len(matches)}} matches",
            "matches": matches[:10],  # Return first 10
            "stage": {index},
        }}'''

    def _generate_peek_stage(self, index: int, description: str) -> str:
        """Generate a PEEK stage (file reading/preview)."""
        return f'''    def _stage_{index}(
        self,
        context: str,
        query: str,
        prev_output: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """{description}"""
        # Peek at the context structure
        preview_len = 500
        preview = context[:preview_len]

        return {{
            "answer": f"Context preview ({{len(context)}} chars total)",
            "preview": preview,
            "length": len(context),
            "stage": {index},
        }}'''

    def _generate_generic_stage(
        self, index: int, description: str, stage_kind: str
    ) -> str:
        """Generate a generic processing stage."""
        return f'''    def _stage_{index}(
        self,
        context: str,
        query: str,
        prev_output: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """{description}"""
        # Generic processing stage ({stage_kind})
        result = prev_output.get("answer", "") if prev_output else context[:200]

        return {{
            "answer": result,
            "stage": {index},
            "kind": "{stage_kind}",
        }}'''

    def _generate_agent_md(self, agent: Any) -> str:
        """Generate AGENT.md content with YAML frontmatter.

        Args:
            agent: DiscoveredAgent specification.

        Returns:
            Complete AGENT.md file content.
        """
        capabilities = [agent.task_type, "pattern-optimized", "synthesized"]

        # Build tool list based on stages
        tools_needed = set()
        for stage in agent.stages:
            kind = stage.get("kind", "")
            if kind == "GREP":
                tools_needed.add("regex")
            elif kind == "PEEK":
                tools_needed.add("file-read")
            elif kind == "QUERY":
                tools_needed.add("llm")

        allowed_tools = sorted(tools_needed) if tools_needed else ["llm"]

        # Sanitize all values for safe YAML interpolation
        safe_name = _sanitize_for_yaml(agent.name)
        safe_description = _sanitize_for_yaml(agent.description)
        safe_task_type = _sanitize_for_yaml(agent.task_type)

        frontmatter = f"""---
name: {safe_name}
version: 0.1.0
description: {safe_description}
capabilities:
{self._yaml_list(capabilities)}
model: claude-sonnet-4.5
allowed-tools:
{self._yaml_list(allowed_tools)}
max-turns: 5
tags:
  - synthesized
  - pattern-based
  - {safe_task_type}
---"""

        instructions = f"""
# {agent.name.replace('-', ' ').title()}

You are a specialized agent synthesized from observed execution patterns.

## Task

Handle **{agent.task_type}** tasks using a {len(agent.stages)}-stage optimized pipeline.

## Performance Profile

- **Estimated speedup**: {agent.estimated_speedup:.1f}x over baseline
- **Estimated cost reduction**: {agent.estimated_cost_reduction:.1%}
- **Source pattern**: {agent.source_pattern.pattern_id}
- **Success rate**: {agent.source_pattern.success_rate:.1%} (from \
{agent.source_pattern.frequency} traces)

## Strategy

This agent follows a proven pattern extracted from successful task executions:

{self._format_stages(agent.stages)}

## Execution

1. Receive task with context and query
2. Execute stages in sequence, passing output forward
3. Return final result with reasoning trace

## Quality Assurance

- Each stage validates its output before proceeding
- Failed stages trigger fallback to generic reasoning
- Performance metrics are logged for continuous improvement

## When to Use

Invoke this agent for **{agent.task_type}** tasks matching the pattern profile:
- Context size: typical for this pattern
- Query complexity: handled by the {len(agent.stages)}-stage pipeline
- Performance requirements: speed and cost-optimized

For tasks outside this profile, delegate to the general-purpose RLM agent.
"""

        return frontmatter + "\n" + textwrap.dedent(instructions).strip() + "\n"

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _class_name(agent_name: str) -> str:
        """Convert hyphenated agent name to PascalCase class name.

        Args:
            agent_name: Hyphenated name like 'code-analyzer'.

        Returns:
            PascalCase class name like 'CodeAnalyzer'.
        """
        return "".join(
            word.capitalize() for word in agent_name.split("-")
        )

    @staticmethod
    def _yaml_list(items: list[str], indent: int = 2) -> str:
        """Format a list as YAML array items.

        Args:
            items: List of strings.
            indent: Indentation spaces.

        Returns:
            YAML-formatted list.
        """
        prefix = " " * indent
        return "\n".join(f"{prefix}- {_sanitize_for_yaml(item)}" for item in items)

    @staticmethod
    def _format_stages(stages: list[dict[str, Any]]) -> str:
        """Format stage descriptions as numbered list.

        Args:
            stages: List of stage dictionaries.

        Returns:
            Markdown-formatted numbered list.
        """
        lines = []
        for idx, stage in enumerate(stages, start=1):
            kind = stage.get("kind", "PROCESS")
            desc = stage.get("description", f"Stage {idx}")
            # Sanitize to prevent markdown injection
            safe_kind = re.sub(r"[^a-zA-Z0-9_ -]", "", kind)
            safe_desc = desc.replace("\n", " ").replace("\r", " ")
            lines.append(f"{idx}. **{safe_kind}**: {safe_desc}")
        return "\n".join(lines)
