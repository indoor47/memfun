"""Shared data types for the optimizer trace analysis pipeline.

These types represent the abstraction layers in the optimization process:
- Strategy steps: classified atomic operations in an execution trace
- Strategy signatures: fingerprints of entire execution strategies
- Strategy patterns: recurring multi-trace patterns
- Discovered agents: synthesized agent specifications from patterns
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StepKind(StrEnum):
    """Classification of RLM strategy steps.

    Each step represents a distinct operation category that can be
    recognized in the code field of a TraceStep.
    """

    PEEK = "peek"  # Read/open files
    GREP = "grep"  # Search patterns in text
    SEARCH = "search"  # Find/glob/walk filesystem
    PARTITION = "partition"  # Split data into chunks
    QUERY = "query"  # Single LLM sub-call
    BATCH_QUERY = "batch_query"  # Multiple/batched LLM sub-calls
    AGGREGATE = "aggregate"  # Collect/summarize results
    TRANSFORM = "transform"  # Transform/format data
    SUBMIT = "submit"  # Set final answer
    OTHER = "other"  # Unclassifiable


@dataclass(frozen=True, slots=True)
class StrategyStep:
    """A classified step in an execution strategy.

    Represents one TraceStep after classification by the StrategyExtractor.
    """

    kind: StepKind
    iteration: int
    confidence: float  # 0.0-1.0: classifier confidence
    raw_code: str
    raw_reasoning: str


@dataclass(frozen=True, slots=True)
class StrategySignature:
    """A fingerprint for one trace's execution strategy.

    Captures the sequence of step kinds, success outcome, and resource cost.
    """

    trace_id: str
    task_type: str
    steps: list[StrategyStep]
    total_iterations: int
    success: bool
    duration_ms: float
    token_cost: int  # Total tokens consumed


@dataclass(frozen=True, slots=True)
class StrategyPattern:
    """A recurring pattern across multiple execution traces.

    Identified by clustering similar strategy signatures. Patterns with
    high frequency, confidence, and success rate are candidates for
    agent synthesis.
    """

    pattern_id: str
    step_sequence: list[StepKind]  # Normalized step kind sequence
    frequency: int  # Number of traces exhibiting this pattern
    confidence: float  # Average classifier confidence across steps
    avg_duration_ms: float
    avg_token_cost: float
    success_rate: float  # Fraction of successful traces
    example_trace_ids: list[str]  # Sample traces for inspection


@dataclass(frozen=True, slots=True)
class DiscoveredAgent:
    """Specification for a synthesized agent derived from a pattern.

    This is the output of the discovery pipeline: a concrete agent
    definition that can be instantiated to execute the discovered
    strategy more efficiently.
    """

    name: str
    description: str
    task_type: str
    source_pattern: StrategyPattern
    stages: list[dict]  # Stage definitions (stage name, step kinds, etc.)
    estimated_speedup: float  # Expected speedup vs. baseline RLM
    estimated_cost_reduction: float  # Expected token cost reduction
