"""Discovery pipeline orchestrator.

The DiscoveryPipeline coordinates the full trace analysis workflow:
1. Load traces from TraceCollector
2. Extract strategy signatures
3. Analyze patterns
4. Discover agents from dominant patterns
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_optimizer.patterns import PatternAnalyzer
from memfun_optimizer.signatures import StrategyExtractor
from memfun_optimizer.types import (
    DiscoveredAgent,
    StepKind,
)

if TYPE_CHECKING:
    from memfun_agent.traces import ExecutionTrace, TraceCollector

    from memfun_optimizer.types import StrategyPattern

logger = get_logger("optimizer.discovery")


class DiscoveryPipeline:
    """Orchestrates the full agent discovery pipeline.

    Loads traces, extracts strategy signatures, analyzes patterns,
    and synthesizes agent specifications from dominant patterns.

    Usage::

        pipeline = DiscoveryPipeline(trace_collector, min_traces=3)
        discovered = await pipeline.run(task_type="code_analysis")

        for agent in discovered:
            print(f"Discovered: {agent.name}")
            print(f"  Speedup: {agent.estimated_speedup:.2f}x")
            print(f"  Cost reduction: {agent.estimated_cost_reduction:.1%}")
    """

    def __init__(
        self,
        trace_collector: TraceCollector,
        *,
        min_traces: int = 3,
        min_frequency: int = 2,
        min_confidence: float = 0.5,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize the discovery pipeline.

        Args:
            trace_collector: Source of execution traces
            min_traces: Minimum traces required to run discovery
            min_frequency: Minimum pattern frequency to consider
            min_confidence: Minimum pattern confidence to consider
            similarity_threshold: Similarity threshold for pattern clustering
        """
        self._collector = trace_collector
        self._min_traces = min_traces
        self._min_frequency = min_frequency
        self._min_confidence = min_confidence

        self._extractor = StrategyExtractor()
        self._analyzer = PatternAnalyzer(
            similarity_threshold=similarity_threshold
        )

    async def run(
        self, task_type: str | None = None
    ) -> list[DiscoveredAgent]:
        """Run the full discovery pipeline.

        Args:
            task_type: Optional filter for task type; if None, analyze all traces

        Returns:
            List of discovered agent specifications
        """
        logger.info(
            "Starting discovery pipeline (task_type=%s)", task_type or "all"
        )

        # 1. Load traces
        traces = await self._load_traces(task_type)
        if len(traces) < self._min_traces:
            logger.warning(
                "Insufficient traces for discovery: %d < %d",
                len(traces),
                self._min_traces,
            )
            return []

        # 2. Extract signatures
        signatures = self._extractor.extract_batch(traces)

        # 3. Analyze patterns
        patterns = self._analyzer.analyze(signatures)

        # 4. Find dominant patterns
        dominant = self._analyzer.find_dominant(
            patterns,
            min_frequency=self._min_frequency,
            min_confidence=self._min_confidence,
        )

        if not dominant:
            logger.info("No dominant patterns found")
            return []

        # 5. Rank patterns
        ranked = self._analyzer.rank(dominant)

        # 6. Convert patterns to agent specs
        discovered = [
            self._pattern_to_agent(pattern, task_type or "general")
            for pattern in ranked
        ]

        logger.info("Discovered %d agents", len(discovered))
        return discovered

    # ── Internal Methods ───────────────────────────────────────────

    async def _load_traces(
        self, task_type: str | None
    ) -> list[ExecutionTrace]:
        """Load traces from the collector, optionally filtered by task type.

        Args:
            task_type: Optional task type filter

        Returns:
            List of execution traces
        """
        logger.info("Loading traces (task_type=%s)", task_type or "all")

        trace_ids = await self._collector.list_trace_ids(limit=1000)
        traces: list[ExecutionTrace] = []

        for trace_id in trace_ids:
            trace = await self._collector.load(trace_id)
            if trace is None:
                continue

            # Filter by task type if specified
            if task_type is not None and trace.task_type != task_type:
                continue

            traces.append(trace)

        logger.info("Loaded %d traces", len(traces))
        return traces

    def _pattern_to_agent(
        self, pattern: StrategyPattern, task_type: str
    ) -> DiscoveredAgent:
        """Convert a pattern to a discovered agent specification.

        Args:
            pattern: The strategy pattern to convert
            task_type: The task type for the agent

        Returns:
            A discovered agent spec
        """
        # Generate name from step sequence
        seq_names = [self._step_name(kind) for kind in pattern.step_sequence]
        name = f"{task_type}_{pattern.pattern_id}"

        # Generate description
        description = (
            f"Synthesized agent for {task_type}. "
            f"Strategy: {' → '.join(seq_names)}. "
            f"Observed in {pattern.frequency} traces with "
            f"{pattern.success_rate:.1%} success rate."
        )

        # Break step sequence into logical stages
        stages = self._sequence_to_stages(pattern.step_sequence)

        # Estimate speedup and cost reduction
        # Heuristic: specialized agents are ~2-3x faster and use ~40-60% less tokens
        # based on reduced reasoning overhead and targeted operations
        estimated_speedup = 2.0 + (pattern.success_rate * 1.0)
        estimated_cost_reduction = 0.4 + (pattern.confidence * 0.2)

        return DiscoveredAgent(
            name=name,
            description=description,
            task_type=task_type,
            source_pattern=pattern,
            stages=stages,
            estimated_speedup=estimated_speedup,
            estimated_cost_reduction=estimated_cost_reduction,
        )

    def _step_name(self, kind: StepKind) -> str:
        """Get a human-readable name for a step kind.

        Args:
            kind: The step kind

        Returns:
            Human-readable name
        """
        names = {
            StepKind.PEEK: "Read",
            StepKind.GREP: "Search",
            StepKind.SEARCH: "Find",
            StepKind.PARTITION: "Split",
            StepKind.QUERY: "Query",
            StepKind.BATCH_QUERY: "BatchQuery",
            StepKind.AGGREGATE: "Collect",
            StepKind.TRANSFORM: "Transform",
            StepKind.SUBMIT: "Submit",
            StepKind.OTHER: "Process",
        }
        return names.get(kind, "Unknown")

    def _sequence_to_stages(
        self, sequence: list[StepKind]
    ) -> list[dict]:
        """Convert a step sequence to logical stages.

        Groups consecutive similar steps into stages.

        Args:
            sequence: List of step kinds

        Returns:
            List of stage definitions (dicts with name, steps, etc.)
        """
        if not sequence:
            return []

        stages: list[dict] = []
        current_kind = sequence[0]
        current_steps = [current_kind]

        for kind in sequence[1:]:
            if kind == current_kind:
                current_steps.append(kind)
            else:
                # Finish current stage
                stages.append({
                    "name": self._step_name(current_kind),
                    "steps": current_steps.copy(),
                    "count": len(current_steps),
                })
                # Start new stage
                current_kind = kind
                current_steps = [kind]

        # Add final stage
        stages.append({
            "name": self._step_name(current_kind),
            "steps": current_steps,
            "count": len(current_steps),
        })

        return stages
