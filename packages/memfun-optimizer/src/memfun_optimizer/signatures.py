"""Strategy signature extraction from execution traces.

The StrategyExtractor classifies each TraceStep by analyzing its code field,
producing a StrategySignature that fingerprints the entire execution.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_optimizer.types import (
    StepKind,
    StrategySignature,
    StrategyStep,
)

if TYPE_CHECKING:
    from memfun_agent.traces import ExecutionTrace, TraceStep

logger = get_logger("optimizer.signatures")


# Pattern definitions for step classification
_PATTERNS: dict[StepKind, list[tuple[str, float]]] = {
    StepKind.PEEK: [
        (r"\bopen\s*\(", 0.9),
        (r"\.read\s*\(", 0.9),
        (r"\.read_text\s*\(", 0.9),
        (r"\.read_bytes\s*\(", 0.9),
        (r"\bPath\s*\(.*\)\.read", 0.85),
        (r"\bwith\s+open\s*\(", 0.9),
    ],
    StepKind.GREP: [
        (r"\bre\.findall\s*\(", 0.9),
        (r"\bre\.search\s*\(", 0.9),
        (r"\bre\.match\s*\(", 0.9),
        (r"\bgrep\s+", 0.85),
        (r"\brg\s+", 0.85),
        (r"\.search\s*\(.*pattern", 0.7),
    ],
    StepKind.SEARCH: [
        (r"\bglob\s*\(", 0.9),
        (r"\b\.glob\s*\(", 0.9),
        (r"\bfind\s+", 0.8),
        (r"\bos\.walk\s*\(", 0.9),
        (r"\bPath\s*\(.*\)\.glob", 0.9),
        (r"\brglob\s*\(", 0.9),
    ],
    StepKind.PARTITION: [
        (r"\.split\s*\(", 0.7),
        (r"\.chunk\s*\(", 0.9),
        (r"\bchunks?\s*=", 0.8),
        (r"\.partition\s*\(", 0.9),
        (r"\[\s*\d+:\d+\s*\]", 0.6),  # Slicing
    ],
    StepKind.QUERY: [
        (r"\bllm_query\s*\((?!.*batched)", 0.95),
        (r"\bquery\s*\((?!.*batch)", 0.7),
    ],
    StepKind.BATCH_QUERY: [
        (r"\bllm_query_batched\s*\(", 0.95),
        (r"\bllm_query\s*\(.*\bfor\b", 0.8),  # Loop with llm_query
        (r"\b(map|parallel).*llm_query", 0.85),
    ],
    StepKind.AGGREGATE: [
        (r"\.append\s*\(", 0.6),
        (r"\.extend\s*\(", 0.6),
        (r"\.join\s*\(", 0.7),
        (r"\bsum\s*\(", 0.65),
        (r"\bcollect", 0.7),
        (r"\baggregate", 0.85),
        (r"\bsummar(y|ize)", 0.8),
    ],
    StepKind.TRANSFORM: [
        (r"\.format\s*\(", 0.6),
        (r"\.replace\s*\(", 0.6),
        (r"\.strip\s*\(", 0.5),
        (r"\bjson\.dumps\s*\(", 0.75),
        (r"\bjson\.loads\s*\(", 0.75),
        (r"\btransform", 0.85),
        (r"\bmap\s*\(", 0.6),
    ],
    StepKind.SUBMIT: [
        (r"state\s*\[\s*['\"]FINAL['\"]", 0.95),
        (r"state\s*\[\s*['\"]FINAL_VAR['\"]", 0.95),
        (r"\breturn\s+.*final", 0.7),
    ],
}


class StrategyExtractor:
    """Extracts strategy signatures from execution traces.

    Classifies each TraceStep by analyzing its code field against
    regex patterns, producing a StrategySignature that fingerprints
    the execution strategy.

    Usage::

        extractor = StrategyExtractor()
        signature = extractor.extract(trace)
        batch_sigs = extractor.extract_batch(traces)
    """

    def __init__(self) -> None:
        self._compiled_patterns: dict[
            StepKind, list[tuple[re.Pattern[str], float]]
        ] = {
            kind: [(re.compile(pat, re.IGNORECASE), conf) for pat, conf in pats]
            for kind, pats in _PATTERNS.items()
        }

    def extract(self, trace: ExecutionTrace) -> StrategySignature:
        """Extract a strategy signature from a single trace.

        Args:
            trace: The execution trace to analyze

        Returns:
            A strategy signature fingerprinting the trace's strategy
        """
        steps = [self.classify_step(ts) for ts in trace.trajectory]

        return StrategySignature(
            trace_id=trace.trace_id,
            task_type=trace.task_type,
            steps=steps,
            total_iterations=len(trace.trajectory),
            success=trace.success,
            duration_ms=trace.duration_ms,
            token_cost=trace.token_usage.total_tokens,
        )

    def extract_batch(
        self, traces: list[ExecutionTrace]
    ) -> list[StrategySignature]:
        """Extract strategy signatures from multiple traces.

        Args:
            traces: List of execution traces to analyze

        Returns:
            List of strategy signatures, one per trace
        """
        logger.info("Extracting signatures from %d traces", len(traces))
        signatures = [self.extract(trace) for trace in traces]
        logger.info("Extracted %d signatures", len(signatures))
        return signatures

    def classify_step(self, step: TraceStep) -> StrategyStep:
        """Classify a single trace step by analyzing its code.

        Args:
            step: The trace step to classify

        Returns:
            A classified strategy step with kind and confidence
        """
        code = step.code
        best_kind = StepKind.OTHER
        best_confidence = 0.0

        # Try to match against all patterns
        for kind, patterns in self._compiled_patterns.items():
            for pattern, base_confidence in patterns:
                if pattern.search(code) and base_confidence > best_confidence:
                    best_kind = kind
                    best_confidence = base_confidence

        # If multiple patterns match, average their confidences
        # (current implementation uses max; could be enhanced)

        return StrategyStep(
            kind=best_kind,
            iteration=step.iteration,
            confidence=best_confidence if best_confidence > 0 else 0.1,
            raw_code=step.code,
            raw_reasoning=step.reasoning,
        )
