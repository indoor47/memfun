"""A/B testing framework for comparing agent performance.

Provides tools to compare original agents against synthesized versions,
measuring accuracy, latency, token cost, and overall quality. Uses
statistical analysis to make promotion recommendations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("optimizer.comparison")


# ── Data Structures ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Result from running an agent on a single task.

    Attributes:
        task_id: Identifier for the task.
        answer: Agent's answer/output.
        duration_ms: Execution time in milliseconds.
        token_cost: Estimated token cost (arbitrary units).
        success: Whether the run completed successfully.
        error: Error message if success is False.
    """

    task_id: str
    answer: str
    duration_ms: float
    token_cost: float
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Statistical comparison of two agents.

    Attributes:
        original_metrics: Aggregated metrics for the original agent.
        synthesized_metrics: Aggregated metrics for the synthesized agent.
        speedup: Average speedup (original_time / synthesized_time).
        cost_reduction: Fractional cost reduction (0.0-1.0).
        accuracy_delta: Difference in accuracy (synthesized - original).
        recommendation: Action recommendation: "promote" | "needs-more-data" | "no-improvement".
    """

    original_metrics: dict[str, float]
    synthesized_metrics: dict[str, float]
    speedup: float
    cost_reduction: float
    accuracy_delta: float
    recommendation: str


# ── Comparison Runner ──────────────────────────────────────────


class ComparisonRunner:
    """A/B testing framework for agent comparison.

    Runs both agents on the same set of test tasks and collects
    performance metrics. Provides statistical comparison and
    recommendations for deployment.

    Usage::

        runner = ComparisonRunner(num_runs=5)
        report = await runner.compare(
            original_agent,
            synthesized_agent,
            test_tasks=[
                {"task_id": "1", "query": "...", "context": "..."},
                {"task_id": "2", "query": "...", "context": "..."},
            ]
        )

        print(f"Speedup: {report.speedup:.1f}x")
        print(f"Cost reduction: {report.cost_reduction:.1%}")
        print(f"Recommendation: {report.recommendation}")

        if report.recommendation == "promote":
            deploy_agent(synthesized_agent)
    """

    _MAX_NUM_RUNS = 100
    _MAX_TEST_TASKS = 1000

    def __init__(self, *, num_runs: int = 5) -> None:
        """Initialize the comparison runner.

        Args:
            num_runs: Number of times to run each task (for averaging).
                Capped at 100 to prevent resource exhaustion.
        """
        self._num_runs = min(num_runs, self._MAX_NUM_RUNS)

    async def compare(
        self,
        original_agent: Any,
        synthesized_agent: Any,
        test_tasks: list[dict[str, Any]],
    ) -> ComparisonReport:
        """Compare two agents on the same test tasks.

        Args:
            original_agent: The baseline agent.
            synthesized_agent: The optimized/synthesized agent.
            test_tasks: List of task dicts with task_id, query, context.

        Returns:
            ComparisonReport with detailed metrics and recommendation.

        Raises:
            ValueError: If test_tasks is empty.
        """
        if not test_tasks:
            msg = "test_tasks cannot be empty"
            raise ValueError(msg)

        if len(test_tasks) > self._MAX_TEST_TASKS:
            msg = f"test_tasks exceeds maximum ({len(test_tasks)} > {self._MAX_TEST_TASKS})"
            raise ValueError(msg)

        logger.info(
            "Comparing agents on %d tasks (%d runs each)",
            len(test_tasks),
            self._num_runs,
        )

        # Run both agents on all tasks
        original_results = await self._run_all_tasks(
            original_agent, test_tasks, "original"
        )
        synthesized_results = await self._run_all_tasks(
            synthesized_agent, test_tasks, "synthesized"
        )

        # Compute aggregated metrics
        original_metrics = self._compute_metrics(original_results)
        synthesized_metrics = self._compute_metrics(
            synthesized_results
        )

        # Calculate comparison statistics
        speedup = self._calculate_speedup(
            original_metrics, synthesized_metrics
        )
        cost_reduction = self._calculate_cost_reduction(
            original_metrics, synthesized_metrics
        )
        accuracy_delta = self._calculate_accuracy_delta(
            original_metrics, synthesized_metrics
        )

        # Make recommendation
        recommendation = self._make_recommendation(
            speedup, cost_reduction, accuracy_delta
        )

        logger.info(
            "Comparison complete: speedup=%.1fx, cost_reduction=%.1f%%, "
            "accuracy_delta=%.2f, recommendation=%s",
            speedup,
            cost_reduction * 100,
            accuracy_delta,
            recommendation,
        )

        return ComparisonReport(
            original_metrics=original_metrics,
            synthesized_metrics=synthesized_metrics,
            speedup=speedup,
            cost_reduction=cost_reduction,
            accuracy_delta=accuracy_delta,
            recommendation=recommendation,
        )

    # ── Internal methods ───────────────────────────────────────

    async def _run_all_tasks(
        self,
        agent: Any,
        tasks: list[dict[str, Any]],
        agent_name: str,
    ) -> list[AgentRunResult]:
        """Run an agent on all tasks, repeating each num_runs times.

        Args:
            agent: Agent instance to run.
            tasks: List of task dictionaries.
            agent_name: Name for logging.

        Returns:
            List of AgentRunResult objects.
        """
        results = []

        for task in tasks:
            task_id = task.get("task_id", "unknown")

            for run_idx in range(self._num_runs):
                result = await self._run_agent(
                    agent, task, run_idx=run_idx
                )
                results.append(result)

                logger.debug(
                    "%s agent, task=%s, run=%d: success=%s, duration=%.1fms",
                    agent_name,
                    task_id,
                    run_idx,
                    result.success,
                    result.duration_ms,
                )

        return results

    async def _run_agent(
        self,
        agent: Any,
        task: dict[str, Any],
        run_idx: int = 0,
    ) -> AgentRunResult:
        """Execute one agent on one task.

        Args:
            agent: Agent instance (DSPy module or BaseAgent).
            task: Task dict with query, context, etc.
            run_idx: Run index for logging.

        Returns:
            AgentRunResult with timing and answer.
        """
        task_id = task.get("task_id", f"task_{run_idx}")
        query = task.get("query", "")
        context = task.get("context", "")

        start = time.perf_counter()
        success = True
        answer = ""
        error_msg = None
        token_cost = 0.0

        try:
            # Try DSPy module interface first
            if hasattr(agent, "forward"):
                result = agent.forward(context=context, query=query)
                answer = str(getattr(result, "answer", ""))
                # Estimate token cost (rough heuristic)
                token_cost = (len(query) + len(context) + len(answer)) / 4.0

            # Try BaseAgent interface
            elif hasattr(agent, "handle"):
                from memfun_core.types import TaskMessage

                msg = TaskMessage(
                    task_id=task_id,
                    agent_id="comparison",
                    payload={
                        "type": task.get("type", "ask"),
                        "query": query,
                        "context": context,
                    },
                )
                task_result = await agent.handle(msg)
                success = task_result.success
                answer = str(task_result.result.get("answer", ""))
                error_msg = task_result.error
                # TODO: Extract actual token usage from result
                token_cost = (len(query) + len(context)) / 4.0

            else:
                # Fallback: try calling agent as function
                result = await agent(query=query, context=context)
                answer = str(result)
                token_cost = len(query) / 4.0

        except Exception as exc:
            success = False
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "Agent run failed for task %s: %s", task_id, error_msg
            )

        duration_ms = (time.perf_counter() - start) * 1000.0

        return AgentRunResult(
            task_id=task_id,
            answer=answer,
            duration_ms=duration_ms,
            token_cost=token_cost,
            success=success,
            error=error_msg,
        )

    def _compute_metrics(
        self, results: list[AgentRunResult]
    ) -> dict[str, float]:
        """Aggregate metrics from agent run results.

        Args:
            results: List of AgentRunResult objects.

        Returns:
            Dictionary with aggregated metrics.
        """
        if not results:
            return {
                "count": 0.0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "avg_token_cost": 0.0,
                "p50_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
            }

        successful = [r for r in results if r.success]
        success_rate = len(successful) / len(results)

        durations = [r.duration_ms for r in successful]
        token_costs = [r.token_cost for r in successful]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_cost = sum(token_costs) / len(token_costs) if token_costs else 0.0

        # Compute percentiles
        sorted_durations = sorted(durations)
        p50_idx = int(len(sorted_durations) * 0.50)
        p95_idx = int(len(sorted_durations) * 0.95)

        p50_duration = (
            sorted_durations[p50_idx]
            if sorted_durations and p50_idx < len(sorted_durations)
            else 0.0
        )
        p95_duration = (
            sorted_durations[p95_idx]
            if sorted_durations and p95_idx < len(sorted_durations)
            else 0.0
        )

        return {
            "count": float(len(results)),
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
            "avg_token_cost": avg_cost,
            "p50_duration_ms": p50_duration,
            "p95_duration_ms": p95_duration,
        }

    @staticmethod
    def _calculate_speedup(
        original: dict[str, float],
        synthesized: dict[str, float],
    ) -> float:
        """Calculate speedup: original_time / synthesized_time.

        Args:
            original: Original agent metrics.
            synthesized: Synthesized agent metrics.

        Returns:
            Speedup multiplier (e.g., 2.0 = 2x faster).
        """
        orig_duration = original.get("avg_duration_ms", 1.0)
        synth_duration = synthesized.get("avg_duration_ms", 1.0)

        if synth_duration <= 0.0:
            return 1.0

        return orig_duration / synth_duration

    @staticmethod
    def _calculate_cost_reduction(
        original: dict[str, float],
        synthesized: dict[str, float],
    ) -> float:
        """Calculate fractional cost reduction.

        Args:
            original: Original agent metrics.
            synthesized: Synthesized agent metrics.

        Returns:
            Fraction (0.0-1.0) of cost saved (e.g., 0.3 = 30% savings).
        """
        orig_cost = original.get("avg_token_cost", 0.0)
        synth_cost = synthesized.get("avg_token_cost", 0.0)

        if orig_cost <= 0.0:
            return 0.0

        reduction = (orig_cost - synth_cost) / orig_cost
        return max(0.0, min(1.0, reduction))  # Clamp to [0, 1]

    @staticmethod
    def _calculate_accuracy_delta(
        original: dict[str, float],
        synthesized: dict[str, float],
    ) -> float:
        """Calculate accuracy difference (synthesized - original).

        Args:
            original: Original agent metrics.
            synthesized: Synthesized agent metrics.

        Returns:
            Accuracy delta (e.g., 0.05 = 5% more accurate).
        """
        orig_accuracy = original.get("success_rate", 0.0)
        synth_accuracy = synthesized.get("success_rate", 0.0)

        return synth_accuracy - orig_accuracy

    @staticmethod
    def _make_recommendation(
        speedup: float,
        cost_reduction: float,
        accuracy_delta: float,
    ) -> str:
        """Make a deployment recommendation based on metrics.

        Args:
            speedup: Speed improvement multiplier.
            cost_reduction: Fractional cost reduction.
            accuracy_delta: Accuracy difference.

        Returns:
            One of: "promote", "needs-more-data", "no-improvement".
        """
        # Thresholds for promotion
        min_speedup = 1.2  # At least 20% faster
        min_cost_reduction = 0.1  # At least 10% cheaper
        max_accuracy_loss = -0.05  # No more than 5% accuracy loss

        # Decision logic
        if accuracy_delta < max_accuracy_loss:
            # Too much accuracy loss
            return "no-improvement"

        if speedup >= min_speedup or cost_reduction >= min_cost_reduction:
            # Significant performance gain without accuracy loss
            if accuracy_delta >= 0.0:
                return "promote"
            else:
                # Small accuracy loss but good perf gain
                return "needs-more-data"

        if accuracy_delta > 0.05:
            # Significant accuracy gain, even without perf improvement
            return "promote"

        # No meaningful improvement
        return "no-improvement"
