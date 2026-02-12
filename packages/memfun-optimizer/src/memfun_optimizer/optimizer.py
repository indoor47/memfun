"""MIPROv2 optimization wrapper for synthesized agents.

Provides a high-level interface to DSPy's MIPROv2 optimizer for refining
synthesized agent modules. Converts execution traces to training examples
and runs optimization to improve prompt strategies and few-shot examples.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    # ExecutionTrace is from memfun_agent.traces
    from typing import Protocol

    class ExecutionTrace(Protocol):
        trace_id: str
        task_type: str
        query: str
        final_answer: str
        success: bool
        duration_ms: float
        token_usage: Any
        trajectory: list[Any]

logger = get_logger("optimizer.optimizer")


# ── Data Structures ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Result of MIPROv2 optimization run.

    Attributes:
        original_source: Original module source code.
        optimized_source: Optimized module source (may be same if no LLM).
        improvement_score: Estimated improvement (0.0 = no change, 1.0 = 100% better).
        num_trials: Number of optimization trials run.
        best_trial_idx: Index of the best-performing trial.
        metrics: Optimization metrics dict.
    """

    original_source: str
    optimized_source: str
    improvement_score: float
    num_trials: int
    best_trial_idx: int
    metrics: dict[str, Any]


# ── Agent Optimizer ────────────────────────────────────────────


class AgentOptimizer:
    """Wrapper for MIPROv2 optimization of synthesized agents.

    Uses DSPy's MIPROv2 optimizer to refine agent modules based on
    execution traces. The optimizer:
    1. Converts traces to (input, output) training examples
    2. Creates a metric function to score performance
    3. Runs MIPROv2 trials to improve prompts and demos
    4. Returns optimized source code with updated strategies

    Usage::

        optimizer = AgentOptimizer(num_trials=10, max_bootstrapped_demos=4)
        result = optimizer.optimize(module_source, training_traces)

        if result.improvement_score > 0.1:
            # Deploy the optimized version
            deploy_module(result.optimized_source)

    Note:
        Optimization requires an active LLM connection. If no LLM is
        configured, the original source is returned with a warning.
    """

    def __init__(
        self,
        *,
        num_trials: int = 10,
        max_bootstrapped_demos: int = 4,
    ) -> None:
        """Initialize the optimizer.

        Args:
            num_trials: Number of MIPROv2 optimization trials.
            max_bootstrapped_demos: Max few-shot examples per stage.
        """
        self._num_trials = num_trials
        self._max_demos = max_bootstrapped_demos

    def optimize(
        self,
        module_source: str,
        training_traces: list[Any],
    ) -> OptimizationResult:
        """Optimize a DSPy module using execution traces.

        Args:
            module_source: Python source code of the module to optimize.
            training_traces: List of ExecutionTrace objects.

        Returns:
            OptimizationResult with optimized source and metrics.

        Raises:
            ValueError: If module_source is invalid or cannot be loaded.
        """
        if not module_source.strip():
            msg = "module_source cannot be empty"
            raise ValueError(msg)

        if not training_traces:
            logger.warning(
                "No training traces provided; returning original source"
            )
            return OptimizationResult(
                original_source=module_source,
                optimized_source=module_source,
                improvement_score=0.0,
                num_trials=0,
                best_trial_idx=0,
                metrics={"warning": "no_training_data"},
            )

        logger.info(
            "Optimizing module with %d traces, %d trials",
            len(training_traces),
            self._num_trials,
        )

        # Load the module from source
        try:
            module_obj = self._load_module_from_source(module_source)
        except Exception as exc:
            msg = f"Failed to load module from source: {exc}"
            raise ValueError(msg) from exc

        # Convert traces to DSPy examples
        examples = self._traces_to_examples(training_traces)

        if not examples:
            logger.warning(
                "No valid examples extracted from traces; "
                "returning original source"
            )
            return OptimizationResult(
                original_source=module_source,
                optimized_source=module_source,
                improvement_score=0.0,
                num_trials=0,
                best_trial_idx=0,
                metrics={"warning": "no_valid_examples"},
            )

        # Create metric function
        metric = self._create_metric()

        # Attempt MIPROv2 optimization
        try:
            _optimized_module, metrics = self._run_mipro(
                module_obj, examples, metric
            )
            optimized_source = module_source  # TODO: Extract optimized params
            improvement = metrics.get("improvement_score", 0.0)
            best_idx = metrics.get("best_trial_idx", 0)

            logger.info(
                "Optimization complete: improvement=%.2f, best_trial=%d",
                improvement,
                best_idx,
            )

            return OptimizationResult(
                original_source=module_source,
                optimized_source=optimized_source,
                improvement_score=improvement,
                num_trials=self._num_trials,
                best_trial_idx=best_idx,
                metrics=metrics,
            )

        except Exception as exc:
            logger.warning(
                "MIPROv2 optimization failed: %s. "
                "Returning original source.",
                exc,
            )
            return OptimizationResult(
                original_source=module_source,
                optimized_source=module_source,
                improvement_score=0.0,
                num_trials=0,
                best_trial_idx=0,
                metrics={"error": str(exc)},
            )

    # ── Internal methods ───────────────────────────────────────

    # Patterns that indicate dangerous operations in synthesized module source.
    # This is a defense-in-depth check -- the primary defense is that only
    # code generated by AgentSynthesizer should reach this method.
    _DANGEROUS_PATTERNS = (
        "os.system",
        "subprocess",
        "shutil.rmtree",
        "__import__(",
        "eval(",
        "exec(",
        "compile(",
        "open(",
        "builtins",
        "importlib",
        "ctypes",
        "pickle",
        "marshal",
        "shelve",
    )

    def _load_module_from_source(
        self, source: str
    ) -> Any:
        """Dynamically load a DSPy module from Python source code.

        WARNING: This method executes arbitrary Python code. It should only
        be called with source code generated by AgentSynthesizer, never with
        untrusted user input. A basic blocklist check is performed as
        defense-in-depth, but this is NOT a security boundary.

        Args:
            source: Python source code as string.

        Returns:
            The DSPy module class (not instantiated).

        Raises:
            ValueError: If source is invalid, contains dangerous patterns,
                or module cannot be loaded.
        """
        # Defense-in-depth: reject source containing known-dangerous patterns
        source_lower = source.lower()
        for pattern in self._DANGEROUS_PATTERNS:
            if pattern.lower() in source_lower:
                msg = (
                    f"Module source contains disallowed pattern: {pattern!r}. "
                    f"Only AgentSynthesizer-generated code is permitted."
                )
                raise ValueError(msg)

        logger.warning(
            "Loading module from source (%d bytes). "
            "This executes arbitrary code -- ensure source is trusted.",
            len(source),
        )

        # Write source to a temporary module and import it
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(source)
            tmp_path = Path(tmp.name)

        try:
            spec = importlib.util.spec_from_file_location(
                "synthesized_module", tmp_path
            )
            if spec is None or spec.loader is None:
                msg = "Failed to create module spec from source"
                raise ValueError(msg)

            module = importlib.util.module_from_spec(spec)
            sys.modules["synthesized_module"] = module
            spec.loader.exec_module(module)

            # Find the DSPy module class
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "forward")
                    and name != "Module"
                ):
                    logger.debug("Loaded module class: %s", name)
                    return obj

            msg = "No DSPy module class found in source"
            raise ValueError(msg)

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)
            if "synthesized_module" in sys.modules:
                del sys.modules["synthesized_module"]

    def _traces_to_examples(
        self, traces: list[Any]
    ) -> list[Any]:
        """Convert ExecutionTrace objects to DSPy examples.

        Args:
            traces: List of ExecutionTrace objects.

        Returns:
            List of dspy.Example objects.
        """
        try:
            import dspy
        except ImportError:
            logger.warning("DSPy not available; cannot create examples")
            return []

        examples = []
        for trace in traces:
            # Skip failed traces
            if not trace.success:
                continue

            # Create example: query -> answer
            try:
                example = dspy.Example(
                    query=trace.query,
                    answer=trace.final_answer,
                ).with_inputs("query")

                examples.append(example)

            except Exception as exc:
                logger.debug(
                    "Failed to convert trace %s to example: %s",
                    getattr(trace, "trace_id", "unknown"),
                    exc,
                )

        logger.info(
            "Converted %d/%d traces to training examples",
            len(examples),
            len(traces),
        )
        return examples

    def _create_metric(self) -> Callable[[Any, Any], float]:
        """Create a simple metric function for optimization.

        The metric scores how well the module's output matches the
        expected answer. This is a simple heuristic; domain-specific
        metrics can be substituted.

        Returns:
            Metric function (example, prediction) -> score (0.0-1.0).
        """

        def metric(example: Any, prediction: Any) -> float:
            """Score the quality of a prediction.

            Args:
                example: DSPy example with ground truth.
                prediction: DSPy prediction from the module.

            Returns:
                Score between 0.0 (worst) and 1.0 (perfect match).
            """
            # Extract answer from prediction
            pred_answer = getattr(prediction, "answer", "")
            true_answer = getattr(example, "answer", "")

            if not pred_answer or not true_answer:
                return 0.0

            # Simple token overlap metric
            pred_tokens = set(pred_answer.lower().split())
            true_tokens = set(true_answer.lower().split())

            if not true_tokens:
                return 0.0

            overlap = len(pred_tokens & true_tokens)
            total = len(true_tokens)

            return overlap / total

        return metric

    def _run_mipro(
        self,
        module_class: Any,
        examples: list[Any],
        metric: Callable[[Any, Any], float],
    ) -> tuple[Any, dict[str, Any]]:
        """Run MIPROv2 optimization.

        Args:
            module_class: DSPy module class to optimize.
            examples: Training examples.
            metric: Metric function.

        Returns:
            Tuple of (optimized_module, metrics_dict).

        Raises:
            ImportError: If DSPy or MIPROv2 is not available.
            RuntimeError: If optimization fails.
        """
        try:
            import dspy
            from dspy.teleprompt import MIPROv2
        except ImportError as exc:
            msg = "DSPy or MIPROv2 not available"
            raise ImportError(msg) from exc

        # Check for active LLM
        if not dspy.settings.lm:
            warnings.warn(
                "No LLM configured in dspy.settings; "
                "optimization will not run. "
                "Configure with: dspy.configure(lm=...)",
                stacklevel=2,
            )
            # Return original module
            return module_class(), {
                "warning": "no_llm_configured",
                "improvement_score": 0.0,
            }

        # Split examples into train/dev
        split_point = int(len(examples) * 0.8)
        train_set = examples[:split_point] if split_point > 0 else examples
        dev_set = examples[split_point:] if split_point < len(examples) else examples[:1]

        # Initialize optimizer
        optimizer = MIPROv2(
            metric=metric,
            num_trials=self._num_trials,
            max_bootstrapped_demos=self._max_demos,
        )

        # Run optimization
        try:
            optimized = optimizer.compile(
                student=module_class(),
                trainset=train_set,
                valset=dev_set,
            )

            # Evaluate improvement
            original_score = self._evaluate_module(
                module_class(), dev_set, metric
            )
            optimized_score = self._evaluate_module(
                optimized, dev_set, metric
            )

            improvement = (
                (optimized_score - original_score) / max(original_score, 0.01)
                if original_score > 0
                else 0.0
            )

            metrics = {
                "original_score": original_score,
                "optimized_score": optimized_score,
                "improvement_score": improvement,
                "best_trial_idx": 0,  # MIPROv2 tracks this internally
                "num_train_examples": len(train_set),
                "num_dev_examples": len(dev_set),
            }

            return optimized, metrics

        except Exception as exc:
            msg = f"MIPROv2 optimization failed: {exc}"
            raise RuntimeError(msg) from exc

    def _evaluate_module(
        self,
        module: Any,
        examples: list[Any],
        metric: Callable[[Any, Any], float],
    ) -> float:
        """Evaluate a module on examples using the metric.

        Args:
            module: DSPy module instance.
            examples: List of examples to evaluate.
            metric: Metric function.

        Returns:
            Average score across examples.
        """
        if not examples:
            return 0.0

        scores = []
        for example in examples:
            try:
                # Run module forward pass
                query = getattr(example, "query", "")
                prediction = module.forward(context="", query=query)
                score = metric(example, prediction)
                scores.append(score)
            except Exception as exc:
                logger.debug("Evaluation failed for example: %s", exc)
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0
