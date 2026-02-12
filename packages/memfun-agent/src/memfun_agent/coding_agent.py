"""RLM Coding Agent: the main Memfun autonomous coding agent.

Dispatches coding tasks (analyze, fix, review, ask, explain) using
the RLM module for complex/large-context tasks and direct DSPy
predictors for simpler ones. Collects execution traces for later
optimization via MIPROv2/GEPA.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import dspy
from memfun_core.logging import get_logger
from memfun_runtime.agent import BaseAgent, agent

from memfun_agent.rlm import RLMConfig, RLMModule, RLMResult
from memfun_agent.signatures import (
    BugFix,
    CodeAnalysis,
    CodeExplanation,
    CodeReview,
    QueryTriage,
)
from memfun_agent.tool_bridge import MCPToolBridge, create_tool_bridge
from memfun_agent.traces import (
    ExecutionTrace,
    TraceCollector,
    TraceStep,
)

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult
    from memfun_runtime.context import RuntimeContext

logger = get_logger("agent.coding")


# ── Task type enumeration ──────────────────────────────────────

_TASK_TYPES = frozenset({
    "analyze",
    "fix",
    "review",
    "ask",
    "explain",
})


# ── The agent ──────────────────────────────────────────────────


@agent(
    name="rlm-coder",
    version="1.0",
    capabilities=["code-analysis", "bug-fixing", "code-review"],
)
class RLMCodingAgent(BaseAgent):
    """Autonomous coding agent powered by DSPy's RLM pattern.

    Handles coding tasks by dispatching to:
    - **RLM module** for large-context tasks (exploring repos, multi-file
      analysis, complex debugging).
    - **Direct DSPy predictors** for short, focused tasks (explain a
      function, review a small diff).

    The agent collects execution traces for every task, enabling
    downstream optimization via MIPROv2 or agent synthesis.

    Configuration is read from ``RuntimeContext.config.llm``.
    """

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context)
        self._tool_bridge: MCPToolBridge | None = None
        self._trace_collector: TraceCollector | None = None
        self._rlm: RLMModule | None = None
        self.on_step: Any | None = None
        self.on_plan: Any | None = None
        self.on_step_status: Any | None = None

        # DSPy predictors for simple (non-RLM) tasks
        self._analyzers: dict[str, dspy.Module] = {}
        self._triage = dspy.Predict(QueryTriage)

    # ── Lifecycle ──────────────────────────────────────────

    async def on_start(self) -> None:
        """Initialize tool bridge, trace collector, and RLM module."""
        await super().on_start()

        # Set up trace collection
        self._trace_collector = TraceCollector(
            state_store=self._context.state_store,
        )

        # Initialize MCP tool bridge (best-effort)
        try:
            self._tool_bridge = await create_tool_bridge()
        except Exception as exc:
            logger.warning(
                "MCP Tool Bridge initialization failed: %s. "
                "Tools will be unavailable in the REPL.",
                exc,
            )
            self._tool_bridge = MCPToolBridge()

        # Build extra tools for the RLM REPL
        extra_tools: dict[str, Any] = {}
        if self._tool_bridge is not None:
            extra_tools = self._tool_bridge.get_repl_tools()

        # Configure RLM module
        rlm_config = RLMConfig(
            max_iterations=_DEFAULT_RLM_ITERATIONS,
            verbose=True,
        )

        # Use local REPL (not sandbox) so that:
        # 1. Helper functions (write_file, run_cmd, etc.) work
        #    (sandbox serializes them to strings via JSON)
        # 2. Files are created in the user's project CWD
        #    (sandbox uses a temp dir that gets deleted)
        self._rlm = RLMModule(
            config=rlm_config,
            sandbox=None,
            extra_tools=extra_tools,
            on_step=self.on_step,
            on_plan=self.on_plan,
            on_step_status=self.on_step_status,
        )

        # Set up direct predictors for simple tasks
        self._analyzers = {
            "analyze": dspy.ChainOfThought(CodeAnalysis),
            "fix": dspy.ChainOfThought(BugFix),
            "review": dspy.ChainOfThought(CodeReview),
            "explain": dspy.ChainOfThought(CodeExplanation),
        }

        logger.info(
            "RLMCodingAgent started (max_iter=%d, tools=%d)",
            rlm_config.max_iterations,
            len(extra_tools),
        )

    async def on_stop(self) -> None:
        """Clean up resources."""
        await super().on_stop()
        logger.info("RLMCodingAgent stopped")

    # ── Main dispatch ───────────────────────────────────────

    async def handle(self, task: TaskMessage) -> TaskResult:
        """Process a coding task.

        Expected payload keys:
        - ``type``: One of analyze, fix, review, ask, explain.
        - ``query``: The user's question or task description.
        - ``code`` / ``context``: Source code or large context string.
        - ``bug_description``: (for fix) Description of the bug.
        - ``audience``: (for explain) Target audience level.

        Returns:
            TaskResult with the agent's response.
        """
        from memfun_core.types import TaskResult as TaskRes

        start = time.perf_counter()
        payload = task.payload
        task_type = payload.get("type", "ask")
        query = payload.get("query", "")
        code = payload.get("code", "")
        context = payload.get("context", code)

        if task_type not in _TASK_TYPES:
            return TaskRes(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                error=(
                    f"Unknown task type: {task_type}. "
                    f"Valid types: {', '.join(sorted(_TASK_TYPES))}"
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )

        try:
            # LLM-based triage: decide how to handle this query
            category = await self._triage_query(query, context)

            # Guard: very short queries with conversation history
            # are almost certainly follow-ups, not standalone questions
            has_history = (
                "=== LAST COMPLETED TASK ===" in context
            )
            if (
                category == "direct"
                and has_history
                and len(query.strip()) < 20
            ):
                logger.info(
                    "Overriding triage 'direct' -> 'project'"
                    " (short follow-up with history)"
                )
                category = "project"

            logger.info("Query triage: %s", category)

            if category == "direct":
                result_data = await self._handle_direct_answer(
                    query
                )
            elif (
                category in ("project", "task", "web")
                and self._rlm is not None
            ):
                # Enable planning for "task" category;
                # "web" uses RLM so the agent can call
                # web_search/web_fetch helpers in the REPL.
                if category in ("task", "web"):
                    self._rlm.config = RLMConfig(
                        max_iterations=self._rlm.config.max_iterations,
                        max_output_chars=self._rlm.config.max_output_chars,
                        preview_length=self._rlm.config.preview_length,
                        max_history_chars=self._rlm.config.max_history_chars,
                        verbose=self._rlm.config.verbose,
                        enable_planning=True,
                    )
                else:
                    self._rlm.config = RLMConfig(
                        max_iterations=self._rlm.config.max_iterations,
                        max_output_chars=self._rlm.config.max_output_chars,
                        preview_length=self._rlm.config.preview_length,
                        max_history_chars=self._rlm.config.max_history_chars,
                        verbose=self._rlm.config.verbose,
                        enable_planning=False,
                    )
                result_data = await self._handle_rlm(
                    task_type, query, context, payload
                )
            else:
                result_data = await self._handle_direct(
                    task_type, query, context, payload
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Collect trace
            trace = self._build_trace(
                task_type=task_type,
                query=query,
                context_length=len(context),
                result_data=result_data,
                duration_ms=elapsed_ms,
            )
            if self._trace_collector is not None:
                await self._trace_collector.save(trace)

            return TaskRes(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result=result_data,
                duration_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Task %s failed: %s", task.task_id, exc
            )
            return TaskRes(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms,
            )

    # ── RLM path ──────────────────────────────────────────

    async def _handle_rlm(
        self,
        task_type: str,
        query: str,
        context: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a task using the full RLM exploration loop."""
        assert self._rlm is not None

        # Build a richer query incorporating the task type
        full_query = self._build_rlm_query(
            task_type, query, payload
        )

        # Extract conversation history from payload
        history = payload.get("conversation_history")

        rlm_result: RLMResult = await self._rlm.aforward(
            query=full_query,
            context=context,
            conversation_history=history,
        )

        return {
            "answer": rlm_result.answer,
            "task_type": task_type,
            "method": "rlm",
            "iterations": rlm_result.iterations,
            "trajectory_length": len(rlm_result.trajectory),
            "final_reasoning": rlm_result.final_reasoning,
            "total_tokens": rlm_result.token_usage.total_tokens,
            "ops": [
                {
                    "type": o[0],
                    "target": o[1],
                    "detail": o[2],
                }
                for o in rlm_result.ops
            ],
            "files_created": rlm_result.files_created,
        }

    # ── Direct predictor path ───────────────────────────────

    async def _handle_direct(
        self,
        task_type: str,
        query: str,
        context: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a task using a direct DSPy predictor (no RLM loop)."""
        predictor = self._analyzers.get(task_type)

        if predictor is None:
            # Fallback to generic ask via ChainOfThought
            fallback = dspy.ChainOfThought(
                "context, query -> answer"
            )
            result = await asyncio.to_thread(
                fallback, context=context, query=query
            )
            return {
                "answer": str(getattr(result, "answer", "")),
                "task_type": task_type,
                "method": "direct_fallback",
            }

        # Dispatch to the appropriate signature
        if task_type == "analyze":
            result = await asyncio.to_thread(
                predictor, code=context, query=query
            )
            return {
                "answer": str(
                    getattr(result, "analysis", "")
                ),
                "issues": _to_str_list(
                    getattr(result, "issues", [])
                ),
                "suggestions": _to_str_list(
                    getattr(result, "suggestions", [])
                ),
                "task_type": task_type,
                "method": "direct",
            }

        if task_type == "fix":
            bug_desc = payload.get(
                "bug_description", query
            )
            result = await asyncio.to_thread(
                predictor,
                code=context,
                bug_description=bug_desc,
            )
            return {
                "root_cause": str(
                    getattr(result, "root_cause", "")
                ),
                "fixed_code": str(
                    getattr(result, "fixed_code", "")
                ),
                "explanation": str(
                    getattr(result, "explanation", "")
                ),
                "task_type": task_type,
                "method": "direct",
            }

        if task_type == "review":
            review_context = payload.get("review_context", "")
            result = await asyncio.to_thread(
                predictor,
                code=context,
                context=review_context,
            )
            return {
                "summary": str(
                    getattr(result, "summary", "")
                ),
                "findings": _to_str_list(
                    getattr(result, "findings", [])
                ),
                "approved": bool(
                    getattr(result, "approved", False)
                ),
                "task_type": task_type,
                "method": "direct",
            }

        if task_type == "explain":
            audience = payload.get("audience", "intermediate")
            result = await asyncio.to_thread(
                predictor, code=context, audience=audience
            )
            return {
                "explanation": str(
                    getattr(result, "explanation", "")
                ),
                "key_concepts": _to_str_list(
                    getattr(result, "key_concepts", [])
                ),
                "task_type": task_type,
                "method": "direct",
            }

        # Should not reach here given _TASK_TYPES validation
        return {
            "answer": "",
            "task_type": task_type,
            "method": "unknown",
        }

    async def _triage_query(
        self,
        query: str,
        context: str,
    ) -> str:
        """Use the LLM to classify the query handling strategy.

        Returns one of: 'direct', 'project', 'task', 'web'.
        """
        # Extract recent conversation from context
        recent_conv = ""
        if "=== LAST COMPLETED TASK ===" in context:
            start = context.index(
                "=== LAST COMPLETED TASK ==="
            )
            # Find the next section or end
            end = context.find("=== ", start + 10)
            if end == -1:
                end = len(context)
            recent_conv = context[start:end].strip()[:1500]

        # Build a brief project summary (not the full context)
        ctx_lines = context.split("\n")[:5]
        project_summary = " | ".join(
            line.strip()
            for line in ctx_lines
            if line.strip()
        )[:200]

        try:
            result = await asyncio.to_thread(
                self._triage,
                query=query,
                recent_conversation=recent_conv
                or "(no prior conversation)",
                project_summary=project_summary
                or "(no project context)",
            )
            category = str(
                getattr(result, "category", "project")
            ).strip().lower()
            # Normalize to valid categories
            if category not in (
                "direct", "project", "task", "web"
            ):
                category = "project"
            return category
        except Exception as exc:
            logger.warning(
                "Query triage failed: %s, defaulting to project",
                exc,
            )
            return "project"

    async def _handle_direct_answer(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Answer a query directly without project context or RLM.

        Used for general knowledge questions, simple explanations, etc.
        """
        try:
            predictor = dspy.ChainOfThought("query -> answer")
            result = await asyncio.to_thread(
                predictor, query=query
            )
            answer = str(getattr(result, "answer", ""))
            return {
                "answer": answer,
                "task_type": "ask",
                "method": "direct_triage",
            }
        except Exception as exc:
            return {
                "answer": f"Error: {exc}",
                "task_type": "ask",
                "method": "direct_triage",
            }

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _build_rlm_query(
        task_type: str,
        query: str,
        payload: dict[str, Any],
    ) -> str:
        """Build an enriched query string for the RLM module."""
        prefixes = {
            "analyze": (
                "Analyze the following codebase. "
                "Focus on structure, quality, and potential issues."
            ),
            "fix": (
                "Find and fix the bug described below. "
                "Produce corrected code with an explanation."
            ),
            "review": (
                "Review the following code. Provide structured "
                "feedback with severity-tagged findings."
            ),
            "ask": "Answer the following question about the code.",
            "explain": (
                "Explain how the following code works in clear, "
                "accessible language."
            ),
        }
        prefix = prefixes.get(task_type, "")

        parts = [prefix, f"\nQuery: {query}"]

        # Add extra context from payload
        if task_type == "fix" and "bug_description" in payload:
            parts.append(
                f"\nBug description: {payload['bug_description']}"
            )
        if task_type == "explain" and "audience" in payload:
            parts.append(
                f"\nTarget audience: {payload['audience']}"
            )

        return "\n".join(parts)

    def _build_trace(
        self,
        task_type: str,
        query: str,
        context_length: int,
        result_data: dict[str, Any],
        duration_ms: float,
    ) -> ExecutionTrace:
        """Build an ExecutionTrace from a completed task."""
        trajectory: list[TraceStep] = []

        # If we used RLM, the trajectory info is in result_data
        traj_len = result_data.get("trajectory_length", 0)
        if traj_len > 0:
            # The actual TraceSteps are in the RLM result;
            # we store a summary step referencing the RLM run.
            trajectory.append(
                TraceStep(
                    iteration=1,
                    reasoning=result_data.get(
                        "final_reasoning", ""
                    ),
                    code=f"(RLM loop: {traj_len} iterations)",
                    output=result_data.get("answer", "")[:500],
                )
            )

        return ExecutionTrace(
            task_type=task_type,
            query=query,
            context_length=context_length,
            trajectory=trajectory,
            final_answer=str(
                result_data.get(
                    "answer",
                    result_data.get("explanation", ""),
                )
            ),
            success=True,
            duration_ms=duration_ms,
            agent_id=self.agent_id,
            metadata={
                "method": result_data.get("method", ""),
                "iterations": result_data.get("iterations", 0),
            },
        )


# ── Module-level constants ─────────────────────────────────────

_DEFAULT_RLM_ITERATIONS = 25


# ── Utilities ──────────────────────────────────────────────────


def _to_str_list(value: Any) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)] if value else []
