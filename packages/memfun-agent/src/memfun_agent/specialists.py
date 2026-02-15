"""Specialist agents for multi-agent workflows.

Nine specialist agents wrap :class:`RLMModule` with focused system
prompts, reduced iteration caps, and shared-spec awareness.  Each
is registered via the ``@agent`` decorator so that
:class:`AgentManager` can instantiate them by name.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger
from memfun_runtime.agent import BaseAgent, agent

from memfun_agent.rlm import RLMConfig, RLMModule
from memfun_agent.shared_spec import SharedSpecStore

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult
    from memfun_runtime.context import RuntimeContext

    from memfun_agent.rlm import RLMResult

logger = get_logger("agent.specialist")


# ── Per-agent live activity (for UI spinners) ─────────────────

# Maps sub_task_id → {iteration, max_iter, last_op, agent_name}.
# Updated by on_step callbacks inside handle(); read by the CLI
# spinner in _make_workflow_progress().
AGENT_ACTIVITY: dict[str, dict[str, Any]] = {}


# ── Agent-type to name mapping ────────────────────────────────

AGENT_TYPE_MAP: dict[str, str] = {
    "file": "file-agent",
    "coder": "coder-agent",
    "test": "test-agent",
    "review": "review-agent",
    "web_search": "web-search-agent",
    "web_fetch": "web-fetch-agent",
    "planner": "planner-agent",
    "debug": "debug-agent",
    "security": "security-agent",
}


def agent_name_for_type(agent_type: str) -> str:
    """Map a sub-task ``agent_type`` to a registered agent name."""
    return AGENT_TYPE_MAP.get(agent_type, "coder-agent")


# ── Base specialist ───────────────────────────────────────────


class _SpecialistBase(BaseAgent):
    """Shared base for all specialist agents."""

    _SPECIALIST_TYPE: str = ""
    _MAX_ITERATIONS: int = 10
    _SYSTEM_PREFIX: str = ""

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context)
        self._rlm: RLMModule | None = None
        self._extra_tools: dict[str, Any] = {}
        self._spec_store: SharedSpecStore | None = None

        # Callbacks for progress tracking (set by WorkflowEngine).
        self.on_step: Any | None = None
        self.on_plan: Any | None = None
        self.on_step_status: Any | None = None

    async def on_start(self) -> None:
        await super().on_start()
        self._spec_store = SharedSpecStore(self._context.state_store)

        try:
            from memfun_agent.tool_bridge import create_tool_bridge

            bridge = await create_tool_bridge()
            self._extra_tools = bridge.get_repl_tools()
        except Exception:
            logger.debug("MCP tool bridge unavailable for %s", self.agent_id)

        self._rlm = RLMModule(
            config=RLMConfig(
                max_iterations=self._MAX_ITERATIONS,
                verbose=True,
                enable_planning=False,
            ),
            extra_tools=self._extra_tools,
            on_step=self.on_step,
            on_plan=self.on_plan,
            on_step_status=self.on_step_status,
        )
        logger.info(
            "%s started (max_iter=%d)",
            self.agent_id,
            self._MAX_ITERATIONS,
        )

    async def handle(self, task: TaskMessage) -> TaskResult:
        from memfun_core.types import TaskResult as TaskRes

        start = time.perf_counter()
        payload = task.payload
        query = payload.get("query", "")
        context = payload.get("context", "")
        workflow_id = payload.get("workflow_id", "")
        history = payload.get("conversation_history")

        # Augment context with sub-task specific input files if
        # they weren't already gathered by the shared context pass.
        inputs = payload.get("inputs", [])
        if inputs and "=== FILE:" not in context:
            try:
                from memfun_agent.context_first import ContextGatherer

                gatherer = ContextGatherer(
                    max_bytes=100_000, max_files=10,
                )
                extra = await gatherer.agather(
                    files=inputs, project_root=os.getcwd(),
                )
                if extra.strip():
                    context = extra + "\n\n" + context
            except Exception:
                logger.debug(
                    "Per-specialist context augmentation failed",
                    exc_info=True,
                )

        # Pre-read OUTPUT files so the agent sees their current content
        # before writing.  This prevents destructive overwrites where
        # the agent rewrites a file from scratch, losing existing code.
        outputs = payload.get("outputs", [])
        if outputs:
            try:
                from memfun_agent.context_first import read_affected_files

                project_root = Path(os.getcwd())
                existing = await read_affected_files(
                    outputs, project_root,
                )
                if existing.strip():
                    context = (
                        "=== EXISTING OUTPUT FILES (DO NOT REWRITE "
                        "FROM SCRATCH — use edit_file to modify) "
                        "===\n"
                        + existing
                        + "\n\n"
                        + context
                    )
            except Exception:
                logger.debug(
                    "Pre-reading output files failed",
                    exc_info=True,
                )

        # Inject shared spec into query.
        spec_context = ""
        if workflow_id and self._spec_store:
            spec = await self._spec_store.load(workflow_id)
            if spec:
                spec_context = spec.to_agent_context()

        full_query = f"{self._SYSTEM_PREFIX}\n\n=== YOUR TASK ===\n{query}"
        if spec_context:
            full_query += f"\n\n{spec_context}"

        # Create a per-task RLM with a live-activity callback so the
        # CLI spinner can show iteration progress per specialist.
        sub_task_id = payload.get("sub_task_id", task.task_id)
        max_iter = self._MAX_ITERATIONS

        def _on_step(step: Any) -> None:
            """Update AGENT_ACTIVITY with current iteration info."""
            # Extract a short description of the last operation from
            # the code the LLM produced (first meaningful line).
            code_lines = (step.code or "").strip().splitlines()
            last_op = ""
            for line in code_lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    last_op = stripped
                    break
            AGENT_ACTIVITY[sub_task_id] = {
                "iteration": step.iteration,
                "max_iter": max_iter,
                "last_op": last_op,
                "agent_name": self.agent_id,
            }

        task_rlm = RLMModule(
            config=RLMConfig(
                max_iterations=max_iter,
                verbose=True,
                enable_planning=False,
            ),
            extra_tools=self._extra_tools,
            on_step=_on_step,
            on_plan=self.on_plan,
            on_step_status=self.on_step_status,
        )

        try:
            rlm_result = await task_rlm.aforward(
                query=full_query,
                context=context,
                conversation_history=history,
            )
        finally:
            # Clean up activity entry when done.
            AGENT_ACTIVITY.pop(sub_task_id, None)

        # Post-process: verify + consistency review + polish.
        if rlm_result.success and rlm_result.files_created:
            rlm_result = await self._post_process(
                query=query,
                rlm_result=rlm_result,
                sub_task_id=sub_task_id,
            )

        # Report file creations back to shared spec.
        if workflow_id and self._spec_store and rlm_result.files_created:
            for fpath in rlm_result.files_created:
                await self._spec_store.append_finding(
                    workflow_id, self.agent_id, f"Created file: {fpath}"
                )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TaskRes(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=rlm_result.success,
            result={
                "answer": rlm_result.answer,
                "method": f"specialist_{self._SPECIALIST_TYPE}",
                "iterations": rlm_result.iterations,
                "ops": [
                    {"type": o[0], "target": o[1], "detail": o[2]}
                    for o in rlm_result.ops
                ],
                "files_created": rlm_result.files_created,
                "trajectory_length": len(rlm_result.trajectory),
                "final_reasoning": rlm_result.final_reasoning,
            },
            error=rlm_result.error,
            duration_ms=elapsed_ms,
        )


    async def _post_process(
        self,
        query: str,
        rlm_result: RLMResult,
        sub_task_id: str,
    ) -> RLMResult:
        """Verify, consistency-check, and polish specialist output.

        Runs the same quality pipeline as the context-first solver:
        1. Auto-detect linters -> run -> fix if needed (0-1 LLM calls)
        2. Consistency review against sub-task description (1 LLM call)
        3. Polish if consistency issues found (0-1 LLM calls)

        Returns an updated RLMResult with any extra ops/files appended.
        """
        from memfun_agent.context_first import (
            ConsistencyReviewer,
            FixSolver,
            OperationExecutor,
            Verifier,
            _detect_verify_commands,
            read_affected_files,
        )

        project_root = Path(os.getcwd())
        extra_ops: list[tuple[str, str, Any]] = []
        extra_files: list[str] = []

        def _update_activity(label: str) -> None:
            AGENT_ACTIVITY[sub_task_id] = {
                "iteration": rlm_result.iterations,
                "max_iter": self._MAX_ITERATIONS,
                "last_op": label,
                "agent_name": self.agent_id,
            }

        try:
            # ── Phase 1: Verification (linters) ────────────────
            verify_cmds = _detect_verify_commands(project_root)
            if verify_cmds and rlm_result.files_created:
                _update_activity("Verifying...")
                verifier = Verifier(project_root)
                vr = await verifier.averify(verify_cmds)

                if not vr.passed:
                    logger.info(
                        "%s: verification failed, attempting fix: %s",
                        self.agent_id, vr.errors[:200],
                    )
                    _update_activity("Fixing lint errors...")
                    current = await read_affected_files(
                        rlm_result.files_created, project_root,
                    )
                    if current:
                        fix_solver = FixSolver()
                        try:
                            fix_ops = await fix_solver.afix(
                                query=query,
                                full_context=current,
                                verification_errors=vr.errors,
                            )
                            if fix_ops:
                                executor = OperationExecutor(
                                    project_root, edit_only=True,
                                )
                                await executor.execute(fix_ops)
                                extra_ops.extend(executor.ops)
                                extra_files.extend(executor.files_created)
                        except Exception as exc:
                            logger.warning(
                                "%s: fix solver failed: %s",
                                self.agent_id, exc,
                            )

            # ── Phase 2: Consistency review + polish ───────────
            all_files = list(set(rlm_result.files_created + extra_files))
            if all_files:
                _update_activity("Reviewing consistency...")
                actual = await read_affected_files(all_files, project_root)
                if actual:
                    reviewer = ConsistencyReviewer()
                    review = await reviewer.areview(
                        user_request=query,
                        intended_changes=(
                            f"Agent answer:\n{rlm_result.answer[:2000]}"
                        ),
                        actual_file_contents=actual,
                    )
                    if review.has_issues and review.issues:
                        logger.info(
                            "%s: %d consistency issues, polishing",
                            self.agent_id, len(review.issues),
                        )
                        _update_activity("Polishing...")
                        issues_text = (
                            "CONSISTENCY REVIEW ISSUES:\n"
                            + "\n".join(
                                f"- {i}" for i in review.issues
                            )
                            + f"\n\nReasoning: {review.reasoning}"
                        )
                        fix_solver = FixSolver()
                        polish_ops = await fix_solver.afix(
                            query=query,
                            full_context=actual,
                            verification_errors=issues_text,
                        )
                        if polish_ops:
                            executor = OperationExecutor(
                                project_root, edit_only=True,
                            )
                            await executor.execute(polish_ops)
                            extra_ops.extend(executor.ops)
                            extra_files.extend(executor.files_created)
                    else:
                        logger.info(
                            "%s: consistency check passed",
                            self.agent_id,
                        )

        except Exception as exc:
            # Post-processing is best-effort — never block the pipeline.
            logger.warning(
                "%s: post-processing failed (non-fatal): %s",
                self.agent_id, exc,
            )
        finally:
            AGENT_ACTIVITY.pop(sub_task_id, None)

        # Merge extras into result.
        if extra_ops or extra_files:
            rlm_result.ops = list(rlm_result.ops) + extra_ops
            rlm_result.files_created = list(set(
                rlm_result.files_created + extra_files
            ))

        return rlm_result


# ── Concrete specialists ──────────────────────────────────────


@agent(
    name="file-agent",
    version="1.0",
    capabilities=["file-analysis", "codebase-exploration"],
)
class FileAgent(_SpecialistBase):
    """Reads and analyses files — never creates code."""

    _SPECIALIST_TYPE = "file"
    _MAX_ITERATIONS = 8
    _SYSTEM_PREFIX = (
        "You are a FILE ANALYSIS specialist. Your job is to READ and ANALYZE "
        "files, understand codebase structure, discover patterns, and document "
        "findings. You DO NOT write new code or create files. You only read, "
        "analyze, and report.\n\n"
        "Use read_file(), list_files(), and search to understand the codebase "
        "thoroughly. Set state['FINAL'] to a detailed analysis with specific "
        "file paths and line references."
    )


@agent(
    name="coder-agent",
    version="1.0",
    capabilities=["code-generation", "file-creation"],
)
class CoderAgent(_SpecialistBase):
    """Writes production-quality code."""

    _SPECIALIST_TYPE = "coder"
    _MAX_ITERATIONS = 15
    _SYSTEM_PREFIX = (
        "You are a CODE GENERATION specialist. Your job is to WRITE "
        "production-quality code based on the task description and shared "
        "specification. Follow the naming conventions and interface contracts "
        "in the shared spec exactly.\n\n"
        "CRITICAL FILE EDITING RULES:\n"
        "- ALWAYS use read_file(path) FIRST to see the current content "
        "of any file you will modify.\n"
        "- For EXISTING files: use edit_file(path, old_text, new_text) "
        "to make targeted changes. NEVER rewrite an entire existing file "
        "with write_file() — this destroys existing code.\n"
        "- For NEW files only: use write_file(path, content).\n"
        "- Install dependencies with run_cmd().\n"
        "- Verify your code compiles/imports correctly."
    )


@agent(
    name="test-agent",
    version="1.0",
    capabilities=["test-generation", "test-execution"],
)
class TestAgent(_SpecialistBase):
    """Writes and runs tests for code from other agents."""

    _SPECIALIST_TYPE = "test"
    _MAX_ITERATIONS = 10
    _SYSTEM_PREFIX = (
        "You are a TESTING specialist. Your job is to WRITE and RUN tests "
        "for code produced by other agents. Read the files created, "
        "understand their interfaces, and write comprehensive tests.\n\n"
        "CRITICAL FILE EDITING RULES:\n"
        "- ALWAYS use read_file(path) FIRST before modifying any file.\n"
        "- For EXISTING test files: use edit_file(path, old_text, new_text) "
        "to add or modify tests without destroying existing ones.\n"
        "- For NEW test files: use write_file(path, content).\n\n"
        "Run the tests with pytest or the appropriate test runner. "
        "Report which tests pass and which fail. "
        "Set state['FINAL'] to the test results."
    )


@agent(
    name="review-agent",
    version="1.0",
    capabilities=["code-review", "quality-assurance"],
)
class ReviewAgent(_SpecialistBase):
    """Reviews code from other agents for quality and consistency."""

    _SPECIALIST_TYPE = "review"
    _MAX_ITERATIONS = 8
    _SYSTEM_PREFIX = (
        "You are a CODE REVIEW specialist. Your job is to REVIEW code "
        "produced by other agents for consistency, quality, security, and "
        "adherence to the shared specification.\n\n"
        "Read all files created. Check:\n"
        "1. Do they follow the shared spec's naming conventions?\n"
        "2. Are there conflicts between files from different agents?\n"
        "3. Do imports resolve correctly?\n"
        "4. Are there security issues?\n"
        "5. Is error handling adequate?\n\n"
        "Set state['FINAL'] to a structured review. Start with "
        "'approved: true' or 'approved: false', then list specific issues "
        "with format: '[task_id] severity: description'."
    )


# ── Extended specialists ─────────────────────────────────────


@agent(
    name="web-search-agent",
    version="1.0",
    capabilities=["web-search", "information-retrieval"],
)
class WebSearchAgent(_SpecialistBase):
    """Searches the web for current information, documentation, and APIs."""

    _SPECIALIST_TYPE = "web_search"
    _MAX_ITERATIONS = 8
    _SYSTEM_PREFIX = (
        "You are a WEB SEARCH specialist. Your job is to SEARCH the web "
        "for current information relevant to the task. Use web_search() "
        "to find documentation, API references, library versions, best "
        "practices, tutorials, and any external information needed by "
        "other agents.\n\n"
        "Search strategically: use specific technical queries, check "
        "multiple sources, and verify information across results. "
        "Set state['FINAL'] to a structured summary of findings with "
        "URLs and key facts extracted from each source."
    )


@agent(
    name="web-fetch-agent",
    version="1.0",
    capabilities=["web-fetch", "content-extraction"],
)
class WebFetchAgent(_SpecialistBase):
    """Fetches and extracts content from specific URLs."""

    _SPECIALIST_TYPE = "web_fetch"
    _MAX_ITERATIONS = 8
    _SYSTEM_PREFIX = (
        "You are a WEB FETCH specialist. Your job is to FETCH specific "
        "URLs and extract relevant content. Use web_fetch() to retrieve "
        "web pages, documentation pages, API references, README files, "
        "and other online resources.\n\n"
        "Extract the specific information requested -- do not just dump "
        "raw page content. Summarize, extract code examples, find "
        "configuration details, or pull out whatever specific data is "
        "needed. Set state['FINAL'] to the extracted information "
        "organized clearly with source URLs."
    )


@agent(
    name="planner-agent",
    version="1.0",
    capabilities=["planning", "decomposition"],
)
class PlannerAgent(_SpecialistBase):
    """Decomposes sub-problems and creates execution plans."""

    _SPECIALIST_TYPE = "planner"
    _MAX_ITERATIONS = 6
    _SYSTEM_PREFIX = (
        "You are a PLANNING specialist. Your job is to ANALYZE a "
        "sub-problem and produce a structured plan for how to solve it. "
        "You DO NOT write code or create files. You only read files, "
        "analyze, plan, and report.\n\n"
        "Read relevant files to understand the current state, identify "
        "what needs to change, and produce a step-by-step plan with:\n"
        "1. What files need to be created or modified\n"
        "2. What interfaces or contracts must be followed\n"
        "3. What order the steps should happen in\n"
        "4. What risks or edge cases to watch for\n\n"
        "Set state['FINAL'] to the structured plan. Other agents will "
        "use this plan to guide their implementation."
    )


@agent(
    name="debug-agent",
    version="1.0",
    capabilities=["debugging", "error-diagnosis"],
)
class DebugAgent(_SpecialistBase):
    """Diagnoses errors, reads logs, and traces issues."""

    _SPECIALIST_TYPE = "debug"
    _MAX_ITERATIONS = 12
    _SYSTEM_PREFIX = (
        "You are a DEBUGGING specialist. Your job is to DIAGNOSE errors, "
        "trace issues, and identify root causes. Read error logs, stack "
        "traces, test output, and source code to find the source of "
        "problems.\n\n"
        "Use read_file() to examine source code and logs. Use run_cmd() "
        "to run failing tests or reproduce errors. Use grep to search "
        "for error patterns across the codebase.\n\n"
        "If you apply fixes: use edit_file(path, old_text, new_text) for "
        "targeted changes. NEVER rewrite entire files with write_file().\n\n"
        "Your analysis must include:\n"
        "1. The exact error and where it occurs\n"
        "2. The root cause (not just symptoms)\n"
        "3. A specific fix recommendation with file paths and code\n"
        "4. How to verify the fix works\n\n"
        "Set state['FINAL'] to a structured diagnosis with root cause "
        "and recommended fix."
    )


@agent(
    name="security-agent",
    version="1.0",
    capabilities=["security-review", "vulnerability-detection"],
)
class SecurityAgent(_SpecialistBase):
    """Reviews code for security vulnerabilities and best practices."""

    _SPECIALIST_TYPE = "security"
    _MAX_ITERATIONS = 8
    _SYSTEM_PREFIX = (
        "You are a SECURITY REVIEW specialist. Your job is to analyze "
        "code for security vulnerabilities, unsafe patterns, and "
        "compliance issues. You focus EXCLUSIVELY on security -- unlike "
        "the general review agent which covers quality and consistency.\n\n"
        "Check for:\n"
        "1. Injection vulnerabilities (SQL, command, template, XSS)\n"
        "2. Authentication/authorization flaws\n"
        "3. Hardcoded secrets, API keys, credentials\n"
        "4. Insecure cryptography or hashing\n"
        "5. SSRF, path traversal, unsafe deserialization\n"
        "6. Missing input validation and output encoding\n"
        "7. Dependency vulnerabilities (known CVEs)\n"
        "8. Insecure default configurations\n\n"
        "Set state['FINAL'] to a structured security report. Each "
        "finding must include: severity (critical/high/medium/low), "
        "file path, line number, description, and recommended fix."
    )
