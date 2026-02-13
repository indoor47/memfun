"""Specialist agents for multi-agent workflows.

Nine specialist agents wrap :class:`RLMModule` with focused system
prompts, reduced iteration caps, and shared-spec awareness.  Each
is registered via the ``@agent`` decorator so that
:class:`AgentManager` can instantiate them by name.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger
from memfun_runtime.agent import BaseAgent, agent

from memfun_agent.rlm import RLMConfig, RLMModule
from memfun_agent.shared_spec import SharedSpecStore

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult
    from memfun_runtime.context import RuntimeContext

logger = get_logger("agent.specialist")


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
        self._spec_store: SharedSpecStore | None = None

        # Callbacks for progress tracking (set by WorkflowEngine).
        self.on_step: Any | None = None
        self.on_plan: Any | None = None
        self.on_step_status: Any | None = None

    async def on_start(self) -> None:
        await super().on_start()
        self._spec_store = SharedSpecStore(self._context.state_store)

        extra_tools: dict[str, Any] = {}
        try:
            from memfun_agent.tool_bridge import create_tool_bridge

            bridge = await create_tool_bridge()
            extra_tools = bridge.get_repl_tools()
        except Exception:
            logger.debug("MCP tool bridge unavailable for %s", self.agent_id)

        self._rlm = RLMModule(
            config=RLMConfig(
                max_iterations=self._MAX_ITERATIONS,
                verbose=True,
                enable_planning=False,
            ),
            extra_tools=extra_tools,
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

        # Inject shared spec into query.
        spec_context = ""
        if workflow_id and self._spec_store:
            spec = await self._spec_store.load(workflow_id)
            if spec:
                spec_context = spec.to_agent_context()

        full_query = f"{self._SYSTEM_PREFIX}\n\n=== YOUR TASK ===\n{query}"
        if spec_context:
            full_query += f"\n\n{spec_context}"

        assert self._rlm is not None
        rlm_result = await self._rlm.aforward(
            query=full_query,
            context=context,
            conversation_history=history,
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
        "Use write_file() for every file. Install dependencies with "
        "run_cmd(). Verify your code compiles/imports correctly."
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
