"""Core RLM (Recursive Language Model) module for the Memfun coding agent.

Implements the RLM pattern from Zhang, Krasta & Khattab (2025): large context
is stored as Python variables in a sandboxed REPL; the LLM writes exploration
code rather than consuming the full context in its token window.

Key components:
- ``RLMModule`` -- DSPy module implementing the RLM exploration loop.
- ``LocalREPL`` -- In-process Python REPL for when no sandbox backend exists.
- ``build_context_metadata`` -- Produces compact metadata for the LLM.
"""
from __future__ import annotations

import asyncio
import io
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import dspy
from memfun_core.logging import get_logger

from memfun_agent.signatures import RLMExploration, TaskPlanning
from memfun_agent.traces import TokenUsage, TraceStep

if TYPE_CHECKING:
    from collections.abc import Callable

    from memfun_core.types import ExecutionResult
    from memfun_runtime.protocols.sandbox import SandboxAdapter

logger = get_logger("agent.rlm")

# ── Constants ──────────────────────────────────────────────────

_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_MAX_OUTPUT_CHARS = 0  # 0 = no limit (rely on history window)
_PREVIEW_LENGTH = 500
_MAX_HISTORY_CHARS = 120_000


# ── Context metadata helper ────────────────────────────────────


def build_context_metadata(
    context: str | bytes,
    preview_length: int = _PREVIEW_LENGTH,
    *,
    has_history: bool = False,
    history_len: int = 0,
) -> str:
    """Build a compact metadata string for the LLM about the context.

    Returns a human-readable summary containing type, length, and a
    short preview of the context value.
    """
    ctx_type = type(context).__name__
    ctx_len = len(context)

    if isinstance(context, bytes):
        try:
            preview = context[:preview_length].decode(
                "utf-8", errors="replace"
            )
        except Exception:
            preview = repr(context[:preview_length])
    else:
        preview = context[:preview_length]

    lines = [
        f"type: {ctx_type}",
        f"length: {ctx_len:,} characters",
        f"preview ({min(preview_length, ctx_len)} chars):",
        textwrap.indent(preview, "  "),
    ]

    lines.append("")
    lines.append("Available helper functions in the REPL:")
    lines.append(
        "  write_file(path: str, content: str) -> str"
        "  # Write a file, returns abs path"
    )
    lines.append(
        "  read_file(path: str) -> str"
        "  # Read file contents as string"
    )
    lines.append(
        "  run_cmd(cmd: str) -> str"
        "  # Run shell command, returns stdout string"
    )
    lines.append(
        "  edit_file(path: str, old: str, new: str) -> bool"
        "  # Replace text in file"
    )
    lines.append(
        "  list_files(path: str = '.', with_times: bool = False)"
        " -> list[str]"
        "  # Returns LIST of file paths (with_times adds dates)"
    )
    lines.append(
        "  llm_query(question: str, context: str) -> str"
        "  # Ask sub-LM, returns answer string"
    )
    lines.append(
        "  web_search(query: str, max_results: int = 5)"
        " -> list[dict]"
        "  # Search web, returns [{title, url, snippet}]"
    )
    lines.append(
        "  web_fetch(url: str, max_length: int = 50000)"
        " -> str"
        "  # Fetch URL content as string"
    )
    if has_history:
        lines.append("")
        lines.append(
            f"CONVERSATION HISTORY: {history_len} entries "
            f"available as `conversation_history` variable "
            f"(list of dicts, chronological order)."
        )
        lines.append(
            "  Each entry has: role ('user'/'assistant'), "
            "content (str), and optionally: "
            "files_created (list), ops (list), method (str)"
        )
        lines.append(
            "  search_history(keyword: str) -> list[dict]"
            "  # Search history for entries containing keyword"
        )
        lines.append(
            "  IMPORTANT: Always check conversation_history "
            "for context about WHY things were done, past "
            "decisions, and user preferences."
        )
    lines.append("")
    lines.append("CRITICAL RULES:")
    lines.append(
        "- Output ONLY pure Python code. NO markdown, "
        "NO ```python fences, NO explanatory text."
    )
    lines.append(
        "- list_files() returns a list[str]. "
        "Iterate with for-loop, do NOT call .split() on it."
    )
    lines.append(
        "- read_file() returns a str. "
        "Use .split('\\n') to get lines."
    )
    lines.append(
        "- Always use write_file() instead of raw open()."
    )
    lines.append(
        "- Always use run_cmd() instead of subprocess."
    )
    lines.append(
        "- Use web_search() when you need current information "
        "from the internet (APIs, docs, prices, news, etc.)."
    )
    lines.append(
        "- Use web_fetch() to read full content from a URL."
    )
    lines.append(
        "- You are fully autonomous: install deps, create "
        "files, run commands, deploy, verify — do everything."
    )
    lines.append(
        "- Never tell the user to do something you can do."
    )
    lines.append(
        "- Write production-quality code with proper "
        "structure and styling."
    )
    lines.append(
        "- Always verify: check files exist, test the app."
    )

    return "\n".join(lines)


# ── Local REPL (fallback when no sandbox) ──────────────────────


@dataclass(slots=True)
class REPLResult:
    """Result from executing code in the local REPL."""

    stdout: str = ""
    stderr: str = ""
    success: bool = True
    duration_ms: float = 0.0


class LocalREPL:
    """In-process Python REPL for RLM exploration.

    Provides a persistent namespace where context variables and helper
    functions live across iterations. Used as a fallback when no real
    SandboxAdapter is available.

    Security note: This executes code in the current process. In
    production, use a proper SandboxAdapter (Docker/Modal/Deno).
    """

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {"__builtins__": __builtins__}

    @property
    def namespace(self) -> dict[str, Any]:
        """The REPL's variable namespace (for injection)."""
        return self._namespace

    def execute(
        self,
        code: str,
        timeout: int = 30,
    ) -> REPLResult:
        """Execute Python code in the local namespace.

        Args:
            code: Python source code to execute.
            timeout: Not enforced in local REPL (for interface compat).

        Returns:
            REPLResult with captured stdout/stderr.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.perf_counter()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self._namespace)
            success = True
        except Exception:
            stderr_buf.write(traceback.format_exc())
            success = False

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            success=success,
            duration_ms=elapsed_ms,
        )


# ── RLM Module ─────────────────────────────────────────────────


@dataclass(slots=True)
class RLMConfig:
    """Configuration for the RLM module."""

    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS
    preview_length: int = _PREVIEW_LENGTH
    max_history_chars: int = _MAX_HISTORY_CHARS
    verbose: bool = False
    enable_planning: bool = True


@dataclass(slots=True)
class RLMResult:
    """Result of an RLM execution."""

    answer: str = ""
    trajectory: list[TraceStep] = field(default_factory=list)
    final_reasoning: str = ""
    success: bool = True
    error: str | None = None
    iterations: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    plan: list[str] = field(default_factory=list)
    ops: list[tuple] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)


class RLMModule(dspy.Module):
    """DSPy module implementing the RLM exploration loop.

    The RLM pattern:
    1. Store large context as a variable in a sandboxed REPL.
    2. Show the LLM only metadata (type, length, preview).
    3. LLM generates Python code to explore/search/process context.
    4. Code executes; only truncated output metadata goes back.
    5. LLM can call ``llm_query(sub_query, sub_context)`` for focused
       sub-LM reasoning on snippets.
    6. Loop until ``state['FINAL']`` is set or max iterations reached.

    Usage::

        rlm = RLMModule(config=RLMConfig(max_iterations=15))
        result = await rlm.aforward(
            query="Find all security vulnerabilities",
            context=large_codebase_string,
        )
        print(result.answer)
        print(result.trajectory)
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        sandbox: SandboxAdapter | None = None,
        sub_lm: dspy.LM | None = None,
        extra_tools: dict[str, Callable[..., Any]] | None = None,
        on_step: Callable[[TraceStep], None] | None = None,
        on_plan: Callable[[list[str]], None] | None = None,
        on_step_status: Callable[[int, str, str], None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RLMConfig()
        self._sandbox = sandbox
        self._sub_lm = sub_lm
        self._extra_tools = extra_tools or {}
        self._on_step = on_step
        self._on_plan = on_plan
        self._on_step_status = on_step_status

        # DSPy predictor for generating exploration code
        self.explorer = dspy.Predict(RLMExploration)
        self.planner = dspy.Predict(TaskPlanning)

    # ── Public API ─────────────────────────────────────────

    def forward(
        self,
        query: str,
        context: str | bytes,
        **kwargs: Any,
    ) -> RLMResult:
        """Synchronous forward -- delegates to the RLM loop.

        For async usage prefer ``aforward``.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop; create a new
            # thread to avoid deadlock.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.aforward(
                        query=query, context=context, **kwargs
                    ),
                )
                return future.result()

        return asyncio.run(
            self.aforward(query=query, context=context, **kwargs)
        )

    async def aforward(
        self,
        query: str,
        context: str | bytes,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RLMResult:
        """Async forward: run the RLM exploration loop.

        Args:
            query: The user's question or task.
            context: Large context string (project state, etc.).
            conversation_history: Optional list of conversation
                entries (dicts with role, content, files_created, etc.)
                to inject as a searchable variable in the REPL.
        """
        start = time.perf_counter()
        trajectory: list[TraceStep] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        sub_lm_calls = 0
        sub_lm_tokens = 0

        # Build context metadata for the LLM
        ctx_meta = build_context_metadata(
            context,
            self.config.preview_length,
            has_history=bool(conversation_history),
            history_len=len(conversation_history or []),
        )

        # Set up REPL environment
        repl = self._create_repl(
            context,
            conversation_history=conversation_history,
        )
        code_history = ""

        answer = ""
        final_reasoning = ""
        error: str | None = None
        success = True
        iteration = 0

        # ── Planning phase ────────────────────────────────
        plan: list[str] = []
        current_plan_step = -1

        if self.config.enable_planning:
            try:
                plan_result = await asyncio.to_thread(
                    self.planner,
                    query=query,
                    context_metadata=ctx_meta,
                )
                raw_steps = getattr(plan_result, "steps", [])
                if isinstance(raw_steps, list) and len(raw_steps) >= 2:
                    plan = [str(s) for s in raw_steps[:7]]
                    if self._on_plan is not None:
                        try:
                            self._on_plan(plan)
                        except Exception:
                            logger.debug(
                                "on_plan callback failed",
                                exc_info=True,
                            )

                    # Inject plan into query so the LLM follows it
                    plan_text = "\n".join(
                        f"  {i + 1}. {s}" for i, s in enumerate(plan)
                    )
                    query = (
                        f"{query}\n\n"
                        f"=== EXECUTION PLAN ===\n"
                        f"Follow this plan step by step:\n"
                        f"{plan_text}\n\n"
                        f"=== AUTONOMY DIRECTIVE ===\n"
                        f"You are fully autonomous. Do EVERYTHING "
                        f"yourself:\n"
                        f"- Install dependencies: "
                        f"run_cmd('pip install ...')\n"
                        f"- Create ALL files with write_file()\n"
                        f"- Run shell commands with run_cmd()\n"
                        f"- Edit files with edit_file()\n"
                        f"- Deploy if asked (ngrok, etc.)\n"
                        f"- VERIFY your work: check files exist, "
                        f"run the app, test endpoints\n"
                        f"- NEVER say 'the user should...' — "
                        f"do it yourself\n\n"
                        f"=== QUALITY STANDARDS ===\n"
                        f"- Write production-ready code, not "
                        f"stubs or TODOs\n"
                        f"- Clean structure, proper error "
                        f"handling, good naming\n"
                        f"- Frontend: modern CSS, responsive, "
                        f"polished UI/UX\n"
                        f"- Backend: proper routes, validation, "
                        f"error responses\n"
                        f"- Each iteration: ask 'is this the "
                        f"best way?' before writing\n\n"
                        f"=== EXECUTION RULES ===\n"
                        f"- Set state['current_step'] = N "
                        f"(0-indexed) when starting step N\n"
                        f"- When ALL steps are done AND "
                        f"verified, set state['FINAL'] = "
                        f"summary\n"
                        f"- Summary MUST list all files created "
                        f"and commands run\n"
                        f"- If a step fails, fix it and retry "
                        f"before moving on\n"
                    )
            except Exception:
                logger.debug(
                    "Planning phase failed, continuing without plan",
                    exc_info=True,
                )

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                # Truncate history to avoid blowing up token budget
                if len(code_history) > self.config.max_history_chars:
                    code_history = (
                        "... (earlier history truncated) ...\n"
                        + code_history[-self.config.max_history_chars:]
                    )

                # Ask LLM for next exploration step
                prediction = await asyncio.to_thread(
                    self.explorer,
                    context_metadata=ctx_meta,
                    query=query,
                    code_history=code_history or "(no history yet)",
                )

                reasoning = getattr(prediction, "reasoning", "")
                raw_code = getattr(prediction, "next_code", "")
                next_code = _clean_generated_code(raw_code)
                final_reasoning = reasoning

                # Try to extract token usage from DSPy LM
                try:
                    lm = dspy.settings.lm
                    if hasattr(lm, "history") and lm.history:
                        last = lm.history[-1]
                        usage = {}
                        if isinstance(last, dict):
                            usage = last.get("usage", {})
                            if not usage:
                                resp = last.get("response")
                                if resp and hasattr(resp, "usage"):
                                    usage = resp.usage or {}
                        if isinstance(usage, dict):
                            total_prompt_tokens += usage.get(
                                "prompt_tokens", 0
                            )
                            total_completion_tokens += usage.get(
                                "completion_tokens", 0
                            )
                        elif usage:
                            total_prompt_tokens += getattr(
                                usage, "prompt_tokens", 0
                            )
                            total_completion_tokens += getattr(
                                usage, "completion_tokens", 0
                            )
                except Exception:
                    pass

                if not next_code.strip():
                    logger.debug(
                        "Iteration %d: LLM produced empty code, stopping",
                        iteration,
                    )
                    break

                # Execute the code
                step_start = time.perf_counter()
                exec_result = await self._execute_code(
                    repl, next_code
                )
                step_ms = (
                    (time.perf_counter() - step_start) * 1000.0
                )

                # Build output string
                output = exec_result.stdout
                output_truncated = False
                if (
                    self.config.max_output_chars > 0
                    and len(output) > self.config.max_output_chars
                ):
                    output = (
                        output[: self.config.max_output_chars]
                        + f"\n... (truncated, total {len(exec_result.stdout)} chars)"
                    )
                    output_truncated = True

                if exec_result.stderr:
                    output += f"\n[stderr]: {exec_result.stderr[:2000]}"

                # Record step
                step = TraceStep(
                    iteration=iteration,
                    reasoning=reasoning,
                    code=next_code,
                    output=output,
                    output_truncated=output_truncated,
                    duration_ms=step_ms,
                    cumulative_tokens=(
                        total_prompt_tokens
                        + total_completion_tokens
                    ),
                )
                trajectory.append(step)

                # Notify callback (for live UI updates)
                if self._on_step is not None:
                    try:
                        self._on_step(step)
                    except Exception:
                        logger.debug(
                            "on_step callback failed",
                            exc_info=True,
                        )

                # Track plan step progress
                if plan:
                    state = repl.namespace.get("state", {})
                    new_step = state.get("current_step", -1)

                    # Fallback: auto-advance based on iteration
                    if not isinstance(new_step, int) or new_step < 0:
                        steps_per_iter = max(
                            1,
                            self.config.max_iterations // len(plan),
                        )
                        new_step = min(
                            (iteration - 1) // steps_per_iter,
                            len(plan) - 1,
                        )

                    if new_step != current_plan_step:
                        # Mark previous step completed
                        if (
                            current_plan_step >= 0
                            and self._on_step_status is not None
                        ):
                            try:
                                self._on_step_status(
                                    current_plan_step,
                                    "completed",
                                    reasoning[:120],
                                )
                            except Exception:
                                logger.debug(
                                    "on_step_status callback"
                                    " failed",
                                    exc_info=True,
                                )
                        # Mark new step in-progress
                        current_plan_step = new_step
                        if self._on_step_status is not None:
                            try:
                                detail = (
                                    plan[new_step]
                                    if new_step < len(plan)
                                    else ""
                                )
                                self._on_step_status(
                                    new_step,
                                    "in_progress",
                                    detail,
                                )
                            except Exception:
                                logger.debug(
                                    "on_step_status callback"
                                    " failed",
                                    exc_info=True,
                                )

                # Append to history for next iteration
                if not exec_result.success:
                    code_history += (
                        f"\n--- Iteration {iteration} "
                        f"(FAILED) ---\n"
                        f"Code:\n{next_code}\n"
                        f"ERROR:\n{output[:1500]}\n"
                        f"FIX: Write pure Python only. "
                        f"No markdown. Check return types "
                        f"(list_files returns list, "
                        f"read_file returns str).\n"
                    )
                else:
                    code_history += (
                        f"\n--- Iteration {iteration} ---\n"
                        f"Code:\n{next_code}\n"
                        f"Output ({len(output)} chars):\n"
                        f"{output[:1000]}\n"
                    )

                # Add running state summary so LLM knows
                # what's been accomplished
                state = repl.namespace.get("state", {})
                ops_so_far = state.get("_ops", [])
                if ops_so_far:
                    writes = [
                        o[1] for o in ops_so_far
                        if o[0] == "write"
                    ]
                    edits = [
                        o[1] for o in ops_so_far
                        if o[0] == "edit"
                    ]
                    reads = [
                        o[1] for o in ops_so_far
                        if o[0] == "read"
                    ]
                    cmds = [
                        o[1] for o in ops_so_far
                        if o[0] == "cmd"
                    ]
                    web_searches = [
                        o[1] for o in ops_so_far
                        if o[0] == "web_search"
                    ]
                    web_fetches = [
                        o[1] for o in ops_so_far
                        if o[0] == "web_fetch"
                    ]
                    status_parts = []
                    if reads:
                        # Show unique files read.
                        unique_reads = list(dict.fromkeys(reads))
                        status_parts.append(
                            f"Files read: "
                            f"{', '.join(unique_reads[-5:])}"
                        )
                    if writes:
                        status_parts.append(
                            f"Files written: "
                            f"{', '.join(writes[-5:])}"
                        )
                    if edits:
                        unique_edits = list(dict.fromkeys(edits))
                        status_parts.append(
                            f"Files edited: "
                            f"{', '.join(unique_edits[-5:])}"
                        )
                    if cmds:
                        status_parts.append(
                            f"Commands run: "
                            f"{', '.join(cmds[-5:])}"
                        )
                    if web_searches:
                        status_parts.append(
                            f"Web searches: "
                            f"{', '.join(web_searches[-3:])}"
                        )
                    if web_fetches:
                        status_parts.append(
                            f"URLs fetched: "
                            f"{len(web_fetches)}"
                        )
                    code_history += (
                        f"[State: {'; '.join(status_parts)}]\n"
                    )

                # ── Iteration budget + stall detection ──────
                remaining = self.config.max_iterations - iteration
                code_history += (
                    f"[Budget: {remaining} iterations remaining"
                    f" out of {self.config.max_iterations}]\n"
                )

                if remaining <= 2:
                    code_history += (
                        "⚠ FINAL ITERATIONS: You MUST set "
                        "state['FINAL'] now. Summarize what you "
                        "did, list all files created/modified, "
                        "and set FINAL immediately.\n"
                    )
                elif remaining <= self.config.max_iterations // 2:
                    # Check for stall: if we've done reads but
                    # no writes/edits/commands
                    state = repl.namespace.get("state", {})
                    ops_so_far = state.get("_ops", [])
                    action_ops = [
                        o for o in ops_so_far
                        if o[0] in ("write", "edit", "cmd")
                    ]
                    if not action_ops:
                        code_history += (
                            "⚠ STALL DETECTED: You've been "
                            "reading files but haven't written "
                            "or edited anything yet. You have "
                            "enough context. STOP READING and "
                            "START ACTING: use write_file() or "
                            "edit_file() to implement your fix "
                            "NOW. Do not read more files.\n"
                        )
                    else:
                        # Check for re-read loop: if the agent
                        # has action ops but keeps re-reading
                        # the same files (common after edits)
                        read_ops = [
                            o[1] for o in ops_so_far
                            if o[0] == "read"
                        ]
                        if len(read_ops) >= 4:
                            from collections import Counter

                            read_counts = Counter(read_ops)
                            hot_files = [
                                (f, c) for f, c in
                                read_counts.most_common(3)
                                if c >= 3
                            ]
                            if hot_files:
                                names = ", ".join(
                                    f"{f.rsplit('/', 1)[-1]}"
                                    f" ({c}x)"
                                    for f, c in hot_files
                                )
                                code_history += (
                                    f"⚠ RE-READ LOOP: You've "
                                    f"read {names} multiple "
                                    f"times. The content hasn't "
                                    f"changed. If your task is "
                                    f"done, set state['FINAL'] "
                                    f"with a summary of what you "
                                    f"did. If not done, make "
                                    f"your remaining edits NOW "
                                    f"without re-reading.\n"
                                )

                if self.config.verbose:
                    logger.info(
                        "RLM iteration %d: %s",
                        iteration,
                        reasoning[:120],
                    )

                # Check if the LLM set FINAL
                state = repl.namespace.get("state", {})
                if "FINAL" in state:
                    answer = str(state["FINAL"])
                    break

                # Also check FINAL_VAR pattern
                if "FINAL_VAR" in state:
                    var_name = state["FINAL_VAR"]
                    answer = str(
                        repl.namespace.get(var_name, "")
                    )
                    break
            else:
                # Max iterations reached without FINAL
                logger.warning(
                    "RLM reached max iterations (%d) without "
                    "FINAL answer",
                    self.config.max_iterations,
                )
                # Build auto-summary from tracked operations
                state = repl.namespace.get("state", {})
                ops = state.get("_ops", [])
                files = state.get("_files", [])

                if ops or files:
                    summary_parts: list[str] = []
                    writes = [
                        o for o in ops if o[0] == "write"
                    ]
                    edits = [
                        o for o in ops if o[0] == "edit"
                    ]
                    cmds = [
                        o for o in ops if o[0] == "cmd"
                    ]
                    searches = [
                        o for o in ops
                        if o[0] == "web_search"
                    ]
                    fetches = [
                        o for o in ops
                        if o[0] == "web_fetch"
                    ]

                    if searches:
                        summary_parts.append(
                            "**Web searches:**"
                        )
                        for _, query, count in searches:
                            summary_parts.append(
                                f"- `{query}` "
                                f"({count} results)"
                            )
                    if fetches:
                        summary_parts.append(
                            "\n**URLs fetched:**"
                        )
                        for _, url, size in fetches:
                            summary_parts.append(
                                f"- `{url}` ({size})"
                            )
                    if writes:
                        summary_parts.append(
                            "\n**Files created:**"
                        )
                        for _, path, size in writes:
                            summary_parts.append(
                                f"- `{path}` ({size} chars)"
                            )
                    if edits:
                        summary_parts.append(
                            "\n**Files edited:**"
                        )
                        for _, path, _ in edits:
                            summary_parts.append(
                                f"- `{path}`"
                            )
                    if cmds:
                        summary_parts.append(
                            "\n**Commands run:**"
                        )
                        for _, cmd, rc in cmds:
                            status = (
                                "ok"
                                if rc == 0
                                else f"exit {rc}"
                            )
                            summary_parts.append(
                                f"- `{cmd}` {status}"
                            )

                    answer = "\n".join(summary_parts)
                else:
                    # No tracked ops - try partial_answer
                    answer = str(
                        state.get(
                            "partial_answer",
                            state.get(
                                "FINAL",
                                "(max iterations reached "
                                "without final answer)",
                            ),
                        )
                    )

            # Mark remaining plan steps
            if plan and self._on_step_status is not None:
                for i in range(len(plan)):
                    if i > current_plan_step:
                        try:
                            self._on_step_status(i, "skipped", "")
                        except Exception:
                            logger.debug(
                                "on_step_status callback failed",
                                exc_info=True,
                            )
                if current_plan_step >= 0:
                    try:
                        self._on_step_status(
                            current_plan_step, "completed", ""
                        )
                    except Exception:
                        logger.debug(
                            "on_step_status callback failed",
                            exc_info=True,
                        )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            success = False
            logger.error("RLM execution failed: %s", error)

        (time.perf_counter() - start) * 1000.0

        # Extract tracked operations from REPL state
        final_state = repl.namespace.get("state", {})
        final_ops = final_state.get("_ops", [])
        final_files = list(
            set(final_state.get("_files", []))
        )

        return RLMResult(
            answer=answer,
            trajectory=trajectory,
            final_reasoning=final_reasoning,
            success=success,
            error=error,
            iterations=iteration,
            token_usage=TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=(
                    total_prompt_tokens + total_completion_tokens
                ),
                sub_lm_calls=sub_lm_calls,
                sub_lm_tokens=sub_lm_tokens,
            ),
            plan=plan,
            ops=final_ops,
            files_created=final_files,
        )

    # ── Internal helpers ───────────────────────────────────

    def _create_repl(
        self,
        context: str | bytes,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> LocalREPL:
        """Create and initialize a REPL with the context and tools."""
        repl = LocalREPL()

        # Inject the context as a variable
        repl.namespace["context"] = context
        repl.namespace["state"] = {}

        # Inject conversation history as a searchable variable
        history = conversation_history or []
        repl.namespace["conversation_history"] = history
        repl.namespace["search_history"] = (
            _make_search_history(history)
        )

        # Inject llm_query for sub-LM calls
        repl.namespace["llm_query"] = self._make_llm_query()
        repl.namespace["llm_query_batched"] = (
            self._make_llm_query_batched()
        )

        # Inject extra tools (e.g., MCP tool wrappers)
        for name, func in self._extra_tools.items():
            repl.namespace[name] = func

        # Standard library imports available in the REPL
        repl.execute(
            "import os, re, json, collections, "
            "itertools, math, textwrap, subprocess"
        )

        # Initialize operations tracker
        repl.namespace["state"]["_ops"] = []
        repl.namespace["state"]["_files"] = []

        # Inject robust helper functions
        repl.namespace["write_file"] = _make_write_file(
            repl.namespace
        )
        repl.namespace["read_file"] = _make_read_file(
            repl.namespace
        )
        repl.namespace["run_cmd"] = _make_run_cmd(
            repl.namespace
        )
        repl.namespace["edit_file"] = _make_edit_file(
            repl.namespace
        )
        repl.namespace["list_files"] = _make_list_files()
        repl.namespace["web_search"] = _make_web_search(
            repl.namespace
        )
        repl.namespace["web_fetch"] = _make_web_fetch(
            repl.namespace
        )

        return repl

    def _make_llm_query(
        self,
    ) -> Callable[[str, str], str]:
        """Create the ``llm_query`` function for use in the REPL.

        This allows the RLM to spawn sub-LM calls on focused snippets.
        """
        sub_lm = self._sub_lm

        def llm_query(
            sub_query: str,
            sub_context: str = "",
        ) -> str:
            """Query a sub-LM with a focused question and context.

            Args:
                sub_query: The question to answer.
                sub_context: Relevant context snippet.

            Returns:
                The sub-LM's answer as a string.
            """
            try:
                predictor = dspy.Predict("context, query -> answer")
                if sub_lm is not None:
                    with dspy.context(lm=sub_lm):
                        result = predictor(
                            context=sub_context,
                            query=sub_query,
                        )
                else:
                    result = predictor(
                        context=sub_context,
                        query=sub_query,
                    )
                return str(getattr(result, "answer", ""))
            except Exception as exc:
                return f"[llm_query error]: {exc}"

        return llm_query

    def _make_llm_query_batched(
        self,
    ) -> Callable[[list[tuple[str, str]]], list[str]]:
        """Create ``llm_query_batched`` for parallel sub-LM calls."""
        llm_query = self._make_llm_query()

        def llm_query_batched(
            queries: list[tuple[str, str]],
        ) -> list[str]:
            """Run multiple sub-LM queries.

            Args:
                queries: List of (sub_query, sub_context) tuples.

            Returns:
                List of answer strings, one per query.
            """
            return [
                llm_query(q, c) for q, c in queries
            ]

        return llm_query_batched

    async def _execute_code(
        self,
        repl: LocalREPL,
        code: str,
    ) -> _ExecOutput:
        """Execute code using sandbox or local REPL.

        Tries the SandboxAdapter first; falls back to LocalREPL.
        """
        if self._sandbox is not None:
            try:
                result: ExecutionResult = (
                    await self._sandbox.execute(
                        code,
                        context=repl.namespace,
                        timeout=30,
                    )
                )
                return _ExecOutput(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    success=result.exit_code == 0,
                )
            except NotImplementedError:
                logger.debug(
                    "Sandbox not available, falling back to "
                    "local REPL"
                )
            except Exception as exc:
                logger.warning(
                    "Sandbox execution failed: %s, falling "
                    "back to local REPL",
                    exc,
                )

        # Fallback: local REPL
        result_local = repl.execute(code)
        return _ExecOutput(
            stdout=result_local.stdout,
            stderr=result_local.stderr,
            success=result_local.success,
        )


# ── REPL helper function factories ─────────────────────────────


def _make_write_file(
    ns: dict[str, Any],
) -> Callable[[str, str], str]:
    """Create a ``write_file`` helper bound to *ns*."""
    import os as _os

    def write_file(path: str, content: str) -> str:
        """Write *content* to *path*. Creates parent dirs.

        Returns the absolute path written.

        **Overwrite guard**: If the file already exists and the new
        content would lose >30% of lines, the write is blocked and
        a warning is printed suggesting ``edit_file()`` instead.
        """
        path = _os.path.abspath(path)

        # Guard: detect destructive overwrites of existing files.
        if _os.path.isfile(path):
            try:
                with open(path) as fh:
                    existing = fh.read()
            except (OSError, UnicodeDecodeError):
                existing = ""

            old_lines = len(existing.splitlines())
            new_lines = len(content.splitlines())

            if old_lines > 10 and new_lines < old_lines * 0.7:
                loss_pct = (1 - new_lines / old_lines) * 100
                print(
                    f"BLOCKED: write_file would lose {loss_pct:.0f}% "
                    f"of {path} ({old_lines} -> {new_lines} lines). "
                    f"Use edit_file(path, old_text, new_text) to make "
                    f"targeted changes instead of rewriting the whole "
                    f"file. Or use read_file(path) first to see the "
                    f"current content."
                )
                return path

        _os.makedirs(
            _os.path.dirname(path) or ".", exist_ok=True
        )
        with open(path, "w") as fh:
            fh.write(content)
        ns["state"]["_ops"].append(
            ("write", path, len(content))
        )
        ns["state"]["_files"].append(path)
        print(f"Wrote {path} ({len(content)} chars)")
        return path

    return write_file


def _make_read_file(
    ns: dict[str, Any],
) -> Callable[[str], str]:
    """Create a ``read_file`` helper bound to *ns*.

    Includes read caching: if the same file is read again and its
    content hasn't changed, returns a short summary instead of the
    full content.  This prevents the RLM from burning context by
    re-reading unchanged files every iteration.
    """
    import hashlib
    import os as _os

    # Track {abs_path: content_md5} for cache invalidation.
    _read_cache: dict[str, str] = {}

    def read_file(path: str) -> str:
        """Read and return file contents.

        If the file was already read and hasn't changed, returns a
        short summary instead of the full content.  The original
        content is still accessible as a Python variable from a
        prior iteration.
        """
        path = _os.path.abspath(path)
        with open(path) as fh:
            content = fh.read()

        content_hash = hashlib.md5(
            content.encode("utf-8", errors="replace")
        ).hexdigest()

        ns["state"]["_ops"].append(
            ("read", path, len(content))
        )

        # Check if content is identical to last read.
        prev_hash = _read_cache.get(path)
        _read_cache[path] = content_hash

        if prev_hash == content_hash:
            # File unchanged — return summary to save context.
            lines = len(content.splitlines())
            print(
                f"Already read {path} ({len(content)} chars, "
                f"{lines} lines) — content unchanged since last "
                f"read. Use your existing knowledge of this file "
                f"instead of re-reading it."
            )
            return (
                f"[File already read and unchanged: {path} "
                f"({len(content)} chars, {lines} lines). "
                f"Content is identical to your previous read. "
                f"Proceed with what you already know.]"
            )

        print(f"Read {path} ({len(content)} chars)")
        return content

    return read_file


def _make_run_cmd(
    ns: dict[str, Any],
) -> Callable[[str], str]:
    """Create a ``run_cmd`` helper bound to *ns*."""
    import os as _os
    import subprocess as _sp

    def run_cmd(cmd: str) -> str:
        """Run a shell command. Returns stdout string.

        Stderr is printed if non-empty.
        """
        result = _sp.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_os.getcwd(),
        )
        ns["state"]["_ops"].append(
            ("cmd", cmd, result.returncode)
        )
        if result.stdout:
            print(result.stdout[:5000])
        if result.stderr:
            print(f"[stderr]: {result.stderr[:2000]}")
        return result.stdout

    return run_cmd


def _make_edit_file(
    ns: dict[str, Any],
) -> Callable[[str, str, str], bool]:
    """Create an ``edit_file`` helper bound to *ns*."""
    import os as _os

    from memfun_agent.context_first import _fuzzy_find_and_replace

    def edit_file(
        path: str, old_text: str, new_text: str
    ) -> bool:
        """Replace first occurrence of *old_text* in file.

        Uses fuzzy matching (whitespace normalization, then line-level
        difflib) when exact matching fails.

        Returns ``True`` if the replacement was made.
        """
        path = _os.path.abspath(path)
        with open(path) as fh:
            content = fh.read()

        new_content, ratio, snippet, strategy = _fuzzy_find_and_replace(
            content, old_text, new_text
        )

        if new_content is None:
            print(
                f"Text not found in {path} "
                f"(best ratio={ratio:.2f}, strategy={strategy})"
            )
            if snippet:
                print(f"  Closest match: {snippet[:200]}")
            return False

        if strategy != "exact":
            print(
                f"Fuzzy match in {path} "
                f"(ratio={ratio:.2f}, strategy={strategy})"
            )

        with open(path, "w") as fh:
            fh.write(new_content)
        ns["state"]["_ops"].append(
            ("edit", path, len(new_content))
        )
        ns["state"]["_files"].append(path)
        print(f"Edited {path}")
        return True

    return edit_file


def _make_list_files() -> Callable[[str], list[str]]:
    """Create a ``list_files`` helper."""
    import os as _os

    def list_files(
        path: str = ".", with_times: bool = False
    ) -> list[str]:
        """List all files recursively under *path*.

        Args:
            path: Directory to search.
            with_times: If True, returns 'path (modified: YYYY-MM-DD)'.

        Returns:
            Sorted list of file paths.
        """
        import datetime

        files: list[str] = []
        for root, dirs, filenames in _os.walk(path):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d != "__pycache__"
            ]
            for fname in filenames:
                fpath = _os.path.join(root, fname)
                if with_times:
                    try:
                        mtime = _os.path.getmtime(fpath)
                        dt = datetime.datetime.fromtimestamp(
                            mtime
                        ).strftime("%Y-%m-%d %H:%M")
                        fpath = f"{fpath} (modified: {dt})"
                    except OSError:
                        pass
                files.append(fpath)
        return sorted(files)

    return list_files


def _make_search_history(
    history: list[dict[str, Any]],
) -> Callable[[str], list[dict[str, Any]]]:
    """Create a ``search_history`` helper for conversation history."""

    def search_history(
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Search conversation history for entries containing keyword.

        Args:
            keyword: Text to search for (case-insensitive).

        Returns:
            List of matching history entries (dicts with role,
            content, and turn_number).
        """
        keyword_lower = keyword.lower()
        results = []
        for i, entry in enumerate(history):
            content = str(entry.get("content", ""))
            # Search in content, files_created, ops
            searchable = content.lower()
            files = entry.get("files_created", [])
            if files:
                searchable += " " + " ".join(files).lower()
            ops = entry.get("ops", [])
            if ops:
                for op in ops:
                    if isinstance(op, dict):
                        searchable += " " + str(
                            op.get("target", "")
                        ).lower()

            if keyword_lower in searchable:
                result = {
                    "turn_number": i // 2 + 1,
                    "role": entry.get("role", "unknown"),
                    "content": content[:500],
                }
                if files:
                    result["files_created"] = files
                results.append(result)

        if results:
            print(
                f"Found {len(results)} history entries "
                f"matching '{keyword}'"
            )
        else:
            print(
                f"No history entries matching '{keyword}'"
            )
        return results

    return search_history


def _make_web_search(
    ns: dict[str, Any],
) -> Callable[..., list[dict[str, str]]]:
    """Create a ``web_search`` helper bound to *ns*.

    Uses DuckDuckGo (free, no API key) with fallback
    error handling. Results are tracked in state['_ops'].
    """

    def web_search(
        query: str, max_results: int = 5
    ) -> list[dict[str, str]]:
        """Search the web using DuckDuckGo.

        Args:
            query: Search query string.
            max_results: Maximum number of results (1-10).

        Returns:
            List of dicts with 'title', 'url', 'snippet'.
        """
        max_results = min(max(1, max_results), 10)
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                raw = list(
                    ddgs.text(query, max_results=max_results)
                )
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
            ns["state"]["_ops"].append(
                ("web_search", query, len(results))
            )
            # Print summary for the LLM to see in output
            for i, r in enumerate(results, 1):
                print(
                    f"{i}. {r['title']}\n"
                    f"   {r['url']}\n"
                    f"   {r['snippet'][:150]}\n"
                )
            return results
        except ImportError:
            print(
                "[web_search] ddgs not installed. "
                "Run: pip install ddgs"
            )
            return []
        except Exception as exc:
            print(f"[web_search error]: {exc}")
            ns["state"]["_ops"].append(
                ("web_search", query, f"error: {exc}")
            )
            return []

    return web_search


def _make_web_fetch(
    ns: dict[str, Any],
) -> Callable[..., str]:
    """Create a ``web_fetch`` helper bound to *ns*.

    Fetches a URL and returns content as markdown text.
    Uses httpx (sync) with markdownify for HTML conversion.
    """

    def web_fetch(
        url: str, max_length: int = 50_000
    ) -> str:
        """Fetch a URL and return its content as text.

        HTML is converted to markdown. Other content types
        are returned as-is (up to max_length).

        Args:
            url: The URL to fetch.
            max_length: Max chars to return.

        Returns:
            Content as a string.
        """
        try:
            import httpx

            resp = httpx.get(
                url,
                timeout=30,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Memfun-Agent/1.0 "
                        "(autonomous coding agent)"
                    )
                },
            )
            resp.raise_for_status()
            content_type = resp.headers.get(
                "content-type", ""
            )
            text = resp.text

            # Convert HTML to markdown
            if "html" in content_type:
                try:
                    from markdownify import markdownify

                    text = markdownify(
                        text,
                        heading_style="ATX",
                        strip=["img", "script", "style"],
                    )
                except ImportError:
                    # Basic fallback: strip tags
                    import re

                    text = re.sub(
                        r"<[^>]+>", "", text
                    )

            # Truncate
            if len(text) > max_length:
                text = (
                    text[:max_length]
                    + "\n... (truncated)"
                )

            ns["state"]["_ops"].append(
                ("web_fetch", url, len(text))
            )
            print(
                f"Fetched {url} ({len(text)} chars,"
                f" {content_type.split(';')[0]})"
            )
            return text
        except ImportError:
            print(
                "[web_fetch] httpx not installed."
                " Run: pip install httpx"
            )
            return ""
        except Exception as exc:
            print(f"[web_fetch error]: {exc}")
            ns["state"]["_ops"].append(
                ("web_fetch", url, f"error: {exc}")
            )
            return ""

    return web_fetch


def _clean_generated_code(code: str) -> str:
    """Strip markdown fences and common LLM artifacts from generated code.

    LLMs frequently wrap code in ```python ... ``` blocks.
    This must be removed before exec().
    """
    import re

    stripped = code.strip()

    # Remove markdown code fences: ```python ... ```
    # Handle ```python, ```py, ``` variants
    if stripped.startswith("```"):
        # Find the end of the opening fence line
        first_newline = stripped.find("\n")
        if first_newline == -1:
            return ""
        # Remove opening fence
        stripped = stripped[first_newline + 1:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()
            stripped = stripped[: stripped.rfind("```")]

    # Also handle case where code has multiple fenced blocks
    # (LLM sometimes outputs explanatory text + code blocks)
    if "```python" in stripped or "```py" in stripped:
        blocks: list[str] = []
        in_block = False
        current: list[str] = []
        for line in stripped.split("\n"):
            if re.match(r"^```(?:python|py)?\s*$", line.strip()):
                if in_block:
                    # Closing fence
                    blocks.append("\n".join(current))
                    current = []
                    in_block = False
                else:
                    # Opening fence
                    in_block = True
                continue
            if in_block:
                current.append(line)
        # If we found code blocks, use them
        if blocks:
            stripped = "\n\n".join(blocks)
        elif current:
            # Unclosed block
            stripped = "\n".join(current)

    return stripped.strip()


@dataclass(frozen=True, slots=True)
class _ExecOutput:
    """Internal normalized execution output."""

    stdout: str = ""
    stderr: str = ""
    success: bool = True
