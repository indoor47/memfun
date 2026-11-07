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
#: Re-asks of ONE loop step whose reply could not be parsed, before the loop gives up.
STEP_RETRIES = 2
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
    root: str | None = None,
) -> str:
    """Build a compact metadata string for the LLM about the context.

    Returns a human-readable summary containing type, length, and a
    short preview of the context value.
    """
    ctx_type = type(context).__name__
    ctx_len = len(context)

    if isinstance(context, bytes):
        try:
            preview = context[:preview_length].decode("utf-8", errors="replace")
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
    if root is not None:
        lines.append(f"WORKING DIRECTORY: {root}")
        lines.append(
            "  write_file/read_file/edit_file/list_files resolve relative "
            "paths against this directory and reject paths outside it. "
            "run_cmd starts commands in this directory but does NOT "
            "confine them (cd / absolute paths still reach elsewhere), so "
            "keep commands inside it. Do NOT use os.chdir(), raw open() "
            "or subprocess -- they are not rooted."
        )
        lines.append("")
    lines.append("Available helper functions in the REPL:")
    lines.append("  write_file(path: str, content: str) -> str  # Write a file, returns abs path")
    lines.append(
        "  read_file(path: str, offset: int | None = None, limit: int | None = None) -> str"
        "  # Whole file, or a line-numbered slice from line `offset` (1-based)"
    )
    lines.append(
        "  grep(pattern: str, path: str = '.', glob: str | None = None,"
        " ignore_case: bool = False, context: int = 0, max_results: int = 200) -> str"
        "  # ripgrep regex search; returns 'file:line: text' lines. Use this to find code"
    )
    lines.append(
        "  find_files(pattern: str, path: str = '.', max_results: int = 500) -> list[str]"
        "  # Files matching e.g. '**/*.py', newest first"
    )
    lines.append("  run_cmd(cmd: str) -> str  # Run shell command, returns stdout string")
    lines.append("  edit_file(path: str, old: str, new: str) -> bool  # Replace text in file")
    lines.append(
        "  list_files(path: str = '.', with_times: bool = False)"
        " -> list[str]"
        "  # Returns LIST of file paths (with_times adds dates)"
    )
    lines.append(
        "  llm_query(question: str, context: str) -> str  # Ask sub-LM, returns answer string"
    )
    lines.append(
        "  web_search(query: str, max_results: int = 5)"
        " -> list[dict]"
        "  # Search web, returns [{title, url, snippet}]"
    )
    lines.append(
        "  web_fetch(url: str, max_length: int = 50000) -> str  # Fetch URL content as string"
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
        "- Output ONLY pure Python code. NO markdown, NO ```python fences, NO explanatory text."
    )
    lines.append(
        "- list_files() returns a list[str]. Iterate with for-loop, do NOT call .split() on it."
    )
    lines.append("- read_file() returns a str. Use .split('\\n') to get lines.")
    lines.append("- Always use write_file() instead of raw open().")
    lines.append("- Always use run_cmd() instead of subprocess.")
    lines.append(
        "- sys.exit() (incl. argparse errors/--help) is caught and reported as an"
        " error; don't run __main__ blocks in the REPL, use run_cmd('python -m ...')."
    )
    lines.append(
        "- Use web_search() when you need current information "
        "from the internet (APIs, docs, prices, news, etc.)."
    )
    lines.append("- Use web_fetch() to read full content from a URL.")
    lines.append(
        "- You are fully autonomous: install deps, create "
        "files, run commands, deploy, verify — do everything."
    )
    lines.append("- Never tell the user to do something you can do.")
    lines.append("- Write production-quality code with proper structure and styling.")
    lines.append("- Always verify: check files exist, test the app.")

    return "\n".join(lines)


# ── Local REPL (fallback when no sandbox) ──────────────────────


def _system_exit_message(exc: SystemExit) -> str:
    """Explain a blocked ``sys.exit()`` to the model so it can self-correct."""
    return (
        f"\nYour code called sys.exit({exc.code!r}): this would terminate the "
        "whole program, so it was blocked. argparse calls sys.exit(2) on usage "
        "errors and sys.exit(0) after --help: wrap parse_args in "
        "try/except SystemExit, use parse_known_args, or test the function's "
        "return value directly instead of running the module's __main__ block "
        "(use run_cmd('python -m ...') to exercise the real CLI).\n"
    )


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
            timeout: NOT enforced in the local REPL (kept for interface
                compatibility). ``exec`` runs synchronously on the calling
                thread -- which is the asyncio event-loop thread when called
                from ``RLMModule`` -- so an infinite loop or a blocking call
                in agent code blocks the whole loop and no ``asyncio``
                timeout can fire. Use a real ``SandboxAdapter`` when that
                matters.

        Returns:
            REPLResult with captured stdout/stderr.

        ``SystemExit`` raised by the executed code (``sys.exit()``,
        ``exit()``, argparse usage errors and ``--help``) is caught and
        reported as a failed result: left alone it is a ``BaseException``
        that escapes ``asyncio.run`` and kills every coroutine in the
        process. ``KeyboardInterrupt`` (and subclasses such as signal-to-
        exception adapters) still propagates, so a real Ctrl-C / SIGTERM
        stops the program. ``os._exit()`` / ``os.abort()`` terminate the
        process without raising and cannot be intercepted here.
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
        except SystemExit as exc:
            # Attributed to the agent code: memfun installs no signal
            # handler that raises SystemExit (worker.py uses loop callbacks,
            # the RunPod kit raises a KeyboardInterrupt subclass).
            stderr_buf.write(traceback.format_exc())
            stderr_buf.write(_system_exit_message(exc))
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
    #: Stop after this many consecutive iterations that did nothing new (see
    #: :func:`_made_progress`).  ``0`` disables.  This, not ``max_iterations``, is the
    #: intended stop for an agent that is stuck.
    stall_steps: int = 12
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS
    preview_length: int = _PREVIEW_LENGTH
    max_history_chars: int = _MAX_HISTORY_CHARS
    verbose: bool = False
    enable_planning: bool = True
    # Per-task working directory for the REPL file/command helpers.
    # When set, ``write_file``/``read_file``/``edit_file``/``list_files``
    # resolve relative paths against it (and reject paths that resolve
    # outside it, symlinks included), ``run_cmd`` runs with ``cwd=root``,
    # and the path-scoped MCP bridge tools get their ``path`` argument
    # rooted the same way.  ``None`` keeps the legacy behaviour: paths
    # resolve against the *process* cwd at call time.  This is
    # deliberately not ``os.chdir`` / ``MEMFUN_PROJECT_ROOT`` -- both are
    # process-global and concurrent tasks need different roots.
    root: str | None = None


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
    #: Why the loop ended without a FINAL answer: ``"stalled: ..."`` or
    #: ``"max_iterations"``; empty when the agent finished normally.
    stop_reason: str = ""


def _made_progress(
    new_ops: list[tuple],
    seen_reads: set[str],
    cmds_since_edit: set[str],
) -> bool:
    """Did these newly recorded operations move the task forward?

    Progress is a write or edit, a command not yet run since the last write/edit, a
    file not read before, or any other tool call with arguments not seen before.
    Re-reading a file already read, re-running a command with no edit in between, or
    running nothing is not progress.  Mutates the two tracking sets.
    """
    progress = False
    for op in new_ops:
        kind = str(op[0])
        target = str(op[1]) if len(op) > 1 else ""
        if kind in ("write", "edit"):
            progress = True
            cmds_since_edit.clear()
        elif kind == "cmd":
            if target not in cmds_since_edit:
                progress = True
                cmds_since_edit.add(target)
        elif kind == "read":
            # A slice read (path, size, start) is a different read per start line.
            key = f"{target}@{op[3]}" if len(op) > 3 else target
            if key not in seen_reads:
                progress = True
                seen_reads.add(key)
        else:
            key = f"{kind}:{target}"
            if key not in seen_reads:
                progress = True
                seen_reads.add(key)
    return progress


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
                    self.aforward(query=query, context=context, **kwargs),
                )
                return future.result()

        return asyncio.run(self.aforward(query=query, context=context, **kwargs))

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
            root=_canonical_root(self.config.root),
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
        stop_reason = ""
        seen_ops = 0
        idle_steps = 0
        seen_reads: set[str] = set()
        cmds_since_edit: set[str] = set()

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
                    plan_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))
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
                        + code_history[-self.config.max_history_chars :]
                    )

                # Ask LLM for next exploration step.  A reply the adapter cannot parse
                # (AdapterParseError) used to end the whole loop; retry the STEP with a
                # fresh sample (rollout_id changes DSPy's cache key) up to STEP_RETRIES times.
                prediction = None
                for step_try in range(STEP_RETRIES + 1):
                    try:
                        prediction = await asyncio.to_thread(
                            self.explorer,
                            context_metadata=ctx_meta,
                            query=query,
                            code_history=code_history or "(no history yet)",
                            **(
                                {"config": {"rollout_id": 1000 * iteration + step_try}}
                                if step_try
                                else {}
                            ),
                        )
                        break
                    except Exception as step_exc:
                        if step_try >= STEP_RETRIES:
                            raise
                        logger.warning(
                            "RLM step %d: model reply unusable (%s); retry %d/%d",
                            iteration,
                            type(step_exc).__name__,
                            step_try + 1,
                            STEP_RETRIES,
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
                            total_prompt_tokens += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get("completion_tokens", 0)
                        elif usage:
                            total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                            total_completion_tokens += getattr(usage, "completion_tokens", 0)
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
                exec_result = await self._execute_code(repl, next_code)
                step_ms = (time.perf_counter() - step_start) * 1000.0

                # Build output string
                output = exec_result.stdout
                output_truncated = False
                if self.config.max_output_chars > 0 and len(output) > self.config.max_output_chars:
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
                    cumulative_tokens=(total_prompt_tokens + total_completion_tokens),
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
                        if current_plan_step >= 0 and self._on_step_status is not None:
                            try:
                                self._on_step_status(
                                    current_plan_step,
                                    "completed",
                                    reasoning[:120],
                                )
                            except Exception:
                                logger.debug(
                                    "on_step_status callback failed",
                                    exc_info=True,
                                )
                        # Mark new step in-progress
                        current_plan_step = new_step
                        if self._on_step_status is not None:
                            try:
                                detail = plan[new_step] if new_step < len(plan) else ""
                                self._on_step_status(
                                    new_step,
                                    "in_progress",
                                    detail,
                                )
                            except Exception:
                                logger.debug(
                                    "on_step_status callback failed",
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
                    writes = [o[1] for o in ops_so_far if o[0] == "write"]
                    edits = [o[1] for o in ops_so_far if o[0] == "edit"]
                    reads = [o[1] for o in ops_so_far if o[0] == "read"]
                    cmds = [o[1] for o in ops_so_far if o[0] == "cmd"]
                    web_searches = [o[1] for o in ops_so_far if o[0] == "web_search"]
                    web_fetches = [o[1] for o in ops_so_far if o[0] == "web_fetch"]
                    status_parts = []
                    if reads:
                        # Show unique files read.
                        unique_reads = list(dict.fromkeys(reads))
                        status_parts.append(f"Files read: {', '.join(unique_reads[-5:])}")
                    if writes:
                        status_parts.append(f"Files written: {', '.join(writes[-5:])}")
                    if edits:
                        unique_edits = list(dict.fromkeys(edits))
                        status_parts.append(f"Files edited: {', '.join(unique_edits[-5:])}")
                    if cmds:
                        status_parts.append(f"Commands run: {', '.join(cmds[-5:])}")
                    if web_searches:
                        status_parts.append(f"Web searches: {', '.join(web_searches[-3:])}")
                    if web_fetches:
                        status_parts.append(f"URLs fetched: {len(web_fetches)}")
                    code_history += f"[State: {'; '.join(status_parts)}]\n"

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
                    action_ops = [o for o in ops_so_far if o[0] in ("write", "edit", "cmd")]
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
                        read_ops = [o[1] for o in ops_so_far if o[0] == "read"]
                        if len(read_ops) >= 4:
                            from collections import Counter

                            read_counts = Counter(read_ops)
                            hot_files = [(f, c) for f, c in read_counts.most_common(3) if c >= 3]
                            if hot_files:
                                names = ", ".join(
                                    f"{f.rsplit('/', 1)[-1]} ({c}x)" for f, c in hot_files
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
                    answer = str(repl.namespace.get(var_name, ""))
                    break

                # Stall stop: nothing new for ``stall_steps`` iterations in a row.
                if self.config.stall_steps > 0:
                    ops_now = state.get("_ops", [])
                    if _made_progress(ops_now[seen_ops:], seen_reads, cmds_since_edit):
                        idle_steps = 0
                    else:
                        idle_steps += 1
                    seen_ops = len(ops_now)
                    if idle_steps >= self.config.stall_steps:
                        stop_reason = (
                            f"stalled: {idle_steps} consecutive iterations with no new file "
                            "read, edit, write or command"
                        )
                        logger.warning("RLM STOPPED at iteration %d (%s)", iteration, stop_reason)
                        answer = f"[STOPPED: {stop_reason}]\n" + self._auto_summary(state)
                        break
            else:
                # Max iterations reached without FINAL
                logger.warning(
                    "RLM reached max iterations (%d) without FINAL answer",
                    self.config.max_iterations,
                )
                stop_reason = "max_iterations"
                answer = self._auto_summary(repl.namespace.get("state", {}))

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
                        self._on_step_status(current_plan_step, "completed", "")
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
        final_files = list(set(final_state.get("_files", [])))

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
                total_tokens=(total_prompt_tokens + total_completion_tokens),
                sub_lm_calls=sub_lm_calls,
                sub_lm_tokens=sub_lm_tokens,
            ),
            plan=plan,
            ops=final_ops,
            files_created=final_files,
            stop_reason=stop_reason,
        )

    def _auto_summary(self, state: dict[str, Any]) -> str:
        """Summary of tracked operations for a loop that ended without FINAL."""
        ops = state.get("_ops", [])
        files = state.get("_files", [])

        if ops or files:
            summary_parts: list[str] = []
            writes = [o for o in ops if o[0] == "write"]
            edits = [o for o in ops if o[0] == "edit"]
            cmds = [o for o in ops if o[0] == "cmd"]
            searches = [o for o in ops if o[0] == "web_search"]
            fetches = [o for o in ops if o[0] == "web_fetch"]

            if searches:
                summary_parts.append("**Web searches:**")
                for _, query, count in searches:
                    summary_parts.append(f"- `{query}` ({count} results)")
            if fetches:
                summary_parts.append("\n**URLs fetched:**")
                for _, url, size in fetches:
                    summary_parts.append(f"- `{url}` ({size})")
            if writes:
                summary_parts.append("\n**Files created:**")
                for _, path, size in writes:
                    summary_parts.append(f"- `{path}` ({size} chars)")
            if edits:
                summary_parts.append("\n**Files edited:**")
                for _, path, _ in edits:
                    summary_parts.append(f"- `{path}`")
            if cmds:
                summary_parts.append("\n**Commands run:**")
                for _, cmd, rc in cmds:
                    status = "ok" if rc == 0 else f"exit {rc}"
                    summary_parts.append(f"- `{cmd}` {status}")

            return "\n".join(summary_parts)
        else:
            # No tracked ops - try partial_answer
            return str(
                state.get(
                    "partial_answer",
                    state.get(
                        "FINAL",
                        "(max iterations reached without final answer)",
                    ),
                )
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
        repl.namespace["search_history"] = _make_search_history(history)

        # Inject llm_query for sub-LM calls
        repl.namespace["llm_query"] = self._make_llm_query()
        repl.namespace["llm_query_batched"] = self._make_llm_query_batched()

        root = _canonical_root(self.config.root)

        # Inject extra tools (e.g., MCP tool wrappers).  When a task
        # root is set, the bridge's path-scoped tools are wrapped so
        # their ``path`` argument resolves inside the root instead of
        # the process cwd (see ``_root_extra_tools``).
        extra_tools = (
            _root_extra_tools(self._extra_tools, root) if root is not None else self._extra_tools
        )
        for name, func in extra_tools.items():
            repl.namespace[name] = func

        # Standard library imports available in the REPL
        repl.execute("import os, re, json, collections, itertools, math, textwrap, subprocess")

        # Initialize operations tracker
        repl.namespace["state"]["_ops"] = []
        repl.namespace["state"]["_files"] = []

        # Inject robust helper functions
        repl.namespace["write_file"] = _make_write_file(repl.namespace, root=root)
        repl.namespace["read_file"] = _make_read_file(repl.namespace, root=root)
        repl.namespace["run_cmd"] = _make_run_cmd(repl.namespace, root=root)
        repl.namespace["edit_file"] = _make_edit_file(repl.namespace, root=root)
        repl.namespace["list_files"] = _make_list_files(root=root)
        repl.namespace["grep"] = _make_grep(repl.namespace, root=root)
        repl.namespace["find_files"] = _make_find_files(repl.namespace, root=root)
        if root is not None:
            repl.namespace["PROJECT_ROOT"] = root
        repl.namespace["web_search"] = _make_web_search(repl.namespace)
        repl.namespace["web_fetch"] = _make_web_fetch(repl.namespace)

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
            return [llm_query(q, c) for q, c in queries]

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
                result: ExecutionResult = await self._sandbox.execute(
                    code,
                    context=repl.namespace,
                    timeout=30,
                )
                return _ExecOutput(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    success=result.exit_code == 0,
                )
            except NotImplementedError:
                logger.debug("Sandbox not available, falling back to local REPL")
            except Exception as exc:
                logger.warning(
                    "Sandbox execution failed: %s, falling back to local REPL",
                    exc,
                )

        # Fallback: local REPL
        result_local = repl.execute(code)
        return _ExecOutput(
            stdout=result_local.stdout,
            stderr=result_local.stderr,
            success=result_local.success,
        )


# ── Task-root path resolution ──────────────────────────────────

# MCP bridge tools (see ``MCPToolBridge.get_repl_tools``) whose ``path``
# keyword names a file or directory.  Without wrapping they resolve
# against the process cwd (or ``MEMFUN_PROJECT_ROOT``), bypassing the
# per-task root.  ``read_file``/``write_file`` are listed for
# completeness even though the RLM's own rooted helpers shadow them.
_PATH_SCOPED_BRIDGE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "list_directory",
        "glob_files",
        "grep",
        "git_status",
        "git_diff",
        "git_log",
        "repo_map",
    }
)


class PathOutsideRootError(PermissionError):
    """Raised when a REPL helper path resolves outside the task root."""


def _canonical_root(root: str | None) -> str | None:
    """Return the symlink-resolved absolute form of *root* (or None)."""
    if root is None:
        return None
    from pathlib import Path

    return str(Path(root).resolve())


def _resolve_path(path: str, root: str | None) -> str:
    """Resolve *path* for a REPL helper.

    With ``root=None`` this is exactly the legacy behaviour:
    ``os.path.abspath(path)`` against the process cwd.

    With a root, relative paths are joined onto the root, the result is
    resolved with symlinks followed (``Path.resolve``), and anything
    landing outside the root -- ``../`` escapes, absolute paths
    elsewhere, symlinks inside the root pointing out -- raises
    :class:`PathOutsideRootError`.  *root* must already be canonical
    (see :func:`_canonical_root`).
    """
    import os as _os
    from pathlib import Path

    if root is None:
        return _os.path.abspath(path)

    root_path = Path(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_path):
        # Don't echo the resolved target: for a symlink it reveals where
        # outside the root it points.
        raise PathOutsideRootError(f"path outside task root: {path!r}")
    return str(resolved)


def _root_extra_tools(
    tools: dict[str, Callable[..., Any]],
    root: str,
) -> dict[str, Callable[..., Any]]:
    """Wrap path-scoped MCP bridge tools so ``path`` resolves in *root*.

    Only names in :data:`_PATH_SCOPED_BRIDGE_TOOLS` are wrapped; the
    bridge wrappers accept keyword arguments only.  A missing ``path``
    defaults to the root itself (the bridge tools default to ``"."``,
    i.e. the process cwd).  Escapes return a ``[tool error ...]`` string
    in the same shape the bridge uses for failures.

    Non-path arguments (``glob_files``'s ``pattern``, ``grep``'s ``glob``,
    ``git_diff``'s ``file``) are checked by the bridge itself (absolute and
    ``..`` patterns rejected) and the filesystem tools bound their matches
    to the root.  Tools outside the known set are passed through unwrapped.
    """
    wrapped: dict[str, Callable[..., Any]] = {}
    for name, func in tools.items():
        if name not in _PATH_SCOPED_BRIDGE_TOOLS:
            wrapped[name] = func
            continue
        wrapped[name] = _wrap_rooted_tool(name, func, root)
    return wrapped


def _wrap_rooted_tool(
    name: str,
    func: Callable[..., Any],
    root: str,
) -> Callable[..., Any]:
    def rooted_tool(**kwargs: Any) -> Any:
        raw = kwargs.get("path", ".")
        try:
            kwargs["path"] = _resolve_path(str(raw), root)
        except PathOutsideRootError as exc:
            return f"[tool error: {name}]: {exc}"
        return func(**kwargs)

    rooted_tool.__name__ = getattr(func, "__name__", name)
    rooted_tool.__doc__ = getattr(func, "__doc__", None)
    return rooted_tool


# ── REPL helper function factories ─────────────────────────────


def _make_write_file(
    ns: dict[str, Any],
    *,
    root: str | None = None,
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
        path = _resolve_path(path, root)

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

        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        ns["state"]["_ops"].append(("write", path, len(content)))
        ns["state"]["_files"].append(path)
        print(f"Wrote {path} ({len(content)} chars)")
        return path

    return write_file


def _make_read_file(
    ns: dict[str, Any],
    *,
    root: str | None = None,
) -> Callable[[str], str]:
    """Create a ``read_file`` helper bound to *ns*.

    Includes read caching: if the same file is read again and its
    content hasn't changed, returns a short summary instead of the
    full content.  This prevents the RLM from burning context by
    re-reading unchanged files every iteration.
    """
    import hashlib

    # Track {abs_path: content_md5} for cache invalidation.
    _read_cache: dict[str, str] = {}

    def read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read and return file contents.

        With *offset* (1-based first line) and/or *limit* (line count),
        returns just that slice with ``cat -n`` style line numbers, which
        is how to look at part of a large file found via ``grep``.

        Without them, if the file was already read and hasn't changed,
        returns a short summary instead of the full content.  The
        original content is still accessible as a Python variable from a
        prior iteration.
        """
        path = _resolve_path(path, root)
        with open(path, errors="replace") as fh:
            content = fh.read()

        if offset is not None or limit is not None:
            # Count lines exactly like grep (``\\n`` only), so a grep hit's
            # line number is the line read_file(offset=...) returns.
            all_lines = content.split("\n")
            if all_lines and all_lines[-1] == "":
                all_lines.pop()
            start = _as_int(offset, 1, low=1, high=10**9)
            count = _as_int(limit, _READ_SLICE_DEFAULT, low=0, high=10**6)
            if start > len(all_lines):
                note = f"[{path}: offset {start} is past the end ({len(all_lines)} lines)]"
                print(note)
                return note
            chunk = all_lines[start - 1 : start - 1 + count]
            # 4th element: the slice start, so paging through one file is not a re-read.
            ns["state"]["_ops"].append(("read", path, sum(len(x) for x in chunk), start))
            body = "\n".join(f"{start + i:6d}\t{text}" for i, text in enumerate(chunk))
            end = start + len(chunk) - 1
            note = f"[{path}: lines {start}-{end} of {len(all_lines)}]"
            print(note)
            return f"{note}\n{body}" if body else f"{note}\n(no lines in range)"

        content_hash = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()

        ns["state"]["_ops"].append(("read", path, len(content)))

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
    *,
    root: str | None = None,
) -> Callable[[str], str]:
    """Create a ``run_cmd`` helper bound to *ns*.

    With *root* set, commands run with ``cwd=root``.  This sets the
    starting directory only; it is not a sandbox (a command can still
    ``cd`` elsewhere or use absolute paths).
    """
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
            cwd=root if root is not None else _os.getcwd(),
        )
        ns["state"]["_ops"].append(("cmd", cmd, result.returncode))
        if result.stdout:
            print(result.stdout[:5000])
        if result.stderr:
            print(f"[stderr]: {result.stderr[:2000]}")
        return result.stdout

    return run_cmd


def _make_edit_file(
    ns: dict[str, Any],
    *,
    root: str | None = None,
) -> Callable[[str, str, str], bool]:
    """Create an ``edit_file`` helper bound to *ns*."""
    from memfun_agent.context_first import _fuzzy_find_and_replace

    def edit_file(path: str, old_text: str, new_text: str) -> bool:
        """Replace first occurrence of *old_text* in file.

        Uses fuzzy matching (whitespace normalization, then line-level
        difflib) when exact matching fails.

        Returns ``True`` if the replacement was made.
        """
        path = _resolve_path(path, root)
        with open(path) as fh:
            content = fh.read()

        new_content, ratio, snippet, strategy = _fuzzy_find_and_replace(
            content, old_text, new_text
        )

        if new_content is None:
            print(f"Text not found in {path} (best ratio={ratio:.2f}, strategy={strategy})")
            if snippet:
                print(f"  Closest match: {snippet[:200]}")
            return False

        if strategy != "exact":
            print(f"Fuzzy match in {path} (ratio={ratio:.2f}, strategy={strategy})")

        with open(path, "w") as fh:
            fh.write(new_content)
        ns["state"]["_ops"].append(("edit", path, len(new_content)))
        ns["state"]["_files"].append(path)
        print(f"Edited {path}")
        return True

    return edit_file


def _make_list_files(
    *,
    root: str | None = None,
) -> Callable[[str], list[str]]:
    """Create a ``list_files`` helper.

    With *root* set, *path* is resolved inside the root (escapes raise
    :class:`PathOutsideRootError`) but the returned entries keep the
    legacy shape ``os.path.join(path, <relative>)`` so ``list_files('.')``
    still yields ``./...`` entries, now relative to the root.
    """
    import os as _os

    def list_files(path: str = ".", with_times: bool = False) -> list[str]:
        """List all files recursively under *path*.

        Args:
            path: Directory to search.
            with_times: If True, returns 'path (modified: YYYY-MM-DD)'.

        Returns:
            Sorted list of file paths.
        """
        import datetime

        files: list[str] = []
        walk_dir = path if root is None else _resolve_path(path, root)
        for dirpath, dirs, filenames in _os.walk(walk_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for fname in filenames:
                real_path = _os.path.join(dirpath, fname)
                if root is None:
                    fpath = real_path
                else:
                    rel = _os.path.relpath(real_path, walk_dir)
                    fpath = _os.path.join(path, rel)
                if with_times:
                    try:
                        mtime = _os.path.getmtime(real_path)
                        dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                        fpath = f"{fpath} (modified: {dt})"
                    except OSError:
                        pass
                files.append(fpath)
        return sorted(files)

    return list_files


_READ_SLICE_DEFAULT = 2000
_GREP_TIMEOUT_S = 30
_SEARCH_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv"})


def _as_int(value: object, default: int, *, low: int, high: int) -> int:
    """Clamp an LLM-supplied number; None/garbage falls back to *default*."""
    try:
        n = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        n = default
    return max(low, min(high, n))


def _make_grep(
    ns: dict[str, Any],
    *,
    root: str | None = None,
) -> Callable[..., str]:
    """Create a ``grep`` helper: ripgrep search rooted like the file helpers.

    Uses ``rg`` when installed (respects .gitignore, skips binaries, does
    not follow symlinks) and a pure-Python fallback otherwise.  Results
    are ``path:line: text`` with paths relative to the search base,
    capped at *max_results* lines with an explicit truncation note.  rg's
    output is streamed and the process is stopped once the cap is
    reached, so a huge result never sits in memory.  Never goes through
    a shell; the pattern is passed as an argument.
    """
    import os as _os
    import shutil
    import subprocess as _sp
    import time as _time

    def grep(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        context: int = 0,
        max_results: int = 200,
    ) -> str:
        base = root if root is not None else _os.getcwd()
        target = _resolve_path(path, root)
        rel_target = _os.path.relpath(target, base)
        cap = _as_int(max_results, 200, low=1, high=5000)
        ctx = _as_int(context, 0, low=0, high=20)
        note = ""
        rg = shutil.which("rg")
        if rg:
            argv = [rg, "--line-number", "--with-filename", "--no-heading", "--color", "never"]
            argv += ["--max-columns", "400", "--max-columns-preview"]
            if ignore_case:
                argv.append("--ignore-case")
            if ctx:
                argv += ["--context", str(ctx)]
            if glob:
                argv += ["--glob", str(glob)]
            argv += ["-e", str(pattern)]
            if rel_target != ".":
                # Searching "." would print "./"-prefixed paths; omitting the
                # path makes rg search cwd (= base) with plain relative paths.
                argv += ["--", rel_target]
            lines, more, err, code = _stream_capped(argv, base, cap, _sp, _time)
            if "No files were searched" in err:
                err, code = "", 1  # empty tree / everything ignored: no matches
            if code not in (0, 1) and not lines:
                return f"[grep error: {err.strip()[:500] or f'rg exited {code}'}]"
            if err.strip() and lines:
                note = f"\n[some files could not be searched: {err.strip().splitlines()[0][:200]}]"
        else:
            all_lines = _python_grep(
                pattern, target, base, root=root, glob=glob, ignore_case=ignore_case
            )
            if all_lines and all_lines[0].startswith("[grep error:"):
                return all_lines[0]
            lines, more = all_lines[:cap], max(0, len(all_lines) - cap)
        ns["state"]["_ops"].append(("grep", pattern, len(lines) + max(more, 0)))
        if not lines:
            return f"No matches for {pattern!r} in {rel_target}"
        out = "\n".join(lines)
        if more:
            out += (
                f"\n[truncated: {more if more > 0 else 'more'} more lines;"
                " narrow path/glob or raise max_results]"
            )
        print(f"grep {pattern!r}: {len(lines)} line(s) shown")
        return out + note

    return grep


def _stream_capped(
    argv: list[str], cwd: str, cap: int, _sp: Any, _time: Any
) -> tuple[list[str], int, str, int]:
    """Run *argv*, keep at most *cap* output lines, stop the process after.

    Returns ``(lines, more, stderr, returncode)`` where ``more`` is ``-1``
    when the process was stopped early (count unknown) and the number of
    dropped lines otherwise.  Undecodable bytes are replaced, never fatal.
    """
    proc = _sp.Popen(
        argv,
        cwd=cwd,
        stdout=_sp.PIPE,
        stderr=_sp.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = _time.monotonic() + _GREP_TIMEOUT_S
    lines: list[str] = []
    more = 0
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line == "--":
            continue
        if len(lines) < cap:
            lines.append(line)
        else:
            more = -1  # stop early; exact remainder unknown
            break
        if _time.monotonic() > deadline:
            more = -1
            break
    if more == -1:
        proc.kill()
    try:
        _, err = proc.communicate(timeout=5)
    except _sp.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    return lines, more, err or "", proc.returncode if more != -1 else 0


def _python_grep(
    pattern: str,
    target: str,
    base: str,
    *,
    root: str | None,
    glob: str | None,
    ignore_case: bool,
) -> list[str]:
    """Fallback search when ripgrep is not installed (no context lines).

    Like rg it does not follow symlinks; with a task root, anything that
    resolves outside it is skipped.  Undecodable bytes are replaced.
    """
    import fnmatch
    import os as _os
    import re
    from pathlib import Path

    try:
        rx = re.compile(str(pattern), re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return [f"[grep error: invalid regex: {exc}]"]
    files: list[str] = []
    if _os.path.isfile(target):
        files = [target]
    else:
        for dirpath, dirs, filenames in _os.walk(target):
            dirs[:] = [d for d in dirs if d not in _SEARCH_SKIP_DIRS and not d.startswith(".")]
            files += [_os.path.join(dirpath, f) for f in filenames if not f.startswith(".")]
    out: list[str] = []
    for fpath in sorted(files):
        if _os.path.islink(fpath):
            continue
        if root is not None and not Path(fpath).resolve().is_relative_to(root):
            continue
        rel = _os.path.relpath(fpath, base)
        if glob and not (
            fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(_os.path.basename(rel), glob)
        ):
            continue
        try:
            with open(fpath, "rb") as fh:
                head = fh.read(8192)
                if b"\0" in head:
                    continue  # binary, like rg
                fh.seek(0)
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.split("\n"), 1):
            if rx.search(line):
                out.append(f"{rel}:{n}:{line.rstrip()[:400]}")
    return out


def _make_find_files(
    ns: dict[str, Any],
    *,
    root: str | None = None,
) -> Callable[..., list[str]]:
    """Create a ``find_files`` helper (Claude Code's Glob): newest first.

    Named ``find_files`` rather than ``glob`` so it can't clash with the
    stdlib ``glob`` module inside the REPL.  Patterns are relative
    (``**/*.py``); absolute patterns and ``..`` segments are refused, and
    with a task root every match must resolve inside it (symlinks
    pointing out are dropped).
    """
    import os as _os
    from pathlib import Path

    def find_files(pattern: str, path: str = ".", max_results: int = 500) -> list[str]:
        pattern = str(pattern or "").strip()
        if not pattern:
            return []
        if _os.path.isabs(pattern) or ".." in Path(pattern).parts:
            raise PathOutsideRootError(f"pattern must stay relative: {pattern!r}")
        cap = _as_int(max_results, 500, low=1, high=10000)
        base = root if root is not None else _os.getcwd()
        start = Path(_resolve_path(path, root))
        found: list[tuple[float, str]] = []
        for p in start.glob(pattern):
            if any(part in _SEARCH_SKIP_DIRS for part in p.relative_to(start).parts):
                continue
            if not p.is_file():
                continue
            real = p.resolve()
            if root is not None and not real.is_relative_to(root):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            found.append((mtime, _os.path.relpath(p, base)))
        found.sort(key=lambda t: (-t[0], t[1]))
        result = [f for _, f in found[:cap]]
        ns["state"]["_ops"].append(("find_files", pattern, len(result)))
        return result

    return find_files


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
                        searchable += " " + str(op.get("target", "")).lower()

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
            print(f"Found {len(results)} history entries matching '{keyword}'")
        else:
            print(f"No history entries matching '{keyword}'")
        return results

    return search_history


def _make_web_search(
    ns: dict[str, Any],
) -> Callable[..., list[dict[str, str]]]:
    """Create a ``web_search`` helper bound to *ns*.

    Uses DuckDuckGo (free, no API key) with fallback
    error handling. Results are tracked in state['_ops'].
    """

    def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
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
                raw = list(ddgs.text(query, max_results=max_results))
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
            ns["state"]["_ops"].append(("web_search", query, len(results)))
            # Print summary for the LLM to see in output
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:150]}\n")
            return results
        except ImportError:
            print("[web_search] ddgs not installed. Run: pip install ddgs")
            return []
        except Exception as exc:
            print(f"[web_search error]: {exc}")
            ns["state"]["_ops"].append(("web_search", query, f"error: {exc}"))
            return []

    return web_search


def _make_web_fetch(
    ns: dict[str, Any],
) -> Callable[..., str]:
    """Create a ``web_fetch`` helper bound to *ns*.

    Fetches a URL and returns content as markdown text.
    Uses httpx (sync) with markdownify for HTML conversion.
    """

    def web_fetch(url: str, max_length: int = 50_000) -> str:
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
                headers={"User-Agent": ("Memfun-Agent/1.0 (autonomous coding agent)")},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
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

                    text = re.sub(r"<[^>]+>", "", text)

            # Truncate
            if len(text) > max_length:
                text = text[:max_length] + "\n... (truncated)"

            ns["state"]["_ops"].append(("web_fetch", url, len(text)))
            print(f"Fetched {url} ({len(text)} chars, {content_type.split(';')[0]})")
            return text
        except ImportError:
            print("[web_fetch] httpx not installed. Run: pip install httpx")
            return ""
        except Exception as exc:
            print(f"[web_fetch error]: {exc}")
            ns["state"]["_ops"].append(("web_fetch", url, f"error: {exc}"))
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
        stripped = stripped[first_newline + 1 :]
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
