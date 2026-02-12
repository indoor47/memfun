"""Interactive chat interface for Memfun — Claude Code-style REPL.

Launch with ``memfun`` (no subcommand) or ``memfun chat``.  The agent
stays alive across turns, maintains conversation context via
file-based persistent history, and displays live progress for each
RLM iteration.

Key insight: Unlike traditional chat-based agents that need context
compaction (summarizing old turns to fit the LLM's window), Memfun's
RLM pattern stores conversation history as a Python variable in the
REPL namespace.  The LLM only sees metadata — it explores history by
writing code.  This means history can grow indefinitely with zero
information loss.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import select
import sys
import termios
import time
import tty
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memfun_core.config import MemfunConfig
from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

if TYPE_CHECKING:
    from memfun_agent.coding_agent import RLMCodingAgent
    from memfun_agent.traces import TraceStep
    from memfun_core.types import TaskResult
    from memfun_runtime.context import RuntimeContext

    from memfun_cli.learning import LearningManager

logger = get_logger("cli.chat")

console = Console()

# History safety limits
_MAX_HISTORY_ENTRIES = 10_000  # 5,000 turns
_MAX_HISTORY_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_SINGLE_MESSAGE_LENGTH = 100_000  # 100 KB per message

_MAX_FILE_READ_BYTES = 256_000
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".ex", ".exs", ".kt",
    ".swift", ".toml", ".yaml", ".yml", ".json", ".md", ".sh",
})
_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "dist", "build", ".venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})


# ── History persistence ───────────────────────────────────────────


def _history_file() -> Path:
    """Path to the current session's conversation file.

    Uses .memfun/conversations/ for session-based storage.
    Falls back to .memfun/conversation_history.json for compat.
    """
    convs_dir = Path.cwd() / ".memfun" / "conversations"
    if convs_dir.exists():
        latest = convs_dir / "latest.json"
        return latest
    # Fallback for backward compat
    p = Path.cwd() / ".memfun" / "conversation_history.json"
    parent = p.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            parent.chmod(0o700)
    return p


def _load_history() -> list[dict[str, Any]]:
    """Load full conversation history from disk.

    Enforces a maximum file size and maximum entry count to prevent
    unbounded memory consumption from a corrupted or inflated history file.
    """
    path = _history_file()
    if not path.exists():
        return []
    try:
        # Check file size before loading
        file_size = path.stat().st_size
        if file_size > _MAX_HISTORY_FILE_BYTES:
            logger.warning(
                "History file too large (%d bytes), truncating to last %d entries",
                file_size,
                _MAX_HISTORY_ENTRIES,
            )

        raw = path.read_text()
        data = json.loads(raw)

        # Validate structure
        if not isinstance(data, list):
            logger.warning("History file is not a list, starting fresh")
            return []

        # Cap to max entries (keep most recent)
        if len(data) > _MAX_HISTORY_ENTRIES:
            data = data[-_MAX_HISTORY_ENTRIES:]

        # Validate each entry is a dict with expected keys
        # Preserve extra metadata (ops, files, plan, etc.)
        validated: list[dict[str, Any]] = []
        for entry in data:
            if (
                isinstance(entry, dict)
                and "role" in entry
                and "content" in entry
            ):
                role = str(entry["role"])
                if role not in ("user", "assistant"):
                    continue
                clean: dict[str, Any] = {
                    "role": role,
                    "content": str(entry["content"]),
                }
                # Preserve rich metadata for assistant turns
                if role == "assistant":
                    for key in (
                        "method", "iterations", "plan",
                        "ops", "files_created",
                        "trajectory_length",
                        "final_reasoning",
                    ):
                        if key in entry:
                            clean[key] = entry[key]
                validated.append(clean)

        return validated
    except Exception:
        logger.debug("Failed to load history, starting fresh", exc_info=True)
        return []


def _save_history(history: list[dict[str, Any]]) -> None:
    """Persist conversation history to disk."""
    if len(history) > _MAX_HISTORY_ENTRIES:
        history = history[-_MAX_HISTORY_ENTRIES:]

    path = _history_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2)
        )
    except Exception:
        logger.debug("Failed to save history", exc_info=True)


# ── Context scanning ──────────────────────────────────────────────


def _read_file(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
        if len(text) > _MAX_FILE_READ_BYTES:
            text = text[:_MAX_FILE_READ_BYTES] + "\n... (truncated)"
        return text
    except Exception as exc:
        return f"(Error reading {path}: {exc})"


def _scan_cwd_context(cwd: Path) -> str:
    """Build a project context string from *cwd*."""
    parts: list[str] = []
    for name in ("README.md", "pyproject.toml", "package.json", "Cargo.toml"):
        candidate = cwd / name
        if candidate.exists():
            parts.append(f"--- {name} ---\n{_read_file(candidate)}")

    source_files: list[Path] = []
    for child in sorted(cwd.rglob("*")):
        if child.is_file() and child.suffix in _SOURCE_EXTENSIONS:
            rel = child.relative_to(cwd)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if any(p in _SKIP_DIRS for p in rel.parts):
                continue
            source_files.append(child)
            if len(source_files) >= 10:
                break

    for sf in source_files:
        rel = sf.relative_to(cwd)
        parts.append(f"--- {rel} ---\n{_read_file(sf)}")

    return "\n\n".join(parts) if parts else "(No readable source files)"


# ── Slash command handlers ─────────────────────────────────────────


_HELP_TEXT = """\
[bold]Commands:[/bold]

  /help      Show this help
  /exit      Exit (or Ctrl+D, or type 'exit')
  /clear     Clear conversation history
  /context   Rescan project files
  /history   Show history stats
  /model     Show current LLM model
  /traces    List recent traces
  /remember  Remember a preference or fact
  /memory    View current memory contents
  /forget    Forget a memory entry
  /debug-learning  Test the learning extraction pipeline

[dim]Esc or Ctrl+C cancels a running request.[/dim]
"""

# Models available for switching via /model
_AVAILABLE_MODELS: dict[str, list[tuple[str, str]]] = {
    "anthropic": [
        ("claude-opus-4-6", "Opus 4.6 — most intelligent"),
        ("claude-sonnet-4-5", "Sonnet 4.5 — fast + smart"),
        ("claude-haiku-4-5", "Haiku 4.5 — fastest"),
    ],
    "openai": [
        ("gpt-4.1", "GPT-4.1 — smartest non-reasoning"),
        ("gpt-4.1-mini", "GPT-4.1 Mini — fast"),
        ("gpt-4.1-nano", "GPT-4.1 Nano — cheapest"),
        ("o3", "o3 — reasoning"),
        ("o4-mini", "o4-mini — fast reasoning"),
    ],
    "ollama": [
        ("llama3.1", "Llama 3.1"),
        ("codellama", "Code Llama"),
        ("mistral", "Mistral"),
    ],
}

_SLASH_COMPLETER = WordCompleter(
    [
        "/help", "/exit", "/quit", "/clear",
        "/context", "/history", "/model", "/traces",
        "/remember", "/memory", "/forget",
        "/debug-learning",
    ],
    sentence=True,
)


# ── ChatSession ────────────────────────────────────────────────────


class ChatSession:
    """Manages the persistent agent and conversation state.

    Conversation history is persisted to a JSON file and injected
    into the RLM REPL namespace as the ``conversation_history``
    variable.  The RLM pattern means the LLM explores this history
    by writing Python code — it never needs to fit the full history
    into its token window, so no compaction is ever required.
    """

    def __init__(self) -> None:
        self._config: MemfunConfig | None = None
        self._runtime: RuntimeContext | None = None
        self._agent: RLMCodingAgent | None = None
        self._learning_manager: LearningManager | None = None
        self._history: list[dict[str, Any]] = []
        self._cwd_context: str = ""
        self._last_step: TraceStep | None = None
        self._completed_steps: list[TraceStep] = []
        self._plan: list[str] = []
        self._plan_status: dict[int, str] = {}
        self._plan_detail: dict[int, str] = {}
        self._plan_step_start: dict[int, float] = {}
        self._plan_step_duration: dict[int, float] = {}

    async def start(self) -> None:
        """Load config, build runtime, create agent, scan cwd."""
        from memfun_agent.coding_agent import RLMCodingAgent
        from memfun_runtime.builder import RuntimeBuilder

        from memfun_cli.learning import LearningManager

        self._config = MemfunConfig.load()

        # Configure DSPy LM before creating the agent
        _configure_dspy(self._config)

        self._runtime = await RuntimeBuilder(self._config).build()
        self._agent = RLMCodingAgent(self._runtime)
        self._agent.on_step = self._on_step_callback
        self._agent.on_plan = self._on_plan_callback
        self._agent.on_step_status = self._on_step_status_callback
        await self._agent.on_start()

        # Initialize persistent learning memory
        self._learning_manager = LearningManager(
            self._runtime.state_store
        )

        # Load persistent conversation history
        self._history = _load_history()

        self._cwd_context = _scan_cwd_context(Path.cwd())
        logger.info(
            "Chat session started (history: %d turns)", len(self._history) // 2
        )

    def _on_step_callback(self, step: TraceStep) -> None:
        """Called by RLM after each iteration — stored for live display."""
        # Archive previous step as completed
        if self._last_step is not None:
            self._completed_steps.append(self._last_step)
        self._last_step = step

    def _on_plan_callback(self, steps: list[str]) -> None:
        """Called when the RLM generates an execution plan."""
        self._plan = steps
        self._plan_status = {i: "pending" for i in range(len(steps))}
        self._plan_detail = {}

    def _on_step_status_callback(
        self, step_num: int, status: str, detail: str
    ) -> None:
        """Called when a plan step changes status."""
        self._plan_status[step_num] = status
        if detail:
            self._plan_detail[step_num] = detail
        if status == "in_progress":
            self._plan_step_start[step_num] = time.monotonic()
        elif status == "completed":
            start = self._plan_step_start.get(step_num)
            if start is not None:
                self._plan_step_duration[step_num] = (
                    time.monotonic() - start
                )

    async def chat_turn(self, user_input: str) -> TaskResult:
        """Process one user message and return the agent's response."""
        assert self._agent is not None

        # Rescan CWD so context reflects files created by prior turns
        self._cwd_context = _scan_cwd_context(Path.cwd())

        # Build context with CLEAR PRIORITY: learnings first,
        # then current state, then recent work, then older history.
        context_parts: list[str] = []

        # 0. LEARNED PREFERENCES (highest priority)
        if self._learning_manager is not None:
            try:
                learnings_section = (
                    await self._learning_manager
                    .get_relevant_learnings(user_input)
                )
                if learnings_section:
                    context_parts.append(learnings_section)
            except Exception:
                logger.debug(
                    "Failed to retrieve learnings",
                    exc_info=True,
                )

        # 1. CURRENT PROJECT STATE
        context_parts.append(
            f"=== CURRENT PROJECT STATE ===\n"
            f"{self._cwd_context}"
        )

        # 2. MOST RECENT TURN (what was just done)
        if len(self._history) >= 2:
            last_asst = self._history[-1]
            if last_asst.get("role") == "assistant":
                last_user = self._history[-2]
                recent_summary = (
                    f"=== LAST COMPLETED TASK ===\n"
                    f"User asked: "
                    f"{str(last_user.get('content', ''))[:300]}"
                )
                files = last_asst.get("files_created", [])
                if files:
                    recent_summary += (
                        f"\nFiles created: "
                        f"{', '.join(files[:15])}"
                    )
                ops = last_asst.get("ops", [])
                cmds = [
                    o["target"]
                    for o in ops
                    if o.get("type") == "cmd"
                ]
                if cmds:
                    recent_summary += (
                        f"\nCommands run: "
                        f"{', '.join(cmds[:10])}"
                    )
                content = str(
                    last_asst.get("content", "")
                )
                if content:
                    recent_summary += (
                        f"\nAgent response: {content[:2000]}"
                    )
                context_parts.append(recent_summary)

        # 3. OLDER HISTORY (reference only, lower priority)
        n_turns = len(self._history) // 2
        if n_turns > 1:
            # Summarize older turns compactly
            older = self._history[:-2] if len(self._history) > 2 else []
            if older:
                older_lines: list[str] = []
                # Only last 4 older entries (2 turns)
                for entry in older[-4:]:
                    role = entry.get("role", "user")
                    content = str(entry.get("content", ""))
                    tag = "User" if role == "user" else "Agent"
                    older_lines.append(
                        f"  {tag}: {content[:200]}"
                    )
                    if role == "assistant":
                        files = entry.get(
                            "files_created", []
                        )
                        if files:
                            older_lines.append(
                                f"    [files: "
                                f"{', '.join(files[:5])}]"
                            )
                context_parts.append(
                    f"=== EARLIER CONTEXT "
                    f"(reference only) ===\n"
                    f"{n_turns} total turns. "
                    f"Recent:\n"
                    + "\n".join(older_lines)
                )

        # 4. History variable pointer
        if self._history:
            context_parts.append(
                f"conversation_history variable has "
                f"{len(self._history)} entries with full "
                f"details (injected in REPL namespace). "
                f"Use search_history('keyword') to find "
                f"past discussions. Use list_files() and "
                f"read_file() to review created files."
            )

        context = "\n\n".join(context_parts)

        task_msg = TaskMessage(
            task_id=uuid.uuid4().hex,
            agent_id=self._agent.agent_id,
            payload={
                "type": "ask",
                "query": user_input,
                "context": context,
                "conversation_history": self._history,
            },
        )

        self._last_step = None
        self._completed_steps = []
        self._plan = []
        self._plan_status = {}
        self._plan_detail = {}
        self._plan_step_start = {}
        self._plan_step_duration = {}
        result = await self._agent.handle(task_msg)

        # Append to persistent history with rich metadata
        data = result.result or {}
        answer = data.get("answer", data.get("explanation", ""))
        truncated_input = user_input[:_MAX_SINGLE_MESSAGE_LENGTH]
        truncated_answer = answer[:_MAX_SINGLE_MESSAGE_LENGTH]
        self._history.append(
            {"role": "user", "content": truncated_input}
        )

        # Store full turn data so RLM can reference prior work
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": truncated_answer,
            "method": data.get("method", ""),
            "iterations": data.get("iterations", 0),
        }
        # Include plan steps
        if self._plan:
            assistant_entry["plan"] = self._plan
        # Include operations performed
        ops = data.get("ops", [])
        if ops:
            assistant_entry["ops"] = ops
        files = data.get("files_created", [])
        if files:
            assistant_entry["files_created"] = files
        # Include trajectory summary (compact)
        traj_len = data.get("trajectory_length", 0)
        if traj_len:
            assistant_entry["trajectory_length"] = traj_len
        reasoning = data.get("final_reasoning", "")
        if reasoning:
            assistant_entry["final_reasoning"] = (
                reasoning[:500]
            )

        self._history.append(assistant_entry)

        # Persist to disk after every turn
        _save_history(self._history)

        # Extract learnings from this turn
        if self._learning_manager is not None:
            try:
                stored = (
                    await self._learning_manager
                    .extract_and_store(
                        user_input, truncated_answer
                    )
                )
                if stored:
                    logger.info(
                        "Learned %d items: %s",
                        len(stored),
                        ", ".join(
                            s[:40] for s in stored
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "Learning extraction failed: %s",
                    exc,
                    exc_info=True,
                )

        return result

    def clear_history(self) -> None:
        self._history.clear()
        _save_history(self._history)

    def rescan_context(self) -> None:
        self._cwd_context = _scan_cwd_context(Path.cwd())

    @property
    def history_stats(self) -> str:
        n = len(self._history) // 2
        path = _history_file()
        size = path.stat().st_size if path.exists() else 0
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        return f"{n} turns, {size_str} on disk"

    @property
    def model_name(self) -> str:
        if self._config:
            return self._config.llm.model
        return "unknown"

    @property
    def provider(self) -> str:
        if self._config:
            return self._config.llm.provider
        return "anthropic"

    def switch_model(self, new_model: str) -> None:
        """Switch the active LLM model and reconfigure DSPy."""
        if self._config is None:
            return
        # Rebuild config with new model
        from dataclasses import replace

        new_llm = replace(self._config.llm, model=new_model)
        self._config = replace(self._config, llm=new_llm)
        _configure_dspy(self._config)

    @property
    def last_step(self) -> TraceStep | None:
        return self._last_step

    @property
    def completed_steps(self) -> list[TraceStep]:
        return self._completed_steps

    @property
    def plan(self) -> list[str]:
        return self._plan

    @property
    def plan_status(self) -> dict[int, str]:
        return self._plan_status

    @property
    def plan_detail(self) -> dict[int, str]:
        return self._plan_detail

    @property
    def plan_step_duration(self) -> dict[int, float]:
        return self._plan_step_duration

    async def list_traces(self) -> list[str]:
        """List recent trace IDs from the state store."""
        if self._runtime is None:
            return []
        keys = await self._runtime.state_store.list_keys("memfun:traces:*")
        return [k.removeprefix("memfun:traces:") for k in keys[-10:]]

    async def stop(self) -> None:
        if self._agent:
            await self._agent.on_stop()
        logger.info("Chat session ended")


# ── Display helpers ────────────────────────────────────────────────


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time: '12.3s' under 60s, '2m 40s' above."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s"


def _format_tokens(tokens: int) -> str:
    """Format token count: '1.2k', '12.5k', '150k'."""
    if tokens <= 0:
        return ""
    if tokens < 1000:
        return str(tokens)
    return f"{tokens / 1000:.1f}k"


def _display_ops_summary(
    ops: list[dict], files_created: list[str]
) -> None:
    """Show a summary of file and shell operations performed.

    Each item gets an empty line after it for vertical breathing room.
    """
    if not ops and not files_created:
        return

    writes = [o for o in ops if o.get("type") == "write"]
    edits = [o for o in ops if o.get("type") == "edit"]
    cmds = [o for o in ops if o.get("type") == "cmd"]
    searches = [
        o for o in ops if o.get("type") == "web_search"
    ]
    fetches = [
        o for o in ops if o.get("type") == "web_fetch"
    ]

    if searches or fetches:
        console.print()
        console.print("  [bold]Web:[/bold]")
        console.print()
        for op in searches:
            query = op.get("target", "")
            count = op.get("detail", 0)
            console.print(
                f"    [magenta]?[/magenta] "
                f"search: {query}"
                f" [dim]({count} results)[/dim]"
            )
            console.print()
        for op in fetches:
            url = op.get("target", "")
            size = op.get("detail", 0)
            console.print(
                f"    [magenta]>[/magenta] "
                f"fetch: {url}"
                f" [dim]({size})[/dim]"
            )
            console.print()

    if writes or edits:
        console.print()
        console.print("  [bold]Files:[/bold]")
        console.print()
        seen: set[str] = set()
        for op in writes:
            path = op.get("target", "")
            if path and path not in seen:
                seen.add(path)
                size = op.get("detail", 0)
                console.print(
                    f"    [green]✓[/green] {path}"
                    f" [dim]({size} chars)[/dim]"
                )
                console.print()
        for op in edits:
            path = op.get("target", "")
            if path and path not in seen:
                seen.add(path)
                console.print(
                    f"    [cyan]~[/cyan] {path}"
                    f" [dim](edited)[/dim]"
                )
                console.print()

    if cmds:
        console.print()
        console.print("  [bold]Commands:[/bold]")
        console.print()
        for op in cmds:
            cmd = op.get("target", "")
            rc = op.get("detail", 0)
            if rc == 0:
                console.print(
                    f"    [green]✓[/green] $ {cmd}"
                )
            else:
                console.print(
                    f"    [red]✗[/red] $ {cmd}"
                    f" [dim](exit {rc})[/dim]"
                )
            console.print()

    # Suggest next steps if files were created
    if files_created:
        console.print()
        console.print("  [bold]Next steps:[/bold]")
        console.print()
        # Check for common patterns
        has_py = any(
            f.endswith(".py") for f in files_created
        )
        has_html = any(
            f.endswith(".html") for f in files_created
        )
        has_requirements = any(
            "requirements" in f for f in files_created
        )
        has_package_json = any(
            "package.json" in f for f in files_created
        )

        if has_requirements:
            console.print(
                "    [dim]→[/dim]"
                " pip install -r requirements.txt"
            )
            console.print()
        if has_package_json:
            console.print(
                "    [dim]→[/dim] npm install"
            )
            console.print()
        if has_py:
            mains = [
                f for f in files_created
                if any(
                    n in f
                    for n in (
                        "app.py", "main.py", "server.py",
                    )
                )
            ]
            for m in mains:
                console.print(
                    f"    [dim]→[/dim] python {m}"
                )
                console.print()
        if has_html:
            htmls = [
                f for f in files_created
                if f.endswith(".html")
            ]
            if htmls:
                console.print(
                    f"    [dim]→[/dim] open {htmls[0]}"
                )
                console.print()


def _display_answer(result: TaskResult) -> None:
    """Render the agent's answer with operations summary."""
    if not result.success:
        error = result.error or "Unknown error"
        console.print(f"\n  [red]✗ Error: {error}[/red]\n")
        return

    data = result.result or {}
    answer = data.get("answer", data.get("explanation", ""))
    method = data.get("method", "")
    iterations = data.get("iterations", 0)
    duration = (
        _format_elapsed(result.duration_ms / 1000)
        if result.duration_ms
        else ""
    )
    ops = data.get("ops", [])
    files_created = data.get("files_created", [])
    total_tokens = data.get("total_tokens", 0)

    # Build metadata line
    meta_parts: list[str] = []
    if method:
        meta_parts.append(method)
    if iterations:
        meta_parts.append(f"{iterations} iter")
    if duration:
        meta_parts.append(duration)
    tok_str = _format_tokens(total_tokens)
    if tok_str:
        meta_parts.append(f"{tok_str} tokens")
    meta = ", ".join(meta_parts)

    # Detect max-iterations failure
    is_max_iter = answer.startswith("(max iterations")

    if answer and not is_max_iter:
        console.print()
        console.print(
            f"  [green]✓[/green] [dim]({meta})[/dim]"
        )
        console.print()
        console.print(Markdown(answer))
        console.print()
    else:
        # No final answer or max iterations reached
        console.print()
        console.print(
            f"  [yellow]⚠[/yellow] Task completed"
            f" [dim]({meta})[/dim]"
        )
        console.print()
        traj_len = data.get("trajectory_length", 0)
        if traj_len:
            console.print(
                f"  [dim]{traj_len} iterations"
                f" executed.[/dim]"
            )
            console.print()
        final_reasoning = data.get("final_reasoning", "")
        if final_reasoning:
            console.print(
                f"  [dim]Last: {final_reasoning}[/dim]"
            )
            console.print()

    # Show operations summary
    _display_ops_summary(ops, files_created)
    console.print()


def _make_progress_renderable(
    session: ChatSession, elapsed: float
) -> Text:
    """Build a minimal live-updating spinner for the current step.

    Completed plan steps are printed permanently via
    ``_print_completed_steps``. This function only renders the
    *active* state: current step spinner, reasoning, and timer.
    The timer is always the last line.
    """
    frame = _SPINNER[int(elapsed * 8) % len(_SPINNER)]
    lines: list[str] = [""]  # leading blank line for spacing

    step = session.last_step

    if step is None:
        lines.append(
            f"  {frame} Thinking..."
            f"  [dim]({_format_elapsed(elapsed)})[/dim]"
        )
        lines.append("")
        return Text.from_markup("\n".join(lines))

    # Show reasoning (full sentences, no truncation)
    if step.reasoning:
        for rline in step.reasoning.split(". ")[:4]:
            rline = rline.strip()
            if rline:
                lines.append(f"    [dim]│[/dim] {rline}")
        lines.append("")

    # Show key operations from current code
    if step.code:
        shown = 0
        for code_line in step.code.split("\n"):
            if shown >= 6:
                break
            stripped = code_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            file_ops = (
                "write_file(", "read_file(",
                "edit_file(", "open(",
            )
            shell_ops = (
                "run_cmd(", "subprocess",
                "os.system",
            )
            web_ops = (
                "web_search(", "web_fetch(",
            )

            if any(op in stripped for op in file_ops):
                lines.append(
                    f"    [dim]│ >[/dim]"
                    f" [cyan]{stripped}[/cyan]"
                )
                lines.append("")
                shown += 1
            elif any(op in stripped for op in shell_ops):
                lines.append(
                    f"    [dim]│ $[/dim]"
                    f" [yellow]{stripped}[/yellow]"
                )
                lines.append("")
                shown += 1
            elif any(op in stripped for op in web_ops):
                lines.append(
                    f"    [dim]│ ?[/dim]"
                    f" [magenta]{stripped}[/magenta]"
                )
                lines.append("")
                shown += 1
            elif "state[" in stripped:
                lines.append(
                    f"    [dim]│ =[/dim]"
                    f" [green]{stripped}[/green]"
                )
                lines.append("")
                shown += 1

    # Show output or errors (no truncation)
    if step.output:
        out_lines = step.output.strip().split("\n")
        has_error = "[stderr]" in step.output
        if has_error:
            lines.append("")
            for ol in out_lines[-4:]:
                ol = ol.strip()
                if ol:
                    lines.append(
                        f"    [dim]│[/dim]"
                        f" [red]{ol}[/red]"
                    )
            lines.append("")
        else:
            for ol in out_lines[:3]:
                ol = ol.strip()
                if ol:
                    lines.append(
                        f"    [dim]│ →[/dim] {ol}"
                    )
            lines.append("")

    # Timer + iteration + tokens always at the bottom
    lines.append("")
    timer_line = (
        f"  {frame} Iteration {step.iteration}"
        f"  [dim]({_format_elapsed(elapsed)})[/dim]"
    )
    tok_str = _format_tokens(step.cumulative_tokens)
    if tok_str:
        timer_line += f"  [dim]{tok_str} tokens[/dim]"
    lines.append(timer_line)
    lines.append("")  # trailing space before prompt

    return Text.from_markup("\n".join(lines))


def _print_completed_steps(
    session: ChatSession,
    printed: set[int],
    target_console: Console | None = None,
) -> None:
    """Print newly completed plan steps permanently.

    Called in the main loop so completed steps stream above the
    Live spinner. *printed* tracks which step indices have already
    been output.

    Args:
        session: The current ChatSession.
        printed: Set of step indices already printed.
        target_console: Console to print to (e.g., live.console
            for printing above the Live area).
    """
    out = target_console or console

    if not session.plan:
        return

    # Print plan header once
    if not printed and any(
        session.plan_status.get(i) in ("in_progress", "completed")
        for i in range(len(session.plan))
    ):
        out.print()
        out.print("  [bold]Plan:[/bold]")
        out.print()
        printed.add(-1)  # sentinel: header printed

    for i, step_desc in enumerate(session.plan):
        if i in printed:
            continue
        status = session.plan_status.get(i, "pending")
        if status == "completed":
            dur = session.plan_step_duration.get(i)
            dur_str = (
                f"  [dim]thought for"
                f" {_format_elapsed(dur)}[/dim]"
                if dur
                else ""
            )
            out.print(
                f"    [green]✓[/green] {i + 1}."
                f" {step_desc}{dur_str}"
            )
            out.print()
            printed.add(i)
        elif status == "in_progress" and i not in printed:
            out.print(
                f"    [cyan]⠋[/cyan] {i + 1}. {step_desc}"
            )
            out.print()
            printed.add(i)


def _print_completed_iterations(
    session: ChatSession,
    printed_count: list[int],
    target_console: Console | None = None,
) -> None:
    """Print newly completed RLM iterations permanently.

    Called in the main loop so completed iteration outputs stream
    above the Live spinner. *printed_count* is a single-element
    list tracking how many iterations have been printed so far.

    Args:
        session: The current ChatSession.
        printed_count: Single-element list [n] — number already printed.
        target_console: Console for printing above Live area.
    """
    out = target_console or console
    steps = session.completed_steps
    already = printed_count[0]

    for step in steps[already:]:
        # Header with iteration number and duration
        dur_ms = step.duration_ms
        dur_str = ""
        if dur_ms > 0:
            dur_str = (
                f"  [dim]({dur_ms / 1000:.1f}s)[/dim]"
            )
        out.print(
            f"  [dim]Iteration {step.iteration}[/dim]"
            f"{dur_str}"
        )

        # Show reasoning snippet (first 2 sentences)
        if step.reasoning:
            sentences = step.reasoning.split(". ")[:2]
            for s in sentences:
                s = s.strip()
                if s:
                    out.print(f"    [dim]│[/dim] {s}")

        # Show key operations from code
        if step.code:
            shown = 0
            for code_line in step.code.split("\n"):
                if shown >= 4:
                    break
                stripped = code_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                ops = (
                    "write_file(", "read_file(",
                    "edit_file(", "run_cmd(",
                    "web_search(", "web_fetch(",
                    "list_files(", "llm_query(",
                )
                if any(op in stripped for op in ops):
                    out.print(
                        f"    [dim]│ >[/dim]"
                        f" [cyan]{stripped[:80]}[/cyan]"
                    )
                    shown += 1

        # Show output snippet (errors are highlighted)
        if step.output:
            output_preview = step.output.strip()[:200]
            if "error" in output_preview.lower() or \
               "Traceback" in output_preview:
                # Show error output in red
                for oline in output_preview.split("\n")[:3]:
                    oline = oline.strip()
                    if oline:
                        out.print(
                            f"    [dim]│[/dim]"
                            f" [red]{oline}[/red]"
                        )
            elif output_preview and len(output_preview) > 5:
                first_line = output_preview.split("\n")[0][:80]
                out.print(
                    f"    [dim]│ → {first_line}[/dim]"
                )

        out.print()  # spacing between iterations
        printed_count[0] += 1


def _print_plan_final(session: ChatSession) -> None:
    """Print any remaining unprinted plan steps after completion."""
    if not session.plan:
        return
    for i, step_desc in enumerate(session.plan):
        status = session.plan_status.get(i, "pending")
        if status == "skipped":
            console.print(
                f"    [dim]-[/dim] {i + 1}."
                f" [dim]{step_desc}[/dim]"
            )
            console.print()


# ── Main chat loop ─────────────────────────────────────────────────


def _get_version() -> str:
    """Get the memfun-cli package version."""
    try:
        from importlib.metadata import version
        return version("memfun-cli")
    except Exception:
        return "0.1.2"


def _load_credentials() -> None:
    """Load API keys from credentials files into environment.

    Checks global (~/.memfun/) first, then project (.memfun/).
    Project credentials override global ones.
    """
    _load_creds_file(Path.home() / ".memfun" / "credentials.json")
    _load_creds_file(Path.cwd() / ".memfun" / "credentials.json")


def _load_creds_file(creds_path: Path) -> None:
    """Load a single credentials.json file into env."""
    if not creds_path.exists():
        return
    try:
        import os

        creds = json.loads(creds_path.read_text())
        if isinstance(creds, dict):
            for key, value in creds.items():
                if (
                    isinstance(key, str)
                    and isinstance(value, str)
                    and value
                ):
                    os.environ[key] = value
    except Exception:
        logger.debug(
            "Failed to load credentials from %s",
            creds_path,
            exc_info=True,
        )


def _configure_dspy(config: MemfunConfig) -> None:
    """Configure DSPy with the LLM from memfun config."""
    import os

    import dspy

    provider = config.llm.provider
    model = config.llm.model
    api_key = os.environ.get(config.llm.api_key_env, "")

    if not api_key and provider != "ollama":
        logger.warning(
            "No API key found for %s (set %s). LLM calls will fail.",
            provider,
            config.llm.api_key_env,
        )
        return

    # DSPy uses LiteLLM model names: provider/model
    if provider == "anthropic":
        lm_model = f"anthropic/{model}"
    elif provider == "openai":
        lm_model = f"openai/{model}"
    elif provider == "ollama":
        lm_model = f"ollama_chat/{model}"
        api_key = "ollama"  # LiteLLM requires a non-empty key
    else:
        lm_model = model

    kwargs: dict[str, Any] = {
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if config.llm.base_url:
        kwargs["api_base"] = config.llm.base_url

    try:
        lm = dspy.LM(lm_model, **kwargs)
        dspy.configure(lm=lm)
        logger.info("DSPy configured with %s", lm_model)
    except Exception as exc:
        logger.warning("Failed to configure DSPy LM: %s", exc)


async def _async_chat_loop() -> None:
    """The async chat loop — called from chat_command."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    # Credentials and auto-init are handled in chat_command()
    # (sync context) before asyncio.run() starts.

    session = ChatSession()

    with console.status(
        "[bold cyan]Starting agent...[/bold cyan]", spinner="dots"
    ):
        try:
            await session.start()
        except Exception as exc:
            console.print(f"\n  [red]✗ Failed to start: {type(exc).__name__}: {exc}[/red]\n")
            return

    # Compact welcome banner
    console.print()
    console.print(
        f"  [bold cyan]memfun[/bold cyan] v{_get_version()} [dim]│[/dim] {session.model_name} "
        f"[dim]│[/dim] {Path.cwd().name}/ [dim]│[/dim] {session.history_stats}"
    )
    console.print("  [dim]Type a message or /help. Ctrl+C to cancel. Ctrl+D to exit.[/dim]")
    console.print()

    # Set up prompt_toolkit with file history and slash command completion
    history_path = Path.home() / ".memfun" / "chat_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=_SLASH_COMPLETER,
        complete_while_typing=True,
    )

    typed_ahead = ""  # keystrokes buffered during processing

    try:
        while True:
            # Get user input with dashed borders
            term_width = console.width
            dash_line = "─" * term_width
            console.print(f"[dim]{dash_line}[/dim]")
            try:
                user_input = await prompt_session.prompt_async(
                    "> ",
                    default=typed_ahead,
                )
                typed_ahead = ""
            except EOFError:
                break
            except KeyboardInterrupt:
                typed_ahead = ""
                continue
            finally:
                console.print(f"[dim]{dash_line}[/dim]")

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle exit/quit without slash
            if user_input.lower() in ("exit", "quit", "q"):
                break

            # Handle slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/exit", "/quit"):
                    break
                if cmd == "/help":
                    console.print(_HELP_TEXT)
                    continue
                if cmd == "/clear":
                    session.clear_history()
                    console.print("[dim]History cleared.[/dim]")
                    continue
                if cmd == "/context":
                    session.rescan_context()
                    console.print("[dim]Project context rescanned.[/dim]")
                    continue
                if cmd == "/history":
                    console.print(
                        f"[dim]History: {session.history_stats}[/dim]"
                    )
                    continue
                if cmd == "/model":
                    provider = session.provider
                    models = _AVAILABLE_MODELS.get(
                        provider, []
                    )
                    current = session.model_name
                    console.print(
                        f"  [bold]Model:[/bold] {current}"
                        f" [dim]({provider})[/dim]"
                    )
                    if models:
                        console.print()
                        for i, (mid, desc) in enumerate(
                            models, 1
                        ):
                            marker = (
                                "[green]●[/green]"
                                if mid == current
                                else "[dim]○[/dim]"
                            )
                            console.print(
                                f"    {marker} {i}. {desc}"
                                f" [dim]({mid})[/dim]"
                            )
                        console.print(
                            "    [dim]○[/dim] 0."
                            " Custom model ID"
                        )
                        console.print()
                        try:
                            choice = (
                                await prompt_session.prompt_async(
                                    "  Pick [1-"
                                    f"{len(models)}"
                                    ", 0=custom, Enter=cancel]: ",
                                )
                            ).strip()
                        except (EOFError, KeyboardInterrupt):
                            choice = ""
                        if not choice:
                            continue
                        try:
                            n = int(choice)
                        except ValueError:
                            console.print(
                                "[yellow]Invalid.[/yellow]"
                            )
                            continue
                        if n == 0:
                            try:
                                custom = (
                                    await prompt_session.prompt_async(
                                        "  Model ID: ",
                                    )
                                ).strip()
                            except (
                                EOFError,
                                KeyboardInterrupt,
                            ):
                                custom = ""
                            if custom:
                                session.switch_model(custom)
                                console.print(
                                    f"  [green]✓[/green]"
                                    f" Switched to {custom}"
                                )
                        elif 1 <= n <= len(models):
                            mid = models[n - 1][0]
                            session.switch_model(mid)
                            console.print(
                                f"  [green]✓[/green]"
                                f" Switched to {mid}"
                            )
                        else:
                            console.print(
                                "[yellow]Invalid.[/yellow]"
                            )
                    continue
                if cmd == "/traces":
                    traces = await session.list_traces()
                    if traces:
                        for t in traces:
                            console.print(f"  {t}")
                    else:
                        console.print("[dim]No traces found.[/dim]")
                    continue
                if cmd == "/remember":
                    from memfun_cli.memory import remember

                    text = user_input[len("/remember") :].strip()
                    if not text:
                        console.print(
                            "[yellow]Usage:"
                            " /remember <text>[/yellow]\n"
                            "[dim]Example: /remember"
                            " Use port 8080 for dev"
                            " servers[/dim]\n"
                            "[dim]Use --global to save"
                            " globally[/dim]"
                        )
                        continue
                    is_global = text.startswith("--global ")
                    if is_global:
                        text = text[len("--global ") :].strip()
                    msg = remember(
                        text, project=not is_global
                    )
                    console.print(f"  [green]{msg}[/green]")
                    continue
                if cmd == "/memory":
                    from memfun_cli.memory import (
                        get_memory_display,
                    )

                    console.print(get_memory_display())
                    continue
                if cmd == "/forget":
                    from memfun_cli.memory import forget

                    target = user_input[len("/forget") :].strip()
                    if not target:
                        console.print(
                            "[yellow]Usage:"
                            " /forget <number-or-text>"
                            "[/yellow]\n"
                            "[dim]Example: /forget 3[/dim]\n"
                            "[dim]Example: /forget"
                            " port 5000[/dim]"
                        )
                        continue
                    is_global = target.startswith("--global ")
                    if is_global:
                        target = target[
                            len("--global ") :
                        ].strip()
                    msg = forget(
                        target, project=not is_global
                    )
                    console.print(f"  [dim]{msg}[/dim]")
                    continue
                if cmd == "/debug-learning":
                    console.print(
                        "\n  [bold]Learning Pipeline"
                        " Diagnostic[/bold]\n"
                    )
                    if session._learning_manager is None:
                        console.print(
                            "  [red]LearningManager not"
                            " initialized[/red]"
                        )
                        continue
                    # Step 1: Check DSPy LM
                    console.print(
                        "  [cyan]1.[/cyan] Checking"
                        " DSPy LM..."
                    )
                    try:
                        import dspy
                        lm = dspy.settings.lm
                        if lm is None:
                            console.print(
                                "    [red]No LM configured"
                                " in DSPy![/red]"
                            )
                            continue
                        console.print(
                            f"    [green]OK[/green]"
                            f" LM: {lm.model}"
                        )
                    except Exception as exc:
                        console.print(
                            f"    [red]Error: {exc}[/red]"
                        )
                        continue
                    # Step 2: Test extraction
                    console.print(
                        "  [cyan]2.[/cyan] Running test"
                        " extraction..."
                    )
                    test_user = (
                        "I prefer using port 8080"
                        " for all dev servers"
                    )
                    test_response = (
                        "I've configured the Flask app"
                        " to run on port 8080 as you"
                        " prefer."
                    )
                    try:
                        stored = await (
                            session._learning_manager
                            .extract_and_store(
                                test_user, test_response
                            )
                        )
                        if stored:
                            console.print(
                                f"    [green]OK[/green]"
                                f" Extracted {len(stored)}"
                                f" learnings:"
                            )
                            for s in stored:
                                console.print(
                                    f"      - {s}"
                                )
                        else:
                            console.print(
                                "    [yellow]No learnings"
                                " extracted (empty"
                                " result)[/yellow]"
                            )
                    except Exception as exc:
                        console.print(
                            f"    [red]Extraction"
                            f" failed: {exc}[/red]"
                        )
                        import traceback
                        console.print(
                            f"    [dim]{traceback.format_exc()}"
                            f"[/dim]"
                        )
                    # Step 3: Check DB entries
                    console.print(
                        "  [cyan]3.[/cyan] Checking"
                        " MemoryStore..."
                    )
                    try:
                        entries = await (
                            session._learning_manager
                            ._memory.list_entries(limit=10)
                        )
                        console.print(
                            f"    [green]OK[/green]"
                            f" {len(entries)} entries"
                            f" in database"
                        )
                        for e in entries[:5]:
                            console.print(
                                f"      [{e.topic}]"
                                f" {e.content[:60]}"
                            )
                    except Exception as exc:
                        console.print(
                            f"    [red]DB error:"
                            f" {exc}[/red]"
                        )
                    # Step 4: Check MEMORY.md
                    console.print(
                        "  [cyan]4.[/cyan] Checking"
                        " MEMORY.md..."
                    )
                    from memfun_cli.memory import (
                        load_memory_context,
                        project_memory_path,
                    )
                    ppath = project_memory_path()
                    if ppath.exists():
                        ctx = load_memory_context()
                        lines = [
                            ln for ln in ctx.split("\n")
                            if ln.strip().startswith("- ")
                        ]
                        console.print(
                            f"    [green]OK[/green]"
                            f" {ppath} ({len(lines)}"
                            f" entries)"
                        )
                    else:
                        console.print(
                            f"    [dim]No file:"
                            f" {ppath}[/dim]"
                        )
                    console.print()
                    continue
                console.print(
                    f"[yellow]Unknown command: {cmd}. "
                    f"Type /help for options.[/yellow]"
                )
                continue

            # Process message with streaming output
            t0 = time.monotonic()
            result: TaskResult | None = None
            task: asyncio.Task[TaskResult] | None = None
            cancelled = False
            typed_ahead = ""  # buffer keystrokes for next prompt
            printed_steps: set[int] = set()
            printed_iters: list[int] = [0]

            # Set up terminal for Escape key detection
            fd = sys.stdin.fileno()
            old_term = termios.tcgetattr(fd)

            try:
                tty.setcbreak(fd)

                with Live(
                    _make_progress_renderable(session, 0.0),
                    console=console,
                    refresh_per_second=4,
                    transient=True,
                ) as live:
                    task = asyncio.create_task(
                        session.chat_turn(user_input)
                    )
                    while not task.done():
                        # Non-blocking key check
                        if select.select(
                            [sys.stdin], [], [], 0
                        )[0]:
                            ch = sys.stdin.read(1)
                            if ch == "\x1b":
                                task.cancel()
                                cancelled = True
                                break
                            # Buffer other keystrokes
                            if ch.isprintable() or ch == " ":
                                typed_ahead += ch

                        # Stream completed plan steps above
                        # the Live spinner
                        _print_completed_steps(
                            session,
                            printed_steps,
                            live.console,
                        )

                        # Stream completed RLM iterations
                        _print_completed_iterations(
                            session,
                            printed_iters,
                            live.console,
                        )

                        elapsed = time.monotonic() - t0
                        live.update(
                            _make_progress_renderable(
                                session, elapsed
                            )
                        )
                        await asyncio.sleep(0.15)

                    if not cancelled and task.done():
                        result = task.result()
            except (KeyboardInterrupt, asyncio.CancelledError):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                cancelled = True
            except Exception as exc:
                console.print(
                    f"\n  [red]✗ {type(exc).__name__}:"
                    f" {exc}[/red]\n"
                )
                continue
            finally:
                termios.tcsetattr(
                    fd, termios.TCSADRAIN, old_term
                )

            if cancelled:
                typed_ahead = ""
                console.print(
                    "\n  [yellow]Cancelled.[/yellow]\n"
                )
                continue

            # Print any remaining plan steps (skipped, etc.)
            _print_plan_final(session)

            # Spacing between processing and answer
            console.print()

            if result is not None:
                _display_answer(result)
            console.print()

    finally:
        with console.status(
            "[dim]Shutting down...[/dim]", spinner="dots"
        ):
            await session.stop()
        console.print("[dim]Goodbye.[/dim]")


def chat_command() -> None:
    """Launch the interactive chat interface."""
    # Auto-setup BEFORE entering the async event loop,
    # because InquirerPy prompts use their own asyncio.run()
    # internally and nested event loops are not allowed.

    # Ensure global config exists (first-run wizard)
    global_config = Path.home() / ".memfun" / "config.toml"
    if not global_config.exists():
        from memfun_cli.commands.init import run_global_setup

        run_global_setup()

    # Load credentials (sync, before event loop)
    _load_credentials()

    # Ensure .memfun/ exists in CWD
    project_dir = Path.cwd() / ".memfun"
    if not project_dir.exists():
        from memfun_cli.commands.init import run_project_init

        run_project_init()

    asyncio.run(_async_chat_loop())
