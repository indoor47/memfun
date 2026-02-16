"""Agent CLI commands: ask, analyze, fix, review, explain.

Bridges the synchronous Typer CLI to the async RLMCodingAgent by loading
configuration, building a RuntimeContext, and dispatching TaskMessages.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from memfun_core.config import MemfunConfig
from memfun_core.types import TaskMessage, TaskResult
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Maximum bytes to read from a single file for context
_MAX_FILE_READ_BYTES = 256_000

# Extensions we consider "source code" when scanning a directory
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".ex", ".exs", ".kt",
    ".swift", ".toml", ".yaml", ".yml", ".json", ".md", ".sh",
})


# ── Helpers ────────────────────────────────────────────────────────


def _load_config() -> MemfunConfig:
    """Load MemfunConfig from memfun.toml in cwd, falling back to defaults."""
    toml_path = Path("memfun.toml")
    if toml_path.exists():
        console.print(
            "[dim]Loaded configuration from memfun.toml[/dim]"
        )
        return MemfunConfig.from_toml(toml_path)
    console.print(
        "[dim]No memfun.toml found; using default configuration[/dim]"
    )
    return MemfunConfig()


def _read_file_context(path: Path) -> str:
    """Read a single file and return its contents (truncated if huge)."""
    try:
        text = path.read_text(errors="replace")
        if len(text) > _MAX_FILE_READ_BYTES:
            text = text[:_MAX_FILE_READ_BYTES] + "\n... (truncated)"
        return text
    except Exception as exc:
        return f"(Error reading {path}: {exc})"


def _read_path_context(path: Path) -> str:
    """Build a context string from *path*.

    - If *path* is a file, return its contents.
    - If *path* is a directory, concatenate a handful of key files
      (README, pyproject.toml, plus up to 10 source files).
    """
    if path.is_file():
        return _read_file_context(path)

    parts: list[str] = []

    # Priority files
    for name in ("README.md", "pyproject.toml", "package.json", "Cargo.toml"):
        candidate = path / name
        if candidate.exists():
            parts.append(f"--- {candidate.name} ---\n{_read_file_context(candidate)}")

    # Gather source files (non-hidden, shallow scan)
    source_files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix in _SOURCE_EXTENSIONS:
            # Skip hidden dirs and common noise
            rel = child.relative_to(path)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if any(
                p in {"node_modules", "__pycache__", ".git", "dist", "build"}
                for p in rel.parts
            ):
                continue
            source_files.append(child)
            if len(source_files) >= 10:
                break

    for sf in source_files:
        rel = sf.relative_to(path)
        parts.append(f"--- {rel} ---\n{_read_file_context(sf)}")

    if not parts:
        return "(No readable source files found in directory)"

    return "\n\n".join(parts)


def _load_credentials() -> None:
    """Load API keys from ~/.memfun/credentials.json into environment."""
    import json
    import os

    for creds_path in [
        Path.home() / ".memfun" / "credentials.json",
        Path.cwd() / ".memfun" / "credentials.json",
    ]:
        if not creds_path.exists():
            continue
        try:
            creds = json.loads(creds_path.read_text())
            if isinstance(creds, dict):
                for key, value in creds.items():
                    if isinstance(key, str) and isinstance(value, str) and value:
                        os.environ[key] = value
        except Exception:
            pass


def _configure_dspy(config: MemfunConfig) -> None:
    """Configure DSPy with the LLM from memfun config."""
    import os

    import dspy

    provider = config.llm.provider
    model = config.llm.model
    api_key = os.environ.get(config.llm.api_key_env, "")

    if not api_key and provider != "ollama":
        console.print(
            f"[yellow]Warning: No API key for {provider} "
            f"(set {config.llm.api_key_env})[/yellow]"
        )
        return

    if provider == "anthropic":
        lm_model = f"anthropic/{model}"
    elif provider == "openai":
        lm_model = f"openai/{model}"
    elif provider == "ollama":
        lm_model = f"ollama_chat/{model}"
        api_key = "ollama"
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
    except Exception as exc:
        console.print(f"[yellow]Warning: Failed to configure DSPy: {exc}[/yellow]")


async def run_agent_task(
    task_type: str,
    query: str,
    context: str = "",
    path: str | None = None,
) -> TaskResult:
    """Build the runtime, start the agent, dispatch a task, and return the result.

    This is the core helper shared by all agent CLI commands. It:
    1. Loads credentials from ``~/.memfun/credentials.json``.
    2. Loads ``MemfunConfig`` and configures DSPy.
    3. Builds a ``RuntimeContext`` via ``RuntimeBuilder``.
    4. Instantiates ``RLMCodingAgent`` with that context.
    5. Calls ``on_start()``, then ``handle()`` with a ``TaskMessage``.
    6. Returns the ``TaskResult``.
    """
    from memfun_agent.coding_agent import RLMCodingAgent
    from memfun_runtime.builder import RuntimeBuilder

    _load_credentials()
    config = _load_config()
    _configure_dspy(config)
    runtime_ctx = await RuntimeBuilder(config).build()
    agent = RLMCodingAgent(runtime_ctx)

    try:
        await agent.on_start()

        payload: dict[str, Any] = {
            "type": task_type,
            "query": query,
            "context": context,
        }
        if task_type == "fix":
            payload["bug_description"] = query

        task_msg = TaskMessage(
            task_id=uuid.uuid4().hex,
            agent_id=agent.agent_id,
            payload=payload,
        )

        result = await agent.handle(task_msg)
        return result
    finally:
        await agent.on_stop()


# ── Display helpers ────────────────────────────────────────────────


def _display_result(result: TaskResult, task_type: str) -> None:
    """Render a TaskResult using Rich panels and syntax highlighting."""
    if not result.success:
        console.print(Panel(
            f"[red]{result.error}[/red]",
            title="Error",
            border_style="red",
        ))
        return

    data = result.result

    # Common "answer" field (used by RLM path and ask)
    answer = data.get("answer", "")
    method = data.get("method", "unknown")
    duration = f"{result.duration_ms:.0f}ms"

    if task_type == "analyze":
        _display_analyze(data, method, duration)
    elif task_type == "fix":
        _display_fix(data, method, duration)
    elif task_type == "review":
        _display_review(data, method, duration)
    elif task_type == "explain":
        _display_explain(data, method, duration)
    else:
        _display_generic(answer, method, duration)


def _display_analyze(
    data: dict[str, Any], method: str, duration: str,
) -> None:
    """Display analysis results."""
    answer = data.get("answer", data.get("analysis", ""))
    issues = data.get("issues", [])
    suggestions = data.get("suggestions", [])

    console.print(Panel(
        Markdown(answer) if answer else "[dim]No analysis returned.[/dim]",
        title=f"Analysis  [dim]({method}, {duration})[/dim]",
        border_style="cyan",
    ))

    if issues:
        console.print()
        issue_text = "\n".join(f"  - {i}" for i in issues)
        console.print(Panel(
            issue_text,
            title="Issues",
            border_style="yellow",
        ))

    if suggestions:
        console.print()
        suggestion_text = "\n".join(f"  - {s}" for s in suggestions)
        console.print(Panel(
            suggestion_text,
            title="Suggestions",
            border_style="green",
        ))


def _display_fix(
    data: dict[str, Any], method: str, duration: str,
) -> None:
    """Display bug-fix results."""
    answer = data.get("answer", "")
    root_cause = data.get("root_cause", "")
    fixed_code = data.get("fixed_code", "")
    explanation = data.get("explanation", "")

    # RLM path returns "answer"; direct path returns structured fields
    if answer and not root_cause:
        console.print(Panel(
            Markdown(answer),
            title=f"Fix  [dim]({method}, {duration})[/dim]",
            border_style="green",
        ))
        return

    if root_cause:
        console.print(Panel(
            Markdown(root_cause),
            title="Root Cause",
            border_style="yellow",
        ))

    if fixed_code:
        console.print()
        console.print(Panel(
            Syntax(fixed_code, "python", theme="monokai", word_wrap=True),
            title=f"Fixed Code  [dim]({method}, {duration})[/dim]",
            border_style="green",
        ))

    if explanation:
        console.print()
        console.print(Panel(
            Markdown(explanation),
            title="Explanation",
            border_style="cyan",
        ))


def _display_review(
    data: dict[str, Any], method: str, duration: str,
) -> None:
    """Display code-review results."""
    answer = data.get("answer", "")
    summary = data.get("summary", "")
    findings = data.get("findings", [])
    approved = data.get("approved")

    if answer and not summary:
        console.print(Panel(
            Markdown(answer),
            title=f"Review  [dim]({method}, {duration})[/dim]",
            border_style="cyan",
        ))
        return

    if summary:
        status = (
            "[green]APPROVED[/green]"
            if approved
            else "[yellow]CHANGES REQUESTED[/yellow]"
        )
        console.print(Panel(
            Markdown(summary),
            title=f"Review Summary  {status}  [dim]({method}, {duration})[/dim]",
            border_style="cyan",
        ))

    if findings:
        console.print()
        findings_text = "\n".join(f"  - {f}" for f in findings)
        console.print(Panel(
            findings_text,
            title="Findings",
            border_style="yellow",
        ))


def _display_explain(
    data: dict[str, Any], method: str, duration: str,
) -> None:
    """Display explanation results."""
    answer = data.get("answer", "")
    explanation = data.get("explanation", "")
    key_concepts = data.get("key_concepts", [])

    text = answer or explanation
    console.print(Panel(
        Markdown(text) if text else "[dim]No explanation returned.[/dim]",
        title=f"Explanation  [dim]({method}, {duration})[/dim]",
        border_style="cyan",
    ))

    if key_concepts:
        console.print()
        concepts_text = "\n".join(f"  - {c}" for c in key_concepts)
        console.print(Panel(
            concepts_text,
            title="Key Concepts",
            border_style="green",
        ))


def _display_generic(answer: str, method: str, duration: str) -> None:
    """Display a generic answer (for ask and fallback)."""
    console.print(Panel(
        Markdown(answer) if answer else "[dim]No response returned.[/dim]",
        title=f"Answer  [dim]({method}, {duration})[/dim]",
        border_style="cyan",
    ))


# ── CLI-facing functions (called from main.py) ────────────────────


def ask_command(question: str) -> None:
    """Ask the agent a question. Reads cwd for context."""
    cwd_context = _read_path_context(Path.cwd())
    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
        try:
            result = asyncio.run(
                run_agent_task("ask", question, context=cwd_context)
            )
        except Exception as exc:
            console.print(Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Agent Error",
                border_style="red",
            ))
            return
    _display_result(result, "ask")


def analyze_command(path: str | None = None) -> None:
    """Analyze code at the given path (defaults to cwd)."""
    target = Path(path) if path else Path.cwd()
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        return

    console.print(f"[dim]Analyzing: {target.resolve()}[/dim]")
    context = _read_path_context(target)

    query = f"Analyze the code at {target.resolve()}"
    with console.status("[bold cyan]Analyzing...[/bold cyan]", spinner="dots"):
        try:
            result = asyncio.run(
                run_agent_task("analyze", query, context=context)
            )
        except Exception as exc:
            console.print(Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Agent Error",
                border_style="red",
            ))
            return
    _display_result(result, "analyze")


def fix_command(description: str) -> None:
    """Fix a bug described in the given text."""
    cwd_context = _read_path_context(Path.cwd())
    with console.status("[bold cyan]Fixing...[/bold cyan]", spinner="dots"):
        try:
            result = asyncio.run(
                run_agent_task("fix", description, context=cwd_context)
            )
        except Exception as exc:
            console.print(Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Agent Error",
                border_style="red",
            ))
            return
    _display_result(result, "fix")


def review_command(path: str | None = None) -> None:
    """Review code at the given path (defaults to cwd)."""
    target = Path(path) if path else Path.cwd()
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        return

    console.print(f"[dim]Reviewing: {target.resolve()}[/dim]")
    context = _read_path_context(target)

    query = f"Review the code at {target.resolve()}"
    with console.status("[bold cyan]Reviewing...[/bold cyan]", spinner="dots"):
        try:
            result = asyncio.run(
                run_agent_task("review", query, context=context)
            )
        except Exception as exc:
            console.print(Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Agent Error",
                border_style="red",
            ))
            return
    _display_result(result, "review")


def explain_command(path: str | None = None) -> None:
    """Explain code at the given path (defaults to cwd)."""
    target = Path(path) if path else Path.cwd()
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        return

    console.print(f"[dim]Explaining: {target.resolve()}[/dim]")
    context = _read_path_context(target)

    query = f"Explain the code at {target.resolve()}"
    with console.status("[bold cyan]Explaining...[/bold cyan]", spinner="dots"):
        try:
            result = asyncio.run(
                run_agent_task("explain", query, context=context)
            )
        except Exception as exc:
            console.print(Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Agent Error",
                border_style="red",
            ))
            return
    _display_result(result, "explain")
