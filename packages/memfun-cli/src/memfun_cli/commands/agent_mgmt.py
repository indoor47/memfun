"""Agent management commands: list, info, validate, run."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from memfun_agent.definitions import AgentLoader, AgentValidator, parse_agent_md
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

if TYPE_CHECKING:
    from memfun_agent.definitions.types import AgentDefinition

console = Console()

agent_app = typer.Typer(
    no_args_is_help=True,
)


def _build_loader() -> AgentLoader:
    """Build an AgentLoader with default discovery paths."""
    return AgentLoader()


@agent_app.command("list")
def agent_list() -> None:
    """List all discovered agent definitions."""
    loader = _build_loader()
    agents = loader.discover()

    if not agents:
        console.print(
            "[yellow]No agents found.[/yellow] "
            "Place AGENT.md files in ./agents/ or ~/.memfun/agents/."
        )
        raise typer.Exit(0)

    table = Table(
        title="Discovered Agents",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("Version", justify="center")
    table.add_column("Description")
    table.add_column("Capabilities")

    for agent in sorted(agents, key=lambda a: a.name):
        caps_str = ", ".join(agent.capabilities) if agent.capabilities else "-"
        table.add_row(
            agent.name,
            agent.version,
            agent.description,
            caps_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(agents)} agent(s) found.[/dim]")


@agent_app.command("info")
def agent_info(
    name: str = typer.Argument(..., help="Name of the agent to inspect"),
) -> None:
    """Show detailed information about an agent definition."""
    loader = _build_loader()
    agents = loader.discover()

    match = next((a for a in agents if a.name == name), None)
    if match is None:
        console.print(f"[red]Agent not found:[/red] '{name}'")
        available = [a.name for a in agents]
        if available:
            console.print(
                f"[dim]Available agents: {', '.join(sorted(available))}[/dim]"
            )
        raise typer.Exit(1)

    # Metadata panel
    meta_lines = [
        f"[bold]Name:[/bold]          {match.name}",
        f"[bold]Version:[/bold]       {match.version}",
        f"[bold]Description:[/bold]   {match.description}",
    ]

    if match.capabilities:
        meta_lines.append(
            f"[bold]Capabilities:[/bold]  {', '.join(match.capabilities)}"
        )

    if match.allowed_tools:
        meta_lines.append(
            f"[bold]Allowed tools:[/bold] {', '.join(match.allowed_tools)}"
        )

    if match.delegates_to:
        meta_lines.append(
            f"[bold]Delegates to:[/bold]  {', '.join(match.delegates_to)}"
        )

    if match.model:
        meta_lines.append(f"[bold]Model:[/bold]         {match.model}")

    meta_lines.append(f"[bold]Max turns:[/bold]     {match.max_turns}")

    if match.source_path:
        meta_lines.append(f"[bold]Source:[/bold]        {match.source_path}")

    console.print(Panel(
        "\n".join(meta_lines),
        title=f"Agent: {match.name}",
        border_style="cyan",
    ))

    # Instructions preview
    if match.instructions:
        preview = match.instructions
        if len(preview) > 500:
            preview = preview[:500] + "\n\n... (truncated)"
        console.print()
        console.print(Panel(
            Syntax(preview, "markdown", theme="monokai", word_wrap=True),
            title="Instructions (preview)",
            border_style="dim",
        ))
    else:
        console.print("\n[dim]No instructions body defined.[/dim]")


@agent_app.command("validate")
def agent_validate(
    path: str | None = typer.Argument(
        None,
        help="Path to an agent directory or AGENT.md file. "
        "If omitted, validates all discovered agents.",
    ),
) -> None:
    """Validate agent definition(s) and report errors."""
    validator = AgentValidator()

    if path is not None:
        _validate_single(Path(path), validator)
    else:
        _validate_all(validator)


def _validate_single(path: Path, validator: AgentValidator) -> None:
    """Validate a single agent at *path*."""
    resolved = path.expanduser().resolve()

    # Accept either a directory or an AGENT.md file
    if resolved.is_file() and resolved.name == "AGENT.md":
        agent_file = resolved
    elif resolved.is_dir():
        agent_file = resolved / "AGENT.md"
    else:
        console.print(
            f"[red]Invalid path:[/red] '{path}' is not a directory "
            "or an AGENT.md file."
        )
        raise typer.Exit(1)

    if not agent_file.exists():
        console.print(
            f"[red]AGENT.md not found at:[/red] {agent_file}"
        )
        raise typer.Exit(1)

    try:
        agent = parse_agent_md(agent_file)
    except Exception as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        raise typer.Exit(1) from None

    errors = validator.validate(agent)
    _report_validation(agent.name, errors)

    if errors:
        raise typer.Exit(1)


def _validate_all(validator: AgentValidator) -> None:
    """Validate all discovered agents."""
    loader = _build_loader()
    agents = loader.discover()

    if not agents:
        console.print(
            "[yellow]No agents found to validate.[/yellow] "
            "Place AGENT.md files in ./agents/ or ~/.memfun/agents/."
        )
        raise typer.Exit(0)

    total_errors = 0
    for agent in sorted(agents, key=lambda a: a.name):
        errors = validator.validate(agent)
        _report_validation(agent.name, errors)
        total_errors += len(errors)

    console.print()
    if total_errors == 0:
        console.print(
            f"[green]All {len(agents)} agent(s) passed validation.[/green]"
        )
    else:
        console.print(
            f"[red]{total_errors} error(s) across {len(agents)} agent(s).[/red]"
        )
        raise typer.Exit(1)


def _report_validation(name: str, errors: list[str]) -> None:
    """Print validation results for a single agent."""
    if not errors:
        console.print(f"  [green]OK[/green]  {name}")
    else:
        console.print(f"  [red]FAIL[/red] {name}")
        for err in errors:
            console.print(f"        [red]-[/red] {err}")


@agent_app.command("run")
def agent_run(
    name: str = typer.Argument(..., help="Name of the agent to run"),
    task: str = typer.Argument(..., help="Task description for the agent"),
) -> None:
    """Run an agent definition against a task (dry-run: assembles prompt, shows it)."""
    loader = _build_loader()
    agents = loader.discover()

    match = next((a for a in agents if a.name == name), None)
    if match is None:
        console.print(f"[red]Agent not found:[/red] '{name}'")
        available = [a.name for a in agents]
        if available:
            console.print(
                f"[dim]Available agents: {', '.join(sorted(available))}[/dim]"
            )
        raise typer.Exit(1)

    # Build the prompt the same way DefinedAgent would
    prompt = _build_dry_run_prompt(match, task)

    # Show agent metadata
    meta_lines = [
        f"[bold]Agent:[/bold]       {match.name} v{match.version}",
        f"[bold]Model:[/bold]       {match.model or 'default'}",
        f"[bold]Max turns:[/bold]   {match.max_turns}",
    ]
    if match.allowed_tools:
        meta_lines.append(
            f"[bold]Tools:[/bold]       {', '.join(match.allowed_tools)}"
        )
    if match.delegates_to:
        meta_lines.append(
            f"[bold]Delegates:[/bold]   {', '.join(match.delegates_to)}"
        )

    console.print(Panel(
        "\n".join(meta_lines),
        title=f"Dry-run: {match.name}",
        subtitle="agent run",
        border_style="cyan",
    ))

    # Show assembled prompt
    console.print()
    console.print(Panel(
        Syntax(prompt, "markdown", theme="monokai", word_wrap=True),
        title="Assembled Prompt",
        border_style="dim",
    ))

    console.print()
    console.print(
        "[yellow]Dry-run mode: the prompt above would be sent to the "
        "agent in a live execution.[/yellow]"
    )


def _build_dry_run_prompt(
    agent: AgentDefinition,
    task: str,
) -> str:
    """Build a prompt from an agent definition and a task string.

    Mirrors the prompt-building logic in DefinedAgent._build_prompt
    without requiring a RuntimeContext or TaskMessage.
    """
    parts: list[str] = []

    if agent.instructions:
        parts.append(agent.instructions)

    parts.append("\n## Task\n")
    parts.append(f"**query**: {task}\n")

    return "\n".join(parts)
