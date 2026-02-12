"""Skill invocation: resolve a skill by name and display its instructions."""
from __future__ import annotations

import typer
from memfun_skills import SkillLoader
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def invoke_skill(
    skill_name: str,
    args: list[str] | None = None,
) -> None:
    """Discover skills, find the named one, and print its instructions.

    This is the handler for slash-command invocation (``memfun /skill-name``).
    For now it displays what *would* be sent to the agent; actual RLM
    execution is deferred to a later phase.

    Args:
        skill_name: The skill name (without the leading ``/``).
        args: Additional positional arguments passed after the skill name.

    Raises:
        typer.Exit: If the skill is not found or not user-invocable.
    """
    loader = SkillLoader()
    skills = loader.discover()

    match = next((s for s in skills if s.name == skill_name), None)

    if match is None:
        console.print(f"[red]Skill not found:[/red] '{skill_name}'")
        available = [s.name for s in skills if s.user_invocable]
        if available:
            console.print(
                "[dim]Available skills: "
                f"{', '.join(sorted(available))}[/dim]"
            )
        else:
            console.print(
                "[dim]No user-invocable skills discovered. "
                "Place SKILL.md files in ./skills/ or "
                "~/.memfun/skills/.[/dim]"
            )
        raise typer.Exit(1)

    if not match.user_invocable:
        console.print(
            f"[red]Skill '{skill_name}' is not user-invocable.[/red] "
            "It can only be activated by the agent."
        )
        raise typer.Exit(1)

    # Header
    args_display = " ".join(args) if args else ""
    title = f"/{match.name}"
    if args_display:
        title += f" {args_display}"

    meta_lines = [
        f"[bold]Skill:[/bold]   {match.name} v{match.version}",
        f"[bold]Source:[/bold]  {match.source_path}",
    ]
    if match.tags:
        meta_lines.append(f"[bold]Tags:[/bold]    {', '.join(match.tags)}")
    if match.allowed_tools:
        meta_lines.append(
            f"[bold]Tools:[/bold]   {', '.join(match.allowed_tools)}"
        )
    if args_display:
        meta_lines.append(f"[bold]Args:[/bold]    {args_display}")

    console.print(Panel(
        "\n".join(meta_lines),
        title=title,
        subtitle="skill invocation",
        border_style="cyan",
    ))

    # Instructions that would be sent to the agent
    if match.instructions:
        console.print()
        console.print(Panel(
            Syntax(
                match.instructions,
                "markdown",
                theme="monokai",
                word_wrap=True,
            ),
            title="Instructions (would be sent to agent)",
            border_style="dim",
        ))
    else:
        console.print(
            "\n[dim]This skill has no instruction body.[/dim]"
        )

    console.print()
    console.print(
        "[yellow]Agent execution not yet implemented. "
        "The instructions above would be injected into the agent "
        "context in a future phase.[/yellow]"
    )
