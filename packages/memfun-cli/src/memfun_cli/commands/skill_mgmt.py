"""Extended skill commands: search, invoke (dry-run), create scaffold.

These functions are registered onto the existing ``skill_app`` Typer
sub-application in ``main.py``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from memfun_skills import (
    SkillActivator,
    SkillExecutionContext,
    SkillExecutor,
    SkillLoader,
    SkillRouter,
)
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def _build_loader() -> SkillLoader:
    """Build a SkillLoader with default discovery paths."""
    return SkillLoader()


def skill_search(
    query: str = typer.Argument(..., help="Query to search for matching skills"),
) -> None:
    """Search for skills matching a query."""
    loader = _build_loader()
    skills = loader.discover()

    if not skills:
        console.print(
            "[yellow]No skills found.[/yellow] "
            "Place SKILL.md files in ./skills/ or ~/.memfun/skills/."
        )
        raise typer.Exit(0)

    router = SkillRouter()
    results = router.route_all(query, skills)

    if not results:
        console.print(
            f"[yellow]No skills matched query:[/yellow] '{query}'"
        )
        console.print(
            f"[dim]{len(skills)} skill(s) available. "
            "Try a different query or run 'memfun skill list'.[/dim]"
        )
        raise typer.Exit(0)

    table = Table(
        title=f"Skills matching: '{query}'",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Match Reasons")
    table.add_column("Description")

    for scored in results:
        reasons_str = "; ".join(scored.match_reasons) if scored.match_reasons else "-"
        table.add_row(
            scored.skill.name,
            str(scored.score),
            reasons_str,
            scored.skill.description,
        )

    console.print(table)
    console.print(f"\n[dim]{len(results)} result(s).[/dim]")


def skill_invoke(
    name: str = typer.Argument(..., help="Name of the skill to invoke"),
    query: str = typer.Argument(
        "", help="Query or input for the skill (optional)"
    ),
) -> None:
    """Invoke a skill by name (dry-run: builds execution prompt)."""
    loader = _build_loader()
    skills = loader.discover()

    match = next((s for s in skills if s.name == name), None)
    if match is None:
        console.print(f"[red]Skill not found:[/red] '{name}'")
        available = [s.name for s in skills]
        if available:
            console.print(
                f"[dim]Available skills: {', '.join(sorted(available))}[/dim]"
            )
        raise typer.Exit(1)

    # Activate the skill (resolve references, map tools)
    activator = SkillActivator()
    context = SkillExecutionContext(
        skill=match,
        arguments={"query": query} if query else {},
        working_dir=Path.cwd(),
    )

    try:
        activated = asyncio.run(activator.activate(match, context))
    except Exception as exc:
        console.print(
            f"[red]Activation error:[/red] {type(exc).__name__}: {exc}"
        )
        raise typer.Exit(1) from None

    # Execute in dry-run mode (no LLM callback)
    executor = SkillExecutor(llm_callback=None)
    task_payload = {"query": query} if query else {}

    try:
        result = asyncio.run(executor.execute(activated, task_payload))
    except Exception as exc:
        console.print(
            f"[red]Execution error:[/red] {type(exc).__name__}: {exc}"
        )
        raise typer.Exit(1) from None

    # Show skill metadata
    meta_lines = [
        f"[bold]Skill:[/bold]   {match.name} v{match.version}",
        f"[bold]Source:[/bold]  {match.source_path}",
    ]
    if activated.mapped_tools:
        meta_lines.append(
            f"[bold]Tools:[/bold]   {', '.join(activated.mapped_tools)}"
        )
    meta_lines.append(f"[bold]Duration:[/bold] {result.duration_ms:.1f}ms")

    console.print(Panel(
        "\n".join(meta_lines),
        title=f"Skill Invocation: {match.name}",
        subtitle="dry-run",
        border_style="cyan",
    ))

    # Show assembled prompt
    console.print()
    console.print(Panel(
        Syntax(result.output, "markdown", theme="monokai", word_wrap=True),
        title="Assembled Prompt (dry-run)",
        border_style="dim",
    ))

    console.print()
    console.print(
        "[yellow]Dry-run mode: no LLM was called. The prompt above "
        "would be sent to the model in a live execution.[/yellow]"
    )


def skill_create(
    name: str = typer.Argument(..., help="Name for the new skill"),
) -> None:
    """Scaffold a new SKILL.md in the skills/ directory."""
    skills_dir = Path.cwd() / "skills"
    skill_dir = skills_dir / name

    if skill_dir.exists():
        console.print(
            f"[red]Skill directory already exists:[/red] {skill_dir}"
        )
        raise typer.Exit(1)

    # Create the skill directory structure
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Generate the SKILL.md template
    template = _build_skill_template(name)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(template, encoding="utf-8")

    console.print(Panel(
        f"[bold]Created:[/bold] {skill_file}\n"
        f"[bold]Directory:[/bold] {skill_dir}\n\n"
        "Edit the SKILL.md file to customize the skill's metadata\n"
        "and instructions. Then validate with:\n\n"
        f"  memfun skill validate {skill_dir}",
        title=f"New Skill: {name}",
        border_style="green",
    ))


def _build_skill_template(name: str) -> str:
    """Build a SKILL.md template with frontmatter and placeholder instructions."""
    return f"""---
name: {name}
description: "A brief description of what this skill does"
version: "0.1.0"
user-invocable: true
model-invocable: true
allowed-tools:
  - read-file
  - write-file
tags:
  - custom
---

# {name}

## Purpose

Describe the purpose and goals of this skill.

## Instructions

Provide step-by-step instructions for how the agent should
execute this skill. Be specific about:

1. What inputs to expect
2. What steps to perform
3. What output format to produce

## Constraints

- List any constraints or limitations
- Specify what the skill should NOT do
"""
