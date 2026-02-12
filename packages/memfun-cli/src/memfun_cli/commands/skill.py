"""Skill management commands: list, info, validate."""
from __future__ import annotations

from pathlib import Path

import typer
from memfun_skills import SkillLoader, SkillValidator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

skill_app = typer.Typer(
    no_args_is_help=True,
)


def _build_loader() -> SkillLoader:
    """Build a SkillLoader with default discovery paths."""
    return SkillLoader()


@skill_app.command("list")
def skill_list() -> None:
    """List all discovered skills."""
    loader = _build_loader()
    skills = loader.discover()

    if not skills:
        console.print(
            "[yellow]No skills found.[/yellow] "
            "Place SKILL.md files in ./skills/ or ~/.memfun/skills/."
        )
        raise typer.Exit(0)

    table = Table(
        title="Discovered Skills",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Version", justify="center")
    table.add_column("Tags")

    for skill in sorted(skills, key=lambda s: s.name):
        tags_str = ", ".join(skill.tags) if skill.tags else "-"
        table.add_row(
            skill.name,
            skill.description,
            skill.version,
            tags_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(skills)} skill(s) found.[/dim]")


@skill_app.command("info")
def skill_info(
    name: str = typer.Argument(..., help="Name of the skill to inspect"),
) -> None:
    """Show detailed information about a skill."""
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

    # Metadata panel
    meta_lines = [
        f"[bold]Name:[/bold]        {match.name}",
        f"[bold]Description:[/bold] {match.description}",
        f"[bold]Version:[/bold]     {match.version}",
        f"[bold]Source:[/bold]      {match.source_path}",
        f"[bold]User invocable:[/bold]  {'yes' if match.user_invocable else 'no'}",
        f"[bold]Model invocable:[/bold] {'yes' if match.model_invocable else 'no'}",
    ]

    if match.tags:
        meta_lines.append(f"[bold]Tags:[/bold]        {', '.join(match.tags)}")

    if match.allowed_tools:
        meta_lines.append(
            f"[bold]Allowed tools:[/bold]  {', '.join(match.allowed_tools)}"
        )

    if match.scripts_dir:
        meta_lines.append(f"[bold]Scripts dir:[/bold]  {match.scripts_dir}")

    if match.references_dir:
        meta_lines.append(f"[bold]References dir:[/bold] {match.references_dir}")

    console.print(Panel(
        "\n".join(meta_lines),
        title=f"Skill: {match.name}",
        border_style="cyan",
    ))

    # Instructions preview
    if match.instructions:
        preview = match.instructions
        if len(preview) > 2000:
            preview = preview[:2000] + "\n\n... (truncated)"
        console.print()
        console.print(Panel(
            Syntax(preview, "markdown", theme="monokai", word_wrap=True),
            title="Instructions",
            border_style="dim",
        ))
    else:
        console.print("\n[dim]No instructions body defined.[/dim]")


@skill_app.command("validate")
def skill_validate(
    path: str | None = typer.Argument(
        None,
        help="Path to a skill directory or SKILL.md file. "
        "If omitted, validates all discovered skills.",
    ),
) -> None:
    """Validate skill definition(s) and report errors."""
    validator = SkillValidator()

    if path is not None:
        _validate_single(Path(path), validator)
    else:
        _validate_all(validator)


def _validate_single(path: Path, validator: SkillValidator) -> None:
    """Validate a single skill at *path*."""
    resolved = path.expanduser().resolve()

    # Accept either a directory or a SKILL.md file
    if resolved.is_file() and resolved.name == "SKILL.md":
        skill_file = resolved
    elif resolved.is_dir():
        skill_file = resolved / "SKILL.md"
    else:
        console.print(
            f"[red]Invalid path:[/red] '{path}' is not a directory "
            "or a SKILL.md file."
        )
        raise typer.Exit(1)

    if not skill_file.exists():
        console.print(
            f"[red]SKILL.md not found at:[/red] {skill_file}"
        )
        raise typer.Exit(1)

    from memfun_skills import parse_skill_md

    try:
        skill = parse_skill_md(skill_file)
    except Exception as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        raise typer.Exit(1) from None

    errors = validator.validate(skill)
    _report_validation(skill.name, errors)

    if errors:
        raise typer.Exit(1)


def _validate_all(validator: SkillValidator) -> None:
    """Validate all discovered skills."""
    loader = _build_loader()
    skills = loader.discover()

    if not skills:
        console.print(
            "[yellow]No skills found to validate.[/yellow] "
            "Place SKILL.md files in ./skills/ or ~/.memfun/skills/."
        )
        raise typer.Exit(0)

    total_errors = 0
    for skill in sorted(skills, key=lambda s: s.name):
        errors = validator.validate(skill)
        _report_validation(skill.name, errors)
        total_errors += len(errors)

    console.print()
    if total_errors == 0:
        console.print(
            f"[green]All {len(skills)} skill(s) passed validation.[/green]"
        )
    else:
        console.print(
            f"[red]{total_errors} error(s) across {len(skills)} skill(s).[/red]"
        )
        raise typer.Exit(1)


def _report_validation(name: str, errors: list[str]) -> None:
    """Print validation results for a single skill."""
    if not errors:
        console.print(f"  [green]OK[/green]  {name}")
    else:
        console.print(f"  [red]FAIL[/red] {name}")
        for err in errors:
            console.print(f"        [red]-[/red] {err}")
