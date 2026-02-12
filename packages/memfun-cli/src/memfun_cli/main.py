from __future__ import annotations

import sys

import typer

from memfun_cli.commands.agent_commands import (
    analyze_command,
    ask_command,
    explain_command,
    fix_command,
    review_command,
)
from memfun_cli.commands.agent_mgmt import agent_app
from memfun_cli.commands.chat import chat_command
from memfun_cli.commands.config import config_app
from memfun_cli.commands.init import init_command
from memfun_cli.commands.invoke import invoke_skill
from memfun_cli.commands.skill import skill_app
from memfun_cli.commands.skill_mgmt import skill_create, skill_invoke, skill_search

app = typer.Typer(
    name="memfun",
    help="Memfun — autonomous coding agent",
    invoke_without_command=True,
)

app.command("init")(init_command)
app.add_typer(
    config_app,
    name="config",
    help="View and manage configuration",
)
app.add_typer(skill_app, name="skill", help="Manage skills")
app.add_typer(agent_app, name="agent", help="Manage agent definitions")

# Register extended skill commands onto the existing skill sub-app
skill_app.command("search")(skill_search)
skill_app.command("invoke")(skill_invoke)
skill_app.command("create")(skill_create)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Launch interactive chat when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        chat_command()


@app.command()
def chat() -> None:
    """Launch the interactive chat interface."""
    chat_command()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question or task for the agent"),
) -> None:
    """Ask the agent a question or give it a task."""
    ask_command(question)


@app.command()
def analyze(
    path: str | None = typer.Argument(
        None, help="File or directory to analyze (defaults to cwd)"
    ),
) -> None:
    """Analyze code at a path for structure, quality, and issues."""
    analyze_command(path)


@app.command()
def fix(
    description: str = typer.Argument(
        ..., help="Description of the bug to fix"
    ),
) -> None:
    """Fix a bug based on its description."""
    fix_command(description)


@app.command()
def review(
    path: str | None = typer.Argument(
        None, help="File or directory to review (defaults to cwd)"
    ),
) -> None:
    """Review code at a path and provide structured feedback."""
    review_command(path)


@app.command()
def explain(
    path: str | None = typer.Argument(
        None, help="File or directory to explain (defaults to cwd)"
    ),
) -> None:
    """Explain how code at a path works."""
    explain_command(path)


@app.command()
def version() -> None:
    """Show the Memfun version."""
    from rich.console import Console
    Console().print("memfun 0.1.0")


def _handle_slash_command() -> bool:
    """Intercept slash-command invocation (``memfun /skill-name [args]``).

    Checks whether the first CLI argument starts with ``/``.  If so,
    strips the leading slash and routes to :func:`invoke_skill`.

    Returns:
        *True* if a slash command was handled (caller should exit),
        *False* otherwise (normal Typer dispatch should proceed).
    """
    # sys.argv[0] is the program name; the first real arg is sys.argv[1]
    if len(sys.argv) < 2:
        return False

    first_arg = sys.argv[1]
    if not first_arg.startswith("/"):
        return False

    skill_name = first_arg.lstrip("/")
    if not skill_name:
        return False

    remaining_args = sys.argv[2:]
    invoke_skill(skill_name, remaining_args)
    return True


def main() -> None:
    """Entry-point that supports both normal commands and /slash invocation."""
    if _handle_slash_command():
        return
    app()


if __name__ == "__main__":
    main()
