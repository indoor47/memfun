from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import typer
from InquirerPy import inquirer
from rich.console import Console
from rich.syntax import Syntax

console = Console()

config_app = typer.Typer(
    name="config",
    help="View and manage Memfun configuration",
    invoke_without_command=True,
)


@config_app.callback(invoke_without_command=True)
def config_command(
    ctx: typer.Context,
    show_global: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Show global config only",
    ),
) -> None:
    """View merged configuration."""
    if ctx.invoked_subcommand is not None:
        return

    if show_global:
        path = Path.home() / ".memfun" / "config.toml"
        label = "~/.memfun/config.toml"
    else:
        # Show merged config info
        global_path = Path.home() / ".memfun" / "config.toml"
        project_path = Path.cwd() / ".memfun" / "config.toml"
        if not project_path.exists():
            project_path = Path.cwd() / "memfun.toml"

        # Show both files
        if global_path.exists():
            console.print(
                "[bold]Global[/bold]"
                " (~/.memfun/config.toml):"
            )
            content = global_path.read_text()
            console.print(
                Syntax(
                    content,
                    "toml",
                    theme="monokai",
                )
            )
            console.print()

        if project_path.exists():
            rel = project_path.relative_to(Path.cwd())
            console.print(f"[bold]Project[/bold] ({rel}):")
            content = project_path.read_text()
            console.print(
                Syntax(
                    content,
                    "toml",
                    theme="monokai",
                )
            )
        elif not global_path.exists():
            console.print(
                "[yellow]No config found."
                " Run `memfun init`.[/yellow]"
            )
            raise typer.Exit(1)
        return

    if not path.exists():
        console.print(f"[yellow]{label} not found.[/yellow]")
        raise typer.Exit(1)

    content = path.read_text()
    console.print(f"[bold]{label}:[/bold]")
    console.print(Syntax(content, "toml", theme="monokai"))


@config_app.command("key")
def config_key(
    use_global: bool = typer.Option(
        True,
        "--global/--project",
        help="Store in global or project config",
    ),
) -> None:
    """Add or change an API key interactively."""
    from memfun_core.config import MemfunConfig

    config = MemfunConfig.load()
    provider = config.llm.provider
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "custom": "MEMFUN_API_KEY",
    }.get(provider)

    if not env_var:
        console.print(
            f"[yellow]Provider '{provider}'"
            " doesn't use an API key.[/yellow]"
        )
        return

    console.print(
        f"[bold]Provider:[/bold] {provider}" f" ({env_var})"
    )

    key = inquirer.secret(
        message="Paste your API key",
    ).execute()

    if not key or not key.strip():
        console.print("[yellow]No key provided.[/yellow]")
        return

    key = key.strip()

    if use_global:
        creds_path = (
            Path.home() / ".memfun" / "credentials.json"
        )
    else:
        creds_path = (
            Path.cwd() / ".memfun" / "credentials.json"
        )

    # Load existing creds and merge
    existing = {}
    if creds_path.exists():
        with contextlib.suppress(Exception):
            existing = json.loads(creds_path.read_text())

    existing[env_var] = key
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps(existing, indent=2))
    creds_path.chmod(0o600)

    os.environ[env_var] = key
    console.print("[green]✓[/green] API key saved")


@config_app.command("reset")
def config_reset() -> None:
    """Re-run the global setup wizard."""
    from memfun_cli.commands.init import run_global_setup

    run_global_setup()
