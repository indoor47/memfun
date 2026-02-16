from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import typer
from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel

console = Console()

ASCII_BANNER = """\
 ███╗   ███╗███████╗███╗   ███╗███████╗██╗   ██╗███╗   ██╗
 ████╗ ████║██╔════╝████╗ ████║██╔════╝██║   ██║████╗  ██║
 ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██║   ██║██╔██╗ ██║
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║   ██║██║╚██╗██║
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║     ╚██████╔╝██║ ╚████║
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═══╝"""


def init_command(
    backend: str | None = typer.Option(
        None, help="Backend: memory, sqlite, redis, nats"
    ),
    llm: str | None = typer.Option(
        None, help="LLM provider: anthropic, openai, ollama, custom"
    ),
    sandbox: str | None = typer.Option(
        None, help="Sandbox: local, docker, modal"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-y"
    ),
) -> None:
    """Initialize Memfun (global setup + project init)."""
    global_config = Path.home() / ".memfun" / "config.toml"
    project_dir = Path.cwd() / ".memfun"

    # Step 1: Global setup if needed
    if not global_config.exists():
        if non_interactive:
            _non_interactive_global_setup(llm)
        else:
            run_global_setup()
    else:
        console.print(
            "[dim]Global config: ~/.memfun/config.toml (exists)[/dim]"
        )

    # Step 2: Project init if needed
    if not project_dir.exists():
        if non_interactive:
            run_project_init(
                backend=backend or "sqlite",
                sandbox=sandbox or "local",
            )
            console.print(
                f"[green]✓[/green] Project initialized: "
                f"{Path.cwd().name}/"
            )
        else:
            _interactive_project_init(backend, sandbox)
    else:
        console.print("[dim]Project config: .memfun/ (exists)[/dim]")

    console.print()
    console.print(
        "[green]Ready.[/green] Run [bold cyan]memfun[/bold cyan] "
        "to start chatting."
    )


def run_global_setup() -> None:
    """First-time global setup: LLM provider + API key."""
    console.print()
    console.print(Panel.fit(
        ASCII_BANNER,
        border_style="cyan",
        title="[bold cyan]Welcome to Memfun[/bold cyan]",
        subtitle="Autonomous coding agent",
    ))
    console.print()
    console.print("[bold]First-time setup[/bold]")
    console.print()

    # LLM Provider
    provider = inquirer.select(
        message="Which LLM provider?",
        choices=[
            {
                "name": "Anthropic (Claude) — recommended",
                "value": "anthropic",
            },
            {"name": "OpenAI (GPT)", "value": "openai"},
            {"name": "Ollama / vLLM (local)", "value": "ollama"},
            {"name": "Custom endpoint", "value": "custom"},
        ],
        default="anthropic",
    ).execute()

    config = {"llm": {"provider": provider}}

    # API key
    env_var = _get_api_key_env(provider)
    api_key = None
    if env_var:
        config["llm"]["api_key_env"] = env_var
        existing = os.environ.get(env_var, "")
        if existing:
            console.print(
                f"  [green]✓[/green] Found {env_var} in environment"
            )
            api_key = existing
        else:
            key = inquirer.secret(
                message=f"Paste your API key ({env_var})",
            ).execute()
            if key and key.strip():
                api_key = key.strip()
                console.print("  [green]✓[/green] API key saved")
            else:
                console.print(
                    f"  [yellow]![/yellow] Set {env_var} before "
                    "using Memfun"
                )

    if provider == "custom":
        base_url = inquirer.text(
            message="Enter base URL",
            default="http://localhost:11434/v1",
        ).execute()
        config["llm"]["base_url"] = base_url

    # Write global config
    global_dir = Path.home() / ".memfun"
    global_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        global_dir.chmod(0o700)

    _write_toml(global_dir / "config.toml", config)

    # Save credentials
    if api_key and env_var:
        creds_path = global_dir / "credentials.json"
        creds_path.write_text(json.dumps({env_var: api_key}, indent=2))
        creds_path.chmod(0o600)
        os.environ[env_var] = api_key
        console.print(
            "  [green]✓[/green] Credentials saved to "
            "~/.memfun/credentials.json"
        )

    # Create dirs
    (global_dir / "agents").mkdir(exist_ok=True)
    (global_dir / "skills").mkdir(exist_ok=True)

    console.print()
    console.print("[green]✓[/green] Global setup complete")
    console.print()


def run_project_init(
    project_dir: Path | None = None,
    backend: str = "sqlite",
    sandbox: str = "local",
) -> None:
    """Initialize .memfun/ in a project directory."""
    if project_dir is None:
        project_dir = Path.cwd()

    memfun_dir = project_dir / ".memfun"
    memfun_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        memfun_dir.chmod(0o700)

    project_name = project_dir.name

    # Write project config
    config = {
        "project": {"name": project_name},
        "backend": {"tier": backend},
        "sandbox": {"backend": sandbox},
    }
    if backend == "sqlite":
        config["backend"]["sqlite_path"] = ".memfun/memfun.db"
    elif backend == "redis":
        redis_url = os.environ.get(
            "MEMFUN_REDIS_URL", "redis://localhost:6379",
        )
        config["backend"]["redis_url"] = redis_url
    elif backend == "nats":
        nats_url = os.environ.get(
            "MEMFUN_NATS_URL", "nats://localhost:4222",
        )
        config["backend"]["nats_url"] = nats_url

    config_path = memfun_dir / "config.toml"
    if not config_path.exists():
        _write_toml(config_path, config)

    # Create subdirectories
    for subdir in ("conversations", "tasks", "agents", "skills"):
        (memfun_dir / subdir).mkdir(exist_ok=True)

    # Create starter MEMORY.md
    from memfun_cli.memory import create_starter_memory

    create_starter_memory(memfun_dir / "MEMORY.md")

    # Migration: move old conversation_history.json
    old_history = memfun_dir / "conversation_history.json"
    if old_history.exists():
        convs_dir = memfun_dir / "conversations"
        dest = convs_dir / "migrated.json"
        if not dest.exists():
            old_history.rename(dest)


def _interactive_project_init(
    backend: str | None, sandbox: str | None
) -> None:
    """Interactive project-level setup."""
    console.print()
    console.print(
        f"[bold]Project:[/bold] [cyan]{Path.cwd().name}[/cyan]"
    )

    setup_mode = inquirer.select(
        message="Setup mode:",
        choices=[
            {
                "name": "Quick — defaults (SQLite + local)",
                "value": "quick",
            },
            {
                "name": "Custom — choose backend & sandbox",
                "value": "custom",
            },
        ],
        default="quick",
    ).execute()

    if setup_mode == "quick":
        run_project_init()
        console.print(
            "[green]✓[/green] Project initialized with defaults"
        )
        return

    # Custom: pick backend
    if not backend:
        backend = inquirer.select(
            message="Runtime backend:",
            choices=[
                {"name": "SQLite (recommended)", "value": "sqlite"},
                {"name": "In-memory", "value": "memory"},
                {"name": "Redis", "value": "redis"},
                {"name": "NATS", "value": "nats"},
            ],
            default="sqlite",
        ).execute()

    if backend == "redis":
        redis_url = inquirer.text(
            message="Redis URL:",
            default="redis://localhost:6379",
        ).execute()
        os.environ["MEMFUN_REDIS_URL"] = redis_url

    if backend == "nats":
        nats_url = inquirer.text(
            message="NATS URL:",
            default="nats://localhost:4222",
        ).execute()
        os.environ["MEMFUN_NATS_URL"] = nats_url

    if not sandbox:
        sandbox = inquirer.select(
            message="Sandbox:",
            choices=[
                {"name": "Local (recommended)", "value": "local"},
                {"name": "Docker", "value": "docker"},
                {"name": "Modal", "value": "modal"},
            ],
            default="local",
        ).execute()

    run_project_init(backend=backend, sandbox=sandbox)
    console.print("[green]✓[/green] Project initialized")


def _write_toml(path: Path, config: dict) -> None:
    """Write a dict as TOML to a file."""
    lines = []
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                lines.append(f'{k} = "{v}"')
            lines.append("")
    path.write_text("\n".join(lines))


def _non_interactive_global_setup(llm: str | None) -> None:
    """Non-interactive global setup with defaults."""
    global_dir = Path.home() / ".memfun"
    global_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        global_dir.chmod(0o700)

    provider = llm or "anthropic"
    config = {
        "llm": {
            "provider": provider,
            "api_key_env": (
                _get_api_key_env(provider) or "ANTHROPIC_API_KEY"
            ),
        },
    }
    _write_toml(global_dir / "config.toml", config)
    (global_dir / "agents").mkdir(exist_ok=True)
    (global_dir / "skills").mkdir(exist_ok=True)


def _get_api_key_env(provider: str) -> str | None:
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,
        "custom": "MEMFUN_API_KEY",
    }.get(provider)
