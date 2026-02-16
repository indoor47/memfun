from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
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
    project_config = project_dir / "config.toml"
    if not project_config.exists():
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
        redis_url = _setup_service("redis")
        os.environ["MEMFUN_REDIS_URL"] = redis_url

    if backend == "nats":
        nats_url = _setup_service("nats")
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


_DOCKER_CONFIGS: dict[str, dict[str, str | list[str]]] = {
    "redis": {
        "image": "redis:alpine",
        "container": "memfun-redis",
        "ports": ["6379:6379"],
        "default_url": "redis://localhost:6379",
        "extra_args": [],
    },
    "nats": {
        "image": "nats:alpine",
        "container": "memfun-nats",
        "ports": ["4222:4222", "8222:8222"],
        "default_url": "nats://localhost:4222",
        "extra_args": ["-js"],  # enable JetStream
    },
}


def _has_docker() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _docker_start(service: str) -> str:
    """Start a service via Docker and return its URL."""
    cfg = _DOCKER_CONFIGS[service]
    container = str(cfg["container"])
    image = str(cfg["image"])
    default_url = str(cfg["default_url"])
    ports = cfg["ports"]
    extra = cfg["extra_args"]

    # Check if container already exists
    check = subprocess.run(
        ["docker", "inspect", container],
        capture_output=True,
    )
    if check.returncode == 0:
        # Container exists — start it if stopped
        subprocess.run(
            ["docker", "start", container],
            capture_output=True,
        )
        console.print(
            f"  [green]✓[/green] {service.title()} container "
            f"already exists, started"
        )
        return default_url

    # Build docker run command
    cmd = ["docker", "run", "-d", "--name", container]
    assert isinstance(ports, list)
    for p in ports:
        cmd.extend(["-p", str(p)])
    cmd.append(image)
    assert isinstance(extra, list)
    cmd.extend(str(a) for a in extra)

    console.print(f"  [dim]Starting {service} via Docker...[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(
            f"  [red]✗[/red] Failed to start {service}: "
            f"{result.stderr.strip()}"
        )
        console.print(
            "  [yellow]Falling back to manual URL entry[/yellow]"
        )
        return inquirer.text(
            message=f"{service.title()} URL:",
            default=default_url,
        ).execute()

    console.print(
        f"  [green]✓[/green] {service.title()} started "
        f"at {default_url}"
    )
    return default_url


def _setup_service(service: str) -> str:
    """Prompt user to auto-start or connect to an existing service."""
    cfg = _DOCKER_CONFIGS[service]
    default_url = str(cfg["default_url"])
    has_docker = _has_docker()

    choices = []
    if has_docker:
        choices.append({
            "name": f"Auto-start {service.title()} via Docker (recommended)",
            "value": "auto",
        })
    choices.extend([
        {
            "name": f"Connect to existing {service.title()}",
            "value": "custom",
        },
        {
            "name": f"Use default ({default_url})",
            "value": "default",
        },
    ])

    mode = inquirer.select(
        message=f"{service.title()} setup:",
        choices=choices,
        default="auto" if has_docker else "custom",
    ).execute()

    if mode == "auto":
        return _docker_start(service)
    if mode == "default":
        return default_url
    return inquirer.text(
        message=f"{service.title()} URL:",
        default=default_url,
    ).execute()


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
