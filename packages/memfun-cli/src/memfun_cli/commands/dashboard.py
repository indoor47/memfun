"""``memfun dashboard`` — standalone dashboard server.

Starts only the FastAPI dashboard, without a chat REPL.  Reads state
from ``.memfun/dashboard.json`` (written by previous chat sessions) and
broadcasts live events ingested via the ``/api/event`` HTTP endpoint.

Use when you want to watch a swarm / chat session from a headless
machine, a Docker container, or over an SSH tunnel without holding
open a chat prompt locally.

Example::

    memfun dashboard --port 8081
    # then in another terminal:
    memfun swarm "..."
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger("cli.dashboard")
console = Console()


def dashboard_command(
    port: int = typer.Option(
        8081,
        "--port",
        min=1,
        max=65535,
        help="Port to bind.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind.",
    ),
) -> None:
    """Start ONLY the dashboard server (no chat REPL required).

    Useful for headless / Docker / SSH-tunnel scenarios.
    """
    project_root = Path.cwd()
    memfun_dir = project_root / ".memfun"
    if not memfun_dir.exists():
        console.print(
            Panel(
                f"[red]No [bold].memfun/[/bold] directory in "
                f"{project_root}.[/red]\n\n"
                "Run [bold]memfun init[/bold] first, or [bold]cd[/bold] "
                "into a project that has already been initialised.",
                title="Dashboard error",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    try:
        from memfun_cli.dashboard import server as _dashboard_mod
        from memfun_cli.dashboard.server import create_app
    except ImportError as exc:
        console.print(
            Panel(
                f"[red]Missing dependency: {exc}[/red]\n\n"
                "Install with: [bold]pip install fastapi uvicorn[/bold]",
                title="Dashboard error",
                border_style="red",
            )
        )
        raise typer.Exit(code=2) from exc

    # Set the port BEFORE create_app() so the lifespan can write the
    # lockfile to ``.memfun/dashboard.lock``.
    _dashboard_mod._dashboard_port = port

    app = create_app(
        event_bus=None,  # standalone: read state from disk + HTTP ingest
        project_name=project_root.name,
    )

    url = f"http://{host}:{port}"
    console.print(
        Panel(
            f"[bold]Dashboard URL:[/bold] {url}\n"
            f"[dim]project:[/dim] {project_root.name}\n"
            f"[dim]state file:[/dim] {memfun_dir / 'dashboard.json'}\n\n"
            "Press [bold]Ctrl-C[/bold] to stop.",
            title="memfun dashboard",
            border_style="cyan",
        )
    )

    # Block on uvicorn — Ctrl-C cleanly stops the server via uvicorn's
    # signal handler.
    _run_uvicorn(app, host=host, port=port)


def _run_uvicorn(
    app: object,
    *,
    host: str,
    port: int,
) -> None:
    """Run uvicorn with the dashboard app.

    Extracted as a small wrapper so tests can patch this entry-point
    without spinning up a real HTTP server.
    """
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
