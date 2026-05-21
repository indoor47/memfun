"""Tests for the ``memfun dashboard`` standalone command.

These tests patch the uvicorn entry-point so no real HTTP server is
started.  Coverage:

1. Missing ``.memfun/`` directory -> clean error, exit 2.
2. ``--port`` / ``--host`` are plumbed through to uvicorn.
3. The dashboard module's ``_dashboard_port`` global is set BEFORE
   ``create_app`` is called so the lifespan can write the lockfile.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer
from memfun_cli.commands.dashboard import dashboard_command
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _build_app() -> typer.Typer:
    app = typer.Typer()
    app.command("dashboard")(dashboard_command)
    return app


@pytest.fixture
def initialised_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A tmpdir containing ``.memfun/`` (i.e. ``memfun init`` was run)."""
    (tmp_path / ".memfun").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dashboard_errors_when_no_memfun_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``.memfun/`` the command prints an error and exits 2."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(_build_app(), [])

    assert result.exit_code == 2, result.output
    assert ".memfun" in result.output
    assert "memfun init" in result.output


def test_dashboard_uses_provided_port(
    initialised_project: Path,
) -> None:
    """``memfun dashboard --port 9999`` calls uvicorn.run with port=9999."""
    fake_uvicorn = MagicMock()
    fake_app = MagicMock()

    with (
        patch(
            "memfun_cli.commands.dashboard._run_uvicorn",
            fake_uvicorn,
        ),
        patch(
            "memfun_cli.dashboard.server.create_app",
            return_value=fake_app,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(), ["--port", "9999"],
        )

    assert result.exit_code == 0, result.output
    assert fake_uvicorn.call_count == 1
    kwargs = fake_uvicorn.call_args.kwargs
    assert kwargs["port"] == 9999
    # The dashboard URL panel renders the chosen port.
    assert "9999" in result.output


def test_dashboard_uses_provided_host(
    initialised_project: Path,
) -> None:
    """``--host 0.0.0.0`` is forwarded to uvicorn."""
    fake_uvicorn = MagicMock()
    fake_app = MagicMock()

    with (
        patch(
            "memfun_cli.commands.dashboard._run_uvicorn",
            fake_uvicorn,
        ),
        patch(
            "memfun_cli.dashboard.server.create_app",
            return_value=fake_app,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(),
            ["--host", "0.0.0.0", "--port", "8081"],
        )

    assert result.exit_code == 0, result.output
    kwargs = fake_uvicorn.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"


def test_dashboard_sets_module_port_before_create_app(
    initialised_project: Path,
) -> None:
    """``_dashboard_port`` must be set BEFORE ``create_app`` so the
    FastAPI lifespan can write the lockfile correctly.
    """
    fake_uvicorn = MagicMock()
    fake_app = MagicMock()

    observed: dict[str, int] = {}

    def capture_port_at_create_app(*args: object, **kwargs: object) -> object:
        from memfun_cli.dashboard import server as _dashboard_mod
        observed["port_at_create_app"] = _dashboard_mod._dashboard_port
        return fake_app

    with (
        patch(
            "memfun_cli.commands.dashboard._run_uvicorn",
            fake_uvicorn,
        ),
        patch(
            "memfun_cli.dashboard.server.create_app",
            side_effect=capture_port_at_create_app,
        ),
    ):
        result = CliRunner().invoke(
            _build_app(), ["--port", "9876"],
        )

    assert result.exit_code == 0, result.output
    assert observed["port_at_create_app"] == 9876


# ── Tabbed event panel (Events / Logs / Stats) ────────────────


def test_dashboard_html_contains_tab_markup() -> None:
    """The single-page dashboard exposes the three tab buttons + panes.

    UI-only feature: tabbing the right column of the dashboard.  This
    smoke test asserts the markup the JS depends on is actually in
    the served HTML.
    """
    from fastapi.testclient import TestClient
    from memfun_cli.dashboard.server import create_app

    app = create_app()
    response = TestClient(app).get("/")
    assert response.status_code == 200
    html = response.text

    # Tab bar + per-tab buttons.
    assert "tab-bar" in html
    assert 'data-tab="events"' in html
    assert 'data-tab="logs"' in html
    assert 'data-tab="stats"' in html

    # Tab panes the JS targets by id.
    assert 'id="events"' in html
    assert 'id="logs"' in html
    assert 'id="stats"' in html

    # The new render functions (proves the JS shipped).
    assert "renderLogs" in html
    assert "renderStatsPane" in html


def test_dashboard_api_state_reflects_ingested_events(
    initialised_project: Path,
) -> None:
    """After ingesting a few events via ``/api/event``, ``/api/state``
    returns them — proving the JS will have data to render in the
    Logs / Stats tabs.
    """
    import time

    from fastapi.testclient import TestClient
    from memfun_cli.dashboard import server as _server

    # Reset module state — these globals leak across tests.
    _server._events.clear()
    _server._tasks.clear()
    _server._workers.clear()
    _server._requests.clear()

    app = _server.create_app()
    client = TestClient(app)

    events = [
        {
            "event_type": "worker.online",
            "worker_id": "w1",
            "agent_name": "coder-agent",
            "ts": time.time(),
        },
        {
            "event_type": "task.published",
            "task_id": "t-abc123",
            "agent_name": "coder-agent",
            "workflow_id": "wf-1",
            "ts": time.time(),
        },
        {
            "event_type": "task.picked_up",
            "task_id": "t-abc123",
            "worker_id": "w1",
            "ts": time.time(),
        },
        {
            "event_type": "task.completed",
            "task_id": "t-abc123",
            "worker_id": "w1",
            "success": True,
            "duration_ms": 1234,
            "ts": time.time(),
        },
    ]
    for ev in events:
        r = client.post("/api/event", json=ev)
        assert r.status_code == 200

    state = client.get("/api/state").json()
    assert state["type"] == "state"
    received_types = {e["event_type"] for e in state["events"]}
    # All event types should appear in the snapshot the page first
    # loads — this is what the JS uses to populate logs/stats.
    assert {
        "worker.online",
        "task.published",
        "task.picked_up",
        "task.completed",
    } <= received_types
