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
