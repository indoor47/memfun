"""Remote access to the dashboard: token auth for non-loopback clients.

``memfun chat`` binds the dashboard to 0.0.0.0, so anything on the LAN or
VPN can reach it.  Loopback stays token-free; every other client needs the
token, and with no token configured remote access is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from memfun_cli.dashboard import server
from memfun_cli.dashboard.server import create_app
from starlette.websockets import WebSocketDisconnect

if TYPE_CHECKING:
    from pathlib import Path

TOKEN = "s3cret-token-for-tests"
REMOTE = ("10.0.0.9", 50000)
LOCAL = ("127.0.0.1", 50000)


@pytest.fixture(autouse=True)
def _isolated_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MEMFUN_DASHBOARD_TOKEN", raising=False)
    monkeypatch.setattr(server, "_TOKEN_FILE", tmp_path / "dashboard_token")


def _client(addr: tuple[str, int]) -> TestClient:
    return TestClient(create_app(), client=addr)


def test_loopback_needs_no_token() -> None:
    assert _client(LOCAL).get("/api/state").status_code == 200


def test_remote_refused_when_no_token_is_configured() -> None:
    # Fail closed: a 0.0.0.0 bind without a token must not expose the dashboard.
    r = _client(REMOTE).get("/api/state")
    assert r.status_code == 403


def test_remote_with_wrong_or_missing_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    c = _client(REMOTE)
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", params={"token": "nope"}).status_code == 401
    assert c.post("/api/event", json={"event_type": "x"}).status_code == 401


def test_remote_query_token_sets_cookie_for_websocket_and_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    c = _client(REMOTE)
    page = c.get("/", params={"token": TOKEN})
    assert page.status_code == 200
    cookie = page.headers["set-cookie"]
    assert "memfun_dash=" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    # The page's own fetches and WebSocket carry the cookie, not the query.
    assert c.get("/api/state").status_code == 200
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"


def test_bearer_header_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    r = _client(REMOTE).get("/api/state", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_remote_websocket_without_token_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        _client(REMOTE).websocket_connect("/ws") as ws,
    ):
        ws.receive_json()
    assert exc.value.code == 1008


def test_remote_health_probe_reveals_only_liveness() -> None:
    # Apps probe reachability before they have a token; no project name leaks.
    r = _client(REMOTE).get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_token_file_is_created_private(tmp_path: Path) -> None:
    token = server.ensure_dashboard_token()
    path = server._TOKEN_FILE
    assert token and path.read_text().strip() == token
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert server.ensure_dashboard_token() == token  # stable once created


def test_non_ip_peer_counts_as_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    # Behind a local proxy uvicorn may report e.g. ("unknown", 0); that must
    # not be mistaken for loopback.
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    assert _client(("unknown", 0)).get("/api/state").status_code == 401
    assert _client(("testclient", 50000)).get("/api/state").status_code == 401


def test_non_ascii_token_is_rejected_not_a_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", TOKEN)
    assert _client(REMOTE).get("/api/state", params={"token": "é"}).status_code == 401


def test_cross_origin_websocket_refused_even_from_loopback() -> None:
    # Any web page in a browser on this Mac could otherwise read the swarm state.
    c = _client(LOCAL)
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        c.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws,
    ):
        ws.receive_json()
    assert exc.value.code == 1008


def test_cross_origin_post_refused_same_origin_allowed() -> None:
    c = _client(LOCAL)
    evil = c.post(
        "/api/event", json={"event_type": "x"}, headers={"origin": "https://evil.example"}
    )
    assert evil.status_code == 403
    same = c.post("/api/event", json={"event_type": "x"}, headers={"origin": "http://testserver"})
    assert same.status_code == 200
    # Non-browser clients (memfun processes posting events) send no Origin.
    assert c.post("/api/event", json={"event_type": "x"}).status_code == 200


def test_page_strips_token_from_the_address_bar() -> None:
    html = _client(LOCAL).get("/").text
    assert "searchParams.delete('token')" in html and "history.replaceState" in html
