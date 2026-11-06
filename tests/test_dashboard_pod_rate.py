"""Server-provided default for the dashboard's pod cost rate.

``MEMFUN_POD_RATE_USD`` lets a launch kit supply the pod $/h so a viewer
that cannot type into the page (a phone app wrapping the dashboard) still
gets a working cost meter.  The server exposes it as ``pod_rate`` on both
``/api/state`` (also the WebSocket snapshot) and ``/api/server-metrics``;
the page prefills the ``pod-rate`` box from it unless the viewer typed one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from memfun_cli.dashboard import server
from memfun_cli.dashboard.server import create_app, pod_rate_default

LOCAL = ("127.0.0.1", 50000)


def _client() -> TestClient:
    return TestClient(create_app(), client=LOCAL)


@pytest.fixture(autouse=True)
def no_metrics_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMFUN_INFERENCE_METRICS_URL", raising=False)


def test_unset_env_reports_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMFUN_POD_RATE_USD", raising=False)
    c = _client()
    assert c.get("/api/state").json()["pod_rate"] is None
    assert c.get("/api/server-metrics").json()["pod_rate"] is None


@pytest.mark.parametrize(("raw", "expected"), [("8.99", 8.99), (" 12 ", 12.0), ("0", 0.0)])
def test_valid_value_is_a_number(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
) -> None:
    monkeypatch.setenv("MEMFUN_POD_RATE_USD", raw)
    assert pod_rate_default() == expected
    c = _client()
    assert c.get("/api/state").json()["pod_rate"] == expected
    # Present even though MEMFUN_INFERENCE_METRICS_URL is unset.
    assert c.get("/api/server-metrics").json() == {"enabled": False, "pod_rate": expected}


@pytest.mark.parametrize("raw", ["", "   ", "abc", "-1", "-0.01", "nan", "inf", "-inf", "$8"])
def test_invalid_values_are_ignored(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("MEMFUN_POD_RATE_USD", raw)
    assert pod_rate_default() is None
    assert _client().get("/api/state").json()["pod_rate"] is None


def test_existing_keys_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_POD_RATE_USD", "8.99")
    state = _client().get("/api/state").json()
    assert set(state) == {"type", "workers", "tasks", "events", "requests", "project", "pod_rate"}
    assert state["type"] == "state"


def test_rate_rides_along_with_live_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_INFERENCE_METRICS_URL", "http://127.0.0.1:30000/metrics")
    monkeypatch.setenv("MEMFUN_POD_RATE_USD", "8.99")

    def fake_fetch(url: str, timeout: float) -> str:
        return "sglang:num_running_reqs 4\n"

    monkeypatch.setattr(server, "_fetch_text", fake_fetch)
    body = _client().get("/api/server-metrics").json()
    assert body["enabled"] is True and body["running"] == 4.0 and body["pod_rate"] == 8.99


def test_websocket_snapshot_carries_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_POD_RATE_USD", "3.5")
    with _client().websocket_connect("/ws") as ws:
        assert ws.receive_json()["pod_rate"] == 3.5


def test_page_prefills_from_server_without_overriding_typed_value() -> None:
    js = _client().get("/").text.split("<script>", 1)[1]
    body = js[js.index("function applyServerPodRate") :][:400]
    assert "el.dataset.userSet === '1'" in body and "el.value = String(v)" in body
    # Both server payloads feed it, and typing marks the value as the user's.
    assert "applyServerPodRate(msg.pod_rate)" in js
    assert "applyServerPodRate(SW.server.pod_rate)" in js
    assert "el.dataset.userSet = el.value ? '1' : ''" in js
