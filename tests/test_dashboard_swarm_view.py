"""Tests for the dashboard's swarm view support.

The views themselves are client-side canvas code; what is testable on
the server side is (1) the Prometheus parser that feeds the inference
server strip, (2) that ``/api/server-metrics`` only ever fetches the
operator-configured URL and degrades instead of erroring, and (3) that
the page ships the three views and no longer draws parallel sub-tasks
as an arrow chain.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from memfun_cli.dashboard import server
from memfun_cli.dashboard.server import create_app, parse_inference_metrics

SGLANG_TEXT = """\
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="qwen",dp="0"} 10
sglang:num_running_reqs{model_name="qwen",dp="1"} 6
sglang:num_queue_reqs{model_name="qwen"} 2
sglang:token_usage{model_name="qwen",dp="0"} 0.30
sglang:token_usage{model_name="qwen",dp="1"} 0.50
sglang:cache_hit_rate{model_name="qwen"} 0.875
sglang:gen_throughput{model_name="qwen"} 4321.5
sglang:unrelated_metric 99
"""


def test_sglang_counts_are_summed_and_ratios_averaged() -> None:
    # Two data-parallel ranks: requests add up, KV usage is a mean, not 0.8.
    m = parse_inference_metrics(SGLANG_TEXT)
    assert m == {
        "running": 16.0,
        "queued": 2.0,
        "kv_usage": pytest.approx(0.40),
        "cache_hit_rate": 0.875,
        "gen_tps": 4321.5,
    }


def test_vllm_names_map_to_the_same_fields() -> None:
    m = parse_inference_metrics(
        "vllm:num_requests_running 3\nvllm:num_requests_waiting 1\nvllm:kv_cache_usage_perc 0.2\n"
    )
    assert (m["running"], m["queued"], m["kv_usage"]) == (3.0, 1.0, 0.2)
    # Missing families are reported as unknown, never as zero.
    assert m["cache_hit_rate"] is None and m["gen_tps"] is None


def test_malformed_lines_are_skipped() -> None:
    m = parse_inference_metrics("sglang:num_running_reqs\nsglang:num_queue_reqs{x} not-a-number\n")
    assert m["running"] is None and m["queued"] is None


def test_server_metrics_disabled_without_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMFUN_INFERENCE_METRICS_URL", raising=False)
    client = TestClient(create_app())
    assert client.get("/api/server-metrics").json() == {"enabled": False}


def test_server_metrics_fetch_only_the_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_fetch(url: str, timeout: float) -> str:
        seen.append(url)
        return SGLANG_TEXT

    monkeypatch.setenv("MEMFUN_INFERENCE_METRICS_URL", "http://127.0.0.1:30000/metrics")
    monkeypatch.setattr(server, "_fetch_text", fake_fetch)
    client = TestClient(create_app())
    body = client.get("/api/server-metrics", params={"url": "http://evil.example/"}).json()
    assert seen == ["http://127.0.0.1:30000/metrics"], "a query param must never pick the URL"
    assert body["enabled"] is True and body["ok"] is True and body["running"] == 16.0


def test_unreachable_server_degrades_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, timeout: float) -> str:
        raise OSError("connection refused")

    monkeypatch.setenv("MEMFUN_INFERENCE_METRICS_URL", "http://127.0.0.1:1/metrics")
    monkeypatch.setattr(server, "_fetch_text", boom)
    body = TestClient(create_app()).get("/api/server-metrics").json()
    assert body == {"enabled": True, "ok": False, "error": "OSError"}


def test_page_ships_the_three_views_without_arrow_chain() -> None:
    html = TestClient(create_app()).get("/").text
    for view in ("swarm", "timeline", "tasks"):
        assert f'data-view="{view}"' in html
    assert 'id="swarm-canvas"' in html and 'id="timeline-canvas"' in html
    # Sibling sub-tasks run in parallel; the old flow joined them with arrows.
    assert "task-arrow" not in html.split("<script>", 1)[1]
