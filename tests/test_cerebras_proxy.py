"""The Cerebras rate-limiting proxy against an in-process fake upstream (no network, no key)."""

from __future__ import annotations

import http.server
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import cerebras_proxy as cp  # noqa: E402
from evals.swarm import run_smoke as rs  # noqa: E402

SECRET = "csk-test-secret-value"
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 300,
    "total_tokens": 1300,
    "prompt_tokens_details": {"cached_tokens": 128},
    "completion_tokens_details": {"reasoning_tokens": 250},
}


class Upstream:
    """Fake Cerebras: ``message.reasoning`` next to ``content``, scripted failures."""

    def __init__(self) -> None:
        self.script: list[tuple[int, dict[str, str]]] = []  # failures served before success
        self.seen: list[dict[str, Any]] = []
        self.auth: list[str] = []
        self.agents: list[str] = []
        self.stream_lines: list[bytes] | None = None  # raw SSE lines instead of the default
        self.fail_status = 0  # answer every POST with this status (a non-retryable 4xx)
        up = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:
                pass

            def _send(self, status: int, doc: Any, headers: dict[str, str] | None = None) -> None:
                data = json.dumps(doc).encode()
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                up.auth.append(self.headers.get("authorization", ""))
                self._send(200, {"data": [{"id": "qwen-3.8-27b"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                up.seen.append(body)
                up.auth.append(self.headers.get("authorization", ""))
                up.agents.append(self.headers.get("user-agent", ""))
                if up.fail_status:
                    self._send(up.fail_status, {"error": {"message": "bad request"}})
                    return
                if up.script:
                    status, headers = up.script.pop(0)
                    msg = "Too many tokens: uncached tokens per minute" if status == 429 else "x"
                    self._send(status, {"error": {"message": msg}}, headers)
                    return
                msg = {"role": "assistant", "content": "42", "reasoning": "think think"}
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    if up.stream_lines is not None:
                        for raw in up.stream_lines:
                            self.wfile.write(raw + b"\n\n")
                        return
                    for delta in ({"reasoning": "think"}, {"content": "42"}):
                        chunk = {"choices": [{"index": 0, "delta": delta}]}
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    final = {"choices": [], "usage": USAGE}
                    self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                doc = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
                self._send(200, {**doc, "usage": USAGE})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def upstream() -> Any:
    u = Upstream()
    yield u
    u.close()


def _proxy(upstream: Upstream, **kw: Any) -> cp.CerebrasProxy:
    kw.setdefault("sleep", lambda _s: None)
    kw.setdefault("backoff_base", 0.0)
    return cp.CerebrasProxy(SECRET, upstream=upstream.url, **kw)


def _post(base: str, ep: str, body: dict[str, Any] | None = None) -> tuple[int, Any, Any]:
    req = urllib.request.Request(
        f"{base}/e/{ep}/v1/chat/completions",
        data=json.dumps(body or {"model": "m", "messages": [], "max_tokens": 100}).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer dummy"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), exc.headers


def test_reasoning_field_passes_through_and_usage_is_accounted(upstream: Upstream) -> None:
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        status, doc, _ = _post(base, "ep1")
        assert status == 200
        msg = doc["choices"][0]["message"]
        assert msg["content"] == "42" and msg["reasoning"] == "think think"
        snap = proxy.accounts.get("ep1").as_dict()
        assert snap["prompt_tokens"] == 1000 and snap["completion_tokens"] == 300
        assert snap["reasoning_tokens"] == 250 and snap["cached_tokens"] == 128
        assert snap["cost_usd"] == pytest.approx(1000 * 0.99e-6 + 300 * 1.49e-6)
    finally:
        proxy.stop()


def test_key_is_injected_and_the_clients_own_header_is_not_forwarded(upstream: Upstream) -> None:
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        _post(base, "ep1")
        assert upstream.auth == [f"Bearer {SECRET}"]
    finally:
        proxy.stop()


def test_a_429_is_retried_transparently_and_never_reaches_the_episode(upstream: Upstream) -> None:
    upstream.script = [(429, {"retry-after": "0"}), (429, {}), (503, {})]
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        status, doc, _ = _post(base, "ep1")
        assert status == 200 and doc["choices"][0]["message"]["content"] == "42"
        snap = proxy.accounts.get("ep1")
        assert snap.retries_429 == 2 and snap.retries_other == 1 and snap.gave_up == 0
        assert len(upstream.seen) == 4
    finally:
        proxy.stop()


def test_persistent_429_gives_up_with_a_marked_provider_error(upstream: Upstream) -> None:
    upstream.script = [(429, {})] * 10
    proxy = _proxy(upstream, max_retries=3)
    base = proxy.start()
    try:
        status, _doc, headers = _post(base, "ep1")
        assert status in (429, 503)
        assert headers.get("x-memfun-provider-gave-up") == "1"
        assert proxy.accounts.get("ep1").gave_up == 1
    finally:
        proxy.stop()


def test_spend_cap_refuses_with_402_and_stops_calling_upstream(upstream: Upstream) -> None:
    proxy = _proxy(upstream, max_cost_usd=0.0015)  # one call costs about $0.00144 + ...
    base = proxy.start()
    try:
        assert _post(base, "a")[0] == 200
        assert _post(base, "b")[0] == 200  # 0.00144 < cap: the second still runs
        status, doc, _ = _post(base, "c")
        assert status == 402 and "spend cap" in doc["error"]["message"]
        assert len(upstream.seen) == 2
        assert proxy.accounts.get("c").spend_cap_refusals == 1
    finally:
        proxy.stop()


def test_streaming_is_relayed_and_usage_counted(upstream: Upstream) -> None:
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        req = urllib.request.Request(
            f"{base}/e/s1/v1/chat/completions",
            data=json.dumps(
                {"model": "m", "messages": [], "stream": True, "max_tokens": 50}
            ).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
        assert '"reasoning": "think"' in text and "[DONE]" in text
        snap = proxy.accounts.get("s1")
        assert snap.completion_tokens == 300 and snap.reasoning_tokens == 250
    finally:
        proxy.stop()


def test_reasoning_effort_default_is_injected_but_never_overrides(upstream: Upstream) -> None:
    proxy = _proxy(upstream, reasoning_effort="none")
    base = proxy.start()
    try:
        _post(base, "e")
        _post(base, "e", {"model": "m", "messages": [], "reasoning_effort": "high"})
        assert [b["reasoning_effort"] for b in upstream.seen] == ["none", "high"]
    finally:
        proxy.stop()


def test_models_endpoint_is_forwarded_and_other_paths_are_refused(upstream: Upstream) -> None:
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
            assert json.loads(r.read())["data"][0]["id"] == "qwen-3.8-27b"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/v1/files", timeout=10)
        assert exc.value.code in (403, 404)
    finally:
        proxy.stop()


def test_limiter_queues_when_a_bucket_is_empty_and_settle_gives_back() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    lim = cp.Limiter(
        cp.Limits(rpm=600, uncached_tpm=60_000, total_tpm=600_000, safety=1.0),
        clock=lambda: now[0],
        sleep=sleep,
    )
    r1 = lim.acquire(50_000, 50_000)
    assert not slept  # fits the full bucket
    r2 = lim.acquire(50_000, 50_000)  # only 10k left: has to wait for 40k at 1k/s
    assert slept and now[0] >= 39.0
    lim.settle(r1, 1_000, 1_000)  # real usage was tiny: the quota comes back
    lim.settle(r2, 1_000, 1_000)
    before = len(slept)
    lim.acquire(50_000, 50_000)
    assert len(slept) == before


def test_estimate_counts_the_output_cap_like_cerebras() -> None:
    unc, tot = cp.estimate_tokens({"max_tokens": 16_384}, 32_000, 0.0)
    assert tot == pytest.approx(32_000 / cp.CHARS_PER_TOKEN + 16_384)
    assert unc == tot
    unc2, _ = cp.estimate_tokens({"max_completion_tokens": 10**9}, 3_200, 0.5)
    assert unc2 == pytest.approx(500 + cp.MAX_OUTPUT_TOKENS)  # capped at the paid-tier maximum


def test_read_key_never_echoes_the_key_in_errors(tmp_path: Path) -> None:
    f = tmp_path / "key"
    f.write_text(f"  {SECRET}\n")
    f.chmod(0o600)
    assert cp.read_key(f) == SECRET
    with pytest.raises(SystemExit) as exc:
        cp.read_key(tmp_path / "missing")
    assert SECRET not in str(exc.value)


def test_proxy_sends_its_own_user_agent_cloudflare_blocks_the_default(upstream: Upstream) -> None:
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        _post(base, "ep1")
        assert upstream.agents == [cp.USER_AGENT] and "Python" not in cp.USER_AGENT
    finally:
        proxy.stop()


def _stream(base: str, ep: str, max_tokens: int = 50) -> str:
    body = {"model": "m", "messages": [], "stream": True, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{base}/e/{ep}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def test_the_runners_preflight_works_through_the_proxy(upstream: Upstream) -> None:
    # the live run died here once: "_preflight" was rejected as an episode id (HTTP 400)
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        model, fatal = rs.preflight(f"{base}/e/_preflight/v1", None)
        assert (model, fatal) == ("qwen-3.8-27b", "")
    finally:
        proxy.stop()


def test_a_malformed_stream_event_neither_breaks_the_relay_nor_leaks_the_reservation(
    upstream: Upstream,
) -> None:
    upstream.stream_lines = [
        b"data: [1, 2]",
        b'data: {"choices": [1, {"delta": 5}]}',
        b"data: not json",
        b'data: {"choices": [{"delta": {"content": "42"}}]}',
        b"data: [DONE]",
    ]
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        before = proxy.limiter.state()["uncached_available"]
        text = _stream(base, "m1")
        assert "42" in text and "[DONE]" in text
        snap = proxy.accounts.get("m1")
        assert snap.requests == 1 and snap.unmetered_streams == 1 and snap.prompt_tokens > 0
        assert proxy.accounts.over_cap() is False
        assert proxy.accounts._reserved == 0  # pyright: ignore[reportPrivateUsage]
        # no usage block seen: the reservation is deliberately kept (safe side), not leaked twice
        assert proxy.limiter.state()["uncached_available"] <= before
    finally:
        proxy.stop()


def test_a_4xx_from_upstream_gives_the_quota_back_and_costs_nothing(upstream: Upstream) -> None:
    upstream.fail_status = 400
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        before = proxy.limiter.state()["uncached_available"]
        assert _post(base, "bad")[0] == 400
        assert proxy.limiter.state()["uncached_available"] >= before - 5  # refilled, not held
        snap = proxy.accounts.get("bad")
        assert snap.requests == 0 and snap.prompt_tokens == 0
    finally:
        proxy.stop()


def test_in_flight_worst_case_counts_against_the_cap(upstream: Upstream) -> None:
    proxy = _proxy(upstream, max_cost_usd=0.10)
    base = proxy.start()
    try:
        proxy.accounts.reserve(0.10)  # what a big request in flight holds
        status, doc, _ = _post(base, "c")
        assert status == 402 and "spend cap" in doc["error"]["message"]
        proxy.accounts.release(0.10)
        assert _post(base, "c")[0] == 200
        assert proxy.accounts._reserved == 0  # pyright: ignore[reportPrivateUsage]
    finally:
        proxy.stop()


def test_a_client_that_hangs_up_before_reading_still_gets_charged(upstream: Upstream) -> None:
    import socket

    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        port = int(base.rsplit(":", 1)[1])
        body = json.dumps({"model": "m", "messages": [], "max_tokens": 50}).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        sock.sendall(
            b"POST /e/hang/v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1:%d\r\n"
            b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n"
            % (port, len(body))
            + body
        )
        sock.close()  # gone before the answer
        for _ in range(100):
            if proxy.accounts.get("hang").requests:
                break
            import time

            time.sleep(0.05)
        snap = proxy.accounts.get("hang")
        assert snap.requests == 1 and snap.prompt_tokens == 1000  # usage parsed before writing
    finally:
        proxy.stop()


def test_wrong_host_header_and_form_posts_are_refused(upstream: Upstream) -> None:
    import http.client

    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        port = int(base.rsplit(":", 1)[1])
        for host, ctype, want in (
            ("evil.example", "application/json", 421),  # DNS rebinding
            (f"127.0.0.1:{port}", "text/plain", 415),  # cross-site form POST
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.putrequest("POST", "/e/x/v1/chat/completions", skip_host=True)
            conn.putheader("Host", host)
            conn.putheader("Content-Type", ctype)
            conn.putheader("Content-Length", "2")
            conn.endheaders(b"{}")
            assert conn.getresponse().status == want
            conn.close()
        assert upstream.seen == []
    finally:
        proxy.stop()


def test_a_429_before_a_streamed_answer_is_retried_transparently(upstream: Upstream) -> None:
    upstream.script = [(429, {"retry-after": "0"})]
    proxy = _proxy(upstream)
    base = proxy.start()
    try:
        text = _stream(base, "s429")
        assert '"content": "42"' in text and "[DONE]" in text
        snap = proxy.accounts.get("s429")
        assert snap.retries_429 == 1 and snap.requests == 1 and snap.completion_tokens == 300
    finally:
        proxy.stop()
