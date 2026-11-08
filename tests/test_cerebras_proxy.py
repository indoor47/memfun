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
