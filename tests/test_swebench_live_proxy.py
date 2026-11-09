"""The recording LLM proxy of the SWE-bench-Live runner: hermetic, against a fake upstream."""

from __future__ import annotations

import http.client
import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.swebench_live import proxy as px


class Upstream:
    """A fake OpenAI-compatible server: scripted statuses, records what it was sent."""

    def __init__(self, statuses: list[int] | None = None, stream: bool = False) -> None:
        self.seen: list[dict[str, Any]] = []
        self.auth: list[str] = []
        self.statuses = list(statuses or [])
        up = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._reply(200, json.dumps({"data": [{"id": "m"}]}).encode(), "application/json")

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                up.seen.append({"path": self.path, "body": json.loads(body)})
                up.auth.append(self.headers.get("Authorization") or "")
                status = up.statuses.pop(0) if up.statuses else 200
                if status != 200:
                    self._reply(status, b'{"error": {"message": "busy"}}', "application/json")
                elif stream:
                    data = (
                        b'data: {"choices": [{"delta": {"content": "he"}}]}\n\n'
                        b'data: {"choices": [], "usage": {"prompt_tokens": 7, '
                        b'"completion_tokens": 2}}\n\ndata: [DONE]\n\n'
                    )
                    self._reply(200, data, "text/event-stream")
                else:
                    doc = {
                        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 60},
                        },
                    }
                    self._reply(200, json.dumps(doc).encode(), "application/json")

            def _reply(self, status: int, data: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def call(port: int, token: str, body: Any, method: str = "POST", path: str = "chat/completions"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    conn.request(method, f"/i/{token}/v1/{path}", body=raw if method == "POST" else None)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def lines(traj: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (traj / px.LLM_LOG).read_text("utf-8").splitlines()]


@pytest.fixture
def upstream():
    up = Upstream()
    yield up
    up.stop()


def make(up: Upstream, **kw: Any) -> px.RecordingProxy:
    p = px.RecordingProxy(up.url, sleep=lambda _s: None, **kw)
    p.start()
    return p


def test_records_request_and_response_exactly_and_counts_tokens(upstream, tmp_path: Path) -> None:
    p = make(upstream, api_key="sk-secret-value", price_in=1.0, price_out=2.0)
    try:
        tok = p.register("owner__repo-123", tmp_path / "t")
        req = {"model": "x", "messages": [{"role": "user", "content": "first prompt ü"}]}
        status, data = call(p.port, tok, req)
        assert status == 200 and json.loads(data)["choices"][0]["message"]["content"] == "hello"
        (rec,) = lines(tmp_path / "t")
        assert rec["seq"] == 1 and rec["request"] == req and rec["status"] == 200
        assert rec["response"] == json.loads(data)  # what the agent received
        assert rec["t_response"] >= rec["t_request"] and "forced_by_proxy" not in rec
        u = p.usage(tok)
        assert (u.calls, u.prompt_tokens, u.completion_tokens, u.cached_tokens) == (1, 100, 20, 60)
        assert p.cost_usd(tok) == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
        # the key went upstream and nowhere else; the instance id is not in the path the agent uses
        assert upstream.auth == ["Bearer sk-secret-value"]
        assert "sk-secret-value" not in (tmp_path / "t" / px.LLM_LOG).read_text("utf-8")
        assert "owner__repo-123" not in tok and upstream.seen[0]["path"] == "/v1/chat/completions"
    finally:
        p.stop()


def test_forced_fields_are_applied_and_recorded_next_to_the_original(upstream, tmp_path) -> None:
    p = make(upstream, force_fields={"reasoning_effort": "none", "model": "served"})
    try:
        tok = p.register("i", tmp_path / "t")
        call(p.port, tok, {"model": "served", "messages": [], "reasoning_effort": "high"})
        assert upstream.seen[0]["body"]["reasoning_effort"] == "none"
        (rec,) = lines(tmp_path / "t")
        assert rec["request"]["reasoning_effort"] == "high"  # as the agent sent it
        assert rec["forced_by_proxy"] == {"reasoning_effort": "none"}  # only what changed
    finally:
        p.stop()


def test_token_budget_refuses_further_calls_and_records_the_refusal(upstream, tmp_path) -> None:
    p = make(upstream)
    try:
        tok = p.register("i", tmp_path / "t", max_total_tokens=100)
        assert call(p.port, tok, {"messages": []})[0] == 200  # 120 tokens: over the budget now
        status, data = call(p.port, tok, {"messages": []})
        assert status == 403 and json.loads(data)["error"]["type"] == "budget_exhausted"
        recs = lines(tmp_path / "t")
        assert len(recs) == 2 and recs[1]["refused_by_proxy"] and len(upstream.seen) == 1
        assert p.usage(tok).refused == 1
    finally:
        p.stop()


def test_deadline_and_close_refuse(upstream, tmp_path: Path) -> None:
    now = [1000.0]
    p = make(upstream, clock=lambda: now[0])
    try:
        tok = p.register("i", tmp_path / "t", max_seconds=60)
        assert call(p.port, tok, {"messages": []})[0] == 200
        now[0] += 61
        assert call(p.port, tok, {"messages": []})[0] == 403
        tok2 = p.register("j", tmp_path / "u")
        p.close(tok2)
        assert call(p.port, tok2, {"messages": []})[0] == 403
    finally:
        p.stop()


def test_one_rollout_per_instance_and_unknown_paths(upstream, tmp_path: Path) -> None:
    p = make(upstream)
    try:
        p.register("i", tmp_path / "t")
        with pytest.raises(FileExistsError):
            p.register("i", tmp_path / "t")  # never append to an earlier rollout's trajectory
        assert call(p.port, "0" * 32, {"messages": []})[0] == 404
        conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=b"{}")
        assert conn.getresponse().status == 404
        conn.close()
    finally:
        p.stop()


def test_upstream_429_is_retried_and_a_dead_upstream_is_a_recorded_502(tmp_path: Path) -> None:
    up = Upstream(statuses=[429, 503])
    p = make(up)
    try:
        tok = p.register("i", tmp_path / "t")
        assert call(p.port, tok, {"messages": []})[0] == 200
        assert lines(tmp_path / "t")[0]["upstream_attempts"] == 3
    finally:
        p.stop()
        up.stop()
    dead = px.RecordingProxy("http://127.0.0.1:9/v1", sleep=lambda _s: None, max_retries=1)
    dead.start()
    try:
        tok = dead.register("i", tmp_path / "d")
        assert call(dead.port, tok, {"messages": []})[0] == 502
        rec = lines(tmp_path / "d")[0]
        assert rec["status"] == 502 and dead.usage(tok).errors == 1
    finally:
        dead.stop()


def test_streamed_answers_are_relayed_and_recorded_raw_with_usage(tmp_path: Path) -> None:
    up = Upstream(stream=True)
    p = make(up)
    try:
        tok = p.register("i", tmp_path / "t")
        status, data = call(p.port, tok, {"messages": [], "stream": True})
        assert status == 200 and data.endswith(b"data: [DONE]\n\n")
        (rec,) = lines(tmp_path / "t")
        assert rec["response"] == data.decode() and rec["usage"]["prompt_tokens"] == 7
    finally:
        p.stop()
        up.stop()


def test_a_body_that_is_not_json_is_recorded_as_text_and_models_is_not_a_call(
    upstream, tmp_path: Path
) -> None:
    p = make(upstream)
    try:
        tok = p.register("i", tmp_path / "t")
        assert call(p.port, tok, None, method="GET", path="models")[0] == 200
        assert lines(tmp_path / "t") == [] and p.usage(tok).calls == 0
        assert call(p.port, tok, None, method="GET", path="chat/completions")[0] == 405
    finally:
        p.stop()


def _watch_answers(p: px.RecordingProxy, traj: Path) -> list[int]:
    """Make *p* note, each time it starts answering a client, how many records are on disk."""
    seen: list[int] = []

    def on_disk() -> int:
        f = traj / px.LLM_LOG
        return len(f.read_text("utf-8").splitlines()) if f.exists() else 0

    send, finish = p._send, p._finish_stream

    def watched_send(h: Any, status: int, body: bytes, ctype: str) -> None:
        seen.append(on_disk())
        send(h, status, body, ctype)

    def watched_finish(h: Any, held: bytes) -> None:
        seen.append(on_disk())
        assert held, "the end of a stream must be held back until the record is written"
        finish(h, held)

    p._send = watched_send  # type: ignore[method-assign]
    p._finish_stream = watched_finish  # type: ignore[method-assign]
    return seen


def test_a_call_is_on_disk_before_its_answer_is_sent(upstream, tmp_path: Path) -> None:
    """The order "record, then answer" for every kind of call that belongs in a trajectory.
    (A client that reads the log right after a refusal used to find the record missing.)"""
    p = make(upstream)
    try:
        traj = tmp_path / "t"
        tok = p.register("i", traj, max_total_tokens=100)
        seen = _watch_answers(p, traj)
        assert call(p.port, tok, {"messages": []})[0] == 200  # forwarded
        assert call(p.port, tok, {"messages": []})[0] == 403  # token budget spent
        conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        conn.putrequest("POST", f"/i/{tok}/v1/chat/completions")
        conn.putheader("Content-Length", str(px.MAX_BODY_BYTES + 1))
        conn.endheaders()
        assert conn.getresponse().status == 413  # too large: refused unread, but recorded
        conn.close()
        p.close(tok)
        assert call(p.port, tok, {"messages": []})[0] == 403  # rollout finished
        assert seen == [1, 2, 3, 4]
        recs = lines(traj)
        assert [r["status"] for r in recs] == [200, 403, 413, 403]
        assert recs[2]["declared_length"] == px.MAX_BODY_BYTES + 1 and recs[2]["request"] is None
        assert p.usage(tok).refused == 3 and len(upstream.seen) == 1
    finally:
        p.stop()
    dead = px.RecordingProxy("http://127.0.0.1:9/v1", sleep=lambda _s: None, max_retries=0)
    dead.start()
    try:
        tok = dead.register("i", tmp_path / "d")
        seen = _watch_answers(dead, tmp_path / "d")
        assert call(dead.port, tok, {"messages": []})[0] == 502  # no answer from the model
        assert seen == [1]
    finally:
        dead.stop()


def test_the_end_of_a_stream_is_sent_only_after_the_record(tmp_path: Path) -> None:
    up = Upstream(stream=True)
    p = make(up)
    try:
        traj = tmp_path / "t"
        tok = p.register("i", traj)
        seen = _watch_answers(p, traj)
        status, data = call(p.port, tok, {"messages": [], "stream": True})
        assert status == 200 and data.endswith(b"data: [DONE]\n\n")
        assert seen == [1]  # the last chunk left the proxy with the record already on disk
        assert lines(traj)[0]["response"] == data.decode()
    finally:
        p.stop()
        up.stop()
