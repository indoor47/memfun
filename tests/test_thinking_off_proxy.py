"""evals/swarm/thinking_off_proxy.py: adds enable_thinking=false to chat requests (hermetic)."""

from __future__ import annotations

import http.client
import http.server
import json
import sys
import threading
from pathlib import Path
from typing import ClassVar

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import thinking_off_proxy as tp  # noqa: E402


class _Upstream(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        out = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self) -> None:
        out = b'{"data": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a: object) -> None:
        pass


def _serve(handler: type[http.server.BaseHTTPRequestHandler]) -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(port: int, path: str, doc: dict) -> int:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(doc), {"Content-Type": "application/json"})
    status = c.getresponse().status
    c.close()
    return status


def test_chat_requests_get_enable_thinking_false_and_other_paths_pass_through() -> None:
    _Upstream.seen = []
    up = _serve(_Upstream)
    tp.Handler.upstream_port = up.server_address[1]
    proxy = _serve(tp.Handler)
    port = proxy.server_address[1]
    try:
        assert _post(port, "/v1/chat/completions", {"model": "m", "messages": []}) == 200
        assert _Upstream.seen[-1]["chat_template_kwargs"] == {"enable_thinking": False}
        # a caller that sets it explicitly keeps its own value; other kwargs are kept
        _post(
            port,
            "/v1/chat/completions",
            {"chat_template_kwargs": {"enable_thinking": True, "x": 1}, "model": "m"},
        )
        assert _Upstream.seen[-1]["chat_template_kwargs"] == {"enable_thinking": True, "x": 1}
        # not a chat request: body untouched
        _post(port, "/v1/completions", {"prompt": "p"})
        assert "chat_template_kwargs" not in _Upstream.seen[-1]
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/v1/models")
        assert json.loads(c.getresponse().read()) == {"data": []}
    finally:
        proxy.shutdown()
        up.shutdown()
