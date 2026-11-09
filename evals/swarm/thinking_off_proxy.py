#!/usr/bin/env python3
"""Loopback proxy: adds chat_template_kwargs={"enable_thinking": false} to every chat request.

For a SGLang pod that was started WITHOUT ``--default-chat-template-kwargs`` (thinking on).
Streaming and non-streaming both pass through; GETs (models, metrics) are forwarded untouched.
Never logs bodies.  Usage: thinking_off_proxy.py --listen 18001 --upstream 18000
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "proxy-authenticate",
    "upgrade",
    "te",
    "trailer",
}


class Handler(BaseHTTPRequestHandler):
    upstream_port = 18000
    stats: ClassVar[dict[str, int]] = {"patched": 0, "other": 0}
    lock: ClassVar[threading.Lock] = threading.Lock()
    protocol_version = "HTTP/1.1"

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        patched = False
        if method == "POST" and self.path.rstrip("/").endswith("/chat/completions"):
            try:
                doc = json.loads(body)
                if isinstance(doc, dict):
                    kw = doc.get("chat_template_kwargs")
                    if not isinstance(kw, dict):
                        kw = {}
                    kw.setdefault("enable_thinking", False)
                    doc["chat_template_kwargs"] = kw
                    body = json.dumps(doc).encode()
                    patched = True
            except ValueError:
                pass
        with self.lock:
            self.stats["patched" if patched else "other"] += 1
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP | {"host", "content-length"}
        }
        headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=3600)
        try:
            conn.request(method, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            self.send_error(502, f"upstream: {type(exc).__name__}")
            return
        self.send_response(resp.status)
        stream = "text/event-stream" in (resp.getheader("Content-Type") or "")
        for k, v in resp.getheaders():
            if k.lower() not in HOP | {"content-length", "server", "date"}:
                self.send_header(k, v)
        if stream or resp.getheader("Content-Length") is None:
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        conn.close()

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def log_message(self, *a: object) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=18001)
    ap.add_argument("--upstream", type=int, default=18000)
    a = ap.parse_args()
    Handler.upstream_port = a.upstream
    srv = ThreadingHTTPServer(("127.0.0.1", a.listen), Handler)
    print(f"thinking-off proxy 127.0.0.1:{a.listen} -> 127.0.0.1:{a.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
