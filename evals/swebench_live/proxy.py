#!/usr/bin/env python3
"""The LLM proxy between the task containers and the model server: records, budgets, forwards.

Every agent rollout of the SWE-bench-Live runner talks to the model ONLY through this proxy
(the task containers have no other network route).  Per instance it

* **records the raw trajectory**: every request body exactly as the agent sent it and every
  response body exactly as the agent received it, with timestamps, status and token usage, one
  JSON line per call in ``<trajs>/<instance>/llm.jsonl`` (the leaderboard requires every LLM
  input and output, starting with the first prompt);
* **accounts** calls, tokens and dollars, and **refuses** further calls once the instance's
  token budget or wall-clock deadline is spent (the refusal is recorded too);
* keeps one **order** for every call that belongs in a trajectory (forwarded, failed upstream,
  refused): the record is written and synced to disk BEFORE the agent gets the answer.  A
  streamed answer is relayed as it arrives except for its last chunk, which is sent only after
  the record is on disk;
* **forwards** to one fixed OpenAI-compatible upstream (SGLang through a tunnel, or the local
  Cerebras proxy), optionally forcing a few request fields for every call (thinking off, a fixed
  ``reasoning_effort``, the served model name).  Forced fields are part of the frozen run
  settings and are recorded next to the original request.

An instance is addressed by an opaque random token in the path (``/i/<token>/v1/...``): the
instance id never appears in anything the agent can see (it contains the fix's PR number).
Only ``POST .../chat/completions`` and ``GET .../models`` are served.  The upstream API key, when
one is needed, is read from a file by the caller, injected here and never written anywhere.
"""

from __future__ import annotations

import contextlib
import dataclasses
import http.client
import http.server
import json
import os
import re
import secrets
import ssl
import threading
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

PATH_RE = re.compile(r"^/i/([A-Za-z0-9]{16,64})/v1/(chat/completions|models)/?$")
MAX_BODY_BYTES = 32 * 1024 * 1024
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Answers that mean the model path did not serve the call (as opposed to a request the model
#: server rejected as wrong): no answer at all, credentials or credit refused, rate limit or
#: server errors that outlasted the retries.
UNSERVED_STATUS = frozenset({401, 402, 403, 429, 500, 502, 503, 504})
HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "upgrade", "te", "trailer", "host"}
)
LLM_LOG = "llm.jsonl"


def new_token() -> str:
    """An opaque path token for one instance (reveals nothing about it)."""
    return secrets.token_hex(16)


@dataclasses.dataclass
class Usage:
    calls: int = 0
    refused: int = 0
    errors: int = 0
    unserved: int = 0  # of the errors: the model path did not serve the call (an outage)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    calls_without_usage: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclasses.dataclass
class Instance:
    """One registered rollout: where its trajectory goes and what it may spend."""

    name: str
    traj_dir: Path
    max_total_tokens: int  # 0 = no token budget
    deadline: float  # 0 = no deadline (same clock as the proxy's)
    usage: Usage = dataclasses.field(default_factory=Usage)
    seq: int = 0
    closed: bool = False
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def usage_from_json(doc: Any) -> dict[str, int] | None:
    """Token counts of an OpenAI-style response document (or SSE chunk), None when absent."""
    u = doc.get("usage") if isinstance(doc, dict) else None
    if not isinstance(u, dict):
        return None

    def num(v: Any) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0

    pd = u.get("prompt_tokens_details")
    cd = u.get("completion_tokens_details")
    return {
        "prompt_tokens": num(u.get("prompt_tokens")),
        "completion_tokens": num(u.get("completion_tokens")),
        "cached_tokens": num(pd.get("cached_tokens")) if isinstance(pd, dict) else 0,
        "reasoning_tokens": num(cd.get("reasoning_tokens")) if isinstance(cd, dict) else 0,
    }


def usage_from_sse(raw: bytes) -> dict[str, int] | None:
    """The usage of a streamed response: the last ``data:`` chunk that carries one."""
    found: dict[str, int] | None = None
    for line in raw.splitlines():
        if not line.startswith(b"data:") or b'"usage"' not in line:
            continue
        with contextlib.suppress(ValueError):
            found = usage_from_json(json.loads(line[5:].strip())) or found
    return found


def _body_for_log(raw: bytes) -> Any:
    """A body as it goes into the trajectory: parsed JSON when it is JSON, else the exact text."""
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


class RecordingProxy:
    """See the module docstring.  ``start()`` returns the bound port; ``stop()`` shuts down."""

    def __init__(
        self,
        upstream: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        api_key: str = "",
        force_fields: dict[str, Any] | None = None,
        price_in: float = 0.0,
        price_out: float = 0.0,
        upstream_timeout: float = 1800.0,
        max_retries: int = 4,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        u = urllib.parse.urlsplit(upstream.rstrip("/"))
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"upstream must be an http(s) URL, got {upstream!r}")
        self._tls = u.scheme == "https"
        self._up_host = u.hostname
        self._up_port = u.port or (443 if self._tls else 80)
        #: Path prefix of the upstream's API root, WITHOUT a trailing ``/v1``.
        self._up_prefix = re.sub(r"/v1$", "", u.path.rstrip("/"))
        self._key = api_key
        self.force_fields = dict(force_fields or {})
        self.price_in, self.price_out = price_in, price_out
        self._timeout = upstream_timeout
        self._max_retries = max(0, max_retries)
        self._clock, self._sleep = clock, sleep
        self._instances: dict[str, Instance] = {}
        self._lock = threading.Lock()
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                proxy.handle(self, "GET")

            def do_POST(self) -> None:
                proxy.handle(self, "POST")

            def log_message(self, *args: object) -> None:  # bodies and paths are never logged
                pass

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def upstream_address(self) -> tuple[str, int]:
        """Host and port of the model server this proxy forwards to."""
        return self._up_host, self._up_port

    def start(self) -> int:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ── instances ────────────────────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        traj_dir: Path,
        *,
        max_total_tokens: int = 0,
        max_seconds: float = 0.0,
    ) -> str:
        """Open an instance's account; returns its path token.  The trajectory file must not
        exist yet: one rollout per instance, never appended to an earlier one."""
        traj_dir.mkdir(parents=True, exist_ok=True)
        log = traj_dir / LLM_LOG
        if log.exists():
            raise FileExistsError(f"{log} exists: a rollout of {name} was already recorded")
        log.touch()
        token = new_token()
        inst = Instance(
            name=name,
            traj_dir=traj_dir,
            max_total_tokens=max(0, max_total_tokens),
            deadline=(self._clock() + max_seconds) if max_seconds > 0 else 0.0,
        )
        with self._lock:
            self._instances[token] = inst
        return token

    def close(self, token: str) -> Usage:
        """End an instance's rollout: later calls with its token are refused (and recorded)."""
        inst = self._instances[token]
        with inst.lock:
            inst.closed = True
        return inst.usage

    def usage(self, token: str) -> Usage:
        return self._instances[token].usage

    def cost_usd(self, token: str) -> float:
        u = self._instances[token].usage
        return (u.prompt_tokens * self.price_in + u.completion_tokens * self.price_out) / 1e6

    # ── request handling ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _send(h: http.server.BaseHTTPRequestHandler, status: int, body: bytes, ctype: str) -> None:
        with contextlib.suppress(OSError):
            h.send_response(status)
            h.send_header("Content-Type", ctype)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)

    @staticmethod
    def _error_body(kind: str, msg: str) -> bytes:
        return json.dumps({"error": {"message": msg, "type": kind, "code": kind}}).encode()

    def _error(
        self, h: http.server.BaseHTTPRequestHandler, status: int, kind: str, msg: str
    ) -> bytes:
        """Answer with an error that is NOT part of a trajectory (unknown path, wrong method).
        Anything that belongs in a trajectory is recorded first: see :meth:`_refuse`."""
        body = self._error_body(kind, msg)
        self._send(h, status, body, "application/json")
        return body

    def _refuse(
        self,
        h: http.server.BaseHTTPRequestHandler,
        inst: Instance,
        status: int,
        kind: str,
        why: str,
        t_req: float,
        request_log: Any,
        **extra: Any,
    ) -> None:
        """Refuse a call of a known instance: count it, RECORD it, and only then answer, the
        same order as for a forwarded call (the record is on disk before the agent can act on
        the answer)."""
        body = self._error_body(kind, why)
        with inst.lock:
            inst.usage.refused += 1
        self._record(
            inst,
            {
                "t_request": t_req,
                "t_response": self._clock(),
                "request": request_log,
                "status": status,
                "response": _body_for_log(body),
                "refused_by_proxy": why,
                **extra,
            },
        )
        self._send(h, status, body, "application/json")

    def _record(self, inst: Instance, rec: dict[str, Any]) -> None:
        with inst.lock:
            inst.seq += 1
            rec = {"seq": inst.seq, **rec}
            with (inst.traj_dir / LLM_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                with contextlib.suppress(OSError):  # durable before anyone gets the answer
                    os.fsync(fh.fileno())

    def _refusal(self, inst: Instance) -> str:
        with inst.lock:
            if inst.closed:
                return "this rollout is finished"
            if inst.deadline and self._clock() >= inst.deadline:
                return "the instance's wall-clock budget is spent"
            if inst.max_total_tokens and inst.usage.total_tokens >= inst.max_total_tokens:
                return "the instance's token budget is spent"
        return ""

    def handle(self, h: http.server.BaseHTTPRequestHandler, method: str) -> None:
        m = PATH_RE.match(urllib.parse.urlsplit(h.path).path)
        inst = self._instances.get(m.group(1)) if m else None
        if m is None or inst is None:
            self._error(h, 404, "not_found", "unknown path")
            return
        endpoint = m.group(2)
        if (endpoint == "models") != (method == "GET"):
            self._error(h, 405, "method_not_allowed", "method not allowed")
            return
        try:
            length = int(h.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            if method == "GET":
                self._error(h, 413, "too_large", "request body too large")
            else:  # an LLM call the agent tried: in the trajectory (the body is never read)
                self._refuse(
                    h, inst, 413, "too_large", "request body too large", self._clock(), None,
                    declared_length=length,
                )  # fmt: skip
            return
        raw = h.rfile.read(length) if length else b""
        if method == "GET":
            status, _, data, _, _ = self._forward("GET", "/v1/models", b"", h, stream_ok=False)
            # /models is not an LLM call: relayed, not recorded, not budgeted.
            if status == 0:
                self._error(h, 502, "upstream_error", "the model server did not answer")
            else:
                self._send(h, status, data, "application/json")
            return
        t_req = self._clock()
        request_log = _body_for_log(raw)
        refusal = self._refusal(inst)
        if refusal:
            self._refuse(h, inst, 403, "budget_exhausted", refusal, t_req, request_log)
            return
        forced: dict[str, Any] = {}
        out = raw
        if isinstance(request_log, dict) and self.force_fields:
            doc = dict(request_log)
            for k, v in self.force_fields.items():
                if doc.get(k) != v:
                    forced[k] = v
                    doc[k] = v
            if forced:
                out = json.dumps(doc).encode()
        status, ctype, data, attempts, held = self._forward(
            "POST", "/v1/chat/completions", out, h, stream_ok=True
        )
        stream = "text/event-stream" in ctype
        unserved = status == 0 or status in UNSERVED_STATUS
        if status == 0:
            status, ctype = 502, "application/json"
            data = json.dumps(
                {"error": {"message": "the model server did not answer", "type": "upstream_error"}}
            ).encode()
        use = (usage_from_sse(data) if stream else None) or (
            usage_from_json(_body_for_log(data)) if not stream else None
        )
        with inst.lock:
            inst.usage.calls += 1
            if status >= 400:
                inst.usage.errors += 1
                inst.usage.unserved += unserved
            elif use is None:
                inst.usage.calls_without_usage += 1
            if use:
                inst.usage.prompt_tokens += use["prompt_tokens"]
                inst.usage.completion_tokens += use["completion_tokens"]
                inst.usage.cached_tokens += use["cached_tokens"]
                inst.usage.reasoning_tokens += use["reasoning_tokens"]
        rec: dict[str, Any] = {
            "t_request": t_req,
            "t_response": self._clock(),
            "request": request_log,
            "status": status,
            "response": data.decode("utf-8", "replace") if stream else _body_for_log(data),
            "usage": use,
        }
        if forced:
            rec["forced_by_proxy"] = forced  # run-wide settings applied on top of the request
        if attempts > 1:
            rec["upstream_attempts"] = attempts
        # Recorded BEFORE the agent has the answer.  A stream was relayed as it arrived except
        # for its last chunk, which is held back until the record is on disk.
        self._record(inst, rec)
        if stream:
            self._finish_stream(h, held)
        else:
            self._send(h, status, data, ctype)

    # ── upstream ─────────────────────────────────────────────────────────────────────────────

    def probe(self, model: str, timeout: float = 60.0) -> bool:
        """Is the model path serving?  One minimal call (one output token) sent straight to
        the upstream: it belongs to no instance, is recorded in no trajectory and counts
        against no budget.  A model list is not enough: it also answers when the account has
        no credit or the model itself is down."""
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        ).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        conn = self._connect()
        conn.timeout = timeout
        try:
            conn.request(
                "POST", self._up_prefix + "/v1/chat/completions", body=body, headers=headers
            )
            resp = conn.getresponse()
            resp.read()
            return resp.status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            conn.close()

    def _connect(self) -> http.client.HTTPConnection:
        if self._tls:
            return http.client.HTTPSConnection(
                self._up_host,
                self._up_port,
                timeout=self._timeout,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(self._up_host, self._up_port, timeout=self._timeout)

    def _forward(
        self,
        method: str,
        path: str,
        body: bytes,
        h: http.server.BaseHTTPRequestHandler,
        *,
        stream_ok: bool,
    ) -> tuple[int, str, bytes, int, bytes]:
        """Send to the upstream (retrying 429 / 5xx / connection errors).  A streamed answer is
        relayed to the client as it arrives, except its last chunk; any other answer is
        returned for the caller to record first and send afterwards.  Returns
        ``(status, content_type, body, attempts, held)``: status 0 = no answer (nothing was
        sent to the client); ``held`` = the last chunk of a stream, not sent yet."""
        headers = {"Content-Type": "application/json", "Accept": h.headers.get("Accept") or "*/*"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        if body:
            headers["Content-Length"] = str(len(body))
        attempts = 0
        while True:
            attempts += 1
            conn = self._connect()
            try:
                conn.request(method, self._up_prefix + path, body=body or None, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                ctype = resp.getheader("Content-Type") or "application/json"
                if status in RETRY_STATUS and attempts <= self._max_retries:
                    resp.read()
                    retry_after = resp.getheader("Retry-After") or ""
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0
                    self._sleep(min(60.0, max(delay, 2.0**attempts)))
                    continue
                if stream_ok and "text/event-stream" in ctype:
                    got, held = self._relay_stream(h, resp, status, ctype)
                    return status, ctype, got, attempts, held
                return status, ctype, resp.read(), attempts, b""
            except (OSError, http.client.HTTPException):
                if attempts <= self._max_retries:
                    self._sleep(min(60.0, 2.0**attempts))
                    continue
                return 0, "", b"", attempts, b""
            finally:
                conn.close()

    @staticmethod
    def _relay_stream(
        h: http.server.BaseHTTPRequestHandler,
        resp: http.client.HTTPResponse,
        status: int,
        ctype: str,
    ) -> tuple[bytes, bytes]:
        """Relay a streamed answer; returns ``(everything received, the last chunk)``.  Each
        chunk is sent when the NEXT one arrives, so the last chunk (the end of the answer) is
        still unsent on return: the caller records the call and then sends it."""
        got = bytearray()
        held = b""
        client_ok = True
        try:
            h.send_response(status)
            h.send_header("Content-Type", ctype)
            h.send_header("Connection", "close")
            h.end_headers()
        except OSError:
            client_ok = False
        h.close_connection = True
        while True:
            try:
                chunk = resp.read1(65536)
            except (OSError, http.client.HTTPException):
                break
            if not chunk:
                break
            got += chunk
            if client_ok and held:
                try:
                    h.wfile.write(held)
                    h.wfile.flush()
                except OSError:
                    client_ok = False  # keep reading: the record must hold what the model sent
            held = chunk
        return bytes(got), (held if client_ok else b"")

    @staticmethod
    def _finish_stream(h: http.server.BaseHTTPRequestHandler, held: bytes) -> None:
        """Send the held-back last chunk of a stream (called after the call is recorded)."""
        if held:
            with contextlib.suppress(OSError):
                h.wfile.write(held)
                h.wfile.flush()
