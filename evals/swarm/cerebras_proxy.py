#!/usr/bin/env python3
"""A local rate-limiting proxy between the swarm episodes and the Cerebras API.

Why a proxy: ``run_smoke.py`` runs every episode as its own process, so no in-process limiter
can keep the whole run under Cerebras' per-organization limits (live x-ratelimit headers
for ``qwen-3.8-27b``: 450 RPM, 150K tokens/min; docs: 750K *total* TPM).  One local
choke point can:

* hold the API key (read from a file, injected here, never sent to or logged for an episode);
* queue requests in a three-bucket limiter (requests/min, uncached tokens/min, total
  tokens/min).  Cerebras estimates a request as *input + max_completion_tokens* BEFORE it
  runs and settles afterwards; this limiter reserves the same estimate and settles with the
  real usage, so a big ``max_tokens`` costs quota exactly as it does server-side;
* turn 429 / 5xx / connection errors into a pause and a transparent retry with backoff, so an
  episode never sees a rate limit and a rate limit is never mistaken for a model failure;
* account tokens and dollars per episode (``/e/<episode>/v1/...`` path prefix) and refuse
  further spend past a hard dollar cap.

Only ``POST /v1/chat/completions`` and ``GET /v1/models`` are forwarded, to one fixed upstream.
Request bodies, headers and the key are never logged.  Facts checked against
https://inference-docs.cerebras.ai (rate limits, qwen-3.8-27b, prompt caching, reasoning).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import http.client
import http.server
import json
import random
import re
import ssl
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

UPSTREAM = "https://api.cerebras.ai"
MODEL = "qwen-3.8-27b"
#: Live headers (450 RPM, 150K tokens/min); the docs' 750K total-TPM bucket stays as a ceiling.
RPM, UNCACHED_TPM, TOTAL_TPM = 450, 150_000, 750_000
#: Paid-tier maximum output (the Free Trial's is 32,768) and context window.
MAX_OUTPUT_TOKENS = 40_960
CONTEXT_TOKENS = 131_072
#: $ per million tokens (cached input is billed at the same rate).
PRICE_IN, PRICE_OUT = 0.99, 1.49
#: Stay this far under the published limits: the estimate is ours, theirs may differ.
SAFETY = 0.8
CHARS_PER_TOKEN = 3.2  # conservative for code (real is about 3.5 to 4)
DEFAULT_KEY_FILE = Path.home() / ".config" / "cerebras" / "key"
EPISODE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,79}$")
MAX_BODY_BYTES = 12 * 1024 * 1024  # Cerebras' own limit is 10 MiB
MAX_RETRIES = 10
BACKOFF_BASE_S, BACKOFF_CAP_S = 2.0, 30.0
#: Set on a reply the proxy gave up retrying: a provider problem, not a model failure.
USER_AGENT = "memfun-swarm/1.0"
GAVE_UP_HEADER = "x-memfun-provider-gave-up"
TRANSIENT = frozenset({500, 502, 503, 504})


# ── rate limiter ─────────────────────────────────────────────────────────────


class TokenBucket:
    """A continuously refilling bucket (Cerebras uses the same token-bucket model)."""

    def __init__(self, per_minute: float, clock: Callable[[], float]) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.level = float(per_minute)
        self._clock = clock
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.level = min(self.capacity, self.level + (now - self._t) * self.rate)
        self._t = now

    def wait_for(self, amount: float) -> float:
        """Seconds until *amount* is available (0 when it is now)."""
        self._refill()
        need = min(amount, self.capacity) - self.level
        return 0.0 if need <= 0 else need / self.rate

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= min(amount, self.capacity)

    def give_back(self, amount: float) -> None:
        self._refill()
        self.level = min(self.capacity, self.level + amount)

    def drain(self) -> None:
        self._refill()
        self.level = 0.0


@dataclasses.dataclass(frozen=True)
class Limits:
    rpm: int = RPM
    uncached_tpm: int = UNCACHED_TPM
    total_tpm: int = TOTAL_TPM
    safety: float = SAFETY

    def scaled(self) -> tuple[float, float, float]:
        return (
            self.rpm * self.safety,
            self.uncached_tpm * self.safety,
            self.total_tpm * self.safety,
        )


@dataclasses.dataclass
class Reservation:
    uncached: float
    total: float


class Limiter:
    """Three buckets (requests, uncached tokens, total tokens) reserved atomically."""

    def __init__(
        self,
        limits: Limits | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limits = limits or Limits()
        rpm, unc, tot = self.limits.scaled()
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self.requests = TokenBucket(rpm, clock)
        self.uncached = TokenBucket(unc, clock)
        self.total = TokenBucket(tot, clock)
        self.waited_s = 0.0

    def acquire(self, est_uncached: float, est_total: float) -> Reservation:
        """Block until one request of this estimated size fits all three buckets."""
        started = self._clock()
        while True:
            with self._lock:
                wait = max(
                    self.requests.wait_for(1),
                    self.uncached.wait_for(est_uncached),
                    self.total.wait_for(est_total),
                )
                if wait <= 0:
                    self.requests.take(1)
                    self.uncached.take(est_uncached)
                    self.total.take(est_total)
                    self.waited_s += self._clock() - started
                    return Reservation(est_uncached, est_total)
            self._sleep(min(max(wait, 0.01), 5.0))

    def settle(self, res: Reservation, actual_uncached: float, actual_total: float) -> None:
        """Replace the reservation by the real usage (as Cerebras settles its estimate)."""
        with self._lock:
            self.uncached.give_back(res.uncached - actual_uncached)
            self.total.give_back(res.total - actual_total)

    def penalize(self, bucket: str | None) -> None:
        """A 429 says our estimate was too optimistic: empty the named bucket (all if unknown)."""
        with self._lock:
            if bucket in (None, "uncached"):
                self.uncached.drain()
            if bucket in (None, "total"):
                self.total.drain()
            if bucket in (None, "requests"):
                self.requests.drain()

    def state(self) -> dict[str, float]:
        with self._lock:
            for b in (self.requests, self.uncached, self.total):
                b._refill()
            return {
                "requests_available": round(self.requests.level, 1),
                "uncached_available": round(self.uncached.level),
                "total_available": round(self.total.level),
                "queue_wait_s": round(self.waited_s, 1),
            }


def estimate_tokens(
    body: dict[str, Any], raw_len: int, cache_hit_ratio: float
) -> tuple[float, float]:
    """``(uncached, total)`` token estimate: input + the requested output cap."""
    est_in = raw_len / CHARS_PER_TOKEN
    cap = body.get("max_completion_tokens") or body.get("max_tokens") or MAX_OUTPUT_TOKENS
    est_out = (
        float(min(int(cap), MAX_OUTPUT_TOKENS))
        if isinstance(cap, int)
        else float(MAX_OUTPUT_TOKENS)
    )
    est_total = est_in + est_out
    est_uncached = est_in * (1.0 - min(max(cache_hit_ratio, 0.0), 0.95)) + est_out
    return est_uncached, est_total


# ── accounting ───────────────────────────────────────────────────────────────


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT) / 1_000_000


@dataclasses.dataclass
class EpisodeUsage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    retries_429: int = 0
    retries_other: int = 0
    gave_up: int = 0
    spend_cap_refusals: int = 0
    unmetered_streams: int = 0  # streamed responses without a usage block (estimated)
    cache_ratio: float = 0.0  # moving average, feeds the uncached estimate

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["cost_usd"] = round(cost_usd(self.prompt_tokens, self.completion_tokens), 6)
        d["cache_ratio"] = round(self.cache_ratio, 3)
        return d


class Accounts:
    def __init__(self, max_cost_usd: float | None) -> None:
        self.max_cost_usd = max_cost_usd
        self._lock = threading.Lock()
        self._eps: dict[str, EpisodeUsage] = {}
        self._reserved = 0.0  # worst-case $ of requests in flight

    def reserve(self, usd: float) -> None:
        with self._lock:
            self._reserved += usd

    def release(self, usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - usd)

    def get(self, ep: str) -> EpisodeUsage:
        with self._lock:
            return self._eps.setdefault(ep, EpisodeUsage())

    def add(self, ep: str, **kw: int) -> None:
        with self._lock:
            u = self._eps.setdefault(ep, EpisodeUsage())
            for k, v in kw.items():
                setattr(u, k, getattr(u, k) + v)

    def observe_cache(self, ep: str, prompt: int, cached: int) -> None:
        if prompt <= 0:
            return
        with self._lock:
            u = self._eps.setdefault(ep, EpisodeUsage())
            u.cache_ratio = 0.5 * u.cache_ratio + 0.5 * (cached / prompt)

    def total_cost(self) -> float:
        with self._lock:
            return sum(cost_usd(u.prompt_tokens, u.completion_tokens) for u in self._eps.values())

    def over_cap(self) -> bool:
        """Spent plus the worst case of what is in flight has reached the cap."""
        if self.max_cost_usd is None:
            return False
        with self._lock:
            spent = sum(cost_usd(u.prompt_tokens, u.completion_tokens) for u in self._eps.values())
            return spent + self._reserved >= self.max_cost_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            eps = {k: v.as_dict() for k, v in sorted(self._eps.items())}
        tot = {
            k: sum(e[k] for e in eps.values())
            for k in ("requests", "prompt_tokens", "completion_tokens", "cached_tokens",
                      "reasoning_tokens", "retries_429", "retries_other", "gave_up",
                      "spend_cap_refusals")
        }  # fmt: skip
        tot["cost_usd"] = round(cost_usd(tot["prompt_tokens"], tot["completion_tokens"]), 6)
        return {"episodes": eps, "total": tot, "max_cost_usd": self.max_cost_usd}


def usage_of(doc: Any) -> tuple[int, int, int, int] | None:
    """``(prompt, completion, cached, reasoning)`` from a response/usage object, if present."""
    if not isinstance(doc, dict):
        return None
    usage = doc.get("usage")
    if not isinstance(usage, dict):
        return None
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(pt, int) or not isinstance(ct, int):
        return None
    pd = usage.get("prompt_tokens_details") or {}
    cd = usage.get("completion_tokens_details") or {}
    cached = pd.get("cached_tokens") if isinstance(pd, dict) else 0
    reasoning = cd.get("reasoning_tokens") if isinstance(cd, dict) else 0
    return (
        pt,
        ct,
        cached if isinstance(cached, int) else 0,
        reasoning if isinstance(reasoning, int) else 0,
    )


# ── the proxy ────────────────────────────────────────────────────────────────


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def _retry_after(headers: Any) -> float | None:
    raw = headers.get("Retry-After") if headers is not None else None
    try:
        return max(0.0, min(float(raw), 120.0)) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _bucket_in(message: str) -> str | None:
    low = message.lower()
    if "uncached" in low:
        return "uncached"
    if "total" in low and "token" in low:
        return "total"
    if "request" in low:
        return "requests"
    return None


class CerebrasProxy:
    """The proxy as an object: ``start()`` returns the base URL episodes should use."""

    def __init__(
        self,
        key: str,
        *,
        upstream: str = UPSTREAM,
        limits: Limits | None = None,
        max_cost_usd: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE_S,
        read_timeout: float = 900.0,
        port: int = 0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.reasoning_effort = reasoning_effort
        if not key.strip():
            raise ValueError("empty API key")
        self._key = key.strip()
        self.upstream = urllib.parse.urlsplit(upstream)
        self.limiter = Limiter(limits, clock, sleep)
        self.accounts = Accounts(max_cost_usd)
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.read_timeout = read_timeout
        self._pause_until = 0.0
        self._pause_lock = threading.Lock()
        self._clock = clock
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            server_version = "memfun-cerebras-proxy"
            protocol_version = "HTTP/1.0"  # the body ends when the connection closes (streams)

            def log_message(self, *_a: object) -> None:  # never log requests
                pass

            def do_GET(self) -> None:
                proxy.handle(self, "GET")

            def do_POST(self) -> None:
                proxy.handle(self, "POST")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread: threading.Thread | None = None

    # -- lifecycle --
    def start(self) -> str:
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        if self.thread is not None:  # shutdown() blocks forever if serve_forever never ran
            self.server.shutdown()
        self.server.server_close()

    # -- request handling --
    @staticmethod
    def _reply(
        h: http.server.BaseHTTPRequestHandler, status: int, doc: Any, gave_up: bool = False
    ) -> None:
        data = json.dumps(doc).encode()
        h.send_response(status)
        if gave_up:
            h.send_header(GAVE_UP_HEADER, "1")
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def handle(self, h: http.server.BaseHTTPRequestHandler, method: str) -> None:
        # Loopback only, and only for a request addressed to us as 127.0.0.1/localhost with a
        # JSON body: a web page (DNS rebinding, text/plain form POST) cannot spend the quota.
        port = self.server.server_address[1]
        if (h.headers.get("host") or "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._reply(h, 421, {"error": {"message": "bad Host header"}})
            return
        if method == "POST" and not (h.headers.get("content-type") or "").startswith(
            "application/json"
        ):
            self._reply(h, 415, {"error": {"message": "content-type must be application/json"}})
            return
        path = urllib.parse.urlsplit(h.path).path
        if method == "GET" and path.startswith("/_stats"):
            ep = path[len("/_stats") :].strip("/")
            snap = self.accounts.snapshot()
            doc: Any = (
                snap["episodes"].get(ep, {}) if ep else {**snap, "limiter": self.limiter.state()}
            )
            self._reply(h, 200, doc)
            return
        ep = "_other"
        m = re.match(r"^/e/([^/]+)(/v1/.*)$", path)
        if m:
            ep, path = m.group(1), m.group(2)
            if not EPISODE_RE.match(ep):
                self._reply(h, 400, {"error": {"message": "bad episode id"}})
                return
        if (method, path) not in (("POST", "/v1/chat/completions"), ("GET", "/v1/models")):
            self._reply(h, 404 if method == "POST" or path != "/v1/chat/completions" else 405,
                        {"error": {"message": f"{method} {path} is not proxied"}})  # fmt: skip
            return
        body = b""
        if method == "POST":
            try:
                length = int(h.headers.get("content-length", 0))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY_BYTES:
                self._reply(h, 413, {"error": {"message": "request body too large"}})
                return
            body = h.rfile.read(length)
        self.forward(h, method, path, body, ep)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        base = (
            retry_after
            if retry_after is not None
            else min(BACKOFF_CAP_S, self.backoff_base * 2**attempt)
        )
        return base * (1.0 + random.random() * 0.25)

    def _pause(self, seconds: float) -> None:
        """A rate limit pauses EVERY request, not just the one that hit it."""
        with self._pause_lock:
            self._pause_until = max(self._pause_until, self._clock() + seconds)

    def _wait_pause(self) -> None:
        while True:
            with self._pause_lock:
                left = self._pause_until - self._clock()
            if left <= 0:
                return
            self.sleep(min(left, 5.0))

    def _connect(self) -> http.client.HTTPConnection:
        host, port = self.upstream.hostname or "", self.upstream.port
        if self.upstream.scheme == "https":
            return http.client.HTTPSConnection(
                host, port, timeout=self.read_timeout, context=ssl.create_default_context()
            )
        return http.client.HTTPConnection(host, port, timeout=self.read_timeout)

    def forward(
        self, h: http.server.BaseHTTPRequestHandler, method: str, path: str, body: bytes, ep: str
    ) -> None:
        doc: dict[str, Any] = {}
        streaming = False
        if method == "POST":
            try:
                parsed = json.loads(body)
                doc = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                self._reply(h, 400, {"error": {"message": "request body is not JSON"}})
                return
            streaming = bool(doc.get("stream"))
            if self.reasoning_effort and "reasoning_effort" not in doc:
                doc["reasoning_effort"] = self.reasoning_effort  # default; never overrides
                body = json.dumps(doc).encode()
        if self.accounts.over_cap():
            self.accounts.add(ep, spend_cap_refusals=1)
            msg = f"local spend cap reached (${self.accounts.max_cost_usd:.2f}); request refused"
            self._reply(h, 402, {"error": {"message": msg}})
            return
        est_unc, est_tot = estimate_tokens(doc, len(body), self.accounts.get(ep).cache_ratio)
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
            # Cloudflare in front of the API answers error 1010 to a client with no
            # (or Python's default) User-Agent.
            "User-Agent": USER_AGENT,
        }
        if method == "POST":
            headers["Content-Type"] = "application/json"
        last: tuple[int, bytes] | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_pause()
            flight = (
                Flight(self, ep, self.limiter.acquire(est_unc, est_tot), est_tot, len(body))
                if method == "POST"
                else None
            )
            try:
                outcome = self._attempt(h, flight, method, path, body, headers, ep, streaming)
            finally:
                if flight is not None:
                    flight.close()  # settles exactly once, whatever happened
            if outcome is None:
                return  # relayed (or the episode hung up): done
            status, data, retry_after = outcome
            last = (status, data)
            if status == 429:
                msg = ""
                with contextlib.suppress(ValueError, AttributeError):
                    msg = str((json.loads(data).get("error") or {}).get("message", ""))
                self.limiter.penalize(_bucket_in(msg))
                self.accounts.add(ep, retries_429=1)
            else:
                self.accounts.add(ep, retries_other=1)
            if attempt >= self.max_retries:
                break
            if status == 429 or status in TRANSIENT:
                self._pause(self._backoff(attempt, retry_after))
            else:  # connection-level failure
                self.sleep(self._backoff(attempt, None))
        # retries exhausted: hand the provider's own error back (the runner never counts it as
        # a model failure: see ``provider_stopped`` in run_smoke.classify)
        self.accounts.add(ep, gave_up=1)
        status, data = last if last is not None else (502, b"{}")
        text = _redact(data.decode("utf-8", "replace"), self._key).encode()
        with contextlib.suppress(OSError):
            h.send_response(status)
            h.send_header(GAVE_UP_HEADER, "1")
            h.send_header("content-type", "application/json")
            h.send_header("content-length", str(len(text)))
            h.end_headers()
            h.wfile.write(text)

    def _attempt(
        self,
        h: http.server.BaseHTTPRequestHandler,
        flight: Flight | None,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
        ep: str,
        streaming: bool,
    ) -> tuple[int, bytes, float | None] | None:
        """One upstream try.  ``None`` = answered the episode; else ``(status, body, retry)``
        of a retryable failure (connection errors are reported as status 0)."""
        conn = self._connect()
        try:
            try:
                conn.request(
                    method, path, body=body if method == "POST" else None, headers=headers
                )
                if flight is not None:
                    flight.started = True
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException):
                return (0, b"{}", None)
            if resp.status == 429 or resp.status in TRANSIENT:
                try:
                    data = resp.read()
                except (OSError, http.client.HTTPException):
                    data = b"{}"
                if flight is not None:
                    flight.started = False  # a rejected request never ran
                return (resp.status, data, _retry_after(resp.msg))
            self._relay(h, resp, ep, flight, streaming)
            return None
        finally:
            conn.close()

    def _relay(
        self,
        h: http.server.BaseHTTPRequestHandler,
        resp: http.client.HTTPResponse,
        ep: str,
        flight: Flight | None,
        streaming: bool,
    ) -> None:
        ctype = resp.getheader("content-type") or "application/json"
        key = self._key.encode()
        try:
            if not streaming or "event-stream" not in ctype:
                data = resp.read()
                if flight is not None and resp.status < 400:
                    flight.usage = _usage_in(data)  # BEFORE writing: a hang-up cannot lose it
                    flight.settled_by_upstream = True
                elif flight is not None:
                    flight.started = False  # a 4xx/5xx body never ran the model
                if flight is not None:
                    flight.close()  # charge before the episode can see the answer
                text = _redact(data.decode("utf-8", "replace"), self._key).encode()
                h.send_response(resp.status)
                h.send_header("content-type", ctype)
                h.send_header("content-length", str(len(text)))
                h.end_headers()
                h.wfile.write(text)
                return
            h.send_response(resp.status)
            h.send_header("content-type", ctype)
            h.send_header("cache-control", "no-cache")
            h.end_headers()
            buf = b""
            while True:
                chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                if not chunk:
                    break
                if flight is not None:
                    # parse BEFORE writing: usage of a stream the episode abandons still counts
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        flight.feed_sse_line(line)
                    if resp.status >= 400:
                        flight.started = False
                h.wfile.write(chunk.replace(key, b"***") if key in chunk else chunk)
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, http.client.HTTPException):
            pass  # the episode hung up (loop guard, timeout): what we saw is still accounted


def _usage_in(data: bytes) -> tuple[int, int, int, int] | None:
    try:
        return usage_of(json.loads(data))
    except ValueError:
        return None


class Flight:
    """One upstream attempt's claim on the limiter and the spend cap, settled exactly once."""

    def __init__(
        self,
        proxy: CerebrasProxy,
        ep: str,
        res: Reservation,
        est_tot: float,
        body_len: int,
    ) -> None:
        self.proxy, self.ep, self.res = proxy, ep, res
        self.est_in = int(body_len / CHARS_PER_TOKEN)
        self.est_out_cap = max(0, int(est_tot - body_len / CHARS_PER_TOKEN))
        self.cost_reserved = cost_usd(self.est_in, self.est_out_cap)
        proxy.accounts.reserve(self.cost_reserved)
        self.started = False  # the request reached the server (it may have been billed)
        self.settled_by_upstream = False
        self.usage: tuple[int, int, int, int] | None = None
        self.out_chars = 0
        self.streamed = False
        self._closed = False

    def feed_sse_line(self, line: bytes) -> None:
        self.streamed = True
        if not line.startswith(b"data:") or b"[DONE]" in line:
            return
        try:
            evt = json.loads(line[5:])
            if not isinstance(evt, dict):
                return
            found = usage_of(evt)
            if found:
                self.usage = found
            for choice in evt.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    continue
                for k in ("content", "reasoning", "reasoning_content"):
                    if isinstance(delta.get(k), str):
                        self.out_chars += len(delta[k])
        except (ValueError, AttributeError, TypeError):
            return  # a malformed event must never break the relay or the accounting

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acc, lim, res = self.proxy.accounts, self.proxy.limiter, self.res
        acc.release(self.cost_reserved)
        if not self.started:  # it never ran: the quota comes back
            lim.settle(res, 0, 0)
            return
        if self.usage is None:  # ran, but no usage block seen: charge the estimate
            est_out = int(self.out_chars / CHARS_PER_TOKEN)
            acc.add(
                self.ep,
                requests=1,
                prompt_tokens=self.est_in,
                completion_tokens=est_out,
                unmetered_streams=1,
            )
            return  # the reservation stays: the safe side for the limiter
        pt, ct, cached, reasoning = self.usage
        acc.add(
            self.ep,
            requests=1,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        )
        acc.observe_cache(self.ep, pt, cached)
        lim.settle(res, max(0, pt - cached) + ct, pt + ct)


# ── command line ─────────────────────────────────────────────────────────────


def read_key(path: Path) -> str:
    """The API key from *path*; the content is never printed (only problems with the file)."""
    try:
        st = path.stat()
    except OSError as exc:
        raise SystemExit(f"no Cerebras key file at {path}: {exc.strerror}") from None
    if st.st_mode & 0o077:
        print(f"WARNING: {path} is readable by others (chmod 600 it)", file=sys.stderr)
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"{path} is empty")
    return key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--max-cost-usd", type=float, default=None)
    ap.add_argument("--safety", type=float, default=SAFETY)
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="inject this reasoning_effort into requests that do not set one",
    )
    args = ap.parse_args(argv)
    proxy = CerebrasProxy(
        read_key(args.key_file),
        limits=Limits(safety=args.safety),
        max_cost_usd=args.max_cost_usd,
        port=args.port,
        reasoning_effort=args.reasoning_effort,
    )
    base = proxy.start()
    print(
        f"cerebras proxy on {base} (episodes use {base}/e/<episode>/v1); Ctrl-C to stop",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        proxy.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
