"""RunPod launch kit for memfun swarm runs (stdlib only).

Subcommands::

    gpus     list GPU types and prices                       (read-only)
    launch   plan, and only with explicit intent, rent a pod  (dry-run by default)
    watch    local watchdog: live $ spent, terminate at deadline / Ctrl-C / error
    tunnel   SSH-forward a local port to the pod's SGLang port (read-only API)
    status   show one pod                                     (read-only)
    terminate  terminate ONE pod and verify it is gone (needs --yes)
    logs     stream a pod's container/system log (REST SSE)  (read-only)

Safety model (see ``docs/runpod-launch.md``):

* Every read path goes through :class:`ReadOnlyTransport`, which refuses any GraphQL
  document containing ``mutation`` and any non-GET REST call.  ``launch`` is a dry run
  unless BOTH ``--no-dry-run`` and ``--yes-spend-money`` are given; the mutating transport
  is constructed only after both checks pass.
* The account API key is wrapped in :class:`Secret`, is only ever placed in an HTTP
  ``Authorization`` header, and is redacted from every line this module prints and every
  error it raises.  It is never put into the pod environment.
* Three independent hard stops: the pod-create ``terminateAfter`` field, an in-pod
  deadline (``pod_boot.sh``, uses the pod-scoped ``runpodctl``), and the local watchdog.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

# ---------------------------------------------------------------------------
# Constants.  Every default is a flag; nothing below is used without being shown.
# ---------------------------------------------------------------------------

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
#: Pod log stream (Server-Sent Events, GET + Bearer auth).  Params: source=container|system,
#: tail<=5000, since; header Last-Event-ID.  Returns 404 once the pod is gone (logs vanish).
LOGS_URL = "https://api.runpod.io/v2/pods/{pod_id}/logs"
DEFAULT_LOG_DIR = Path("~/.cache/memfun/runpod-logs")
POD_ID_RE = re.compile(r"[A-Za-z0-9]{1,64}")
MAX_LOG_TAIL = 5000
#: Docker image-pull progress lines from the system log; hidden unless --raw.
PULL_NOISE = re.compile(
    r"^[0-9a-f]{12} (Extracting|Downloading|Pulling fs layer|Waiting|Verifying Checksum"
    r"|Download complete|Pull complete|Already exists)\s*$"
)
KEY_PATH = Path("~/.config/runpod/key")
KEY_ENV = "RUNPOD_API_KEY"
#: Name of the dummy env var memfun's [llm] api_key_env points at (SGLang ignores it).
LLM_KEY_ENV = "MEMFUN_SGLANG_DUMMY_KEY"

#: RunPod ``gpuTypeId`` for one B300.  Verified with the read-only ``gpuTypes`` query
#: (``runpod_kit.py gpus --filter B300``) on 2026-11-06.  For H200 pass
#: ``--gpu-type "NVIDIA H200"``; ``gpus`` lists every id.
DEFAULT_GPU_TYPE = "NVIDIA B300 SXM6 AC"

#: Official SGLang image.  VERIFY BEFORE LAUNCH: checked on Docker Hub
#: (hub.docker.com/r/lmsysorg/sglang/tags) on 2026-11-06, pushed 2026-10-30.  The SGLang
#: install docs recommend the ``cu130`` images for B300/GB300 (SM103) and CUDA 13.
DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.23-cu130-runtime"

#: Verified to exist (HF API, not gated, 81 files, ~30.9 GB) on 2026-11-06.
DEFAULT_MODEL_REPO = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_SERVED_NAME = "qwen3.8-27b-fp8"

#: SGLang's default ``--port`` (docs.sglang.io server arguments; install docs use 30000).
SGLANG_DEFAULT_PORT = 30000
DEFAULT_LOCAL_PORT = 18000
#: Local ports the tunnel must never bind (8434 is the owner's live LLM server).
FORBIDDEN_LOCAL_PORTS = frozenset({8434})

VOLUME_MOUNT = "/workspace"
#: Weights are ~30.9 GB; 80 GB leaves room for run trajectories and a second revision.
DEFAULT_VOLUME_GB = 80
DEFAULT_MAX_MINUTES = 120
MAX_MINUTES_CAP = 720
DEFAULT_MAX_COST_USD = 40.0

BOOT_SCRIPT = Path(__file__).with_name("pod_boot.sh")
SECRET_ENV_HINT = re.compile(r"(api[_-]?key|token|secret|password|passwd)", re.IGNORECASE)
#: RunPod keys are [A-Za-z0-9_-]; anything else (newline, space, NUL...) is refused so it
#: can never reach an HTTP header where http.client would echo it in an exception.
KEY_CHARSET = re.compile(r"[A-Za-z0-9_\-]{8,512}")
#: Hugging Face style ``org/name`` (or ``name``); no ``..``/absolute paths (see _check_name).
REPO_NAME_RE = re.compile(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?")
REVISION_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
MAX_GPU_COUNT = 16
MAX_COST_CAP_USD = 10_000.0


class KitError(Exception):
    """User-facing failure.  Messages are redacted before they are raised."""


class MutationRefusedError(KitError):
    """A read-only transport was asked to mutate."""


class LogStreamError(KitError):
    """The pod log stream failed.  ``status`` is the HTTP code (None for I/O errors);
    ``idle`` means the read timed out with the connection still open (no new lines)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        idle: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.idle = idle


def _d(value: Any) -> dict[str, Any]:
    """``value`` as a JSON object, or ``{}``."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _l(value: Any) -> list[Any]:
    """``value`` as a JSON array, or ``[]``."""
    return cast("list[Any]", value) if isinstance(value, list) else []


# ---------------------------------------------------------------------------
# Secrets and redaction.
# ---------------------------------------------------------------------------


class Secret:
    """A string that refuses to print itself."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    def __str__(self) -> str:
        return "***"

    def __bool__(self) -> bool:
        return bool(self._value)


class Redactor:
    """Replaces every registered secret (and its obvious encodings) with ``***``."""

    def __init__(self) -> None:
        self._needles: list[str] = []

    def add(self, secret: Secret) -> None:
        raw = secret.reveal()
        if len(raw) < _FRAG_PREFIX:
            return
        for needle in (raw, base64.b64encode(raw.encode()).decode(), json.dumps(raw)[1:-1]):
            if needle and needle not in self._needles:
                self._needles.append(needle)
        self._needles.sort(key=len, reverse=True)

    def __call__(self, text: str) -> str:
        text = str(text)
        for needle in self._needles:
            text = text.replace(needle, "***")
        for needle in self._needles:
            text = _redact_fragments(text, needle)
        return text


#: A key prefix this long (or any inner run of _FRAG_INNER chars) is treated as the key.
_FRAG_PREFIX = 6
_FRAG_INNER = 8


def _redact_fragments(text: str, secret: str) -> str:
    """Hide truncated pieces of ``secret``: any prefix >= 6 chars, any inner run >= 8."""
    if len(secret) < _FRAG_PREFIX:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        best = 0
        # Longest run starting at text[i] that equals some slice of the secret.
        for start in range(len(secret)):
            if secret[start] != text[i]:
                continue
            n = 0
            while (
                i + n < len(text) and start + n < len(secret) and text[i + n] == secret[start + n]
            ):
                n += 1
            need = _FRAG_PREFIX if start == 0 else _FRAG_INNER
            if n >= need and n > best:
                best = n
        if best:
            out.append("***")
            i += best
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


REDACT = Redactor()


def emit(message: str = "", *, err: bool = False) -> None:
    """The only way this module writes to stdout/stderr."""
    stream = sys.stderr if err else sys.stdout
    print(REDACT(message), file=stream, flush=True)


def load_api_key(env: dict[str, str] | None = None, path: Path | None = None) -> Secret:
    """Read the key from ``$RUNPOD_API_KEY`` or ``~/.config/runpod/key``.  Never echoes it."""
    env = dict(os.environ) if env is None else env
    raw = env.get(KEY_ENV, "")
    source = f"${KEY_ENV}"
    if not raw.strip():
        key_file = (path or KEY_PATH).expanduser()
        if not key_file.is_file():
            raise KitError(f"no RunPod API key: set ${KEY_ENV} or create {key_file} (chmod 600)")
        mode = stat.S_IMODE(key_file.stat().st_mode)
        if mode & 0o077:
            emit(f"warning: {key_file} is mode {oct(mode)}; run chmod 600 on it", err=True)
        try:
            raw = key_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise KitError(f"cannot read {key_file} as UTF-8 text") from None
        source = str(key_file)
    # Register the raw value and every piece of it before validating, so even a rejected
    # key can never be printed.
    REDACT.add(Secret(raw.strip()))
    for piece in re.split(r"[\s\x00]+", raw):
        if len(piece) >= _FRAG_INNER:
            REDACT.add(Secret(piece))
    value = raw.strip()
    if not value:
        raise KitError("RunPod API key is empty")
    if not KEY_CHARSET.fullmatch(value):
        raise KitError(
            f"RunPod API key from {source} is malformed: it must be one line of 8-512 "
            "characters from [A-Za-z0-9_-] (no spaces, comments or control characters). "
            "The value is not shown."
        )
    secret = Secret(value)
    REDACT.add(secret)
    return secret


def assert_no_secret(obj: Any, secret: Secret, what: str) -> None:
    """Refuse to continue if ``secret`` appears anywhere inside ``obj``."""
    if secret.reveal() and secret.reveal() in json.dumps(obj, default=str):
        raise KitError(f"refusing: the account API key would be sent inside {what}")


def redact_payload(obj: Any) -> Any:
    """Copy of a launch payload with secret-looking env values hidden, for printing."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in _d(obj).items():
            out[str(k)] = redact_payload(v)
        key = out.get("key")
        if isinstance(key, str) and "value" in out and SECRET_ENV_HINT.search(key):
            out["value"] = "***"
        return out
    if isinstance(obj, list):
        return [redact_payload(v) for v in _l(obj)]
    if isinstance(obj, str) and len(obj) > 160:
        return obj[:80] + f"...<{len(obj)} chars>"
    return obj


# ---------------------------------------------------------------------------
# Transports.
# ---------------------------------------------------------------------------


class Transport(Protocol):
    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any: ...


_MUTATION = re.compile(r"\bmutation\b", re.IGNORECASE)


def is_mutation(query: str) -> bool:
    """Conservative: any occurrence of the word ``mutation``, or a non-query operation."""
    if _MUTATION.search(query):
        return True
    head = query.lstrip()
    return not (head.startswith("{") or head.lower().startswith("query"))


class ReadOnlyTransport:
    """Wraps a transport and refuses anything that could change account state."""

    def __init__(self, inner: Transport) -> None:
        self._inner = inner

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if is_mutation(query):
            raise MutationRefusedError("read-only transport refused a GraphQL mutation")
        return self._inner.graphql(query, variables)

    def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if method.upper() != "GET" or body is not None:
            raise MutationRefusedError(f"read-only transport refused REST {method.upper()}")
        return self._inner.rest(method, path, None)

    def log_stream(self) -> LogStream | None:
        """The inner transport's GET-only log streamer, or None if it has none."""
        return log_stream_for(self._inner)


type LogStream = Callable[[str, dict[str, str], str | None, float], Iterator[str]]


def log_stream_for(t: object) -> LogStream | None:
    """A transport's ``stream_logs`` (GET only by construction), or None."""
    if isinstance(t, ReadOnlyTransport):
        return t.log_stream()
    fn = getattr(t, "stream_logs", None)
    return cast("LogStream", fn) if callable(fn) else None


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the Bearer key must not travel to another URL."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        detail = f"refusing HTTP {code} redirect from {req.full_url} to {newurl}"
        raise KitError(REDACT(detail + ": the API key is never sent to a redirect target")[:400])


#: The only opener the kit uses for authenticated requests (no redirects followed).
_AUTH_OPENER = urllib.request.build_opener(_RefuseRedirects)


def _auth_open(req: urllib.request.Request, timeout: float) -> Any:
    """Open an authenticated request (redirects refused).  Tests patch this."""
    return _AUTH_OPENER.open(req, timeout=timeout)


class HttpTransport:
    """Real RunPod transport.  The key only ever goes into the Authorization header, as an
    unredirected header, through an opener that refuses redirects."""

    def __init__(self, key: Secret, *, timeout: float = 30.0, logs_url: str = LOGS_URL) -> None:
        self._key = key
        self._timeout = timeout
        self._logs_url = logs_url

    def _request(self, method: str, url: str, body: dict[str, Any] | None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_unredirected_header("Authorization", f"Bearer {self._key.reveal()}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "memfun-runpod-kit/1")
        # Every failure becomes a KitError whose text was redacted BEFORE truncation;
        # no original exception (which may quote the header) is chained or re-raised.
        try:
            with _auth_open(req, timeout=self._timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            msg = REDACT(f"RunPod HTTP {exc.code} for {url}: {detail}")
            raise KitError(msg[:500]) from None
        except Exception as exc:
            msg = REDACT(f"RunPod request to {url} failed: {type(exc).__name__}: {exc}")
            raise KitError(msg[:500]) from None
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            msg = REDACT(f"RunPod returned non-JSON from {url}: {text}")
            raise KitError(msg[:300]) from None

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query}
        if variables is not None:
            body["variables"] = variables
        resp = self._request("POST", GRAPHQL_URL, body)
        if not isinstance(resp, dict):
            raise KitError("RunPod GraphQL returned an unexpected body")
        payload = _d(resp)
        errors = payload.get("errors")
        if errors:
            msgs = "; ".join(str(_d(e).get("message", e)) for e in _l(errors))
            raise KitError(REDACT(f"RunPod GraphQL error: {msgs}"))
        data = payload.get("data")
        return _d(data)

    def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request(method.upper(), REST_URL + path, body)

    def stream_logs(
        self, pod_id: str, params: dict[str, str], last_event_id: str | None, timeout: float
    ) -> Iterator[str]:
        """GET the pod's SSE log stream and yield its raw text lines.  Read-only."""
        if not POD_ID_RE.fullmatch(pod_id):
            raise KitError(f"invalid pod id {pod_id!r}")
        url = self._logs_url.format(pod_id=pod_id)
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, method="GET")
        req.add_unredirected_header("Authorization", f"Bearer {self._key.reveal()}")
        req.add_header("Accept", "text/event-stream")
        req.add_header("User-Agent", "memfun-runpod-kit/1")
        if last_event_id and re.fullmatch(r"[\x20-\x7e]{1,256}", last_event_id):
            req.add_header("Last-Event-ID", last_event_id)
        try:
            resp = _auth_open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2000).decode("utf-8", "replace")
            except Exception:
                detail = ""
            ra = _retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            msg = REDACT(f"log stream HTTP {exc.code}: {detail}")[:400]
            raise LogStreamError(msg, status=int(exc.code), retry_after=ra) from None
        except Exception as exc:
            msg = REDACT(f"log stream request failed: {type(exc).__name__}: {exc}")[:400]
            raise LogStreamError(msg) from None
        with resp:
            while True:
                try:
                    raw = resp.readline(1 << 20)  # one line is at most 1 MiB
                except TimeoutError:
                    raise LogStreamError("log stream idle (read timeout)", idle=True) from None
                except Exception as exc:
                    msg = REDACT(f"log stream read failed: {type(exc).__name__}: {exc}")[:400]
                    raise LogStreamError(msg) from None
                if not raw:
                    return
                yield raw.decode("utf-8", "replace").rstrip("\r\n")


# ---------------------------------------------------------------------------
# API helpers.
# ---------------------------------------------------------------------------


def _lit(value: str) -> str:
    """A GraphQL string literal (JSON string escaping is a valid subset)."""
    return json.dumps(value)


@dataclass(frozen=True, slots=True)
class GpuType:
    id: str
    display_name: str
    memory_gb: float
    secure_price: float | None
    community_price: float | None
    secure_cloud: bool
    community_cloud: bool

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> GpuType:
        return cls(
            id=str(raw.get("id", "")),
            display_name=str(raw.get("displayName", "")),
            memory_gb=float(raw.get("memoryInGb") or 0),
            secure_price=_opt_float(raw.get("securePrice")),
            community_price=_opt_float(raw.get("communityPrice")),
            secure_cloud=bool(raw.get("secureCloud")),
            community_cloud=bool(raw.get("communityCloud")),
        )

    def list_price(self, cloud: str) -> float | None:
        return self.secure_price if cloud == "secure" else self.community_price


@dataclass(frozen=True, slots=True)
class Offer:
    stock_status: str | None
    lowest_price: float | None
    available_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Volume:
    id: str
    name: str
    size_gb: int
    data_center_id: str


@dataclass(frozen=True, slots=True)
class PodInfo:
    id: str
    name: str
    desired_status: str
    cost_per_hr: float | None
    uptime_seconds: int | None
    ports: tuple[dict[str, Any], ...]

    def ssh_endpoint(self) -> tuple[str, int] | None:
        for port in self.ports:
            if (
                port.get("privatePort") == 22
                and port.get("isIpPublic")
                and str(port.get("type", "tcp")).lower() == "tcp"
                and port.get("ip")
                and port.get("publicPort")
            ):
                return str(port["ip"]), int(port["publicPort"])
        return None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


GPU_TYPES_QUERY = (
    "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud "
    "securePrice communityPrice } }"
)


def list_gpu_types(t: Transport) -> list[GpuType]:
    data = t.graphql(GPU_TYPES_QUERY)
    return [GpuType.from_api(_d(g)) for g in _l(data.get("gpuTypes"))]


def resolve_gpu_type(types: list[GpuType], wanted: str) -> GpuType:
    """Exact id, else exact (case-insensitive) display name.  Never a fuzzy pick."""
    for g in types:
        if g.id == wanted:
            return g
    for g in types:
        if g.display_name.lower() == wanted.lower():
            return g
    near = [g.id for g in types if wanted.lower() in (g.id + " " + g.display_name).lower()]
    hint = f" Close matches: {near}." if near else " Run the `gpus` subcommand to list ids."
    raise KitError(f"unknown GPU type {wanted!r}.{hint}")


def gpu_offer(t: Transport, gpu_id: str, count: int, cloud: str, data_center: str | None) -> Offer:
    dc = f", dataCenterIds: [{_lit(data_center)}]" if data_center else ""
    secure = "true" if cloud == "secure" else "false"
    query = (
        f"query {{ gpuTypes(input: {{id: {_lit(gpu_id)}}}) {{ id "
        f"lowestPrice(input: {{gpuCount: {int(count)}, secureCloud: {secure}{dc}}}) "
        f"{{ stockStatus uninterruptablePrice availableGpuCounts }} }} }}"
    )
    data = t.graphql(query)
    rows = _l(data.get("gpuTypes"))
    lp = _d(_d(rows[0]).get("lowestPrice")) if rows else {}
    counts = tuple(int(c) for c in _l(lp.get("availableGpuCounts")))
    return Offer(
        stock_status=lp.get("stockStatus"),
        lowest_price=_opt_float(lp.get("uninterruptablePrice")),
        available_counts=counts,
    )


@dataclass(frozen=True, slots=True)
class DcStock:
    id: str
    availability: str
    supports_network_volumes: bool


def datacenter_stock(t: Transport, gpu_id: str, count: int, cloud: str) -> list[DcStock]:
    """Per-data-center availability (``lowestPrice.gpuTypeDatacenters``), read-only."""
    secure = "true" if cloud == "secure" else "false"
    data = t.graphql(
        f"query {{ gpuTypes(input: {{id: {_lit(gpu_id)}}}) {{ lowestPrice(input: "
        f"{{gpuCount: {int(count)}, secureCloud: {secure}}}) {{ gpuTypeDatacenters "
        "{ dataCenterId supportNetworkVolumes availability } } } }"
    )
    rows = _l(data.get("gpuTypes"))
    lp = _d(_d(rows[0]).get("lowestPrice")) if rows else {}
    return [
        DcStock(
            id=str(dc.get("dataCenterId")),
            availability=str(dc.get("availability")),
            supports_network_volumes=bool(dc.get("supportNetworkVolumes")),
        )
        for dc in map(_d, _l(lp.get("gpuTypeDatacenters")))
    ]


def account_summary(t: Transport) -> tuple[float | None, list[Volume]]:
    data = t.graphql(
        "query { myself { clientBalance networkVolumes { id name size dataCenterId } } }"
    )
    me = _d(data.get("myself"))
    vols = [
        Volume(
            id=str(v.get("id")),
            name=str(v.get("name")),
            size_gb=int(v.get("size") or 0),
            data_center_id=str(v.get("dataCenterId")),
        )
        for v in map(_d, _l(me.get("networkVolumes")))
    ]
    return _opt_float(me.get("clientBalance")), vols


POD_FIELDS = (
    "id name desiredStatus costPerHr "
    "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }"
)


def get_pod(t: Transport, pod_id: str) -> PodInfo | None:
    data = t.graphql(f"query {{ pod(input: {{podId: {_lit(pod_id)}}}) {{ {POD_FIELDS} }} }}")
    raw = _d(data.get("pod"))
    if not raw:
        return None
    runtime = _d(raw.get("runtime"))
    uptime = runtime.get("uptimeInSeconds")
    return PodInfo(
        id=str(raw.get("id")),
        name=str(raw.get("name")),
        desired_status=str(raw.get("desiredStatus")),
        cost_per_hr=_opt_float(raw.get("costPerHr")),
        uptime_seconds=int(uptime) if uptime is not None else None,
        ports=tuple(_d(p) for p in _l(runtime.get("ports"))),
    )


DEPLOY_MUTATION = (
    "mutation Deploy($input: PodFindAndDeployOnDemandInput!) { "
    "podFindAndDeployOnDemand(input: $input) { id name costPerHr desiredStatus } }"
)


def deploy_pod(t: Transport, payload: dict[str, Any]) -> PodInfo:
    data = t.graphql(DEPLOY_MUTATION, {"input": payload})
    raw = _d(data.get("podFindAndDeployOnDemand"))
    if not raw or not raw.get("id"):
        raise KitError("RunPod did not return a pod id (no capacity?)")
    return PodInfo(
        id=str(raw["id"]),
        name=str(raw.get("name")),
        desired_status=str(raw.get("desiredStatus")),
        cost_per_hr=_opt_float(raw.get("costPerHr")),
        uptime_seconds=None,
        ports=(),
    )


def terminate_pod(t: Transport, pod_id: str) -> None:
    # podTerminate returns a scalar; no selection set (docs.runpod.io GraphQL manage-pods).
    t.graphql(f"mutation {{ podTerminate(input: {{podId: {_lit(pod_id)}}}) }}")


def create_network_volume(t: Transport, name: str, size_gb: int, data_center: str) -> Volume:
    raw = t.rest(
        "POST", "/networkvolumes", {"name": name, "size": size_gb, "dataCenterId": data_center}
    )
    vol = _d(raw)
    if not vol.get("id"):
        raise KitError("RunPod did not return a network volume id")
    return Volume(str(vol["id"]), str(vol.get("name")), int(vol.get("size") or 0), data_center)


# ---------------------------------------------------------------------------
# Planning (pure; everything it needs comes from read-only queries).
# ---------------------------------------------------------------------------


@dataclass
class LaunchPlan:
    gpu: GpuType
    gpu_count: int
    cloud: str
    offer: Offer
    price_per_gpu_hr: float
    max_minutes: int
    volume: Volume | None
    volume_to_create: tuple[str, int, str] | None
    balance: float | None
    payload: dict[str, Any]
    data_center: str | None = None
    dc_availability: str | None = None
    warnings: list[str] = field(default_factory=list[str])

    @property
    def est_max_cost(self) -> float:
        return estimate_cost(self.price_per_gpu_hr, self.gpu_count, self.max_minutes)


def estimate_cost(price_per_gpu_hr: float, gpu_count: int, minutes: float) -> float:
    return round(price_per_gpu_hr * gpu_count * minutes / 60.0, 2)


def conservative_price(gpu: GpuType, offer: Offer, cloud: str) -> float:
    """The higher of the list price and the live lowest price: an upper bound."""
    reported = [p for p in (gpu.list_price(cloud), offer.lowest_price) if p is not None]
    bad = [p for p in reported if not (math.isfinite(p) and p > 0)]
    if bad:
        raise KitError(f"RunPod reported an invalid {cloud} price {bad} for {gpu.id}; refusing")
    if not reported:
        raise KitError(f"RunPod reported no {cloud} price for {gpu.id}; refusing to estimate")
    return max(reported)


def build_boot_env(
    args: argparse.Namespace, deadline_epoch: int, public_key: str
) -> list[dict[str, str]]:
    boot = BOOT_SCRIPT.read_bytes()
    env = {
        "PUBLIC_KEY": public_key,
        "MEMFUN_BOOT_B64": base64.b64encode(boot).decode(),
        "MEMFUN_MAX_MINUTES": str(args.max_minutes),
        "MEMFUN_DEADLINE_EPOCH": str(deadline_epoch),
        "MEMFUN_MODEL_DIR": f"{VOLUME_MOUNT}/models/{args.model_repo}",
        "MEMFUN_MODEL_REPO": args.model_repo,
        "MEMFUN_MODEL_REVISION": args.model_revision,
        "MEMFUN_SERVED_NAME": args.served_name,
        "MEMFUN_SGLANG_PORT": str(args.sglang_port),
        "MEMFUN_SGLANG_ARGS": args.sglang_args,
        "MEMFUN_ALLOW_PULL": "1" if args.pull_in_pod else "0",
        "MEMFUN_CRASH_GRACE_MIN": str(args.crash_grace_minutes),
        "MEMFUN_FAIL_GRACE_MIN": str(getattr(args, "fail_grace_minutes", 2)),
    }
    return [{"key": k, "value": v} for k, v in env.items()]


DOCKER_ARGS = (
    'bash -c \'printf %s "$MEMFUN_BOOT_B64" | base64 -d > /tmp/memfun_boot.sh '
    "&& exec bash /tmp/memfun_boot.sh'"
)


def build_payload(
    args: argparse.Namespace,
    gpu: GpuType,
    volume_id: str | None,
    data_center: str | None,
    now: dt.datetime,
    public_key: str,
) -> dict[str, Any]:
    deadline = now + dt.timedelta(minutes=args.max_minutes)
    payload: dict[str, Any] = {
        "name": args.name,
        "cloudType": args.cloud.upper(),
        "gpuTypeId": gpu.id,
        "gpuCount": args.gpu_count,
        "imageName": args.image,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": VOLUME_MOUNT,
        "ports": "22/tcp",
        "supportPublicIp": True,
        "startSsh": True,
        "dockerArgs": DOCKER_ARGS,
        "env": build_boot_env(args, int(deadline.timestamp()), public_key),
    }
    if volume_id:
        payload["networkVolumeId"] = volume_id
    if data_center:
        payload["dataCenterIds"] = [data_center]
    if args.server_terminate_after:
        payload["terminateAfter"] = deadline.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


def read_public_key(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8").strip()
    if "PRIVATE KEY" in text:
        raise KitError(f"{p} looks like a PRIVATE key; pass the .pub file")
    return text


def make_plan(t: ReadOnlyTransport, args: argparse.Namespace, now: dt.datetime) -> LaunchPlan:
    warnings: list[str] = []
    gpu = resolve_gpu_type(list_gpu_types(t), args.gpu_type)
    balance, volumes = account_summary(t)

    volume: Volume | None = None
    to_create: tuple[str, int, str] | None = None
    if args.volume_id:
        matches = [v for v in volumes if v.id == args.volume_id]
        if not matches:
            raise KitError(f"network volume {args.volume_id!r} not found on this account")
        volume = matches[0]
    elif args.volume_name:
        matches = [v for v in volumes if v.name == args.volume_name]
        if len(matches) > 1:
            raise KitError(f"{len(matches)} volumes named {args.volume_name!r}; use --volume-id")
        if matches:
            volume = matches[0]
        else:
            if not args.data_center:
                raise KitError(
                    "creating a network volume needs --data-center (pods are pinned to the "
                    "volume's data center); pick one where the GPU is in stock"
                )
            to_create = (args.volume_name, args.volume_gb, args.data_center)
    else:
        warnings.append("no network volume: weights and runs are lost when the pod terminates")
        if not args.pull_in_pod:
            warnings.append("no volume and no --pull-in-pod: the pod will find no weights and "
                            "self-terminate at boot")  # fmt: skip

    data_center = volume.data_center_id if volume else args.data_center
    if args.data_center and volume and args.data_center != volume.data_center_id:
        raise KitError(
            f"--data-center {args.data_center} != volume's data center {volume.data_center_id}"
        )
    if volume and volume.size_gb < 40:
        warnings.append(f"volume {volume.id} is {volume.size_gb} GB; FP8 weights need ~31 GB")
    if args.cloud == "community" and (volume or to_create):
        warnings.append("UNVERIFIED: network volumes may not attach to community-cloud pods")

    offer = gpu_offer(t, gpu.id, args.gpu_count, args.cloud, data_center)
    if offer.stock_status in (None, "None"):
        warnings.append(
            f"no {args.cloud} stock reported for {gpu.id}"
            + (f" in {data_center}" if data_center else "")
        )
    price = conservative_price(gpu, offer, args.cloud)
    dc_availability: str | None = None
    if data_center:
        stock = {d.id: d for d in datacenter_stock(t, gpu.id, args.gpu_count, args.cloud)}
        row = stock.get(data_center)
        if row is None:
            warnings.append(f"{data_center} does not list {gpu.id}; try `gpus --where`")
        else:
            dc_availability = row.availability
            if row.availability != "AVAILABLE":
                warnings.append(f"{gpu.id} is {row.availability} in {data_center}")
            if (volume or to_create) and not row.supports_network_volumes:
                warnings.append(f"{data_center} does not support network volumes")

    public_key = read_public_key(args.ssh_public_key)
    if not public_key:
        warnings.append(f"no SSH public key at {args.ssh_public_key}: tunnel will not work")

    payload = build_payload(
        args,
        gpu,
        volume.id if volume else "<created-at-launch>" if to_create else None,
        data_center,
        now,
        public_key,
    )
    return LaunchPlan(
        gpu=gpu,
        gpu_count=args.gpu_count,
        cloud=args.cloud,
        offer=offer,
        price_per_gpu_hr=price,
        max_minutes=args.max_minutes,
        volume=volume,
        volume_to_create=to_create,
        balance=balance,
        payload=payload,
        data_center=data_center,
        dc_availability=dc_availability,
        warnings=warnings,
    )


def print_plan(plan: LaunchPlan) -> None:
    g = plan.gpu
    emit("=== RunPod launch plan ===")
    emit(f"GPU          : {plan.gpu_count} x {g.id} ({g.display_name}, {g.memory_gb:g} GB)")
    emit(f"cloud        : {plan.cloud}")
    emit(f"data center  : {plan.data_center or 'any'}  availability: {plan.dc_availability}")
    emit(
        f"stock        : {plan.offer.stock_status}  available counts: "
        f"{list(plan.offer.available_counts)}"
    )
    emit(
        f"price/GPU/h  : list {g.list_price(plan.cloud)}  live-lowest {plan.offer.lowest_price}"
        f"  -> estimate uses ${plan.price_per_gpu_hr:.2f}"
    )
    emit(f"max minutes  : {plan.max_minutes}  (hard stop in pod + watchdog + terminateAfter)")
    emit(f"EST. MAX COST: ${plan.est_max_cost:.2f}  (GPU only; volume storage billed separately)")
    if plan.balance is not None:
        emit(f"balance      : ${plan.balance:.2f}")
        if plan.balance < plan.est_max_cost:
            plan.warnings.append("account balance is below the estimated max cost")
    if plan.volume:
        v = plan.volume
        emit(f"volume       : attach {v.id} '{v.name}' {v.size_gb} GB in {v.data_center_id}")
    elif plan.volume_to_create:
        n, s, d = plan.volume_to_create
        emit(f"volume       : CREATE '{n}' {s} GB in {d} (at launch)")
    for w in plan.warnings:
        emit(f"WARNING      : {w}")
    emit("payload (secrets redacted):")
    emit(json.dumps(redact_payload(plan.payload), indent=2))


# ---------------------------------------------------------------------------
# Watchdog.
# ---------------------------------------------------------------------------


type Clock = Callable[[], float]
type Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class WatchResult:
    reason: str
    terminated: bool
    verified_gone: bool
    spent_usd: float


_GONE_ERR = re.compile(r"not\s*found|does not exist|no pod", re.IGNORECASE)

#: Pods this process has VERIFIED gone (absent from ``myself.pods`` or TERMINATED).
VERIFIED_GONE: set[str] = set()
#: Last desiredStatus seen for a pod that is still listed (for loud reporting).
LAST_STATUS: dict[str, str] = {}
POD_LIST_QUERY = "query { myself { pods { id desiredStatus } } }"


def _pod_gone(t: Transport, pod_id: str) -> bool | None:
    """True gone, False still there, None unknown.

    Authoritative check: the account's ``myself { pods }`` listing.  Absent, or
    desiredStatus TERMINATED => gone; listed with ANY other status (RUNNING with a stopped
    container, EXITED, ERROR...) => NOT gone (it can keep billing).  Only if the listing
    fails is ``pod(input)`` used; a failed/unknown lookup is never "gone".
    """
    try:
        data = t.graphql(POD_LIST_QUERY)
        myself = data.get("myself")
        pods = _d(myself).get("pods")
        if isinstance(myself, dict) and isinstance(pods, list):
            for raw in map(_d, _l(pods)):
                if str(raw.get("id")) == pod_id:
                    status = str(raw.get("desiredStatus"))
                    if status.upper() == "TERMINATED":
                        return True
                    LAST_STATUS[pod_id] = status
                    return False
            return True
    except Exception as exc:
        emit(f"[watchdog] pod listing failed ({type(exc).__name__}); falling back", err=True)
    try:
        pod = get_pod(t, pod_id)
    except Exception as exc:
        return True if isinstance(exc, KitError) and _GONE_ERR.search(str(exc)) else None
    if pod is not None and pod.desired_status.upper() != "TERMINATED":
        LAST_STATUS[pod_id] = pod.desired_status
    return pod is None or pod.desired_status.upper() == "TERMINATED"


@contextlib.contextmanager
def signals_ignored() -> Generator[None]:
    """Ignore SIGTERM/SIGHUP/SIGINT while termination runs (restored afterwards).  A
    no-op outside the main thread (signal handlers can only be changed there)."""
    saved: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                saved[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in saved.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, handler if handler is not None else signal.SIG_DFL)


def _quiet(fn: Callable[[], object]) -> None:
    """Run a side step (print, sleep) that must never abort termination (this includes
    a KeyboardInterrupt injected or delivered mid-termination)."""
    with contextlib.suppress(BaseException):
        fn()


def terminate_and_verify(
    t: Transport,
    pod_id: str,
    *,
    sleep: Sleeper,
    retries: int = 6,
    backoff_seconds: float = 10.0,
    verify_polls: int = 3,
) -> tuple[bool, bool]:
    """Terminate with retries; return (terminate call succeeded once, verified gone).

    Survives a KeyboardInterrupt (or any exception) at every line: each step is guarded,
    and the caller should run it inside :func:`signals_ignored`.
    """
    called = False
    gone: bool | None = None
    for attempt in range(1, retries + 1):
        try:
            terminate_pod(t, pod_id)
            called = True
            _quiet(lambda a=attempt: emit(
                f"[watchdog] terminate request {a}/{retries} accepted for {pod_id}"))  # fmt: skip
        except BaseException as exc:
            _quiet(lambda a=attempt, e=exc: emit(
                f"[watchdog] terminate attempt {a}/{retries} failed: {type(e).__name__}: {e}",
                err=True))  # fmt: skip
        for _ in range(verify_polls):
            try:
                gone = _pod_gone(t, pod_id)
            except BaseException:
                gone = None
            if gone:
                VERIFIED_GONE.add(pod_id)
                _quiet(lambda: emit(f"[watchdog] verified: pod {pod_id} is gone"))
                return called, True
            _quiet(lambda: sleep(backoff_seconds))
        if gone is False:
            status = LAST_STATUS.get(pod_id, "?")
            _quiet(lambda st=status: emit(
                f"[watchdog] pod {pod_id} is still listed with desiredStatus={st} (NOT gone; "
                "it may still bill even if its container exited); retrying podTerminate",
                err=True))  # fmt: skip
    _quiet(lambda: emit(
        f"[watchdog] !!! COULD NOT VERIFY POD {pod_id} IS TERMINATED (last status "
        f"{LAST_STATUS.get(pod_id, 'unknown')}). Terminate it now at "
        "https://console.runpod.io/pods or: launch_pod.py terminate --pod-id "
        f"{pod_id} --yes !!!",
        err=True))  # fmt: skip
    return called, False


def terminate_guarded(
    t: Transport,
    pod_id: str,
    *,
    sleep: Sleeper,
    retries: int = 6,
    backoff_seconds: float = 10.0,
) -> tuple[bool, bool]:
    """terminate_and_verify with signals ignored, repeated if anything escapes it."""
    called = False
    with signals_ignored():
        for _ in range(3):
            try:
                c, gone = terminate_and_verify(
                    t, pod_id, sleep=sleep, retries=retries, backoff_seconds=backoff_seconds
                )
            except BaseException:
                continue
            called = called or c
            if gone:
                return called, True
            break
    return called, False


def run_watchdog(
    t: Transport,
    pod_id: str,
    deadline: float,
    *,
    clock: Clock = time.time,
    sleep: Sleeper = time.sleep,
    started_at: float | None = None,
    poll_seconds: float = 30.0,
    retries: int = 6,
    backoff_seconds: float = 10.0,
    max_poll_errors: int = 5,
) -> WatchResult:
    """Poll the pod, print live spend, and terminate at the deadline or on any exit path.

    Termination sits in a ``finally``: no exception (a second signal inside an ``except``
    handler, a failing print...) can skip it.  It only returns without terminating when
    the pod was VERIFIED gone already.
    """
    start = started_at if started_at is not None else 0.0
    spent = 0.0
    cost_hr: float | None = None
    reason = "unknown"
    result: WatchResult | None = None
    try:
        try:
            start = clock() if started_at is None else started_at
            poll_errors = 0
            while True:
                now = clock()
                if now >= deadline:
                    reason = "deadline"
                    break
                try:
                    pod = get_pod(t, pod_id)
                except KitError as exc:
                    poll_errors += 1
                    emit(f"[watchdog] poll error {poll_errors}/{max_poll_errors}: {exc}", err=True)
                    if poll_errors >= max_poll_errors:
                        raise
                    sleep(max(0.0, min(poll_seconds, deadline - now)))
                    continue
                poll_errors = 0
                if pod is None or pod.desired_status.upper() == "TERMINATED":
                    if _pod_gone(t, pod_id) is True:  # confirm with the listing
                        VERIFIED_GONE.add(pod_id)
                        emit(f"[watchdog] pod {pod_id} is already gone")
                        result = WatchResult("already-gone", False, True, spent)
                        return result
                    reason = "pod state unclear"
                    break
                if pod.cost_per_hr is not None:
                    cost_hr = pod.cost_per_hr
                if cost_hr is not None:
                    spent = round(cost_hr * (now - start) / 3600.0, 4)
                left = deadline - now
                rate = f"${cost_hr:.2f}/h" if cost_hr is not None else "rate n/a"
                emit(
                    f"[watchdog] {pod_id} {pod.desired_status} | {rate} | spent ${spent:.2f} "
                    f"| {int(left // 60)}m{int(left % 60):02d}s to hard stop"
                )
                sleep(max(0.0, min(poll_seconds, left)))
        except KeyboardInterrupt:
            reason = "interrupt"
        except Exception as exc:
            reason = f"error: {REDACT(str(exc))}"
    except BaseException:  # a second signal while handling the first
        reason = "interrupt"
    finally:
        if result is None:
            result = _stop_pod(t, pod_id, reason, cost_hr, start, clock, sleep,
                               retries, backoff_seconds)  # fmt: skip
    return result


def _stop_pod(
    t: Transport,
    pod_id: str,
    reason: str,
    cost_hr: float | None,
    start: float,
    clock: Clock,
    sleep: Sleeper,
    retries: int,
    backoff_seconds: float,
) -> WatchResult:
    with signals_ignored():
        spent = 0.0
        try:
            if cost_hr is not None:
                spent = round(cost_hr * (clock() - start) / 3600.0, 4)
        except BaseException:
            pass
        msg = f"[watchdog] stopping pod {pod_id} ({reason}); spent so far ~${spent:.2f}"
        _quiet(lambda: emit(msg))
        called, gone = terminate_guarded(
            t, pod_id, sleep=sleep, retries=retries, backoff_seconds=backoff_seconds
        )
    return WatchResult(reason, called, gone, spent)


# ---------------------------------------------------------------------------
# Pod logs (REST SSE).  Survive the pod only if captured locally while it runs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SseEvent:
    data: str
    event_id: str | None


class SseParser:
    """Line-fed Server-Sent Events parser.  ``data:`` lines accumulate until a blank line,
    or dispatch at once when the accumulated data is already complete JSON (RunPod sends
    one JSON object per event)."""

    def __init__(self) -> None:
        self._data: list[str] = []
        self.last_id: str | None = None

    def _dispatch(self) -> list[SseEvent]:
        if not self._data:
            return []
        ev = SseEvent("\n".join(self._data), self.last_id)
        self._data = []
        return [ev]

    def feed(self, line: str) -> list[SseEvent]:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return []  # comment / keep-alive
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "id":
            if "\x00" not in value:
                self.last_id = value
            return []
        if name == "data":
            self._data.append(value)
            try:
                json.loads("\n".join(self._data))
            except ValueError:
                return []
            return self._dispatch()
        return []  # event:, retry:, unknown fields

    def flush(self) -> list[SseEvent]:
        return self._dispatch()


def format_log_event(ev: SseEvent) -> tuple[str, bool]:
    """(printable line, is image-pull noise).  Bad JSON is shown raw, never dropped."""
    try:
        obj = json.loads(ev.data)
    except ValueError:
        return f"[raw] {ev.data}", False
    if not isinstance(obj, dict):
        return f"[raw] {ev.data}", False
    d = _d(obj)
    line = str(d.get("line", ""))
    src = str(d.get("source", "?"))[:4]
    return f"{d.get('ts', '')} [{src}] {line}", bool(PULL_NOISE.match(line))


@dataclass
class LogStats:
    events: int = 0
    hidden: int = 0
    reconnects: int = 0
    outcome: str = ""


def follow_logs(
    stream: LogStream,
    pod_id: str,
    sink: Callable[[str], None],
    *,
    follow: bool,
    source: str = "both",
    tail: int = 2000,
    raw: bool = False,
    clock: Clock = time.time,
    sleep: Sleeper = time.sleep,
    overall_timeout: float = 30.0,
    read_timeout: float = 15.0,
    max_errors: int = 5,
    backoff: float = 5.0,
    stop: threading.Event | None = None,
    is_gone: Callable[[], bool | None] | None = None,
) -> LogStats:
    """Read the pod log stream into ``sink`` (each line redacted first).

    Outcomes: ``gone`` (404: pod removed), ``ended`` (backlog read, non-follow),
    ``timeout`` (non-follow hard cap), ``stopped`` (``stop`` set), ``error`` (bounded
    consecutive failures).  Reconnects send Last-Event-ID and skip already-seen events.
    """
    params = {"tail": str(max(1, min(int(tail), MAX_LOG_TAIL)))}
    if source != "both":
        params["source"] = source
    parser = SseParser()
    stats = LogStats()
    seen: set[str] = set()
    errors = 0
    not_yet = 0
    start = clock()

    def out(text: str) -> None:
        sink(REDACT(text))

    replay = {"on": False}  # after a reconnect, skip the replayed backlog we already saw

    def handle(ev: SseEvent) -> None:
        text, noise = format_log_event(ev)
        if ev.event_id is not None:
            if ev.event_id in seen:
                return
            key = "id:" + ev.event_id
        else:
            key = "tx:" + text
            if replay["on"] and key in seen:
                return
            replay["on"] = False  # first new line: no more replay skipping
        if len(seen) < 50_000:
            seen.add(ev.event_id if ev.event_id is not None else key)
        stats.events += 1
        if noise and not raw:
            stats.hidden += 1
            return
        out(text)

    def done(outcome: str) -> LogStats:
        stats.outcome = outcome
        return stats

    while True:
        if stop is not None and stop.is_set():
            return done("stopped")
        if not follow and clock() - start >= overall_timeout:
            return done("timeout")
        replay["on"] = stats.reconnects > 0
        try:
            for line in stream(pod_id, params, parser.last_id, read_timeout):
                for ev in parser.feed(line):
                    handle(ev)
                    errors = 0
                if stop is not None and stop.is_set():
                    return done("stopped")
                if not follow and clock() - start >= overall_timeout:
                    return done("timeout")
            for ev in parser.flush():
                handle(ev)
            if not follow:
                return done("ended")
            stats.reconnects += 1
            sleep(backoff)
            continue
        except LogStreamError as exc:
            for ev in parser.flush():
                handle(ev)
            if exc.status == 404:
                # Confirm before giving up: a pod that is still initializing (no container
                # log yet) or a lookup hiccup also 404s.  Without a checker, trust the 404.
                state = True if is_gone is None else _safe_state(is_gone)
                if state is True:
                    return done("gone")
                if not follow:
                    out("# HTTP 404 but the pod still exists: no logs yet (still initializing?)")
                    return done("no-logs-yet")
                not_yet += 1
                if state is None:
                    errors += 1
                    out(f"# HTTP 404 and the pod state is unknown ({errors}/{max_errors})")
                    if errors >= max_errors:
                        return done("error")
                elif not_yet == 1:
                    out("# HTTP 404 but the pod still exists: waiting for its logs")
                stats.reconnects += 1
                sleep(min(backoff * not_yet, 30.0))
                continue
            if exc.idle:
                if not follow:
                    return done("ended")
                stats.reconnects += 1
                continue
            errors += 1
            if exc.status == 429:
                wait = exc.retry_after if exc.retry_after is not None else backoff * errors
                wait = max(1.0, min(wait, 300.0))
                out(f"# log stream rate-limited (429); retrying in {wait:g}s")
            else:
                wait = backoff * errors
                out(f"# log stream error {errors}/{max_errors}: {exc}")
        except Exception as exc:  # a fake/buggy stream must never kill the caller
            errors += 1
            wait = backoff * errors
            out(f"# log stream error {errors}/{max_errors}: {type(exc).__name__}: {exc}")
        if errors >= max_errors:
            return done("error")
        stats.reconnects += 1
        sleep(wait)


def _safe_state(is_gone: Callable[[], bool | None]) -> bool | None:
    try:
        return is_gone()
    except Exception:
        return None


def _check_pod_id(pod_id: str) -> str:
    if not POD_ID_RE.fullmatch(pod_id):
        raise KitError(f"invalid pod id {pod_id!r}")
    return pod_id


_NEW_FILE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def open_new_private_file(path: Path) -> int:
    """Create ``path`` 0600; refuse anything already there (incl. a planted symlink)."""
    try:
        return os.open(path, _NEW_FILE_FLAGS, 0o600)
    except FileExistsError:
        what = "a symlink" if path.is_symlink() else "an existing file"
        raise KitError(f"refusing to write {path}: it is {what}") from None
    except OSError as exc:
        raise KitError(f"cannot create {path}: {exc.strerror}") from None


def open_capture_file(out_dir: Path, pod_id: str) -> tuple[Path, int]:
    """``<out_dir>/pod-<id>.log`` (or ``pod-<id>.N.log`` if taken), created 0600 O_EXCL."""
    _check_pod_id(pod_id)
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for n in range(20):
        path = out_dir / (f"pod-{pod_id}.log" if n == 0 else f"pod-{pod_id}.{n}.log")
        try:
            return path, open_new_private_file(path)
        except KitError as exc:
            if path.is_symlink():
                emit(f"warning: {exc}; not following it", err=True)
            continue
    raise KitError(f"no free log file name for pod {pod_id} in {out_dir}")


class LogCapture:
    """Daemon thread that follows a pod's log stream into a local 0600 file until the
    pod is gone.  Nothing here may affect the watchdog: every failure is swallowed."""

    #: Seconds between reconnects (and the base of the error backoff).
    backoff: float = 5.0
    #: Local file size cap; the capture stops with a final line when reached.
    max_bytes: int = 200 * 1024 * 1024

    def __init__(
        self,
        stream: LogStream,
        pod_id: str,
        path: Path,
        fd: int,
        is_gone: Callable[[], bool | None] | None = None,
    ) -> None:
        self.path = path
        self._is_gone = is_gone
        self._written = 0
        self.stats: LogStats | None = None
        self._stream = stream
        self._pod_id = pod_id
        self._fh = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"podlogs-{pod_id}", daemon=True)

    @classmethod
    def start(cls, transport: object, pod_id: str, out_dir: Path) -> LogCapture | None:
        try:
            stream = log_stream_for(transport)
            if stream is None:
                emit("log capture: this transport cannot stream pod logs (skipped)")
                return None
            path, fd = open_capture_file(out_dir, pod_id)
            graphql = getattr(transport, "graphql", None)
            checker = (
                (lambda: _pod_gone(cast("Transport", transport), pod_id)) if graphql else None
            )
            cap = cls(stream, pod_id, path, fd, checker)
            cap._thread.start()
            return cap
        except Exception as exc:
            emit(f"log capture disabled: {REDACT(str(exc))}", err=True)
            return None

    def _write(self, text: str) -> None:
        with self._lock:
            if self._fh.closed or self._written >= self.max_bytes:
                return
            line = REDACT(text) + "\n"
            self._written += len(line.encode("utf-8", "replace"))
            if self._written >= self.max_bytes:
                self._fh.write(
                    f"# capture stopped: file reached its {self.max_bytes} byte cap "
                    f"(use the logs command for the rest)\n"
                )
                self._stop.set()
                return
            self._fh.write(line)

    def _run(self) -> None:
        try:
            self.stats = follow_logs(
                self._stream, self._pod_id, self._write, follow=True, stop=self._stop,
                is_gone=self._is_gone,
                read_timeout=60.0, backoff=self.backoff, max_errors=20,
            )  # fmt: skip
            self._write(f"# capture ended: {self.stats.outcome} ({self.stats.events} events)")
        except BaseException as exc:  # the thread must never raise
            with contextlib.suppress(Exception):
                self._write(f"# capture failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock, contextlib.suppress(Exception):
                self._fh.close()

    def finish(self, wait: float = 5.0) -> None:
        """Give the stream ``wait`` s to deliver the last lines (404 ends it), then stop."""
        with contextlib.suppress(Exception):
            self._thread.join(wait)
            self._stop.set()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


def logs_command(pod_id: str) -> str:
    return f"{KIT_CMD} logs --pod-id {pod_id} --follow"


def no_ssh_endpoint_message(pod: PodInfo) -> str:
    up = pod.uptime_seconds
    uptime = f"{up // 60}m{up % 60:02d}s" if up is not None else "n/a (not started)"
    return (
        f"pod {pod.id} has no public TCP 22 mapping yet: it is still initializing "
        f"(status {pod.desired_status}, uptime {uptime}). RunPod leaves the port mapping "
        "empty until the container starts, and the image pull alone can take 10+ minutes "
        f"on some hosts. Watch the boot with: {logs_command(pod.id)}"
    )


# ---------------------------------------------------------------------------
# Tunnel.
# ---------------------------------------------------------------------------


def check_local_port(port: int) -> None:
    if port in FORBIDDEN_LOCAL_PORTS:
        raise KitError(f"refusing local port {port}: reserved for the owner's local LLM server")
    if not 1024 <= port <= 65535:
        raise KitError(f"local port {port} out of range 1024-65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            raise KitError(f"local port {port} is already in use; pick another") from None


def default_known_hosts(pod_id: str) -> Path:
    """Per-pod known_hosts outside ~/.ssh (RunPod reuses IP:port across pods)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", pod_id)
    return Path(tempfile.gettempdir()) / f"memfun-runpod-{safe}.known_hosts"


def build_ssh_command(
    host: str,
    ssh_port: int,
    local_port: int,
    remote_port: int,
    identity: str | None,
    known_hosts: Path | None = None,
) -> list[str]:
    """ssh -N -L ...; trust-on-first-use against a per-pod known_hosts, never ~/.ssh."""
    if local_port in FORBIDDEN_LOCAL_PORTS:
        raise KitError(f"refusing local port {local_port}")
    hosts = known_hosts if known_hosts is not None else default_known_hosts(f"{host}-{ssh_port}")
    cmd = [
        "ssh", "-N",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        "-p", str(ssh_port),
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
    ]  # fmt: skip
    if identity:
        cmd += ["-i", str(Path(identity).expanduser())]
    cmd.append(f"root@{host}")
    return cmd


def ssh_env(base: dict[str, str]) -> dict[str, str]:
    """Environment for ssh: drop the RunPod key and anything that looks like a credential."""
    return {
        k: v
        for k, v in base.items()
        if k != KEY_ENV and not k.startswith("RUNPOD_") and not SECRET_ENV_HINT.search(k)
    }


def memfun_snippet(local_port: int, served_name: str) -> str:
    """Qwen everywhere: one [llm] block that every memfun role inherits.  No cloud key."""
    return (
        "# memfun.toml: every role (orchestrator, decomposer, workers, auditors, reviewer)\n"
        "# inherits [llm], so all of them use the model served by SGLang on the pod.\n"
        "[llm]\n"
        'provider = "openai"\n'
        f'base_url = "http://127.0.0.1:{local_port}/v1"\n'
        f'model = "{served_name}"\n'
        f'api_key_env = "{LLM_KEY_ENV}"  # NAME of a dummy env var; SGLang needs no key\n'
        "# Once per-role models land ([llm.roles.<role>]), single roles can be overridden,\n"
        "# e.g. Claude Opus 5.5 for orchestrator/decomposer as a comparison arm.\n"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


type TransportFactory = Callable[[Secret], Transport]


def _default_factory(key: Secret) -> Transport:
    return HttpTransport(key)


def _gpu_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--gpu-count must be an integer, got {text!r}") from None
    if not 1 <= value <= MAX_GPU_COUNT:
        raise argparse.ArgumentTypeError(f"--gpu-count must be 1..{MAX_GPU_COUNT}")
    return value


def _cost_cap(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--max-cost-usd must be a number, got {text!r}"
        ) from None
    if not (math.isfinite(value) and 0 < value <= MAX_COST_CAP_USD):
        raise argparse.ArgumentTypeError(
            f"--max-cost-usd must be finite and in (0, {MAX_COST_CAP_USD:g}]"
        )
    return value


def _check_name(value: str, what: str) -> str:
    if (
        not REPO_NAME_RE.fullmatch(value)
        or value.startswith("-")
        or any(part.startswith("-") for part in value.split("/"))
        or any(part in (".", "..") for part in value.split("/"))
        or ".." in value
    ):
        raise KitError(f"{what} {value!r} must look like org/name using [A-Za-z0-9._-]")
    return value


def validate_launch_args(args: argparse.Namespace) -> None:
    """Belt and braces on top of the argparse types; runs before any transport exists."""
    if not (isinstance(args.gpu_count, int) and 1 <= args.gpu_count <= MAX_GPU_COUNT):
        raise KitError(f"--gpu-count must be an integer 1..{MAX_GPU_COUNT}")
    if not (isinstance(args.max_minutes, int) and 1 <= args.max_minutes <= MAX_MINUTES_CAP):
        raise KitError(f"--max-minutes must be an integer 1..{MAX_MINUTES_CAP}")
    cap = args.max_cost_usd
    if not (isinstance(cap, float | int) and math.isfinite(cap) and 0 < cap <= MAX_COST_CAP_USD):
        raise KitError("--max-cost-usd must be finite and > 0")
    _check_name(args.model_repo, "--model-repo")
    _check_name(args.served_name, "--served-name")
    rev = args.model_revision
    if not REVISION_RE.fullmatch(rev) or ".." in rev or rev.startswith("-"):
        raise KitError("--model-revision must be a branch, tag or commit sha")


def _positive_minutes(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--max-minutes must be an integer, got {text!r}"
        ) from None
    if not 1 <= value <= MAX_MINUTES_CAP:
        raise argparse.ArgumentTypeError(f"--max-minutes must be 1..{MAX_MINUTES_CAP}")
    return value


def _grace_minutes(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"grace must be an integer, got {text!r}") from None
    if not 0 <= value <= 60:
        raise argparse.ArgumentTypeError("grace must be 0..60 minutes")
    return value


def _log_timeout(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--timeout must be a number, got {text!r}") from None
    if not (math.isfinite(value) and 1 <= value <= 3600):
        raise argparse.ArgumentTypeError("--timeout must be finite, 1..3600 seconds")
    return value


def _log_tail(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--tail must be an integer, got {text!r}") from None
    if not 1 <= value <= MAX_LOG_TAIL:
        raise argparse.ArgumentTypeError(f"--tail must be 1..{MAX_LOG_TAIL}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runpod_kit", description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gpus", help="list GPU types and prices (read-only)")
    g.add_argument("--filter", default="", help="substring of id/displayName")
    g.add_argument("--where", default=None, metavar="GPU_TYPE_ID",
                   help="show per-data-center stock for this gpuTypeId")  # fmt: skip
    g.add_argument("--gpu-count", type=int, default=1)
    g.add_argument("--cloud", choices=("secure", "community"), default="secure")

    la = sub.add_parser("launch", help="plan (default) or rent a pod")
    la.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="RunPod gpuTypeId")
    la.add_argument("--gpu-count", type=_gpu_count, default=1)
    la.add_argument("--cloud", choices=("secure", "community"), default="secure")
    la.add_argument("--image", default=DEFAULT_IMAGE, help="verify the tag before launch")
    la.add_argument("--name", default="memfun-swarm")
    la.add_argument("--volume-id", default=None, help="attach this existing network volume")
    la.add_argument("--volume-name", default="memfun-weights", help="attach by name or create")
    la.add_argument("--no-volume", action="store_true", help="run without a network volume")
    la.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    la.add_argument("--data-center", default=None, help="e.g. EU-RO-1 (needed to create)")
    la.add_argument("--container-disk-gb", type=int, default=50)
    la.add_argument("--max-minutes", type=_positive_minutes, default=DEFAULT_MAX_MINUTES)
    la.add_argument("--max-cost-usd", type=_cost_cap, default=DEFAULT_MAX_COST_USD)
    la.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    la.add_argument("--model-revision", default="main", help="branch/tag/sha to serve")
    la.add_argument("--served-name", default=DEFAULT_SERVED_NAME)
    la.add_argument("--sglang-port", type=int, default=SGLANG_DEFAULT_PORT)
    la.add_argument("--sglang-args", default="", help="extra launch_server args")
    la.add_argument("--pull-in-pod", action="store_true", help="download weights if missing")
    la.add_argument("--crash-grace-minutes", type=int, default=10)
    la.add_argument(
        "--fail-grace-minutes",
        type=_grace_minutes,
        default=2,
        help="keep a pod whose boot failed this long so its logs can be read (not the deadline)",
    )
    la.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    la.add_argument(
        "--server-terminate-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also set the pod's terminateAfter field (server-side stop)",
    )
    la.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    la.add_argument("--yes-spend-money", action="store_true", help="required with --no-dry-run")
    la.add_argument("--poll-seconds", type=float, default=30.0)

    w = sub.add_parser("watchdog", help="terminate an existing pod at a deadline / Ctrl-C")
    w.add_argument("--pod-id", required=True)
    w.add_argument("--max-minutes", type=_positive_minutes, required=True,
                   help="terminate this many minutes from now")  # fmt: skip
    w.add_argument("--poll-seconds", type=float, default=30.0)

    tu = sub.add_parser("tunnel", help="SSH tunnel local port -> pod SGLang")
    tu.add_argument("--pod-id", required=True)
    tu.add_argument("--local-port", type=int, default=DEFAULT_LOCAL_PORT)
    tu.add_argument("--remote-port", type=int, default=SGLANG_DEFAULT_PORT)
    tu.add_argument("--identity", default=None, help="ssh private key path (-i)")
    tu.add_argument("--served-name", default=DEFAULT_SERVED_NAME)
    tu.add_argument("--run", action="store_true", help="actually start ssh (foreground)")

    sp = sub.add_parser("spend", help="live $ spent for a pod (read-only)")
    sp.add_argument("--pod-id", required=True)

    wa = sub.add_parser(
        "watch",
        help="tunnel + readiness check + start memfun (dashboard) with metrics/rate env",
    )
    wa.add_argument("--pod-id", required=True)
    wa.add_argument("--local-port", type=int, default=DEFAULT_LOCAL_PORT)
    wa.add_argument("--remote-port", type=int, default=SGLANG_DEFAULT_PORT)
    wa.add_argument("--identity", default=None, help="ssh private key path (-i)")
    wa.add_argument("--served-name", default=DEFAULT_SERVED_NAME)
    wa.add_argument("--ready-retries", type=int, default=60)
    wa.add_argument("--ready-interval", type=float, default=10.0)
    wa.add_argument("--no-caffeinate", action="store_true")
    wa.add_argument(
        "command", nargs=argparse.REMAINDER, help="memfun command after --; default: memfun chat"
    )

    te = sub.add_parser("terminate", help="terminate ONE pod and verify it is gone (destructive)")
    te.add_argument("--pod-id", required=True)
    te.add_argument("--yes", action="store_true", help="required: really terminate it")

    st = sub.add_parser("status", help="show a pod (read-only)")
    st.add_argument("--pod-id", required=True)

    lg = sub.add_parser(
        "logs", help="print a pod's container/system log (read-only; gone with the pod)"
    )
    lg.add_argument("--pod-id", required=True)
    lg.add_argument("--follow", action="store_true", help="keep streaming until the pod is gone")
    lg.add_argument("--out", default=None, help="also write to this NEW file (0600)")
    lg.add_argument("--source", choices=("container", "system", "both"), default="both")
    lg.add_argument("--tail", type=_log_tail, default=2000, help=f"1..{MAX_LOG_TAIL}")
    lg.add_argument("--raw", action="store_true", help="show image-pull progress lines too")
    lg.add_argument(
        "--timeout", type=_log_timeout, default=30.0, help="non-follow overall cap, 1..3600 s"
    )
    for sp_ in (la, wa):
        sp_.add_argument(
            "--log-dir", default=str(DEFAULT_LOG_DIR), help="local pod-log capture directory"
        )
    return p


def cmd_gpus(ro: ReadOnlyTransport, args: argparse.Namespace) -> int:
    if args.where:
        rows_dc = datacenter_stock(ro, args.where, args.gpu_count, args.cloud)
        emit(f"{args.gpu_count} x {args.where} ({args.cloud}) by data center:")
        emit(f"{'dataCenterId':<14} {'availability':<14} network-volumes")
        for r in sorted(rows_dc, key=lambda r: r.id):
            vols = "yes" if r.supports_network_volumes else "no"
            emit(f"{r.id:<14} {r.availability:<14} {vols}")
        if not rows_dc:
            emit("(no data center reports this GPU type)")
        return 0
    rows = [
        g for g in list_gpu_types(ro)
        if args.filter.lower() in (g.id + " " + g.display_name).lower()
    ]  # fmt: skip
    emit(f"{'gpuTypeId':<34} {'displayName':<22} {'GB':>5} {'secure$/h':>10} {'comm$/h':>9}")
    for g in sorted(rows, key=lambda r: r.id):
        emit(
            f"{g.id:<34} {g.display_name:<22} {g.memory_gb:>5g} "
            f"{g.secure_price!s:>10} {g.community_price!s:>9}"
        )
    return 0


# ---------------------------------------------------------------------------
# Run card, live spend, and the watch wrapper (tunnel -> readiness -> memfun dashboard).
# ---------------------------------------------------------------------------

KIT_CMD = "uv run python scripts/runpod/launch_pod.py"
METRICS_ENV = "MEMFUN_INFERENCE_METRICS_URL"
RATE_ENV = "MEMFUN_POD_RATE_USD"


def metrics_url(local_port: int) -> str:
    return f"http://127.0.0.1:{local_port}/metrics"


def format_rate(rate: float) -> str:
    return f"{rate:.4f}".rstrip("0").rstrip(".")


def run_card(
    *,
    pod_id: str | None,
    gpu_id: str,
    gpu_count: int,
    rate_per_hr: float,
    rate_source: str,
    deadline_utc: dt.datetime,
    minutes: int,
    local_port: int = DEFAULT_LOCAL_PORT,
    log_file: Path | None = None,
) -> list[str]:
    """The launch printout.  Contains no secrets by construction (no key is passed in)."""
    dry = pod_id is None
    pid = pod_id or "<POD_ID>"
    local = deadline_utc.astimezone()
    lines = [
        "=== DRY RUN: run card (nothing created) ===" if dry else "=== run card ===",
        f"pod id     : {'<none: dry run>' if dry else pod_id}",
        f"GPU        : {gpu_count} x {gpu_id}",
        f"rate       : ${rate_per_hr:.2f}/h ({rate_source})",
        f"hard stop  : {local:%Y-%m-%d %H:%M:%S %Z} local (in {minutes} min, auto-terminate)",
        f"live spend : {KIT_CMD} spend --pod-id {pid}",
        f"tunnel     : {KIT_CMD} tunnel --pod-id {pid} --run",
        f"watch      : {KIT_CMD} watch --pod-id {pid} -- memfun chat",
        f"env        : export {METRICS_ENV}={metrics_url(local_port)}",
        f"             export {RATE_ENV}={format_rate(rate_per_hr)}",
        f"pod logs   : {logs_command(pid)}   (gone once the pod is removed)",
    ]
    if log_file is not None:
        lines.append(f"log file   : {log_file}  (captured locally; survives the pod)")
    lines += [
        "note       : 'no public TCP 22 mapping yet' = pod still initializing (image pull can",
        "             take 10+ min on some hosts); the hard stop counts from LAUNCH time",
        "dashboard: watch in the Swarm app",
    ]
    return lines


def cmd_logs(
    ro: ReadOnlyTransport, args: argparse.Namespace, *, clock: Clock, sleep: Sleeper
) -> int:
    pod_id = _check_pod_id(args.pod_id)
    stream = ro.log_stream()
    if stream is None:
        raise KitError("this transport cannot stream pod logs")
    fh: Any = None
    if args.out:
        fh = os.fdopen(open_new_private_file(Path(args.out).expanduser()), "w", encoding="utf-8")

    def sink(text: str) -> None:
        emit(text)
        if fh is not None:
            fh.write(REDACT(text) + "\n")
            fh.flush()

    try:
        stats = follow_logs(
            stream, pod_id, sink, follow=args.follow, source=args.source, tail=args.tail,
            raw=args.raw, clock=clock, sleep=sleep, overall_timeout=args.timeout,
            is_gone=lambda: _pod_gone(ro, pod_id),
        )  # fmt: skip
    finally:
        if fh is not None:
            fh.close()
    hidden = f" ({stats.hidden} image-pull progress lines hidden; --raw shows them)"
    emit(f"# {stats.events} log events{hidden if stats.hidden else ''}; end: {stats.outcome}")
    if stats.outcome == "gone":
        emit(
            f"# pod {pod_id} is gone or does not exist (HTTP 404, and not in the pod list): "
            "RunPod deletes a pod's logs with it"
        )
        return 0
    return 1 if stats.outcome == "error" else 0


def cmd_terminate(
    factory: TransportFactory, key: Secret, args: argparse.Namespace, *, sleep: Sleeper
) -> int:
    """Deliberate, explicit termination of ONE pod.  Needs --yes; exit 0 only if verified."""
    pod_id = _check_pod_id(args.pod_id)
    if not args.yes:
        emit(f"refusing: terminating pod {pod_id} is destructive; add --yes", err=True)
        return 2
    t = factory(key)  # mutating transport only after the explicit --yes
    VERIFIED_GONE.discard(pod_id)
    called, gone = terminate_guarded(t, pod_id, sleep=sleep)
    last = LAST_STATUS.get(pod_id, "unknown")
    state = (
        "gone (absent from the pod list or TERMINATED)"
        if gone
        else f"NOT verified gone (last desiredStatus {last})"
    )
    emit(f"terminate {pod_id}: request accepted={called}; final state: {state}")
    return 0 if gone else 3


def cmd_spend(ro: ReadOnlyTransport, pod_id: str) -> int:
    pod = get_pod(ro, pod_id)
    if pod is None:
        raise KitError(f"pod {pod_id} not found (terminated?)")
    if pod.cost_per_hr is None:
        raise KitError(f"RunPod reports no costPerHr for {pod_id}")
    up = pod.uptime_seconds or 0
    spent = pod.cost_per_hr * up / 3600.0
    emit(
        f"{pod_id} {pod.desired_status}: ${pod.cost_per_hr:.2f}/h x {up // 60}m{up % 60:02d}s "
        f"uptime = ~${spent:.2f} (uptime-based; billing can also include start-up time)"
    )
    return 0


def _http_get_local(url: str, timeout: float) -> tuple[int, bytes]:
    if not url.startswith("http://127.0.0.1:"):
        raise KitError(f"refusing non-local readiness URL {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, OSError) as exc:
        raise KitError(f"GET {url} failed: {exc}") from None


@dataclass
class WatchDeps:
    """Side effects of ``watch``, injectable for tests."""

    popen: Callable[..., Any] = subprocess.Popen
    which: Callable[[str], str | None] = shutil.which
    http_get: Callable[[str, float], tuple[int, bytes]] = _http_get_local
    sleep: Sleeper = time.sleep
    environ: dict[str, str] | None = None


def wait_ready(
    local_port: int,
    served_name: str,
    deps: WatchDeps,
    retries: int,
    interval: float,
    tunnel: Any = None,
) -> list[str]:
    """Read-only readiness: GET /metrics and /v1/models through the tunnel, with retries."""
    base = f"http://127.0.0.1:{local_port}"
    last = "not tried"
    for attempt in range(1, max(1, retries) + 1):
        if tunnel is not None and tunnel.poll() is not None:
            raise KitError(f"ssh tunnel exited (rc={tunnel.returncode}); check ssh access")
        try:
            m_status, _ = deps.http_get(f"{base}/metrics", 10.0)
            v_status, body = deps.http_get(f"{base}/v1/models", 10.0)
            if m_status == 200 and v_status == 200:
                models = [str(_d(m).get("id")) for m in _l(_d(json.loads(body)).get("data"))]
                emit(f"ready: /metrics 200, /v1/models {models}")
                if served_name not in models:
                    emit(f"warning: served name {served_name!r} not in {models}", err=True)
                return models
            last = f"/metrics {m_status}, /v1/models {v_status}"
            if m_status == 404:
                last += " (is SGLang running with --enable-metrics?)"
        except (KitError, json.JSONDecodeError) as exc:
            last = str(exc)
        emit(f"waiting for SGLang ({attempt}/{retries}): {last}")
        deps.sleep(interval)
    raise KitError(f"SGLang not ready through the tunnel after {retries} tries: {last}")


def build_watch_env(base: dict[str, str], local_port: int, rate: float | None) -> dict[str, str]:
    env = {k: v for k, v in base.items() if k != KEY_ENV}  # never hand memfun the RunPod key
    env[METRICS_ENV] = metrics_url(local_port)
    if rate is not None:
        env[RATE_ENV] = format_rate(rate)
    else:
        env.pop(RATE_ENV, None)
    env.setdefault(LLM_KEY_ENV, "unused")
    return env


def caffeinate_argv(path: str, pid: int) -> list[str]:
    return [path, "-dimsu", "-w", str(pid)]


def cmd_watch(ro: ReadOnlyTransport, args: argparse.Namespace, deps: WatchDeps) -> int:
    check_local_port(args.local_port)
    pod = get_pod(ro, args.pod_id)
    if pod is None:
        raise KitError(f"pod {args.pod_id} not found")
    endpoint = pod.ssh_endpoint()
    if endpoint is None:
        raise KitError(no_ssh_endpoint_message(pod))
    capture = LogCapture.start(ro, args.pod_id, Path(args.log_dir))
    if capture is not None:
        emit(f"pod log capture: {capture.path}")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]  # only the first separator belongs to us
    command = command or ["memfun", "chat"]
    base_env = dict(os.environ) if deps.environ is None else deps.environ
    run_dir = Path(tempfile.mkdtemp(prefix="memfun-runpod-"))
    ssh = build_ssh_command(
        endpoint[0],
        endpoint[1],
        args.local_port,
        args.remote_port,
        args.identity,
        known_hosts=run_dir / "known_hosts",
    )
    emit(f"export {METRICS_ENV}={metrics_url(args.local_port)}")
    if pod.cost_per_hr is not None:
        emit(f"export {RATE_ENV}={format_rate(pod.cost_per_hr)}")
    else:
        emit(f"warning: RunPod reports no costPerHr; {RATE_ENV} not set", err=True)
    emit("tunnel: " + shlex.join(ssh))
    emit(memfun_snippet(args.local_port, args.served_name))
    # New sessions: Ctrl-C inside memfun must not kill the tunnel or caffeinate.
    tunnel = deps.popen(ssh, start_new_session=True, env=ssh_env(base_env))
    caff: Any = None
    child: Any = None
    try:
        wait_ready(
            args.local_port,
            args.served_name,
            deps,
            args.ready_retries,
            args.ready_interval,
            tunnel,
        )
        env = build_watch_env(base_env, args.local_port, pod.cost_per_hr)
        emit("starting: " + shlex.join(command))
        child = deps.popen(command, env=env)
        caff_path = None if args.no_caffeinate else deps.which("caffeinate")
        if caff_path:
            caff = deps.popen(caffeinate_argv(caff_path, child.pid), start_new_session=True)
            emit(f"caffeinate: keeping the Mac awake until pid {child.pid} exits")
        else:
            emit("caffeinate not available (not macOS?) or disabled: the machine may sleep")
        emit("dashboard: watch in the Swarm app")
        while True:
            try:
                return int(child.wait())
            except TerminateRequestedError:
                emit(
                    "watch: terminating memfun and the tunnel. The POD KEEPS RUNNING until the "
                    "launch watchdog / in-pod hard stop ends it (or Ctrl-C the launch terminal).",
                    err=True,
                )
                return 143
            except KeyboardInterrupt:
                continue  # Ctrl-C belongs to memfun; the wrapper leaves when memfun exits
    finally:
        for proc in (child, caff, tunnel):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        shutil.rmtree(run_dir, ignore_errors=True)
        if capture is not None:
            capture.finish(wait=0.5)


def cmd_launch(
    key: Secret,
    factory: TransportFactory,
    args: argparse.Namespace,
    *,
    clock: Clock,
    sleep: Sleeper,
    now: dt.datetime,
) -> int:
    validate_launch_args(args)
    if args.no_volume:
        args.volume_name = None
        args.volume_id = None
    ro = ReadOnlyTransport(factory(key))
    plan = make_plan(ro, args, now)
    print_plan(plan)
    if args.dry_run:
        emit("")
        card = run_card(
            pod_id=None,
            gpu_id=plan.gpu.id,
            gpu_count=plan.gpu_count,
            rate_per_hr=plan.price_per_gpu_hr * plan.gpu_count,
            rate_source="plan estimate from RunPod prices",
            deadline_utc=now + dt.timedelta(minutes=plan.max_minutes),
            minutes=plan.max_minutes,
        )
        for line in card:
            emit(line)
        emit("\n" + memfun_snippet(DEFAULT_LOCAL_PORT, args.served_name))
        emit("\nDRY RUN: nothing was created. To rent, add --no-dry-run --yes-spend-money.")
        return 0
    if not args.yes_spend_money:
        emit("refusing: --no-dry-run also needs --yes-spend-money", err=True)
        return 2
    est = plan.est_max_cost
    if not (math.isfinite(est) and est > 0):
        emit(f"refusing: the cost estimate {est!r} is not a positive number", err=True)
        return 2
    if est > args.max_cost_usd:
        emit(
            f"refusing: estimated max cost ${plan.est_max_cost:.2f} exceeds "
            f"--max-cost-usd {args.max_cost_usd:.2f}",
            err=True,
        )
        return 2
    if plan.offer.stock_status in (None, "None"):
        emit("refusing: RunPod reports no stock for this GPU/cloud/data center", err=True)
        return 2
    if plan.data_center and plan.dc_availability != "AVAILABLE":
        emit(f"refusing: {plan.gpu.id} is not AVAILABLE in {plan.data_center}", err=True)
        return 2
    if not any(e["key"] == "PUBLIC_KEY" and e["value"] for e in plan.payload["env"]):
        emit("refusing: no SSH public key (see --ssh-public-key)", err=True)
        return 2

    # Only past this point does a mutating transport exist.
    rw = factory(key)
    ro_rw = ReadOnlyTransport(rw)
    emit(f"\nLAUNCHING: up to ${est:.2f} will be spent.")
    created_volume: Volume | None = None
    if plan.volume_to_create:
        n, sz, dc = plan.volume_to_create
        created_volume = create_network_volume(rw, n, sz, dc)
        emit(f"created network volume {created_volume.id} ({sz} GB, {dc})")
        plan.payload["networkVolumeId"] = created_volume.id
    # A unique name lets us find the pod if the create call dies after RunPod made it.
    pod_name = f"{args.name}-{uuid.uuid4().hex[:10]}"
    plan.payload["name"] = pod_name
    assert_no_secret(plan.payload, key, "the pod payload")
    started = clock()
    deadline = started + plan.max_minutes * 60
    pod_id: str | None = None
    handed_to_watchdog = False
    capture: LogCapture | None = None
    try:
        try:
            pod = deploy_pod(rw, plan.payload)
            pod_id = pod.id
        except BaseException as exc:
            emit(f"pod create failed or was interrupted: {type(exc).__name__}: {exc}", err=True)
            pod_id = find_pod_id_by_name(ro_rw, pod_name, sleep=sleep)
            if pod_id is None:
                _warn_orphans(pod_name, created_volume)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                return 4
            emit(f"RunPod did create pod {pod_id} ({pod_name}); terminating it", err=True)
            return 4
        emit(f"pod {pod.id} created ({pod.desired_status}) as {pod_name}")
        # Logs vanish with the pod: follow them into a local file from the start.
        capture = LogCapture.start(ro_rw, pod.id, Path(args.log_dir))
        if pod.cost_per_hr is None:
            fresh = get_pod(rw, pod.id)
            pod = dataclasses.replace(pod, cost_per_hr=fresh.cost_per_hr if fresh else None)
        reported = pod.cost_per_hr is not None
        card = run_card(
            pod_id=pod.id,
            gpu_id=plan.gpu.id,
            gpu_count=plan.gpu_count,
            rate_per_hr=pod.cost_per_hr
            if pod.cost_per_hr is not None
            else plan.price_per_gpu_hr * plan.gpu_count,
            rate_source="reported by RunPod for this pod" if reported else "plan estimate",
            deadline_utc=now + dt.timedelta(minutes=plan.max_minutes),
            minutes=plan.max_minutes,
            log_file=capture.path if capture is not None else None,
        )
        for line in card:
            emit(line)
        emit("\n" + memfun_snippet(DEFAULT_LOCAL_PORT, args.served_name))
        handed_to_watchdog = True  # run_watchdog terminates on every exit path itself
        result = run_watchdog(
            rw, pod.id, deadline, clock=clock, sleep=sleep, started_at=started,
            poll_seconds=args.poll_seconds,
        )  # fmt: skip
        emit(f"done: {result.reason}; ~${result.spent_usd:.2f} spent; gone={result.verified_gone}")
        return 0 if result.verified_gone else 3
    finally:
        # Safety net, independent of handed_to_watchdog: a created pod that was not
        # VERIFIED gone is terminated here, whatever escaped above (e.g. a second signal).
        if pod_id is not None and pod_id not in VERIFIED_GONE:
            what = "was handed to the watchdog" if handed_to_watchdog else "is not watched yet"
            _quiet(lambda: emit(
                f"[launch] pod {pod_id} {what} but is not verified gone: terminating it",
                err=True))  # fmt: skip
            terminate_guarded(rw, pod_id, sleep=sleep)
        if capture is not None:
            capture.finish()
            emit(f"pod log saved: {capture.path}")


def find_pod_id_by_name(
    t: Transport, name: str, *, sleep: Sleeper, attempts: int = 3, backoff: float = 10.0
) -> str | None:
    """Read-only ``myself { pods }`` lookup of the kit-generated unique pod name."""
    for attempt in range(1, attempts + 1):
        try:
            data = t.graphql("query { myself { pods { id name desiredStatus } } }")
            for pod in map(_d, _l(_d(data.get("myself")).get("pods"))):
                if pod.get("name") == name and pod.get("id"):
                    return str(pod["id"])
            return None
        except KitError as exc:
            emit(f"[launch] pod lookup {attempt}/{attempts} failed: {exc}", err=True)
            sleep(backoff)
    return None


def _warn_orphans(pod_name: str, volume: Volume | None) -> None:
    emit(
        f"!!! COULD NOT CONFIRM WHETHER POD '{pod_name}' EXISTS. Check "
        "https://console.runpod.io/pods now. If it exists, terminate it with:\n"
        f"    runpodctl pod delete <POD_ID>   # the pod named {pod_name}\n"
        "or the console's Terminate button. !!!",
        err=True,
    )
    if volume is not None:
        emit(
            f"!!! network volume {volume.id} ('{volume.name}', {volume.size_gb} GB) was created "
            "for this launch and keeps billing storage. The kit never deletes volumes; delete "
            "it in the console if unwanted. !!!",
            err=True,
        )


class TerminateRequestedError(KeyboardInterrupt):
    """SIGTERM/SIGHUP.  A KeyboardInterrupt, so every launch/watchdog path that terminates
    on Ctrl-C also terminates on these; ``watch`` tells it apart from a plain Ctrl-C."""


_EXIT_SIGNALLED = {"seen": False}


def _exit_signal_handler(signum: int, _frame: object) -> None:
    # Latch: only the FIRST SIGTERM/SIGHUP raises.  A second one (e.g. `uv run` forwarding
    # plus process-group delivery) would otherwise land between the first one's handler and
    # the termination path and skip it.  Termination itself also ignores signals.
    if _EXIT_SIGNALLED["seen"]:
        return
    _EXIT_SIGNALLED["seen"] = True
    raise TerminateRequestedError(f"signal {signum}")


def install_exit_signals() -> None:
    """Turn SIGTERM/SIGHUP (closed terminal, kill) into TerminateRequestedError so the
    watchdog's termination path runs.  Only called from the CLI entry points."""

    def handler(signum: int, frame: object) -> None:
        _exit_signal_handler(signum, frame)

    _EXIT_SIGNALLED["seen"] = False
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


def main(
    argv: list[str] | None = None,
    *,
    watch_deps: WatchDeps | None = None,
    factory: TransportFactory = _default_factory,
    env: dict[str, str] | None = None,
    key_path: Path | None = None,
    clock: Clock = time.time,
    sleep: Sleeper = time.sleep,
    now: dt.datetime | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    VERIFIED_GONE.clear()  # per invocation (tests call main() repeatedly in one process)
    LAST_STATUS.clear()
    try:
        key = load_api_key(env, key_path)
        if args.cmd == "gpus":
            return cmd_gpus(ReadOnlyTransport(factory(key)), args)
        if args.cmd == "launch":
            return cmd_launch(
                key, factory, args, clock=clock, sleep=sleep, now=now or dt.datetime.now(dt.UTC)
            )
        if args.cmd == "status":
            pod = get_pod(ReadOnlyTransport(factory(key)), args.pod_id)
            emit(json.dumps(None if pod is None else dataclasses.asdict(pod), indent=2))
            return 0 if pod else 1
        if args.cmd == "logs":
            return cmd_logs(ReadOnlyTransport(factory(key)), args, clock=clock, sleep=sleep)
        if args.cmd == "spend":
            return cmd_spend(ReadOnlyTransport(factory(key)), args.pod_id)
        if args.cmd == "watch":
            return cmd_watch(ReadOnlyTransport(factory(key)), args, watch_deps or WatchDeps())
        if args.cmd == "watchdog":
            deadline = clock() + args.max_minutes * 60
            wt = factory(key)
            try:
                run_watchdog(
                    wt, args.pod_id, deadline, clock=clock, sleep=sleep,
                    poll_seconds=args.poll_seconds,
                )  # fmt: skip
            finally:
                if args.pod_id not in VERIFIED_GONE:
                    terminate_guarded(wt, args.pod_id, sleep=sleep)
            return 0 if args.pod_id in VERIFIED_GONE else 3
        if args.cmd == "terminate":
            return cmd_terminate(factory, key, args, sleep=sleep)
        if args.cmd == "tunnel":
            check_local_port(args.local_port)
            pod = get_pod(ReadOnlyTransport(factory(key)), args.pod_id)
            if pod is None:
                raise KitError(f"pod {args.pod_id} not found")
            endpoint = pod.ssh_endpoint()
            if endpoint is None:
                raise KitError(no_ssh_endpoint_message(pod))
            hosts_file = default_known_hosts(args.pod_id)
            cmd = build_ssh_command(
                endpoint[0],
                endpoint[1],
                args.local_port,
                args.remote_port,
                args.identity,
                known_hosts=hosts_file,
            )
            emit("ssh command: " + shlex.join(cmd))
            emit(f"(host keys: trust-on-first-use in {hosts_file}, not ~/.ssh/known_hosts)")
            emit("\n" + memfun_snippet(
                args.local_port, args.served_name))  # fmt: skip
            emit("export MEMFUN_SGLANG_DUMMY_KEY=unused")
            emit(f"check: curl -s http://127.0.0.1:{args.local_port}/v1/models")
            if not args.run:
                emit("(not started; add --run to open the tunnel)")
                return 0
            try:
                return subprocess.call(cmd, env=ssh_env(dict(os.environ)))
            finally:
                hosts_file.unlink(missing_ok=True)
    except KitError as exc:
        emit(f"error: {REDACT(str(exc))}", err=True)
        return 1
    except KeyboardInterrupt:
        emit("interrupted", err=True)
        return 130
    except Exception as exc:
        # Never let a raw traceback (which could quote a header) reach the terminal.
        emit(f"error: {type(exc).__name__}: {REDACT(str(exc))}", err=True)
        return 1
    return 2


if __name__ == "__main__":
    install_exit_signals()
    sys.exit(main())
