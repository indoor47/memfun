#!/usr/bin/env python3
"""Run the swarm smoke bundles through ``memfun swarm`` and grade them with hidden tests.

Each episode (one bundle x one trial) gets a fresh throwaway git repo built from the
bundle's ``repo/`` with its own ``memfun.toml`` (one ``[llm]`` block for the served model),
runs ``memfun swarm`` in its own process group, then grades the units' owned files with
the isolated grader from ``validate_bundles.py`` (owned files over a clean ``repo/``,
hidden tests from a separate grader dir, one pytest process per unit).  All episodes'
events are forwarded to ONE dashboard.  See ``README-run-smoke.md`` next to this file for
commands, the ``results.json`` schema, the classification rules and the threat model
(grading is NOT a sandbox).

    uv run python evals/swarm/run_smoke.py --base-url http://127.0.0.1:18000/v1 \\
        --bundles textkit-3 formats-2 --parallel-tasks 2 --dry-plan
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import errno
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:  # also runnable as a plain script
    sys.path.insert(0, str(_REPO_ROOT))

from evals.swarm import validate_bundles as vb  # noqa: E402

SCHEMA_VERSION = 1
SMOKE_ROOT = HERE / "smoke"
#: Outside the memfun checkout on purpose: never write real results into the repo, and
#: keep ``smoke/<bundle>/solution`` + ``hidden/`` off a short ``../`` path from episodes.
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "memfun-swarm-runs"
DEFAULT_BUNDLES = (
    "textkit-3",
    "kvstores-4",
    "minicli-3",
    "formats-2",
    "validators-4",
    "control-calc",
)
#: Ports the runner never talks to: 8434 is the owner's live model server.
FORBIDDEN_PORTS = frozenset({8434})
#: NAME of the dummy env var the episode's ``[llm] api_key_env`` points at (same as the
#: RunPod kit).  The value is a dummy: SGLang / MTPLX need no key.
DUMMY_KEY_ENV = "MEMFUN_SGLANG_DUMMY_KEY"
DUMMY_KEY_VALUE = "dummy-not-a-secret"
#: memfun_cli.commands.chat.NO_DASHBOARD_AUTOSTART_ENV: episodes never start a dashboard
#: (it would run under the episode's throwaway HOME, without the owner's token, and could
#: take the port the shared dashboard needs).  Episodes only reuse the one named by their
#: ``.memfun/dashboard.lock``; the runner owns the single real dashboard.
NO_DASHBOARD_ENV = "MEMFUN_NO_DASHBOARD_AUTOSTART"
#: memfun_runtime.distributed.EVENT_TOPIC (kept literal so the runner is stdlib-only).
EVENT_TOPIC = "memfun.distributed.events"
VOID_THRESHOLD = 0.05
OUTCOMES = ("pass", "partial", "model_failure", "harness_error")
#: Per-call output limit for every role: well inside the served model's context (262144
#: for Qwen3.8-27B) and large enough for a whole module or a big decomposition plan.
DEFAULT_MAX_TOKENS = 32768
SMOKE_EPISODE_TIMEOUT = 2400.0
COMMIT0_EPISODE_TIMEOUT = 5400.0  # 90 min: tasks should finish, not be cut off
COMMIT0_GRADER_TIMEOUT = 900.0
#: The worker's own truncation report and the stall stop, as logged by memfun.
TRUNCATED_LOG_RE = re.compile(r"LLM RESPONSE TRUNCATED")
STALL_LOG_RE = re.compile(r"RLM STOPPED at iteration \d+ \(stalled")
SPEC_SUFFIX = (
    "Implement this in the existing stub modules of this repository (create a module the "
    "spec names if it does not exist yet). Do not modify anything under tests/: hidden "
    "tests grade the result."
)
#: Variables copied from the runner's environment into child processes; nothing else is.
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR")
#: Extra variables the shared dashboard process may see (names only are ever printed).
DASHBOARD_ENV_PASS = (
    "MEMFUN_DASHBOARD_TOKEN",
    "MEMFUN_INFERENCE_METRICS_URL",
    "MEMFUN_POD_RATE_USD",
)
GITIGNORE = ".memfun/\n__pycache__/\n*.pyc\n.pytest_cache/\n.ruff_cache/\n"
#: Pins pytest's rootdir to the episode repo for the agents' own test runs.
EPISODE_PYTEST_INI = "[pytest]\n# written by run_smoke.py: rootdir = this repo\n"
PYTEST_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
GIT_IDENTITY = ("-c", "user.name=memfun-smoke", "-c", "user.email=smoke@memfun.invalid")
WARNING_RE = re.compile(
    r"MCP tool discovery|Failed to discover MCP tools|Task decomposition failed|Traceback"
)
#: Error classes of an unreachable / failing model server (structured llm.call errors).
SERVER_ERROR_RE = re.compile(
    r"APIConnectionError|Connection ?refused|ConnectError|ServiceUnavailable|"
    r"InternalServerError|BadGateway|APITimeoutError|NotFoundError: .*model|"
    r"HTTP(?: status)? 5\d\d|status[_ ]code[=: ]+5\d\d",
    re.IGNORECASE,
)
#: A LOG line only counts when it also names the client stack (not agent code text).
SERVER_LOG_CONTEXT_RE = re.compile(r"litellm|openai\.|httpx|httpcore")
#: Task ids the engine derives from a sub-task (pair re-dispatch, revision, review).
DERIVED_TASK_RE = re.compile(r"_(pair\d+|rev\d+|review_.*|pr_review)$")
#: The fail-closed review parser's rejection of an auditor answer without a valid
#: findings JSON block carrying the dispatch's review id.
REVIEWER_REJECT_RE = re.compile(r"Reviewer output had no valid findings JSON block")
RESULT_MARKER = "Swarm result"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LOG_TAIL_LINES = 40
MAX_OWNED_BYTES = 1_000_000
MAX_LOG_BYTES = 20_000_000
MAX_AGENTS = 100  # memfun swarm --max-pool range
MAX_PARALLEL = 64
MAX_TRIALS = 1000
GRADE_WORKERS = 4
LOOPBACK_NAMES = frozenset({"localhost"})
#: memfun_cli.commands.swarm.RESULT_JSON_ENV and its schema (the swarm's own result file).
SWARM_RESULT_ENV = "MEMFUN_SWARM_RESULT_JSON"
SWARM_RESULT_SCHEMA = 1
SWARM_RESULT_NAME = "swarm-result.json"
MAX_SWARM_RESULT_BYTES = 20_000_000
TOOL_TYPES = ("grep", "find_files", "read", "edit", "write", "cmd", "web_search", "web_fetch")
SEARCH_TOOLS = ("grep", "find_files")


class UsageError(Exception):
    """Bad command line (exit 2)."""


# ── Bundles ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Unit:
    id: str
    owns: tuple[str, ...]
    hidden_test: str

    @property
    def module(self) -> str:
        return vb.unit_module(self.hidden_test)


@dataclasses.dataclass(frozen=True)
class Bundle:
    id: str
    path: Path
    kind: str = "?"
    spec: str = ""
    units: tuple[Unit, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def owned(self) -> list[str]:
        return [p for u in self.units for p in u.owns]

    @property
    def has_integration(self) -> bool:
        return (self.path / "hidden" / f"{vb.INTEGRATION}.py").is_file()


def load_bundle(path: Path) -> Bundle:
    """Parse ``task.toml``; schema problems (missing files etc.) end up in ``problems``."""
    try:
        task: dict[str, Any] = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return Bundle(id=path.name, path=path, problems=(f"task.toml unreadable: {exc}",))
    problems = vb.schema_problems(path, task)
    units: list[Unit] = []
    raw_units = task.get("units")
    for raw in cast("list[Any]", raw_units if isinstance(raw_units, list) else []):
        if isinstance(raw, dict):
            unit = cast("dict[str, Any]", raw)
            owns = unit.get("owns")
            units.append(
                Unit(
                    id=str(unit.get("id")),
                    owns=tuple(str(p) for p in cast("list[Any]", owns or [])),
                    hidden_test=str(unit.get("hidden_test")),
                )
            )
    return Bundle(
        id=path.name,
        path=path,
        kind=str(task.get("kind", "?")),
        spec=str(task.get("spec", "")),
        units=tuple(units),
        problems=tuple(problems),
    )


def known_bundles(root: Path) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/task.toml"))


def episode_spec(bundle: Bundle) -> str:
    return f"{bundle.spec.strip()}\n\n{SPEC_SUFFIX}"


# ── URLs, ports, model discovery ───────────────────────────────────────────────


def url_port(url: str) -> int:
    parts = urllib.parse.urlsplit(url)
    if parts.port is not None:
        return parts.port
    return 443 if parts.scheme == "https" else 80


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_http_url(url: str, flag: str) -> str:
    """http(s) only, valid port, never a forbidden port.  Returns it without trailing /."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UsageError(f"{flag} must be an http(s) URL, got {url!r}")
    try:
        port = url_port(url)
    except ValueError as exc:
        raise UsageError(f"{flag} has an invalid port: {url!r}") from exc
    if port in FORBIDDEN_PORTS:
        raise UsageError(f"refusing {flag} on port {port}: that is the owner's live model server")
    return url.rstrip("/")


def check_base_url(url: str) -> str:
    return check_http_url(url, "--base-url")


def check_port(port: int, flag: str) -> int:
    if port in FORBIDDEN_PORTS:
        raise UsageError(f"refusing {flag} {port}: that is the owner's live model server")
    if not 1 <= port <= 65535:
        raise UsageError(f"{flag} out of range: {port}")
    return port


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to the same scheme, host and port."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if (old.scheme, old.hostname, old.port) != (new.scheme, new.hostname, new.port):
            raise urllib.error.HTTPError(
                req.full_url, code, f"refused cross-host redirect to {newurl}", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SameHostRedirects())

HttpGet = Callable[[str, float], Any]


def http_get_json(url: str, timeout: float = 10.0) -> Any:
    """GET ``url`` (no proxies, no auth headers, same-host redirects only), decode JSON."""
    with _OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read(5_000_000).decode("utf-8", "replace"))


def served_models(base_url: str, get: HttpGet = http_get_json) -> list[str]:
    body = get(f"{base_url}/models", 10.0)
    data = cast("dict[str, Any]", body).get("data") if isinstance(body, dict) else None
    models: list[str] = []
    for item in cast("list[Any]", data if isinstance(data, list) else []):
        if isinstance(item, dict) and cast("dict[str, Any]", item).get("id"):
            models.append(str(cast("dict[str, Any]", item)["id"]))
    return models


def preflight(base_url: str, model: str | None, get: HttpGet = http_get_json) -> tuple[str, str]:
    """Return ``(model, error)``; ``error`` is ``""`` when the server is usable."""
    try:
        models = served_models(base_url, get)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return model or "", f"model server unreachable: GET {base_url}/models: {exc}"
    if model:
        if models and model not in models:
            return model, f"model {model!r} is not served (served: {models})"
        return model, ""
    if len(models) != 1:
        return "", f"--model not given and {base_url}/models lists {len(models)} models: {models}"
    return models[0], ""


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def prom_counters(text: str, names: Iterable[str]) -> dict[str, float]:
    """Sum Prometheus samples by metric name (labels ignored)."""
    wanted = set(names)
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[{\s]", line, maxsplit=1)[0]
        if name in wanted:
            with contextlib.suppress(ValueError, IndexError):
                out[name] = out.get(name, 0.0) + float(line.rsplit(None, 1)[1])
    return out


PROM_PROMPT = ("sglang:prompt_tokens_total", "vllm:prompt_tokens_total")
PROM_GEN = ("sglang:generation_tokens_total", "vllm:generation_tokens_total")


def server_usage_snapshot(metrics_url: str) -> dict[str, float] | None:
    try:
        with _OPENER.open(metrics_url, timeout=5.0) as resp:
            text = resp.read(20_000_000).decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return None
    c = prom_counters(text, (*PROM_PROMPT, *PROM_GEN))
    return {
        "prompt": sum(c.get(n, 0.0) for n in PROM_PROMPT),
        "completion": sum(c.get(n, 0.0) for n in PROM_GEN),
    }


# ── Private file I/O (episode dirs are reachable by unconfined agent code) ─────────


def write_private(path: Path, data: str | bytes) -> None:
    """Atomically write ``path``: exclusive no-follow temp file, then rename over it.

    Renaming replaces whatever entry is at ``path`` (a planted symlink included) and
    never writes through it.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    tmp = path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json_atomic(path: Path, obj: Any) -> None:
    write_private(path, json.dumps(obj, indent=2, default=str))


@dataclasses.dataclass(frozen=True)
class Probe:
    """A file read with no symlink following.  ``data`` None: absent or refused."""

    data: bytes | None = None
    mtime: float = 0.0
    problem: str = ""  # symlink | symlinked parent | not a regular file | too large | ...


def safe_read(root: Path, rel: str, limit: int = MAX_OWNED_BYTES) -> Probe:
    """Read ``root/rel`` only if every component is a real dir and the leaf a regular file."""
    pure = PurePosixPath(rel)
    if pure.is_absolute() or not pure.parts or any(p in ("", ".", "..") for p in pure.parts):
        return Probe(problem=f"bad owned path {rel!r}")
    try:
        st = os.lstat(root)
    except FileNotFoundError:
        return Probe()
    if stat.S_ISLNK(st.st_mode):
        return Probe(problem="symlinked root")
    cur = root
    for i, comp in enumerate(pure.parts[:-1]):
        cur = cur / comp
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return Probe()
        if stat.S_ISLNK(st.st_mode):
            return Probe(problem=f"symlinked parent {'/'.join(pure.parts[: i + 1])}")
        if not stat.S_ISDIR(st.st_mode):
            return Probe()
    try:
        fd = os.open(cur / pure.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return Probe()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return Probe(problem="symlink")
        return Probe(problem=f"unreadable ({exc.strerror})")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return Probe(problem="not a regular file")
        if st.st_size > limit:
            return Probe(problem=f"too large ({st.st_size} bytes > {limit})")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                return Probe(problem=f"too large (> {limit} bytes)")
        return Probe(data=b"".join(chunks), mtime=st.st_mtime)
    finally:
        os.close(fd)


def read_log(path: Path) -> str:
    """The tail of a log the runner created, read without following a planted symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return ""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return ""
        if st.st_size > MAX_LOG_BYTES:
            os.lseek(fd, st.st_size - MAX_LOG_BYTES, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", "replace")
    finally:
        os.close(fd)


def _tail_text(text: str, lines: int = LOG_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def archive_dir(src: Path, ep_dir: Path, name: str) -> str:
    """Copy a runner-private dir into ``ep_dir/name``; refuse if the name is taken."""
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=ep_dir))
    try:
        shutil.copytree(src, staging, symlinks=True, dirs_exist_ok=True)
        os.rename(staging, ep_dir / name)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return f"artifact {name}/ not archived ({exc.strerror or exc}): planted by the agent?"
    return ""


# ── Child environment and process groups ─────────────────────────────────────────


def child_env(
    home: Path, base: Mapping[str, str] | None = None, result_json: Path | None = None
) -> dict[str, str]:
    """Clean allowlist env for ``memfun swarm``: no real credentials, a throwaway HOME.

    HOME points at the episode's own directory so ``~/.memfun/credentials.json`` and
    ``~/.memfun/config.toml`` (real keys, per-role cloud models) never reach the swarm.
    """
    src = os.environ if base is None else base
    env = {k: src[k] for k in ENV_ALLOW if k in src}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["HOME"] = str(home)
    env[DUMMY_KEY_ENV] = DUMMY_KEY_VALUE
    env["PYTHONUNBUFFERED"] = "1"
    env["COLUMNS"] = "160"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env[NO_DASHBOARD_ENV] = "1"  # the token is never passed to episodes (not in ENV_ALLOW)
    if result_json is not None:
        env[SWARM_RESULT_ENV] = str(result_json)
    return env


def dashboard_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Env for the shared dashboard: allowlist + real HOME (it owns the token file)."""
    src = os.environ if base is None else base
    env = {k: src[k] for k in (*ENV_ALLOW, "HOME", *DASHBOARD_ENV_PASS) if k in src}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_groups(
    pgids: Iterable[int], grace: float = 5.0, reap: Callable[[], object] | None = None
) -> None:
    """SIGTERM every group in ``pgids`` (and only those), one shared grace, then SIGKILL.

    ``reap`` (e.g. ``proc.poll``) lets the caller reap its own leader while waiting, so
    a zombie leader does not keep its group looking alive for the whole grace period.
    """
    groups = list(pgids)
    for pgid in groups:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while any(_group_alive(g) for g in groups) and time.monotonic() < deadline:
        if reap is not None:
            reap()
        time.sleep(0.05)
    for pgid in groups:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def kill_group(pgid: int, grace: float = 5.0, reap: Callable[[], object] | None = None) -> None:
    kill_groups([pgid], grace, reap)


@dataclasses.dataclass
class ProcOutcome:
    returncode: int | None
    timed_out: bool
    pid: int


def run_in_group(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: float,
    grace: float = 5.0,
    on_start: Callable[[int], object] | None = None,
    on_end: Callable[[int], object] | None = None,
) -> ProcOutcome:
    """Run ``cmd`` in a NEW process group; at ``timeout`` kill only that group.

    Whatever the outcome, leftover members of the group (children the command left
    behind) are terminated too, and the leader is reaped.
    """
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if on_start is not None:
            on_start(proc.pid)
        timed_out = False
        try:
            rc: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = None
        finally:
            kill_group(proc.pid, grace, proc.poll)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=grace + 5)
            if on_end is not None:
                on_end(proc.pid)
    return ProcOutcome(returncode=rc, timed_out=timed_out, pid=proc.pid)


# ── Events: read from the episode's event-bus DB, forward to the dashboard ─────────


def read_events(db: Path, after_rowid: int = 0) -> list[tuple[int, dict[str, Any]]]:
    """Dashboard events from a memfun T1 (SQLite) event bus DB, oldest first.

    Strictly read-only (``mode=ro``): any error (no table yet, locked, a WAL db without
    its ``-shm``) returns ``[]`` and the caller simply polls again later.
    """
    try:
        st = os.lstat(db)
    except OSError:
        return []
    if not stat.S_ISREG(st.st_mode):
        return []
    try:
        con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT rowid, payload FROM events WHERE topic = ? AND rowid > ? ORDER BY rowid",
            (EVENT_TOPIC, after_rowid),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: list[tuple[int, dict[str, Any]]] = []
    for rowid, payload in rows:
        try:
            raw = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
            event = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(event, dict):
            out.append((int(rowid), cast("dict[str, Any]", event)))
    return out


def post_event(port: int, event: dict[str, Any], timeout: float = 3.0) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/event",
        data=json.dumps(event).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        resp.read()


class EventForwarder(threading.Thread):
    """Tail one episode's event-bus DB and POST each new event to the shared dashboard."""

    def __init__(self, db: Path, port: int, interval: float = 0.5) -> None:
        super().__init__(daemon=True, name=f"fwd-{db.parent.parent.name}")
        self.db = db
        self.port = port
        self.interval = interval
        self.last_rowid = 0
        self.forwarded = 0
        self.errors = 0
        self.stuck = False
        self._stop_evt = threading.Event()

    def pump(self) -> None:
        for rowid, event in read_events(self.db, self.last_rowid):
            self.last_rowid = rowid
            try:
                post_event(self.port, event)
                self.forwarded += 1
            except (OSError, urllib.error.URLError):
                self.errors += 1

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval):
            self.pump()

    def finish(self, join_timeout: float = 30.0) -> None:
        """Stop, then do the final drain ONLY if the thread is gone (no double sends)."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=join_timeout)
        if self.is_alive():
            self.stuck = True
            return
        self.pump()


def dashboard_health(port: int, get: HttpGet = http_get_json) -> bool:
    try:
        body = get(f"http://127.0.0.1:{port}/api/health", 2.0)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(body, dict) and cast("dict[str, Any]", body).get("status") == "ok"


@dataclasses.dataclass
class SharedDashboard:
    """The ONE dashboard all episodes report to (started by the runner, or reused)."""

    port: int
    lock_pid: int
    owned: bool
    proc: subprocess.Popen[bytes] | None = None
    error: str = ""

    def lock_payload(self) -> dict[str, int]:
        return {"port": self.port, "pid": self.lock_pid}

    def stop(self) -> None:
        """Stop the dashboard iff the runner started it; a reused one is never touched."""
        if self.owned and self.proc is not None:
            kill_group(self.proc.pid, reap=self.proc.poll)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=10)


_TERM_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})


@contextlib.contextmanager
def _term_signals_deferred() -> Generator[bool]:
    """Block SIGINT/SIGTERM in the main thread for the block (pending ones fire after)."""
    if threading.current_thread() is not threading.main_thread():
        yield False
        return
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _TERM_SIGNALS)
    try:
        yield True
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _unblock_term_signals() -> None:
    """Child side of :func:`_term_signals_deferred`: the dashboard must stay killable."""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _TERM_SIGNALS)


def record_pgid(run_dir: Path | None, pgid: int, label: str) -> None:
    """Append ``<pgid> <label>`` to ``<run_dir>/pgids.txt`` (operator cleanup after SIGKILL)."""
    if run_dir is None:
        return
    with contextlib.suppress(OSError):
        fd = os.open(
            run_dir / "pgids.txt", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o644
        )
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(f"{pgid} {label}\n")


def start_dashboard(
    memfun_cmd: list[str],
    port: int,
    workdir: Path,
    *,
    wait_s: float = 60.0,
    get: HttpGet = http_get_json,
    register: Callable[[SharedDashboard], object] | None = None,
) -> SharedDashboard:
    """Reuse a healthy memfun dashboard on ``port`` or start ``memfun dashboard --remote``.

    ``register`` receives a runner-started dashboard right after it is spawned, so an
    interrupt during the health wait can still stop it.
    """
    if dashboard_health(port, get):
        return SharedDashboard(port=port, lock_pid=os.getpid(), owned=False)
    if port_in_use(port):
        return SharedDashboard(
            port=port,
            lock_pid=0,
            owned=False,
            error=f"dashboard port {port} is in use by something that is not a memfun dashboard",
        )
    (workdir / ".memfun").mkdir(parents=True, exist_ok=True)
    log = (workdir / "dashboard.log").open("wb")
    dash: SharedDashboard | None = None
    try:
        # A Ctrl-C / SIGTERM landing between Popen() returning and register() would
        # orphan the dashboard: defer those signals across spawn + register.
        with _term_signals_deferred() as deferred:
            try:
                proc = subprocess.Popen(
                    [*memfun_cmd, "dashboard", "--remote", "--port", str(port)],
                    cwd=workdir,
                    env=dashboard_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=_unblock_term_signals if deferred else None,
                )
            except OSError as exc:
                return SharedDashboard(port, 0, False, error=f"dashboard failed to start: {exc}")
            dash = SharedDashboard(port=port, lock_pid=proc.pid, owned=True, proc=proc)
            if register is not None:
                register(dash)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if dashboard_health(port, get):
                return dash
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    except BaseException:
        if dash is not None:
            dash.stop()
        raise
    finally:
        log.close()
    dash.stop()
    tail = _tail_text(read_log(workdir / "dashboard.log"), 10)
    dash.error = f"dashboard failed to start on port {port}:\n{tail}"
    return dash


# ── Episode repo ─────────────────────────────────────────────────────────────────


def memfun_toml(base_url: str, model: str, max_tokens: int) -> str:
    return (
        "# Generated by evals/swarm/run_smoke.py: every role inherits this [llm] block.\n"
        "[llm]\n"
        'provider = "openai"\n'
        f"base_url = {json.dumps(base_url)}\n"
        f"model = {json.dumps(model)}\n"
        f'api_key_env = "{DUMMY_KEY_ENV}"  # NAME of a dummy env var; the server needs no key\n'
        f"max_tokens = {int(max_tokens)}\n"
        "temperature = 0.0\n"
        "\n"
        "[backend]\n"
        'tier = "sqlite"\n'
        'sqlite_path = ".memfun/memfun.db"\n'
    )


def _git(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(
        ["git", *GIT_IDENTITY, "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )


def create_episode_repo(
    bundle: Bundle, repo: Path, home: Path, base_url: str, model: str, max_tokens: int
) -> None:
    home.mkdir(parents=True, exist_ok=False)
    (home / ".gitconfig").write_text(
        "[user]\n\tname = memfun-smoke\n\temail = smoke@memfun.invalid\n"
        "[commit]\n\tgpgsign = false\n",
        encoding="utf-8",
    )
    shutil.copytree(bundle.path / "repo", repo)
    (repo / "memfun.toml").write_text(memfun_toml(base_url, model, max_tokens), "utf-8")
    (repo / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if not any((repo / name).exists() for name in PYTEST_CONFIG_FILES):
        (repo / "pytest.ini").write_text(EPISODE_PYTEST_INI, encoding="utf-8")
    env = child_env(home)
    _git(["init", "-q", "-b", "main"], repo, env)
    _git(["add", "-A"], repo, env)
    _git(["commit", "-q", "-m", f"smoke start tree: {bundle.id}"], repo, env)
    (repo / ".memfun").mkdir(exist_ok=True)


def swarm_cmd(memfun_cmd: list[str], spec: str, max_agents: int, task_timeout: int) -> list[str]:
    return [
        *memfun_cmd,
        "swarm",
        spec,
        "--max-pool",
        str(max_agents),
        "--timeout",
        str(task_timeout),
        "--no-push",
    ]


# ── Owned-file collection (symlink-safe) ─────────────────────────────────────────

_SKIP_PARTS = {".git", ".memfun", "__pycache__", ".pytest_cache", ".ruff_cache"}
_SKIP_ROOT_FILES = {"memfun.toml", ".gitignore", "pytest.ini"}


def _start_bytes(start: Path, rel: str) -> bytes | None:
    path = start / rel
    return path.read_bytes() if path.is_file() else None  # the bundle: trusted


def find_worktrees(repo: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """``(label, path)`` of real ``.memfun/worktrees/<wf>/<task>`` dirs, plus tampering.

    Symlinked entries (at any of these levels) are skipped and reported.
    """
    tamper: list[str] = []
    base = repo
    for comp in (".memfun", "worktrees"):
        base = base / comp
        try:
            st = os.lstat(base)
        except FileNotFoundError:
            return [], tamper
        if stat.S_ISLNK(st.st_mode):
            tamper.append(f"{base.relative_to(repo)} is a symlink")
            return [], tamper
        if not stat.S_ISDIR(st.st_mode):
            return [], tamper
    out: list[tuple[str, Path]] = []
    for wf in sorted(os.scandir(base), key=lambda e: e.name):
        if wf.is_symlink():
            tamper.append(f"worktree dir {wf.name} is a symlink")
            continue
        if not wf.is_dir(follow_symlinks=False):
            continue
        for task in sorted(os.scandir(wf.path), key=lambda e: e.name):
            if task.is_symlink():
                tamper.append(f"worktree {wf.name}/{task.name} is a symlink")
                continue
            if task.is_dir(follow_symlinks=False):
                out.append((f"{wf.name}/{task.name}", Path(task.path)))
    return out, tamper


def _tree_entries(root: Path) -> Iterable[tuple[str, bool]]:
    """``(rel, is_symlink)`` for files and symlinks under ``root`` (never follows links)."""
    for dirpath, dirnames, filenames in os.walk(root):
        keep: list[str] = []
        for d in sorted(dirnames):
            if d in _SKIP_PARTS:
                continue
            if os.path.islink(os.path.join(dirpath, d)):
                yield (Path(dirpath) / d).relative_to(root).as_posix(), True
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if name not in _SKIP_PARTS:
                full = Path(dirpath) / name
                yield full.relative_to(root).as_posix(), os.path.islink(full)


def collect_owned(bundle: Bundle, repo: Path, dest: Path) -> dict[str, Any]:
    """Pick each owned file's graded copy from the worktrees (project root as fallback).

    Only regular files read with no symlink following count.  For each owned path, the
    copies that differ from the bundle's start tree are candidates.  Worktree copies win;
    when several worktrees changed it the newest by mtime is taken and ``duplicate`` is
    set.  The project-root copy is used only when no worktree changed the path.  A
    symlink / symlinked parent / non-regular / oversized owned path anywhere is
    tampering: that unit contributes NO files and fails.  Chosen bytes go to ``dest``.
    """
    start = bundle.path / "repo"
    worktrees, tampering = find_worktrees(repo)
    files: dict[str, dict[str, Any]] = {}
    chosen_bytes: dict[str, bytes] = {}
    bad_paths: set[str] = set()
    for rel in bundle.owned:
        base = _start_bytes(start, rel)
        changed: list[tuple[str, Probe]] = []
        for label, wt in worktrees:
            probe = safe_read(wt, rel)
            if probe.problem:
                tampering.append(f"{label}:{rel}: {probe.problem}")
                bad_paths.add(rel)
            elif probe.data is not None and probe.data != base:
                changed.append((label, probe))
        root_probe = safe_read(repo, rel)
        if root_probe.problem:
            tampering.append(f"root:{rel}: {root_probe.problem}")
            bad_paths.add(rel)
        chosen: tuple[str, Probe] | None = None
        if changed:
            chosen = max(changed, key=lambda c: c[1].mtime)
        elif root_probe.data is not None and root_probe.data != base:
            chosen = ("root", root_probe)
        if chosen is not None and chosen[1].data is not None:
            chosen_bytes[rel] = chosen[1].data
        files[rel] = {
            "source": chosen[0] if chosen else None,
            "changed_in": [label for label, _ in changed],
            "duplicate": len(changed) > 1,
            "tampered": rel in bad_paths,
        }
    units: dict[str, dict[str, Any]] = {}
    for unit in bundle.units:
        infos = [files[p] for p in unit.owns]
        tampered = any(i["tampered"] for i in infos)
        tasks = sorted({lbl.split("/", 1)[1] for i in infos for lbl in i["changed_in"]})
        if not tampered:
            for rel in unit.owns:
                if rel in chosen_bytes:
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_private(target, chosen_bytes[rel])
        units[unit.id] = {
            "output": not tampered and any(i["source"] for i in infos),
            "owned_changed": sum(1 for i in infos if i["source"]),
            "sub_tasks": tasks,
            "duplicate_work": any(i["duplicate"] for i in infos),
            "symlink_tampering": tampered,
        }
    owned = set(bundle.owned)
    stray: list[str] = []
    for label, root in [*worktrees, ("root", repo)]:
        for rel, is_link in _tree_entries(root):
            if rel in owned or rel in _SKIP_ROOT_FILES:
                continue
            if is_link:
                stray.append(f"{label}:{rel} (symlink)")
                continue
            probe = safe_read(root, rel)
            if probe.problem or probe.data != _start_bytes(start, rel):
                stray.append(f"{label}:{rel}")
    return {
        "worktrees": [label for label, _ in worktrees],
        "files": files,
        "units": units,
        "tampering": tampering[:50],
        "stray_changes": stray[:50],
        "tests_modified": any(s.split(":", 1)[1].startswith("tests/") for s in stray if ":" in s),
    }


# ── Metrics from events ──────────────────────────────────────────────────────────


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _rel(ts: float | None, origin: float | None) -> float | None:
    return round(ts - origin, 3) if ts is not None and origin is not None else None


def peak_overlap(intervals: Iterable[tuple[float, float]]) -> int:
    points: list[tuple[float, int]] = []
    for start, end in intervals:
        points.append((start, 1))
        points.append((max(start, end), -1))
    points.sort(key=lambda p: (p[0], p[1]))  # ends before starts at the same instant
    peak = cur = 0
    for _, delta in points:
        cur += delta
        peak = max(peak, cur)
    return peak


def event_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokens, agents, concurrency and timings from one episode's dashboard events."""
    llm = [e for e in events if e.get("event_type") == "llm.call"]
    tokens: dict[str, Any] = {"prompt": 0, "completion": 0, "calls": len(llm)}
    with_usage = 0
    by_role: dict[str, dict[str, int]] = {}
    llm_errors: list[str] = []
    for e in llm:
        p, c = _num(e.get("prompt_tokens")), _num(e.get("completion_tokens"))
        if p is not None or c is not None:
            with_usage += 1
        tokens["prompt"] += int(p or 0)
        tokens["completion"] += int(c or 0)
        slot = by_role.setdefault(str(e.get("role") or "unknown"), {"prompt": 0, "completion": 0})
        slot["prompt"] += int(p or 0)
        slot["completion"] += int(c or 0)
        if e.get("error"):
            llm_errors.append(str(e["error"])[:300])
    tokens["total"] = tokens["prompt"] + tokens["completion"]
    tokens["calls_with_usage"] = with_usage
    tokens["by_role"] = by_role
    tokens["source"] = (
        "llm.call events" if with_usage else ("llm.call events (no usage)" if llm else "none")
    )

    tasks: dict[str, dict[str, Any]] = {}
    for e in events:
        etype = str(e.get("event_type") or "")
        tid = e.get("task_id")
        ts = _num(e.get("ts"))
        if (
            not tid
            or ts is None
            or etype not in ("task.published", "task.picked_up", "task.completed", "task.failed")
        ):
            continue
        slot = tasks.setdefault(str(tid), {"task_id": str(tid), "agent_name": e.get("agent_name")})
        key = {"task.published": "published", "task.picked_up": "start"}.get(etype, "end")
        if key == "end":
            slot["success"] = bool(e.get("success")) if etype == "task.completed" else False
        slot.setdefault(key, ts)
        slot["derived"] = bool(DERIVED_TASK_RE.search(str(tid)))

    all_ts = [t for e in events if (t := _num(e.get("ts"))) is not None]
    last_ts = max(all_ts) if all_ts else 0.0
    intervals = [(t["start"], t.get("end", last_ts)) for t in tasks.values() if "start" in t]
    llm_intervals: list[tuple[float, float]] = []
    for e in llm:
        ts = _num(e.get("ts"))
        if ts is not None:
            llm_intervals.append((ts - (_num(e.get("duration_ms")) or 0.0) / 1000.0, ts))

    started = [t for e in events if e.get("event_type") == "workflow.started"
               and (t := _num(e.get("ts"))) is not None]  # fmt: skip
    published = [t["published"] for t in tasks.values() if "published" in t]
    starts = [t["start"] for t in tasks.values() if "start" in t]
    ends = [t["end"] for t in tasks.values() if "end" in t]
    agent_ids = {t for t in tasks if "start" in tasks[t]} | {
        str(e["task_id"]) for e in llm if e.get("task_id")
    }
    names = {str(e["agent_name"]) for e in events if e.get("agent_name")} - {"workflow-engine"}
    wf0 = min(started) if started else None
    return {
        "tokens": tokens,
        "llm_errors": llm_errors[:20],
        "llm_error_count": len(llm_errors),
        "llm_server_error_count": sum(1 for err in llm_errors if SERVER_ERROR_RE.search(err)),
        "agents_used": len(agent_ids),
        "agent_names": sorted(names),
        "peak_concurrency": peak_overlap(intervals),
        "peak_llm_concurrency": peak_overlap(llm_intervals),
        "decompose_s": round(min(published) - wf0, 3) if wf0 is not None and published else None,
        "exec_wall_s": round(max(ends) - min(starts), 3) if starts and ends else None,
        "primary_sub_tasks": sum(1 for t in tasks.values() if not t["derived"]),
        "workflow_ids": sorted({str(e["workflow_id"]) for e in events if e.get("workflow_id")}),
        "sub_tasks": [
            {**t, "start_rel_s": _rel(t.get("start"), wf0), "end_rel_s": _rel(t.get("end"), wf0)}
            for t in sorted(tasks.values(), key=lambda t: (t.get("start", 0.0), t["task_id"]))
        ],
        "event_count": len(events),
    }


# ── The swarm's own result file (MEMFUN_SWARM_RESULT_JSON) ─────────────────────


def _op_counts(ops: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in cast("list[Any]", ops if isinstance(ops, list) else []):
        typ = cast("dict[str, Any]", op).get("type") if isinstance(op, dict) else None
        if isinstance(typ, str) and typ:
            counts[typ] = counts.get(typ, 0) + 1
    return counts


def load_swarm_result(ep_dir: Path, workflow_ids: list[str]) -> tuple[dict[str, Any] | None, str]:
    """Read and validate the swarm's result file.  ``(doc, "")`` or ``(None, problem)``.

    Read with no symlink following; it must be a JSON object with the expected schema
    version, a ``sub_tasks`` object, and a workflow id consistent with the episode's
    event bus (when both are known).  Anything else is "unavailable" and the runner
    falls back to the log and events.  NOTE: unconfined agent code could forge it, like
    the event DB; it is telemetry, never used for grading.
    """
    probe = safe_read(ep_dir, SWARM_RESULT_NAME, MAX_SWARM_RESULT_BYTES)
    if probe.problem:
        return None, f"{SWARM_RESULT_NAME}: {probe.problem}"
    if probe.data is None:
        return None, f"{SWARM_RESULT_NAME} missing"
    try:
        doc = json.loads(probe.data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"{SWARM_RESULT_NAME} is not JSON: {exc}"
    if not isinstance(doc, dict):
        return None, f"{SWARM_RESULT_NAME} is not a JSON object"
    d = cast("dict[str, Any]", doc)
    if d.get("schema_version") != SWARM_RESULT_SCHEMA:
        return None, f"unsupported schema_version {d.get('schema_version')!r}"
    if not isinstance(d.get("sub_tasks"), dict):
        return None, "sub_tasks missing"
    wf = d.get("workflow_id")
    if wf and workflow_ids and wf not in workflow_ids:
        return None, f"workflow_id {wf!r} not in the episode's events {workflow_ids}"
    return d, ""


def tool_usage(doc: dict[str, Any] | None, problem: str = "") -> dict[str, Any]:
    """Per-episode and per-sub-task tool counts from the swarm's result file."""
    if doc is None:
        return {"source": "unavailable", "problem": problem}
    subs = cast("dict[str, Any]", doc["sub_tasks"])
    per_task: dict[str, Any] = {}
    for tid, raw in sorted(subs.items()):
        st = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
        counts = _op_counts(st.get("ops"))
        per_task[tid] = {
            "agent": st.get("agent"),
            "status": st.get("status"),
            "success": st.get("success"),
            "iterations": st.get("iterations"),
            "counts": counts,
            "truncated": bool(st.get("truncated")),
            "used_search_tools": any(counts.get(t, 0) for t in SEARCH_TOOLS),
        }
    totals: dict[str, int] = {}
    raw_totals = doc.get("ops_total")
    if isinstance(raw_totals, dict):
        for k, v in cast("dict[str, Any]", raw_totals).items():
            if isinstance(v, int) and not isinstance(v, bool):
                totals[str(k)] = v
    else:
        for t in per_task.values():
            for k, v in t["counts"].items():
                totals[k] = totals.get(k, 0) + v
    decomposition = doc.get("decomposition")
    n_sub = (
        cast("dict[str, Any]", decomposition).get("n_sub_tasks")
        if isinstance(decomposition, dict)
        else None
    )
    agents = {t["agent"] for t in per_task.values() if t["agent"]}
    unaccepted = doc.get("unaccepted_tasks")
    return {
        "source": SWARM_RESULT_NAME,
        "workflow_id": doc.get("workflow_id"),
        "success": doc.get("success"),
        "n_sub_tasks": n_sub if isinstance(n_sub, int) else len(per_task),
        # Distinct agent instance ids are not in the file: one instance per sub-task.
        "agents_used": len(per_task),
        "agent_types": sorted(str(a) for a in agents),
        "unaccepted_tasks": list(cast("list[Any]", unaccepted))
        if isinstance(unaccepted, list)
        else [],
        "counts": totals,
        "used_search_tools": any(totals.get(t, 0) for t in SEARCH_TOOLS),
        "sub_tasks_used_search": sum(1 for t in per_task.values() if t["used_search_tools"]),
        "sub_tasks": per_task,
    }


# ── Logs ─────────────────────────────────────────────────────────────────────────


def log_signals(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    warnings = [ln.strip()[:300] for ln in lines if WARNING_RE.search(ln)]
    server = [
        ln.strip()[:300]
        for ln in lines
        if SERVER_ERROR_RE.search(ln) and SERVER_LOG_CONTEXT_RE.search(ln)
    ]
    return {
        "reviewer_json_rejections": len(REVIEWER_REJECT_RE.findall(text)),
        "truncated_responses": len(TRUNCATED_LOG_RE.findall(text)),
        "stalled_stops": len(STALL_LOG_RE.findall(text)),
        "warnings": warnings[:50],
        "server_error_lines": server[:20],
        "server_error_line_count": len(server),
        "produced_result": RESULT_MARKER in text,
    }


# ── Classification ───────────────────────────────────────────────────────────────


def classify(
    *,
    fatal: str = "",
    timed_out: bool = False,
    timeout_s: float = 0.0,
    returncode: int | None = 0,
    produced_result: bool = True,
    grade_error: str = "",
    hidden_passed: int = 0,
    all_pass: bool = False,
    any_output: bool = True,
    server_errors: int = 0,
) -> tuple[str, str]:
    """One of OUTCOMES plus a reason.  Same rules for passing and failing episodes.

    A harness fault is blamed only when it explains the outcome: the runner could not
    set up or grade the episode, or the episode produced NO owned output while the swarm
    timed out, crashed without a result, or saw model-server errors.  Whenever owned
    output exists it is graded on its merits (timeouts / crashes / server errors become
    flags, never a pass-rate exclusion).
    """
    if fatal:
        return "harness_error", fatal
    if grade_error:
        return "harness_error", grade_error
    if not any_output:
        if timed_out:
            return "harness_error", f"episode timeout after {timeout_s:g}s with no output"
        if returncode not in (0, None) and not produced_result:
            return "harness_error", f"memfun swarm exited {returncode} without a result"
        if server_errors:
            return "harness_error", f"model server errors ({server_errors}) and no output"
    if all_pass:
        return "pass", "all hidden tests pass"
    if hidden_passed:
        return "partial", "some hidden tests pass"
    return "model_failure", "no hidden test passes"


# ── Grading ──────────────────────────────────────────────────────────────────────


def _unit_result(res: vb.RunResult, module: str) -> dict[str, Any]:
    suite = module == vb.ALL_MODULES
    if module == vb.ALL_MODULES:  # whole-suite unit (Commit0): every collected test
        stats = {"passed": res.passed, "failed": res.failed}
    else:
        stats = res.by_file.get(module, {"passed": 0, "failed": 0})
    return {
        "passed": res.returncode == 0 and not vb.unit_fails(res, module),
        "rc": res.returncode,
        "tests": dict(stats),
        # Suite (Commit0) units run the library's whole test tree: code the agent left
        # broken at import makes pytest exit 2/4/5 (conftest import error, no tests),
        # which is a model failure, not a harness crash.
        "crash": res.returncode not in ((0, 1, 2, 4, 5, -9) if suite else (0, 1, -9)),
        "timeout": res.returncode == -9,
        "output_tail": res.output_tail,
    }


def grade_candidate(
    bundle: Bundle, candidate: Path | None, scratch: Path, timeout: float
) -> dict[str, Any]:
    """Grade with validate_bundles' isolated grader, ONE pytest process per unit.

    The graded tree is a clean ``repo/`` plus only the owned files from ``candidate``.
    Each unit's hidden test file (and the integration test) runs in its own process with
    its own timeout, so a crash at import in one unit only fails that unit.
    """
    tree = vb.graded_tree(bundle.path, bundle.owned, candidate, scratch / "graded")
    jobs: list[tuple[str, str, Path]] = []
    for unit in bundle.units:
        jobs.append((unit.id, unit.module, bundle.path / unit.hidden_test.rstrip("/")))
    if bundle.has_integration:
        jobs.append(
            ("<integration>", vb.INTEGRATION, bundle.path / "hidden" / f"{vb.INTEGRATION}.py")
        )

    def run_one(job: tuple[str, str, Path]) -> tuple[str, dict[str, Any]]:
        key, module, test_file = job
        slot = scratch / f"unit-{key.strip('<>')}"
        tests = slot / "tests"
        tests.mkdir(parents=True)
        if test_file.is_dir():
            shutil.copytree(test_file, tests, dirs_exist_ok=True)
        else:
            shutil.copy2(test_file, tests / test_file.name)
        return key, _unit_result(vb.run_grader(tree, tests, slot, timeout), module)

    with concurrent.futures.ThreadPoolExecutor(max_workers=GRADE_WORKERS) as pool:
        results = dict(pool.map(run_one, jobs))
    units = {u.id: results[u.id] for u in bundle.units}
    integration = results.get("<integration>")
    crashes = {k: r["rc"] for k, r in results.items() if r["crash"]}
    timeouts = [k for k, r in results.items() if r["timeout"]]
    return {
        "hidden_passed": sum(r["tests"]["passed"] for r in results.values()),
        "hidden_failed": sum(r["tests"]["failed"] for r in results.values()),
        "units": {
            k: {kk: vv for kk, vv in r.items() if kk != "output_tail"} for k, r in units.items()
        },
        "integration": None
        if integration is None
        else {k: v for k, v in integration.items() if k != "output_tail"},
        "all_pass": bool(units)
        and all(r["passed"] for r in units.values())
        and (integration is None or integration["passed"]),
        "crashes": crashes,
        "timeouts": timeouts,
        "output": "\n".join(
            f"== {k} (rc={r['rc']})\n{r['output_tail']}" for k, r in results.items()
        ),
    }


# ── Episodes ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    bundle: Bundle
    trial: int
    ep_dir: Path

    @property
    def name(self) -> str:
        return f"{self.bundle.id}-t{self.trial}"

    @property
    def repo(self) -> Path:
        # The directory name is the dashboard's project label, so it must be unique.
        return self.ep_dir / self.name

    @property
    def home(self) -> Path:
        return self.ep_dir / "home"


@dataclasses.dataclass
class RunContext:
    base_url: str
    model: str
    memfun_cmd: list[str]
    max_agents: int
    task_timeout: int
    episode_timeout: float
    grader_timeout: float
    max_tokens: int
    dashboard: SharedDashboard | None = None
    fatal: str = ""
    kill_grace: float = 5.0
    run_dir: Path | None = None
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    running: int = 0
    peak_parallel: int = 0
    aborted: bool = False
    active_groups: dict[int, str] = dataclasses.field(default_factory=lambda: dict[int, str]())
    _ref_cache: dict[str, str] = dataclasses.field(default_factory=lambda: dict[str, str]())
    _ref_locks: dict[str, threading.Lock] = dataclasses.field(
        default_factory=lambda: dict[str, threading.Lock]()
    )

    def track(self, pgid: int, episode: str) -> None:
        with self._lock:
            self.active_groups[pgid] = episode
            record_pgid(self.run_dir, pgid, episode)
        if self.aborted:  # started after abort() took its snapshot
            kill_group(pgid, self.kill_grace)

    def untrack(self, pgid: int) -> None:
        with self._lock:
            self.active_groups.pop(pgid, None)

    def abort(self) -> None:
        """SIGTERM every running episode's group (only those), shared grace, SIGKILL."""
        with self._lock:
            self.aborted = True
            groups = list(self.active_groups)
        kill_groups(groups, self.kill_grace)

    def enter(self) -> None:
        with self._lock:
            self.running += 1
            self.peak_parallel = max(self.peak_parallel, self.running)

    def leave(self) -> None:
        with self._lock:
            self.running -= 1

    def reference_error(self, bundle: Bundle) -> str:
        """Grade the reference solution once per bundle (cached).  "" when it passes."""
        with self._lock:
            lock = self._ref_locks.setdefault(bundle.id, threading.Lock())
        with lock:
            if bundle.id not in self._ref_cache:
                self._ref_cache[bundle.id] = reference_check(bundle, self.grader_timeout)
            return self._ref_cache[bundle.id]


def reference_check(bundle: Bundle, timeout: float) -> str:
    scratch = Path(tempfile.mkdtemp(prefix=f"smoke-ref-{bundle.id}-"))
    try:
        grade = grade_candidate(bundle, bundle.path / "solution", scratch, timeout)
    except Exception as exc:
        return f"reference grading crashed: {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if grade["all_pass"] and not grade["crashes"] and not grade["timeouts"]:
        return ""
    bad = [k for k, u in grade["units"].items() if not u["passed"]]
    return f"reference solution does not pass the grader (units {bad}, crashes {grade['crashes']})"


def run_episode(plan: EpisodePlan, ctx: RunContext) -> dict[str, Any]:
    bundle = plan.bundle
    t0 = time.time()
    rec: dict[str, Any] = {
        "bundle": bundle.id,
        "kind": bundle.kind,
        "trial": plan.trial,
        "episode": plan.name,
        "dir": str(plan.ep_dir),
        "repo": str(plan.repo),
        "started_at": t0,
        "flags": [],
        "warnings": [],
    }
    log_path = plan.ep_dir / "swarm.log"
    log_text = ""
    fatal = ""
    ctx.enter()
    forwarder: EventForwarder | None = None
    private = Path(tempfile.mkdtemp(prefix=f"smoke-{plan.name}-"))
    try:
        plan.ep_dir.mkdir(parents=True, exist_ok=False)
        if ctx.fatal:
            fatal = ctx.fatal
        elif bundle.problems:
            fatal = "missing/invalid bundle files: " + "; ".join(bundle.problems)
        elif ctx.aborted:
            fatal = "run aborted (interrupted)"
        else:
            ref = ctx.reference_error(bundle)
            if ref:
                fatal = f"harness: {ref}"
        if not fatal:
            try:
                create_episode_repo(
                    bundle, plan.repo, plan.home, ctx.base_url, ctx.model, ctx.max_tokens
                )
            except subprocess.CalledProcessError as exc:
                err = (exc.stderr or b"").decode("utf-8", "replace")[-500:]
                fatal = f"git setup failed: {' '.join(map(str, exc.cmd[-3:]))}: {err}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                    raise
                fatal = f"episode repo setup failed: {exc}"
        if fatal:
            outcome, reason = classify(fatal=fatal)
            rec.update(outcome=outcome, reason=reason)
            return rec

        if ctx.dashboard is not None:
            write_json_atomic(
                plan.repo / ".memfun" / "dashboard.lock", ctx.dashboard.lock_payload()
            )
            forwarder = EventForwarder(plan.repo / ".memfun" / "memfun.db", ctx.dashboard.port)
            forwarder.start()

        cmd = swarm_cmd(ctx.memfun_cmd, episode_spec(bundle), ctx.max_agents, ctx.task_timeout)
        result_json = plan.ep_dir / SWARM_RESULT_NAME  # must not exist yet: memfun uses O_EXCL
        proc = run_in_group(
            cmd,
            cwd=plan.repo,
            env=child_env(plan.home, result_json=result_json),
            log_path=log_path,
            timeout=ctx.episode_timeout,
            grace=ctx.kill_grace,
            on_start=lambda pgid: ctx.track(pgid, plan.name),
            on_end=ctx.untrack,
        )
        rec["returncode"] = proc.returncode
        rec["timed_out"] = proc.timed_out
        if forwarder is not None:
            forwarder.finish()
            rec["dashboard"] = {"forwarded": forwarder.forwarded, "post_errors": forwarder.errors}
            if forwarder.errors:
                rec["warnings"].append(f"dashboard: {forwarder.errors} event POSTs failed")
            if forwarder.stuck:
                rec["warnings"].append("dashboard: forwarder did not stop; final drain skipped")
            forwarder = None
        if ctx.aborted:
            rec.update(outcome="harness_error", reason="run aborted (interrupted)")
            return rec

        events = [e for _, e in read_events(plan.repo / ".memfun" / "memfun.db")]
        metrics = event_metrics(events)
        log_text = read_log(log_path)
        signals = log_signals(log_text)
        rec["warnings"].extend(signals["warnings"])
        server_errors = signals["server_error_line_count"] + metrics["llm_server_error_count"]
        rec["server_errors"] = server_errors
        rec["server_error_lines"] = signals["server_error_lines"]
        rec["reviewer_json_rejections"] = signals["reviewer_json_rejections"]
        rec["truncated_responses"] = signals["truncated_responses"]
        rec["stalled_stops"] = signals["stalled_stops"]
        swarm_doc, swarm_problem = load_swarm_result(plan.ep_dir, metrics["workflow_ids"])
        usage = tool_usage(swarm_doc, swarm_problem)
        rec["tool_usage"] = usage
        rec["tool_usage_source"] = usage["source"]
        # The result file is the primary signal that memfun swarm finished; the Rich
        # "Swarm result" panel text is only the fallback.
        rec["produced_result"] = swarm_doc is not None or signals["produced_result"]
        signals["produced_result"] = rec["produced_result"]
        if swarm_doc is not None:
            rec["n_sub_tasks"] = usage["n_sub_tasks"]
            rec["agents_used"] = usage["agents_used"]
            rec["unaccepted_tasks"] = usage["unaccepted_tasks"]
        else:
            rec["n_sub_tasks"] = metrics["primary_sub_tasks"]
            rec["agents_used"] = metrics["agents_used"]
            rec["unaccepted_tasks"] = None

        collected_dir = private / "collected"
        collected_dir.mkdir()
        collected = collect_owned(bundle, plan.repo, collected_dir)
        rec["collection"] = collected
        flags: list[str] = rec["flags"]
        if bundle.kind == "fanout" and max(rec["n_sub_tasks"], len(collected["worktrees"])) <= 1:
            flags.append("decomposition_fallback")
        if any(u["duplicate_work"] for u in collected["units"].values()):
            flags.append("duplicate_work")
        if collected["tests_modified"]:
            flags.append("tests_modified")
        if collected["tampering"]:
            flags.append("symlink_tampering")
        if proc.timed_out:
            flags.append("episode_timeout")
        if proc.returncode not in (0, None) and not signals["produced_result"]:
            flags.append("swarm_crash")
        if server_errors:
            flags.append("server_errors")

        grade_error = ""
        grade: dict[str, Any] | None = None
        try:
            grade = grade_candidate(bundle, collected_dir, private, ctx.grader_timeout)
        except OSError:
            raise
        except Exception as exc:
            grade_error = f"grader crashed: {type(exc).__name__}: {exc}"
        if grade is not None:
            if grade["crashes"]:
                flags.append("grader_crash")
            if grade["timeouts"]:
                flags.append("grader_timeout")
            write_private(plan.ep_dir / "grader.txt", grade.pop("output"))
            for name in ("collected", "graded"):
                problem = archive_dir(private / name, plan.ep_dir, name)
                if problem:
                    rec["warnings"].append(problem)
        rec["grade"] = grade

        units: dict[str, Any] = {}
        for unit in bundle.units:
            info = collected["units"][unit.id]
            g = (grade or {}).get("units", {}).get(unit.id, {})
            if info["symlink_tampering"]:
                status = "fail"
            elif not info["output"]:
                status = "no_output"
            else:
                status = "pass" if g.get("passed") else "fail"
            units[unit.id] = {
                **info,
                "status": status,
                "tests": g.get("tests", {"passed": 0, "failed": 0}),
                "grader_rc": g.get("rc"),
            }
        rec["units"] = units

        outcome, reason = classify(
            timed_out=proc.timed_out,
            timeout_s=ctx.episode_timeout,
            returncode=proc.returncode,
            produced_result=signals["produced_result"],
            grade_error=grade_error,
            hidden_passed=grade["hidden_passed"] if grade else 0,
            all_pass=bool(grade and grade["all_pass"]),
            any_output=any(u["output"] for u in collected["units"].values()),
            server_errors=server_errors,
        )
        rec.update(outcome=outcome, reason=reason, metrics=metrics)
        return rec
    except OSError as exc:
        label = "disk full" if exc.errno == errno.ENOSPC else "OS error"
        rec.update(outcome="harness_error", reason=f"{label}: {exc}")
        rec["traceback"] = traceback.format_exc()[-2000:]
        return rec
    except Exception as exc:
        reason = f"runner exception: {type(exc).__name__}: {exc}"
        rec.update(outcome="harness_error", reason=reason)
        rec["traceback"] = traceback.format_exc()[-2000:]
        return rec
    finally:
        if forwarder is not None:
            forwarder.finish()
        shutil.rmtree(private, ignore_errors=True)
        ctx.leave()
        rec["wall_s"] = round(time.time() - t0, 3)
        if rec.get("outcome") != "pass":
            rec["log_tail"] = _tail_text(log_text or read_log(log_path))
        else:
            rec["log_tail"] = ""
        if plan.ep_dir.is_dir():
            with contextlib.suppress(OSError):
                write_json_atomic(plan.ep_dir / "episode.json", rec)


# ── Aggregation and summary ──────────────────────────────────────────────────────


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _units_of(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return cast("dict[str, dict[str, Any]]", episode.get("units") or {})


def _tool_block(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean tool counts per episode and search-tool use, over episodes with the file."""
    usages = [
        cast("dict[str, Any]", e["tool_usage"])
        for e in eps
        if isinstance(e.get("tool_usage"), dict) and e["tool_usage"].get("source") != "unavailable"
    ]
    n = len(usages)
    types = sorted({k for u in usages for k in u["counts"]} | set(TOOL_TYPES))
    sub_total = sum(len(u["sub_tasks"]) for u in usages)
    return {
        "episodes_with_data": n,
        "episodes_without_data": sum(
            1 for e in eps if e.get("tool_usage_source") == "unavailable"
        ),
        "mean_counts": {t: round(sum(u["counts"].get(t, 0) for u in usages) / n, 3) for t in types}
        if n
        else {},
        "episodes_used_search": _rate(sum(1 for u in usages if u["used_search_tools"]), n),
        "sub_tasks_used_search": _rate(sum(u["sub_tasks_used_search"] for u in usages), sub_total),
        "mean_sub_tasks": round(sum(u["n_sub_tasks"] for u in usages) / n, 3) if n else None,
    }


def aggregate(episodes: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    def counts(eps: list[dict[str, Any]]) -> dict[str, int]:
        return {o: sum(1 for e in eps if e.get("outcome") == o) for o in OUTCOMES}

    def block(eps: list[dict[str, Any]]) -> dict[str, Any]:
        c = counts(eps)
        graded = [e for e in eps if e.get("outcome") != "harness_error"]
        unit_rows = [u for e in graded for u in _units_of(e).values()]
        mets = [e["metrics"] for e in eps if e.get("metrics")]
        return {
            "episodes": len(eps),
            "outcomes": c,
            "episode_pass_rate": _rate(c["pass"], len(graded)),
            "unit_pass_rate": _rate(sum(u["status"] == "pass" for u in unit_rows), len(unit_rows)),
            "harness_error_rate": _rate(c["harness_error"], len(eps)),
            "tokens": {
                "prompt": sum(m["tokens"]["prompt"] for m in mets),
                "completion": sum(m["tokens"]["completion"] for m in mets),
            },
            "wall_s_mean": round(sum(e.get("wall_s", 0.0) for e in eps) / len(eps), 3)
            if eps
            else None,
            "agents_used_max": max((m["agents_used"] for m in mets), default=0),
            "peak_concurrency": max((m["peak_concurrency"] for m in mets), default=0),
            "reviewer_json_rejections": sum(e.get("reviewer_json_rejections", 0) for e in eps),
            "truncated_responses": sum(e.get("truncated_responses", 0) for e in eps),
            "stalled_stops": sum(e.get("stalled_stops", 0) for e in eps),
            "flags": sorted({f for e in eps for f in e.get("flags", [])}),
            "tool_usage": _tool_block(eps),
        }

    by_bundle: dict[str, list[dict[str, Any]]] = {}
    for e in episodes:
        by_bundle.setdefault(str(e["bundle"]), []).append(e)
    units: dict[str, dict[str, Any]] = {}
    for bid, eps in by_bundle.items():
        for e in eps:
            if e.get("outcome") == "harness_error":
                continue
            for uid, u in _units_of(e).items():
                slot = units.setdefault(f"{bid}/{uid}", {"graded": 0, "passed": 0, "no_output": 0})
                slot["graded"] += 1
                slot["passed"] += u["status"] == "pass"
                slot["no_output"] += u["status"] == "no_output"
    for slot in units.values():
        slot["pass_rate"] = _rate(slot["passed"], slot["graded"])
    overall = block(episodes)
    rate = overall["harness_error_rate"] or 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        **meta,
        "void": rate > VOID_THRESHOLD,
        "void_threshold": VOID_THRESHOLD,
        "overall": overall,
        "bundles": {bid: block(eps) for bid, eps in sorted(by_bundle.items())},
        "units": dict(sorted(units.items())),
        "harness_errors": [
            {
                "episode": e["episode"],
                "reason": e.get("reason", ""),
                "log_tail": e.get("log_tail", ""),
            }
            for e in episodes
            if e.get("outcome") == "harness_error"
        ],
        "episodes": episodes,
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def summary_table(results: dict[str, Any]) -> str:
    head = (
        f"{'bundle':<14} {'eps':>3} {'pass':>4} {'part':>4} {'fail':>4} {'harn':>4} "
        f"{'ep%':>5} {'unit%':>5} {'wall_s':>7} {'tok_in':>8} {'tok_out':>7} "
        f"{'agents':>6} {'peak':>4}  flags"
    )
    rows = [head, "-" * len(head)]

    def row(name: str, b: dict[str, Any]) -> str:
        o = b["outcomes"]
        return (
            f"{name:<14} {b['episodes']:>3} {o['pass']:>4} {o['partial']:>4} "
            f"{o['model_failure']:>4} {o['harness_error']:>4} {_pct(b['episode_pass_rate']):>5} "
            f"{_pct(b['unit_pass_rate']):>5} {b['wall_s_mean'] or 0:>7.1f} "
            f"{b['tokens']['prompt']:>8} {b['tokens']['completion']:>7} "
            f"{b['agents_used_max']:>6} {b['peak_concurrency']:>4}  {','.join(b['flags'])}"
        )

    for bid, b in results["bundles"].items():
        rows.append(row(bid, b))
    rows.append("-" * len(head))
    rows.append(row("ALL", results["overall"]))
    o = results["overall"]["outcomes"]
    rows.append(
        f"raw counts: episodes={results['overall']['episodes']} pass={o['pass']} "
        f"partial={o['partial']} model_failure={o['model_failure']} "
        f"harness_error={o['harness_error']}  (ep% and unit% exclude harness errors; "
        "harness rate is over ALL episodes)"
    )
    rows.append("")
    rows.append("tool usage (mean per episode; search = grep/find_files):")
    for name, b in [*results["bundles"].items(), ("ALL", results["overall"])]:
        tu = b["tool_usage"]
        if not tu["episodes_with_data"]:
            rows.append(f"  {name:<14} no data ({tu['episodes_without_data']} unavailable)")
            continue
        mc = tu["mean_counts"]
        rows.append(
            f"  {name:<14} grep {mc.get('grep', 0):g} find {mc.get('find_files', 0):g} "
            f"read {mc.get('read', 0):g} edit {mc.get('edit', 0):g} "
            f"write {mc.get('write', 0):g} cmd {mc.get('cmd', 0):g} | sub-tasks "
            f"{tu['mean_sub_tasks']:g} | search: eps {_pct(tu['episodes_used_search'])} "
            f"sub-tasks {_pct(tu['sub_tasks_used_search'])} "
            f"(data {tu['episodes_with_data']}/{b['episodes']})"
        )
    rows.append("")
    rows.append("units (graded episodes only):")
    for uid, u in results["units"].items():
        rows.append(f"  {uid:<32} {u['passed']}/{u['graded']} pass  ({u['no_output']} no_output)")
    if results["harness_errors"]:
        rows.append("")
        rows.append("harness errors (not model failures):")
        for h in results["harness_errors"]:
            rows.append(f"  {h['episode']}: {(h['reason'].splitlines() or [''])[0][:150]}")
    rows.append("")
    rows.append(
        f"wall-clock {results.get('wall_s', 0):.1f}s, peak parallel episodes "
        f"{results.get('peak_parallel_episodes', 0)}, model {results.get('model')!r}"
    )
    if results.get("interrupted"):
        rows.append(f"INTERRUPTED: {results.get('not_run', 0)} episode(s) not run")
    if results["void"]:
        rows.append(
            f"ARM VOID: harness error rate {_pct(results['overall']['harness_error_rate'])} "
            f"> {VOID_THRESHOLD:.0%}"
        )
    return "\n".join(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (…/v1)")
    ap.add_argument("--model", help="served model id (default: the single one at GET /models)")
    ap.add_argument(
        "--suite",
        choices=("smoke", "commit0"),
        default="smoke",
        help="bundle source: the 6 toy smoke bundles, or Commit0 libraries built by "
        "commit0_adapter.py into --commit0-dir",
    )
    ap.add_argument(
        "--commit0-dir",
        type=Path,
        default=Path.home() / ".cache" / "memfun-commit0",
        help="commit0_adapter.py cache (bundles under <dir>/bundles)",
    )
    ap.add_argument(
        "--bundles",
        nargs="+",
        default=None,
        help="bundle ids (space or comma separated); default: all 6 smoke bundles, or "
        "every built Commit0 bundle",
    )
    ap.add_argument("--bundles-root", type=Path, default=SMOKE_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-agents", type=int, default=4, help="memfun swarm --max-pool")
    ap.add_argument("--parallel-tasks", type=int, default=1, help="episodes run at once")
    ap.add_argument(
        "--episode-timeout",
        type=float,
        default=None,
        help=f"hard cap, seconds (default {SMOKE_EPISODE_TIMEOUT:g}; "
        f"{COMMIT0_EPISODE_TIMEOUT:g} = 90 min for --suite commit0)",
    )
    ap.add_argument(
        "--task-timeout",
        type=int,
        default=None,
        help="memfun swarm --timeout (default 900; --suite commit0: the episode timeout, "
        "so the episode cap is the only wall-clock limit)",
    )
    ap.add_argument(
        "--grader-timeout",
        type=float,
        default=None,
        help="per unit test run, seconds (default 300; --suite commit0: 900)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="[llm] max_tokens for EVERY role (per call); a truncated response is reported "
        "as an error, never silently accepted",
    )
    ap.add_argument("--memfun-cmd", default="memfun", help="memfun executable (shell-split)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--dashboard-port", type=int, default=8081)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument(
        "--metrics-url", default="", help="optional Prometheus URL for run-level token counters"
    )
    ap.add_argument("--dry-plan", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--kill-grace", type=float, default=5.0, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _bounded(value: int, lo: int, hi: int, flag: str) -> None:
    if not lo <= value <= hi:
        raise UsageError(f"{flag} must be in {lo}..{hi}, got {value}")


def _apply_suite_defaults(args: argparse.Namespace) -> None:
    commit0 = args.suite == "commit0"
    if args.episode_timeout is None:
        args.episode_timeout = COMMIT0_EPISODE_TIMEOUT if commit0 else SMOKE_EPISODE_TIMEOUT
    if args.task_timeout is None:
        args.task_timeout = int(args.episode_timeout) if commit0 else 900
    if args.grader_timeout is None:
        args.grader_timeout = COMMIT0_GRADER_TIMEOUT if commit0 else 300.0


def _validate_args(args: argparse.Namespace) -> None:
    _apply_suite_defaults(args)
    args.base_url = check_base_url(args.base_url)
    check_port(args.dashboard_port, "--dashboard-port")
    _bounded(args.trials, 1, MAX_TRIALS, "--trials")
    _bounded(args.max_agents, 1, MAX_AGENTS, "--max-agents")
    _bounded(args.parallel_tasks, 1, MAX_PARALLEL, "--parallel-tasks")
    if args.task_timeout < 1 or args.max_tokens < 1:
        raise UsageError("--task-timeout and --max-tokens must be >= 1")
    if args.episode_timeout <= 0 or args.grader_timeout <= 0:
        raise UsageError("timeouts must be > 0")
    if args.metrics_url:
        args.metrics_url = check_http_url(args.metrics_url, "--metrics-url")
    for flag, url in (("--base-url", args.base_url), ("--metrics-url", args.metrics_url)):
        if url and not is_loopback_host(urllib.parse.urlsplit(url).hostname):
            print(
                f"WARNING: {flag} {url} is NOT a loopback address: model traffic and episode "
                "specs leave this machine (use an SSH tunnel to 127.0.0.1)",
                file=sys.stderr,
            )
    if args.suite == "commit0":
        args.bundles_root = args.commit0_dir.expanduser().resolve() / "bundles"
    known = set(known_bundles(args.bundles_root))
    if args.bundles is None:
        args.bundles = list(DEFAULT_BUNDLES) if args.suite == "smoke" else sorted(known)
    selected: list[str] = []
    for item in args.bundles:
        for bid in (b.strip() for b in item.split(",")):
            if not bid:
                continue
            if not BUNDLE_ID_RE.match(bid) or bid not in known:
                raise UsageError(f"unknown bundle {bid!r} (known: {', '.join(sorted(known))})")
            if bid not in selected:
                selected.append(bid)
    if not selected:
        raise UsageError("--bundles selects no bundle")
    args.bundles = selected
    args.run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    if not RUN_ID_RE.match(args.run_id) or ".." in args.run_id:
        raise UsageError(f"--run-id must match {RUN_ID_RE.pattern} and not contain '..'")
    out = args.out_dir.expanduser().resolve()
    if out == _REPO_ROOT or _REPO_ROOT in out.parents:
        raise UsageError(
            f"--out-dir {out} is inside the memfun checkout: episodes would sit a short "
            "relative path from the bundles' solution/ and hidden/ and pick up memfun's "
            "pytest config. Use a directory outside the repo."
        )
    args.out_dir = out
    if not shlex.split(args.memfun_cmd):
        raise UsageError("--memfun-cmd is empty")


def plan_episodes(args: argparse.Namespace, run_dir: Path) -> list[EpisodePlan]:
    plans: list[EpisodePlan] = []
    for bid in args.bundles:
        bundle = load_bundle(args.bundles_root / bid)
        for trial in range(1, args.trials + 1):
            plans.append(EpisodePlan(bundle, trial, run_dir / bid / str(trial)))
    return plans


def dry_plan(args: argparse.Namespace, run_dir: Path, plans: list[EpisodePlan]) -> str:
    memfun_cmd = shlex.split(args.memfun_cmd)
    model = args.model or f"<auto: the single model at GET {args.base_url}/models>"
    dash_cmd = shlex.join(
        [*memfun_cmd, "dashboard", "--remote", "--port", str(args.dashboard_port)]
    )
    out = [
        "DRY PLAN (nothing is created, started or contacted)",
        f"run id        : {args.run_id}",
        f"run dir       : {run_dir}",
        f"model server  : {args.base_url}  (port {url_port(args.base_url)}; 8434 is refused)",
        f"model         : {model}",
        f"memfun        : {shlex.join(memfun_cmd)}",
        f"parallel      : {args.parallel_tasks} episode(s) at once, --max-pool {args.max_agents}",
        f"timeouts      : episode {args.episode_timeout:g}s (process-group kill), "
        f"memfun --timeout {args.task_timeout}s, grader {args.grader_timeout:g}s per unit",
    ]
    if args.no_dashboard:
        out.append("dashboard     : disabled (episodes never start one either)")
    else:
        out.append(
            f"dashboard     : ONE on port {args.dashboard_port}: reuse it if a memfun dashboard "
            f"answers there, else start `{dash_cmd}` in {run_dir / '_dashboard'}; each episode "
            "gets .memfun/dashboard.lock -> it, and the runner forwards each episode's events "
            "to POST /api/event"
        )
    out.append(f"child env     : {', '.join(ENV_ALLOW)}, HOME=<episode>/home, {DUMMY_KEY_ENV}")
    out.append("reference     : each bundle's solution/ is graded once first (harness check)")
    out.append(f"episodes      : {len(plans)}")
    for p in plans:
        spec = episode_spec(p.bundle)
        first = spec.strip().splitlines()[0][:60] if spec.strip() else ""
        cmd = swarm_cmd(memfun_cmd, f"<spec: {first}...>", args.max_agents, args.task_timeout)
        problem = f"  !! {'; '.join(p.bundle.problems)}" if p.bundle.problems else ""
        out.append(f"  {p.name:<18} kind={p.bundle.kind:<8} units={len(p.bundle.units)}{problem}")
        out.append(f"      repo : {p.repo}")
        out.append(f"      cmd  : (cd repo) {shlex.join(cmd)}")
        out.append(f"      grade: hidden/ of {p.bundle.path}, one pytest per unit, private dir")
    return "\n".join(out)


def _install_signals() -> dict[int, Any]:
    """SIGTERM -> SystemExit in the main thread, so the abort path runs."""
    if threading.current_thread() is not threading.main_thread():
        return {}

    def _on_term(_signum: int, _frame: object) -> None:
        raise SystemExit(143)

    previous: dict[int, Any] = {signal.SIGINT: signal.getsignal(signal.SIGINT)}
    previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, _on_term)
    return previous


def _quiet_signals() -> None:
    """During cleanup a second Ctrl-C / SIGTERM must not cut the cleanup short."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _restore_signals(previous: dict[int, Any]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def run(args: argparse.Namespace) -> int:
    run_dir = args.out_dir / args.run_id
    plans = plan_episodes(args, run_dir)
    if args.dry_plan:
        print(dry_plan(args, run_dir, plans))
        return 0
    if run_dir.exists():
        raise UsageError(f"run dir already exists (pick another --run-id): {run_dir}")
    run_dir.mkdir(parents=True)
    memfun_cmd = shlex.split(args.memfun_cmd)
    t0 = time.time()

    model, fatal = args.model or "", ""
    dashboard: SharedDashboard | None = None
    started: list[SharedDashboard] = []
    ctx: RunContext | None = None
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    episodes: list[dict[str, Any]] = []
    usage_before: dict[str, float] | None = None
    interrupted = False

    def meta() -> dict[str, Any]:
        return {
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "created_at": dt.datetime.fromtimestamp(t0, dt.UTC).isoformat(),
            "base_url": args.base_url,
            "model": model,
            "memfun_cmd": memfun_cmd,
            "bundles": args.bundles,
            "trials": args.trials,
            "max_agents": args.max_agents,
            "parallel_tasks": args.parallel_tasks,
            "episode_timeout": args.episode_timeout,
            "task_timeout": args.task_timeout,
            "dashboard": None
            if dashboard is None
            else {"port": dashboard.port, "started_by_runner": dashboard.owned},
            "fatal": fatal,
            "interrupted": interrupted,
            "not_run": len(plans) - len(episodes),
            "wall_s": round(time.time() - t0, 3),
            "peak_parallel_episodes": ctx.peak_parallel if ctx else 0,
            "server_usage": None,
        }

    def _register(dash: SharedDashboard) -> None:
        started.append(dash)
        record_pgid(run_dir, dash.lock_pid, "_dashboard")

    previous = _install_signals()
    try:
        model, fatal = preflight(args.base_url, args.model)
        usage_before = server_usage_snapshot(args.metrics_url) if args.metrics_url else None
        if not args.no_dashboard and not fatal:
            dash = start_dashboard(
                memfun_cmd, args.dashboard_port, run_dir / "_dashboard", register=_register
            )
            if dash.error:
                fatal = dash.error
            else:
                dashboard = dash
                if dash.owned:
                    print(f"dashboard: started http://localhost:{dash.port}", flush=True)
                else:
                    print(
                        f"dashboard: REUSING the memfun dashboard already on port {dash.port}; "
                        "episode events will appear there (--no-dashboard to avoid)",
                        flush=True,
                    )
        ctx = RunContext(
            base_url=args.base_url,
            model=model,
            memfun_cmd=memfun_cmd,
            max_agents=args.max_agents,
            task_timeout=args.task_timeout,
            episode_timeout=args.episode_timeout,
            grader_timeout=args.grader_timeout,
            max_tokens=args.max_tokens,
            dashboard=dashboard,
            fatal=fatal,
            kill_grace=args.kill_grace,
            run_dir=run_dir,
        )
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_tasks)
        futures = [pool.submit(run_episode, p, ctx) for p in plans]
        for fut in concurrent.futures.as_completed(futures):
            ep = fut.result()
            print(f"  {ep['episode']:<18} {ep['outcome']:<14} {ep['reason'][:80]}", flush=True)
            episodes.append(ep)
            # Partial results after every episode: survive a SIGKILL of the runner.
            write_json_atomic(run_dir / "results.json", aggregate(episodes, meta()))
    except (KeyboardInterrupt, SystemExit):
        # The swarms run in their own sessions, so they would outlive us.  Kill exactly
        # the groups we started, drop queued episodes, keep what finished.
        interrupted = True
        _quiet_signals()
        print("interrupted: stopping running episodes", file=sys.stderr, flush=True)
        if ctx is not None:
            ctx.abort()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        seen = {id(e) for e in episodes}
        for fut in futures:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                ep = fut.result()
                if id(ep) not in seen:
                    episodes.append(ep)
    finally:
        _quiet_signals()
        if pool is not None:
            pool.shutdown(wait=True)
        for dash in started:  # runner-started only; a reused dashboard is never stopped
            dash.stop()
        _restore_signals(previous)
    episodes.sort(key=lambda e: (e["bundle"], e["trial"]))
    final_meta = meta()
    if args.metrics_url and not interrupted:
        usage_after = server_usage_snapshot(args.metrics_url)
        server_usage: dict[str, Any] = {"available": bool(usage_before and usage_after)}
        if usage_before and usage_after:
            server_usage.update(
                prompt=usage_after["prompt"] - usage_before["prompt"],
                completion=usage_after["completion"] - usage_before["completion"],
                note="whole-server counters: include any other traffic on this server",
            )
        final_meta["server_usage"] = server_usage
    results = aggregate(episodes, final_meta)
    write_json_atomic(run_dir / "results.json", results)
    print()
    print(summary_table(results))
    print(f"\nresults: {run_dir / 'results.json'}")
    if interrupted:
        return 130
    return 1 if results["void"] else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        return run(args)
    except UsageError as exc:
        print(f"run_smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
