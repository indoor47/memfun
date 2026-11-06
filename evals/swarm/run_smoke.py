#!/usr/bin/env python3
"""Run the swarm smoke bundles through ``memfun swarm`` and grade them with hidden tests.

Each episode (one bundle x one trial) gets a fresh throwaway git repo built from the
bundle's ``repo/`` with its own ``memfun.toml`` (one ``[llm]`` block for the served model),
runs ``memfun swarm`` in its own process group, then grades the units' owned files with
the isolated grader from ``validate_bundles.py`` (owned files over a clean ``repo/``,
hidden tests from a separate grader dir).  All episodes' events are forwarded to ONE
dashboard.  See ``README-run-smoke.md`` next to this file for commands, the
``results.json`` schema and the classification rules.

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
import json
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:  # also runnable as a plain script
    sys.path.insert(0, str(_REPO_ROOT))

from evals.swarm import validate_bundles as vb  # noqa: E402

SCHEMA_VERSION = 1
SMOKE_ROOT = HERE / "smoke"
DEFAULT_OUT_DIR = HERE / "runs"
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
#: memfun_runtime.distributed.EVENT_TOPIC (kept literal so the runner is stdlib-only).
EVENT_TOPIC = "memfun.distributed.events"
VOID_THRESHOLD = 0.05
OUTCOMES = ("pass", "partial", "model_failure", "harness_error")
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
GIT_IDENTITY = ("-c", "user.name=memfun-smoke", "-c", "user.email=smoke@memfun.invalid")
WARNING_RE = re.compile(
    r"MCP tool discovery|Failed to discover MCP tools|Task decomposition failed|Traceback"
)
SERVER_ERROR_RE = re.compile(
    r"APIConnectionError|Connection ?refused|ConnectError|ServiceUnavailable|"
    r"InternalServerError|BadGateway|APITimeoutError|NotFoundError: .*model|"
    r"HTTP(?: status)? 5\d\d|status[_ ]code[=: ]+5\d\d",
    re.IGNORECASE,
)
#: Task ids the engine derives from a sub-task (pair re-dispatch, revision, review).
DERIVED_TASK_RE = re.compile(r"_(pair\d+|rev\d+|review_.*|pr_review)$")
RESULT_MARKER = "Swarm result"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
LOG_TAIL_LINES = 40


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
        return Path(self.hidden_test).stem


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


def episode_spec(bundle: Bundle) -> str:
    return f"{bundle.spec.strip()}\n\n{SPEC_SUFFIX}"


# ── URLs, ports, model discovery ───────────────────────────────────────────────


def url_port(url: str) -> int:
    parts = urllib.parse.urlsplit(url)
    if parts.port is not None:
        return parts.port
    return 443 if parts.scheme == "https" else 80


def check_base_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UsageError(f"--base-url must be an http(s) URL, got {url!r}")
    try:
        port = url_port(url)
    except ValueError as exc:
        raise UsageError(f"--base-url has an invalid port: {url!r}") from exc
    if port in FORBIDDEN_PORTS:
        raise UsageError(
            f"refusing --base-url on port {port}: that is the owner's live model server"
        )
    return url.rstrip("/")


def check_port(port: int, flag: str) -> int:
    if port in FORBIDDEN_PORTS:
        raise UsageError(f"refusing {flag} {port}: that is the owner's live model server")
    if not 1 <= port <= 65535:
        raise UsageError(f"{flag} out of range: {port}")
    return port


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

HttpGet = Callable[[str, float], Any]


def http_get_json(url: str, timeout: float = 10.0) -> Any:
    """GET ``url`` (no proxies, no auth headers) and decode JSON."""
    with _NO_PROXY_OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


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
        with _NO_PROXY_OPENER.open(metrics_url, timeout=5.0) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return None
    c = prom_counters(text, (*PROM_PROMPT, *PROM_GEN))
    return {
        "prompt": sum(c.get(n, 0.0) for n in PROM_PROMPT),
        "completion": sum(c.get(n, 0.0) for n in PROM_GEN),
    }


# ── Child environment and process groups ─────────────────────────────────────────


def child_env(home: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
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


def kill_group(pgid: int, grace: float = 5.0, reap: Callable[[], object] | None = None) -> None:
    """SIGTERM then SIGKILL the process group ``pgid`` (and only it).

    ``reap`` (e.g. ``proc.poll``) lets the caller reap its own leader while waiting, so
    a zombie leader does not keep the group looking alive for the whole grace period.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _group_alive(pgid) and time.monotonic() < deadline:
        if reap is not None:
            reap()
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


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
    with log_path.open("wb") as log:
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
    """Dashboard events from a memfun T1 (SQLite) event bus DB, oldest first.  Read-only."""
    if not db.is_file():
        return []
    rows: list[Any] = []
    # Read-only first; a WAL db left without its -shm (crashed writer) needs a plain open.
    for uri in (f"{db.as_uri()}?mode=ro", db.as_uri()):
        try:
            con = sqlite3.connect(uri, uri=True, timeout=5.0)
        except sqlite3.Error:
            continue
        try:
            rows = con.execute(
                "SELECT rowid, payload FROM events WHERE topic = ? AND rowid > ? ORDER BY rowid",
                (EVENT_TOPIC, after_rowid),
            ).fetchall()
            break
        except sqlite3.Error:
            rows = []
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
    with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
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

    def finish(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=30)
        self.pump()  # final drain


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
        if self.owned and self.proc is not None:
            kill_group(self.proc.pid, reap=self.proc.poll)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=10)


def start_dashboard(
    memfun_cmd: list[str],
    port: int,
    workdir: Path,
    *,
    wait_s: float = 60.0,
    get: HttpGet = http_get_json,
) -> SharedDashboard:
    """Reuse a healthy memfun dashboard on ``port`` or start ``memfun dashboard --remote``."""
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
    try:
        proc = subprocess.Popen(
            [*memfun_cmd, "dashboard", "--remote", "--port", str(port)],
            cwd=workdir,
            env=dashboard_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.close()
        return SharedDashboard(port, 0, False, error=f"dashboard failed to start: {exc}")
    log.close()
    dash = SharedDashboard(port=port, lock_pid=proc.pid, owned=True, proc=proc)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if dashboard_health(port, get):
            return dash
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    dash.stop()
    tail = _tail(workdir / "dashboard.log", 10)
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
    home.mkdir(parents=True, exist_ok=True)
    (home / ".gitconfig").write_text(
        "[user]\n\tname = memfun-smoke\n\temail = smoke@memfun.invalid\n"
        "[commit]\n\tgpgsign = false\n",
        encoding="utf-8",
    )
    shutil.copytree(bundle.path / "repo", repo)
    (repo / "memfun.toml").write_text(memfun_toml(base_url, model, max_tokens), "utf-8")
    (repo / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
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


# ── Owned-file collection ────────────────────────────────────────────────────────

_SKIP_PARTS = {".git", ".memfun", "__pycache__", ".pytest_cache", ".ruff_cache"}
_SKIP_ROOT_FILES = {"memfun.toml", ".gitignore"}


def _differs(start: Path, candidate: Path) -> bool:
    if not candidate.is_file():
        return False
    if not start.is_file():
        return True
    return start.read_bytes() != candidate.read_bytes()


def find_worktrees(repo: Path) -> list[tuple[str, Path]]:
    """``(label, path)`` for ``.memfun/worktrees/<workflow_id>/<task_id>`` dirs."""
    base = repo / ".memfun" / "worktrees"
    if not base.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for wf in sorted(p for p in base.iterdir() if p.is_dir()):
        for task in sorted(p for p in wf.iterdir() if p.is_dir()):
            out.append((f"{wf.name}/{task.name}", task))
    return out


def _tree_files(root: Path) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_PARTS)
        for name in sorted(filenames):
            if name not in _SKIP_PARTS:
                yield (Path(dirpath) / name).relative_to(root).as_posix()


def collect_owned(bundle: Bundle, repo: Path, dest: Path) -> dict[str, Any]:
    """Pick each owned file's graded copy from the worktrees (project root as fallback).

    For each owned path: the copies that differ from the bundle's start tree are
    candidates.  Worktree copies win; when several worktrees changed it the newest by
    mtime is taken and ``duplicate`` is set.  The project-root copy is used only when no
    worktree changed the path.  Chosen copies are copied to ``dest/<path>``.
    """
    start = bundle.path / "repo"
    worktrees = find_worktrees(repo)
    files: dict[str, dict[str, Any]] = {}
    for rel in bundle.owned:
        changed = [(label, wt / rel) for label, wt in worktrees if _differs(start / rel, wt / rel)]
        chosen: tuple[str, Path] | None = None
        if changed:
            chosen = max(changed, key=lambda c: c[1].stat().st_mtime)
        elif _differs(start / rel, repo / rel):
            chosen = ("root", repo / rel)
        if chosen is not None:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chosen[1], target)
        files[rel] = {
            "source": chosen[0] if chosen else None,
            "changed_in": [label for label, _ in changed],
            "duplicate": len(changed) > 1,
        }
    units: dict[str, dict[str, Any]] = {}
    for unit in bundle.units:
        infos = [files[p] for p in unit.owns]
        tasks = sorted({lbl.split("/", 1)[1] for i in infos for lbl in i["changed_in"]})
        units[unit.id] = {
            "output": any(i["source"] for i in infos),
            "owned_changed": sum(1 for i in infos if i["source"]),
            "sub_tasks": tasks,
            "duplicate_work": any(i["duplicate"] for i in infos),
        }
    owned = set(bundle.owned)
    stray: list[str] = []
    for label, root in [*worktrees, ("root", repo)]:
        for rel in _tree_files(root):
            if rel in owned or rel in _SKIP_ROOT_FILES:
                continue
            if _differs(start / rel, root / rel):
                stray.append(f"{label}:{rel}")
    return {
        "worktrees": [label for label, _ in worktrees],
        "files": files,
        "units": units,
        "stray_changes": stray[:50],
        "tests_modified": any(s.split(":", 1)[1].startswith("tests/") for s in stray),
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
            or etype
            not in (
                "task.published",
                "task.picked_up",
                "task.completed",
                "task.failed",
            )
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


# ── Logs ─────────────────────────────────────────────────────────────────────────


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def log_signals(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    warnings = [ln.strip()[:300] for ln in lines if WARNING_RE.search(ln)]
    server = [ln.strip()[:300] for ln in lines if SERVER_ERROR_RE.search(ln)]
    return {
        "warnings": warnings[:50],
        "server_error_lines": server[:20],
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
    server_errors: int = 0,
) -> tuple[str, str]:
    """One of OUTCOMES plus a reason.  Harness problems are never blamed on the model."""
    if fatal:
        return "harness_error", fatal
    if timed_out:
        return "harness_error", f"episode timeout after {timeout_s:g}s (process group killed)"
    if returncode not in (0, None) and not produced_result:
        return "harness_error", f"memfun swarm exited {returncode} without a result"
    if grade_error:
        return "harness_error", grade_error
    if all_pass:
        return "pass", "all hidden tests pass"
    if server_errors:
        return "harness_error", f"model server errors during the episode ({server_errors})"
    if hidden_passed:
        return "partial", "some hidden tests pass"
    return "model_failure", "no hidden test passes"


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
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    running: int = 0
    peak_parallel: int = 0
    aborted: bool = False
    active_groups: set[int] = dataclasses.field(default_factory=lambda: set[int]())

    def track(self, pgid: int) -> None:
        with self._lock:
            self.active_groups.add(pgid)
        if self.aborted:  # started after abort() took its snapshot
            kill_group(pgid, self.kill_grace)

    def untrack(self, pgid: int) -> None:
        with self._lock:
            self.active_groups.discard(pgid)

    def abort(self) -> None:
        """Kill the process groups of every running episode (and only those)."""
        with self._lock:
            self.aborted = True
            groups = list(self.active_groups)
        for pgid in groups:
            kill_group(pgid, self.kill_grace)

    def enter(self) -> None:
        with self._lock:
            self.running += 1
            self.peak_parallel = max(self.peak_parallel, self.running)

    def leave(self) -> None:
        with self._lock:
            self.running -= 1


def grade_candidate(
    bundle: Bundle, candidate: Path | None, scratch: Path, timeout: float
) -> dict[str, Any]:
    """Grade with validate_bundles' isolated grader (owned files over a clean repo/)."""
    tree = vb.graded_tree(bundle.path, bundle.owned, candidate, scratch / "graded")
    res = vb.run_grader(tree, bundle.path / "hidden", scratch / "grading", timeout)
    units = {u.id: not vb.unit_fails(res, u.module) for u in bundle.units}
    integration = not vb.unit_fails(res, vb.INTEGRATION) if bundle.has_integration else None
    crashed = res.returncode not in (0, 1, -9) or (res.returncode == 1 and not res.by_file)
    return {
        "returncode": res.returncode,
        "hidden_passed": res.passed,
        "hidden_failed": res.failed,
        "by_file": res.by_file,
        "units": units,
        "integration": integration,
        "all_pass": bool(units) and all(units.values()) and integration is not False,
        "grader_timeout": res.returncode == -9,
        "error": f"grader crashed (pytest rc={res.returncode})" if crashed else "",
        "output_tail": res.output_tail,
    }


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
    fatal = ""
    if ctx.fatal:
        fatal = ctx.fatal
    elif bundle.problems:
        fatal = "missing/invalid bundle files: " + "; ".join(bundle.problems)
    ctx.enter()
    forwarder: EventForwarder | None = None
    try:
        plan.ep_dir.mkdir(parents=True, exist_ok=True)
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
        if ctx.aborted:
            fatal = "run aborted (interrupted)"
        if fatal:
            outcome, reason = classify(fatal=fatal)
            rec.update(outcome=outcome, reason=reason, log_tail="")
            return rec

        if ctx.dashboard is not None:
            (plan.repo / ".memfun" / "dashboard.lock").write_text(
                json.dumps(ctx.dashboard.lock_payload()), encoding="utf-8"
            )
            forwarder = EventForwarder(plan.repo / ".memfun" / "memfun.db", ctx.dashboard.port)
            forwarder.start()

        cmd = swarm_cmd(ctx.memfun_cmd, episode_spec(bundle), ctx.max_agents, ctx.task_timeout)
        log_path = plan.ep_dir / "swarm.log"
        proc = run_in_group(
            cmd,
            cwd=plan.repo,
            env=child_env(plan.home),
            log_path=log_path,
            timeout=ctx.episode_timeout,
            grace=ctx.kill_grace,
            on_start=ctx.track,
            on_end=ctx.untrack,
        )
        rec["returncode"] = proc.returncode
        rec["timed_out"] = proc.timed_out
        if forwarder is not None:
            forwarder.finish()
            rec["dashboard"] = {"forwarded": forwarder.forwarded, "post_errors": forwarder.errors}
            if forwarder.errors:
                rec["warnings"].append(f"dashboard: {forwarder.errors} event POSTs failed")
            forwarder = None
        if ctx.aborted:
            rec.update(outcome="harness_error", reason="run aborted (interrupted)")
            return rec

        events = [e for _, e in read_events(plan.repo / ".memfun" / "memfun.db")]
        metrics = event_metrics(events)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        signals = log_signals(log_text)
        rec["warnings"].extend(signals["warnings"])
        server_errors = len(signals["server_error_lines"]) + sum(
            1 for err in metrics["llm_errors"] if SERVER_ERROR_RE.search(err)
        )
        rec["server_errors"] = server_errors
        rec["server_error_lines"] = signals["server_error_lines"]
        rec["produced_result"] = signals["produced_result"]

        collected = collect_owned(bundle, plan.repo, plan.ep_dir / "collected")
        rec["collection"] = collected
        if (
            bundle.kind == "fanout"
            and max(metrics["primary_sub_tasks"], len(collected["worktrees"])) <= 1
        ):
            rec["flags"].append("decomposition_fallback")
        if any(u["duplicate_work"] for u in collected["units"].values()):
            rec["flags"].append("duplicate_work")
        if collected["tests_modified"]:
            rec["flags"].append("tests_modified")

        grade_error = ""
        try:
            grade = grade_candidate(
                bundle, plan.ep_dir / "collected", plan.ep_dir, ctx.grader_timeout
            )
            grade_error = grade["error"]
            (plan.ep_dir / "grader.txt").write_text(grade["output_tail"], encoding="utf-8")
        except OSError:
            raise
        except Exception as exc:
            grade = None
            grade_error = f"grader crashed: {type(exc).__name__}: {exc}"
        if grade is not None and grade["grader_timeout"]:
            rec["flags"].append("grader_timeout")
        rec["grade"] = grade

        units: dict[str, Any] = {}
        for unit in bundle.units:
            info = collected["units"][unit.id]
            passed = bool(grade and grade["units"].get(unit.id))
            status = "no_output" if not info["output"] else ("pass" if passed else "fail")
            stats = (grade or {}).get("by_file", {}).get(unit.module, {"passed": 0, "failed": 0})
            units[unit.id] = {**info, "status": status, "tests": stats}
        rec["units"] = units

        outcome, reason = classify(
            timed_out=proc.timed_out,
            timeout_s=ctx.episode_timeout,
            returncode=proc.returncode,
            produced_result=signals["produced_result"],
            grade_error=grade_error,
            hidden_passed=grade["hidden_passed"] if grade else 0,
            all_pass=bool(grade and grade["all_pass"]),
            server_errors=server_errors,
        )
        rec.update(
            outcome=outcome,
            reason=reason,
            metrics=metrics,
            log_tail=_tail(log_path) if outcome != "pass" else "",
        )
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
        ctx.leave()
        rec["wall_s"] = round(time.time() - t0, 3)
        rec.setdefault("log_tail", _tail(plan.ep_dir / "swarm.log"))
        with contextlib.suppress(OSError):
            (plan.ep_dir / "episode.json").write_text(
                json.dumps(rec, indent=2, default=str), encoding="utf-8"
            )


# ── Aggregation and summary ──────────────────────────────────────────────────────


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _units_of(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return cast("dict[str, dict[str, Any]]", episode.get("units") or {})


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
            "flags": sorted({f for e in eps for f in e.get("flags", [])}),
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
        "--bundles",
        nargs="+",
        default=list(DEFAULT_BUNDLES),
        help="bundle ids (space or comma separated); default: all 6",
    )
    ap.add_argument("--bundles-root", type=Path, default=SMOKE_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-agents", type=int, default=4, help="memfun swarm --max-pool")
    ap.add_argument("--parallel-tasks", type=int, default=1, help="episodes run at once")
    ap.add_argument("--episode-timeout", type=float, default=2400.0, help="hard cap, seconds")
    ap.add_argument("--task-timeout", type=int, default=900, help="memfun swarm --timeout")
    ap.add_argument("--grader-timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=8192, help="[llm] max_tokens")
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
    args = ap.parse_args(argv)
    args.bundles = [b for item in args.bundles for b in item.split(",") if b]
    return args


def _validate_args(args: argparse.Namespace) -> None:
    args.base_url = check_base_url(args.base_url)
    check_port(args.dashboard_port, "--dashboard-port")
    for name in ("trials", "max_agents", "parallel_tasks", "task_timeout", "max_tokens"):
        if getattr(args, name) < 1:
            raise UsageError(f"--{name.replace('_', '-')} must be >= 1")
    if args.episode_timeout <= 0 or args.grader_timeout <= 0:
        raise UsageError("timeouts must be > 0")
    if args.metrics_url and url_port(args.metrics_url) in FORBIDDEN_PORTS:
        raise UsageError("refusing --metrics-url on a forbidden port")
    args.run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    if not RUN_ID_RE.match(args.run_id):
        raise UsageError(f"--run-id must match {RUN_ID_RE.pattern}")
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
    out = [
        "DRY PLAN (nothing is created, started or contacted)",
        f"run id        : {args.run_id}",
        f"run dir       : {run_dir}",
        f"model server  : {args.base_url}  (port {url_port(args.base_url)}; 8434 is refused)",
        f"model         : {model}",
        f"memfun        : {shlex.join(memfun_cmd)}",
        f"parallel      : {args.parallel_tasks} episode(s) at once, --max-pool {args.max_agents}",
        f"timeouts      : episode {args.episode_timeout:g}s (process-group kill), "
        f"memfun --timeout {args.task_timeout}s, grader {args.grader_timeout:g}s",
    ]
    if args.no_dashboard:
        out.append("dashboard     : disabled (each memfun swarm may start its own)")
    else:
        out.append(
            f"dashboard     : ONE on port {args.dashboard_port}: reuse it if a memfun dashboard "
            f"answers there, else start `{shlex.join([*memfun_cmd, 'dashboard', '--remote', '--port', str(args.dashboard_port)])}`"  # noqa: E501
            f" in {run_dir / '_dashboard'}; each episode gets .memfun/dashboard.lock -> it, "
            "and the runner forwards each episode's events to POST /api/event"
        )
    out.append(f"child env     : {', '.join(ENV_ALLOW)}, HOME=<episode>/home, {DUMMY_KEY_ENV}")
    out.append(f"episodes      : {len(plans)}")
    for p in plans:
        spec = episode_spec(p.bundle)
        first = spec.strip().splitlines()[0][:60] if spec.strip() else ""
        cmd = swarm_cmd(memfun_cmd, f"<spec: {first}...>", args.max_agents, args.task_timeout)
        problem = f"  !! {'; '.join(p.bundle.problems)}" if p.bundle.problems else ""
        out.append(f"  {p.name:<18} kind={p.bundle.kind:<8} units={len(p.bundle.units)}{problem}")
        out.append(f"      repo : {p.repo}")
        out.append(f"      cmd  : (cd repo) {shlex.join(cmd)}")
        out.append(f"      grade: hidden/ of {p.bundle.path} from {p.ep_dir / 'grading'}")
    return "\n".join(out)


def _install_sigterm() -> Any:
    """SIGTERM -> SystemExit in the main thread, so the abort path runs."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def _on_term(_signum: int, _frame: object) -> None:
        raise SystemExit(143)

    return signal.signal(signal.SIGTERM, _on_term)


def _restore_sigterm(previous: Any) -> None:
    if previous is not None:
        signal.signal(signal.SIGTERM, previous)


def run(args: argparse.Namespace) -> int:
    run_dir = (args.out_dir / args.run_id).resolve()
    plans = plan_episodes(args, run_dir)
    if args.dry_plan:
        print(dry_plan(args, run_dir, plans))
        return 0
    if run_dir.exists():
        raise UsageError(f"run dir already exists (pick another --run-id): {run_dir}")
    run_dir.mkdir(parents=True)
    memfun_cmd = shlex.split(args.memfun_cmd)
    t0 = time.time()

    model, fatal = preflight(args.base_url, args.model)
    usage_before = server_usage_snapshot(args.metrics_url) if args.metrics_url else None
    dashboard: SharedDashboard | None = None
    if not args.no_dashboard and not fatal:
        dashboard = start_dashboard(memfun_cmd, args.dashboard_port, run_dir / "_dashboard")
        if dashboard.error:
            fatal = dashboard.error
            dashboard = None
        else:
            how = "started" if dashboard.owned else "reusing"
            print(f"dashboard: {how} http://localhost:{dashboard.port}", flush=True)
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
    )
    episodes: list[dict[str, Any]] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_tasks)
    previous_term = _install_sigterm()
    futures = [pool.submit(run_episode, p, ctx) for p in plans]
    interrupted = False
    try:
        for fut in concurrent.futures.as_completed(futures):
            ep = fut.result()
            print(f"  {ep['episode']:<18} {ep['outcome']:<14} {ep['reason'][:80]}", flush=True)
            episodes.append(ep)
    except (KeyboardInterrupt, SystemExit):
        # Ctrl-C / SIGTERM: the swarms run in their own sessions, so they would outlive
        # us.  Kill exactly the groups we started, drop queued episodes, keep the rest.
        interrupted = True
        print("interrupted: stopping running episodes", file=sys.stderr, flush=True)
        ctx.abort()
        pool.shutdown(wait=True, cancel_futures=True)
        seen = {id(e) for e in episodes}
        for fut in futures:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                ep = fut.result()
                if id(ep) not in seen:
                    episodes.append(ep)
    finally:
        pool.shutdown(wait=True)
        _restore_sigterm(previous_term)
        if dashboard is not None:
            dashboard.stop()
    episodes.sort(key=lambda e: (e["bundle"], e["trial"]))
    usage_after = server_usage_snapshot(args.metrics_url) if args.metrics_url else None
    server_usage: dict[str, Any] | None = None
    if args.metrics_url:
        server_usage = {"available": bool(usage_before and usage_after)}
        if usage_before and usage_after:
            server_usage.update(
                prompt=usage_after["prompt"] - usage_before["prompt"],
                completion=usage_after["completion"] - usage_before["completion"],
            )
    meta: dict[str, Any] = {
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
        "peak_parallel_episodes": ctx.peak_parallel,
        "server_usage": server_usage,
    }
    results = aggregate(episodes, meta)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
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
