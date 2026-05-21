"""Live dashboard server for multi-agent coordination.

Shows workflow requests, task flow, event stream, and worker status.
Can run standalone (Redis) or embedded in the memfun process (any backend).

Usage (standalone)::

    python -m memfun_cli.dashboard.server --redis-url redis://localhost:6379 --port 8080

Usage (embedded — auto-started by memfun chat)::

    from memfun_cli.dashboard.server import create_app
    app = create_app(event_bus=runtime.event_bus, project_name="my-project")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from memfun_core.logging import get_logger

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError as _fastapi_err:
    FastAPI = WebSocket = WebSocketDisconnect = HTMLResponse = None  # type: ignore[assignment, misc]
    _fastapi_err_msg = str(_fastapi_err)

# Suppress websockets/uvicorn DeprecationWarning before they get imported.
# These fire when a browser connects to the dashboard WebSocket.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"uvicorn")
warnings.filterwarnings("ignore", message=".*websockets.*", category=DeprecationWarning)

logger = get_logger("dashboard")

# ── Module-level state ───────────────────────────────────────
_redis_url: str = "redis://localhost:6379"
_project_name: str = ""
_connections: set[Any] = set()  # WebSocket connections
_events: list[dict] = []  # Recent events buffer (max 500)
_workers: dict[str, dict] = {}  # worker_id -> {agent_name, status, last_seen}
_tasks: dict[str, dict] = {}  # task_id -> {agent_name, status, worker_id, ...}
_requests: dict[str, dict] = {}  # workflow_id -> {description, status, ts, ...}
_event_bus: Any | None = None  # Shared event bus (set by create_app)
_project_root: Path | None = None  # Project dir for state persistence

MAX_EVENTS = 2000  # Sized for llm.call frequency (10-30x lifecycle events)
_save_counter: int = 0  # debounce saves (every 3 events)


EVENT_TOPIC = "memfun.distributed.events"


def _state_path() -> Path | None:
    """Path to the persisted dashboard state file."""
    if _project_root is None:
        return None
    return _project_root / ".memfun" / "dashboard.json"


def _save_state(*, force: bool = False) -> None:
    """Persist dashboard state to disk (debounced, every 3rd event)."""
    global _save_counter
    _save_counter += 1
    if not force and _save_counter % 3 != 0:
        return
    path = _state_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "requests": _requests,
            "tasks": _tasks,
            "workers": _workers,
            "events": _events[-MAX_EVENTS:],
        }
        path.write_text(json.dumps(data, default=str), "utf-8")
    except Exception:
        logger.debug("Failed to save dashboard state", exc_info=True)


def _load_state() -> None:
    """Load persisted dashboard state from disk."""
    path = _state_path()
    if path is None or not path.is_file():
        return
    try:
        data = json.loads(path.read_text("utf-8"))
        _requests.update(data.get("requests", {}))
        _tasks.update(data.get("tasks", {}))
        # Don't restore workers — they're per-session
        _events.extend(data.get("events", []))
        # Trim to max
        while len(_events) > MAX_EVENTS:
            _events.pop(0)
        logger.info(
            "Restored dashboard state: %d requests, %d tasks, %d events",
            len(_requests), len(_tasks), len(_events),
        )
    except Exception:
        logger.debug("Failed to load dashboard state", exc_info=True)


def _lockfile_path() -> Path | None:
    """Path to the dashboard lockfile."""
    if _project_root is None:
        return None
    return _project_root / ".memfun" / "dashboard.lock"


def _write_lockfile(port: int) -> None:
    """Write lockfile with port and PID."""
    path = _lockfile_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"port": port, "pid": os.getpid()}
        path.write_text(json.dumps(data), "utf-8")
    except Exception:
        logger.debug("Failed to write dashboard lockfile", exc_info=True)


def _remove_lockfile() -> None:
    """Remove lockfile on shutdown."""
    path = _lockfile_path()
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink()
    except Exception:
        logger.debug("Failed to remove dashboard lockfile", exc_info=True)


def read_lockfile(project_root: Path) -> dict | None:
    """Read and validate the dashboard lockfile for a project.

    Returns ``{"port": int, "pid": int}`` if the lockfile exists and the
    owning process is still alive, otherwise ``None`` (stale lockfiles
    are cleaned up automatically).
    """
    path = project_root / ".memfun" / "dashboard.lock"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
        pid = data.get("pid", 0)
        port = data.get("port", 0)
        if not pid or not port:
            return None
        # Check if the owning process is still alive
        os.kill(pid, 0)
        return {"port": port, "pid": pid}
    except ProcessLookupError:
        # Stale lockfile — owning process is dead
        logger.info("Removing stale dashboard lockfile (pid %s)", data.get("pid"))
        with _suppress_os():
            path.unlink()
        return None
    except Exception:
        return None


def _suppress_os():
    """Contextmanager to suppress OSError."""
    import contextlib

    return contextlib.suppress(OSError)


_dashboard_port: int = 0  # set by create_app lifespan


async def _event_listener(bus: Any) -> None:
    """Subscribe to any event bus and broadcast to WebSocket clients."""
    global _connections
    from memfun_runtime.distributed import EVENT_TOPIC as _TOPIC
    from memfun_runtime.distributed import DistributedEvent

    logger.info("Dashboard event listener started")

    async for msg in bus.subscribe(_TOPIC):
        try:
            event = DistributedEvent.from_bytes(msg.payload)
            event_dict = {
                "event_type": event.event_type,
                "task_id": event.task_id,
                "agent_name": event.agent_name,
                "worker_id": event.worker_id,
                "success": event.success,
                "duration_ms": event.duration_ms,
                "detail": event.detail,
                "ts": event.ts,
                "project": event.project,
                "workflow_id": event.workflow_id,
                # ── LLM-call telemetry (only set for event_type=llm.call) ──
                "model": event.model,
                "prompt_preview": event.prompt_preview,
                "completion_preview": event.completion_preview,
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
                "error": event.error,
            }

            _update_state(event_dict)

            _events.append(event_dict)
            if len(_events) > MAX_EVENTS:
                _events.pop(0)

            message = json.dumps({"type": "event", "data": event_dict})
            dead = set()
            for ws in _connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            _connections -= dead

        except Exception as exc:
            logger.debug("Bad event: %s", exc)


async def _redis_listener() -> None:
    """Standalone mode: create a Redis event bus and listen."""
    from memfun_runtime.backends.redis.event_bus import RedisEventBus

    bus = await RedisEventBus.create(_redis_url)
    logger.info("Dashboard listening on Redis at %s", _redis_url)
    await _event_listener(bus)


def _update_state(event: dict) -> None:
    """Update worker, task, and request state from an event."""
    etype = event.get("event_type", "")
    worker_id = event.get("worker_id")
    task_id = event.get("task_id")
    agent_name = event.get("agent_name")
    workflow_id = event.get("workflow_id")

    # ── Request tracking ────────────────────────────────
    if etype == "workflow.started" and workflow_id:
        _requests[workflow_id] = {
            "workflow_id": workflow_id,
            "description": event.get("detail", ""),
            "status": "running",
            "ts": event.get("ts", time.time()),
            "project": event.get("project", ""),
            "task_count": 0,
            "completed_count": 0,
        }

    # Associate tasks with their workflow request
    if workflow_id and task_id:
        if etype == "task.published":
            if workflow_id in _requests:
                _requests[workflow_id]["task_count"] = (
                    _requests[workflow_id].get("task_count", 0) + 1
                )
        elif etype == "task.completed" and workflow_id in _requests:
            _requests[workflow_id]["completed_count"] = (
                _requests[workflow_id].get("completed_count", 0) + 1
            )
            req = _requests[workflow_id]
            if req["completed_count"] >= req["task_count"] > 0:
                req["status"] = "completed"

    # ── Worker tracking ─────────────────────────────────
    if etype == "worker.online" and worker_id:
        _workers[worker_id] = {
            "agent_name": agent_name,
            "status": "idle",
            "last_seen": time.time(),
            "tasks_done": 0,
        }
    elif etype == "worker.offline" and worker_id:
        _workers.pop(worker_id, None)

    # ── Task tracking ───────────────────────────────────
    if etype == "task.published" and task_id:
        _tasks[task_id] = {
            "agent_name": agent_name,
            "status": "pending",
            "detail": event.get("detail", ""),
            "ts": event.get("ts", time.time()),
            "workflow_id": workflow_id,
        }
    elif etype == "task.picked_up" and task_id:
        if task_id in _tasks:
            _tasks[task_id]["status"] = "running"
            _tasks[task_id]["worker_id"] = worker_id
        if worker_id and worker_id in _workers:
            _workers[worker_id]["status"] = "busy"
            _workers[worker_id]["last_seen"] = time.time()
    elif etype == "task.completed" and task_id:
        if task_id in _tasks:
            _tasks[task_id]["status"] = (
                "completed" if event.get("success") else "failed"
            )
            _tasks[task_id]["duration_ms"] = event.get("duration_ms")
        if worker_id and worker_id in _workers:
            _workers[worker_id]["status"] = "idle"
            _workers[worker_id]["tasks_done"] = (
                _workers[worker_id].get("tasks_done", 0) + 1
            )
            _workers[worker_id]["last_seen"] = time.time()

    # Persist state to disk (debounced)
    _save_state()


def _get_state() -> dict:
    """Current snapshot for newly connected clients."""
    return {
        "type": "state",
        "workers": _workers,
        "tasks": _tasks,
        "events": _events[-50:],
        "requests": _requests,
        "project": _project_name,
    }


# ── FastAPI app ───────────────────────────────────────────────────


def create_app(
    event_bus: Any | None = None,
    project_name: str = "",
) -> FastAPI:
    """Create the FastAPI app for the dashboard.

    Args:
        event_bus: Shared event bus (any backend). If None, falls back
                   to connecting to Redis (standalone mode).
        project_name: Current project name shown in the header.
    """
    if FastAPI is None:
        logger.error("Install fastapi: pip install fastapi uvicorn")
        raise ImportError(_fastapi_err_msg)

    global _event_bus, _project_name, _project_root
    _event_bus = event_bus
    _project_name = project_name or ""
    _project_root = Path.cwd()

    _bg_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        _load_state()
        if _dashboard_port:
            _write_lockfile(_dashboard_port)
        if _event_bus is not None:
            task = asyncio.create_task(_event_listener(_event_bus))
        else:
            task = asyncio.create_task(_redis_listener())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        yield
        _save_state(force=True)  # final flush
        _remove_lockfile()
        task.cancel()

    app = FastAPI(title="Memfun Agent Dashboard", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _DASHBOARD_HTML

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        _connections.add(ws)
        await ws.send_text(json.dumps(_get_state()))
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            _connections.discard(ws)

    @app.get("/api/state")
    async def api_state() -> dict:
        return _get_state()

    @app.get("/api/health")
    async def api_health() -> dict:
        return {"status": "ok", "project": _project_name}

    @app.post("/api/event")
    async def api_ingest_event(body: dict) -> dict:
        """Ingest an event from an external memfun process."""
        global _connections
        _update_state(body)
        _events.append(body)
        if len(_events) > MAX_EVENTS:
            _events.pop(0)
        # Broadcast to WebSocket clients
        message = json.dumps({"type": "event", "data": body})
        dead: set[Any] = set()
        for ws in _connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        _connections -= dead
        return {"ok": True}

    return app


# ── Dashboard HTML (single-file, no build step) ─────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memfun Agent Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149;
    --yellow: #d29922; --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
    height: 100vh; overflow: hidden;
  }

  .header {
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
    height: 48px; flex-shrink: 0;
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .badge {
    background: var(--accent); color: var(--bg);
    padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 700;
  }
  .header .project-name {
    font-size: 13px; color: var(--muted);
    border-left: 1px solid var(--border);
    padding-left: 16px;
  }
  .header .project-name span {
    color: var(--text); font-weight: 600;
  }
  .header .stats {
    margin-left: auto; display: flex; gap: 16px;
    font-size: 13px; color: var(--muted);
  }
  .header .stats span {
    color: var(--text); font-weight: 600;
  }

  .main-layout {
    display: flex; flex-direction: column;
    height: calc(100vh - 48px);
  }

  /* 3-column grid: Requests | Task Flow + Events | Workers */
  .top-area {
    flex: 1; min-height: 120px;
    display: grid;
    grid-template-columns: 220px 1fr 220px;
    grid-template-rows: 1fr 1fr;
    gap: 1px; background: var(--border);
    overflow: hidden;
  }

  .panel {
    background: var(--surface); overflow: hidden;
    display: flex; flex-direction: column;
  }
  .panel-title {
    padding: 8px 16px; font-size: 11px;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--muted);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .panel-content {
    overflow-y: auto; flex: 1; padding: 0;
  }
  .panel-content.padded {
    padding: 8px;
  }

  /* Request items in left panel */
  .show-all-btn {
    padding: 8px 12px; cursor: pointer;
    color: var(--accent); font-size: 12px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  .show-all-btn:hover { background: rgba(88,166,255,0.08); }
  .show-all-btn.active { background: rgba(88,166,255,0.15); }

  .request-item {
    padding: 8px 12px; cursor: pointer;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; font-size: 12px;
  }
  .request-item:hover { background: rgba(255,255,255,0.04); }
  .request-item.active { background: var(--accent); color: #fff; }
  .request-item .req-time {
    font-size: 10px; color: var(--muted);
    margin-right: 6px;
  }
  .request-item.active .req-time { color: rgba(255,255,255,0.7); }
  .request-item .req-status {
    float: right; font-size: 10px;
    padding: 1px 5px; border-radius: 8px;
  }
  .request-item .req-status.running {
    background: rgba(210,153,34,0.2); color: var(--yellow);
  }
  .request-item .req-status.completed {
    background: rgba(63,185,80,0.2); color: var(--green);
  }
  .request-item.active .req-status {
    background: rgba(255,255,255,0.2); color: #fff;
  }

  /* Workers panel (right) */
  .worker {
    padding: 6px 10px; border-radius: 6px;
    margin: 3px 8px;
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.03);
  }
  .worker .dot {
    width: 8px; height: 8px;
    border-radius: 50%; flex-shrink: 0;
  }
  .worker .dot.idle {
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
  }
  .worker .dot.busy {
    background: var(--yellow);
    box-shadow: 0 0 6px var(--yellow);
    animation: pulse 1.5s infinite;
  }
  .worker .dot.offline { background: var(--muted); }
  .worker .name { font-size: 12px; font-weight: 500; }
  .worker .info {
    font-size: 11px; color: var(--muted);
    margin-left: auto;
  }

  /* Task DAG */
  .task-flow { padding: 8px; }
  .task-node {
    display: inline-flex; align-items: center;
    gap: 6px; padding: 4px 10px;
    border-radius: 6px; margin: 3px;
    font-size: 11px;
    border: 1px solid var(--border);
  }
  .task-node.pending {
    border-color: var(--muted); color: var(--muted);
  }
  .task-node.running {
    border-color: var(--yellow);
    color: var(--yellow);
    animation: pulse 1.5s infinite;
  }
  .task-node.completed {
    border-color: var(--green); color: var(--green);
  }
  .task-node.failed {
    border-color: var(--red); color: var(--red);
  }
  .task-arrow { color: var(--muted); margin: 0 2px; }

  /* Event feed */
  .event {
    padding: 4px 10px; font-size: 11px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    display: flex; gap: 6px; align-items: baseline;
  }
  .event .time {
    color: var(--muted); font-size: 10px;
    flex-shrink: 0; width: 72px;
  }
  .event .type {
    font-weight: 600; width: 110px; flex-shrink: 0;
  }
  .event .type.published { color: var(--accent); }
  .event .type.picked_up { color: var(--yellow); }
  .event .type.completed { color: var(--green); }
  .event .type.failed { color: var(--red); }
  .event .type.online { color: var(--purple); }
  .event .type.offline { color: var(--muted); }
  .event .type.started { color: var(--purple); }
  .event .detail {
    color: var(--muted); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }

  /* Tab bar (Events / Logs / Stats) */
  .tab-bar {
    display: flex; gap: 4px;
    padding: 4px 8px;
    align-items: stretch;
  }
  .tab-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    font-family: inherit;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 6px 12px;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }
  .tab-btn:hover {
    background: rgba(255,255,255,0.04);
    color: var(--text);
  }
  .tab-btn.active {
    color: var(--text);
    font-weight: 600;
    border-bottom: 2px solid var(--accent);
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* Logs tab */
  .log-line {
    padding: 3px 10px;
    font-size: 11px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    display: flex; gap: 8px; align-items: baseline;
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }
  .log-ts { color: var(--muted); flex-shrink: 0; }
  .log-task {
    color: var(--accent);
    font-weight: 600;
    flex-shrink: 0;
  }
  .log-type {
    font-weight: 600;
    flex-shrink: 0;
    width: 96px;
  }
  .log-type.type-task-picked_up { color: var(--yellow); }
  .log-type.type-task-completed { color: var(--green); }
  .log-type.type-task-failed { color: var(--red); }
  .log-type.type-workflow-started { color: var(--purple); }
  .log-type.type-workflow-completed { color: var(--purple); }
  .log-type.type-llm-call { color: var(--accent); }
  .log-detail {
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .log-hint {
    padding: 12px 16px;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
    font-style: italic;
  }
  /* Logs toolbar */
  .logs-toolbar {
    display: flex; gap: 12px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,0.02);
    font-size: 11px;
    align-items: center;
    flex-shrink: 0;
  }
  .logs-filter-label {
    display: flex; align-items: center; gap: 4px;
    cursor: pointer; color: var(--muted);
    user-select: none;
  }
  .logs-filter-label input[type="checkbox"] {
    cursor: pointer;
    accent-color: var(--accent);
  }
  .logs-filter-label span {
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }
  /* Two-line LLM-call log entry */
  .log-line.llm {
    display: block;
    padding: 4px 10px;
    white-space: normal;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .log-line.llm .log-header {
    display: flex; gap: 8px;
    align-items: baseline;
    font-size: 11px;
  }
  .log-line.llm .log-llm-meta {
    color: var(--muted);
    font-size: 10px;
    margin-left: 8px;
  }
  .log-line.llm .log-llm-text {
    padding: 2px 0 0 22px;
    font-size: 11px;
    line-height: 1.4;
  }
  .log-line.llm .log-llm-prompt { color: var(--muted); }
  .log-line.llm .log-llm-completion { color: var(--text); }
  .log-line.llm .log-arrow { color: var(--accent); }

  /* Stats tab */
  .stats-wrap { padding: 12px 16px; }
  .stats-list {
    display: grid;
    grid-template-columns: 1fr auto;
    column-gap: 16px;
    row-gap: 6px;
    font-size: 12px;
    margin-bottom: 16px;
  }
  .stats-list dt {
    color: var(--muted);
  }
  .stats-list dd {
    color: var(--text);
    font-weight: 600;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* Stats: timeseries sparkline cards */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 10px;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 100px;
    transition: border-color 0.15s ease;
  }
  .metric-card:hover { border-color: var(--accent); }
  .metric-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 6px;
  }
  .metric-title {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }
  .metric-value {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }
  .spark {
    display: block;
    width: 100%;
    height: 60px;
  }
  .spark-empty {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 18px;
  }
  .stats-section-title {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    margin: 8px 0 6px;
  }
  .stats-evtypes {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  .stats-evtypes td {
    padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .stats-evtypes td.evtype-name { color: var(--muted); }
  .stats-evtypes td.evtype-count {
    text-align: right;
    color: var(--text);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .empty {
    padding: 20px; text-align: center;
    color: var(--muted); font-size: 12px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Memfun</h1>
  <div class="badge">LIVE</div>
  <div class="project-name" id="project-label"></div>
  <div class="stats">
    Workers: <span id="stat-workers">0</span>
    &nbsp;|&nbsp;
    Tasks: <span id="stat-tasks">0</span>
    &nbsp;|&nbsp;
    Resolved: <span id="stat-resolved">0</span>
  </div>
</div>

<div class="main-layout">
  <div class="top-area">
    <!-- Left: Requests (spans both rows) -->
    <div class="panel" style="grid-row: span 2">
      <div class="panel-title">Requests</div>
      <div class="panel-content">
        <div class="show-all-btn active" id="show-all-btn">Show All</div>
        <div id="request-list">
          <div class="empty">No requests yet</div>
        </div>
      </div>
    </div>

    <!-- Middle top: Task Flow -->
    <div class="panel">
      <div class="panel-title">Task Flow</div>
      <div class="panel-content padded" id="tasks">
        <div class="empty">No tasks yet</div>
      </div>
    </div>

    <!-- Right: Workers (spans both rows) -->
    <div class="panel" style="grid-row: span 2">
      <div class="panel-title">Workers</div>
      <div class="panel-content padded" id="workers">
        <div class="empty">Waiting for workers...</div>
      </div>
    </div>

    <!-- Middle bottom: Event Stream / Logs / Stats (tabbed) -->
    <div class="panel">
      <div class="panel-title tab-bar" role="tablist">
        <button class="tab-btn active" data-tab="events" type="button">Events</button>
        <button class="tab-btn" data-tab="logs" type="button">Logs</button>
        <button class="tab-btn" data-tab="stats" type="button">Stats</button>
      </div>
      <div class="panel-content tab-pane active" id="events"></div>
      <div class="panel-content tab-pane" id="logs">
        <div class="logs-toolbar">
          <label class="logs-filter-label">
            <input type="checkbox" id="logs-filter-lifecycle"
              class="logs-filter" data-filter="lifecycle" checked>
            <span>Lifecycle</span>
          </label>
          <label class="logs-filter-label">
            <input type="checkbox" id="logs-filter-llm"
              class="logs-filter" data-filter="llm" checked>
            <span>LLM calls</span>
          </label>
        </div>
        <div id="logs-body"></div>
      </div>
      <div class="panel-content tab-pane" id="stats"></div>
    </div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────
const ws = new WebSocket(`ws://${location.host}/ws`);
const workersEl = document.getElementById('workers');
const tasksEl = document.getElementById('tasks');
const eventsEl = document.getElementById('events');
const logsEl = document.getElementById('logs');
const statsEl = document.getElementById('stats');
const requestListEl = document.getElementById('request-list');
const showAllBtn = document.getElementById('show-all-btn');
const projectLabel = document.getElementById('project-label');

let workers = {};
let tasks = {};
let requests = {};
let allEvents = [];
let activeRequest = null;  // null = show all
let resolved = 0;

// Tabbed right-column state (shared across tabs).
const LIFECYCLE_LOG_TYPES = new Set([
  'task.picked_up', 'task.completed', 'task.failed',
  'workflow.started', 'workflow.completed',
]);
const LLM_LOG_TYPES = new Set(['llm.call']);
// Buffer cap raised — llm.call events are 10-30x more frequent than
// lifecycle events, so 500 fills up in seconds during a swarm run.
const MAX_LOGS = 2000;
let logs = [];                 // newest first
let eventCounts = {};          // event_type -> count
let workflowStart = null;      // first event ts (epoch seconds)
let activeTab = 'events';
// Logs tab filters (default both on).  Read directly off DOM in
// renderLogs() so hot-reloading state isn't required.
let logsFilterLifecycle = true;
let logsFilterLlm = true;

// ── Timeseries state (Stats tab sparkline cards) ──────────
// 10-second buckets, keep the last 30 buckets (5 minutes total).
const SPARK_BUCKET_SEC = 10;
const SPARK_BUCKETS = 30;
const SPARK_WINDOW_SEC = SPARK_BUCKET_SEC * SPARK_BUCKETS;

// Per-metric circular buffers.  Each entry is { ts, value }.
// Counters accumulate per bucket; gauges/avgs use { sum, count }.
const sparkSeries = {
  llmCallsPerMin: [],
  tokensInPerMin: [],
  tokensOutPerMin: [],
  tasksDonePerMin: [],
  agentsBusy: [],
  avgLatencySec: [],
};
// Per-metric "currently-accumulating" bucket state, keyed by bucket
// start timestamp.  Reset when the bucket rolls over.
const sparkAccum = {
  llmCallsPerMin: { ts: 0, count: 0 },
  tokensInPerMin: { ts: 0, sum: 0 },
  tokensOutPerMin: { ts: 0, sum: 0 },
  tasksDonePerMin: { ts: 0, count: 0 },
  agentsBusy: { ts: 0, value: 0 },        // gauge — sampled
  avgLatencySec: { ts: 0, sum: 0, count: 0 },
};

// ── WebSocket ─────────────────────────────────────────────
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'state') {
    workers = msg.workers || {};
    tasks = msg.tasks || {};
    requests = msg.requests || {};
    if (msg.project) {
      projectLabel.innerHTML = '<span>' + escHtml(msg.project) + '</span>';
    }
    resolved = Object.values(tasks)
      .filter(t => t.status === 'completed').length;
    allEvents = [];
    logs = [];
    eventCounts = {};
    workflowStart = null;
    (msg.events || []).forEach(ev => {
      allEvents.push(ev);
      ingestForTabs(ev);
    });
    renderAll();
  } else if (msg.type === 'event') {
    handleEvent(msg.data);
  }
};

// Update logs / counts / workflowStart from a single event.
function ingestForTabs(ev) {
  const et = ev.event_type || '';
  eventCounts[et] = (eventCounts[et] || 0) + 1;
  if (workflowStart === null && ev.ts) workflowStart = ev.ts;
  if (LIFECYCLE_LOG_TYPES.has(et) || LLM_LOG_TYPES.has(et)) {
    logs.unshift(ev);
    if (logs.length > MAX_LOGS) logs.length = MAX_LOGS;
  }
  ingestSpark(ev);
}

// ── Timeseries helpers ────────────────────────────────────
// Compute the start timestamp (epoch seconds) of the bucket
// containing ``ts`` for the configured bucket size.
function bucketStart(ts) {
  if (!ts) ts = Date.now() / 1000;
  return Math.floor(ts / SPARK_BUCKET_SEC) * SPARK_BUCKET_SEC;
}

// Push a finalised bucket value into a series buffer, capping at
// SPARK_BUCKETS most-recent entries.  ``ts`` is the bucket start.
function pushBucket(name, ts, value) {
  const arr = sparkSeries[name];
  if (!arr) return;
  arr.push({ ts: ts, value: value });
  while (arr.length > SPARK_BUCKETS) arr.shift();
}

// Flush whatever's currently in ``sparkAccum[name]`` into the
// series, then start a fresh accumulator at ``newTs``.
function rollBucket(name, newTs) {
  const acc = sparkAccum[name];
  if (!acc) return;
  if (acc.ts !== 0 && acc.ts !== newTs) {
    let value = 0;
    if (name === 'llmCallsPerMin' || name === 'tasksDonePerMin') {
      // counter -> rate per minute
      value = (acc.count || 0) * (60 / SPARK_BUCKET_SEC);
    } else if (name === 'tokensInPerMin' || name === 'tokensOutPerMin') {
      // sum -> rate per minute
      value = (acc.sum || 0) * (60 / SPARK_BUCKET_SEC);
    } else if (name === 'agentsBusy') {
      // gauge — last sampled value
      value = acc.value || 0;
    } else if (name === 'avgLatencySec') {
      value = acc.count > 0 ? (acc.sum / acc.count) : 0;
    }
    pushBucket(name, acc.ts, value);
  }
  // Reset accumulator for the new bucket.
  if (name === 'llmCallsPerMin' || name === 'tasksDonePerMin') {
    sparkAccum[name] = { ts: newTs, count: 0 };
  } else if (name === 'tokensInPerMin' || name === 'tokensOutPerMin') {
    sparkAccum[name] = { ts: newTs, sum: 0 };
  } else if (name === 'agentsBusy') {
    sparkAccum[name] = { ts: newTs, value: 0 };
  } else if (name === 'avgLatencySec') {
    sparkAccum[name] = { ts: newTs, sum: 0, count: 0 };
  }
}

// Ensure every accumulator is rolled forward to the current bucket
// so the sparklines refresh even when no event arrives.
function advanceSparkBuckets() {
  const now = bucketStart();
  Object.keys(sparkAccum).forEach(name => {
    if (sparkAccum[name].ts !== now) {
      rollBucket(name, now);
    }
  });
  // Re-sample the gauge (agents currently busy) every tick.
  try {
    const busy = Object.values(workers)
      .filter(w => w && w.status === 'busy').length;
    sparkAccum.agentsBusy.value = busy;
  } catch (_e) { /* defensive — never crash the dashboard */ }
}

// Ingest a single event into the timeseries accumulators.  Best
// effort — any malformed event is silently ignored so the dashboard
// keeps working even with partial telemetry.
function ingestSpark(ev) {
  try {
    if (!ev) return;
    const ts = ev.ts || (Date.now() / 1000);
    const b = bucketStart(ts);
    const et = ev.event_type || '';

    // LLM calls / min  (fallback: count ALL events when no llm.call
    // events have ever been seen — useful before LLM telemetry lands).
    const isLlm = et === 'llm.call' || et.startsWith('llm.');
    const llmEverSeen = (eventCounts['llm.call'] || 0) > 0
      || Object.keys(eventCounts).some(k => k.startsWith('llm.'));
    if (isLlm || !llmEverSeen) {
      if (sparkAccum.llmCallsPerMin.ts !== b) {
        rollBucket('llmCallsPerMin', b);
      }
      sparkAccum.llmCallsPerMin.count += 1;
    }

    // Tokens in / out — pull from event.usage or top-level fields.
    const usage = ev.usage || ev.detail_usage || {};
    const pin = Number(
      ev.prompt_tokens
        ?? usage.prompt_tokens
        ?? usage.input_tokens
        ?? 0,
    );
    const pout = Number(
      ev.completion_tokens
        ?? usage.completion_tokens
        ?? usage.output_tokens
        ?? 0,
    );
    if (pin > 0) {
      if (sparkAccum.tokensInPerMin.ts !== b) {
        rollBucket('tokensInPerMin', b);
      }
      sparkAccum.tokensInPerMin.sum += pin;
    }
    if (pout > 0) {
      if (sparkAccum.tokensOutPerMin.ts !== b) {
        rollBucket('tokensOutPerMin', b);
      }
      sparkAccum.tokensOutPerMin.sum += pout;
    }

    // Tasks done / min — count successful task.completed only.
    if (et === 'task.completed' && ev.success !== false) {
      if (sparkAccum.tasksDonePerMin.ts !== b) {
        rollBucket('tasksDonePerMin', b);
      }
      sparkAccum.tasksDonePerMin.count += 1;
    }

    // Avg call latency — only events that carry duration_ms.
    const dur = Number(ev.duration_ms || 0);
    if (dur > 0) {
      if (sparkAccum.avgLatencySec.ts !== b) {
        rollBucket('avgLatencySec', b);
      }
      sparkAccum.avgLatencySec.sum += dur / 1000;
      sparkAccum.avgLatencySec.count += 1;
    }
  } catch (_e) {
    // Best effort — keep the dashboard alive.
  }
}

// Build a series snapshot suitable for rendering: finalised buckets
// from sparkSeries PLUS the current accumulator's preliminary value.
function snapshotSeries(name) {
  const arr = (sparkSeries[name] || []).slice();
  const acc = sparkAccum[name];
  if (!acc || !acc.ts) return arr;
  let v = 0;
  if (name === 'llmCallsPerMin' || name === 'tasksDonePerMin') {
    v = (acc.count || 0) * (60 / SPARK_BUCKET_SEC);
  } else if (name === 'tokensInPerMin' || name === 'tokensOutPerMin') {
    v = (acc.sum || 0) * (60 / SPARK_BUCKET_SEC);
  } else if (name === 'agentsBusy') {
    v = acc.value || 0;
  } else if (name === 'avgLatencySec') {
    v = acc.count > 0 ? (acc.sum / acc.count) : 0;
  }
  // Don't duplicate if the most recent finalised bucket has the same ts.
  if (arr.length === 0 || arr[arr.length - 1].ts !== acc.ts) {
    arr.push({ ts: acc.ts, value: v });
  }
  return arr;
}

// Render a single sparkline as inline SVG.  ``points`` is an array
// of { ts, value }.  Returns an HTML string.  ``stroke`` is a CSS
// variable name (e.g. ``--accent``).
function renderSparkSVG(points, stroke) {
  if (!points || points.length === 0) {
    return '<div class="spark-empty">—</div>';
  }
  const w = 100, h = 30, pad = 2;
  const values = points.map(p => p.value);
  const maxV = Math.max.apply(null, values);
  const minV = Math.min.apply(null, values);
  // Avoid divide-by-zero when the series is flat.
  const span = (maxV - minV) > 0 ? (maxV - minV) : 1;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;

  // Distribute points across the full inner width even when the
  // series is shorter than SPARK_BUCKETS — start from the right.
  const xs = points.map((_p, i) => {
    const slot = SPARK_BUCKETS - points.length + i;
    return pad + (slot / (SPARK_BUCKETS - 1)) * innerW;
  });
  const ys = values.map(v => {
    const norm = (v - minV) / span;
    return pad + innerH - norm * innerH;
  });
  const path = xs.map((x, i) =>
    (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + ys[i].toFixed(1),
  ).join(' ');
  // Area fill below the line for visual weight.
  const fillPath = path + ' L' + xs[xs.length - 1].toFixed(1) +
    ' ' + (h - pad).toFixed(1) +
    ' L' + xs[0].toFixed(1) + ' ' + (h - pad).toFixed(1) + ' Z';
  const last = points[points.length - 1];
  const tip = 'value ' + formatSparkValue(last.value, false) +
    ' @ ' + fmtTimeShort(last.ts);
  return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h +
    '" preserveAspectRatio="none">' +
    '<title>' + escHtml(tip) + '</title>' +
    '<path d="' + fillPath + '" fill="var(' + stroke + ')" ' +
      'fill-opacity="0.12" stroke="none"></path>' +
    '<path d="' + path + '" fill="none" stroke="var(' + stroke +
      ')" stroke-width="1.5" stroke-linejoin="round" ' +
      'stroke-linecap="round"></path>' +
    '<circle cx="' + xs[xs.length - 1].toFixed(1) + '" cy="' +
      ys[ys.length - 1].toFixed(1) + '" r="1.6" fill="var(' +
      stroke + ')"></circle>' +
    '</svg>';
}

// Format a metric value for the card header.  ``isInt`` rounds.
function formatSparkValue(v, isInt) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!isFinite(v)) return '—';
  if (isInt) return String(Math.round(v));
  if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// Compute the displayed "current value" for a metric — most recent
// bucket (including the in-progress accumulator).  Returns null
// when the series has no data.
function currentValue(name) {
  const snap = snapshotSeries(name);
  if (snap.length === 0) return null;
  return snap[snap.length - 1].value;
}

// Definition of the six cards shown in the Stats tab grid.
const SPARK_CARDS = [
  {
    name: 'llmCallsPerMin', title: 'LLM calls / min',
    color: '--accent', isInt: false,
  },
  {
    name: 'tokensInPerMin', title: 'Tokens / min (in)',
    color: '--accent', isInt: false,
  },
  {
    name: 'tokensOutPerMin', title: 'Tokens / min (out)',
    color: '--accent', isInt: false,
  },
  {
    name: 'tasksDonePerMin', title: 'Tasks done / min',
    color: '--green', isInt: false,
  },
  {
    name: 'agentsBusy', title: 'Agents busy',
    color: '--yellow', isInt: true,
  },
  {
    name: 'avgLatencySec', title: 'Avg call latency (s)',
    color: '--accent', isInt: false,
  },
];

function renderSparkCards() {
  return '<div class="stats-grid">' +
    SPARK_CARDS.map(card => {
      const snap = snapshotSeries(card.name);
      const cur = currentValue(card.name);
      const valStr = cur === null
        ? '—'
        : formatSparkValue(cur, card.isInt);
      const svg = renderSparkSVG(snap, card.color);
      return '<div class="metric-card">' +
        '<div class="metric-header">' +
          '<span class="metric-title">' + escHtml(card.title) +
            '</span>' +
          '<span class="metric-value">' + escHtml(valStr) +
            '</span>' +
        '</div>' +
        svg +
      '</div>';
    }).join('') +
  '</div>';
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Event handling ────────────────────────────────────────
function handleEvent(ev) {
  const et = ev.event_type;

  // Request tracking
  if (et === 'workflow.started' && ev.workflow_id) {
    requests[ev.workflow_id] = {
      workflow_id: ev.workflow_id,
      description: ev.detail || '',
      status: 'running',
      ts: ev.ts || Date.now() / 1000,
      task_count: 0,
      completed_count: 0,
    };
  }

  // Worker tracking
  if (et === 'worker.online') {
    workers[ev.worker_id] = {
      agent_name: ev.agent_name,
      status: 'idle', tasks_done: 0,
    };
  } else if (et === 'worker.offline') {
    delete workers[ev.worker_id];
  }

  // Task tracking
  if (et === 'task.published') {
    tasks[ev.task_id] = {
      agent_name: ev.agent_name,
      status: 'pending', detail: ev.detail,
      workflow_id: ev.workflow_id,
    };
    if (ev.workflow_id && requests[ev.workflow_id]) {
      requests[ev.workflow_id].task_count =
        (requests[ev.workflow_id].task_count || 0) + 1;
    }
  } else if (et === 'task.picked_up') {
    if (tasks[ev.task_id]) {
      tasks[ev.task_id].status = 'running';
      tasks[ev.task_id].worker_id = ev.worker_id;
    }
    if (workers[ev.worker_id]) {
      workers[ev.worker_id].status = 'busy';
    }
  } else if (et === 'task.completed') {
    if (tasks[ev.task_id]) {
      tasks[ev.task_id].status =
        ev.success ? 'completed' : 'failed';
      tasks[ev.task_id].duration_ms = ev.duration_ms;
    }
    if (ev.worker_id && workers[ev.worker_id]) {
      workers[ev.worker_id].status = 'idle';
      workers[ev.worker_id].tasks_done =
        (workers[ev.worker_id].tasks_done || 0) + 1;
    }
    if (ev.success) resolved++;
    // Update request completion
    const wfid = ev.workflow_id ||
      (tasks[ev.task_id] && tasks[ev.task_id].workflow_id);
    if (wfid && requests[wfid]) {
      requests[wfid].completed_count =
        (requests[wfid].completed_count || 0) + 1;
      if (requests[wfid].completed_count >= requests[wfid].task_count
          && requests[wfid].task_count > 0) {
        requests[wfid].status = 'completed';
      }
    }
  }

  allEvents.push(ev);
  if (allEvents.length > 500) allEvents.shift();

  ingestForTabs(ev);
  renderAll();
}

// ── Filtering ─────────────────────────────────────────────
function matchesFilter(ev) {
  if (!activeRequest) return true;
  return ev.workflow_id === activeRequest;
}

function filteredTasks() {
  if (!activeRequest) return tasks;
  const out = {};
  for (const [tid, t] of Object.entries(tasks)) {
    if (t.workflow_id === activeRequest) out[tid] = t;
  }
  return out;
}

function filteredEvents() {
  return allEvents.filter(matchesFilter);
}

// ── Render ────────────────────────────────────────────────
function renderAll() {
  renderRequests();
  renderWorkers();
  renderTasks();
  renderEvents();
  renderLogs();
  renderStatsPane();
  renderHeaderStats();
}

// Cheap "render only the active right-column pane" used by the
// 1s interval timer.  Events/Logs are timestamp-bearing so the
// "time since" deltas should refresh even when no new event arrives.
function renderActivePane() {
  if (activeTab === 'logs') renderLogs();
  else if (activeTab === 'stats') renderStatsPane();
  // Events tab already reflects allEvents on every WS message; no
  // delta in it changes purely with wall-clock time.
}

function renderRequests() {
  const rkeys = Object.keys(requests);
  if (rkeys.length === 0) {
    requestListEl.innerHTML =
      '<div class="empty">No requests yet</div>';
    return;
  }
  // Sort by timestamp, newest first
  const sorted = rkeys
    .map(k => requests[k])
    .sort((a, b) => (b.ts || 0) - (a.ts || 0));

  requestListEl.innerHTML = sorted.map(r => {
    const isActive = activeRequest === r.workflow_id;
    const t = r.ts
      ? new Date(r.ts * 1000).toLocaleTimeString([], {
          hour: '2-digit', minute: '2-digit',
        })
      : '';
    const desc = escHtml(r.description || r.workflow_id)
      .substring(0, 40);
    const counts = r.task_count
      ? ` (${r.completed_count || 0}/${r.task_count})`
      : '';
    return `<div class="request-item${isActive ? ' active' : ''}"
                data-wfid="${r.workflow_id}">
      <span class="req-time">${t}</span>
      <span class="req-status ${r.status}">${r.status}${counts}</span>
      ${desc}
    </div>`;
  }).join('');

  // Click handlers
  requestListEl.querySelectorAll('.request-item').forEach(el => {
    el.addEventListener('click', () => {
      activeRequest = el.dataset.wfid;
      showAllBtn.classList.remove('active');
      renderAll();
    });
  });
}

function renderWorkers() {
  const wkeys = Object.keys(workers);
  if (wkeys.length === 0) {
    workersEl.innerHTML =
      '<div class="empty">Waiting for workers...</div>';
  } else {
    workersEl.innerHTML = wkeys.map(wid => {
      const w = workers[wid];
      return `<div class="worker">
        <div class="dot ${w.status}"></div>
        <span class="name">${escHtml(w.agent_name || wid)}</span>
        <span class="info">${w.status} (${w.tasks_done || 0})</span>
      </div>`;
    }).join('');
  }
}

function renderTasks() {
  const ft = filteredTasks();
  const tkeys = Object.keys(ft);
  if (tkeys.length === 0) {
    tasksEl.innerHTML =
      '<div class="empty">No tasks yet</div>';
  } else {
    tasksEl.innerHTML =
      '<div class="task-flow">' +
      tkeys.slice(-30).map(tid => {
        const t = ft[tid];
        const d = t.duration_ms
          ? ` ${(t.duration_ms/1000).toFixed(1)}s`
          : '';
        return `<div class="task-node ${t.status}"` +
          ` title="${escHtml(tid)}">` +
          `${escHtml(t.agent_name || '?')}: ` +
          `${tid.substring(0,8)}${d}</div>`;
      }).join('<span class="task-arrow">&rarr;</span>') +
      '</div>';
  }
}

function renderEvents() {
  const fe = filteredEvents();
  eventsEl.innerHTML = '';
  // Show last 200 filtered events, newest first
  const toShow = fe.slice(-200).reverse();
  toShow.forEach(ev => {
    const t = ev.ts
      ? new Date(ev.ts * 1000).toLocaleTimeString()
      : '';
    const tc = ev.event_type.split('.').pop();
    const ag = escHtml(ev.agent_name || '');
    const tid = ev.task_id
      ? ev.task_id.substring(0, 12)
      : '';
    const det = escHtml(ev.detail || ev.worker_id || '');
    const dur = ev.duration_ms
      ? ` (${(ev.duration_ms/1000).toFixed(1)}s)`
      : '';

    const div = document.createElement('div');
    div.className = 'event';
    div.innerHTML =
      `<span class="time">${t}</span>` +
      `<span class="type ${tc}">${ev.event_type}</span>` +
      `<span class="detail">${ag} ${tid} ${det}${dur}</span>`;
    eventsEl.appendChild(div);
  });
}

function renderHeaderStats() {
  const ft = filteredTasks();
  const tkeys = Object.keys(ft);
  const wkeys = Object.keys(workers);
  const res = Object.values(ft)
    .filter(t => t.status === 'completed').length;
  document.getElementById('stat-workers')
    .textContent = wkeys.length;
  document.getElementById('stat-tasks')
    .textContent = tkeys.length;
  document.getElementById('stat-resolved')
    .textContent = res;
}

// ── Logs tab ──────────────────────────────────────────────
function shortType(et) {
  // 'task.picked_up' -> 'picked_up'
  return (et || '').split('.').slice(1).join('.') || et || '';
}

function fmtTimeShort(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return hh + ':' + mm + ':' + ss;
}

function taskLabel(ev) {
  if (!ev.task_id) return '';
  // Show short prefix of the task ID
  return 'T' + ev.task_id.substring(0, 6);
}

function logDetail(ev) {
  const et = ev.event_type || '';
  if (et === 'task.picked_up') {
    return 'by ' + (ev.agent_name || ev.worker_id || 'worker');
  }
  if (et === 'task.completed') {
    const dur = ev.duration_ms
      ? ' in ' + (ev.duration_ms / 1000).toFixed(1) + 's'
      : '';
    return (ev.success === false ? 'failed' : 'ok') + dur;
  }
  if (et === 'task.failed') {
    return ev.detail || 'error';
  }
  if (et === 'workflow.started' || et === 'workflow.completed') {
    return ev.detail || ev.workflow_id || '';
  }
  return ev.detail || '';
}

function renderLogs() {
  const body = document.getElementById('logs-body');
  if (!body) return;

  // Apply request filter + checkbox filters.
  const visible = logs.filter(ev => {
    if (activeRequest && ev.workflow_id !== activeRequest) return false;
    const et = ev.event_type || '';
    if (LLM_LOG_TYPES.has(et)) return logsFilterLlm;
    if (LIFECYCLE_LOG_TYPES.has(et)) return logsFilterLifecycle;
    return false;
  });

  if (visible.length === 0) {
    body.innerHTML =
      '<div class="log-hint">' +
      'Showing lifecycle events and per-agent LLM calls. ' +
      'Toggle the filters above. Switch to Events for the full stream.' +
      '</div>';
    return;
  }

  const rows = visible.slice(0, MAX_LOGS).map(ev => {
    const et = ev.event_type || '';
    if (et === 'llm.call') return renderLlmLogRow(ev);
    return renderLifecycleLogRow(ev);
  });
  body.innerHTML = rows.join('');
}

function renderLifecycleLogRow(ev) {
  const et = ev.event_type || '';
  const cls = 'type-' + et.split('.').join('-');
  return '<div class="log-line">' +
    '<span class="log-ts">[' + escHtml(fmtTimeShort(ev.ts)) + ']</span>' +
    '<span class="log-task">' + escHtml(taskLabel(ev)) + '</span>' +
    '<span class="log-type ' + cls + '">' +
      escHtml(shortType(et)) + '</span>' +
    '<span class="log-detail">' + escHtml(logDetail(ev)) + '</span>' +
  '</div>';
}

function renderLlmLogRow(ev) {
  // Header: [time] T-prefix agent-name#worker (call: 1.2s, 1466→412 tok)
  const ts = escHtml(fmtTimeShort(ev.ts));
  const tlbl = escHtml(taskLabel(ev));
  const agent = escHtml(ev.agent_name || 'agent');
  const worker = ev.worker_id
    ? '#' + escHtml(String(ev.worker_id).substring(0, 6))
    : '';
  const durTxt = ev.duration_ms
    ? (ev.duration_ms / 1000).toFixed(1) + 's'
    : '?';
  let tokTxt = '';
  if (ev.prompt_tokens || ev.completion_tokens) {
    const pt = ev.prompt_tokens || 0;
    const ct = ev.completion_tokens || 0;
    tokTxt = ', ' + pt + '→' + ct + ' tok';
  }
  const errTxt = (ev.success === false && ev.error)
    ? ' <span class="log-llm-prompt">error: ' +
      escHtml(String(ev.error).substring(0, 120)) + '</span>'
    : '';

  const promptText = ev.prompt_preview || '';
  const compText = ev.completion_preview || '';
  const promptLine = promptText
    ? '<div class="log-llm-text log-llm-prompt">' +
      '&ldquo;' + escHtml(promptText) + '&rdquo;</div>'
    : '';
  const compLine = compText
    ? '<div class="log-llm-text log-llm-completion">' +
      '<span class="log-arrow">&rarr;</span> &ldquo;' +
      escHtml(compText) + '&rdquo;</div>'
    : '';

  return '<div class="log-line llm">' +
    '<div class="log-header">' +
      '<span class="log-ts">[' + ts + ']</span>' +
      '<span class="log-task">' + tlbl + '</span>' +
      '<span class="log-type type-llm-call">' + agent + worker + '</span>' +
      '<span class="log-llm-meta">(call: ' + durTxt + tokTxt + ')</span>' +
      errTxt +
    '</div>' +
    promptLine +
    compLine +
  '</div>';
}

// ── Stats tab ─────────────────────────────────────────────
function formatDuration(seconds) {
  if (!seconds || seconds < 0) return '0s';
  const s = Math.floor(seconds);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return m + 'm ' + rs + 's';
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return h + 'h ' + rm + 'm';
}

function renderStatsPane() {
  if (!statsEl) return;
  const totalEvents = Object.values(eventCounts)
    .reduce((a, b) => a + b, 0);
  if (totalEvents === 0) {
    statsEl.innerHTML =
      '<div class="empty">Waiting for events...</div>';
    return;
  }
  const tvals = Object.values(tasks);
  const running = tvals.filter(t => t.status === 'running').length;
  const done = tvals.filter(t => t.status === 'completed').length;
  const failed = tvals.filter(t => t.status === 'failed').length;
  const pending = tvals.filter(t => t.status === 'pending').length;

  const wvals = Object.values(workers);
  const busy = wvals.filter(w => w.status === 'busy').length;
  const idle = wvals.filter(w => w.status === 'idle').length;

  const durations = tvals
    .filter(t => typeof t.duration_ms === 'number')
    .map(t => t.duration_ms);
  const avg = durations.length
    ? Math.round(
        durations.reduce((a, b) => a + b, 0) / durations.length,
      )
    : null;

  const now = Date.now() / 1000;
  const sinceStart = workflowStart
    ? formatDuration(now - workflowStart)
    : '-';

  const grid = [
    ['Total events', String(totalEvents)],
    ['Tasks running', String(running)],
    ['Tasks pending', String(pending)],
    ['Tasks done', String(done)],
    ['Tasks failed', String(failed)],
    ['Workers busy', String(busy)],
    ['Workers idle', String(idle)],
    ['Avg task duration', avg === null ? '-' : avg + ' ms'],
    ['Time since start', sinceStart],
  ];
  const dl =
    '<dl class="stats-list">' +
    grid.map(([k, v]) =>
      '<dt>' + escHtml(k) + '</dt>' +
      '<dd>' + escHtml(v) + '</dd>',
    ).join('') +
    '</dl>';

  // ── LLM call stats (per-agent breakdown) ───────────────
  const llmStatsBlock = renderLlmStats();

  const sortedCounts = Object.entries(eventCounts)
    .sort((a, b) => b[1] - a[1]);
  const table =
    '<div class="stats-section-title">Events by type</div>' +
    '<table class="stats-evtypes"><tbody>' +
    sortedCounts.map(([k, v]) =>
      '<tr>' +
      '<td class="evtype-name">' + escHtml(k) + '</td>' +
      '<td class="evtype-count">' + v + '</td>' +
      '</tr>',
    ).join('') +
    '</tbody></table>';

  // Timeseries cards on top, static counters, per-agent LLM stats,
  // event-type histogram at the bottom.
  const cards = renderSparkCards();
  statsEl.innerHTML = '<div class="stats-wrap">' +
    cards + dl + llmStatsBlock + table + '</div>';
}

// Aggregate llm.call events into headline metrics + per-agent rows.
function renderLlmStats() {
  const llmEvents = logs.filter(ev => ev.event_type === 'llm.call');
  if (llmEvents.length === 0) return '';

  let totalDur = 0;
  let durCount = 0;
  let totalPrompt = 0;
  let totalCompletion = 0;
  // agent_name -> {calls, prompt_tok, completion_tok, ms}
  const perAgent = {};

  for (const ev of llmEvents) {
    if (typeof ev.duration_ms === 'number') {
      totalDur += ev.duration_ms;
      durCount++;
    }
    if (typeof ev.prompt_tokens === 'number') {
      totalPrompt += ev.prompt_tokens;
    }
    if (typeof ev.completion_tokens === 'number') {
      totalCompletion += ev.completion_tokens;
    }
    const a = ev.agent_name || '(unknown)';
    if (!perAgent[a]) {
      perAgent[a] = {
        calls: 0, prompt_tok: 0, completion_tok: 0, ms: 0,
      };
    }
    perAgent[a].calls++;
    if (typeof ev.prompt_tokens === 'number') {
      perAgent[a].prompt_tok += ev.prompt_tokens;
    }
    if (typeof ev.completion_tokens === 'number') {
      perAgent[a].completion_tok += ev.completion_tokens;
    }
    if (typeof ev.duration_ms === 'number') {
      perAgent[a].ms += ev.duration_ms;
    }
  }

  const avgMs = durCount ? Math.round(totalDur / durCount) : 0;
  const headline = [
    ['Total LLM calls', String(llmEvents.length)],
    ['Avg call duration', avgMs + ' ms'],
    ['Total prompt tokens', String(totalPrompt)],
    ['Total completion tokens', String(totalCompletion)],
  ];
  const headlineDl =
    '<dl class="stats-grid">' +
    headline.map(([k, v]) =>
      '<dt>' + escHtml(k) + '</dt>' +
      '<dd>' + escHtml(v) + '</dd>',
    ).join('') +
    '</dl>';

  // Per-agent breakdown — sort by call count desc.
  const agents = Object.entries(perAgent)
    .sort((a, b) => b[1].calls - a[1].calls);
  const rows = agents.map(([name, s]) =>
    '<tr>' +
      '<td class="evtype-name">' + escHtml(name) + '</td>' +
      '<td class="evtype-count">' +
        s.calls + ' calls, ' +
        (s.prompt_tok + s.completion_tok) + ' tok' +
      '</td>' +
    '</tr>'
  ).join('');

  return (
    '<div class="stats-section-title">LLM calls</div>' +
    headlineDl +
    '<div class="stats-section-title">Per-agent breakdown</div>' +
    '<table class="stats-evtypes"><tbody>' + rows +
    '</tbody></table>'
  );
}

// ── Logs filter checkboxes ────────────────────────────────
document.querySelectorAll('.logs-filter').forEach(box => {
  box.addEventListener('change', () => {
    const which = box.dataset.filter;
    if (which === 'lifecycle') logsFilterLifecycle = box.checked;
    else if (which === 'llm') logsFilterLlm = box.checked;
    if (activeTab === 'logs') renderLogs();
  });
});

// ── Tab switching ─────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (!tab) return;
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.classList.toggle('active', p.id === tab);
    });
    renderActivePane();
  });
});

// Refresh the active pane every second so wall-clock fields
// (time-since-start) stay current even with no new events.  Also
// advance the timeseries bucket clock so sparklines decay/refresh
// even during quiet periods.
setInterval(() => {
  try { advanceSparkBuckets(); } catch (_e) { /* defensive */ }
  renderActivePane();
}, 1000);

// ── Show All button ───────────────────────────────────────
showAllBtn.addEventListener('click', () => {
  activeRequest = null;
  showAllBtn.classList.add('active');
  renderAll();
});

// Initial render
renderAll();
</script>
</body>
</html>"""


def main() -> None:
    """Run the dashboard server standalone (Redis backend)."""
    parser = argparse.ArgumentParser(
        description="Memfun Agent Dashboard",
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global _redis_url
    _redis_url = args.redis_url

    import uvicorn

    app = create_app()
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
    )


if __name__ == "__main__":
    main()
