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
import time
from contextlib import asynccontextmanager
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

MAX_EVENTS = 500


EVENT_TOPIC = "memfun.distributed.events"


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

    global _event_bus, _project_name
    _event_bus = event_bus
    _project_name = project_name or ""

    _bg_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if _event_bus is not None:
            task = asyncio.create_task(_event_listener(_event_bus))
        else:
            task = asyncio.create_task(_redis_listener())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        yield
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

    <!-- Middle bottom: Event Stream -->
    <div class="panel">
      <div class="panel-title">Event Stream</div>
      <div class="panel-content" id="events"></div>
    </div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────
const ws = new WebSocket(`ws://${location.host}/ws`);
const workersEl = document.getElementById('workers');
const tasksEl = document.getElementById('tasks');
const eventsEl = document.getElementById('events');
const requestListEl = document.getElementById('request-list');
const showAllBtn = document.getElementById('show-all-btn');
const projectLabel = document.getElementById('project-label');

let workers = {};
let tasks = {};
let requests = {};
let allEvents = [];
let activeRequest = null;  // null = show all
let resolved = 0;

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
    (msg.events || []).forEach(ev => allEvents.push(ev));
    renderAll();
  } else if (msg.type === 'event') {
    handleEvent(msg.data);
  }
};

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
  renderStats();
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

function renderStats() {
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
