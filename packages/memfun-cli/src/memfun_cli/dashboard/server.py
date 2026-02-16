"""Live dashboard server for distributed multi-agent coordination.

Subscribes to Redis event streams and pushes real-time updates to
connected browsers via WebSocket.  Shows:

- **Event Feed**: Live stream of agent events (task published, picked up, completed)
- **Agent Map**: Which workers are online, busy, or idle
- **Task DAG**: Visual flow of tasks through the decompose → execute → review pipeline

Usage::

    python -m memfun_cli.dashboard.server --redis-url redis://localhost:6379 --port 8080

Then open http://localhost:8080 in your browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("dashboard")

# Will be populated when the server starts
_redis_url: str = "redis://localhost:6379"
_connections: set[Any] = set()  # WebSocket connections
_events: list[dict] = []  # Recent events buffer (max 500)
_workers: dict[str, dict] = {}  # worker_id -> {agent_name, status, last_seen}
_tasks: dict[str, dict] = {}  # task_id -> {agent_name, status, worker_id, ...}

MAX_EVENTS = 500


async def _redis_listener() -> None:
    """Subscribe to the distributed events stream and broadcast to WebSockets."""
    from memfun_runtime.backends.redis.event_bus import RedisEventBus
    from memfun_runtime.distributed import EVENT_TOPIC, DistributedEvent

    bus = await RedisEventBus.create(_redis_url)
    logger.info("Dashboard listening on Redis events at %s", _redis_url)

    async for msg in bus.subscribe(EVENT_TOPIC):
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
            }

            # Update state
            _update_state(event_dict)

            # Buffer
            _events.append(event_dict)
            if len(_events) > MAX_EVENTS:
                _events.pop(0)

            # Broadcast to all connected WebSocket clients
            message = json.dumps({"type": "event", "data": event_dict})
            dead = set()
            for ws in _connections:  # noqa: F823
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            _connections -= dead

        except Exception as exc:
            logger.debug("Bad event: %s", exc)


def _update_state(event: dict) -> None:
    """Update worker and task state from an event."""
    etype = event.get("event_type", "")
    worker_id = event.get("worker_id")
    task_id = event.get("task_id")
    agent_name = event.get("agent_name")

    if etype == "worker.online" and worker_id:
        _workers[worker_id] = {
            "agent_name": agent_name,
            "status": "idle",
            "last_seen": time.time(),
            "tasks_done": 0,
        }
    elif etype == "worker.offline" and worker_id:
        _workers.pop(worker_id, None)
    elif etype == "task.published" and task_id:
        _tasks[task_id] = {
            "agent_name": agent_name,
            "status": "pending",
            "detail": event.get("detail", ""),
            "ts": event.get("ts", time.time()),
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
            _tasks[task_id]["status"] = "completed" if event.get("success") else "failed"
            _tasks[task_id]["duration_ms"] = event.get("duration_ms")
        if worker_id and worker_id in _workers:
            _workers[worker_id]["status"] = "idle"
            _workers[worker_id]["tasks_done"] = _workers[worker_id].get("tasks_done", 0) + 1
            _workers[worker_id]["last_seen"] = time.time()


def _get_state() -> dict:
    """Current snapshot for newly connected clients."""
    return {
        "type": "state",
        "workers": _workers,
        "tasks": _tasks,
        "events": _events[-50:],  # Last 50 events
    }


def create_app():
    """Create the FastAPI app for the dashboard."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ImportError:
        logger.error("Install fastapi: pip install fastapi uvicorn")
        raise

    app = FastAPI(title="Memfun Agent Dashboard")

    _bg_tasks: set[asyncio.Task[None]] = set()

    @app.on_event("startup")
    async def startup():
        task = asyncio.create_task(_redis_listener())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _DASHBOARD_HTML

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        _connections.add(ws)
        # Send current state
        await ws.send_text(json.dumps(_get_state()))
        try:
            while True:
                await ws.receive_text()  # Keep alive
        except WebSocketDisconnect:
            _connections.discard(ws)

    @app.get("/api/state")
    async def api_state():
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
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
  }

  .header {
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .badge {
    background: var(--accent); color: var(--bg); padding: 2px 8px;
    border-radius: 12px; font-size: 11px; font-weight: 700;
  }
  .header .stats {
    margin-left: auto; display: flex; gap: 16px;
    font-size: 13px; color: var(--muted);
  }
  .header .stats span { color: var(--text); font-weight: 600; }

  .grid {
    display: grid; grid-template-columns: 300px 1fr; grid-template-rows: auto 1fr;
    height: calc(100vh - 56px); gap: 1px; background: var(--border);
  }

  .panel { background: var(--surface); overflow: hidden; display: flex; flex-direction: column; }
  .panel-title {
    padding: 10px 16px; font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .panel-content { overflow-y: auto; flex: 1; padding: 8px; }

  /* Workers panel */
  .worker {
    padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.03);
  }
  .worker .dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }
  .worker .dot.idle { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .worker .dot.busy {
    background: var(--yellow);
    box-shadow: 0 0 6px var(--yellow);
    animation: pulse 1.5s infinite;
  }
  .worker .dot.offline { background: var(--muted); }
  .worker .name { font-size: 13px; font-weight: 500; }
  .worker .info { font-size: 11px; color: var(--muted); margin-left: auto; }

  /* Task DAG */
  .task-flow { padding: 12px; }
  .task-node {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 6px; margin: 4px;
    font-size: 12px; border: 1px solid var(--border);
  }
  .task-node.pending { border-color: var(--muted); color: var(--muted); }
  .task-node.running {
    border-color: var(--yellow); color: var(--yellow);
    animation: pulse 1.5s infinite;
  }
  .task-node.completed { border-color: var(--green); color: var(--green); }
  .task-node.failed { border-color: var(--red); color: var(--red); }
  .task-arrow { color: var(--muted); margin: 0 4px; }

  /* Event feed */
  .event {
    padding: 6px 12px; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.03);
    display: flex; gap: 8px; align-items: baseline;
  }
  .event .time { color: var(--muted); font-size: 11px; flex-shrink: 0; width: 80px; }
  .event .type { font-weight: 600; width: 120px; flex-shrink: 0; }
  .event .type.published { color: var(--accent); }
  .event .type.picked_up { color: var(--yellow); }
  .event .type.completed { color: var(--green); }
  .event .type.failed { color: var(--red); }
  .event .type.online { color: var(--purple); }
  .event .type.offline { color: var(--muted); }
  .event .detail {
    color: var(--muted); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .empty { padding: 24px; text-align: center; color: var(--muted); font-size: 13px; }
</style>
</head>
<body>

<div class="header">
  <h1>Memfun</h1>
  <div class="badge">LIVE</div>
  <div class="stats">
    Workers: <span id="stat-workers">0</span> &nbsp;|&nbsp;
    Tasks: <span id="stat-tasks">0</span> &nbsp;|&nbsp;
    Resolved: <span id="stat-resolved">0</span>
  </div>
</div>

<div class="grid">
  <div class="panel" style="grid-row: span 2">
    <div class="panel-title">Workers</div>
    <div class="panel-content" id="workers">
      <div class="empty">Waiting for workers...</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Task Flow</div>
    <div class="panel-content" id="tasks">
      <div class="empty">No tasks yet</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Event Stream</div>
    <div class="panel-content" id="events"></div>
  </div>
</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const workersEl = document.getElementById('workers');
const tasksEl = document.getElementById('tasks');
const eventsEl = document.getElementById('events');

let workers = {};
let tasks = {};
let resolved = 0;

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'state') {
    workers = msg.workers || {};
    tasks = msg.tasks || {};
    resolved = Object.values(tasks).filter(t => t.status === 'completed').length;
    (msg.events || []).forEach(addEvent);
    render();
  } else if (msg.type === 'event') {
    handleEvent(msg.data);
  }
};

function handleEvent(ev) {
  const et = ev.event_type;
  if (et === 'worker.online') {
    workers[ev.worker_id] = { agent_name: ev.agent_name, status: 'idle', tasks_done: 0 };
  } else if (et === 'worker.offline') {
    delete workers[ev.worker_id];
  } else if (et === 'task.published') {
    tasks[ev.task_id] = { agent_name: ev.agent_name, status: 'pending', detail: ev.detail };
  } else if (et === 'task.picked_up') {
    if (tasks[ev.task_id]) {
      tasks[ev.task_id].status = 'running';
      tasks[ev.task_id].worker_id = ev.worker_id;
    }
    if (workers[ev.worker_id]) workers[ev.worker_id].status = 'busy';
  } else if (et === 'task.completed') {
    if (tasks[ev.task_id]) {
      tasks[ev.task_id].status = ev.success ? 'completed' : 'failed';
      tasks[ev.task_id].duration_ms = ev.duration_ms;
    }
    if (ev.worker_id && workers[ev.worker_id]) {
      workers[ev.worker_id].status = 'idle';
      workers[ev.worker_id].tasks_done = (workers[ev.worker_id].tasks_done || 0) + 1;
    }
    if (ev.success) resolved++;
  }
  addEvent(ev);
  render();
}

function addEvent(ev) {
  const time = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : '';
  const typeClass = ev.event_type.split('.').pop();
  const agent = ev.agent_name || '';
  const task = ev.task_id ? ev.task_id.substring(0, 12) : '';
  const detail = ev.detail || ev.worker_id || '';
  const dur = ev.duration_ms ? ` (${(ev.duration_ms/1000).toFixed(1)}s)` : '';

  const div = document.createElement('div');
  div.className = 'event';
  div.innerHTML = `
    <span class="time">${time}</span>
    <span class="type ${typeClass}">${ev.event_type}</span>
    <span class="detail">${agent} ${task} ${detail}${dur}</span>
  `;
  eventsEl.insertBefore(div, eventsEl.firstChild);
  if (eventsEl.children.length > 200) eventsEl.removeChild(eventsEl.lastChild);
}

function render() {
  // Workers
  const wkeys = Object.keys(workers);
  if (wkeys.length === 0) {
    workersEl.innerHTML = '<div class="empty">Waiting for workers...</div>';
  } else {
    workersEl.innerHTML = wkeys.map(wid => {
      const w = workers[wid];
      return `<div class="worker">
        <div class="dot ${w.status}"></div>
        <span class="name">${w.agent_name}</span>
        <span class="info">${w.status} (${w.tasks_done || 0} done)</span>
      </div>`;
    }).join('');
  }

  // Tasks
  const tkeys = Object.keys(tasks);
  if (tkeys.length === 0) {
    tasksEl.innerHTML = '<div class="empty">No tasks yet</div>';
  } else {
    tasksEl.innerHTML = '<div class="task-flow">' + tkeys.slice(-30).map(tid => {
      const t = tasks[tid];
      const dur = t.duration_ms ? ` ${(t.duration_ms/1000).toFixed(1)}s` : '';
      return `<div class="task-node ${t.status}" title="${tid}">
        ${t.agent_name || '?'}: ${tid.substring(0,8)}${dur}
      </div>`;
    }).join('<span class="task-arrow">→</span>') + '</div>';
  }

  // Stats
  document.getElementById('stat-workers').textContent = wkeys.length;
  document.getElementById('stat-tasks').textContent = tkeys.length;
  document.getElementById('stat-resolved').textContent = resolved;
}
</script>
</body>
</html>"""


def main() -> None:
    """Run the dashboard server."""
    parser = argparse.ArgumentParser(description="Memfun Agent Dashboard")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global _redis_url
    _redis_url = args.redis_url

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
