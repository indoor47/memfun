"""Live dashboard server for distributed multi-agent coordination.

Subscribes to Redis event streams and pushes real-time updates to
connected browsers via WebSocket.  Shows:

- **Event Feed**: Live stream of agent events (task published, picked up, completed)
- **Agent Map**: Which workers are online, busy, or idle
- **Task DAG**: Visual flow of tasks through the decompose -> execute -> review pipeline
- **Web CLI**: Chat with the memfun agent directly from the browser

Usage::

    python -m memfun_cli.dashboard.server --redis-url redis://localhost:6379 --port 8080

Then open http://localhost:8080 in your browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("dashboard")

# Will be populated when the server starts
_redis_url: str = "redis://localhost:6379"
_connections: set[Any] = set()  # WebSocket connections
_events: list[dict] = []  # Recent events buffer (max 500)
_workers: dict[str, dict] = {}  # worker_id -> {agent_name, status, last_seen}
_tasks: dict[str, dict] = {}  # task_id -> {agent_name, status, worker_id, ...}

# Chat state
_chat_session: Any | None = None
_chat_ws: Any | None = None

MAX_EVENTS = 500


async def _redis_listener() -> None:
    """Subscribe to the distributed events stream and broadcast."""
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

            _update_state(event_dict)

            _events.append(event_dict)
            if len(_events) > MAX_EVENTS:
                _events.pop(0)

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
    }


# ── Chat session helpers ──────────────────────────────────────────


def _load_credentials() -> None:
    """Load API keys from credentials files into environment."""
    for base in (Path.home(), Path.cwd()):
        creds_path = base / ".memfun" / "credentials.json"
        if not creds_path.exists():
            continue
        try:
            creds = json.loads(creds_path.read_text())
            if isinstance(creds, dict):
                for key, value in creds.items():
                    if (
                        isinstance(key, str)
                        and isinstance(value, str)
                        and value
                    ):
                        os.environ[key] = value
        except Exception:
            logger.debug("Failed to load credentials from %s", creds_path)


async def _init_chat_session() -> Any:
    """Create and start a ChatSession (lazy, one-time)."""
    from memfun_cli.commands.chat import ChatSession

    _load_credentials()

    session = ChatSession()
    await session.start()

    # Wire the context-first status callback to push to WebSocket
    original_cb = session._on_context_first_status_callback

    def _cf_status_hook(msg: str) -> None:
        original_cb(msg)
        _push_chat_status(msg)

    if session._agent is not None:
        session._agent.on_context_first_status = _cf_status_hook

    logger.info("Chat session initialized for dashboard")
    return session


def _push_chat_status(text: str) -> None:
    """Best-effort push a status line to the chat WebSocket."""
    if _chat_ws is not None:
        import contextlib

        with contextlib.suppress(Exception):
            asyncio.get_event_loop().create_task(
                _chat_ws.send_text(
                    json.dumps({"type": "status", "text": text})
                )
            )


async def _handle_chat_message(ws: Any, text: str) -> None:
    """Process a chat message and push result back."""
    global _chat_session

    await ws.send_text(json.dumps({
        "type": "status", "text": "Processing...",
    }))

    start = time.monotonic()
    try:
        result = await _chat_session.chat_turn(text)
        elapsed = time.monotonic() - start
        data = result.result or {}
        answer = data.get(
            "answer", data.get("explanation", str(data)),
        )
        await ws.send_text(json.dumps({
            "type": "answer",
            "text": answer,
            "files": data.get("files_created", []),
            "method": data.get("method", ""),
            "duration": round(elapsed, 1),
            "success": result.success,
        }))
    except Exception as exc:
        logger.warning("Chat turn failed: %s", exc, exc_info=True)
        await ws.send_text(json.dumps({
            "type": "error",
            "text": str(exc),
        }))


# ── FastAPI app ───────────────────────────────────────────────────


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
        await ws.send_text(json.dumps(_get_state()))
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            _connections.discard(ws)

    @app.websocket("/ws/chat")
    async def chat_endpoint(ws: WebSocket):
        global _chat_session, _chat_ws
        await ws.accept()
        _chat_ws = ws

        # Lazy-init ChatSession on first connection
        if _chat_session is None:
            await ws.send_text(json.dumps({
                "type": "status",
                "text": "Starting agent...",
            }))
            try:
                _chat_session = await _init_chat_session()
            except Exception as exc:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "text": f"Failed to start agent: {exc}",
                }))
                _chat_ws = None
                return
            await ws.send_text(json.dumps({
                "type": "status",
                "text": "Agent ready.",
            }))

        # Send recent history
        history = _chat_session._history[-20:]
        await ws.send_text(json.dumps({
            "type": "history",
            "messages": [
                {
                    "role": h.get("role", "user"),
                    "content": str(
                        h.get("content", "")
                    )[:2000],
                }
                for h in history
            ],
        }))

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                if data.get("type") == "message":
                    user_text = str(data.get("text", "")).strip()
                    if user_text:
                        await _handle_chat_message(ws, user_text)
        except WebSocketDisconnect:
            _chat_ws = None

    @app.get("/api/state")
    async def api_state():
        return _get_state()

    return app


# ── Dashboard HTML (single-file, no build step) ─────────────────────

_DASHBOARD_HTML = (
    """<!DOCTYPE html>
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

  /* Top area: workers + tasks + events */
  .top-area {
    flex: 1; min-height: 120px;
    display: grid;
    grid-template-columns: 260px 1fr;
    grid-template-rows: auto 1fr;
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
    overflow-y: auto; flex: 1; padding: 8px;
  }

  /* Drag handle */
  .drag-handle {
    height: 6px; background: var(--border);
    cursor: ns-resize; flex-shrink: 0;
    position: relative;
  }
  .drag-handle:hover,
  .drag-handle.active {
    background: var(--accent);
  }
  .drag-handle::after {
    content: ''; position: absolute;
    left: 50%; top: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 2px;
    background: var(--muted); border-radius: 1px;
  }

  /* Terminal panel at bottom */
  .terminal-area {
    height: 280px; min-height: 80px;
    display: flex; flex-direction: column;
    background: var(--bg);
    border-top: 1px solid var(--border);
  }
  .terminal-header {
    padding: 6px 16px; font-size: 11px;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--muted);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px;
    flex-shrink: 0;
  }
  .terminal-header .dot-green {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
  }
  .terminal-header .dot-gray {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted);
  }
  .terminal-output {
    flex: 1; overflow-y: auto; padding: 8px 12px;
    font-size: 13px; line-height: 1.6;
  }
  .terminal-input-row {
    display: flex; align-items: center;
    padding: 6px 12px;
    border-top: 1px solid var(--border);
    background: var(--surface); flex-shrink: 0;
  }
  .terminal-prompt {
    color: var(--accent); font-weight: 600;
    margin-right: 8px; white-space: nowrap;
    font-size: 13px;
  }
  .terminal-input-row input {
    flex: 1; background: transparent;
    color: var(--text); border: none; outline: none;
    font-family: inherit; font-size: 13px;
  }

  /* Terminal line styles */
  .t-line {
    white-space: pre-wrap; word-break: break-word;
  }
  .t-user { color: var(--accent); }
  .t-answer { color: var(--text); }
  .t-status {
    color: var(--muted); font-style: italic;
  }
  .t-error { color: var(--red); }
  .t-meta {
    color: var(--muted); font-size: 11px;
  }
  .t-files { color: var(--green); font-size: 12px; }
  .t-welcome { color: var(--purple); }

  /* Workers panel */
  .worker {
    padding: 6px 10px; border-radius: 6px;
    margin-bottom: 3px;
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

  <div class="drag-handle" id="drag-handle"></div>

  <div class="terminal-area" id="terminal-area">
    <div class="terminal-header">
      <span class="dot-gray" id="agent-dot"></span>
      Terminal
      <span style="margin-left:auto;font-size:10px"
            id="agent-status">connecting...</span>
    </div>
    <div class="terminal-output" id="terminal-output"></div>
    <div class="terminal-input-row">
      <span class="terminal-prompt">memfun&gt;</span>
      <input type="text" id="terminal-input"
             placeholder="Ask memfun anything..."
             autocomplete="off" disabled />
    </div>
  </div>
</div>

<script>
// ── Event dashboard WebSocket ─────────────────────────────
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
    resolved = Object.values(tasks)
      .filter(t => t.status === 'completed').length;
    (msg.events || []).forEach(addEvent);
    render();
  } else if (msg.type === 'event') {
    handleEvent(msg.data);
  }
};

function handleEvent(ev) {
  const et = ev.event_type;
  if (et === 'worker.online') {
    workers[ev.worker_id] = {
      agent_name: ev.agent_name,
      status: 'idle', tasks_done: 0,
    };
  } else if (et === 'worker.offline') {
    delete workers[ev.worker_id];
  } else if (et === 'task.published') {
    tasks[ev.task_id] = {
      agent_name: ev.agent_name,
      status: 'pending', detail: ev.detail,
    };
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
  }
  addEvent(ev);
  render();
}

function addEvent(ev) {
  const t = ev.ts
    ? new Date(ev.ts * 1000).toLocaleTimeString()
    : '';
  const tc = ev.event_type.split('.').pop();
  const ag = ev.agent_name || '';
  const tid = ev.task_id
    ? ev.task_id.substring(0, 12)
    : '';
  const det = ev.detail || ev.worker_id || '';
  const dur = ev.duration_ms
    ? ` (${(ev.duration_ms/1000).toFixed(1)}s)`
    : '';

  const div = document.createElement('div');
  div.className = 'event';
  div.innerHTML =
    `<span class="time">${t}</span>` +
    `<span class="type ${tc}">${ev.event_type}</span>` +
    `<span class="detail">${ag} ${tid} ${det}${dur}</span>`;
  eventsEl.insertBefore(div, eventsEl.firstChild);
  if (eventsEl.children.length > 200) {
    eventsEl.removeChild(eventsEl.lastChild);
  }
}

function render() {
  const wkeys = Object.keys(workers);
  if (wkeys.length === 0) {
    workersEl.innerHTML =
      '<div class="empty">Waiting for workers...</div>';
  } else {
    workersEl.innerHTML = wkeys.map(wid => {
      const w = workers[wid];
      return `<div class="worker">
        <div class="dot ${w.status}"></div>
        <span class="name">${w.agent_name}</span>
        <span class="info">${w.status} """
    """(${w.tasks_done || 0})</span>
      </div>`;
    }).join('');
  }

  const tkeys = Object.keys(tasks);
  if (tkeys.length === 0) {
    tasksEl.innerHTML =
      '<div class="empty">No tasks yet</div>';
  } else {
    tasksEl.innerHTML =
      '<div class="task-flow">' +
      tkeys.slice(-30).map(tid => {
        const t = tasks[tid];
        const d = t.duration_ms
          ? ` ${(t.duration_ms/1000).toFixed(1)}s`
          : '';
        return `<div class="task-node ${t.status}" """
    """title="${tid}">` +
          `${t.agent_name || '?'}: ${tid.substring(0,8)}${d}` +
          `</div>`;
      }).join('<span class="task-arrow">&rarr;</span>') +
      '</div>';
  }

  document.getElementById('stat-workers')
    .textContent = wkeys.length;
  document.getElementById('stat-tasks')
    .textContent = tkeys.length;
  document.getElementById('stat-resolved')
    .textContent = resolved;
}

// ── Terminal / Chat WebSocket ─────────────────────────────
const termOut = document.getElementById('terminal-output');
const termIn = document.getElementById('terminal-input');
const agentDot = document.getElementById('agent-dot');
const agentStatus = document.getElementById('agent-status');

let chatWs = null;
let chatBusy = false;
let statusEl = null;

function connectChat() {
  chatWs = new WebSocket(`ws://${location.host}/ws/chat`);

  chatWs.onopen = () => {
    agentStatus.textContent = 'initializing...';
  };

  chatWs.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'status') {
      updateStatus(msg.text);
      if (msg.text === 'Agent ready.') {
        agentDot.className = 'dot-green';
        agentStatus.textContent = 'ready';
        termIn.disabled = false;
        termIn.placeholder = 'Ask memfun anything...';
        termIn.focus();
      }
    } else if (msg.type === 'answer') {
      clearStatus();
      tLine('t-answer', msg.text);
      const parts = [];
      if (msg.method) parts.push(msg.method);
      if (msg.duration) parts.push(msg.duration + 's');
      if (parts.length) tLine('t-meta', parts.join(' | '));
      if (msg.files && msg.files.length) {
        tLine('t-files',
          'Files: ' + msg.files.join(', '));
      }
      tLine('t-answer', '');  // blank line
      chatBusy = false;
      agentDot.className = 'dot-green';
      agentStatus.textContent = 'ready';
      termIn.disabled = false;
      termIn.placeholder = 'Ask memfun anything...';
      termIn.focus();
    } else if (msg.type === 'error') {
      clearStatus();
      tLine('t-error', 'Error: ' + msg.text);
      chatBusy = false;
      agentDot.className = 'dot-green';
      agentStatus.textContent = 'ready';
      termIn.disabled = false;
      termIn.focus();
    } else if (msg.type === 'history') {
      (msg.messages || []).slice(-10).forEach(m => {
        if (m.role === 'user') {
          tLine('t-user', '> ' + m.content);
        } else {
          tLine('t-answer', m.content);
        }
      });
    }
  };

  chatWs.onclose = () => {
    agentDot.className = 'dot-gray';
    agentStatus.textContent = 'disconnected';
    termIn.disabled = true;
    termIn.placeholder = 'Reconnecting...';
    setTimeout(connectChat, 3000);
  };
}

termIn.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !chatBusy) {
    const text = termIn.value.trim();
    if (!text) return;
    termIn.value = '';
    tLine('t-user', '> ' + text);
    chatWs.send(JSON.stringify({
      type: 'message', text: text,
    }));
    chatBusy = true;
    termIn.disabled = true;
    termIn.placeholder = 'Thinking...';
    agentDot.className = 'dot-gray';
    agentStatus.textContent = 'thinking...';
  }
});

function tLine(cls, text) {
  const div = document.createElement('div');
  div.className = 't-line ' + cls;
  div.textContent = text;
  termOut.appendChild(div);
  termOut.scrollTop = termOut.scrollHeight;
}

function updateStatus(text) {
  if (!statusEl) {
    statusEl = document.createElement('div');
    statusEl.className = 't-line t-status';
    termOut.appendChild(statusEl);
  }
  statusEl.textContent = text;
  agentStatus.textContent = text.substring(0, 40);
  termOut.scrollTop = termOut.scrollHeight;
}

function clearStatus() {
  if (statusEl) { statusEl.remove(); statusEl = null; }
}

// Welcome message
tLine('t-welcome',
  'Memfun Agent Terminal');
tLine('t-meta',
  'Type a message and press Enter to chat with the agent.');
tLine('t-answer', '');

connectChat();

// ── Drag handle for resizing ──────────────────────────────
const handle = document.getElementById('drag-handle');
const termArea = document.getElementById('terminal-area');
let dragging = false;
let startY = 0;
let startH = 0;

handle.addEventListener('mousedown', (e) => {
  dragging = true;
  startY = e.clientY;
  startH = termArea.offsetHeight;
  handle.classList.add('active');
  document.body.style.cursor = 'ns-resize';
  document.body.style.userSelect = 'none';
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const delta = startY - e.clientY;
  const newH = Math.max(80, Math.min(
    window.innerHeight - 200, startH + delta
  ));
  termArea.style.height = newH + 'px';
});

document.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  handle.classList.remove('active');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
});
</script>
</body>
</html>"""
)


def main() -> None:
    """Run the dashboard server."""
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
