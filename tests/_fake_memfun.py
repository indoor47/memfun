"""A fake ``memfun`` for hermetic tests of ``evals/swarm/run_smoke.py``.

    python _fake_memfun.py --state DIR --bundles-root ROOT [--mode M] [--bundle-mode B=M]
        [--hold SECONDS] swarm "<spec>" --max-pool N --timeout S --no-push
    python _fake_memfun.py --state DIR ... dashboard --remote --port P

``swarm`` behaves like ``memfun swarm`` as far as the runner can see: prints the dashboard
banner (from ``.memfun/dashboard.lock``), creates per-task "worktrees" under
``.memfun/worktrees/<wf>/<task>``, writes owned files there, and publishes realistic
dashboard events (``workflow.started``, ``task.*``, ``llm.call`` with tokens) into the
project's T1 event-bus DB using memfun's own schema and ``DistributedEvent``.  Timestamps
come from a fake clock (``FAKE_T0`` + offsets) so metrics are deterministic.

``dashboard`` serves ``/api/health`` and records ``POST /api/event`` bodies to
``<state>/dashboard_events.jsonl``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.server
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

# memfun's own DDL, so the fake's DB has exactly the schema the runner reads.
from memfun_runtime.backends.sqlite.event_bus import (
    _CREATE_EVENTS,  # pyright: ignore[reportPrivateUsage]
    _CREATE_TOPICS,  # pyright: ignore[reportPrivateUsage]
)
from memfun_runtime.distributed import EVENT_TOPIC, DistributedEvent

FAKE_T0 = 1_000_000.0
WF = "wf-fake01"


@contextlib.contextmanager
def _locked(state: Path):
    with (state / "conc.lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _conc(state: Path, delta: int) -> None:
    with _locked(state):
        path = state / "conc.json"
        data = json.loads(path.read_text()) if path.is_file() else {"running": 0, "max": 0}
        data["running"] += delta
        data["max"] = max(data["max"], data["running"])
        path.write_text(json.dumps(data))


class Bus:
    def __init__(self, db: Path) -> None:
        db.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(_CREATE_EVENTS + _CREATE_TOPICS)

    def emit(self, offset: float, event_type: str, **kw: Any) -> None:
        ev = DistributedEvent(event_type=event_type, ts=FAKE_T0 + offset, **kw)
        self.con.execute(
            "INSERT INTO events (id, topic, payload, key, timestamp) VALUES (?, ?, ?, ?, ?)",
            (os.urandom(8).hex(), EVENT_TOPIC, ev.to_bytes(), None, time.time()),
        )
        self.con.commit()


def _write(path: Path, content: bytes, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def _worktree(repo: Path, tid: str) -> Path:
    wt = repo / ".memfun" / "worktrees" / WF / tid
    shutil.copytree(repo, wt, ignore=shutil.ignore_patterns(".git", ".memfun"), dirs_exist_ok=True)
    return wt


def _fake_ops(n: int, owns: list[str]) -> list[dict[str, str]]:
    """Deterministic ops for sub-task ``n``: grep on even n, find_files on n == 0."""
    ops: list[dict[str, str]] = []
    if n % 2 == 0:
        ops.append({"type": "grep", "target": "def ", "detail": "3"})
    if n == 0:
        ops.append({"type": "find_files", "target": "*.py", "detail": "5"})
    ops.append({"type": "read", "target": "README", "detail": "10"})
    for rel in owns:
        ops.append({"type": "read", "target": rel, "detail": "100"})
        ops.append({"type": "write", "target": rel, "detail": "200"})
    return ops


def _write_result(result_mode: str, groups: list[list[int]], units: list[Any]) -> None:
    """Write $MEMFUN_SWARM_RESULT_JSON like memfun swarm does (O_EXCL, no-follow)."""
    target = os.environ.get("MEMFUN_SWARM_RESULT_JSON")
    if not target or result_mode == "none":
        return
    path = Path(target)
    sub_tasks: dict[str, Any] = {}
    totals: dict[str, int] = {}
    for n, idxs in enumerate(groups):
        ops = _fake_ops(n, [rel for i in idxs for rel in units[i]["owns"]])
        for op in ops:
            totals[op["type"]] = totals.get(op["type"], 0) + 1
        sub_tasks[f"T{n + 1}"] = {
            "agent": "coder-agent",
            "status": "completed",
            "success": True,
            "error": None,
            "iterations": 3,
            "ops": ops,
            "ops_count": len(ops),
            "truncated": False,
        }
    doc: dict[str, Any] = {
        "schema_version": 2 if result_mode == "schema2" else 1,
        "workflow_id": WF, "success": True, "error": None, "unaccepted_tasks": [],
        "review_errors": 0, "review_rounds": 1, "files_created": [],
        "decomposition": {"n_sub_tasks": len(groups), "is_single_task": len(groups) == 1,
                          "groups": [[f"T{n + 1}" for n in range(len(groups))]]},
        "sub_tasks": sub_tasks, "ops_total": totals, "wall_seconds": 1.0,
    }  # fmt: skip
    if result_mode == "symlink":  # an agent planting a forged file behind a symlink
        forged = path.parent / "home" / "forged.json"
        forged.write_text(json.dumps(doc))
        path.symlink_to(forged)
        return
    data = "{not json" if result_mode == "invalid" else json.dumps(doc)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(data)


def swarm(args: argparse.Namespace, rest: list[str]) -> int:
    state = Path(args.state)
    repo = Path.cwd()
    project = repo.name
    bundle_id = project.rsplit("-t", 1)[0]
    modes = dict(m.split("=", 1) for m in args.bundle_mode)
    mode = modes.get(bundle_id, args.mode)
    bundle = Path(args.bundles_root) / bundle_id
    task = tomllib.loads((bundle / "task.toml").read_text())
    units = task["units"]

    (state / "calls").mkdir(parents=True, exist_ok=True)
    (state / "calls" / f"{project}.json").write_text(
        json.dumps(
            {
                "argv": rest,
                "env": dict(os.environ),
                "cwd": str(repo),
                "mode": mode,
                "pid": os.getpid(),
            }
        )
    )
    _conc(state, +1)
    try:
        lock = repo / ".memfun" / "dashboard.lock"
        banner = "memfun swarm\nproject: " + str(repo)
        if lock.is_file():
            banner += f"\nDashboard: http://localhost:{json.loads(lock.read_text())['port']}"
        print(banner, flush=True)
        time.sleep(args.hold)

        if mode == "crash":
            print("Traceback (most recent call last):\n  boom\nRuntimeError: fake crash")
            return 3
        if mode == "hang_after_output":
            for i, unit in enumerate(units):
                wt = _worktree(repo, f"T{i + 1}")
                for rel in unit["owns"]:
                    _write(wt / rel, (bundle / "solution" / rel).read_bytes(), time.time())
            mode = "hang"
        if mode == "hang":
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(600)"],
            )
            (state / "hang_child.pid").write_text(str(child.pid))
            time.sleep(600)
            return 0

        bus = Bus(repo / ".memfun" / "memfun.db")
        bus.emit(0, "workflow.started", workflow_id=WF, project=project, detail="spec")
        bus.emit(
            1,
            "llm.call",
            workflow_id=WF,
            role="decomposer",
            model="fake-qwen",
            prompt_tokens=50,
            completion_tokens=5,
            duration_ms=500,
        )
        if mode == "warnings":
            print("WARNING Failed to discover MCP tools: fake")
            print("WARNING Task decomposition failed: fake (falling back)")

        if mode == "single":
            groups = [list(range(len(units)))]
        elif mode == "duplicate":
            groups = [[0], [0]] + [[i] for i in range(1, len(units))]
        else:
            groups = [[i] for i in range(len(units))]

        mtime = time.time() - 1000
        for n, unit_idxs in enumerate(groups):
            tid = f"T{n + 1}"
            full = f"{WF}_{tid}"
            bus.emit(2, "task.published", task_id=full, agent_name="coder-agent",
                     workflow_id=WF, project=project)  # fmt: skip
            bus.emit(2, "task.picked_up", task_id=full, agent_name="coder-agent",
                     workflow_id=WF, project=project)  # fmt: skip
            err = None
            if mode == "server_error" or (mode == "server_error_recovered" and n == 0):
                err = "litellm.APIConnectionError: Connection refused"
            bus.emit(
                3 + n * 0.1,
                "llm.call",
                task_id=full,
                agent_name="coder-agent",
                workflow_id=WF,
                role="worker",
                model="fake-qwen",
                prompt_tokens=None if err else 100 * (n + 1),
                completion_tokens=None if err else 10 * (n + 1),
                duration_ms=500,
                error=err,
            )
            if mode in ("server_error", "noop"):
                continue
            target = repo if mode == "root" else _worktree(repo, tid)
            for i in unit_idxs:
                if mode == "missing" and i == len(units) - 1:
                    continue
                for rel in units[i]["owns"]:
                    sol = bundle / "solution" / rel
                    good = sol.read_bytes() if sol.is_file() else b""
                    wrong = (mode == "fail_unit" and i == 0) or (mode == "duplicate" and n == 0)
                    start = bundle / "repo" / rel
                    bad = (start.read_bytes() if start.is_file() else b"") + b"\n# wrong\n"
                    if mode == "exit_unit" and i == 0:  # unguarded CLI exit at import
                        good = good + b"\nimport sys\nsys.exit(0)\n"
                    mtime += 10
                    if mode == "symlink":  # point the owned path at the reference answer
                        (target / rel).unlink(missing_ok=True)
                        (target / rel).symlink_to(sol.resolve())
                        continue
                    _write(target / rel, bad if wrong else good, mtime)
            if mode == "tamper":
                _write(target / "tests" / "conftest.py", b"import pytest\n", mtime)
                _write(target / "conftest.py", b"raise SystemExit(0)\n", mtime)
            bus.emit(5 + n, "task.completed", task_id=full, agent_name="coder-agent",
                     workflow_id=WF, project=project, success=True)  # fmt: skip
        if mode in ("server_error", "server_error_recovered"):
            print("ERROR litellm.APIConnectionError: Connection refused")
        result_mode = dict(m.split("=", 1) for m in args.result_mode).get(bundle_id, "normal")
        _write_result(result_mode, groups, units)
        print("Swarm result\nsuccess True", flush=True)
        return 0
    finally:
        _conc(state, -1)


def dashboard(args: argparse.Namespace, rest: list[str]) -> int:
    state = Path(args.state)
    port = int(rest[rest.index("--port") + 1])
    out = state / "dashboard_events.jsonl"
    (state / "dashboard_argv.json").write_text(json.dumps(rest))
    (state / "dashboard.pid").write_text(str(os.getpid()))
    time.sleep(args.dash_delay)  # a slow-starting dashboard

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"status": "ok", "project": "fake"}).encode()
            self.send_response(200 if self.path == "/api/health" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(n)
            with out.open("ab") as fh:
                fh.write(data.replace(b"\n", b" ") + b"\n")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--state", required=True)
    ap.add_argument("--bundles-root", default="")
    ap.add_argument("--mode", default="pass")
    ap.add_argument("--bundle-mode", action="append", default=[])
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--dash-delay", type=float, default=0.0)
    ap.add_argument("--result-mode", action="append", default=[])
    ap.add_argument("command")
    args, rest = ap.parse_known_args()
    Path(args.state).mkdir(parents=True, exist_ok=True)
    if args.command == "swarm":
        return swarm(args, rest)
    if args.command == "dashboard":
        return dashboard(args, rest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
