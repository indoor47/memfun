"""``memfun swarm`` exits once its result is written (hermetic: a fake model server, no network).

It used to print its result and then sit idle until the task timeout: the runtime's aiosqlite
connections run on non-daemon worker threads and nothing closed them.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

ANSWER = (
    "[[ ## reasoning ## ]]\nnone\n\n[[ ## answer ## ]]\nok\n\n"
    "[[ ## operations ## ]]\n[]\n\n[[ ## completed ## ]]"
)


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:
        pass

    def do_GET(self) -> None:
        self._send({"data": [{"id": "m"}]})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def _send(self, doc: dict[str, object]) -> None:
        body = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_close_all_connections_ends_the_worker_threads(tmp_path: Path) -> None:
    from memfun_runtime.backends.sqlite._db import close_all_connections, get_connection

    async def go() -> tuple[int, int, int]:
        before = threading.active_count()
        await get_connection(str(tmp_path / "a.db"))
        await get_connection(str(tmp_path / "b.db"))
        open_threads = threading.active_count() - before
        closed = await close_all_connections()
        again = await close_all_connections()  # idempotent
        return open_threads, closed, again

    open_threads, closed, again = asyncio.run(go())
    assert open_threads == 2 and closed == 2 and again == 0


@pytest.mark.parametrize("dashboard", [False, True])
def test_the_swarm_process_exits_after_it_writes_its_result(
    tmp_path: Path, dashboard: bool
) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proj, home = tmp_path / "proj", tmp_path / "home"
    proj.mkdir()
    home.mkdir()
    (proj / "m.py").write_text("def f():\n    pass\n")
    (proj / "memfun.toml").write_text(
        '[llm]\nprovider = "openai"\nmodel = "m"\napi_key_env = "FAKE_KEY"\n'
        f'base_url = "http://127.0.0.1:{server.server_address[1]}/v1"\n'
    )
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"]):
        subprocess.run(["git", *cmd], cwd=proj, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "i"],
        cwd=proj,
        check=True,
        capture_output=True,
    )
    result = tmp_path / "swarm-result.json"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "FAKE_KEY": "x",
        "MEMFUN_SWARM_RESULT_JSON": str(result),
    }
    if not dashboard:
        env["MEMFUN_NO_DASHBOARD_AUTOSTART"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from memfun_cli.main import main; main()",
            "swarm",
            "implement f",
            "--no-decompose",
            "--no-commit",
            "--no-push",
            "--no-use-worktrees",
            "--max-pool",
            "1",
            "--timeout",
            "60",
        ],
        cwd=proj,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 180
        while not result.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert result.exists(), "the swarm never wrote its result"
        # A hang is binary: a healthy process exits within a second or two of the result, a
        # stuck one never does.  The generous bound only keeps a slow CI machine from flaking.
        code = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("the swarm process was still alive 30 s after it wrote its result")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        server.shutdown()
    assert code == 1  # the fake answer edits no file: the swarm reports failure, not a hang
