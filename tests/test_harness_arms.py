"""Harness-benchmark arms (evals/swarm/arms.py): hermetic, a fake model server, no real agents."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import arms  # noqa: E402
from evals.swarm import run_smoke as rs  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / ".memfun").mkdir()
    (repo / "src" / "m.py").write_text("def f():\n    pass\n")
    (repo / "tests" / "test_m.py").write_text(
        "from src.m import f\ndef test_f():\n    assert f() == 1\n"
    )
    (repo / ".memfun" / "verify.json").write_text(
        json.dumps({"commands": ["python -m pytest -q tests/test_m.py::test_f"]})
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "start"]):
        subprocess.run(["git", *cmd], cwd=repo, env=env, check=True)
    return repo


class _Model(http.server.BaseHTTPRequestHandler):
    reply = "=== FILE: src/m.py ===\n```python\ndef f():\n    return 1\n```"
    status = 200
    finish = "stop"
    seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        if self.status != 200:
            out = json.dumps({"error": {"message": "context_length_exceeded: too long"}})
        else:
            out = json.dumps(
                {
                    "choices": [
                        {"message": {"content": self.reply}, "finish_reason": self.finish}
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }
            )
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *a: object) -> None:  # silence
        pass


@pytest.fixture
def model():
    _Model.seen = []
    _Model.status, _Model.finish = 200, "stop"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _run(repo: Path, tmp_path: Path, base: str, arm: str = "single-shot", *extra: str) -> dict:
    spec = tmp_path / "task.txt"
    spec.write_text("Implement f so that it returns 1.")
    result = tmp_path / "result.json"
    env = {**os.environ, "HOME": str(tmp_path), arms.RESULT_ENV: str(result)}
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evals/swarm/arms.py"),
            arm,
            "--spec-file",
            str(spec),
            "--base-url",
            base,
            "--model",
            "m",
            "--owned",
            "src/m.py",
            "--timeout",
            "20",
            *extra,
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
    )
    return json.loads(result.read_text())


def test_single_shot_writes_the_file(tmp_path: Path, model: str) -> None:
    repo = _repo(tmp_path)
    doc = _run(repo, tmp_path, model, "single-shot", "--repetition-penalty", "1.05")
    assert (repo / "src" / "m.py").read_text() == "def f():\n    return 1\n"
    assert doc["success"] and not doc["no_edits"] and doc["failure_class"] == ""
    assert doc["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "calls": 1}
    sent = _Model.seen[0]
    assert sent["temperature"] == 0.7 and sent["repetition_penalty"] == 1.05
    prompt = sent["messages"][1]["content"]
    assert "Implement f so that it returns 1." in prompt  # the task text, verbatim
    assert "VISIBLE TEST FILE: tests/test_m.py" in prompt and "def f():\n    pass" in prompt


def test_single_shot_never_writes_outside_owned(tmp_path: Path, model: str) -> None:
    repo = _repo(tmp_path)
    _Model.reply = "=== FILE: ../evil.py ===\n```python\nx=1\n```"
    try:
        doc = _run(repo, tmp_path, model)
    finally:
        _Model.reply = "=== FILE: src/m.py ===\n```python\ndef f():\n    return 1\n```"
    assert not (tmp_path / "evil.py").exists()
    assert doc["failure_class"] == "format_error" and doc["no_edits"]


def test_context_length_and_truncation_are_their_own_classes(tmp_path: Path, model: str) -> None:
    repo = _repo(tmp_path)
    _Model.status = 400
    doc = _run(repo, tmp_path, model)
    assert doc["failure_class"] == "context_length_exceeded" and doc["context_length_exceeded"]
    assert not doc["no_edits"]
    _Model.status, _Model.finish = 200, "length"
    (tmp_path / "result.json").unlink()
    doc = _run(repo, tmp_path, model)
    assert doc["failure_class"] == "truncated" and doc["truncated_responses"] == 1


def test_unreachable_server_is_a_harness_crash(tmp_path: Path) -> None:
    doc = _run(_repo(tmp_path), tmp_path, "http://127.0.0.1:9/v1")
    assert doc["failure_class"] == "harness_crash"


def test_visible_test_files_found(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert arms.visible_test_files(repo, arms.visible_command(repo)) == ["tests/test_m.py"]
    assert arms.visible_test_files(repo, None) == []


def test_aider_token_lines() -> None:
    m = arms.AIDER_TOKENS_RE.search("Tokens: 3.2k sent, 410 received.")
    assert m and arms.token_count(m.group(1)) == 3200 and arms.token_count(m.group(2)) == 410


def test_mini_and_aider_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns = arms.parse_args(
        [
            "mini-swe-agent",
            "--spec-file",
            "t",
            "--base-url",
            "http://h/v1",
            "--model",
            "m",
            "--repetition-penalty",
            "1.05",
            "--bin",
            "/x/mini",
        ]
    )
    cmd = arms.mini_command(ns, "TASK", tmp_path / "t.json")
    assert cmd[0] == "/x/mini" and "TASK" in cmd and "-y" in cmd
    assert "model.model_kwargs.extra_body.repetition_penalty=1.05" in cmd
    assert "model.model_name=openai/m" in cmd and "litellm_textbased" in cmd
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = _repo(tmp_path)
    ns = arms.parse_args(
        [
            "aider",
            "--spec-file",
            "t",
            "--base-url",
            "http://h/v1",
            "--model",
            "m",
            "--owned",
            "src/m.py",
            "--bin",
            "/x/aider",
            "--edit-format",
            "whole",
        ]
    )
    acmd = arms.aider_command(ns, repo, "TASK TEXT")
    assert acmd[acmd.index("--edit-format") + 1] == "whole"
    # the task travels on the command line: a file outside the episode is unreadable when isolated
    assert acmd[acmd.index("--message") + 1] == "TASK TEXT" and "--message-file" not in acmd
    assert "--auto-test" in acmd and acmd[acmd.index("--test-cmd") + 1].startswith(
        "python -m pytest"
    )
    assert "--yes-always" in acmd and "--no-auto-commits" in acmd and acmd[-1] == "src/m.py"
    settings = json.loads((tmp_path / "aider-model-settings.json").read_text())
    assert settings[0]["extra_params"]["temperature"] == 0.7


def test_run_smoke_arm_cmd_is_identical_task_file(tmp_path: Path) -> None:
    kw = dict(
        sampling=rs.Sampling(temperature=0.7, repetition_penalty=1.05),
        max_tokens=32768,
        timeout=1500.0,
    )
    cmds = [
        rs.arm_cmd(a, tmp_path / "task.txt", "http://h/v1", "m", ["a.py"], **kw) for a in arms.ARMS
    ]
    for c in cmds:
        assert c[c.index("--spec-file") + 1] == str(tmp_path / "task.txt")
        assert c[c.index("--temperature") + 1] == "0.7"
        assert c[c.index("--repetition-penalty") + 1] == "1.05"


def test_arm_crash_is_a_harness_error_not_a_partial() -> None:
    assert rs.classify(arm_crash="boom", any_output=False, hidden_passed=7)[0] == "harness_error"
    assert rs.classify(arm_crash="boom", any_output=True, hidden_passed=7)[0] == "partial"


def test_run_in_group_reaps_a_command_that_wrote_its_result_but_never_exits(
    tmp_path: Path,
) -> None:
    done = tmp_path / "result.json"
    code = f"import pathlib,time; pathlib.Path({str(done)!r}).write_text('{{}}'); time.sleep(120)"
    out = rs.run_in_group(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "log",
        timeout=60,
        grace=1.0,
        done_path=done,
        done_grace=1.0,
    )
    assert out.reaped and not out.timed_out and out.returncode == 0
