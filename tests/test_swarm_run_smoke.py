"""Hermetic tests for ``evals/swarm/run_smoke.py`` (the swarm smoke runner).

A fake ``memfun`` (``tests/_fake_memfun.py``, passed via ``--memfun-cmd``) stands in for
``memfun swarm`` / ``memfun dashboard``; the model server is a tiny loopback ``/v1/models``
responder.  No network, no LLM, no wall-clock assertions: timings come from the fake's
fake clock, concurrency from counters, and timeouts only bound hangs.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SWARM_EVALS = REPO_ROOT / "evals" / "swarm"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

SMOKE = SWARM_EVALS / "smoke"
FAKE = Path(__file__).resolve().parent / "_fake_memfun.py"
ALL: list[str] = list(rs.DEFAULT_BUNDLES)


# ── fixtures / helpers ─────────────────────────────────────────────────────────────


class _Models:
    ids: list[str] = ["fake-qwen"]  # noqa: RUF012


@pytest.fixture
def models_server() -> Iterator[tuple[str, type[_Models]]]:
    """Loopback stand-in for an OpenAI-compatible ``GET /v1/models`` (no model behind it)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"data": [{"id": m} for m in _Models.ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    _Models.ids = ["fake-qwen"]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", _Models
    finally:
        srv.shutdown()
        srv.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _fake_cmd(state: Path, modes: dict[str, str] | None = None, hold: float = 0.0) -> str:
    cmd = [sys.executable, str(FAKE), "--state", str(state), "--bundles-root", str(SMOKE)]
    for bundle, mode in (modes or {}).items():
        cmd += ["--bundle-mode", f"{bundle}={mode}"]
    if hold:
        cmd += ["--hold", str(hold)]
    return shlex.join(cmd)


def _run(
    tmp_path: Path,
    base_url: str,
    bundles: list[str],
    *extra: str,
    modes: dict[str, str] | None = None,
    hold: float = 0.0,
    dashboard: bool = False,
) -> tuple[int, dict[str, Any], Path]:
    state = tmp_path / "state"
    argv = [
        "--base-url",
        base_url,
        "--bundles",
        *bundles,
        "--memfun-cmd",
        _fake_cmd(state, modes, hold),
        "--out-dir",
        str(tmp_path / "runs"),
        "--run-id",
        "r1",
        "--kill-grace",
        "2",
        *extra,
    ]
    if not dashboard:
        argv.append("--no-dashboard")
    rc = rs.main(argv)
    results_path = tmp_path / "runs" / "r1" / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text()) if results_path.is_file() else {}
    )
    return rc, results, state


def _episodes(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["bundle"]: e for e in results["episodes"]}


def _wait_dead(pid: int, tries: int = 400) -> bool:
    """Bounded wait for ``pid`` to disappear (a reparented child is reaped by init)."""
    for _ in range(tries):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(0.05)
    return False


# ── oracle / null through the runner's grader path ───────────────────────────────


def test_reference_solutions_pass_all_bundles_through_runner(
    tmp_path: Path, models_server: tuple[str, type[_Models]], capsys: pytest.CaptureFixture[str]
) -> None:
    base_url, _ = models_server
    rc, results, state = _run(tmp_path, base_url, ALL, "--parallel-tasks", "3")
    assert rc == 0
    assert results["schema_version"] == 1
    for key in ("run_id", "model", "overall", "bundles", "units", "episodes", "void"):
        assert key in results
    assert results["model"] == "fake-qwen"  # discovered from GET /models
    assert results["overall"]["outcomes"]["pass"] == 6
    assert results["overall"]["unit_pass_rate"] == 1.0
    assert results["void"] is False
    eps = _episodes(results)
    assert set(eps) == set(ALL)
    for bid, ep in eps.items():
        assert ep["outcome"] == "pass", (bid, ep["reason"], ep.get("grade"))
        assert ep["grade"]["hidden_failed"] == 0 and ep["grade"]["hidden_passed"] > 0
        assert all(u["status"] == "pass" for u in ep["units"].values())
        assert "decomposition_fallback" not in ep["flags"]
        assert (Path(ep["dir"]) / "episode.json").is_file()
        assert (Path(ep["dir"]) / "swarm.log").is_file()
    # tokens / agents / concurrency from llm.call + task events (fake clock)
    tk = eps["textkit-3"]
    m = tk["metrics"]
    assert m["tokens"]["prompt"] == 50 + 100 + 200 + 300
    assert m["tokens"]["completion"] == 5 + 10 + 20 + 30
    assert m["tokens"]["by_role"]["decomposer"] == {"prompt": 50, "completion": 5}
    assert m["tokens"]["source"] == "llm.call events"
    assert m["agents_used"] == 3
    assert m["peak_concurrency"] == 3
    assert m["decompose_s"] == 2.0
    assert m["exec_wall_s"] == 5.0  # picked up at +2, last completion at +7
    assert tk["collection"]["files"]["textkit/slugify.py"]["source"] == "wf-fake01/T1"
    assert tk["units"]["slugify"]["sub_tasks"] == ["T1"]
    assert results["overall"]["tokens"]["prompt"] == sum(
        e["metrics"]["tokens"]["prompt"] for e in results["episodes"]
    )
    # the episode repo: own memfun.toml with one [llm] block, .gitignore, a commit
    repo = Path(tk["repo"])
    toml = (repo / "memfun.toml").read_text()
    assert 'model = "fake-qwen"' in toml and f'api_key_env = "{rs.DUMMY_KEY_ENV}"' in toml
    assert f'base_url = "{base_url}"' in toml and 'provider = "openai"' in toml
    assert ".memfun/" in (repo / ".gitignore").read_text()
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "smoke start tree" in log.stdout
    call = json.loads((state / "calls" / "textkit-3-t1.json").read_text())
    assert call["argv"][0].startswith("Implement three independent text utilities")
    assert rs.SPEC_SUFFIX in call["argv"][0]
    assert call["argv"][1:] == ["--max-pool", "4", "--timeout", "900", "--no-push"]
    out = capsys.readouterr().out
    assert "ALL" in out and "textkit-3" in out and "kvstores-4/ttl" in out
    assert "VOID" not in out


def test_untouched_trees_pass_no_unit_through_runner(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    modes = dict.fromkeys(ALL, "noop")
    rc, results, _ = _run(tmp_path, base_url, ALL, "--parallel-tasks", "3", modes=modes)
    assert rc == 0
    for bid, ep in _episodes(results).items():
        assert ep["outcome"] in ("model_failure", "partial"), (bid, ep["reason"])
        assert all(u["status"] == "no_output" for u in ep["units"].values()), bid
        assert not any(ep["grade"]["units"].values()), bid
    assert results["overall"]["unit_pass_rate"] == 0.0
    assert all(u["passed"] == 0 for u in results["units"].values())


# ── collection rules and flags ─────────────────────────────────────────────────


def test_failed_unit_missing_unit_duplicates_fallback_and_tampering(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    modes = {
        "textkit-3": "fail_unit",
        "formats-2": "missing",
        "validators-4": "duplicate",
        "minicli-3": "single",
        "kvstores-4": "root",
        "control-calc": "tamper",
    }
    _, results, _ = _run(tmp_path, base_url, ALL, "--parallel-tasks", "3", modes=modes)
    eps = _episodes(results)

    tk = eps["textkit-3"]
    assert tk["outcome"] == "partial"
    assert tk["units"]["slugify"]["status"] == "fail"
    assert tk["units"]["wrap"]["status"] == "pass"

    fm = eps["formats-2"]
    assert fm["outcome"] == "partial"
    assert fm["units"]["inilite"]["status"] == "no_output"
    assert fm["units"]["duration"]["status"] == "pass"

    va = eps["validators-4"]
    assert va["outcome"] == "pass"  # the newest copy (correct) wins
    assert "duplicate_work" in va["flags"]
    assert va["units"]["luhn"]["duplicate_work"] is True
    assert va["units"]["luhn"]["sub_tasks"] == ["T1", "T2"]
    assert va["collection"]["files"]["checks/luhn.py"]["source"] == "wf-fake01/T2"

    mc = eps["minicli-3"]
    assert mc["outcome"] == "pass"
    assert "decomposition_fallback" in mc["flags"]

    kv = eps["kvstores-4"]
    assert kv["outcome"] == "pass"
    assert kv["collection"]["files"]["kv/ttl.py"]["source"] == "root"
    assert "decomposition_fallback" not in kv["flags"]

    cc = eps["control-calc"]
    assert cc["outcome"] == "pass"  # planted conftest files are ignored by the grader
    assert "tests_modified" in cc["flags"]
    assert any(s.endswith(":conftest.py") for s in cc["collection"]["stray_changes"])
    assert "decomposition_fallback" not in cc["flags"]  # control task: 1 unit is expected

    assert results["units"]["textkit-3/slugify"] == {
        "graded": 1,
        "passed": 0,
        "no_output": 0,
        "pass_rate": 0.0,
    }
    assert results["units"]["formats-2/inilite"]["no_output"] == 1


# ── harness errors ─────────────────────────────────────────────────────────────


def test_crash_and_server_errors_are_harness_errors_and_void_the_arm(
    tmp_path: Path, models_server: tuple[str, type[_Models]], capsys: pytest.CaptureFixture[str]
) -> None:
    base_url, _ = models_server
    modes = {"formats-2": "crash", "textkit-3": "server_error", "minicli-3": "warnings"}
    rc, results, _ = _run(
        tmp_path, base_url, ["formats-2", "textkit-3", "minicli-3"], "--parallel-tasks", "3",
        modes=modes,
    )  # fmt: skip
    eps = _episodes(results)
    crash = eps["formats-2"]
    assert crash["outcome"] == "harness_error"
    assert "exited 3 without a result" in crash["reason"]
    assert "fake crash" in crash["log_tail"]
    assert any("Traceback" in w for w in crash["warnings"])

    srv = eps["textkit-3"]
    assert srv["outcome"] == "harness_error"
    assert "model server errors" in srv["reason"]
    assert srv["metrics"]["llm_error_count"] == 3  # one failed worker call per sub-task
    assert srv["metrics"]["tokens"]["calls_with_usage"] == 1  # only the decomposer's
    assert srv["server_errors"] >= 3

    warn = eps["minicli-3"]
    assert warn["outcome"] == "pass"
    assert any("Failed to discover MCP tools" in w for w in warn["warnings"])
    assert any("Task decomposition failed" in w for w in warn["warnings"])

    assert results["overall"]["outcomes"]["harness_error"] == 2
    assert results["void"] is True and rc == 1
    assert {h["episode"] for h in results["harness_errors"]} == {"formats-2-t1", "textkit-3-t1"}
    # harness errors do not count against the model's pass rate
    assert results["overall"]["episode_pass_rate"] == 1.0
    assert "ARM VOID" in capsys.readouterr().out


def test_episode_timeout_kills_the_swarm_process_group(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, state = _run(
        tmp_path, base_url, ["formats-2"], "--episode-timeout", "3", modes={"formats-2": "hang"}
    )
    ep = _episodes(results)["formats-2"]
    assert ep["outcome"] == "harness_error"
    assert "episode timeout" in ep["reason"]
    assert ep["timed_out"] is True
    child = int((state / "hang_child.pid").read_text())
    assert _wait_dead(child)


def test_sigterm_to_the_runner_kills_the_swarm_groups_it_started(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    state = tmp_path / "state"
    runner = subprocess.Popen(
        [
            sys.executable, str(SWARM_EVALS / "run_smoke.py"), "--base-url", base_url,
            "--bundles", "formats-2", "--memfun-cmd", _fake_cmd(state, {"formats-2": "hang"}),
            "--out-dir", str(tmp_path / "runs"), "--run-id", "r1", "--no-dashboard",
            "--episode-timeout", "600", "--kill-grace", "2",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        pidfile = state / "hang_child.pid"
        for _ in range(1200):
            if pidfile.is_file() and pidfile.read_text():
                break
            time.sleep(0.05)
        child = int(pidfile.read_text())
        swarm_pid = json.loads((state / "calls" / "formats-2-t1.json").read_text())["pid"]
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=120) == 130
        assert _wait_dead(swarm_pid) and _wait_dead(child)
        results = json.loads((tmp_path / "runs" / "r1" / "results.json").read_text())
        assert results["interrupted"] is True
        assert results["episodes"][0]["outcome"] == "harness_error"
        assert "aborted" in results["episodes"][0]["reason"]
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(runner.pid, signal.SIGKILL)
        runner.wait()


def test_unreachable_model_server_is_a_harness_error_and_nothing_runs(tmp_path: Path) -> None:
    port = _free_port()  # nothing listens there
    rc, results, state = _run(tmp_path, f"http://127.0.0.1:{port}/v1", ["formats-2", "textkit-3"])
    assert rc == 1 and results["void"] is True
    for ep in results["episodes"]:
        assert ep["outcome"] == "harness_error"
        assert "model server unreachable" in ep["reason"]
    assert not (state / "calls").exists()  # memfun swarm was never started


def test_missing_bundle_is_a_harness_error(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, state = _run(tmp_path, base_url, ["no-such-bundle"])
    ep = results["episodes"][0]
    assert ep["outcome"] == "harness_error"
    assert "missing/invalid bundle files" in ep["reason"]
    assert not (state / "calls").exists()


def test_preflight_model_selection() -> None:
    def get(models: list[str]) -> rs.HttpGet:
        return lambda _url, _t: {"data": [{"id": m} for m in models]}

    assert rs.preflight("http://h/v1", None, get(["q"])) == ("q", "")
    model, err = rs.preflight("http://h/v1", None, get(["a", "b"]))
    assert model == "" and "lists 2 models" in err
    assert rs.preflight("http://h/v1", "a", get(["a", "b"])) == ("a", "")
    assert "not served" in rs.preflight("http://h/v1", "zz", get(["a"]))[1]


# ── parallelism, env ───────────────────────────────────────────────────────────


def test_parallel_tasks_bound_and_clean_child_env(
    tmp_path: Path, models_server: tuple[str, type[_Models]], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, _ = models_server
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "RUNPOD_POD_ID"):
        monkeypatch.setenv(name, "real-looking-secret-value")
    monkeypatch.setenv("MEMFUN_DASHBOARD_TOKEN", "tok-should-not-reach-swarm")
    _, results, state = _run(
        tmp_path, base_url, ["formats-2"], "--trials", "4", "--parallel-tasks", "2",
        "--model", "fake-qwen", hold=0.5,
    )  # fmt: skip
    conc = json.loads((state / "conc.json").read_text())
    assert 1 <= conc["max"] <= 2  # counted by the fake: never more than 2 RUNNING at once
    assert conc["running"] == 0
    assert results["peak_parallel_episodes"] <= 2
    assert [e["trial"] for e in results["episodes"]] == [1, 2, 3, 4]
    assert results["overall"]["outcomes"]["pass"] == 4

    allowed = {*rs.ENV_ALLOW, "HOME", rs.DUMMY_KEY_ENV, "PYTHONUNBUFFERED", "COLUMNS",
               "GIT_CONFIG_NOSYSTEM", "__CF_USER_TEXT_ENCODING", "LC_CTYPE"}  # fmt: skip
    for trial in range(1, 5):
        call = json.loads((state / "calls" / f"formats-2-t{trial}.json").read_text())
        env = call["env"]
        assert set(env) <= allowed, set(env) - allowed
        assert env[rs.DUMMY_KEY_ENV] == rs.DUMMY_KEY_VALUE
        assert env["HOME"] == str(tmp_path / "runs" / "r1" / "formats-2" / str(trial) / "home")
        assert env["HOME"] != os.path.expanduser("~")
        assert "real-looking-secret-value" not in json.dumps(env)
        assert "tok-should-not-reach-swarm" not in json.dumps(env)


def test_run_in_group_kills_only_its_own_group(tmp_path: Path) -> None:
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    )
    try:
        out = rs.run_in_group(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=rs.child_env(tmp_path),
            log_path=tmp_path / "log.txt",
            timeout=3,
            grace=2,
        )
        assert out.timed_out is True
        assert _wait_dead(int(pidfile.read_text()))  # the grandchild died with the group
        assert sleeper.poll() is None  # the unrelated process is untouched
        os.kill(sleeper.pid, 0)
    finally:
        sleeper.send_signal(signal.SIGKILL)
        sleeper.wait()


# ── shared dashboard ─────────────────────────────────────────────────────────


def test_runner_starts_one_dashboard_and_forwards_all_episodes(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    port = _free_port()
    _, results, state = _run(
        tmp_path, base_url, ["formats-2", "textkit-3"], "--parallel-tasks", "2",
        "--dashboard-port", str(port), dashboard=True,
    )  # fmt: skip
    assert results["dashboard"] == {"port": port, "started_by_runner": True}
    assert json.loads((state / "dashboard_argv.json").read_text()) == [
        "--remote",
        "--port",
        str(port),
    ]
    events = [
        json.loads(line)
        for line in (state / "dashboard_events.jsonl").read_text().split("\n")
        if line
    ]
    projects = {e.get("project") for e in events if e.get("event_type") == "task.published"}
    assert projects == {"formats-2-t1", "textkit-3-t1"}
    llm = [e for e in events if e["event_type"] == "llm.call" and e.get("prompt_tokens")]
    assert {e["role"] for e in llm} == {"decomposer", "worker"}
    for ep in results["episodes"]:
        assert ep["outcome"] == "pass"
        assert ep["dashboard"]["post_errors"] == 0
        assert ep["dashboard"]["forwarded"] == ep["metrics"]["event_count"]
        lock = json.loads((Path(ep["repo"]) / ".memfun" / "dashboard.lock").read_text())
        assert lock["port"] == port
        assert f"Dashboard: http://localhost:{port}" in (Path(ep["dir"]) / "swarm.log").read_text()
    assert not rs.dashboard_health(port)  # the runner stopped the dashboard it started


def test_memfun_swarm_reuses_the_shared_dashboard_from_the_lockfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ``memfun swarm`` dashboard helper honours the lockfile the runner writes."""
    from memfun_cli.commands.chat import (
        _start_dashboard_background,  # pyright: ignore[reportPrivateUsage]
    )

    state = tmp_path / "state"
    port = _free_port()
    dash = rs.start_dashboard(shlex.split(_fake_cmd(state)), port, tmp_path / "_dashboard")
    assert dash.error == "" and dash.owned
    try:
        repo = tmp_path / "repo"
        (repo / ".memfun").mkdir(parents=True)
        (repo / ".memfun" / "dashboard.lock").write_text(json.dumps(dash.lock_payload()))
        monkeypatch.chdir(repo)
        got = asyncio.run(_start_dashboard_background(object()))
        assert got == (port, False)  # reused, not a second dashboard
    finally:
        dash.stop()


def test_existing_dashboard_is_reused_and_left_running(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    port = _free_port()
    other_state = tmp_path / "owner"
    owner = subprocess.Popen(
        [*shlex.split(_fake_cmd(other_state)), "dashboard", "--remote", "--port", str(port)],
        start_new_session=True,
    )
    try:
        for _ in range(400):
            if rs.dashboard_health(port):
                break
            time.sleep(0.05)
        _, results, _ = _run(
            tmp_path, base_url, ["formats-2"], "--dashboard-port", str(port), dashboard=True
        )
        assert results["dashboard"] == {"port": port, "started_by_runner": False}
        assert results["episodes"][0]["outcome"] == "pass"
        assert owner.poll() is None and rs.dashboard_health(port)  # not ours: left alone
        assert (other_state / "dashboard_events.jsonl").read_text().strip()
    finally:
        owner.send_signal(signal.SIGKILL)
        owner.wait()


def test_dashboard_port_busy_with_something_else_is_a_harness_error(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = int(s.getsockname()[1])
        _, results, state = _run(
            tmp_path, base_url, ["formats-2"], "--dashboard-port", str(port), dashboard=True
        )
    ep = results["episodes"][0]
    assert ep["outcome"] == "harness_error" and "not a memfun dashboard" in ep["reason"]
    assert not (state / "calls").exists()


# ── CLI guards, dry plan ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--base-url", "http://127.0.0.1:8434/v1"],
        ["--base-url", "https://localhost:8434/v1"],
        ["--base-url", "http://127.0.0.1:18000/v1", "--dashboard-port", "8434"],
        ["--base-url", "ftp://127.0.0.1:18000/v1"],
    ],
)
def test_refuses_forbidden_port_and_bad_urls(
    tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "runs"
    rc = rs.main([*argv, "--out-dir", str(out_dir), "--memfun-cmd", "false"])
    assert rc == 2
    assert "run_smoke:" in capsys.readouterr().err
    assert not out_dir.exists()


def test_dry_plan_prints_everything_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state"
    out_dir = tmp_path / "runs"
    rc = rs.main([
        "--base-url", "http://127.0.0.1:18000/v1", "--bundles", "textkit-3,formats-2",
        "--trials", "2", "--max-agents", "5", "--parallel-tasks", "2",
        "--memfun-cmd", _fake_cmd(state), "--out-dir", str(out_dir), "--run-id", "plan1",
        "--dry-plan",
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY PLAN" in out
    for name in ("textkit-3-t1", "textkit-3-t2", "formats-2-t1", "formats-2-t2"):
        assert name in out
    assert "episodes      : 4" in out
    assert "--max-pool 5" in out and "--timeout 900" in out and "--no-push" in out
    assert "port 18000" in out and "dashboard --remote --port 8081" in out
    assert "GET http://127.0.0.1:18000/v1/models" in out
    assert not out_dir.exists() and not state.exists()


def test_existing_run_dir_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    rc = rs.main([
        "--base-url", "http://127.0.0.1:18000/v1", "--out-dir", str(tmp_path / "runs"),
        "--run-id", "r1", "--no-dashboard",
    ])  # fmt: skip
    assert rc == 2 and "already exists" in capsys.readouterr().err


# ── pure logic ───────────────────────────────────────────────────────────────


def _ev(t: float, etype: str, tid: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"event_type": etype, "ts": t, "task_id": tid, **kw}


def test_event_metrics_concurrency_tokens_and_derived_tasks() -> None:
    events = [
        _ev(0, "workflow.started", workflow_id="wf"),
        _ev(1, "task.published", "wf_A", agent_name="coder-agent"),
        _ev(1, "task.picked_up", "wf_A", agent_name="coder-agent"),
        _ev(2, "task.published", "wf_B", agent_name="test-agent"),
        _ev(2, "task.picked_up", "wf_B", agent_name="test-agent"),
        _ev(
            3,
            "llm.call",
            "wf_A",
            role="worker",
            prompt_tokens=10,
            completion_tokens=1,
            duration_ms=2000,
        ),
        _ev(
            3.5,
            "llm.call",
            "wf_B",
            role="auditor",
            prompt_tokens=20,
            completion_tokens=2,
            duration_ms=1000,
        ),
        _ev(4, "task.completed", "wf_A", success=True),
        _ev(4, "task.published", "wf_A_rev1", agent_name="coder-agent"),
        _ev(4, "task.picked_up", "wf_A_rev1", agent_name="coder-agent"),
        _ev(6, "task.completed", "wf_B", success=False),
        _ev(7, "task.completed", "wf_A_rev1", success=True),
        _ev(7.5, "llm.call", None, role="reviewer", error="boom", duration_ms=100),
    ]
    m = rs.event_metrics(events)
    assert m["peak_concurrency"] == 2  # A ends at 4 exactly when rev1 starts
    assert m["peak_llm_concurrency"] == 2
    assert m["primary_sub_tasks"] == 2  # wf_A_rev1 is derived
    assert m["agents_used"] == 3
    assert m["agent_names"] == ["coder-agent", "test-agent"]
    assert m["tokens"]["prompt"] == 30 and m["tokens"]["completion"] == 3
    assert m["tokens"]["calls"] == 3 and m["tokens"]["calls_with_usage"] == 2
    assert m["tokens"]["by_role"]["auditor"] == {"prompt": 20, "completion": 2}
    assert m["llm_error_count"] == 1
    assert m["decompose_s"] == 1.0 and m["exec_wall_s"] == 6.0
    sub = {s["task_id"]: s for s in m["sub_tasks"]}
    assert sub["wf_B"]["success"] is False and sub["wf_B"]["end_rel_s"] == 6.0


def test_classification_rules() -> None:
    assert rs.classify(fatal="git setup failed")[0] == "harness_error"
    assert rs.classify(timed_out=True, timeout_s=5)[1].startswith("episode timeout after 5s")
    assert rs.classify(returncode=1, produced_result=False)[0] == "harness_error"
    # a swarm that ran but reported failure is the model's problem, not the harness's
    assert rs.classify(returncode=1, produced_result=True)[0] == "model_failure"
    assert rs.classify(grade_error="grader crashed (pytest rc=3)")[0] == "harness_error"
    assert rs.classify(all_pass=True, hidden_passed=9)[0] == "pass"
    assert rs.classify(all_pass=True, server_errors=2)[0] == "pass"
    assert rs.classify(hidden_passed=0, server_errors=1)[0] == "harness_error"
    assert rs.classify(hidden_passed=3)[0] == "partial"
    assert rs.classify(hidden_passed=0)[0] == "model_failure"


def test_void_flag_above_five_percent_harness_errors() -> None:
    def eps(n_harness: int, total: int = 20) -> list[dict[str, Any]]:
        return [
            {"bundle": "b", "trial": i, "episode": f"b-t{i}", "wall_s": 1.0, "flags": [],
             "outcome": "harness_error" if i < n_harness else "pass", "units": {}}
            for i in range(total)
        ]  # fmt: skip

    assert rs.aggregate(eps(1), {})["void"] is False  # exactly 5%
    assert rs.aggregate(eps(2), {})["void"] is True
    res = rs.aggregate(eps(2), {"wall_s": 1.0, "model": "m"})
    assert res["schema_version"] == 1 and len(res["harness_errors"]) == 2
    assert "ARM VOID" in rs.summary_table(res)


def test_runner_constants_match_memfun() -> None:
    from memfun_runtime.distributed import EVENT_TOPIC

    from scripts.runpod import runpod_kit as kit

    assert rs.EVENT_TOPIC == EVENT_TOPIC
    assert rs.DUMMY_KEY_ENV == kit.LLM_KEY_ENV
    assert kit.FORBIDDEN_LOCAL_PORTS <= rs.FORBIDDEN_PORTS
    assert set(rs.DEFAULT_BUNDLES) == {p.parent.name for p in SMOKE.glob("*/task.toml")}


def test_prom_counters_sum_labelled_samples() -> None:
    text = (
        "# TYPE sglang:prompt_tokens_total counter\n"
        'sglang:prompt_tokens_total{model_name="q"} 100\n'
        'sglang:prompt_tokens_total{model_name="r"} 50\n'
        "sglang:generation_tokens_total 7\n"
        "other 1\n"
    )
    assert rs.prom_counters(text, rs.PROM_PROMPT + rs.PROM_GEN) == {
        "sglang:prompt_tokens_total": 150.0,
        "sglang:generation_tokens_total": 7.0,
    }
