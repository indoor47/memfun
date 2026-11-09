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
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.parse
import uuid
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
    redirect_to: str = ""


@pytest.fixture
def models_server() -> Iterator[tuple[str, type[_Models]]]:
    """Loopback stand-in for an OpenAI-compatible ``GET /v1/models`` (no model behind it)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if _Models.redirect_to:
                self.send_response(302)
                self.send_header("Location", _Models.redirect_to)
                self.end_headers()
                return
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
    _Models.redirect_to = ""
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


# ── leak guard: every fake process carries a per-test token and records its pgid ──

_CURRENT: dict[str, str] = {"token": "", "track": ""}
_SESSION_TOKENS: list[str] = []
#: Tests that deliberately SIGKILL the runner clean up via its pgids.txt themselves.
_ORPHANS_EXPECTED = {"test_partial_results_and_pgids_survive_a_sigkilled_runner"}


def _cmdline(pid: int) -> str:
    proc = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return proc.stdout


def _live_tracked(track: Path, token: str) -> list[tuple[int, int]]:
    """``(pid, pgid)`` of recorded fake processes still alive AND still carrying ``token``."""
    out: list[tuple[int, int]] = []
    for rec in sorted(track.glob("*")) if track.is_dir() else []:
        pgid_s, _, tok = rec.read_text().partition(" ")
        pid = int(rec.name)
        if tok != token or token not in _cmdline(pid):  # guards against pid reuse
            continue
        with contextlib.suppress(ProcessLookupError):
            if os.getpgid(pid) == int(pgid_s):
                out.append((pid, int(pgid_s)))
    return out


def _live_by_token(tokens: list[str]) -> list[tuple[int, int]]:
    """``(pid, pgid)`` of ANY process whose command line carries one of ``tokens``.

    Catches fakes that died off before recording themselves (killed during startup).
    """
    listing = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,command="], capture_output=True, text=True, check=False
    ).stdout
    out: list[tuple[int, int]] = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and any(t in parts[2] for t in tokens) and int(parts[0]) != os.getpid():
            out.append((int(parts[0]), int(parts[1])))
    return out


def _kill_leaked(leaked: list[tuple[int, int]]) -> list[str]:
    """Describe, then SIGKILL the groups of ``leaked`` (token-verified pids), then wait."""
    details = [f"{pid} (pgid {pgid}): {_cmdline(pid).strip()[:200]}" for pid, pgid in leaked]
    for _pid, pgid in leaked:
        if pgid > 1 and pgid != os.getpgid(0):  # never our own group
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
    for pid, _pgid in leaked:
        _wait_dead(pid)
    return details


@pytest.fixture(autouse=True)
def fake_process_guard(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    """Kill (by process group, token-verified) any fake the test left alive, and fail."""
    token = f"rstok-{uuid.uuid4().hex}"
    track = tmp_path_factory.mktemp("fake-track")
    _CURRENT.update(token=token, track=str(track))
    _SESSION_TOKENS.append(token)
    try:
        yield token
    finally:
        _CURRENT.update(token="", track="")
        leaked = sorted(set(_live_tracked(track, token)) | set(_live_by_token([token])))
        details = _kill_leaked(leaked)
        if leaked and request.node.originalname not in _ORPHANS_EXPECTED:
            pytest.fail(f"test leaked fake memfun processes (killed in teardown): {details}")


@pytest.fixture(scope="session", autouse=True)
def no_fake_survivors() -> Iterator[None]:
    yield
    alive = _kill_leaked(_live_by_token(_SESSION_TOKENS))
    assert not alive, f"processes from this test session were still alive (killed): {alive}"


def _fake_cmd(
    state: Path,
    modes: dict[str, str] | None = None,
    hold: float = 0.0,
    result_modes: dict[str, str] | None = None,
) -> str:
    cmd = [sys.executable, str(FAKE), "--state", str(state), "--bundles-root", str(SMOKE)]
    if _CURRENT["token"]:
        cmd += ["--fake-token", _CURRENT["token"], "--track-dir", _CURRENT["track"]]
    for bundle, mode in (modes or {}).items():
        cmd += ["--bundle-mode", f"{bundle}={mode}"]
    for bundle, mode in (result_modes or {}).items():
        cmd += ["--result-mode", f"{bundle}={mode}"]
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
    result_modes: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], Path]:
    state = tmp_path / "state"
    argv = [
        "--base-url",
        base_url,
        "--bundles",
        *bundles,
        "--memfun-cmd",
        _fake_cmd(state, modes, hold, result_modes),
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
        assert not any(u["passed"] for u in ep["grade"]["units"].values()), bid
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


def test_broken_bundle_and_broken_reference_are_harness_errors(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    root = tmp_path / "bundles"
    shutil.copytree(SMOKE / "formats-2", root / "formats-2")
    (root / "formats-2" / "hidden" / "test_duration.py").unlink()  # missing bundle file
    shutil.copytree(SMOKE / "textkit-3", root / "textkit-3")
    (root / "textkit-3" / "solution" / "textkit" / "wrap.py").write_text("def wrap(:\n")
    _, results, state = _run(
        tmp_path, base_url, ["formats-2", "textkit-3"], "--bundles-root", str(root)
    )
    eps = _episodes(results)
    assert eps["formats-2"]["outcome"] == "harness_error"
    assert "missing/invalid bundle files" in eps["formats-2"]["reason"]
    assert eps["textkit-3"]["outcome"] == "harness_error"  # the grader itself is suspect
    assert "reference solution does not pass" in eps["textkit-3"]["reason"]
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
               "GIT_CONFIG_NOSYSTEM", "__CF_USER_TEXT_ENCODING", "LC_CTYPE",
               rs.SWARM_RESULT_ENV, rs.NO_DASHBOARD_ENV, rs.TRUST_PROJECT_VERIFY_ENV}  # fmt: skip
    for trial in range(1, 5):
        call = json.loads((state / "calls" / f"formats-2-t{trial}.json").read_text())
        env = call["env"]
        assert set(env) <= allowed, set(env) - allowed
        assert env[rs.DUMMY_KEY_ENV] == rs.DUMMY_KEY_VALUE
        assert env[rs.NO_DASHBOARD_ENV] == "1"
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
    assert _wait_dead(int((state / "dashboard.pid").read_text()))  # the process, not the port


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
    assert dash.proc is not None and _wait_dead(dash.proc.pid)


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
        os.kill(int((other_state / "dashboard.pid").read_text()), 0)  # process still alive
        assert (other_state / "dashboard_events.jsonl").read_text().strip()
    finally:
        owner.send_signal(signal.SIGKILL)
        owner.wait()
    assert _wait_dead(owner.pid)


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
    no_out: dict[str, Any] = {"any_output": False}
    assert rs.classify(timed_out=True, timeout_s=5, **no_out)[1].startswith(
        "episode timeout after 5s"
    )
    # with owned output, a timeout / crash is graded on its merits (no pass-rate exclusion)
    assert rs.classify(timed_out=True, all_pass=True)[0] == "pass"
    assert rs.classify(timed_out=True, hidden_passed=0)[0] == "model_failure"
    assert rs.classify(returncode=1, produced_result=False, **no_out)[0] == "harness_error"
    assert rs.classify(returncode=1, produced_result=False, hidden_passed=2)[0] == "partial"
    # a swarm that ran but reported failure is the model's problem, not the harness's
    assert rs.classify(returncode=1, produced_result=True)[0] == "model_failure"
    assert rs.classify(grade_error="grader crashed (pytest rc=3)")[0] == "harness_error"
    assert rs.classify(all_pass=True, hidden_passed=9)[0] == "pass"
    assert rs.classify(all_pass=True, server_errors=2)[0] == "pass"
    assert rs.classify(hidden_passed=0, server_errors=1, **no_out)[0] == "harness_error"
    assert rs.classify(hidden_passed=0, server_errors=1)[0] == "model_failure"  # symmetric
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


# ── audit fixes: symlink-safe collection ───────────────────────────────────────

TEXTKIT_SOL = SMOKE / "textkit-3" / "solution"


def _repo_with_worktree(tmp_path: Path) -> tuple[rs.Bundle, Path, Path]:
    bundle = rs.load_bundle(SMOKE / "textkit-3")
    repo = tmp_path / "repo"
    shutil.copytree(bundle.path / "repo", repo)
    wt = repo / ".memfun" / "worktrees" / "wf1" / "T1"
    shutil.copytree(bundle.path / "repo", wt)
    for name in ("wrap.py", "roman.py"):  # two legit regular-file units
        shutil.copy(TEXTKIT_SOL / "textkit" / name, wt / "textkit" / name)
    return bundle, repo, wt


def _collect_grade(bundle: rs.Bundle, repo: Path, tmp_path: Path) -> tuple[Any, Any]:
    dest = tmp_path / "collected"
    dest.mkdir()
    collected = rs.collect_owned(bundle, repo, dest)
    grade = rs.grade_candidate(bundle, dest, tmp_path / "g", 120)
    return collected, grade


def _plant(case: str, repo: Path, wt: Path) -> None:
    sol_file = TEXTKIT_SOL / "textkit" / "slugify.py"
    leaf = wt / "textkit" / "slugify.py"
    if case == "absolute":
        leaf.unlink()
        leaf.symlink_to(sol_file.resolve())
    elif case == "relative":
        leaf.unlink()
        leaf.symlink_to(os.path.relpath(sol_file.resolve(), leaf.parent.resolve()))
    elif case == "parent_component":
        shutil.rmtree(wt / "textkit")
        (wt / "textkit").symlink_to((TEXTKIT_SOL / "textkit").resolve())
    elif case == "root_parent_component":
        shutil.rmtree(repo / ".memfun")
        shutil.rmtree(repo / "textkit")
        (repo / "textkit").symlink_to((TEXTKIT_SOL / "textkit").resolve())
    elif case == "worktree_dir":
        shutil.rmtree(repo / ".memfun")
        (repo / ".memfun" / "worktrees" / "wf1").mkdir(parents=True)
        (repo / ".memfun" / "worktrees" / "wf1" / "T9").symlink_to(TEXTKIT_SOL.resolve())
    elif case == "worktrees_root":
        shutil.rmtree(repo / ".memfun")
        fake = repo.parent / "elsewhere" / "wf1"
        shutil.copytree(TEXTKIT_SOL, fake / "T1")
        (repo / ".memfun").mkdir()
        (repo / ".memfun" / "worktrees").symlink_to(fake.parent.resolve())


@pytest.mark.parametrize(
    "case",
    [
        "absolute",
        "relative",
        "parent_component",
        "root_parent_component",
        "worktree_dir",
        "worktrees_root",
    ],
)
def test_symlinked_owned_files_never_count_as_output(tmp_path: Path, case: str) -> None:
    bundle, repo, wt = _repo_with_worktree(tmp_path)
    _plant(case, repo, wt)
    collected, grade = _collect_grade(bundle, repo, tmp_path)
    assert collected["tampering"], case
    assert grade["units"]["slugify"]["passed"] is False, case
    assert not (tmp_path / "collected" / "textkit" / "slugify.py").exists()
    assert collected["units"]["slugify"]["output"] is False
    if case in ("absolute", "relative", "parent_component", "root_parent_component"):
        assert collected["units"]["slugify"]["symlink_tampering"] is True


def test_legit_regular_files_still_pass(tmp_path: Path) -> None:
    bundle, repo, wt = _repo_with_worktree(tmp_path)
    shutil.copy(TEXTKIT_SOL / "textkit" / "slugify.py", wt / "textkit" / "slugify.py")
    collected, grade = _collect_grade(bundle, repo, tmp_path)
    assert collected["tampering"] == []
    assert grade["all_pass"] is True


def test_fifo_and_oversized_owned_files_are_refused(tmp_path: Path) -> None:
    bundle, repo, wt = _repo_with_worktree(tmp_path)
    (wt / "textkit" / "slugify.py").unlink()
    os.mkfifo(wt / "textkit" / "slugify.py")  # opening it must not block
    (wt / "textkit" / "wrap.py").write_bytes(b"#" * (rs.MAX_OWNED_BYTES + 1))
    dest = tmp_path / "collected"
    dest.mkdir()
    collected = rs.collect_owned(bundle, repo, dest)
    problems = " | ".join(collected["tampering"])
    assert "not a regular file" in problems and "too large" in problems
    assert collected["units"]["slugify"]["output"] is False
    assert collected["units"]["wrap"]["output"] is False
    assert collected["units"]["roman"]["output"] is True


def test_runner_flags_symlinked_output_and_does_not_pass(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, _ = _run(tmp_path, base_url, ["textkit-3"], modes={"textkit-3": "symlink"})
    ep = results["episodes"][0]
    assert ep["outcome"] != "pass"
    assert "symlink_tampering" in ep["flags"]
    assert all(u["status"] == "fail" for u in ep["units"].values())


def test_write_private_replaces_a_planted_symlink_without_following_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (tmp_path / "episode.json").symlink_to(victim)
    rs.write_private(tmp_path / "episode.json", "{}")
    assert victim.read_text() == "keep"
    assert not (tmp_path / "episode.json").is_symlink()


def test_archive_refuses_a_planted_artifact_dir(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.py").write_text("x")
    ep = tmp_path / "ep"
    ep.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ep / "graded").symlink_to(elsewhere)
    assert "not archived" in rs.archive_dir(src, ep, "graded")
    assert list(elsewhere.iterdir()) == []


# ── audit fixes: one unit's crash only fails that unit ─────────────────────────

CRASHES = {
    "sys_exit": "import sys\nsys.exit(0)\n",
    "system_exit": "raise SystemExit(0)\n",
    "keyboard_interrupt": "raise KeyboardInterrupt\n",
    "os_exit": "import os\nos._exit(7)\n",
    "syntax_error": "def broken(:\n",
    "pytest_exit": "import pytest\npytest.exit('bye')\n",
    "infinite_loop": "while True:\n    pass\n",
}


@pytest.mark.parametrize("mode", list(CRASHES))
def test_a_crashing_unit_does_not_zero_the_others(tmp_path: Path, mode: str) -> None:
    bundle = rs.load_bundle(SMOKE / "textkit-3")
    cand = tmp_path / "cand"
    shutil.copytree(TEXTKIT_SOL, cand)
    slug = cand / "textkit" / "slugify.py"
    slug.write_text(slug.read_text() + "\n" + CRASHES[mode])  # runs at import
    grade = rs.grade_candidate(bundle, cand, tmp_path / "g", 8 if mode == "infinite_loop" else 120)
    units = grade["units"]
    assert units["slugify"]["passed"] is False
    assert units["wrap"]["passed"] is True and units["roman"]["passed"] is True
    assert units["wrap"]["tests"]["passed"] > 0
    if mode == "infinite_loop":
        assert units["slugify"]["timeout"] is True and "slugify" in grade["timeouts"]
    outcome, _ = rs.classify(hidden_passed=grade["hidden_passed"], all_pass=grade["all_pass"])
    assert outcome == "partial"


def test_runner_unguarded_exit_in_one_unit_is_partial_not_harness_error(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, _ = _run(tmp_path, base_url, ["textkit-3"], modes={"textkit-3": "exit_unit"})
    ep = results["episodes"][0]
    assert ep["outcome"] == "partial", ep["reason"]
    assert ep["units"]["slugify"]["status"] == "fail"
    assert ep["units"]["wrap"]["status"] == "pass"
    assert results["overall"]["outcomes"]["harness_error"] == 0


def test_timeout_with_output_is_graded_and_recovered_server_errors_stay_pass(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, _ = _run(
        tmp_path, base_url, ["formats-2", "textkit-3"], "--episode-timeout", "4",
        "--parallel-tasks", "2",
        modes={"formats-2": "hang_after_output", "textkit-3": "server_error_recovered"},
    )  # fmt: skip
    eps = _episodes(results)
    assert eps["formats-2"]["outcome"] == "pass"
    assert "episode_timeout" in eps["formats-2"]["flags"]
    assert eps["textkit-3"]["outcome"] == "pass"
    assert "server_errors" in eps["textkit-3"]["flags"] and eps["textkit-3"]["server_errors"] >= 1


# ── audit fixes: signals, partial results, operator cleanup ────────────────────


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_during_dashboard_startup_leaves_no_dashboard(
    tmp_path: Path, models_server: tuple[str, type[_Models]], sig: signal.Signals
) -> None:
    base_url, _ = models_server
    state = tmp_path / "state"
    cmd = [*shlex.split(_fake_cmd(state)), "--dash-delay", "120"]
    runner = subprocess.Popen(
        [
            sys.executable, str(SWARM_EVALS / "run_smoke.py"), "--base-url", base_url,
            "--bundles", "formats-2", "--memfun-cmd", shlex.join(cmd),
            "--out-dir", str(tmp_path / "runs"), "--run-id", "r1",
            "--dashboard-port", str(_free_port()), "--kill-grace", "2",
        ],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        pidfile = state / "dashboard.pid"
        for _ in range(1200):
            if pidfile.is_file() and pidfile.read_text():
                break
            time.sleep(0.05)
        dash_pid = int(pidfile.read_text())
        runner.send_signal(sig)
        assert runner.wait(timeout=120) == 130
        assert _wait_dead(dash_pid)  # the process itself is gone, not just the port
        pgids = (tmp_path / "runs" / "r1" / "pgids.txt").read_text()
        assert f"{dash_pid} _dashboard" in pgids
        assert not (state / "calls").exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(runner.pid, signal.SIGKILL)
        runner.wait()


def test_partial_results_and_pgids_survive_a_sigkilled_runner(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    state = tmp_path / "state"
    run_dir = tmp_path / "runs" / "r1"
    runner = subprocess.Popen(
        [
            sys.executable, str(SWARM_EVALS / "run_smoke.py"), "--base-url", base_url,
            "--bundles", "formats-2", "textkit-3", "--parallel-tasks", "2",
            "--memfun-cmd", _fake_cmd(state, {"textkit-3": "hang"}),
            "--out-dir", str(tmp_path / "runs"), "--run-id", "r1", "--no-dashboard",
            "--episode-timeout", "600",
        ],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    pgid = 0
    try:
        for _ in range(2400):
            done = (run_dir / "results.json").is_file() and (state / "hang_child.pid").is_file()
            if done:
                break
            time.sleep(0.05)
        os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
        partial = json.loads((run_dir / "results.json").read_text())
        assert [e["episode"] for e in partial["episodes"]] == ["formats-2-t1"]
        assert partial["episodes"][0]["outcome"] == "pass"
        lines: dict[str, str] = {}
        for line in (run_dir / "pgids.txt").read_text().splitlines():
            group, episode = line.split()
            lines[episode] = group
        pgid = int(lines["textkit-3-t1"])
        os.killpg(pgid, 0)  # the orphaned swarm group is still there ...
        os.killpg(pgid, signal.SIGTERM)  # ... and the README's cleanup removes it
        assert _wait_dead(int((state / "hang_child.pid").read_text()))
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(runner.pid, signal.SIGKILL)
        if pgid:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)


def test_kill_groups_uses_one_shared_grace_and_kills_term_ignorers(tmp_path: Path) -> None:
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    procs = [
        subprocess.Popen([sys.executable, "-c", code], start_new_session=True) for _ in range(3)
    ]
    try:
        time.sleep(0.5)  # let them install the handler (not an assertion)
        rs.kill_groups([p.pid for p in procs], grace=1.0)
        for p in procs:
            assert p.wait(timeout=30) == -signal.SIGKILL
    finally:
        for p in procs:
            with contextlib.suppress(ProcessLookupError):
                p.kill()


def test_forwarder_finish_skips_the_final_drain_when_the_thread_is_stuck(tmp_path: Path) -> None:
    release = threading.Event()

    class Stuck(rs.EventForwarder):
        pumps = 0

        def run(self) -> None:
            release.wait(30)

        def pump(self) -> None:
            Stuck.pumps += 1

    fwd = Stuck(tmp_path / "x.db", 1)
    fwd.start()
    fwd.finish(join_timeout=0.2)
    assert fwd.stuck is True and Stuck.pumps == 0
    release.set()
    fwd.join(10)


def test_read_events_is_strictly_read_only(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "memfun.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE other (x)")
    con.commit()
    con.close()
    before = db.read_bytes()
    assert rs.read_events(db) == []  # no events table yet: empty, retried later
    assert db.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["memfun.db"]


def test_cross_host_redirect_is_refused(models_server: tuple[str, type[_Models]]) -> None:
    base_url, models = models_server
    port = urllib.parse.urlsplit(base_url).port
    models.redirect_to = f"http://localhost:{port}/v1/models"
    _, err = rs.preflight(base_url, None)
    assert "cross-host redirect" in err


# ── audit fixes: argument validation, out-dir ──────────────────────────────────


@pytest.mark.parametrize(
    "extra",
    [
        ["--bundles", "../../../X"],
        ["--bundles", ","],
        ["--bundles", "no-such-bundle"],
        ["--max-agents", "0"],
        ["--max-agents", "101"],
        ["--parallel-tasks", "0"],
        ["--parallel-tasks", "257"],
        ["--trials", "0"],
        ["--run-id", "a..b"],
        ["--run-id", "a/b"],
        ["--metrics-url", "file:///etc/passwd"],
        ["--out-dir", str(REPO_ROOT / "evals" / "swarm" / "runs")],
    ],
)
def test_bad_arguments_are_refused(
    tmp_path: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--base-url", "http://127.0.0.1:18000/v1", "--out-dir", str(tmp_path / "runs")]
    rc = rs.main([*argv, "--memfun-cmd", "false", *extra])
    assert rc == 2, extra
    assert "run_smoke:" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
    assert not (REPO_ROOT / "evals" / "swarm" / "runs").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_bundles_are_deduped_and_non_loopback_urls_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rs.main([
        "--base-url", "http://10.1.2.3:18000/v1", "--bundles", "formats-2,formats-2",
        "textkit-3", "--out-dir", str(tmp_path / "runs"), "--dry-plan",
    ])  # fmt: skip
    cap = capsys.readouterr()
    assert rc == 0 and "episodes      : 2" in cap.out
    assert "WARNING: --base-url" in cap.err and "NOT a loopback" in cap.err


def test_default_out_dir_is_outside_the_checkout() -> None:
    out = rs.DEFAULT_OUT_DIR.resolve()
    assert out != REPO_ROOT and REPO_ROOT not in out.parents


def test_episode_repo_is_its_own_pytest_rootdir(
    tmp_path: Path, models_server: tuple[str, type[_Models]]
) -> None:
    base_url, _ = models_server
    _, results, _ = _run(tmp_path, base_url, ["formats-2"])
    repo = Path(results["episodes"][0]["repo"])
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider", "tests"],
        cwd=repo, capture_output=True, text=True, timeout=120,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )  # fmt: skip
    assert f"rootdir: {repo.resolve()}" in proc.stdout or f"rootdir: {repo}" in proc.stdout
    assert "configfile: pytest.ini" in proc.stdout


# ── tool usage from the swarm's result file ───────────────────────────────────


def test_tool_usage_counts_from_the_swarm_result_file(
    tmp_path: Path, models_server: tuple[str, type[_Models]], capsys: pytest.CaptureFixture[str]
) -> None:
    base_url, _ = models_server
    _, results, state = _run(tmp_path, base_url, ["textkit-3", "minicli-3"],
                             "--parallel-tasks", "2", modes={"minicli-3": "single"})  # fmt: skip
    call = json.loads((state / "calls" / "textkit-3-t1.json").read_text())
    ep_dir = tmp_path / "runs" / "r1" / "textkit-3" / "1"
    assert call["env"][rs.SWARM_RESULT_ENV] == str(ep_dir / rs.SWARM_RESULT_NAME)
    eps = _episodes(results)
    tk = eps["textkit-3"]
    assert tk["tool_usage_source"] == "swarm-result.json"
    tu = tk["tool_usage"]
    # T1: grep find read read write | T2: read read write | T3: grep read read write
    assert tu["counts"] == {"grep": 2, "find_files": 1, "read": 6, "write": 3}
    assert tu["sub_tasks"]["T1"]["counts"] == {"grep": 1, "find_files": 1, "read": 2,
                                               "write": 1}  # fmt: skip
    assert tu["used_search_tools"] is True and tu["sub_tasks_used_search"] == 2
    assert tk["n_sub_tasks"] == 3 and tk["agents_used"] == 3 and tk["unaccepted_tasks"] == []
    assert tk["produced_result"] is True
    mc = eps["minicli-3"]
    assert mc["n_sub_tasks"] == 1 and "decomposition_fallback" in mc["flags"]
    block = results["bundles"]["textkit-3"]["tool_usage"]
    assert block["episodes_with_data"] == 1 and block["mean_counts"]["read"] == 6.0
    assert block["episodes_used_search"] == 1.0
    assert block["sub_tasks_used_search"] == round(2 / 3, 4)
    overall = results["overall"]["tool_usage"]
    assert overall["sub_tasks_used_search"] == round(3 / 4, 4)  # minicli's single T1 searched
    out = capsys.readouterr().out
    assert "tool usage" in out and "textkit-3      grep 2 find 1 read 6" in out


@pytest.mark.parametrize(
    ("result_mode", "problem"),
    [
        ("none", "missing"),
        ("symlink", "symlink"),
        ("invalid", "not JSON"),
        ("schema2", "unsupported schema_version 2"),
    ],
)
def test_missing_or_untrusted_result_file_falls_back(
    tmp_path: Path, models_server: tuple[str, type[_Models]], result_mode: str, problem: str
) -> None:
    base_url, _ = models_server
    _, results, _ = _run(
        tmp_path, base_url, ["textkit-3"], result_modes={"textkit-3": result_mode}
    )
    ep = results["episodes"][0]
    assert ep["tool_usage_source"] == "unavailable"
    assert problem in ep["tool_usage"]["problem"]
    assert ep["outcome"] == "pass"  # the fallback (log + events) still works
    assert ep["n_sub_tasks"] == 3 and ep["agents_used"] == 3  # from events
    assert ep["produced_result"] is True  # from the log marker
    assert results["bundles"]["textkit-3"]["tool_usage"]["episodes_without_data"] == 1


def test_result_file_with_a_foreign_workflow_id_is_ignored(tmp_path: Path) -> None:
    (tmp_path / rs.SWARM_RESULT_NAME).write_text(
        json.dumps({"schema_version": 1, "workflow_id": "wf-other", "sub_tasks": {}})
    )
    doc, problem = rs.load_swarm_result(tmp_path, ["wf-fake01"])
    assert doc is None and "not in the episode's events" in problem
    doc, problem = rs.load_swarm_result(tmp_path, [])
    assert doc is not None and problem == ""


# ── runner: no orphan window between spawning the dashboard and registering it ─


def test_signal_right_after_dashboard_spawn_still_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM arrives the instant Popen returns (before register): no orphan.

    The signal is sent to the MAIN thread (``pthread_kill``), the thread whose mask the
    deferral blocks.  ``os.kill(os.getpid(), ...)`` is process-directed: on Linux the
    kernel may hand it to any thread that does not block SIGTERM (a leftover server
    thread from an earlier test is enough), CPython then runs the Python-level handler
    in the main thread at once and the test raced ``register()`` (issue #73).  A few idle
    threads are started on purpose so that situation exists here too.
    """
    state = tmp_path / "state"
    spawned: list[int] = []
    real_popen = subprocess.Popen
    release = threading.Event()
    idle = [threading.Thread(target=release.wait, daemon=True) for _ in range(4)]
    for t in idle:
        t.start()

    def popen_then_sigterm(*args: Any, **kwargs: Any) -> Any:
        proc = real_popen(*args, **kwargs)
        spawned.append(proc.pid)
        main = threading.main_thread().ident
        assert main is not None
        signal.pthread_kill(main, signal.SIGTERM)  # pending behind the main thread's mask
        return proc

    previous = rs._install_signals()  # pyright: ignore[reportPrivateUsage]
    started: list[rs.SharedDashboard] = []
    try:
        monkeypatch.setattr(rs.subprocess, "Popen", popen_then_sigterm)
        with pytest.raises(SystemExit):
            rs.start_dashboard(
                shlex.split(_fake_cmd(state)), _free_port(), tmp_path / "d",
                register=started.append,
            )  # fmt: skip
    finally:
        monkeypatch.undo()
        rs._restore_signals(previous)  # pyright: ignore[reportPrivateUsage]
        for dash in started:  # what run()'s finally does
            dash.stop()
        release.set()
    assert spawned and started and started[0].lock_pid == spawned[0]
    assert _wait_dead(spawned[0])


def test_dashboard_child_does_not_inherit_blocked_signals(tmp_path: Path) -> None:
    """The deferral is parent-side only: SIGTERM alone still stops the dashboard."""
    state = tmp_path / "state"
    dash = rs.start_dashboard(shlex.split(_fake_cmd(state)), _free_port(), tmp_path / "d")
    assert dash.error == "" and dash.proc is not None
    try:
        os.kill(dash.proc.pid, signal.SIGTERM)  # just SIGTERM, no group kill, no SIGKILL
        assert dash.proc.wait(timeout=30) == -signal.SIGTERM
    finally:
        dash.stop()


def test_episode_env_forbids_dashboard_autostart_and_has_no_token(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "MEMFUN_DASHBOARD_TOKEN": "secret-token",
        "MEMFUN_INFERENCE_METRICS_URL": "http://127.0.0.1:1/metrics",
    }
    env = rs.child_env(tmp_path, base)
    assert env[rs.NO_DASHBOARD_ENV] == "1"
    assert "MEMFUN_DASHBOARD_TOKEN" not in env
    assert "secret-token" not in env.values()


def test_swarm_does_not_start_its_own_dashboard_when_autostart_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No lockfile + MEMFUN_NO_DASHBOARD_AUTOSTART: nothing is started, nothing binds."""
    from memfun_cli.commands import chat

    monkeypatch.chdir(tmp_path)  # no .memfun/dashboard.lock here
    monkeypatch.setenv(rs.NO_DASHBOARD_ENV, "1")
    assert rs.NO_DASHBOARD_ENV == chat.NO_DASHBOARD_AUTOSTART_ENV
    import uvicorn

    runtime = types.SimpleNamespace(event_bus=object())
    started: list[object] = []

    def _record(*_a: object, **_k: object) -> None:
        started.append(1)
        raise RuntimeError("stop here: a dashboard was about to start")

    from memfun_cli.dashboard import server as dash_server

    monkeypatch.setattr(dash_server, "create_app", lambda **_k: object())
    monkeypatch.setattr(uvicorn, "Server", _record)
    got = asyncio.run(
        chat._start_dashboard_background(runtime)  # pyright: ignore[reportPrivateUsage]
    )
    assert got == (None, False) and started == []
    # control: without the flag the same call does try to start one
    monkeypatch.delenv(rs.NO_DASHBOARD_ENV)
    asyncio.run(chat._start_dashboard_background(runtime))  # pyright: ignore[reportPrivateUsage]
    assert started == [1]


def test_worker_reasoning_effort_becomes_a_role_override_in_the_episode_config() -> None:
    import tomllib

    doc = tomllib.loads(
        rs.memfun_toml(
            "http://h/v1",
            "m",
            1000,
            rs.Sampling(reasoning_effort="none", worker_reasoning_effort="medium"),
        )
    )
    assert doc["llm"]["reasoning_effort"] == "none"
    assert doc["llm"]["roles"]["worker"] == {"reasoning_effort": "medium"}
    plain = tomllib.loads(rs.memfun_toml("http://h/v1", "m", 1000, rs.Sampling()))
    assert "roles" not in plain["llm"]


def test_fix_attempts_override_lands_in_verify_json(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "b"
    bundle_dir.mkdir()
    (bundle_dir / "task.toml").write_text(
        '[verify]\ncommands = ["pytest -q"]\nmax_fix_attempts = 4\n', "utf-8"
    )
    repo = tmp_path / "repo"
    (repo / ".memfun").mkdir(parents=True)
    bundle = rs.Bundle(id="b", path=bundle_dir)
    rs.write_project_verify(bundle, repo)
    assert json.loads((repo / ".memfun" / "verify.json").read_text())["max_fix_attempts"] == 4
    rs.write_project_verify(bundle, repo, 12)
    assert json.loads((repo / ".memfun" / "verify.json").read_text())["max_fix_attempts"] == 12


def test_arm_step_limit_reaches_mini_and_arms_use_the_episode_url() -> None:
    from evals.swarm import arms

    cmd = rs.arm_cmd(
        "mini-swe-agent",
        Path("t.txt"),
        "http://127.0.0.1:1/e/ep-t1/v1",
        "m",
        ["a.py"],
        sampling=rs.Sampling(),
        max_tokens=100,
        timeout=60.0,
        step_limit=40,
    )
    assert cmd[cmd.index("--base-url") + 1] == "http://127.0.0.1:1/e/ep-t1/v1"
    ns = arms.parse_args(cmd[2:])
    assert ns.step_limit == 40
    assert "agent.step_limit=40" in arms.mini_command(ns, "TASK", Path("traj.json"))
    ns0 = arms.parse_args(
        ["mini-swe-agent", "--spec-file", "t", "--base-url", "u", "--model", "m"]
    )
    assert not any("step_limit" in c for c in arms.mini_command(ns0, "T", Path("x")))
