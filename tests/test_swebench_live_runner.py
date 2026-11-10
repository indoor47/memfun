"""The SWE-bench-Live runner against a fake ``docker``: the rules it must enforce, hermetic."""

# ruff: noqa: E501  (the embedded fake docker script and table-like test data)

from __future__ import annotations

import itertools
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evals.swebench_live import dataset as ds  # noqa: E402
from evals.swebench_live import runner as rn  # noqa: E402
from tests.test_swebench_live_proxy import Upstream  # noqa: E402

HEAD = "a" * 40
PATCH = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, sys, urllib.request
state = os.environ["FAKE_DOCKER_STATE"]
args = sys.argv[1:]
scen = json.load(open(os.path.join(state, "scenario.json")))
with open(os.path.join(state, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps(args) + "\n")
def mark(name): open(os.path.join(state, name), "w").close()
def has(name): return os.path.exists(os.path.join(state, name))
if args[:2] == ["image", "inspect"]:
    sys.exit(0 if has("pulled") else 1)
if args[0] == "pull":
    if scen.get("pull_fails"): sys.stderr.write("toomanyrequests"); sys.exit(1)
    mark("pulled"); sys.exit(0)
if args[0] == "network":
    if args[1] == "create": mark("net"); sys.exit(0)
    if args[1] == "rm": sys.exit(0)
    if args[1] == "inspect":
        if not has("net"): sys.exit(1)
        if "-f" in args: print("127.0.0.1 0123456789abcdef0123 true")
        sys.exit(0)
if args[0] == "run": print("cid"); sys.exit(0)
if args[0] in ("rm", "rmi", "restart"): sys.exit(0)
if args[0] == "exec":
    step = next(a.split("=", 1)[1] for a in args if a.startswith("MFL_STEP="))
    if step == "findrepo": print("/testbed\n%s" % scen.get("head", "a" * 40))
    elif step == "baseline": print("c" * 40 + "".join("\n" + x for x in scen.get("dirty", [])))
    elif step == "lockdown":
        sys.stdin.read()
        if scen.get("lockdown_fails"): print("lockdown check FAILED: refs beyond HEAD"); sys.exit(1)
        print("lockdown ok: 5 commits reachable, nothing beyond HEAD")
    elif step == "nettest": print(json.dumps(scen.get("net", {"proxy": True, "internet": [], "dns": [], "host_ports": []})))
    elif step == "config":
        open(os.path.join(state, "config.toml"), "w").write(sys.stdin.read())
    elif step == "rollout":
        with open(os.path.join(state, "rollout_env.jsonl"), "a") as fh:
            fh.write(json.dumps([a for a in args if "=" in a and not a.startswith("MFL_")]) + "\n")
        n = len(open(os.path.join(state, "rollout_env.jsonl")).read().splitlines())
        crashes = scen.get("crashes", [])
        if n <= len(crashes):
            print("Traceback (most recent call last)\n" + crashes[n - 1], flush=True)
            sys.exit(1)
        cfg = open(os.path.join(state, "config.toml")).read()
        base = cfg.split('base_url = "')[1].split('"')[0]
        for _ in range(scen.get("llm_calls", 1)):
            req = urllib.request.Request(base + "/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": "p"}]}).encode(), headers={"Content-Type": "application/json"})
            try: urllib.request.urlopen(req, timeout=20).read()
            except Exception as e: print("llm error", e)
        print("agent ran", flush=True)
        if scen.get("writes_result"): mark("result")
        if scen.get("hangs"):
            import time
            time.sleep(scen["hangs"])
    elif step == "done": sys.exit(0 if has("result") else 1)
    elif step == "result":
        if "result_json" not in scen: sys.exit(1)
        sys.stdout.write(json.dumps(scen["result_json"]))
    elif step == "collect": sys.stdout.write(scen.get("patch", ""))
    elif step == "agentstate": sys.stdout.write(scen.get("state", {}).get(args[-1], ""))
    elif step == "patchcheck": sys.exit(0 if scen.get("applies", True) else 1)
    sys.exit(0)
sys.exit(2)
"""
SENTINELS = {
    "instance_id": "octo__widget-4242",
    "pull_number": "4242",
    "hints_text": "HINT-SENTINEL use the frobnicate helper",
    "all_hints_text": "ALLHINT-SENTINEL",
    "test_patch": "diff --git TESTPATCH-SENTINEL",
    "patch": "diff --git GOLD-SENTINEL",
    "FAIL_TO_PASS": ["tests/test_w.py::test_F2P_SENTINEL"],
    "PASS_TO_PASS": ["tests/test_w.py::test_P2P_SENTINEL"],
    "base_commit": HEAD,
    "docker_image": "starryzhang/sweb.eval.x86_64.octo_1776_widget-4242",
    "test_cmds": ["pytest -rA TESTCMD_SENTINEL"],
    "created_at": "2099-01-02T03:04:05",
    "problem_statement": "Widget crashes on empty input.\nSteps: call widget('').",
}


class Env:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, **scenario: Any) -> None:
        self.tmp = tmp
        self.state = tmp / "state"
        self.state.mkdir()
        (self.state / "scenario.json").write_text(json.dumps({"patch": PATCH, **scenario}))
        self.docker = tmp / "docker"
        self.docker.write_text(
            FAKE_DOCKER.replace("#!/usr/bin/env python3", "#!" + sys.executable)
        )
        self.docker.chmod(self.docker.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("FAKE_DOCKER_STATE", str(self.state))
        self.bundle = tmp / "bundle"
        (self.bundle / "venv" / "bin").mkdir(parents=True)
        (self.bundle / "venv" / "bin" / "memfun").write_text("")
        (self.bundle / "VERSION").write_text("memfun-test-1")
        self.data = tmp / "rows.jsonl"
        self.data.write_text(json.dumps(SENTINELS) + "\n")
        self.run_dir = tmp / "run"
        self.up = Upstream()

    def argv(self, *extra: str) -> list[str]:
        return [
            "run", "--dataset", str(self.data), "--run-dir", str(self.run_dir),
            "--upstream", self.up.url, "--model", "m", "--bundle", str(self.bundle),
            "--docker", str(self.docker), "--instance-timeout", "60", *extra,
        ]  # fmt: skip

    def calls(self) -> list[list[str]]:
        return [json.loads(x) for x in (self.state / "calls.jsonl").read_text().splitlines()]

    def record(self) -> dict[str, Any]:
        return json.loads(
            (self.run_dir / "instances" / SENTINELS["instance_id"] / "instance.json").read_text()
        )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    made: list[Env] = []

    def factory(**scenario: Any) -> Env:
        e = Env(tmp_path, monkeypatch, **scenario)
        made.append(e)
        return e

    yield factory
    for e in made:
        e.up.stop()


def test_a_rollout_end_to_end_records_patch_trajectory_and_usage(env) -> None:
    e = env()
    assert rn.main(e.argv("--price-in", "1", "--price-out", "2")) == 0
    rec = e.record()
    assert rec["status"] == "done" and rec["head"] == HEAD and rec["flags"] == []
    assert rec["usage"]["calls"] == 1 and rec["usage"]["prompt_tokens"] == 100
    assert rec["cost_usd"] == pytest.approx(140 / 1e6) and rec["patch_applies"] is True
    assert rec["patch_files"] == 1 and rec["patch_new_files"] == 0
    collect = next(c for c in e.calls() if "MFL_STEP=collect" in c)[-1]
    assert "':(exclude).memfun'" in collect and "':(exclude,glob)**/__pycache__/**'" in collect
    idir = e.run_dir / "instances" / SENTINELS["instance_id"]
    assert (idir / "patch.diff").read_text() == PATCH
    assert len((idir / "llm.jsonl").read_text().splitlines()) == 1  # the raw trajectory
    preds = json.loads((e.run_dir / "preds.json").read_text())
    assert preds == {SENTINELS["instance_id"]: {"model_name_or_path": "m", "model_patch": PATCH}}
    steps = [a.split("=", 1)[1] for c in e.calls() for a in c if a.startswith("MFL_STEP=")]
    # the order is the contract: history lock-down and the network self-test BEFORE the rollout
    assert steps == [
        "findrepo", "lockdown", "baseline", "nettest", "config", "rollout", "result",
        "agentstate", "agentstate", "agentstate", "agentstate", "collect", "patchcheck",
    ]  # fmt: skip
    # the model library must not refresh its price table from the internet at start-up
    rollout = next(c for c in e.calls() if "MFL_STEP=rollout" in c)
    assert "LITELLM_LOCAL_MODEL_COST_MAP=true" in rollout
    kinds = [c[0] for c in e.calls()]
    assert kinds[-3:] == ["rm", "rmi", "network"]  # pull-run-delete, nothing left behind


def test_only_the_problem_statement_reaches_the_container(env) -> None:
    """Rule 1: no other dataset field, and not the instance id, in anything docker was given
    (image name aside: the image is the task) or in the agent's config."""
    e = env()
    assert rn.main(e.argv()) == 0
    blob = (
        "\n".join(json.dumps([a for a in c if a != SENTINELS["docker_image"]]) for c in e.calls())
        + (e.state / "config.toml").read_text()
    )
    for needle in ("SENTINEL", "4242", "octo__widget", "2099-01-02", HEAD[:12] + "zz"):
        assert needle not in blob, needle
    rollout = next(c for c in e.calls() if "MFL_STEP=rollout" in c)
    prompt = rollout[rollout.index("swarm") + 1]
    assert prompt == rn.PROMPT_TEMPLATE.format(problem_statement=SENTINELS["problem_statement"])
    assert ds.episode_input(SENTINELS) == ds.EpisodeInput(SENTINELS["problem_statement"])
    assert {f.name for f in ds.EpisodeInput.__dataclass_fields__.values()} == {"problem_statement"}
    assert set(SENTINELS) - {"problem_statement"} <= set(ds.FORBIDDEN_FIELDS)


def test_agent_options_reach_the_agent_and_are_frozen_with_the_run(env) -> None:
    """The switches of the agent's own verification: written into the repository's
    ``.memfun`` (which is never part of a patch), trusted through the environment, recorded in
    the frozen settings; without options neither the file nor the trust variable exists."""
    e = env()
    opts = ["--agent-option", "discover_tests=true", "--agent-option", "probe_guard=off"]
    assert rn.main(e.argv(*opts)) == 0
    written = next(c for c in e.calls() if "MFL_STEP=agentopts" in c)
    assert written[-2] == "/testbed"
    assert json.loads(written[-1]) == {"discover_tests": True, "probe_guard": "off"}
    assert ".memfun" in rn.PATCH_EXCLUDES and ".memfun/verify.json" in rn.WRITE_OPTIONS
    rollout = next(c for c in e.calls() if "MFL_STEP=rollout" in c)
    assert f"{rn.TRUST_VERIFY_ENV}=1" in rollout
    meta = json.loads((e.run_dir / "run.json").read_text())
    assert meta["settings"]["agent_options"] == {"discover_tests": True, "probe_guard": "off"}
    with pytest.raises(SystemExit, match="agent_options"):  # other options: another run
        rn.main(e.argv("--agent-option", "discover_tests=false"))


def test_a_run_created_before_agent_options_existed_can_be_resumed(env) -> None:
    e = env()
    assert rn.main(e.argv()) == 0
    assert not any("MFL_STEP=agentopts" in c for c in e.calls())  # no options: no file,
    rollout = next(c for c in e.calls() if "MFL_STEP=rollout" in c)
    assert not any(a.startswith(rn.TRUST_VERIFY_ENV) for a in rollout)  # and no trust
    meta = json.loads((e.run_dir / "run.json").read_text())
    del meta["settings"]["agent_options"]
    (e.run_dir / "run.json").write_text(json.dumps(meta))
    assert rn.main(e.argv()) == 0  # same settings: accepted, nothing re-run


def test_the_agents_own_check_log_is_kept_with_the_trajectory(env) -> None:
    e = env(state={".memfun/checks.jsonl": '{"event": "verify"}\n'})
    assert rn.main(e.argv()) == 0
    idir = e.run_dir / "instances" / SENTINELS["instance_id"]
    assert (idir / "checks.jsonl").read_text() == '{"event": "verify"}\n'
    assert e.record()["agent_state"] == ["checks.jsonl"] and not (idir / "probes.jsonl").exists()


def test_the_container_is_isolated_and_limited(env) -> None:
    e = env()
    assert rn.main(e.argv("--cpus", "3", "--memory", "5g")) == 0
    run = next(c for c in e.calls() if c[0] == "run")
    opt = dict(itertools.pairwise(run))
    assert opt["--network"] == "mfl-net" and opt["--cpus"] == "3.0" and opt["--memory"] == "5g"
    assert opt["--hostname"] == "task" and re.fullmatch(r"mfl-[0-9a-f]{12}", opt["--name"])
    assert f"{e.bundle}:{e.bundle}:ro" in run  # the agent bundle is read-only
    create = next(c for c in e.calls() if c[:2] == ["network", "create"])
    assert "--internal" in create


@pytest.mark.parametrize(
    ("scenario", "needle"),
    [
        (
            {"net": {"proxy": True, "internet": ["1.1.1.1"], "dns": [], "host_ports": []}},
            "internet",
        ),
        ({"net": {"proxy": True, "internet": [], "dns": ["github.com"], "host_ports": []}}, "DNS"),
        ({"net": {"proxy": True, "internet": [], "dns": [], "host_ports": [6379]}}, "6379"),
        ({"net": {"proxy": False, "internet": [], "dns": [], "host_ports": []}}, "proxy"),
        ({"lockdown_fails": True}, "lockdown"),
        ({"pull_fails": True}, "pull"),
    ],
)
def test_a_failed_self_test_stops_before_the_rollout(env, monkeypatch, scenario, needle) -> None:
    monkeypatch.setattr(rn.threading.Event, "wait", lambda self, timeout=None: False)
    e = env(**scenario)
    assert rn.main(e.argv()) == 1
    rec = e.record()
    assert rec["status"] == "infra_error" and needle in rec["error"]
    assert not any("MFL_STEP=rollout" in c for c in e.calls())  # the agent never started
    assert json.loads((e.run_dir / "preds.json").read_text()) == {}


def test_no_model_answer_is_an_infrastructure_failure_not_an_empty_patch(env) -> None:
    e = env(llm_calls=0, patch="")
    assert rn.main(e.argv()) == 1
    assert e.record()["status"] == "infra_error"


def test_a_rollout_the_model_path_did_not_serve_is_an_infrastructure_failure(env) -> None:
    e = env(llm_calls=2)
    e.up.statuses = [200, 402]  # the account ran out of credit in the middle of the rollout
    assert rn.main(e.argv()) == 1
    rec = e.record()
    assert rec["status"] == "infra_error" and "did not serve 1 of 2 calls" in rec["error"]
    assert rec["usage"]["unserved"] == 1 and "patch_bytes" not in rec  # no patch is taken from it
    assert json.loads((e.run_dir / "preds.json").read_text()) == {}


def test_an_outage_pauses_the_run_and_restarts_the_instance_from_scratch(env) -> None:
    e = env()
    # probe ok, the rollout's call is refused (no credit), probe refused, probe ok, call ok
    e.up.statuses = [200, 402, 402, 200, 200]
    assert rn.main(e.argv("--outage-wait", "30", "--outage-poll", "0.01")) == 0
    assert e.record()["status"] == "done" and e.record()["patch_bytes"] == len(PATCH.encode())
    (log,) = [json.loads(x) for x in (e.run_dir / "attempts.jsonl").read_text().splitlines()]
    assert log["status"] == "infra_error" and "no answer from the model" in log["error"]
    assert (e.run_dir / log["kept_at"] / "llm.jsonl").exists()  # the failed attempt is kept
    assert len(e.up.seen) == 5 and e.up.statuses == []
    assert sum(c[0] == "run" for c in e.calls()) == 2  # a fresh container for the restart


def test_a_model_path_that_does_not_come_back_starts_nothing(env) -> None:
    e = env()
    e.up.statuses = [402] * 200
    assert rn.main(e.argv("--outage-wait", "0.2", "--outage-poll", "0.01")) == 1
    assert not (e.run_dir / "instances").exists()  # no attempt was used up
    assert not (e.state / "calls.jsonl").exists() or not any(c[0] == "run" for c in e.calls())


def test_the_self_test_covers_a_model_server_on_this_host(env) -> None:
    e = env()
    assert rn.main(e.argv()) == 0
    (nettest,) = [c for c in e.calls() if "MFL_STEP=nettest" in c]
    assert nettest[-1] == e.up.url.split(":")[2].split("/")[0]  # the upstream's port is probed


def test_flags_empty_and_non_applying_patches_and_a_moved_head(env) -> None:
    e = env(patch=PATCH, applies=False, head="b" * 40, dirty=[" M setup.py", "?? gen.py"])
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert set(rec["flags"]) == {
        "patch_does_not_apply", "head_is_not_base_commit", "image_tree_dirty"
    }  # fmt: skip
    assert rec["initial_dirty"] == [" M setup.py", "?? gen.py"] and rec["base_tree"] == "c" * 40
    collect = next(c for c in e.calls() if "MFL_STEP=collect" in c)[-1]
    assert "diff --text --no-color " + "c" * 40 in collect  # against the initial tree, not HEAD


def test_the_patch_is_the_change_from_the_images_initial_tree(tmp_path: Path) -> None:
    """Real git: an image whose tree is already dirty; the agent's patch must hold only the
    agent's change, apply on top of that dirty tree, and leave the index as it was found."""

    def sh(script: str) -> str:
        res = subprocess.run(
            ["sh", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
        )
        assert res.returncode == 0, res.stderr
        return res.stdout

    home = tmp_path.parent / "home"
    snap = lambda n: rn._snapshot(n).replace(rn.HOME_IN, str(home))  # noqa: E731
    spec = rn._pathspec()
    sh("git init -q && git config user.email a@b && git config user.name t && "
       "printf 'a=1\nb=1\n' > m.py && echo keep > k.py && git add . && git commit -qm c")  # fmt: skip
    sh("printf 'a=1\nb=IMAGE\n' > m.py && echo pre > untracked.py && git add k.py")  # dirty image
    index_before = sh("git status --porcelain")
    base = sh(snap("base")).strip()
    assert sh("git status --porcelain") == index_before  # the real index is untouched
    sh("printf 'a=AGENT\nb=IMAGE\n' > m.py && echo new > fix.py && mkdir -p __pycache__ && "
       "touch __pycache__/m.pyc")  # fmt: skip
    patch = sh(
        f"t=$({snap('final')}) && git --no-pager diff --text --no-color {base} $t -- . {spec}"
    )
    assert "+a=AGENT" in patch and "fix.py" in patch
    assert "IMAGE" not in patch.replace(" b=IMAGE", "") and "untracked.py" not in patch
    assert "__pycache__" not in patch
    (home / "patch.diff").write_text(patch)
    chk = tmp_path.parent / "chk"
    chk.mkdir()
    sh(f"git archive {base} | tar -x -C {chk} && cd {chk} && git apply --check {home}/patch.diff")


def test_an_agent_that_wrote_its_result_but_does_not_exit_is_reaped(env, monkeypatch) -> None:
    orig = rn.Box.rollout
    monkeypatch.setattr(
        rn.Box,
        "rollout",
        lambda self, argv, **kw: orig(self, argv, **{**kw, "grace": 1.0, "poll": 0.3}),
    )
    e = env(writes_result=True, hangs=30)
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert rec["status"] == "done" and rec["flags"] == ["reaped_after_result"]
    assert rec["rollout_s"] < 20 and rec["work_s"] <= rec["rollout_s"] and not rec["timed_out"]
    assert rec["patch_bytes"] == len(PATCH)  # the patch is still collected after the reap
    assert any(c[0] == "restart" for c in e.calls())
    idir = e.run_dir / "instances" / SENTINELS["instance_id"]
    assert b"agent ran" in (idir / "agent.log").read_bytes()  # streamed while running


def test_a_timed_out_agent_still_gets_its_patch_collected(env, monkeypatch) -> None:
    orig = rn.Box.rollout
    monkeypatch.setattr(
        rn.Box,
        "rollout",
        lambda self, argv, **kw: orig(self, argv, **{**kw, "timeout": 1.5, "poll": 0.3}),
    )
    e = env(hangs=30)
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert rec["flags"] == ["agent_timed_out"] and rec["timed_out"] is True
    assert rec["status"] == "done" and rec["patch_bytes"] == len(PATCH)


def test_single_rollout_resume_and_documented_infra_retry(env) -> None:
    e = env(net={"proxy": False, "internet": [], "dns": [], "host_ports": []})
    assert rn.main(e.argv()) == 1  # infra failure
    n = len(e.calls())
    assert rn.main(e.argv()) == 1  # not re-run without the explicit flag
    assert not any(c[0] == "run" for c in e.calls()[n:])
    (e.state / "scenario.json").write_text(json.dumps({"patch": PATCH}))
    assert rn.main(e.argv("--retry-infra")) == 0
    (log,) = [json.loads(x) for x in (e.run_dir / "attempts.jsonl").read_text().splitlines()]
    assert log["instance_id"] == SENTINELS["instance_id"] and "proxy" in log["error"]
    assert (e.run_dir / log["kept_at"] / "instance.json").exists()  # the first attempt is kept
    n = len(e.calls())
    assert rn.main(e.argv("--retry-infra")) == 0  # finished: never rolled out again
    assert not any(c[0] == "run" for c in e.calls()[n:])


def test_settings_are_frozen_and_a_finalized_run_cannot_roll_out(env) -> None:
    e = env()
    assert rn.main(e.argv()) == 0
    with pytest.raises(SystemExit, match="temperature"):
        rn.main(e.argv("--temperature", "0.1"))
    assert rn.main(["finalize", "--run-dir", str(e.run_dir)]) == 0
    assert json.loads((e.run_dir / "run.json").read_text())["finalized"] is True
    with pytest.raises(SystemExit, match="finalized"):
        rn.main(e.argv())


def test_finalize_refuses_an_unfinished_run(env) -> None:
    e = env(pull_fails=True)
    import threading

    orig = threading.Event.wait
    threading.Event.wait = lambda self, timeout=None: False  # type: ignore[method-assign]
    try:
        assert rn.main(e.argv()) == 1
    finally:
        threading.Event.wait = orig  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="not finished"):
        rn.main(["finalize", "--run-dir", str(e.run_dir)])


def test_prompts_are_general(tmp_path: Path) -> None:
    """Rule 8: no instance- or repository-specific text in the runner's code or prompt."""
    src = "".join(
        p.read_text("utf-8") for p in (REPO_ROOT / "evals" / "swebench_live").glob("*.py")
    )
    assert not re.search(r"[a-z0-9-]+__[a-z0-9_.-]+-\d+", src.replace("octo__widget", ""))
    for word in ("django", "sympy", "astropy", "matplotlib", "scikit", "pandas", "reflex"):
        assert word not in rn.PROMPT_TEMPLATE.lower()
    assert rn.PROMPT_TEMPLATE.count("{") == 1  # only the problem statement is filled in


def test_firewall_rules_touch_only_our_bridge() -> None:
    rules = rn.firewall_rules("br-0123456789ab", "172.18.0.1", 18200)
    assert all(r[:3] == ["INPUT", "-i", "br-0123456789ab"] for r in rules)
    assert rules[0][-1] == "DROP" and rules[1][-1] == "ACCEPT" and "18200" in rules[1]


def test_lockdown_removes_everything_newer_than_head(tmp_path: Path) -> None:
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=t", *a],
            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60,
        ).stdout.strip()  # fmt: skip

    git("init", "-q", "-b", "main")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-qm", "c1")
    git("tag", "v1.0")
    base = git("rev-parse", "HEAD")
    (tmp_path / "f").write_text("FIX-SECRET")
    git("commit", "-qam", "the fix")
    git("tag", "v2.0")
    git("branch", "future")
    git("checkout", "-q", "--detach", base)
    git("remote", "add", "origin", "https://example.invalid/x.git")
    git("update-ref", "refs/remotes/origin/main", "future")
    (tmp_path / "f").write_text("wip")
    git("stash", "-q")
    (tmp_path / "f").write_text("dirty")
    script = str(rn.LOCKDOWN)
    before = subprocess.run(
        ["sh", script, "--check", str(tmp_path)], capture_output=True, timeout=60
    )
    assert before.returncode == 1 and b"refs beyond HEAD" in before.stdout
    done = subprocess.run(
        ["sh", script, str(tmp_path)], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0 and "lockdown ok" in done.stdout, done.stdout + done.stderr
    assert git("tag") == "v1.0" and git("remote") == "" and git("rev-parse", "HEAD") == base
    assert (tmp_path / "f").read_text() == "dirty"  # the working tree is untouched
    objs = git("cat-file", "--batch-all-objects", "--batch-check").splitlines()
    for line in objs:
        body = subprocess.run(
            ["git", "cat-file", "-p", line.split()[0]],
            cwd=tmp_path,
            capture_output=True,
            timeout=30,
        ).stdout
        assert b"FIX-SECRET" not in body and b"the fix" not in body


def test_dataset_selection_and_dev_sample(tmp_path: Path) -> None:
    rows = [{"instance_id": f"o__r-{i}", "problem_statement": "p"} for i in range(20)]
    assert [r["instance_id"] for r in ds.select(rows, ids=["o__r-3", "o__r-1"])] == [
        "o__r-3",
        "o__r-1",
    ]
    with pytest.raises(ValueError, match="not in the dataset"):
        ds.select(rows, ids=["nope"])
    assert ds.select(rows, sample=5, seed=1) == ds.select(list(reversed(rows)), sample=5, seed=1)
    dev = ds.dev_sample(rows, [rows[:5], rows[5:10]], 4, seed=18)
    assert len(dev) == 4 and all(int(i.rsplit("-", 1)[1]) >= 10 for i in dev)
    assert (
        ds.task_of({"instance_id": "A__b-1"}).docker_image
        == "starryzhang/sweb.eval.x86_64.a_1776_b-1"
    )
    with pytest.raises(ValueError):
        ds.task_of({"instance_id": "../etc"})


# ── the baseline arm (mini-swe-agent on the host, driving the prepared container) ────────────

SUBMISSION = "diff --git a/s.py b/s.py\n--- a/s.py\n+++ b/s.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
FAKE_MINI_DRIVER = r"""
import json, os, sys, time, urllib.request
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
task = sys.stdin.read()
seen = {"argv": sys.argv[1:], "task": task, "env": dict(os.environ)}
Path(__file__).with_name("driver-seen.json").write_text(json.dumps(seen))
if "SLEEP" in task:
    time.sleep(600)
req = urllib.request.Request(args["--base-url"] + "/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": task}]}).encode(),
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=20).read()
out = Path(args["--out"])
(out / "submission.diff").write_text(SUBMISSION)
(out / "mini.traj.json").write_text("{}")
(out / "result.json").write_text(json.dumps({"exit_status": "Submitted", "steps": 1}))
print("driver done")
"""


def mini_argv(e: Env, monkeypatch: pytest.MonkeyPatch, *extra: str) -> list[str]:
    driver = e.tmp / "fake_mini_driver.py"
    driver.write_text(FAKE_MINI_DRIVER.replace("SUBMISSION)", repr(SUBMISSION) + ")"))
    monkeypatch.setattr(rn, "MINI_DRIVER", driver)
    return e.argv("--arm", "mini", "--mini-python", sys.executable, *extra)


def test_the_baseline_arm_sees_only_the_problem_statement_and_its_submission_is_the_patch(
    env, monkeypatch
) -> None:
    e = env()
    assert rn.main(mini_argv(e, monkeypatch)) == 0
    rec = e.record()
    idir = e.run_dir / "instances" / SENTINELS["instance_id"]
    assert rec["status"] == "done" and rec["usage"]["calls"] == 1 and rec["flags"] == []
    assert (idir / "patch.diff").read_text() == SUBMISSION  # what the agent submitted...
    assert (idir / "worktree.diff").read_text() == PATCH  # ...not whatever it left behind
    assert rec["mini"] == {"exit_status": "Submitted", "steps": 1}
    assert (idir / "mini.traj.json").exists() and (idir / "mini-result.json").exists()
    assert "driver done" in (idir / "agent.log").read_text()
    seen = json.loads((e.tmp / "driver-seen.json").read_text())
    assert seen["task"] == SENTINELS["problem_statement"]  # rule 1: exactly this, on stdin
    blob = json.dumps([seen["argv"], seen["env"]])
    for needle in ("SENTINEL", "4242", "octo__widget"):
        assert needle not in blob, needle
    args = dict(zip(seen["argv"][0::2], seen["argv"][1::2], strict=True))
    assert args["--container"].startswith("mfl-") and args["--cwd"] == "/testbed"
    assert args["--config"] == "swebench.yaml" and "/i/" in args["--base-url"]
    assert not (e.run_dir / "tmp" / args["--container"]).exists()  # scratch removed
    assert set(seen["env"]) <= {"PATH", "HOME", "MSWEA_GLOBAL_CONFIG_DIR", "MSWEA_CONFIGURED",
                                "PYTHONUNBUFFERED", "NO_PROXY", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}  # fmt: skip
    settings = json.loads((e.run_dir / "run.json").read_text())["settings"]
    assert settings["arm"] == "mini" and settings["mini_config"] == "swebench.yaml"
    assert not any("MFL_STEP=rollout" in c for c in e.calls())  # nothing ran IN the container
    assert any("MFL_STEP=putpatch" in c for c in e.calls())  # the submission was checked


def test_a_baseline_run_over_its_time_is_killed_and_has_no_patch(env, monkeypatch) -> None:
    rows = dict(SENTINELS, problem_statement="SLEEP forever")
    e = env()
    e.data.write_text(json.dumps(rows) + "\n")
    argv = mini_argv(e, monkeypatch)
    argv[argv.index("--instance-timeout") + 1] = "1"
    rn.main(argv)
    rec = e.record()
    # no model answer at all is an infrastructure failure, exactly as for the memfun arm
    assert rec["status"] == "infra_error" and "no answer from the model" in rec["error"]
    assert any(c[0] == "restart" for c in e.calls())  # nothing keeps editing


def test_the_baseline_arm_needs_its_interpreter(env) -> None:
    e = env()
    with pytest.raises(SystemExit, match="--mini-python"):
        rn.main(e.argv("--arm", "mini"))


# ── an agent that dies before its first model call ───────────────────────────────────────────

CRASH = "MarkupError: closing tag '[/content/x.py]' at position 1949 doesn't match any open tag"


def attempts(e: Env) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (e.run_dir / "attempts.jsonl").read_text().splitlines()]


def test_the_same_crash_three_times_closes_the_instance_instead_of_retrying_for_ever(env) -> None:
    """Zero model calls is not an attempt: the instance runs again from scratch.  But an
    agent that dies the same way three times in a row is failing on this task: it is closed
    with an empty patch and its own class (one pilot instance was retried 39 times)."""
    e = env(crashes=[CRASH] * 9)
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert rec["status"] == "done" and rec["failure_class"] == "agent_crash"
    assert rec["crash"] == CRASH and rec["identical_crashes"] == 3
    assert {"agent_crash", "empty_patch"} <= set(rec["flags"]) and rec["patch_bytes"] == 0
    assert rec["usage"]["calls"] == 0 and "no answer from the model" in rec["error"]
    log = attempts(e)
    assert [(a["attempt"], a["status"], a["crash"]) for a in log] == [
        (1, "infra_error", CRASH),
        (2, "infra_error", CRASH),
        (3, "agent_crash", CRASH),
    ]
    assert log[2]["final"] is True and "not retried" in log[2]["note"]
    assert len((e.state / "rollout_env.jsonl").read_text().splitlines()) == 3  # and no more
    iid = SENTINELS["instance_id"]
    assert (e.run_dir / "attempts" / f"{iid}.1" / "agent.log").read_text().strip().endswith(CRASH)
    preds = json.loads((e.run_dir / "preds.json").read_text())
    assert preds[iid]["model_patch"] == ""
    assert rn.main(e.argv("--retry-infra")) == 0  # final: never rolled out again
    assert len((e.state / "rollout_env.jsonl").read_text().splitlines()) == 3


def test_a_crash_that_does_not_repeat_is_retried_and_the_rollout_counts(env) -> None:
    e = env(crashes=[CRASH, "OSError: [Errno 28] No space left on device"])
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert rec["status"] == "done" and "failure_class" not in rec and rec["patch_bytes"] > 0
    assert [a["status"] for a in attempts(e)] == ["infra_error", "infra_error"]


def test_crashes_that_keep_differing_stay_an_infrastructure_error(env) -> None:
    e = env(crashes=[f"RuntimeError: temporary failure {i}" for i in range(9)])
    assert rn.main(e.argv()) == 1
    assert e.record()["status"] == "infra_error"
    assert len((e.state / "rollout_env.jsonl").read_text().splitlines()) == rn.CRASH_REPEATS


def test_the_crash_signature_is_the_last_line_of_text_not_the_drawn_frame(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        "\u256d\u2500 Traceback (most recent call last) \u2500\u256e\n"
        "\u2502 /x/markup.py:167 in render             \u2502\n"
        "\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256f\n"
        f"{CRASH}\n\n",
        "utf-8",
    )
    assert rn.crash_signature(log) == CRASH
    assert rn.crash_signature(tmp_path / "missing.log") == ""


def test_an_empty_patch_is_never_silent(env) -> None:
    """The agent's own outcome is kept: a rollout that changed nothing says so."""
    why = "no file changes produced (context-first escalated, RLM did not run)"
    e = env(patch="", result_json={"success": False, "no_edits": True, "error": why})
    assert rn.main(e.argv()) == 0
    rec = e.record()
    assert rec["status"] == "done" and "empty_patch" in rec["flags"]
    assert rec["failure_class"] == "no_edits" and rec["agent_no_edits"] is True
    assert rec["agent_error"] == why and rec["agent_success"] is False


def test_the_served_reads_are_kept_with_the_trajectory(env) -> None:
    line = json.dumps({"round": 1, "asked": "sed -n '1,5p' m.py", "served": True})
    e = env(state={".memfun/reads.jsonl": line + "\n"})
    assert rn.main(e.argv()) == 0
    idir = e.run_dir / "instances" / SENTINELS["instance_id"]
    assert (idir / "reads.jsonl").read_text() == line + "\n"
    assert "reads.jsonl" in e.record()["agent_state"]
