#!/usr/bin/env python3
"""Run a coding agent on SWE-bench-Live instances inside the official task containers.

One rollout per instance, under the leaderboard's rules, each enforced by a mechanism here:

1. **Only the problem statement reaches the agent.**  A dataset row is split by
   ``dataset.episode_input`` (the agent's input) and ``dataset.task_of`` (what the runner needs);
   the container, its name, its environment and the prompt carry no instance id or other field.
2. **The official container.**  The row's ``docker_image``, repository at ``/testbed`` (or its
   one sub-directory holding ``.git``), exactly as the official evaluator uses it.
3. **No future history.**  ``lockdown.sh`` runs in the container before the rollout and its
   self-check must pass.
4. **No internet.**  The container sits on an ``--internal`` Docker network; the only reachable
   thing is the recording LLM proxy on the bridge gateway.  A self-test from inside the
   container must show the internet, DNS and other host services unreachable.
5. **No ground-truth feedback.**  Nothing here runs the evaluation; ``grade.py`` refuses a run
   that is not finalized, and a finalized run cannot be rolled out again.
6. **Single rollout, frozen settings.**  Settings are written to ``run.json`` at the first start
   and must be identical on resume.  A finished instance is never re-run; one that failed for
   an infrastructure reason is re-run only with ``--retry-infra``, from scratch, and the earlier
   attempt is kept and logged in ``attempts.jsonl``.
7. **Raw trajectories.**  ``proxy.RecordingProxy`` writes every LLM request and response.

The agent (memfun) runs INSIDE the task container from a read-only bundle mounted at the same
absolute path it was built at; the commands it starts use the container's own environment.

    runner.py run --dataset full.jsonl --ids-file dev.txt --run-dir runs/dev1 \
        --upstream http://127.0.0.1:18100/v1 --model qwen-3.8-27b --bundle /root/memfun-live/bundle
    runner.py finalize --run-dir runs/dev1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.swebench_live import dataset as ds
from evals.swebench_live import proxy as px

HERE = Path(__file__).resolve().parent
LOCKDOWN = HERE / "lockdown.sh"
MINI_DRIVER = HERE / "mini_driver.py"
#: The whole task text given to the agent besides the issue itself: general, never specific to
#: an instance or a repository (a test checks this file for instance-specific text).
PROMPT_TEMPLATE = (
    "The repository in the current working directory has the problem described in the issue "
    "below. Change the repository's source files so that the issue is resolved. Find the "
    "relevant code by searching the repository, reproduce the problem when you can, and check "
    "your change by running the relevant existing tests. Do not rely on network access.\n\n"
    "<issue>\n{problem_statement}\n</issue>"
)
HOME_IN = "/tmp/mfl-home"
REPO_ROOT_IN = "/testbed"
DUMMY_KEY_ENV, DUMMY_KEY_VALUE = "MEMFUN_SGLANG_DUMMY_KEY", "dummy-not-a-secret"
#: The agent reads ``.memfun/verify.json`` only when this is set (the runner wrote the file).
TRUST_VERIFY_ENV = "MEMFUN_TRUST_PROJECT_VERIFY"
WRITE_OPTIONS = 'mkdir -p "$1/.memfun" && printf %s "$2" > "$1/.memfun/verify.json"'
MAX_ARG_BYTES = 120_000  # Linux refuses a single argv string over 128 KiB
#: Never part of a patch: the agent's own state and scratch.
PATCH_EXCLUDES = (".memfun", "memfun.toml")
#: Junk a test run leaves behind when the repository does not ignore it (git glob pathspecs).
PATCH_JUNK = (
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/.coverage",
    "**/.coverage.*",
    "**/htmlcov/**",
    "**/*.egg-info/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/.hypothesis/**",
    "**/.tox/**",
    "**/.nox/**",
    "build/**",
    "dist/**",
)
#: A patch over this size is flagged (never cut): it probably carries generated files.
PATCH_LARGE_BYTES = 1_000_000
FINAL = "done"
INFRA = "infra_error"
#: An agent that exits with an error before its first model call, the same way this many
#: times in a row, is not an infrastructure failure: the instance is closed with an empty
#: patch and this failure class instead of being retried for ever.
CRASH_REPEATS = 3
AGENT_CRASH = "agent_crash"

FIND_REPO = (
    f"cd {REPO_ROOT_IN} && "
    "{ [ -d .git ] || { g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); "
    '[ -n "$g" ] && cd "${g%/.git}"; }; } && pwd && git rev-parse HEAD'
)
NETTEST = r"""
import json, socket, sys
gw, port = sys.argv[1], int(sys.argv[2])
def tcp(host, p):
    try:
        socket.create_connection((host, p), timeout=3).close()
        return True
    except OSError:
        return False
def dns(name):
    try:
        socket.setdefaulttimeout(3)
        socket.getaddrinfo(name, 443)
        return True
    except OSError:
        return False
out = {"proxy": tcp(gw, port)}
out["internet"] = [h for h, p in (("1.1.1.1", 443), ("8.8.8.8", 53), ("140.82.112.3", 443),
                                   ("151.101.0.223", 443)) if tcp(h, p)]
out["dns"] = [n for n in ("github.com", "pypi.org", "huggingface.co") if dns(n)]
ports = (22, 80, 443, 2375, 5432, 6379, 8000, 8080) + tuple(int(a) for a in sys.argv[3:])
out["host_ports"] = [p for p in ports if p != port and tcp(gw, p)]
print(json.dumps(out))
"""


class InfraError(RuntimeError):
    """The rollout could not run for a reason that is not the agent's doing."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """The frozen configuration of a run (rule 6): written once, compared on resume."""

    arm: str
    model: str
    solver: str
    temperature: float
    llm_max_tokens: int
    context_window: int
    reasoning_effort: str
    force_fields: dict[str, Any]
    instance_timeout: int
    max_tokens_per_instance: int
    cpus: float
    memory: str
    prompt_template: str
    agent_version: str
    runner_commit: str
    #: Switches of the agent's own verification, written to ``.memfun/verify.json`` in the
    #: repository (never part of a patch): e.g. ``discover_tests``, ``reproduce``,
    #: ``probe_guard`` (``"off"`` is right here: the container is disposable).
    agent_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Baseline arm (``arm = "mini"``): the shipped mini-swe-agent config and its step limit
    #: (0 = the config's own).  Empty / 0 for the memfun arm.
    mini_config: str = ""
    mini_step_limit: int = 0


@dataclasses.dataclass
class Context:
    run_dir: Path
    settings: Settings
    proxy: px.RecordingProxy
    docker: str
    network: str
    gateway: str
    bundle: Path
    keep_images: bool
    mini_python: str = ""
    outage_wait: float = 0.0  # seconds to wait for a model path that is not serving; 0 = off
    outage_poll: float = 60.0
    outage_restarts: int = 3  # restarts of one instance after an outage, within one run
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    live: set[str] = dataclasses.field(default_factory=set)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


# ── small helpers ────────────────────────────────────────────────────────────────────────────


def write_json(path: Path, doc: Any) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), "utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def sh(
    cmd: list[str], *, stdin: bytes | None = None, timeout: float = 120.0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, check=False)


def text(b: bytes) -> str:
    return b.decode("utf-8", "replace")


def build_prompt(inp: ds.EpisodeInput) -> str:
    return PROMPT_TEMPLATE.format(problem_statement=inp.problem_statement)


def memfun_config(base_url: str, s: Settings) -> str:
    """The agent's config, written to its throwaway HOME (never into the repository)."""
    extra = ""
    if s.reasoning_effort:
        extra += f"reasoning_effort = {json.dumps(s.reasoning_effort)}\n"
    return (
        "[llm]\n"
        'provider = "openai"\n'
        f"base_url = {json.dumps(base_url)}\n"
        f"model = {json.dumps(s.model)}\n"
        f'api_key_env = "{DUMMY_KEY_ENV}"\n'
        f"max_tokens = {int(s.llm_max_tokens)}\n"
        f"context_window = {int(s.context_window)}\n"
        f"temperature = {s.temperature!r}\n"
        f"{extra}"
        "constrained_json = false\n"
        "\n[backend]\n"
        'tier = "sqlite"\n'
        f'sqlite_path = "{HOME_IN}/memfun.db"\n'
    )


def rollout_argv(ctx: Context, prompt: str) -> list[str]:
    s = ctx.settings
    if s.arm != "memfun":
        raise ValueError(f"unknown arm {s.arm!r}")
    return [
        str(ctx.bundle / "venv" / "bin" / "memfun"),
        "swarm",
        prompt,
        "--no-decompose",
        "--no-commit",
        "--no-push",
        "--no-use-worktrees",
        "--max-pool",
        "1",
        "--timeout",
        str(max(60, s.instance_timeout - 120)),
        *([] if s.solver == "context-first" else ["--single-task-solver", s.solver]),
    ]


# ── one instance ─────────────────────────────────────────────────────────────────────────────


class Box:
    """One task container, addressed by an opaque name (the instance id is never in it)."""

    def __init__(self, ctx: Context, image: str) -> None:
        self.ctx, self.image = ctx, image
        self.name = "mfl-" + secrets.token_hex(6)

    def start(self) -> None:
        c, s = self.ctx, self.ctx.settings
        res = sh(
            [
                c.docker, "run", "-d", "--name", self.name, "--hostname", "task",
                "--platform", "linux/amd64", "--network", c.network,
                "--cpus", str(s.cpus), "--memory", s.memory, "--pids-limit", "4096",
                "-v", f"{c.bundle}:{c.bundle}:ro", "-w", REPO_ROOT_IN,
                "--entrypoint", "sleep", self.image, "infinity",
            ],
            timeout=300,
        )  # fmt: skip
        if res.returncode != 0:
            raise InfraError(f"container did not start: {text(res.stderr)[-400:]}")
        with c.lock:
            c.live.add(self.name)

    def exec(
        self,
        step: str,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout: float = 300.0,
        workdir: str = "",
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        cmd = [self.ctx.docker, "exec", "-e", f"MFL_STEP={step}"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        if stdin is not None:
            cmd.append("-i")
        if workdir:
            cmd += ["-w", workdir]
        return sh([*cmd, self.name, *argv], stdin=stdin, timeout=timeout)

    def must(self, step: str, argv: list[str], **kw: Any) -> str:
        res = self.exec(step, argv, **kw)
        if res.returncode != 0:
            tail = (text(res.stdout) + text(res.stderr))[-600:]
            raise InfraError(f"{step} failed (exit {res.returncode}): {tail}")
        return text(res.stdout)

    def rollout(
        self,
        argv: list[str],
        *,
        workdir: str,
        env: dict[str, str],
        log: Path,
        timeout: float,
        done_file: str,
        grace: float = 20.0,
        poll: float = 5.0,
    ) -> tuple[str, int | None, float]:
        """Run the agent, its output streamed to *log*.  Returns ``(how, exit_code, work_s)``:
        ``exited`` on its own; ``reaped`` when it wrote *done_file* (its result) but was still
        alive *grace* seconds later (a finished agent that does not exit must not burn the
        whole timeout); ``timeout``.  In the last two cases the container is restarted, which
        ends every process in it and keeps its files."""
        cmd = [self.ctx.docker, "exec", "-e", "MFL_STEP=rollout"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["-w", workdir, self.name, *argv]
        t0 = time.time()
        done_at: float | None = None
        with log.open("wb") as fh:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh)
            try:
                while True:
                    try:
                        code = proc.wait(timeout=poll)
                        return "exited", code, (done_at or time.time()) - t0
                    except subprocess.TimeoutExpired:
                        pass
                    now = time.time()
                    if done_at is None:
                        seen = self.exec("done", ["test", "-f", done_file], timeout=60)
                        if seen.returncode == 0:
                            done_at = now
                    how = (
                        "reaped"
                        if done_at is not None and now - done_at >= grace
                        else "timeout"
                        if now - t0 >= timeout or self.ctx.stop.is_set()
                        else ""
                    )
                    if how:
                        self.restart()
                        return how, None, (done_at or now) - t0
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=30)

    def restart(self) -> None:
        """Kill everything running in the container; its filesystem stays."""
        sh([self.ctx.docker, "restart", "-t", "0", self.name], timeout=120)

    def remove(self) -> None:
        sh([self.ctx.docker, "rm", "-f", "-v", self.name], timeout=120)
        with self.ctx.lock:
            self.ctx.live.discard(self.name)


def pull(ctx: Context, image: str, attempts: int = 5) -> float:
    t0 = time.time()
    err = ""
    for i in range(attempts):
        if ctx.stop.is_set():
            raise InfraError("run stopped")
        if sh([ctx.docker, "image", "inspect", image], timeout=60).returncode == 0:
            return time.time() - t0
        res = sh([ctx.docker, "pull", "--platform", "linux/amd64", image], timeout=3600)
        if res.returncode == 0:
            return time.time() - t0
        err = text(res.stderr)[-300:]
        ctx.stop.wait(min(300.0, 20.0 * 2**i))
    raise InfraError(f"image pull failed after {attempts} attempts: {err}")


def network_selftest(box: Box, ctx: Context) -> dict[str, Any]:
    py = str(ctx.bundle / "venv" / "bin" / "python")
    # A model server (or key-holding proxy) on this host must be out of reach as well: only
    # the recording proxy may be between the agent and the model.
    host, up_port = ctx.proxy.upstream_address
    extra = [str(up_port)] if host in ("127.0.0.1", "localhost", "::1", ctx.gateway) else []
    out = box.must(
        "nettest", [py, "-c", NETTEST, ctx.gateway, str(ctx.proxy.port), *extra], timeout=120
    )
    try:
        doc = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise InfraError(f"network self-test gave no result: {out[-200:]}") from exc
    bad = []
    if not doc.get("proxy"):
        bad.append("the LLM proxy is not reachable")
    if doc.get("internet"):
        bad.append(f"internet reachable: {doc['internet']}")
    if doc.get("dns"):
        bad.append(f"DNS resolves: {doc['dns']}")
    if doc.get("host_ports"):
        bad.append(f"host services reachable on ports {doc['host_ports']}")
    if bad:
        raise InfraError("network self-test FAILED: " + "; ".join(bad))
    return doc


def _pathspec() -> str:
    return " ".join(
        [f"':(exclude){p}'" for p in PATCH_EXCLUDES]
        + [f"':(exclude,glob){p}'" for p in PATCH_JUNK]
    )


def _snapshot(name: str) -> str:
    """Shell: write the working tree (tracked + untracked, minus junk) as a git tree object
    through a PRIVATE index file, so the repository's own index stays exactly as found."""
    idx = f"{HOME_IN}/index.{name}"
    return (
        f'mkdir -p {HOME_IN} && cp "$(git rev-parse --absolute-git-dir)/index" {idx} 2>/dev/null; '
        f"GIT_INDEX_FILE={idx} git add -A -- . {_pathspec()} >/dev/null 2>&1; "
        f"GIT_INDEX_FILE={idx} git write-tree"
    )


def baseline(box: Box, repo: str) -> tuple[str, list[str]]:
    """The image's INITIAL working tree as a tree object, and what was dirty in it.

    The official evaluator applies a patch to the image as it is, uncommitted changes and
    all, so the patch must be the difference from this tree, not from HEAD."""
    out = box.must(
        "baseline",
        ["sh", "-c", _snapshot("base") + " && git status --porcelain | head -200"],
        workdir=repo,
        timeout=900,
    ).splitlines()
    if not out or len(out[0].strip()) < 40:
        raise InfraError(f"could not snapshot the initial tree: {out[:2]}")
    return out[0].strip(), [ln for ln in out[1:] if ln.strip()]


def collect_patch(box: Box, repo: str, base_tree: str) -> tuple[str, bool]:
    """The agent's change as one diff against the image's initial tree, checked to apply
    cleanly to a copy of that tree.  ``("", True)`` when nothing changed."""
    script = (
        f"t=$({_snapshot('final')}) && "
        f"git --no-pager diff --text --no-color {base_tree} $t -- . {_pathspec()} "
        f"> {HOME_IN}/patch.diff && cat {HOME_IN}/patch.diff"
    )
    res = box.exec("collect", ["sh", "-c", script], workdir=repo, timeout=600)
    if res.returncode != 0:
        raise InfraError(f"patch collection failed: {text(res.stderr)[-300:]}")
    patch = text(res.stdout)
    if not patch.strip():
        return "", True
    check = (
        "rm -rf /tmp/mfl-check && mkdir /tmp/mfl-check && "
        f"git archive {base_tree} | tar -x -C /tmp/mfl-check && cd /tmp/mfl-check && "
        f"git apply --check --whitespace=nowarn {HOME_IN}/patch.diff"
    )
    ok = box.exec("patchcheck", ["sh", "-c", check], workdir=repo, timeout=900).returncode == 0
    return patch, ok


def host_rollout(
    ctx: Context, box: Box, repo: str, base_url: str, task_text: str, log: Path, out: Path
) -> tuple[str, int | None, float]:
    """The baseline arm: mini-swe-agent runs on the host (its own virtualenv) and drives the
    prepared container with ``docker exec``.  It is given the problem statement on stdin and
    an opaque container name and output directory: no instance id anywhere it can see.
    Returns ``(how, exit_code, work_s)`` like :meth:`Box.rollout`; on a timeout the driver's
    process group is killed and the container restarted, so nothing keeps editing."""
    s = ctx.settings
    argv = [
        ctx.mini_python, str(MINI_DRIVER), "--container", box.name, "--cwd", repo,
        "--base-url", base_url, "--model", s.model, "--out", str(out),
        "--config", s.mini_config, "--step-limit", str(s.mini_step_limit),
        "--wall-limit", str(max(60, s.instance_timeout - 120)), "--docker", ctx.docker,
    ]  # fmt: skip
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(out),
        "MSWEA_GLOBAL_CONFIG_DIR": str(out / "config"),
        "MSWEA_CONFIGURED": "true",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "*",
    }
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("wb") as fh:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=fh, stderr=fh, env=env, start_new_session=True
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(task_text.encode())
            proc.stdin.close()
            while True:
                try:
                    return "exited", proc.wait(timeout=5.0), time.time() - t0
                except subprocess.TimeoutExpired:
                    if time.time() - t0 >= s.instance_timeout or ctx.stop.is_set():
                        break
        finally:
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
    box.restart()
    return "timeout", None, time.time() - t0


def submitted_patch(box: Box, repo: str, base_tree: str, patch: str) -> bool:
    """Whether a patch the agent SUBMITTED applies to a copy of the image's initial tree."""
    if not patch.strip():
        return True
    box.must(
        "putpatch",
        ["sh", "-c", f"mkdir -p {HOME_IN} && cat > {HOME_IN}/patch.diff"],
        stdin=patch.encode(),
    )
    check = (
        "rm -rf /tmp/mfl-check && mkdir /tmp/mfl-check && "
        f"git archive {base_tree} | tar -x -C /tmp/mfl-check && cd /tmp/mfl-check && "
        f"git apply --check --whitespace=nowarn {HOME_IN}/patch.diff"
    )
    return box.exec("patchcheck", ["sh", "-c", check], workdir=repo, timeout=900).returncode == 0


#: The agent's own record of what it checked (which tests it ran and what they said, its
#: reproduction script, its experiments): kept with the trajectory, never part of a patch.
AGENT_STATE = (
    (".memfun/checks.jsonl", "checks.jsonl"),
    (".memfun/probes.jsonl", "probes.jsonl"),
    (".memfun/reads.jsonl", "reads.jsonl"),
    (".memfun/repro/issue_repro.py", "issue_repro.py"),
)


def collect_agent_state(box: Box, repo: str, idir: Path) -> list[str]:
    """Copy the files of :data:`AGENT_STATE` that exist out of the container."""
    got: list[str] = []
    for rel, dest in AGENT_STATE:
        script = 'f="$1/$2"; [ -f "$f" ] && head -c 5000000 "$f"'
        res = box.exec("agentstate", ["sh", "-c", script, "sh", repo, rel], timeout=120)
        if res.returncode == 0 and res.stdout:
            (idir / dest).write_bytes(res.stdout)
            got.append(dest)
    return got


def run_instance(task: ds.Task, inp: ds.EpisodeInput, ctx: Context) -> dict[str, Any]:
    """One rollout.  Returns the instance record (also written to ``instance.json``)."""
    s = ctx.settings
    idir = ctx.run_dir / "instances" / task.instance_id
    idir.mkdir(parents=True, exist_ok=False)
    rec: dict[str, Any] = {
        "instance_id": task.instance_id,
        "image": task.docker_image,
        "status": "started",
        "started_at": time.time(),
        "flags": [],
    }
    write_json(idir / "instance.json", rec)
    token = ""
    box: Box | None = None
    try:
        prompt = build_prompt(inp)
        if len(prompt.encode()) > MAX_ARG_BYTES:
            raise InfraError("the problem statement is too long to pass as one argument")
        rec["pull_s"] = round(pull(ctx, task.docker_image), 1)
        box = Box(ctx, task.docker_image)
        box.start()
        found = box.must("findrepo", ["sh", "-c", FIND_REPO]).split()
        if len(found) != 2:
            raise InfraError(f"no git repository under {REPO_ROOT_IN}: {found}")
        repo, head = found[0], found[1]
        rec.update(repo_dir=repo, head=head)
        if task.base_commit and head != task.base_commit:
            rec["flags"].append("head_is_not_base_commit")
        rec["lockdown"] = (
            box.must(
                "lockdown", ["sh", "-s", "--", repo], stdin=LOCKDOWN.read_bytes(), timeout=1800
            )
            .strip()
            .splitlines()[-1]
        )
        base_tree, dirty = baseline(box, repo)
        rec.update(base_tree=base_tree, initial_dirty=dirty)
        if dirty:
            rec["flags"].append("image_tree_dirty")
        token = ctx.proxy.register(
            task.instance_id,
            idir,
            max_total_tokens=s.max_tokens_per_instance,
            max_seconds=s.instance_timeout + 60,
        )
        rec["network_selftest"] = network_selftest(box, ctx)
        base_url = f"http://{ctx.gateway}:{ctx.proxy.port}/i/{token}/v1"
        box.must(
            "config",
            ["sh", "-c", f"mkdir -p {HOME_IN}/.memfun && cat > {HOME_IN}/.memfun/config.toml"],
            stdin=memfun_config(base_url, s).encode(),
        )
        if s.agent_options:
            box.must(
                "agentopts",
                [
                    "sh",
                    "-c",
                    WRITE_OPTIONS,
                    "sh",
                    repo,
                    json.dumps(s.agent_options, sort_keys=True),
                ],
            )
        env = {
            "HOME": HOME_IN,
            DUMMY_KEY_ENV: DUMMY_KEY_VALUE,
            **({TRUST_VERIFY_ENV: "1"} if s.agent_options else {}),
            "MEMFUN_NO_DASHBOARD_AUTOSTART": "1",
            "MEMFUN_SWARM_RESULT_JSON": f"{HOME_IN}/swarm-result.json",
            # The model library refreshes a price table from the internet at start-up unless
            # told to use its bundled copy: no network attempt may appear in a rollout.
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            "PYTHONUNBUFFERED": "1",
            "COLUMNS": "160",
        }
        t0 = time.time()
        mini_out = ctx.run_dir / "tmp" / box.name  # opaque: the driver never sees the id
        if s.arm == "mini":
            how, exit_code, work_s = host_rollout(
                ctx, box, repo, base_url, inp.problem_statement, idir / "agent.log", mini_out
            )
        else:
            how, exit_code, work_s = box.rollout(
                rollout_argv(ctx, prompt),
                workdir=repo,
                env=env,
                log=idir / "agent.log",
                timeout=s.instance_timeout,
                done_file=f"{HOME_IN}/swarm-result.json",
            )
        rec["agent_exit"] = exit_code
        timed_out = how == "timeout"
        if how != "exited":  # nothing may keep editing while the patch is collected
            rec["flags"].append("agent_timed_out" if timed_out else "reaped_after_result")
        rec["rollout_s"] = round(time.time() - t0, 1)  # wall time including any wait to reap
        rec["work_s"] = round(work_s, 1)  # until the agent wrote its result (or stopped)
        rec["timed_out"] = timed_out
        usage = ctx.proxy.close(token)
        rec["usage"] = dataclasses.asdict(usage)
        rec["cost_usd"] = round(ctx.proxy.cost_usd(token), 6)
        result = box.exec("result", ["cat", f"{HOME_IN}/swarm-result.json"], timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            (idir / "swarm-result.json").write_bytes(result.stdout)
            rec.update(agent_outcome(result.stdout))
        if usage.calls - usage.errors <= 0:
            rec["model_path_failed"] = True
            if usage.calls == 0 and exit_code not in (0, None):
                rec["crash"] = crash_signature(idir / "agent.log")
            raise InfraError(
                f"the agent got no answer from the model ({usage.calls} calls, "
                f"{usage.errors} errors): see agent.log"
            )
        if usage.unserved:
            # An outage in the middle of a rollout: what the agent did afterwards is not its
            # own doing.  A documented infrastructure failure, restarted from scratch.
            rec["model_path_failed"] = True
            raise InfraError(
                f"the model path did not serve {usage.unserved} of {usage.calls} calls "
                "(no answer, credentials or credit refused, or errors that outlasted the "
                "retries): see llm.jsonl"
            )
        rec["agent_state"] = collect_agent_state(box, repo, idir)
        patch, applies = collect_patch(box, repo, base_tree)
        if s.arm == "mini":
            # The standard result of mini-swe-agent is the patch the agent SUBMITS; what else
            # it left in the working tree (scratch files, its patch file) is kept for reference.
            (idir / "worktree.diff").write_text(patch, "utf-8")
            kept = {"mini.traj.json": "mini.traj.json", "result.json": "mini-result.json"}
            for name, dest in kept.items():
                if (mini_out / name).is_file():
                    shutil.copyfile(mini_out / name, idir / dest)
            sub = mini_out / "submission.diff"
            patch = sub.read_text("utf-8") if sub.is_file() else ""
            applies = submitted_patch(box, repo, base_tree, patch)
            rec["mini"] = read_json(mini_out / "result.json") or {}
            shutil.rmtree(mini_out, ignore_errors=True)
        (idir / "patch.diff").write_text(patch, "utf-8")
        rec["patch_bytes"] = len(patch.encode())
        rec["patch_applies"] = applies
        rec["patch_files"] = patch.count("\ndiff --git ") + patch.startswith("diff --git ")
        rec["patch_new_files"] = patch.count("\nnew file mode ")
        if len(patch.encode()) > PATCH_LARGE_BYTES:
            rec["flags"].append("patch_large")
        if not patch.strip():
            # Never silent: an empty patch carries the agent's own failure class.
            rec["flags"].append("empty_patch")
            rec["failure_class"] = (
                "no_edits" if rec.get("agent_no_edits") else rec.get("agent_error") or "no_patch"
            )[:300]
        elif not applies:
            rec["flags"].append("patch_does_not_apply")
        rec["status"] = FINAL
    except InfraError as exc:
        rec.update(status=INFRA, error=str(exc))
    except subprocess.TimeoutExpired as exc:
        rec.update(status=INFRA, error=f"a runner step timed out: {exc.cmd[:4]}")
    except Exception as exc:
        rec.update(status=INFRA, error=f"{type(exc).__name__}: {exc}")
    finally:
        if token:
            with contextlib.suppress(Exception):
                ctx.proxy.close(token)
        if box is not None:
            with contextlib.suppress(Exception):
                box.remove()
        if not ctx.keep_images:
            with contextlib.suppress(Exception):
                sh([ctx.docker, "rmi", task.docker_image], timeout=600)
        rec["finished_at"] = time.time()
        write_json(idir / "instance.json", rec)
    return rec


# ── the run ──────────────────────────────────────────────────────────────────────────────────


_BOX_DRAWING = re.compile(r"^[\s\u2500-\u257f]+|[\s\u2500-\u257f]+$")


def crash_signature(log: Path) -> str:
    """The last line of text in an agent log (frames of a drawn traceback left out): what an
    agent that died before its first model call died of, e.g. ``MarkupError: closing tag...``."""
    try:
        tail = log.read_bytes()[-8_000:].decode("utf-8", "replace")
    except OSError:
        return ""
    for line in reversed(tail.splitlines()):
        line = _BOX_DRAWING.sub("", line)
        if line:
            return line[:300]
    return ""


def agent_outcome(result_json: bytes) -> dict[str, Any]:
    """What the agent itself reported (its ``swarm-result.json``): kept in the instance
    record so that a rollout without a patch says why."""
    try:
        doc = json.loads(result_json)
    except ValueError:
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, Any] = {"agent_success": bool(doc.get("success"))}
    if doc.get("no_edits"):
        out["agent_no_edits"] = True
    if doc.get("error"):
        out["agent_error"] = str(doc["error"])[:500]
    return out


def identical_crashes_before(run_dir: Path, iid: str, signature: str) -> int:
    """How many attempts of *iid* in a row, up to now, ended in this same crash."""
    try:
        lines = (run_dir / "attempts.jsonl").read_text("utf-8").splitlines()
    except OSError:
        return 0
    n = 0
    for line in reversed(lines):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if doc.get("instance_id") != iid:
            continue
        if not signature or doc.get("crash") != signature:
            break
        n += 1
    return n


def close_as_crash(run_dir: Path, rec: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Close an instance whose agent keeps crashing before its first model call: final, with
    an empty patch and the failure class ``agent_crash`` (documented in ``attempts.jsonl``)."""
    iid = str(rec["instance_id"])
    idir = run_dir / "instances" / iid
    rec = {
        **rec,
        "status": FINAL,
        "failure_class": AGENT_CRASH,
        "identical_crashes": repeats,
        "patch_bytes": 0,
        "patch_applies": False,
        "patch_files": 0,
        "flags": [*rec.get("flags", []), AGENT_CRASH, "empty_patch"],
    }
    (idir / "patch.diff").write_text("", "utf-8")
    write_json(idir / "instance.json", rec)
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "instance_id": iid,
                    "attempt": repeats,
                    "status": AGENT_CRASH,
                    "final": True,
                    "error": rec.get("error", ""),
                    "crash": rec.get("crash", ""),
                    "llm_calls": 0,
                    "closed_at": time.time(),
                    "note": f"{repeats} identical crashes before the first model call: "
                    "closed with an empty patch, not retried",
                }
            )
            + "\n"
        )
    return rec


def instance_status(run_dir: Path, iid: str) -> str:
    """``""`` (never started), ``done``, or anything else = an attempt that did not finish."""
    idir = run_dir / "instances" / iid
    if not idir.exists():
        return ""
    doc = read_json(idir / "instance.json")
    return str(doc.get("status") or "unknown") if isinstance(doc, dict) else "unknown"


def set_aside(run_dir: Path, iid: str, status: str) -> None:
    """Keep a failed attempt (trajectory included) and log why the instance runs again."""
    n = 1
    while (dest := run_dir / "attempts" / f"{iid}.{n}").exists():
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = read_json(run_dir / "instances" / iid / "instance.json") or {}
    shutil.move(str(run_dir / "instances" / iid), str(dest))
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "instance_id": iid,
                    "attempt": n,
                    "status": status,
                    "error": old.get("error", ""),
                    "crash": old.get("crash", ""),
                    "llm_calls": (old.get("usage") or {}).get("calls", 0),
                    "set_aside_at": time.time(),
                    "kept_at": str(dest.relative_to(run_dir)),
                }
            )
            + "\n"
        )


def write_preds(run_dir: Path, model_name: str) -> dict[str, Any]:
    """``preds.json`` in the official shape, from every finished instance."""
    preds: dict[str, Any] = {}
    for idir in sorted((run_dir / "instances").glob("*")):
        doc = read_json(idir / "instance.json")
        if not isinstance(doc, dict) or doc.get("status") != FINAL:
            continue
        patch = (idir / "patch.diff").read_text("utf-8") if (idir / "patch.diff").exists() else ""
        preds[idir.name] = {"model_name_or_path": model_name, "model_patch": patch}
    write_json(run_dir / "preds.json", preds)
    return preds


def open_run(run_dir: Path, settings: Settings, ids: list[str]) -> None:
    """Create the run or check a resume against its frozen settings and instance list."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(run_dir / "run.json")
    mine = dataclasses.asdict(settings)
    if meta is None:
        write_json(
            run_dir / "run.json",
            {"settings": mine, "instance_ids": ids, "created_at": time.time(), "finalized": False},
        )
        return
    if meta.get("finalized"):
        raise SystemExit("this run is finalized: no further rollouts (rule: single rollout)")
    saved = dict(meta.get("settings") or {})
    for key, default in (("agent_options", {}), ("mini_config", ""), ("mini_step_limit", 0)):
        saved.setdefault(key, default)  # runs created before the setting existed
    if saved != mine:
        diff = sorted(k for k in mine if saved.get(k) != mine[k])
        raise SystemExit(f"settings differ from the frozen run.json in {diff}: start a new run")
    if meta.get("instance_ids") != ids:
        raise SystemExit("the instance list differs from the frozen run.json: start a new run")


def wait_for_model(ctx: Context) -> bool:
    """Block until the model path serves a call (see :meth:`proxy.RecordingProxy.probe`), at
    most ``ctx.outage_wait`` seconds; ``True`` at once when the wait is switched off.  A model
    server that is down, unreachable or out of credit pauses the run instead of failing every
    remaining instance."""
    if ctx.outage_wait <= 0:
        return True
    deadline = time.time() + ctx.outage_wait
    told = False
    while not ctx.stop.is_set():
        if ctx.proxy.probe(ctx.settings.model):
            if told:
                print("the model path answers again", flush=True)
            return True
        if time.time() >= deadline:
            return False
        if not told:
            print(
                f"the model path is not serving calls: waiting up to {ctx.outage_wait:.0f}s",
                flush=True,
            )
            told = True
        ctx.stop.wait(ctx.outage_poll)
    return False


def run_all(
    rows: list[dict[str, Any]], ctx: Context, workers: int, retry_infra: bool
) -> dict[str, int]:
    todo: list[tuple[ds.Task, ds.EpisodeInput]] = []
    counts = {"done_before": 0, "skipped_unfinished": 0, FINAL: 0, INFRA: 0}
    for row in rows:
        task, inp = ds.task_of(row), ds.episode_input(row)
        status = instance_status(ctx.run_dir, task.instance_id)
        if status == FINAL:
            counts["done_before"] += 1
            continue
        if status:
            if not retry_infra:
                counts["skipped_unfinished"] += 1
                print(
                    f"{task.instance_id}: earlier attempt is {status!r}; --retry-infra re-runs it"
                )
                continue
            set_aside(ctx.run_dir, task.instance_id, status)
        todo.append((task, inp))

    def one(item: tuple[ds.Task, ds.EpisodeInput]) -> None:
        iid = item[0].instance_id
        restarts = 0
        crashes = 0
        while True:
            if ctx.stop.is_set():
                return
            if not wait_for_model(ctx):
                if not ctx.stop.is_set():
                    print(f"{iid}: not started, the model path did not come back", flush=True)
                    with ctx.lock:
                        counts["skipped_unfinished"] += 1
                return
            rec = run_instance(item[0], item[1], ctx)
            crash = str(rec.get("crash") or "")
            if crash:
                # The agent died before its first model call.  From scratch again (no model
                # call was made, so it is not an attempt), but the same death CRASH_REPEATS
                # times in a row is the agent's failure on this task: closed, not retried.
                with ctx.lock:
                    seen = identical_crashes_before(ctx.run_dir, iid, crash) + 1
                    if seen >= CRASH_REPEATS:
                        rec = close_as_crash(ctx.run_dir, rec, seen)
                        break
                    crashes += 1
                    if ctx.stop.is_set() or crashes >= CRASH_REPEATS:
                        break  # crashes that differ: left as an infrastructure error
                    print(
                        f"{iid}: agent crashed before its first call ({crash}); again", flush=True
                    )
                    set_aside(ctx.run_dir, iid, str(rec["status"]))
                continue
            # An outage during the rollout: wait for the model path and restart this instance
            # from scratch within the same run (the attempt is kept and logged, as with
            # --retry-infra).  Only with the wait switched on, and a bounded number of times.
            again = rec.get("model_path_failed") and ctx.outage_wait > 0
            if not again or restarts >= ctx.outage_restarts or ctx.stop.is_set():
                break
            restarts += 1
            print(f"{iid}: {rec.get('error', '')}; restart {restarts} from scratch", flush=True)
            with ctx.lock:
                set_aside(ctx.run_dir, iid, str(rec["status"]))
        with ctx.lock:
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            write_preds(ctx.run_dir, ctx.settings.model)
        u = rec.get("usage") or {}
        print(
            f"{rec['instance_id']}: {rec['status']}  "
            f"{rec.get('rollout_s', 0)}s  calls {u.get('calls', 0)}  "
            f"tokens {u.get('prompt_tokens', 0) + u.get('completion_tokens', 0)}  "
            f"${rec.get('cost_usd', 0):.4f}  patch {rec.get('patch_bytes', 0)}B  "
            f"{' '.join(rec['flags'])} {rec.get('error', '')}",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    return counts


# ── network ──────────────────────────────────────────────────────────────────────────────────


def network_up(docker: str, name: str) -> tuple[str, str]:
    """Create the internal network; returns ``(gateway_ip, bridge_interface)``."""
    if sh([docker, "network", "inspect", name]).returncode != 0:
        res = sh([docker, "network", "create", "--internal", "--driver", "bridge", name])
        if res.returncode != 0:
            raise SystemExit(f"cannot create network {name}: {text(res.stderr)[-300:]}")
    res = sh(
        [docker, "network", "inspect", name, "-f",
         "{{(index .IPAM.Config 0).Gateway}} {{.Id}} {{.Internal}}"]
    )  # fmt: skip
    parts = text(res.stdout).split()
    if res.returncode != 0 or len(parts) != 3 or parts[2] != "true":
        raise SystemExit(f"network {name} is not an internal bridge network: {parts}")
    return parts[0], "br-" + parts[1][:12]


def firewall_rules(iface: str, gateway: str, port: int) -> list[list[str]]:
    """INPUT rules for OUR bridge only: containers may reach the proxy port, nothing else on
    the host.  Inserted at the top in this order, so the ACCEPT ends up above the DROP."""
    return [
        ["INPUT", "-i", iface, "-j", "DROP"],
        ["INPUT", "-i", iface, "-p", "tcp", "-d", gateway, "--dport", str(port), "-j", "ACCEPT"],
    ]


def firewall(action: str, rules: list[list[str]], record: Path | None) -> None:
    """``action`` is ``-I`` (insert, and write the exact rules to *record*) or ``-D``."""
    for rule in rules if action == "-I" else reversed(rules):
        chain, rest = rule[0], rule[1:]
        cmd = ["iptables", action, chain, *(["1"] if action == "-I" else []), *rest]
        res = sh(cmd)
        if res.returncode != 0 and action == "-I":
            raise SystemExit(f"iptables failed: {' '.join(cmd)}: {text(res.stderr)[-200:]}")
    if record is not None:
        if action == "-I":
            lines = ["iptables -D " + " ".join(r) for r in rules]
            record.write_text("# rules added by the swebench-live runner; remove with:\n"
                              + "\n".join(lines) + "\n", "utf-8")  # fmt: skip
        else:
            with contextlib.suppress(OSError):
                record.unlink()


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────


def git_commit() -> str:
    res = sh(["git", "-C", str(HERE), "rev-parse", "HEAD"])
    return text(res.stdout).strip() if res.returncode == 0 else "unknown"


def bundle_version(bundle: Path) -> str:
    f = bundle / "VERSION"
    return f.read_text("utf-8").strip() if f.is_file() else "unknown"


def parse_force(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--force-field wants KEY=JSON, got {item!r}")
        try:
            out[key] = json.loads(val)
        except ValueError:
            out[key] = val
    return out


def cmd_run(a: argparse.Namespace) -> int:
    rows = ds.load_rows(a.dataset)
    ids = None
    if a.ids_file:
        ids = [x.strip() for x in a.ids_file.read_text("utf-8").splitlines() if x.strip()]
    rows = ds.select(rows, ids=ids, sample=a.sample, seed=a.seed)
    if not rows:
        raise SystemExit("no instances selected")
    bundle = a.bundle.resolve()
    if not (bundle / "venv" / "bin" / "memfun").exists():
        raise SystemExit(f"no agent bundle at {bundle} (run provision.sh)")
    force = parse_force(a.force_field)
    settings = Settings(
        arm=a.arm,
        model=a.model,
        solver=a.solver,
        temperature=a.temperature,
        llm_max_tokens=a.llm_max_tokens,
        context_window=a.context_window,
        reasoning_effort=a.reasoning_effort,
        force_fields=force,
        instance_timeout=a.instance_timeout,
        max_tokens_per_instance=a.max_tokens_per_instance,
        cpus=a.cpus,
        memory=a.memory,
        prompt_template=PROMPT_TEMPLATE,
        agent_version=bundle_version(bundle),
        runner_commit=git_commit(),
        agent_options=parse_force(a.agent_option),
        mini_config=a.mini_config if a.arm == "mini" else "",
        mini_step_limit=a.mini_step_limit if a.arm == "mini" else 0,
    )
    if a.arm == "mini" and not (a.mini_python and Path(a.mini_python).is_file()):
        raise SystemExit("--arm mini needs --mini-python (a virtualenv with mini-swe-agent)")
    run_dir = a.run_dir.resolve()
    open_run(run_dir, settings, [str(r["instance_id"]) for r in rows])
    key = a.key_file.read_text("utf-8").strip() if a.key_file else ""
    gateway, iface = network_up(a.docker, a.network)
    proxy = px.RecordingProxy(
        a.upstream,
        host=gateway,
        port=a.proxy_port,
        api_key=key,
        force_fields=force,
        price_in=a.price_in,
        price_out=a.price_out,
    )
    proxy.start()
    rules = firewall_rules(iface, gateway, proxy.port) if a.host_firewall else []
    ctx = Context(
        run_dir=run_dir,
        settings=settings,
        proxy=proxy,
        docker=a.docker,
        network=a.network,
        gateway=gateway,
        bundle=bundle,
        keep_images=a.keep_images,
        mini_python=a.mini_python,
        outage_wait=max(0.0, a.outage_wait),
        outage_restarts=max(0, a.outage_restarts),
        outage_poll=max(0.01, a.outage_poll),
    )

    def on_signal(_sig: int, _frm: Any) -> None:
        ctx.stop.set()
        for name in list(ctx.live):
            sh([a.docker, "rm", "-f", "-v", name], timeout=60)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        if rules:
            firewall("-I", rules, a.iptables_record)
        counts = run_all(rows, ctx, a.workers, a.retry_infra)
    finally:
        for name in list(ctx.live):
            sh([a.docker, "rm", "-f", "-v", name], timeout=60)
        if rules:
            firewall("-D", rules, a.iptables_record)
        proxy.stop()
        if not a.keep_network:
            sh([a.docker, "network", "rm", a.network])
    write_preds(run_dir, settings.model)
    print(json.dumps(counts))
    return 0 if not counts.get(INFRA) and not counts.get("skipped_unfinished") else 1


def cmd_finalize(a: argparse.Namespace) -> int:
    """Close the run: every instance must be finished; after this no rollout is possible and
    grading is allowed."""
    run_dir = a.run_dir.resolve()
    meta = read_json(run_dir / "run.json")
    if not isinstance(meta, dict):
        raise SystemExit(f"no run.json in {run_dir}")
    open_ids = [i for i in meta["instance_ids"] if instance_status(run_dir, i) != FINAL]
    if open_ids:
        raise SystemExit(f"{len(open_ids)} instances are not finished: {open_ids[:5]}")
    preds = write_preds(run_dir, meta["settings"]["model"])
    meta.update(finalized=True, finalized_at=time.time(), n_instances=len(preds))
    write_json(run_dir / "run.json", meta)
    print(f"finalized: {len(preds)} predictions in {run_dir / 'preds.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", type=Path, required=True, help="JSONL from dataset.py fetch")
    r.add_argument("--ids-file", type=Path, default=None)
    r.add_argument("--sample", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--upstream", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--key-file", type=Path, default=None, help="upstream API key (never printed)")
    r.add_argument("--model", required=True)
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--arm", default="memfun", choices=("memfun", "mini"))
    r.add_argument("--mini-python", default="", help="python of a venv with mini-swe-agent")
    r.add_argument("--mini-config", default="swebench.yaml", help="a shipped benchmark config")
    r.add_argument("--mini-step-limit", type=int, default=0, help="0 = the config's own limit")
    r.add_argument("--solver", default="context-first", choices=("context-first", "rlm"))
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--llm-max-tokens", type=int, default=32768)
    r.add_argument("--context-window", type=int, default=131072)
    r.add_argument("--reasoning-effort", default="none")
    r.add_argument("--force-field", action="append", default=[], metavar="KEY=JSON")
    r.add_argument(
        "--agent-option", action="append", default=[], metavar="KEY=JSON",
        help="a switch of the agent's own verification (.memfun/verify.json), e.g. "
        "discover_tests=true reproduce=true probe_guard=off; part of the frozen settings",
    )  # fmt: skip
    r.add_argument("--instance-timeout", type=int, default=2400)
    r.add_argument("--max-tokens-per-instance", type=int, default=4_000_000)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--cpus", type=float, default=4.0)
    r.add_argument("--memory", default="16g")
    r.add_argument("--docker", default="docker")
    r.add_argument("--network", default="mfl-net")
    r.add_argument("--proxy-port", type=int, default=0)
    r.add_argument("--price-in", type=float, default=0.0, help="$ per million input tokens")
    r.add_argument("--price-out", type=float, default=0.0)
    r.add_argument("--host-firewall", action="store_true",
                   help="iptables: containers reach only the proxy port on the host")  # fmt: skip
    r.add_argument("--iptables-record", type=Path, default=None)
    r.add_argument("--keep-images", action="store_true")
    r.add_argument("--keep-network", action="store_true")
    r.add_argument("--retry-infra", action="store_true")
    r.add_argument(
        "--outage-wait", type=float, default=0.0, metavar="SECONDS",
        help="before each instance, and after a rollout the model path did not serve, wait up "
        "to this long for the model to answer a one-token call, then restart that instance "
        "from scratch (0 = off: such a rollout is an infrastructure error for --retry-infra)",
    )  # fmt: skip
    r.add_argument("--outage-restarts", type=int, default=3)
    r.add_argument("--outage-poll", type=float, default=60.0, help="seconds between checks")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("finalize")
    f.add_argument("--run-dir", type=Path, required=True)
    f.set_defaults(fn=cmd_finalize)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
