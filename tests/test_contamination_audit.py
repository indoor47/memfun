"""Contamination audit + read-isolation profile: hermetic (no model, no network)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import audit_contamination as ac  # noqa: E402

STUB = "def helper():\n    return 1\n\n\ndef target(x):\n    '''doc'''\n    pass\n"
REF = (
    "def helper():\n    return 1\n\n\ndef target(x):\n    '''doc'''\n"
    "    total_value = compute_something(x)\n    if total_value is None:\n"
    "        return default_result_for(x)\n    return total_value + offset_for(x)\n"
)
CMD = "```mswea_bash_command\n{}\n```"


def _bundle(root: Path) -> Path:
    b = root / "bundles" / "c0s-lib-target"
    (b / "repo" / "lib").mkdir(parents=True)
    (b / "solution" / "lib").mkdir(parents=True)
    (b / "repo" / "lib" / "m.py").write_text(STUB)
    (b / "solution" / "lib" / "m.py").write_text(REF)
    (b / "task.toml").write_text(
        '[source]\nlibrary = "lib"\nfunction = "target"\nfile = "lib/m.py"\n'
    )
    return b


def _episode(run: Path, steps: list[tuple[str, str]], final: str) -> None:
    ep = run / "c0s-lib-target" / "1"
    (ep / "home").mkdir(parents=True)
    (ep / "collected" / "lib").mkdir(parents=True)
    (ep / "collected" / "lib" / "m.py").write_text(final)
    msgs: list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for cmd, out in steps:
        msgs.append({"role": "assistant", "content": CMD.format(cmd)})
        msgs.append(
            {"role": "user", "content": f"<returncode>0</returncode>\n<output>\n{out}\n</output>"}
        )
    (ep / "home" / "mini-trajectory.json").write_text(json.dumps({"messages": msgs}))
    (ep / "episode.json").write_text(
        json.dumps({"episode": "c0s-lib-target-t1", "outcome": "pass", "repo": str(ep / "repo")})
    )
    (run / "results.json").write_text(json.dumps({"meta": {"arm": "mini-swe-agent"}}))


def _audit(tmp: Path, steps: list[tuple[str, str]], final: str = REF) -> dict:
    _bundle(tmp)
    run = tmp / "bench-mini-swe-agent"
    _episode(run, steps, final)
    return ac.audit_run(run, tmp / "bundles")["details"][0]


def test_reading_the_reference_solution_is_a_leak(tmp_path: Path) -> None:
    d = _audit(tmp_path, [("cat /x/bundles/c0s-lib-target/solution/lib/m.py", REF)])
    assert d["leaks"] and "bundle cache" in d["leaks"][0]["source"]
    assert d["reference_path_touches"][0]["class"] == "reference-solution"


def test_echoing_its_own_edit_is_not_a_leak(tmp_path: Path) -> None:
    write = "cat > lib/m.py <<'EOF'\n" + REF + "EOF"
    d = _audit(tmp_path, [(write, ""), ("cat lib/m.py", REF)])
    assert d["leaks"] == [] and d["reference_path_touches"] == []


def test_escaped_newline_edit_is_not_a_leak(tmp_path: Path) -> None:
    cmd = "sed -i 's/pass/" + REF.split("'''doc'''\n")[1].replace("\n", "\\n") + "/' lib/m.py"
    d = _audit(tmp_path, [(cmd, ""), ("cat lib/m.py", REF)])
    assert d["leaks"] == []


def test_probes_are_recorded(tmp_path: Path) -> None:
    d = _audit(tmp_path, [("find / -name m.py", ""), ("pip show lib", "Name: lib")])
    assert "find-root" in d["probes"] and "pip" in d["probes"]
    assert d["leaks"] == []


def test_clean_episode(tmp_path: Path) -> None:
    d = _audit(tmp_path, [("ls", "lib"), ("python -m pytest -q", "1 passed")])
    assert d["probes"] == {} and d["leaks"] == [] and d["original_file_exposures"] == []


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_episode_sandbox_denies_reads_outside_the_episode(tmp_path: Path) -> None:
    secret_dir = Path.home() / ".cache" / "memfun-audit-sandbox-test"
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret = secret_dir / "solution.txt"
    secret.write_text("reference solution")
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    home.mkdir()
    (repo / "mine.txt").write_text("episode file")
    wrapper = REPO_ROOT / "evals/swarm/sandbox/run_in_episode_sandbox.sh"
    env = {**os.environ, "HOME": str(home)}

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(wrapper), *cmd], cwd=repo, env=env, capture_output=True, text=True, check=False
        )

    try:
        assert run("cat", "mine.txt").stdout.strip() == "episode file"  # own repo readable
        assert run("cat", str(secret)).returncode != 0  # the answer cache is not
        assert run("ls", str(Path.home() / "Ventures")).returncode != 0
        assert run("sh", "-c", f"echo x > {secret_dir}/escape").returncode != 0  # no writes out
        assert run("sh", "-c", "echo ok > new.txt").returncode == 0  # writes in
    finally:
        for f in secret_dir.iterdir():
            f.unlink()
        secret_dir.rmdir()


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_read_paths_must_travel_inside_the_command_not_the_environment(tmp_path: Path) -> None:
    """run_smoke launches a harness with a CLEANED environment: SANDBOX_READ_PATHS exported by the
    caller is dropped, so the harness cannot read its own interpreter (exit 126 in every episode).
    The paths have to be passed with env(1) inside the command."""
    tools = Path.home() / ".cache" / "memfun-sandbox-env-test"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "tool.txt").write_text("tool file")
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    home.mkdir()
    wrapper = str(REPO_ROOT / "evals/swarm/sandbox/run_in_episode_sandbox.sh")
    clean = {"PATH": "/usr/bin:/bin", "HOME": str(home)}  # what run_smoke's child_env gives

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(cmd), cwd=repo, env=clean, capture_output=True, text=True, check=False
        )

    try:
        via_env = run(
            "env", f"SANDBOX_READ_PATHS={tools}", wrapper, "cat", str(tools / "tool.txt")
        )
        assert via_env.returncode == 0 and via_env.stdout == "tool file"
        assert run(wrapper, "cat", str(tools / "tool.txt")).returncode != 0  # variable missing
    finally:
        for f in tools.iterdir():
            f.unlink()
        tools.rmdir()


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_the_runner_result_file_is_the_only_write_allowed_outside_the_episode(
    tmp_path: Path,
) -> None:
    # like a real run directory: under the home dir, not in the temp area writes are allowed in
    out = Path.home() / ".cache" / "memfun-result-file-test"
    out.mkdir(parents=True, exist_ok=True)
    repo, home = tmp_path / "repo", tmp_path / "home"
    for d in (repo, home):
        d.mkdir()
    wrapper = str(REPO_ROOT / "evals/swarm/sandbox/run_in_episode_sandbox.sh")
    result, other = out / "swarm-result.json", out / "other.txt"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "MEMFUN_SWARM_RESULT_JSON": str(result)}

    def run(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [wrapper, "sh", "-c", script],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        # O_EXCL create, exactly as memfun does
        created = run(f"set -C; echo '{{}}' > {result}")
        assert created.returncode == 0 and result.read_text() == "{}\n"
        result.unlink()
        # memfun writes a temp file next to the result and renames it over the name
        tmp = out / ".swarm-result.json.ab12cd.tmp"
        assert run(f"echo '{{}}' > {tmp} && mv {tmp} {out}/swarm-result.json").returncode == 0
        assert run(f"echo x > {other}").returncode != 0 and not other.exists()
        assert run(f"echo x > {out}/swarm-result.json.bak").returncode != 0
    finally:
        for f in out.iterdir():
            f.unlink()
        out.rmdir()


def test_a_tagged_run_name_still_selects_the_arm_parser(tmp_path: Path) -> None:
    """A RUN_TAG run name (bench-mini-swe-agent-hardD) is audited as mini, not an unknown arm."""
    _bundle(tmp_path)
    run = tmp_path / "bench-mini-swe-agent-hardD"
    _episode(
        run,
        [("find / -name m.py", ""), ("cat /x/bundles/c0s-lib-target/solution/lib/m.py", REF)],
        REF,
    )
    (run / "results.json").write_text(json.dumps({"meta": {}}))  # no "arm" recorded
    res = ac.audit_run(run, tmp_path / "bundles")
    assert res["arm"] == "mini-swe-agent"
    d = res["details"][0]
    assert d["commands"] == 2 and "find-root" in d["probes"] and d["leaks"]


def test_the_audit_reads_the_look_steps_probe_log_and_the_arm_from_episode_json(
    tmp_path: Path,
) -> None:
    """memfun's probe scripts (.memfun/probes.jsonl) are audited like any command; a run named
    look-P (no arm in results.json) is recognised from the arm recorded in episode.json."""
    _bundle(tmp_path)
    run = tmp_path / "look-P"
    ep = run / "c0s-lib-target" / "1"
    repo = ep / "c0s-lib-target-t1"
    (repo / ".memfun").mkdir(parents=True)
    (ep / "collected" / "lib").mkdir(parents=True)
    (ep / "collected" / "lib" / "m.py").write_text(REF)
    (ep / "swarm-result.json").write_text(json.dumps({"sub_tasks": {}}))
    (repo / ".memfun" / "probes.jsonl").write_text(
        json.dumps(
            {"scripts": ["import glob; print(glob.glob('/Users/x/.cache/*'))"], "output": "[]"}
        )
        + "\n"
    )
    (ep / "episode.json").write_text(
        json.dumps(
            {"episode": "c0s-lib-target-t1", "outcome": "pass", "repo": str(repo), "arm": "memfun"}
        )
    )
    (run / "results.json").write_text(json.dumps({"meta": {}}))
    res = ac.audit_run(run, tmp_path / "bundles")
    assert res["arm"] == "memfun"
    d = res["details"][0]
    assert d["commands"] == 1 and d["probes"]  # the probe script was seen and flagged
