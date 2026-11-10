#!/usr/bin/env python3
"""Baseline arm: mini-swe-agent, run the standard way, on a container the runner prepared.

The runner starts the task container (official image, internal network, history lock-down)
and then starts this driver ON THE HOST, in mini-swe-agent's own virtualenv.  The driver uses
mini-swe-agent unchanged: its shipped SWE-bench configuration, its default agent loop, its
litellm model class and its docker environment.  Two things differ from ``mini-extra
swebench``, both forced by the leaderboard's rules:

* the environment ATTACHES to the runner's container instead of starting its own (so the
  lock-down and the network isolation hold), and never stops it;
* the model endpoint is the runner's recording proxy (every request and response is the
  trajectory), with the instance's token budget enforced there.

The task (the problem statement, nothing else) arrives on stdin.  Output, in ``--out``:
``mini.traj.json`` (mini-swe-agent's own trajectory), ``submission.diff`` (the patch the agent
submitted, empty when it did not submit) and ``result.json`` (exit status, steps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--container", required=True, help="name of the running task container")
    ap.add_argument("--cwd", required=True, help="the repository inside the container")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint (the proxy)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="swebench.yaml", help="a shipped benchmark config")
    ap.add_argument("--step-limit", type=int, default=0, help="0 = the config's own limit")
    ap.add_argument("--wall-limit", type=int, default=0, help="seconds; 0 = none")
    ap.add_argument("--docker", default="docker")
    a = ap.parse_args(argv)
    task = sys.stdin.read()
    a.out.mkdir(parents=True, exist_ok=True)
    # Unknown model for litellm's price table: the proxy does the accounting.
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge

    class AttachedContainer(DockerEnvironment):
        """mini-swe-agent's docker environment on a container that already runs."""

        def _start_container(self) -> None:
            self.container_id = self.config.image  # the runner's container, by name

        def cleanup(self) -> None:  # the runner owns the container
            pass

    config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / a.config))
    agent_over: dict[str, object] = {"output_path": str(a.out / "mini.traj.json"), "cost_limit": 0}
    if a.step_limit:
        agent_over["step_limit"] = a.step_limit
    if a.wall_limit:
        agent_over["wall_time_limit_seconds"] = a.wall_limit
    config = recursive_merge(
        config,
        {
            "agent": agent_over,
            "environment": {"cwd": a.cwd, "executable": a.docker},
            "model": {
                "model_name": f"openai/{a.model}",
                "cost_tracking": "ignore_errors",
                "model_kwargs": {"api_base": a.base_url, "api_key": "dummy-not-a-secret"},
            },
        },
    )
    env_cfg = {k: v for k, v in config["environment"].items() if k != "environment_class"}
    env = AttachedContainer(image=a.container, **env_cfg)
    agent_cfg = {k: v for k, v in config["agent"].items() if k != "agent_class"}
    agent = DefaultAgent(get_model(config=config["model"]), env, **agent_cfg)
    result: dict[str, object] = {}
    error = ""
    try:
        result = agent.run(task) or {}
    except Exception as exc:  # the trajectory so far is still worth keeping
        error = f"{type(exc).__name__}: {exc}"
    submission = str(result.get("submission") or "")
    (a.out / "submission.diff").write_text(submission, "utf-8")
    summary = {
        "exit_status": str(result.get("exit_status") or ""),
        "error": error,
        "steps": int(getattr(agent, "n_calls", 0) or 0),
        "submitted_bytes": len(submission.encode()),
        "config": a.config,
    }
    (a.out / "result.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
