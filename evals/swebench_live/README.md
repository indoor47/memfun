# SWE-bench-Live runner

Runs a coding agent on [SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live) instances
inside the official per-instance Docker image, one rollout per instance, and prepares the
files a leaderboard submission needs (`preds.json`, raw trajectories, the official
`results.json`).

| File | What it does |
|---|---|
| `dataset.py` | Fetches a split to JSONL; splits a row into the agent's input (`problem_statement` only) and what the runner needs; seeded dev sample from `full` minus the frozen splits. |
| `proxy.py` | The only network route out of a task container: records every LLM request and response per instance (`llm.jsonl`), counts tokens and dollars, enforces a token and time budget. |
| `lockdown.sh` | Runs in the container before the rollout: removes remotes, refs, reflogs and objects newer than the checked-out commit; self-checks. |
| `runner.py` | Per instance: pull, start on an internal network, find the repository, lock down history, network self-test, rollout, collect and check the patch, delete the image. Resume, frozen settings, `finalize`. |
| `grade.py` | The official evaluator at a pinned commit; refuses a run that is not finalized. |
| `provision.sh` | Builds the read-only agent bundle on a Linux x86_64 Docker box; teardown. |
| `mini_driver.py` | The baseline arm: mini-swe-agent, unchanged, attached to the runner's container. |

## Rules and the mechanism behind each

| Rule | Mechanism | Test |
|---|---|---|
| Only `problem_statement` reaches the agent | `dataset.episode_input`; opaque container name and proxy token; config in a throwaway `HOME` | `test_only_the_problem_statement_reaches_the_container` |
| Official container | the row's `docker_image`, repository under `/testbed` as the evaluator uses it | `test_a_rollout_end_to_end_...` |
| No future history | `lockdown.sh` and its self-check before the rollout | `test_lockdown_removes_everything_newer_than_head` |
| No internet | `--internal` Docker network; self-test from inside the container (internet, DNS, host services); optional host firewall rule on the runner's bridge only | `test_a_failed_self_test_stops_before_the_rollout` |
| No ground-truth feedback | `grade.py` needs a finalized run; a finalized run cannot roll out | `test_settings_are_frozen_and_a_finalized_run_cannot_roll_out` |
| Single rollout, frozen settings | `run.json`; finished instances never re-run; infrastructure failures re-run only with `--retry-infra`, logged in `attempts.jsonl`, earlier attempt kept | `test_single_rollout_resume_and_documented_infra_retry` |
| Raw trajectories | `proxy.py` records before the agent gets the answer | `tests/test_swebench_live_proxy.py` |
| General prompts only | one prompt template, filled with the problem statement | `test_prompts_are_general` |

## Use

```bash
# on the Docker box (all state under one directory)
evals/swebench_live/provision.sh up /root/memfun-live /root/memfun-live/repo
python3 evals/swebench_live/dataset.py fetch --split full --out data/full.jsonl
python3 evals/swebench_live/runner.py run --dataset data/full.jsonl --ids-file data/dev.txt \
    --run-dir runs/dev1 --upstream http://127.0.0.1:18100/v1 --model MODEL \
    --bundle /root/memfun-live/bundle --host-firewall
python3 evals/swebench_live/runner.py finalize --run-dir runs/dev1
python3 evals/swebench_live/grade.py grade --dir /root/memfun-live/eval --run-dir runs/dev1 --split full
```

Never develop on `lite` or `verified` instances: use `dataset.py dev-sample`.

The agent's own verification is off unless switched on, per run and recorded in the frozen
settings: `--agent-option discover_tests=true --agent-option reproduce=true --agent-option
probe_guard=off` (the runner writes them to `.memfun/verify.json` in the repository, which is
never part of a patch; `probe_guard=off` is right here because the container is disposable).

## Gold check first

`grade.py gold` runs the official evaluator with the reference patch. Do it for every instance
before trusting a verdict on it: an instance whose reference patch does not pass cannot be
resolved by anyone on that machine. On the first dev sample (from `full` minus the frozen
splits), 2 of the first 8 instances judged failed their own gold patch, both because a
pass-to-pass test needs the live internet:

| Instance | Why the gold patch "fails" |
|---|---|
| `deepset-ai__haystack-6847` | the target test passes; 3 pass-to-pass tests fetch a web page whose content has changed |
| `geopandas__geopandas-3240` | the target test passes; 1 pass-to-pass test downloads a file from a URL |

A failed image pull is reported by the evaluator as "not found in registry" and counted as an
error, not a failure: retry it. At the pinned commit the evaluator's gold mode exits with an
error after writing its results whenever a gold patch passes; `grade.py` uses the results
when they are fresh and complete.

## Planning facts for a full run

- **Image size and pull time vary a lot.** Most images of the dev sample are 2 to 4 GB and
  pull in about a minute; one was 22.7 GB and took 5 minutes.
- **Docker Hub limits unauthenticated pulls** to about 100 per 6 hours per address. 300
  instances pulled once for the rollout and again for grading do not fit one machine in one
  window. Options: several machines, authenticated pulls, or grading each instance right after
  its own rollout while its image is cached (keeping every evaluation output away from every
  agent, and never grading an instance whose rollout could still be repeated).
- **Grading time is the instance's whole test suite**: from 1 to 12 minutes per instance on
  an 8-core machine.
- **A finished agent used to hang** instead of exiting (fixed in memfun since); the runner
  still reaps an agent that wrote its result and does not exit (`reaped_after_result`).

## Baseline arm

`--arm mini --mini-python /path/to/venv/bin/python` runs mini-swe-agent on the same container,
network and proxy: `uv venv mini/venv && uv pip install --python mini/venv/bin/python
mini-swe-agent mcp` (litellm imports `mcp`). It runs on the host and drives the prepared
container with `docker exec`; its patch is what the agent submits. Give it a token budget
(`--max-tokens-per-instance`): it is the only spend limit, the model's price is unknown to
litellm.

Known limits: a problem statement
over 120 KB cannot be passed (reported as an infrastructure error); images whose C library is
not glibc cannot run the bundle; submodule repositories are flagged, not locked down.
