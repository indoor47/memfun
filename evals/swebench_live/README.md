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

Known limits: the agent arm is memfun only (a baseline arm is planned); a problem statement
over 120 KB cannot be passed (reported as an infrastructure error); images whose C library is
not glibc cannot run the bundle; submodule repositories are flagged, not locked down.
