# Swarm smoke runner (`run_smoke.py`)

`evals/swarm/run_smoke.py` runs the smoke bundles in `smoke/` through `memfun swarm`
against one OpenAI-compatible model server, grades each episode with hidden tests, and
writes `<out-dir>/<run_id>/results.json` plus a summary table. Every episode reports to
ONE dashboard.

## Threat model: grading is NOT a sandbox

The swarm's agents run **unconfined Python** as your user. The runner keeps grading honest
against *accidental* and *casual* tampering (owned files only, no symlinks, a separate grader
dir, one pytest process per unit, a reference self-check), but anything the agent's process
can read, it can read: the bundles' `hidden/` tests and `solution/` files are on the same
filesystem (absolute paths work from anywhere), and so are the episode dir's artifacts.
The default `--out-dir` is outside the memfun checkout so they are not a short `../` away
and memfun's own pytest config does not apply to episodes, but that is hygiene, not
isolation. For adversarial agents a real run needs a container per episode and an
out-of-process oracle that holds the hidden tests. Event-bus metrics (tokens, timings) come
from a DB the episode can write to and are trusted only as telemetry, never for grading.

## Commands

Look at the plan first. `--dry-plan` prints episodes, repos, commands, ports and timeouts,
and creates, starts and contacts nothing:

```bash
uv run python evals/swarm/run_smoke.py --base-url http://127.0.0.1:8441/v1 \
    --bundles textkit-3 formats-2 --parallel-tasks 2 --dry-plan
```

**Local MTPLX** (2 bundles, 2 at once; `--memfun-cmd` points at the venv that has the tool
and decomposer fixes):

```bash
uv run python evals/swarm/run_smoke.py \
    --base-url http://127.0.0.1:8441/v1 \
    --bundles textkit-3 formats-2 --parallel-tasks 2 --max-agents 3 \
    --memfun-cmd /path/to/venv/bin/memfun \
    --run-id mtplx-smoke-1
```

**B300 through the SSH tunnel** (`launch_pod.py tunnel` / `watch`, local port 18000). Run
all 6 at once, then again with more agents:

```bash
export MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:18000/metrics   # dashboard server strip
uv run python evals/swarm/run_smoke.py --base-url http://127.0.0.1:18000/v1 \
    --parallel-tasks 6 --max-agents 4 --metrics-url http://127.0.0.1:18000/metrics \
    --run-id b300-a4
uv run python evals/swarm/run_smoke.py --base-url http://127.0.0.1:18000/v1 \
    --parallel-tasks 6 --max-agents 8 --metrics-url http://127.0.0.1:18000/metrics \
    --run-id b300-a8
```

| flag | default | meaning |
|---|---|---|
| `--base-url` | required | model server (`.../v1`). Port **8434 is always refused** |
| `--model` | the single id at `GET <base-url>/models` | served model id |
| `--bundles` | all 6 | known ids only, space or comma separated, deduped |
| `--trials` | 1 | episodes per bundle (1..1000) |
| `--max-agents` | 4 | passed as `memfun swarm --max-pool` (1..100) |
| `--parallel-tasks` | 1 | episodes running at once (1..64) |
| `--task-timeout` | 900 | passed as `memfun swarm --timeout` (seconds) |
| `--episode-timeout` | 2400 | hard cap per episode; then the swarm's process group is killed |
| `--grader-timeout` | 300 | per unit's hidden-test process |
| `--max-tokens` | 8192 | `[llm] max_tokens` in the episode's `memfun.toml` |
| `--memfun-cmd` | `memfun` | memfun executable (shell-split, so `uv run memfun` works) |
| `--out-dir` | `$TMPDIR/memfun-swarm-runs` | refused if inside the memfun checkout |
| `--run-id` | UTC timestamp | `[A-Za-z0-9._-]`, no `..`; an existing run dir is refused |
| `--dashboard-port` / `--no-dashboard` | 8081 | the shared dashboard (below) |
| `--metrics-url` | off | Prometheus endpoint for run-level server token counters |

Exit code: 0 done, 1 done but the arm is VOID, 2 bad arguments or refused port, 130
interrupted. On Ctrl-C or SIGTERM (also while the dashboard is still starting) the runner
kills only the process groups it started (the swarms and a dashboard it launched, never a
reused one), drops queued episodes, and still writes `results.json` with
`"interrupted": true` and `not_run`. `results.json` is also rewritten atomically after
every finished episode, so it survives even a SIGKILL of the runner.

**After a SIGKILL of the runner** its swarm process groups keep running (they are in their
own sessions). Every group the runner starts is appended to `<run_dir>/pgids.txt`
(`<pgid> <episode>`). Clean up with:

```bash
while read -r pgid ep; do ps -o pid=,command= -g "$pgid" && kill -TERM -- "-$pgid"; done \
    < <run_dir>/pgids.txt
```

Before anything runs, the runner checks the server with `GET <base-url>/models`. That
request also picks the model when `--model` is not given. `--base-url`/`--metrics-url` must
be http(s); a non-loopback host prints a loud WARNING (use the SSH tunnel). The runner's GETs
send no credentials, use no proxies and refuse cross-host redirects.

## What one episode does

Episode = one bundle x one trial. Its directory is `runs/<run_id>/<bundle>/<trial>/`:

| path | content |
|---|---|
| `<bundle>-t<trial>/` | the throwaway git repo (its name is the dashboard's project label) |
| `home/` | the swarm's `HOME` (keeps `~/.memfun/credentials.json` / `config.toml` out) |
| `swarm.log` | `memfun swarm` stdout+stderr |
| `collected/` | copy of the owned files picked for grading |
| `graded/` | copy of the graded tree (clean `repo/` + collected owned files) |
| `grader.txt` | per-unit grader output tails |
| `episode.json` | every metric of the episode |

Collection and grading happen in a runner-private temp dir; `collected/` and `graded/` are
archived afterwards by an exclusive rename (refused if the agent planted those names), and
`episode.json` / `grader.txt` / `results.json` are written via an exclusive no-follow temp
file + rename, so a planted symlink is replaced, never written through. The runner reads
nothing back from the episode dir for grading.

1. Copy `repo/` and write `memfun.toml` with one `[llm]` block (`provider = "openai"`,
   `base_url`, `model`, `api_key_env = "MEMFUN_SGLANG_DUMMY_KEY"`, `max_tokens`) and
   `[backend] sqlite_path = ".memfun/memfun.db"`. Add a `.gitignore` (`.memfun/`, caches)
   and, if the bundle has no pytest config, an empty `pytest.ini` so the agents' own pytest
   runs use the episode repo as rootdir. Then `git init` and commit.
2. Run `memfun swarm "<task spec + 'implement in the existing stubs, do not modify tests/'>"
   --max-pool N --timeout S --no-push` from the repo, in its own process group. The env is
   built from an allowlist: `PATH LANG LC_ALL LC_CTYPE TERM TMPDIR`, `HOME=<episode>/home`,
   the dummy key, plus `PYTHONUNBUFFERED`, `COLUMNS` and `GIT_CONFIG_NOSYSTEM`. No
   `ANTHROPIC_API_KEY`, `RUNPOD_*` or dashboard token reaches it. At `--episode-timeout`,
   only that process group gets SIGTERM, then SIGKILL. The runner never uses `pkill` and
   never starts or stops a model server.
3. Collect the owned files and grade them with `validate_bundles.graded_tree` and
   `run_grader` (imported, not copied). The graded tree is a clean `repo/` plus only the
   owned files. **Each unit's hidden test file, and `test_integration.py`, runs in its own
   pytest process** from a separate grader dir with a pinned `-c`/`--rootdir`/
   `--confcutdir`, `python -P`, sockets disabled and its own timeout, so an unguarded
   `sys.exit()` / `os._exit()` / `KeyboardInterrupt` at import in one unit only fails that
   unit (flag `grader_crash`; the others keep their results).
4. **Reference self-check:** before the first episode of a bundle, its `solution/` is graded
   once through the same machinery (cached). If that fails, the grader is broken and every
   episode of the bundle is a `harness_error`; if it passes, a crash while grading agent
   code is the agent's (`model_failure`/`partial`).

### Owned-file collection rule

Since PR #45 agents write into their own worktrees
(`.memfun/worktrees/<workflow_id>/<task_id>`), and nothing merges their work back yet. For
each path a unit `owns`:

- Only **regular files**, opened with no symlink following, count: an owned path that is
  a symlink, sits under a symlinked directory, is not a regular file (fifo, device) or is
  larger than 1 MB is **tampering**: the unit contributes no files, fails, and the episode
  gets `symlink_tampering` (details in `collection.tampering`). Symlinked worktree dirs
  (or a symlinked `.memfun/worktrees`) are skipped and reported the same way.
- The candidates are the copies that **differ from the bundle's start tree**. A file that
  is not in `repo/` counts as changed once it exists.
- **Worktree copies win.** If several worktrees changed the path, the newest by mtime is
  graded and the unit gets `duplicate_work`.
- The **project-root copy** is used only when no worktree changed the path.
- If none of a unit's owned paths changed, the unit is `no_output` (and fails).
- A unit's `sub_tasks` are the worktree task ids that changed its owned paths.
- Anything else changed (`stray_changes`) is ignored by grading. Changes under `tests/` set
  `tests_modified`.

The runner collects before it cleans up anything, and it cleans up nothing except the
episode dir it created, which it keeps. It works whether the worktrees are still there or
not.

### Metrics (from the episode's event-bus DB)

`memfun swarm` publishes its dashboard events (`workflow.started`, `task.published`,
`task.picked_up`, `task.completed`, `llm.call` with `prompt_tokens`/`completion_tokens`/
`model`/`role`/`agent_name`/`task_id`) on its T1 event bus, which is
`<repo>/.memfun/memfun.db`, table `events`, topic `memfun.distributed.events`. The runner
reads that table **read-only** and computes:

- `tokens`: prompt/completion totals, `by_role`, `calls`, `calls_with_usage`, and `source`
  (`llm.call events`, `llm.call events (no usage)` or `none`). With `--metrics-url`,
  `results.server_usage` holds the server's own prompt/generation counter deltas over the
  run (`sglang:*_tokens_total` / `vllm:*`). That is the fallback when events are missing.
  It is per run, not per episode, and counts **all** traffic on that server (on a shared
  server such as a local MTPLX on 8441, other clients' tokens too).
- `agents_used`: the number of distinct task ids that were picked up or made an LLM call
  (one agent instance per sub-task). `agent_names` lists the agent types.
- `peak_concurrency`: the largest overlap of `[picked_up, completed]` intervals. The engine
  stamps these per parallel group. `peak_llm_concurrency` is the largest overlap of
  in-flight `llm.call`s.
- `decompose_s` (`workflow.started` to the first `task.published`), `exec_wall_s` (the
  first pick-up to the last completion), and `sub_tasks[]` with start/end.
- `primary_sub_tasks`: sub-tasks that are not derived from another one, so revisions,
  pair re-dispatches and reviews do not count.

Flags (recorded, not a pass/fail reason by themselves):

- `decomposition_fallback`: a fan-out bundle ran as one sub-task or none (a serial
  collapse).
- `duplicate_work`, `tests_modified`, `symlink_tampering`
- `grader_crash` (a unit's pytest exited with rc not 0/1), `grader_timeout`
- `episode_timeout`, `swarm_crash` (non-zero exit without a result), `server_errors`
  (recorded for ALL episodes, passing ones included, with sample lines and a count)

Per-episode `warnings` are log lines that match `MCP tool discovery`,
`Failed to discover MCP tools`, `Task decomposition failed` or `Traceback`.

## Classification (each episode is exactly one)

The same rules apply to passing and failing episodes. A harness fault is blamed only when
it explains the outcome. In order:

1. **`harness_error`** (never counted against the model):
   - the runner could not set up or grade the episode: missing/invalid bundle files, model
     server unreachable at preflight / model not served / ambiguous `/models`, dashboard
     could not start or its port is held by something else, git/repo setup failure, the
     **reference self-check failed**, an exception in the runner's grading code, disk full
     or another OS error, run aborted;
   - or the episode produced **no owned output at all** AND one of: the episode timeout
     hit, `memfun swarm` exited non-zero without a result, or model-server errors were
     seen (structured `llm.call` errors such as `APIConnectionError` / connection refused /
     5xx, or log lines naming the client stack: `litellm`, `openai.`, `httpx`, `httpcore`;
     agent code that merely contains such words does not count).
2. Otherwise the output is graded on its merits, whatever else happened (timeouts, crashes
   and server errors stay flags):
   - **`pass`**: every unit's hidden test file passes, and `test_integration.py` too when
     the bundle has one;
   - **`partial`**: some hidden tests pass;
   - **`model_failure`**: no hidden test passes (this includes a swarm that finished and
     reported failure, agent code that crashes or hangs its unit's grader process).

The record keeps the reason string and the last 40 log lines. `harness_error_rate` is over
ALL episodes; `episode_pass_rate` = pass / (episodes - harness errors) and `unit_pass_rate`
covers the units of graded episodes. The summary prints the raw outcome counts too, so any
rate can be recomputed. If **more than 5 %** of an arm's episodes are harness errors, the
arm (this run) is marked **VOID**: `"void": true`, `ARM VOID` in the summary, exit code 1.

## The shared dashboard

`memfun swarm` starts an embedded dashboard per project root (ports 8081-8099). It first
reads `<project>/.memfun/dashboard.lock` (`{"port", "pid"}`), and if that pid is alive and
`GET 127.0.0.1:<port>/api/health` answers, it **reuses** that dashboard instead of starting
one (`commands/chat.py::_start_dashboard_background`). In parallel, every episode is its
own project root, so without help you would get N dashboards on N ports. The runner works
around this with no production change:

1. **One dashboard.** If a memfun dashboard already answers on `--dashboard-port` (default
   8081), the runner reuses it and never stops it, and prints a one-line notice: that may
   be the owner's own dashboard (e.g. an embedded `memfun chat` one), which then fills with
   episode events. That is the default on purpose (one dashboard for the phone); pass
   `--no-dashboard` or another `--dashboard-port` to avoid it. If the port is free, the runner starts
   `<memfun-cmd> dashboard --remote --port <p>` in `<run_dir>/_dashboard/` in its own
   process group, and stops only that one at the end. If the port is held by something
   else, that is a harness error.
2. **The lock.** Each episode repo gets `.memfun/dashboard.lock` pointing at that
   dashboard. `memfun swarm` then reuses it (banner `Dashboard: http://localhost:<p>`) and
   starts no dashboard of its own.
3. **Forwarding.** In reuse mode `memfun swarm` does not forward events itself. The
   runner tails each episode's `.memfun/memfun.db` read-only and POSTs every event
   unchanged (including `llm.call` model/role/tokens) to `http://127.0.0.1:<p>/api/event`.
   Loopback, so no token is needed.

The owner's phone app watches that one dashboard. Each episode shows up as its own project
(`<bundle>-t<trial>`). Remote clients need the dashboard token as before. `memfun dashboard
--remote` creates the token file if it is missing, and the runner never reads or prints it.
The dashboard process gets the real `HOME` plus `MEMFUN_DASHBOARD_TOKEN`,
`MEMFUN_INFERENCE_METRICS_URL` and `MEMFUN_POD_RATE_USD` when they are set. With
`--no-dashboard`, nothing is shared and each swarm may start its own embedded dashboard.

## results.json (`"schema_version": 1`)

```text
schema_version, run_id, run_dir, created_at, base_url, model, memfun_cmd, bundles, trials,
max_agents, parallel_tasks, episode_timeout, task_timeout,
dashboard: {port, started_by_runner} | null, fatal, interrupted, not_run, wall_s,
peak_parallel_episodes,
server_usage: {available, prompt?, completion?} | null,
void, void_threshold,
overall / bundles.<id>: {episodes, outcomes{pass,partial,model_failure,harness_error},
    episode_pass_rate, unit_pass_rate, harness_error_rate, tokens{prompt,completion},
    wall_s_mean, agents_used_max, peak_concurrency, flags[]},
units."<bundle>/<unit>": {graded, passed, no_output, pass_rate},
harness_errors[]: {episode, reason, log_tail},
episodes[]: the episode.json records:
    bundle, kind, trial, episode, dir, repo, started_at, wall_s, outcome, reason, flags[],
    warnings[], returncode, timed_out, produced_result, server_errors, server_error_lines[],
    log_tail, dashboard{forwarded, post_errors},
    collection{worktrees[], files.<path>{source, changed_in[], duplicate, tampered},
               units.<id>{...}, tampering[], stray_changes[], tests_modified},
    grade{hidden_passed, hidden_failed,
          units.<id>{passed, rc, tests{passed, failed}, crash, timeout},
          integration{...} | null, all_pass, crashes{<unit>: rc}, timeouts[]},
    units.<id>{status: pass|fail|no_output, output, owned_changed, sub_tasks[],
               duplicate_work, symlink_tampering, tests{passed, failed}, grader_rc},
    metrics{tokens{prompt, completion, total, calls, calls_with_usage, by_role, source},
            llm_errors[], llm_error_count, agents_used, agent_names[], peak_concurrency,
            peak_llm_concurrency, decompose_s, exec_wall_s, primary_sub_tasks,
            workflow_ids[], sub_tasks[]{task_id, agent_name, published, start, end,
            success, derived, start_rel_s, end_rel_s}, event_count}
```

Keys that only make sense after the swarm ran are missing from episodes that stopped
earlier, such as a preflight harness error.

## Tests

`tests/test_swarm_run_smoke.py` is hermetic: no network and no model. It drives the runner
with a fake memfun (`tests/_fake_memfun.py`, given via `--memfun-cmd`) and a loopback
`/v1/models` stub. The fake writes events with memfun's own SQLite schema and
`DistributedEvent`. The tests cover:

- the oracle and null arms for all 6 bundles through the runner's grader path
- each collection rule and flag; symlinked owned files (absolute, relative, parent
  component, root parent, worktree dir, worktrees root), fifos and oversized files never
  count; planted artifact names are refused
- one crashing unit (sys.exit, SystemExit, KeyboardInterrupt, os._exit, syntax error,
  pytest.exit, infinite loop) does not zero the others; a broken reference is a harness error
- crash / timeout / server errors without output as harness errors; with output graded
- signals during dashboard startup, partial results + `pgids.txt` after a runner SIGKILL
- argument validation, cross-host redirects, read-only event DB access, pytest rootdir
- VOID
- the parallel bound
- a clean env
- the process-group kill, including SIGTERM to the runner
- the shared dashboard (start, reuse, forwarding, and the real swarm helper honouring the
  lockfile)
- port 8434 refusal
- `--dry-plan`
