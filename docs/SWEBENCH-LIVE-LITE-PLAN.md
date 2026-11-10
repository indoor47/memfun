# SWE-bench-Live Lite: the plan for the one run

Status: preparation. No Lite instance has been run, and none is run until the go is given.
Everything below was developed and tested on dev instances only (split `full` minus `lite`
minus `verified`).

## What is frozen before the first instance starts

One `run.json` per box holds all of it, and the runner refuses to continue a run whose settings
differ:

| What | Where it is recorded |
|---|---|
| the agent: one commit of this repository | `settings.agent_version` |
| the bundle built from it (from `uv.lock`) | `settings.bundle_digest` (digest of everything installed) |
| the runner | `settings.runner_commit` |
| the agent options | `settings.agent_options` |
| model, temperature, output limit, context window, reasoning effort | `settings.*` |
| the per-task stop (tokens) and the time limit | `settings.max_tokens_per_instance`, `settings.instance_timeout` |
| the prompt template (the same text for every instance) | `settings.prompt_template` |

The agent options of the run are the ones of the pilot:
`discover_tests=true reproduce=true repo_search=true probe_guard=off fix_detail=true
probe_rounds=2 max_fix_attempts=4 read_rounds=3 offline=true`.

The same bundle directory is copied to every box (not rebuilt per box) and
`bundle_check.py BUNDLE` is run on each: every box must print the same digest.

## One instance, start to finish

1. **Pull** the instance's image (`pull_s` recorded).
2. **Lock-down and self-tests** inside the container: no remotes, no ref that is not an
   ancestor of HEAD, reflogs expired; from inside, only the model proxy answers (no internet,
   no DNS, no other host port). A failed self-test is an infrastructure error, not an attempt.
3. **Rollout**: `memfun swarm` inside the container with the frozen options; the prompt is the
   template with `problem_statement` and nothing else. Every model request and response is
   recorded by the proxy on the host (`llm.jsonl`).
4. **Patch**: the diff of the working tree against the base tree, without the agent's own
   files; checked to apply to a clean checkout.
5. **Final**: `instance.json` gets `status: done`. A final instance is never rolled out again.
   An instance whose agent made no model call (the endpoint did not answer, or the process died
   before its first call) is not final: it is set aside with its log, listed in
   `attempts.jsonl`, and run again from scratch. The same death three times in a row closes
   the instance with an empty patch (`failure_class: agent_crash`).
6. **Sealed evaluation** (`runner.py --sealed DIR --evaluator DIR --gold-check`): the official
   evaluation of the patch, then of the reference patch, on the image that is already there.
   Everything the evaluator writes, including its own output and exit code, goes to `DIR`
   (outside the run directory, mode 0700). The run directory only gets a line in
   `sealed.jsonl` saying that the evaluations ran. Nothing prints or logs a verdict.
7. **Delete the image** (done by the last evaluation).

After the last instance on every box is final: `runner.py finalize` on each box (from then on
the runner refuses any rollout in that run), the shard run directories and sealed directories
are copied to one box, `submission.py merge` joins them (it refuses shards whose settings
differ or that share an instance), and `sealed.py unseal` (which refuses a run that is not
finalized) runs the official script once over all 300 instances: existing reports are reused,
a missing one is evaluated, and the official `results.json` is written. The same pass over the
sealed gold checks gives `gold-results.json`: which tasks nobody could have resolved on our
boxes.

## Boxes

Measured on the pilot (30 dev instances, a 16-CPU box, three instances at a time):

| Step | median | mean | 90th percentile | max |
|---|---|---|---|---|
| rollout (before the read step) | 44 s | 75 s | 171 s | 369 s |
| official evaluation of one patch | 58 s | 130 s | 174 s | 869 s |
| pull + evaluation of the reference patch | 117 s | 300 s | 809 s | 1372 s |
| image size | 3 GB | 7 GB | 18 GB | 19 GB |

- **Time**: about 8 to 10 minutes per instance through all seven steps. Three at a time on one
  box: about 60 instances in 3 hours, 100 in 5 hours.
- **Disk**: three images at once at up to 19 GB each, plus the evaluator and the bundle (about
  2 GB): 80 GB is enough, 160 GB is comfortable. Images are deleted as the run goes.
- **Image pulls**: Docker Hub allows about 100 unauthenticated pulls per 6 hours per address.
  The plan is 5 boxes with 60 instances each, so that no box comes near the limit and a few
  repeated pulls (a restart from scratch, a missing report at the end) have room. With a
  logged-in Docker Hub account 3 boxes of 100 would do.
- **Model path**: one key-holding proxy with one spend counter on one box; every other box
  reaches it through an SSH forward (`tunnel.sh`). The key is on that one box only. The runner
  waits when the endpoint stops answering (`--outage-wait`) instead of failing instances.
- The shards are fixed before the run: the 300 ids in the dataset's order, cut into five files
  of 60. A box runs only its own file.

## Budget

PILOT_COST_PLACEHOLDER

## Submission folder

`submission.py build --run-dir RUN --dataset lite.jsonl --out submissions/lite/memfun/MODEL
--model-label MODEL` writes what github.com/SWE-bench-Live/submission asks for:

- `preds.json` and `results.json` (the file the official script wrote);
- `trajs/<instance>/`: `llm.jsonl` (every request and response as sent and received, from the
  first prompt, with timestamps and usage), `agent.log`, `patch.diff`, `instance.json`, and the
  agent's own check logs (`checks.jsonl`, `reads.jsonl`, `probes.jsonl`, the reproduction
  script); `trajs/_attempts/` holds every attempt that was set aside for an infrastructure
  reason;
- `README.md`: the scaffold, the frozen settings, one rollout per instance, the number of
  restarts from scratch and why, the result, the audit;
- `run.json`, `attempts.jsonl`, `audit.json`.

It refuses to build while the **self-audit** has a violation. `submission.py audit` checks, for
every instance:

| Rule | Check |
|---|---|
| only `problem_statement` reaches the agent | the first prompt is exactly the template with the problem statement; no long line of `hints_text`, `test_patch` or the reference `patch` is in any prompt before the model itself wrote it; the instance id is in no prompt |
| no internet | the recorded network self-test shows only the proxy; the agent log shows no attempted web search, fetch or metadata refresh |
| no future history | the lock-down result is recorded |
| single rollout | every instance is final exactly once; every attempt that was set aside has a documented infrastructure reason |
| no credentials | no file of the trajectory holds text that looks like a key |

Things that are not violations but should be read by a person are listed as notes: a command
the agent ran that wanted the network (a test that fetches a URL), or the name of a target test
appearing in a prompt.

## Order of work on the day

1. Merge the agent PR; choose the commit; build the bundle once; copy it to the boxes; compare
   digests.
2. `grade.py setup` on every box (the pinned evaluator); fetch the `lite` split to a file;
   write the five shard files.
3. A last check on two dev instances per box with the frozen settings (`--split full`), then
   delete those run directories.
4. Start the five boxes (detached, one command each). Watch only: instances final, spend,
   disk. No verdict exists outside the sealed directories.
5. When all are final: finalize, copy, merge, unseal, audit, build. The research session and
   the user read the folder before any pull request is opened.

## What would stop the run

- The spend counter reaches the cap: the proxy refuses, the runner waits, nothing is lost.
- A box runs out of disk or pulls: its remaining instances are not started; they are run on
  the same box later (same run directory, same settings).
- The agent crashes the same way on many instances: stop, because that is a defect of the
  frozen agent. Fixing it means a new frozen agent and a new run of all 300, which the rules
  allow only if no evaluation result of the first run was seen. The sealed directories make
  that checkable.
