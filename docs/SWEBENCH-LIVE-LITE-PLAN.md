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

The settings of the run are written in one place, `evals/swebench_live/box_run.sh`, so that the
frozen commit names them and every box runs the same ones:

- solver `context-first`, model `qwen-3.8-27b`, temperature 0.7, reasoning effort `none`,
  at most 32,768 output tokens per call, context window 131,072;
- agent options `discover_tests=true reproduce=true repo_search=true probe_guard=off
  fix_detail=true probe_rounds=2 max_fix_attempts=4 read_rounds=2 read_offer=false
  offline=true`;
- per instance: stop at 520,000 tokens, 1,800 s, 2 CPUs, 8 GB.

These are the options of the pilot with the read step answering requests without inviting them
(`read_offer=false`) and two rounds instead of three. That pair was checked on the 12 pilot
tasks that had used the read step (see "What the practice runs say").

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
   files; checked to apply to a clean checkout. When the tree also changed in files the agent
   never edited (the project's tests wrote them while the agent ran them), the patch holds the
   files the agent edited and the whole difference is kept next to it (`worktree.diff`).
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

- **Time on a box**: about 8 to 10 minutes per instance through all seven steps. Three at a
  time on one box: about 60 instances in 3 hours, 100 in 5 hours, when the model path keeps
  up.
- **The model path sets the pace.** The Cerebras account is limited to 150,000 tokens per
  minute (read from the live `x-ratelimit-*` headers; the key-holding proxy queues requests
  to stay at 80% of it and saw one rate-limit refusal, retried, in about 2,200 requests). One
  rollout that is not waiting uses about 100,000 uncached tokens per minute, and a task uses
  about 130,000 in total. So
  300 tasks need at least 5.5 hours of that limit however many boxes there are, and fifteen
  rollouts at once would each crawl at a fifteenth of their speed towards the 30-minute time
  limit. The run therefore has ONE rollout in flight per box (`ROLLOUT_SLOTS=1` in
  `box_run.sh`, five in all) while the box's other two workers grade: a typical rollout then
  takes 3 to 5 minutes and a task at its token stop about 20. Expected wall time: 6 to 7
  hours. An endpoint without such a limit (a rented GPU serving the same model) can carry
  three rollouts per box and finish in about 3 hours; it is a different serving path and has
  to pass the dev check first.
- **Disk**: three images at once at up to 19 GB each, plus the evaluator and the bundle (about
  2 GB): 80 GB is enough, 160 GB is comfortable. Images are deleted as the run goes.
- **Image pulls**: Docker Hub allows about 100 unauthenticated pulls per 6 hours per address.
  The plan is 5 boxes with 60 instances each, so that no box comes near the limit and a few
  repeated pulls (a restart from scratch, a missing report at the end) have room. With a
  logged-in Docker Hub account 3 boxes of 100 would do.
- **Size**: 16 shared CPUs, 32 GB of memory and 360 GB of disk per box (Hetzner `cpx51`, about
  EUR 0.14 per hour; the pilot box was that size with a larger disk). Three instances at a time
  use 6 CPUs and 24 GB for the rollouts; the evaluations use the rest.
- **Model path**: one key-holding proxy with one spend counter on one box; every other box
  reaches it through an SSH forward (`tunnel.sh`). The key is on that one box only. The runner
  waits when the endpoint stops answering (`--outage-wait`) instead of failing instances.
- The shards are fixed before the run: the 300 ids in the dataset's order, dealt out in turn
  to five files of 60 (the split has 70 repositories and the largest has 30 instances; dealing
  in turn spreads a repository with large images or a slow test suite over the boxes). A box
  runs only its own file.

### The commands

`B=/root/memfun-live` on every box. `COMMIT` is the frozen commit of this repository.

Once, on the box that holds the key (box 1):

```bash
# the key-holding proxy with the spend cap of the run (listens on 127.0.0.1 only)
python3 $B/repo/evals/swarm/cerebras_proxy.py --port 18431 --reasoning-effort none \
    --key-file $B/secrets/cerebras.key --max-cost-usd CAP
# the bundle is built ONCE, here, and copied to the other boxes as a directory
git -C $B/repo fetch -q origin && git -C $B/repo checkout -q --detach COMMIT
bash $B/repo/evals/swebench_live/provision.sh up $B $B/repo
python3 $B/repo/evals/swebench_live/dataset.py fetch --split lite --out $B/data/lite.jsonl
python3 - <<'PY'   # five shard files, dealt out in turn in the dataset's order
import json
ids = [json.loads(line)["instance_id"] for line in open("/root/memfun-live/data/lite.jsonl")]
for k in range(5):
    open(f"/root/memfun-live/data/shard-{k + 1}.txt", "w").write("\n".join(ids[k::5]) + "\n")
PY
```

On every box (docker, git and uv installed; the other boxes get `bundle/`, `data/lite.jsonl`
and their shard file copied from box 1, and keep an SSH forward of port 18431 to box 1 open):

```bash
git -C $B/repo fetch -q origin && git -C $B/repo checkout -q --detach COMMIT
python3 $B/repo/evals/swebench_live/bundle_check.py $B/bundle      # the same digest everywhere
python3 $B/repo/evals/swebench_live/grade.py setup --dir $B/eval --uv "$(command -v uv)"
# the last check: two DEV instances through the whole pipeline, then that run is deleted
$B/repo/evals/swebench_live/box_run.sh $B check $B/data/check-ids.txt $B/data/full.jsonl full
rm -rf $B/runs/check $B/sealed/check
# the run (detached; N is the number of the box's shard)
setsid nohup $B/repo/evals/swebench_live/box_run.sh $B lite-N $B/data/shard-N.txt \
    $B/data/lite.jsonl lite > $B/lite-N.out 2>&1 < /dev/null &
```

`box_run.sh` runs passes until every instance of the shard is final and prints `BOXDONE`. It
works on three instances at a time, of which one is in its rollout (`WORKERS`, `ROLLOUT_SLOTS`
in its environment; see "The model path sets the pace"). What
may be looked at while it runs: the lines it prints (instance, status, time, calls, tokens,
dollars, patch size, flags), the proxy's spend counter, `df`. Nothing else: the sealed
directories are not opened.

When every box has printed `BOXDONE`:

```bash
python3 $B/repo/evals/swebench_live/runner.py finalize --run-dir $B/runs/lite-N   # on each box
# copy runs/lite-N and sealed/lite-N of every box to box 1, then on box 1:
mkdir -p $B/sealed/lite && for n in 1 2 3 4 5; do cp -a $B/sealed/lite-$n/. $B/sealed/lite/; done
R=$B/repo/evals/swebench_live
python3 $R/submission.py merge --run-dir $B/runs/lite-1 --run-dir $B/runs/lite-2 \
    --run-dir $B/runs/lite-3 --run-dir $B/runs/lite-4 --run-dir $B/runs/lite-5 --out $B/runs/lite
python3 $R/sealed.py unseal --dir $B/eval --run-dir $B/runs/lite --sealed $B/sealed/lite --split lite
python3 $R/submission.py audit --run-dir $B/runs/lite --dataset $B/data/lite.jsonl > $B/audit.json
# alarms about lines of the test patch or reference patch in a prompt: look them up in the image
python3 $R/submission.py repo-text --run-dir $B/runs/lite --dataset $B/data/lite.jsonl
python3 $R/submission.py audit --run-dir $B/runs/lite --dataset $B/data/lite.jsonl > $B/audit.json
python3 $R/submission.py build --run-dir $B/runs/lite --dataset $B/data/lite.jsonl \
    --out $B/submissions/lite/memfun/Qwen3.8-27B --model-label Qwen3.8-27B
```

## Budget

Model (Cerebras list price for `qwen-3.8-27b`: $0.99 per million input tokens, $1.49 per
million output tokens), measured with the run's settings on dev tasks:

| Kind of task | share in the pilot | cost per task |
|---|---|---|
| never asks to read | 18 of 30 | $0.11 |
| reads, or is one of the tasks that read when invited | 12 of 30 | $0.24 |

That is about $0.16 per task and about $50 for 300. The stop of 520,000 tokens bounds one task
at about $0.52, so the hard ceiling is $156; the proxy's spend cap is set to $65, and when it is
reached the run waits instead of failing instances.

Boxes: five at about EUR 0.14 per hour for 6 to 7 hours of run plus setup and the final
steps, and the key-holding box: under EUR 10.

## What the practice runs say

All on dev tasks (never Lite), official grading, one rollout per task at temperature 0.7,
model `qwen-3.8-27b`.

| Set | Agent | Resolved | Cost |
|---|---|---|---|
| 30 random dev tasks whose reference patch passes | mini-swe-agent, stopped at $0.75 per task | 7 (8 when a cut-off run's edits are graded) | $13.51 |
| the same 30 | memfun before the fixes for runs that ended without a patch | 7, and 7 runs without a patch | $3.69 |
| the same 30 | memfun with those fixes, read step invited, 3 rounds | 8, and 1 run without a patch | $5.80 |
| the 12 of those 30 that had used the read step | memfun with the run's settings (2 rounds, not invited) | 4 (2 in the row above on the same 12), no run without a patch | $2.90 ($3.85) |
| 15 dev tasks that older published agents mostly solved | memfun, three runs across these versions | 12, 14, 13 | $1.56 to about $3 |
| the same 15 | mini-swe-agent | 13 | $4.81 |

**How much these numbers can carry.** One rollout per task at temperature 0.7 moves the count
by several tasks from run to run: the two memfun runs on the 30 tasks (7 and 8 resolved) have
only 4 resolved tasks in common, and 11 tasks were resolved by one run or the other. So 8
against 7 is not an improvement, and neither is 4 against 2 on 12 tasks. What the fixes changed
for certain is the number of runs that hand in nothing (7, then 1, then 0 on the 12). The
resolve rate on this pool is somewhere around a quarter, with a wide margin on 30 tasks.

**What it says about Lite.** The dev pool is `full` minus `lite` minus `verified`. Older
published agents resolve about 10% of it and 16 to 18% of Lite, so Lite is the easier set, by
an amount these runs cannot pin down. No estimate of the Lite score is made here.

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

One alarm needs evidence the audit does not have. A hidden test patch often repeats lines of
tests that are already in the repository, and the agent reads existing tests. The audit sees
those lines in a prompt and cannot know where they came from. On 97 finished dev runs this
raised a violation for 2 distinct tasks (about 4%). `submission.py repo-text` settles each
case: it searches the instance's image, as it was before the rollout and without network, for
every distinctive line of the forbidden fields and records the lines it finds in
`in-repo.json`. The audit does not count those and says so in a note; a line the image does
not hold stays a violation. On the two dev cases every flagged line was in the image (3 and 9
lines). It needs the images of the flagged instances again: about a dozen pulls for 300.

## Order of work on the day

The commands are under "Boxes".

1. Choose the commit; build the bundle once; copy it to the boxes; compare digests.
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
