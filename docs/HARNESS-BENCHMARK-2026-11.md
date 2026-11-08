# Harness benchmark: does memfun's scaffolding add quality over other coding agents?

Run on 2026-11-08 (evening, CET). The same model served by SGLang on one B300 (Qwen3.8-27B FP8,
`qwen3.8-27b-fp8`, thinking off), the same tasks, the same hidden-test grader. Only the agent harness differs.

**Read the contamination section before quoting any number.** The agents ran on the Mac as the user with
write-only isolation. Reads were not restricted, and in one mini-swe-agent episode the agent read the reference
solution. The effect on the tallies is small (one episode) but the run was not clean, so a read-isolated rerun
is proposed below.

## Result in one paragraph

On 15 function-level tasks (3 trials each, 45 episodes per arm), counting an episode as solved when every
hidden test passes: **mini-swe-agent 43/45 (42/45 without the one contaminated episode), memfun 39/45,
aider 35/45, a plain single call 28/45.** mini-swe-agent solved the most and memfun came second, so memfun did
not beat the strongest alternative here. memfun was 4 episodes ahead of aider and 11 ahead of the single
call, and it used about one fourteenth of mini-swe-agent's input tokens and a third of its wall time. The gap to
mini is concentrated in one function, `wcswidth`, where mini iterates for ~105 commands on average and memfun makes
one edit and a bounded fix loop (up to 4 rounds).

## Two ways to count a pass

| | single-shot | memfun | aider | mini-swe-agent |
|---|---|---|---|---|
| **grader `all_pass`** (every hidden test passes) | 28 | 39 | 35 | 43 |
| **runner outcome `pass`** (also requires a changed file) | 26 | 38 | 32 | 40 |

The runner fails closed: an episode that changed no file is a failure even when the hidden tests pass. That only
matters for `cachetools methodkey`: its visible and hidden tests already pass on the untouched stub, so an agent that
correctly reports "nothing to do" gets `all_pass` but no changed file. The research session's tallies use
`all_pass`; the first version of this document used the outcome. Both are shown. `methodkey` is a flawed task (see
caveats) and should be dropped from a rerun.

## Setup

| | |
|---|---|
| Model | Qwen3.8-27B FP8 on one B300 via SGLang, reached through an SSH tunnel (127.0.0.1:18000), thinking off (server default) |
| Tasks | 15 function-level tasks: wcwidth (6: `_bisearch`, `_wcmatch_version`, `_wcversion_value`, `list_versions`, `wcswidth`, `wcwidth`) and cachetools (9: `fifo_cache`, `lfu_cache`, `lru_cache`, `mru_cache`, `rr_cache`, `ttl_cache`, `hashkey`, `methodkey`, `typedkey`; `typedmethodkey` left out, only 4 tests) |
| Trials | 3 per task, all 45 episodes of an arm at once; 25 min cap per episode |
| Task text | byte-identical for every arm (`task.txt`), including the visible test command |
| Agent's view | the episode repo with the visible tests. The hidden grader copy was NOT hidden from reads (see contamination) |
| Sampling | temperature 0.7, repetition_penalty 1.05 sent by every arm, top_p at the server default, `max_tokens` 32768 |
| Grading | `run_smoke.py`'s grader for every arm: hidden tests, symlink-safe; `merge-check` per arm |
| Limits | 25 min wall per task, one agent per task |

Arms and their exact setup:

- **single-shot**: one chat call: task text, target file(s) and the visible test files in, the whole new file out,
  written to disk. No tools, no iteration. A lower bound.
- **memfun**: this repo at `feat/harness-arms` (main plus #77), `memfun swarm --no-decompose`, `--max-pool 1`,
  verifier on (the visible test command runs after the edit and failures go back to the model),
  `[llm] context_window = 262144`. Per episode: about 2 tool operations (one edit) plus the bounded verify-and-fix loop (`max_fix_attempts = 4`).
- **mini-swe-agent**: version 2.4.6, `mini_textbased.yaml` with `litellm_textbased` (text-based bash actions, so
  no server-side tool parser is needed), `agent.mode=yolo`, `cost_limit=0`, no step limit (the 25 min wall is the limit).
- **aider**: aider-chat 0.86.2, `--edit-format diff`, `--yes-always`, `--no-auto-commits`, `--auto-test` with the
  visible test command (the way Commit0's own aider agent iterates), the target file as the editable file, aider's
  default of 3 reflections. aider has no shell tool.
- **claude-code**: not run (it needs an Anthropic-Messages endpoint in front of SGLang; not smoked in time).

## Per arm

| arm | all hidden tests pass (episodes) | runner `pass` | partial | failed | no file change | wall | input / output tokens | hidden tests passed | merge-check wcwidth / cachetools | pod time cost |
|---|---|---|---|---|---|---|---|---|---|---|
| single-shot | 28/45 (62%) | 26 | 10 | 9 | 7 (6 unparseable replies) | 74 s | 296k / 72k | 569/723 (79%) | 35/37, 208/215 | ~$0.2 |
| aider | 35/45 (78%) | 32 | 2 | 11 | 7 + 4 over the context window | 320 s | ~2.33M / 45k | 583/723 (81%) | 35/37, 210/215 | ~$0.8 |
| memfun | 39/45 (87%) | 38 | 3 | 4 | 2 | 338 s | 1.24M / 213k | 671/717 (94%) | 33/37, 210/215 | ~$0.8 |
| mini-swe-agent | 43/45 (96%) | 40 | 1 | 4 | 4 | 1,020 s | 17.6M / 175k | 715/723 (99%) | 37/37, 210/215 | ~$2.6 |

Notes: token numbers are the harnesses' own counts (single-shot, aider, mini) or the server counters (memfun); aider's
are rounded from its log lines. The pod cost is wall time times the pod's hourly rate (the arms ran one after
another), so it ignores grading and idle time. memfun's hidden-test total is 717 rather than 723 because in
`mru_cache` trial 1 its edit broke the module import and only 1 of that task's 7 hidden tests was collected.
mini's numbers include the one contaminated episode (`wcswidth` trial 1).

## Per function (episodes where every hidden test passes, out of 3)

| function | single-shot | memfun | aider | mini |
|---|---|---|---|---|
| wcwidth `_bisearch` | 2 | 3 | 2 | 3 |
| wcwidth `_wcmatch_version` | 1 | 3 | 1 | 3 |
| wcwidth `_wcversion_value` | 2 | 3 | 2 | 3 |
| wcwidth `list_versions` | 2 | 2 | 1 | 3 |
| wcwidth `wcswidth` | 0 | 0 | 0 | **3** (2 without the contaminated trial) |
| wcwidth `wcwidth` | 2 | 3 | 2 | 2 |
| cachetools `fifo_cache` | 3 | 3 | 3 | 3 |
| cachetools `lfu_cache` | 2 | 3 | 3 | 3 |
| cachetools `lru_cache` | 3 | 3 | 3 | 3 |
| cachetools `mru_cache` | 1 | 1 | 3 | 2 |
| cachetools `rr_cache` | 3 | 3 | 3 | 3 |
| cachetools `ttl_cache` | 3 | 3 | 3 | 3 |
| cachetools `hashkey` | 0 | 3 | 3 | 3 |
| cachetools `methodkey` | 1 | 2 | 0 | 0 |
| cachetools `typedkey` | 1 | 3 | 3 | 3 |

(This table uses the runner outcome, which is why `methodkey` is low for the arms that correctly did nothing.
With `all_pass`, `methodkey` is 3/3 for every arm, including the single call: the task cannot tell the arms apart.)

## Contamination audit

Why it was needed: the agents (mini-swe-agent, aider, memfun) ran on the Mac as the user. The sandbox blocked writes
outside the output directory but not reads. The Mac holds the answers: the bundle cache with each task's reference
solution (`bundles/*/solution/`) and the hidden grader tests (`bundles/*/hidden/`), the clones of the original
libraries (`~/.cache/memfun-commit0/clones`), pip caches and this repository.

Method (`evals/swarm/audit_contamination.py`, re-runnable on any run directory): for every episode of every arm, take
the commands the agent ran (mini: its trajectory file; memfun: the recorded tool operations; aider: its session log;
single-shot has no tools) and check (a) whether a command reached outside the episode repo, (b) whether a command
touched the reference-solution, hidden-test, bundle-cache or clones-cache paths, and (c) whether any command output
newly contained the reference implementation of the target function (at least 30% and at least 2 of its distinctive lines,
not already written by the agent or shown earlier). The detector was checked with a positive control, and I fixed two
bugs in it while auditing (a wrong bundle lookup that silently disabled the leak check, and false positives from the
agent re-displaying its own edit).

| arm | episodes | reached outside the repo | touched reference / hidden / clones paths | reference code appeared in an output |
|---|---|---|---|---|
| single-shot | 45 | 0 (no tools) | 0 | 0 |
| memfun | 45 | 0 (its operations are reads and edits of project files) | 0 | 0 |
| aider | 45 | 0 (one log line names a path from the library's own docs on another machine; harmless) | 0 | 0 (no shell tool) |
| **mini-swe-agent** | 45 | **23** | **4 episodes** | **1 confirmed, 1 partly explained, see below** |

What mini did outside the repo (episodes, a command can be in several categories): path-style probes outside the repo
19, listing the home directory 13, `git show/log/...` 9, `find /` or `find ~` 14, the cache directories 6,
`inspect.getsource` / `__file__` 5, `pip show` 4, `site-packages` 4. No `pip install`, `pip download`, `git clone`,
`curl` or `wget` was run, and no command read a `hidden/` directory.

Findings, all in mini-swe-agent:

1. **Confirmed contamination: `wcwidth-wcswidth` trial 1 (counted as solved).** The agent was stuck at 3 failing tests
   from step 118 to 127. At step 128 it searched for the reference (`find ... -name wcwidth.py -path "*solution*"`),
   at step 129 it printed the reference function from the bundle cache (`grep -A 60 "def wcswidth" .../bundles/c0s-wcwidth-wcswidth/solution/wcwidth/wcwidth.py`,
   all 19 distinctive lines of the function appear), and at step 130 the tests passed (23 passed). This episode
   should not be counted. The other two `wcswidth` trials (2 and 3) show no read of outside code: they listed the
   home directory and ran `pip show` / `inspect.getsource` on the episode's own package, so they stand.
2. **`wcwidth-wcwidth` trial 2 (partial):** the agent found the bundle cache and worked inside the bundle's own `repo/`
   copy (including `sed -i` there) instead of its episode repo. No reference code appeared in the outputs. It did not
   solve the task. The modified file is in a scratchpad copy of the bundles (`scratchpad/c0/bundles`, which also gained
   coverage artifacts). The cache the benchmark actually used, `~/.cache/memfun-commit0/bundles`, is a separate copy
   made earlier; I diffed the two and the benchmark's copy is unmodified, so grading was not affected. The scratchpad copy
   should not be reused.
3. **`cachetools-mru_cache` trial 1 (failed):** the agent read `/private/tmp/ep/repo/src/cachetools/func.py`, a leftover
   episode repo from another run, which contained a finished `func.py`. It still failed.
4. Three more episodes (`cachetools-hashkey` trial 3, `wcwidth-_bisearch` trial 1, `wcwidth-wcswidth` trial 1)
   looked for or listed the commit0 cache and clones directories. The lookups written with `~` failed (HOME was the
   episode's temporary home); one with an absolute path (`ls /Users/dmytro/.cache/memfun-commit0/clones/wcwidth`)
   worked and returned file names. No later command read a clones file.
5. The probing itself is the signal: in 23 of 45 episodes mini went looking for the library or the task elsewhere on the machine.

Conclusion: the confirmed leak changes mini's tally by one episode (43 to 42 by `all_pass`, 40 to 39 by outcome, and `wcswidth`
3 to 2 trials). No evidence that memfun, aider or single-shot saw reference code, though for memfun and aider the audit
only sees their tool calls and log, not whatever the model remembers from training. The more important point is that
the sandbox did not guarantee a clean test, and an agent that is willing to search the disk will find the answers.

## Why mini-swe-agent wins: it iterates for far longer

| | commands per episode |
|---|---|
| mini-swe-agent | median 9, mean 19.6, 90th percentile 61, max 132 |
| mini on the passing episodes / partial / failed (means) | 20 / 26 / 14 |
| mini on `wcswidth` (mean) | 105 |
| mini on `_wcmatch_version` (mean) | 46 |
| memfun | mean 2.2 tool operations (one edit, a bounded fix loop of at most 4 rounds) |
| aider | one edit plus up to 3 auto-test reflections |
| single-shot | one call |

The hard functions are won by trying, running the code and reading real output many times. On `wcswidth` (the
function every other arm never solves) mini averages about 105 commands per trial, and on `_wcmatch_version` 46,
against memfun's single edit and at most four fix rounds. That is also why mini costs 14 times memfun's input tokens
(17.6M against 1.24M): every step resends the growing conversation. The lesson is that on tasks like these, quality
tracks the number of run-and-observe cycles the harness allows; memfun's design (one planned edit, a bounded verify
loop) is cheaper and faster, and on easy functions it is just as good (13 of the 15 functions are within one trial
of mini; the exceptions are `wcswidth`, where mini is 3 trials ahead, and `methodkey`, an artifact), but it stops
where mini keeps going. A fair next experiment is memfun with a larger fix-loop budget, or the RLM
solver (`--single-task-solver rlm`), which is iterative, against mini on the same tasks.

## Where memfun loses and wins

- **`wcswidth`:** memfun, aider and the single call fail all three trials; mini solves 2 clean trials of 3 (3 with the
  contaminated one).
- **`list_versions`, `mru_cache`, `wcwidth`:** mini is ahead by one trial (memfun at 2, 1, 3 against mini's 3, 2, 2:
  on `wcwidth` memfun is ahead).
- **Overall:** mini-swe-agent is 3 to 4 episodes ahead of memfun and 44 hidden tests ahead (715 vs 671, with memfun's
  6-test denominator effect from the broken `mru_cache` import).
- **Where memfun wins:** against the single call (+11 episodes) and aider (+4, and aider lost four episodes to prompts
  over the context window and seven to ending without an applied edit), on cost and time (1/14 of mini's input tokens,
  a third of its wall time for 91% of its solved episodes), and on `methodkey` (2 vs 0 by outcome, but see below).

## Proposal: read-isolation for a fair rerun

The rule: the agent can read its episode and its own tools, nothing else; the library under test is not installed anywhere
it can import from; no network except the model.

Implemented and tested in this PR (`evals/swarm/sandbox/`, used by `run_harness_benchmark.sh`):

- `episode.sb`, a macOS `sandbox-exec` profile run per episode through `run_in_episode_sandbox.sh`: it **denies reads of
  everything under `/Users`, `/private/tmp`, `/tmp` and `/Volumes`** except the episode repo, the episode's throwaway
  HOME and up to six explicitly allowed tool paths (the agent's own venv and interpreter prefix); writes only to the
  episode; **network denied except loopback** (the model tunnel). No bundle cache, no clones, no other episodes, no pip
  cache, no repo.
- A self-test at the start of the script refuses to run unless every line passes: bundle cache unreadable, commit0 cache
  and clones unreadable, memfun repo unreadable, other episodes unreadable, writes outside the episode denied, internet
  unreachable, the episode repo usable, and no installed copy of `wcwidth` or `cachetools` for the agent's interpreter.
- A contamination audit runs after every arm and writes `audit.json`.
- Verified locally: reference solution, hidden tests, clones, `~/Ventures` and other episodes all fail with "Operation
  not permitted"; `pip download` fails; `import wcwidth` and `import cachetools` fail outside the episode; pytest runs in
  the episode with the allowed interpreter; mini, aider and `memfun swarm --help` start under the profile. A trap found
  while testing: allowing `(local ip "localhost:*")` re-opens the internet, so only outbound loopback is allowed.

Stronger options if macOS profiles are not trusted: run each episode in a container (Docker or Apple's container
tooling) with only the episode repo mounted and a network limited to the model host; or run the agents under a separate
macOS user whose home and `/private/tmp` are unreadable. Either also removes the dependence on filesystem layout. And
independent of the sandbox: keep the bundle cache and clones on a path the agent's user cannot read, drop `methodkey`
(its tests pass untouched, so it cannot distinguish arms), and write the leak audit's verdict into each arm's table.

## Caveats

- Our function split is not official Commit0, and 15 tasks x 3 trials is small: a gap of 3 or 4 episodes is within
  trial noise. Read the result as "mini-swe-agent and memfun are close; both clearly ahead of the single call; aider in between",
  not as a strict ranking.
- The arms' prompts can only be equalised so far: each harness adds its own system prompt and templates around the
  identical task text. aider got its default `diff` format (a `whole` run was not made) and a text-based mini config.
- One run per arm; no repeated-run variance. The sampling was not greedy.
- `cachetools methodkey` is not a discriminating task: its tests pass on the untouched stub.
- The agentic arms ran on the Mac with bash against the model on the B300, with the isolation flaw described above.
- claude-code was not run.
- The model may remember these public libraries from training; nothing here separates recall from reasoning.
