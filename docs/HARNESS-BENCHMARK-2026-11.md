# Harness benchmark: does memfun's scaffolding add quality over other coding agents?

Run on 2026-11-08 (evening, CET). The same model served by SGLang on one B300 (Qwen3.8-27B FP8,
`qwen3.8-27b-fp8`, thinking off), the same tasks, the same hidden-test grader. Only the agent harness differs.

**A clean, read-isolated rerun with the new memfun loop is in the last section ("Rerun on 2026-11-09").
Use that one for current numbers; the first run below is kept as the record of what went wrong.**

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

## Rerun on 2026-11-09: read-isolated, new memfun loop

One B300 pod (Qwen3.8-27B FP8, SGLang), the same grader, 14 tasks instead of 15 (cachetools `methodkey`
dropped: its tests pass on the untouched stub), 3 trials each, so 42 episodes per arm. Every agent ran under the
read-isolation profile (`evals/swarm/sandbox/`): it could read only its own episode, no reference solutions,
hidden tests, clones or other episodes, and had no network except the model. The contamination audit ran after each
arm. memfun was the version with the new fix loop (PR #81). Thinking was off, sent with every request (see "What went
wrong during the run" for why it is not a server default). The arms ran one after another.

### Per arm

The grader-side tallies were cross-checked independently (the research session's aggregation and mine agree on all
four arms: memfun 683/693 hidden tests and 37/42 episodes, mini-swe-agent 654/693 and 38/42, single-shot 610/671 and
31/42, aider 514/693 and 29/42).

| arm | all hidden tests pass (of 42) | same 14 tasks, first run | hidden tests passed | input / output tokens | wall | failures | contamination audit |
|---|---|---|---|---|---|---|---|
| single-shot | 31 (74%) | 25 | 610/671 | 0.29M / 0.08M | 955 s | 9 partial, 2 failed | no tools: none |
| memfun | 37 (88%) | 36 | 683/693 | 1.39M / 0.25M | 593 s | 5 partial | no outside reads, no leaks |
| aider | 29 (69%) | 32 | 514/693 | 5.83M / 0.06M | 954 s | 3 partial, 10 failed (7 no file change, 3 context length) | no outside reads, no leaks |
| mini-swe-agent | 38 (90%) | 40 | 654/693 | 30.7M / 0.21M | 1,505 s (episode cap) | 3 partial, 1 failed (no file change) | no reference, hidden-test, clones or bundle paths touched; no leaks |

Token numbers are the server's counters for each arm's run window (the arms ran sequentially; a few test
requests of mine are included). The harnesses' own counts are lower where episodes were killed (mini, at its
25-minute cap, loses the usage of unfinished episodes). "Wall" for single-shot and aider includes one episode whose code
hung the grader until its timeout (900 s for single-shot, 240 s for the others); the model calls themselves took
under 90 s. Whole-library merge-check (best trial per function, full test suite): memfun 36/37 wcwidth, 207/215
cachetools; mini 37/37 and 207/215; aider 1/37 (its best-trial merge breaks the import) and 208/215.

### Per function (episodes fully passing, out of 3)

| function | single-shot | aider | memfun | mini-swe-agent |
|---|---|---|---|---|
| wcwidth `_bisearch` | 2 | 1 | 3 | 3 |
| wcwidth `_wcmatch_version` | 2 | 0 | 3 | 2 |
| wcwidth `_wcversion_value` | 3 | 3 | 3 | 3 |
| wcwidth `list_versions` | 2 | 1 | 3 | 2 |
| wcwidth `wcswidth` | 0 | 0 | 0 | 1 |
| wcwidth `wcwidth` | 2 | 1 | 3 | 3 |
| cachetools `fifo_cache` | 3 | 2 | 3 | 3 |
| cachetools `lfu_cache` | 3 | 3 | 3 | 3 |
| cachetools `lru_cache` | 3 | 3 | 3 | 3 |
| cachetools `mru_cache` | 2 | 3 | 2 | 3 |
| cachetools `rr_cache` | 3 | 3 | 3 | 3 |
| cachetools `ttl_cache` | 3 | 3 | 3 | 3 |
| cachetools `hashkey` | 2 | 3 | 2 | 3 |
| cachetools `typedkey` | 1 | 3 | 3 | 3 |

### What it says

- **memfun and mini-swe-agent are level on pass rate here (37 and 38 of 42).** One episode apart is well inside
  the noise of a 42-episode arm. memfun got there with about a twentieth of mini's input tokens (1.4M against 30.7M)
  and, with no episode running to a cap, in less than half the wall time.
- **The gain from the new loop is small on the B300.** On the same 14 tasks memfun went from 36 to 37 fully passing
  episodes. It gained on `mru_cache` and `list_versions` and lost one on `hashkey`. The large gain seen on Cerebras
  (10 to 17 of 20 on the five weakest functions) does not show up here, because the old loop already solved most of
  these tasks on the B300. Read it as "no regression and a possible small improvement", not as a measured win.
- **`wcswidth` is still the unsolved function.** memfun 0 of 3 (19 to 22 of 23 tests), mini 1 of 3, the others 0.
  In the first run mini solved it 3 of 3, but one of those trials had read the reference solution; this clean rerun
  gives 1 of 3. With 3 trials the two results cannot be told apart, so the first run's "mini wins on wcswidth" is not
  confirmed.
- **The isolation worked.** mini looked around outside its episode in 28 of 42 episodes (directory listings, `find`,
  `pip show`), as in the first run, but touched none of the answer locations and no reference code appeared in any
  output. By kind (episodes): path-style probes outside the repo 22, directory listings of the home directory 16,
  `git show/log/...` 15, `find /` or `find ~` 13, `inspect.getsource`/`__file__` 10, `pip show` 9, cache directories 4,
  site-packages 4; up to 207 commands in one episode. The audit (`evals/swarm/audit_contamination.py`) was run on all
  four arms of this run in one pass: 0 function leaks, 0 original-file exposures and 0 touches of the reference,
  hidden-test, bundle or clones paths, for every arm. That is the difference from the first run, where one episode read the reference solution.
- **aider was weaker than in the first run (29 against 32), and the sandbox is not the cause.** I compared the
  two runs' aider logs. The failure mix is the same: context length exceeded 3 then 4, a reply with no edit 5 then 7,
  an edit that did not fix the tests 2 then 2. No aider log in this run contains a permission error ("Operation not
  permitted"). The "no edit" episodes are the model answering with pseudo tool calls (a made-up `<tool_call>` /
  `Read` / `Bash` syntax) instead of the SEARCH/REPLACE blocks aider needs, so aider ends the turn with nothing to
  apply; that behaviour is in both runs. A three-episode difference on 42 is within the noise, and the one change to
  aider (the task now arrives with `--message` instead of `--message-file`) delivers the same text. The one real
  sandbox side effect was the first attempt, below.

### What went wrong during the run (process, so the numbers are read correctly)

1. **Thinking was on at first.** The pod was launched without the SGLang flag the earlier runs used
   (`--default-chat-template-kwargs '{"enable_thinking":false}'`), so the first single-shot attempt ran with the model
   thinking out loud and was discarded (9 episodes ran to the output limit). The pod was not restarted (B300 stock was
   scarce); a small local proxy (`evals/swarm/thinking_off_proxy.py`) adds `enable_thinking=false` to every request, and the
   whole benchmark was rerun through it. The first attempt's partial results are not used. The launch kit's docs now state
   the flag.
2. **aider failed its first attempt.** The runner gave aider its task as a file outside the episode, which the
   read-isolation blocks, so aider got an empty task and did nothing in all 42 episodes (13 s). The task is now passed
   on aider's command line (`--message`), and aider was rerun alone after mini. memfun, mini and single-shot were not
   affected.
3. **One hanging episode can stall an arm until the grader timeout.** The grader timeout was lowered from 900 s to 240 s
   for memfun, mini and aider (single-shot ran with 900 s). A hanging candidate still counts as a failure.
4. A first memfun start was stopped after a minute and restarted when the timeout change was made.

### Cost

About $13.2 for the B300 pod (88 minutes at $8.99/h, of which roughly 8 minutes boot and download and about 10
minutes lost to the thinking-on attempt), taken from the account balance before and after. The earlier Cerebras
experiments that led to the new loop cost about $7.8.

### Caveats

- One run per arm, 14 tasks, 3 trials: differences of a few episodes are noise. The ranking memfun 37, mini 38,
  single-shot 31, aider 29 should be read as "memfun and mini close together, aider and single-shot behind".
- The rerun differs from the first run in more than memfun's loop: read-isolation, thinking switched off per request
  rather than as a server default, the lower grader timeout, and 14 instead of 15 tasks.
- The model may remember these public libraries from training.
- The agents ran on the Mac against the B300 through a tunnel, under a macOS sandbox profile; a stronger setup
  would be per-episode containers.
- claude-code was not run.


## Hard functions on the H200 (2026-11-09)

The functions where some arm missed a full pass in the isolated rerun above, run again on a free H200 (Qwen3.8-27B FP8,
thinking off, read isolation, one arm after another, all 18 episodes of an arm at once): six functions (`wcswidth`,
`_wcmatch_version`, `list_versions`, `wcwidth` in wcwidth; `mru_cache`, `hashkey` in cachetools) x 3 trials.

| arm | fully passing (of 18) | hidden tests | input / output tokens | wall |
|---|---|---|---|---|
| A: memfun as is (fix budget 4) | 14 | 447/459 | 0.71M / 0.15M | 481 s |
| C: memfun, fix budget 12 | 11 | 441/459 | 0.54M / 0.13M | 411 s |
| D: mini-swe-agent | **17** | **457/459** | 24.4M / 0.14M | 820 s |

| function | A | C | D (mini) |
|---|---|---|---|
| `hashkey` | 3 | 1 | 3 |
| `mru_cache` | 3 | 2 | 3 |
| `_wcmatch_version` | 2 | 2 | 3 |
| `list_versions` | 3 | 3 | 3 |
| `wcwidth` | 3 | 3 | 3 |
| `wcswidth` | **0** (18, 20, 20 of 23 tests) | **0** (20, 20, 21) | **2** (23, 23, 21) |

Whole-library merge-check (best trial per function): wcwidth 34/37 (A), 35/37 (C), 37/37 (D); cachetools 215/215 for all three.
Contamination audit: clean for all three arms (no leaks, no touches of the reference, hidden-test, bundle or clones paths).
mini looked around outside its episode in 14 of 18 episodes (5 with output) and found nothing; it ran 38 commands on average, and
112, 135 and 175 commands on `wcswidth`.

### What this says

- **A deeper fix loop does not help (arm C).** Raising the fix budget from 4 to 12 rounds gave 11 against 14 and did not lengthen the median
  episode (116 s in both arms). Most of the gap is `hashkey` (3 passes to 1), which is plausibly sampling noise since both arms
  share the same first attempt. More rounds of the same kind of fixing do not add information, so they do not unstick it.
- **mini wins mainly on `wcswidth`** (2 of 3, memfun 0 of 6 across A and C), at about 34 times memfun's input tokens, and in this run it is the only
  arm that solves `_wcmatch_version` every time (memfun got 2 of 3 on it in both arms). On the other four functions mini and memfun are level.
- Arm B (thinking on for the solver) was not run here; it stays for Cerebras.

### Why mini solves `wcswidth` and memfun does not

I read mini's three `wcswidth` trajectories (112, 135 and 175 commands) and memfun's code and failures. This is a reading of
three trajectories, not a controlled ablation.

1. **Both see the same failure and get stuck at the same place.** The remaining failures are the emoji tests (variation selector 16 and
   zero-width-joiner sequences). pytest prints them as one truncated line,
   `AssertionError: assert [{'expected_w...': '3\ufe0f'}, ...] == []`, which hides *which* sequences are wrong and why. memfun's fix step receives
   exactly this line plus the failing test's source. mini is stuck at "2 failed, 21 passed" for about 60 steps in its successful trial.
2. **mini looks; memfun reasons.** In every trial mini wrote 59 to 130 small Python experiments and 38 to 46 commands that read the unicode
   tables. The decisive habit: it loads the tests' own data files (`tests/emoji-zwj-sequences.txt`,
   `tests/emoji-variation-sequences.txt`; 11, 14 and 35 commands across the trials), runs its candidate over *every* sequence, and prints the
   mismatches. From the pattern it forms a hypothesis (one of them: the variation-selector bonus applies only when the base character starts the string, not
   when it sits mid-sequence; that one still left 2 failures and was dropped, and the next passed all the data), tests it against every sequence in seconds
   ("fails: 0"), and only then edits. In its successful trial this
   took from step 33 to about step 104; the test count went from 21 to 23 only at the end.
3. **memfun does not run experiments of its own.** Its first solve and its fix rounds are written from reading; essentially the only code run is the
   verify command. Its final code is an elaborate hand-built emoji-sequence state machine (skin-tone skipping, "the whole joiner sequence is width 2"),
   built from the model's general knowledge of emoji. The library's own description (`SPEC.txt`, which memfun could see) states a simpler rule,
   and the data files show which sequences disagree with the model's version. memfun does read the same table
   (`table_vs16.py`, 7 KB, fully visible to it), so this is not a context-size problem.
4. **Persistence helps, but it is not the whole story.** mini's trial 1 ran 135 commands and never got past 2 failures; the two trials that passed used the
   experiments to find the pattern. Arm C shows that more of memfun's kind of iteration (patch, rerun, patch) does not substitute for it.

### What to try next (not done)

- Show the fix step the **untruncated failure**: re-run the failing tests with full assertion output (`-vv`), or have the loop print the
  mismatching items itself, so the model sees which sequences fail, not a truncated list.
- A **differential probe**: when a round stalls, let the model run short experiments that load the failing test's data and print where its code
  and the expected values disagree. The earlier probe step (round 2 of the Cerebras experiments) fired rarely, was off by default, and was not told to
  compare against the test data, so that experiment does not rule this out.
- Keep fix-loop depth as it is; arm C shows it is not the lever.

### Caveats

One run per arm, 18 episodes per arm: a difference of one or two episodes is noise, so the solid findings are "mini
is ahead on `wcswidth`" and "a deeper fix loop does not help", not the exact counts. The H200 session also gave the
[concurrency ramp](CONCURRENCY-RAMP-2026-11.md).
\n