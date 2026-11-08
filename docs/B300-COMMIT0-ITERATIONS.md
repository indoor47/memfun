# B300 Commit0 iterations

One section per pod. Goal: ONE Commit0 library passes with good quality on a B300 with
Qwen3.8-27B (FP8, SGLang), by iterating on that one exercise. Numbers are the hidden-grader
test counts of `evals/swarm/run_smoke.py --suite commit0`; the visible tests of a task are the
same files (checked byte for byte), so "visible" and "hidden" agree except for a few episodes
where the agent edited other files.

## Cerebras API (2026-11-08): `qwen-3.8-27b`, no GPU pod

Run from the Mac through `evals/swarm/cerebras_proxy.py` (`run_smoke.py --backend cerebras`): the
proxy holds the key, queues requests under the live limits (450 RPM, 150K tokens/min), retries
429/5xx transparently, and accounts tokens and dollars per episode ($0.99/M in, $1.49/M out).
Total spend for everything below: **$1.52** (cap $8). Cloudflare answers 403 (error 1010) to
Python's default User-Agent: the proxy sends its own.

| Run | Setup | Result | Cost |
|---|---|---|---|
| Probe A | `wcswidth` x3, thinking on, max_tokens 40,960 | 0/3: reasoning ate the whole budget (about 90K reasoning tokens, 5 truncations each), no edit | $0.52 |
| Run B | 18 tasks, thinking off, max_tokens 16,384, 6 parallel | 9 pass, 3 partial (`wcswidth` 18 to 20 of 23), 6 fail | $0.74 |
| Re-runs of the 6 failures | `wcwidth`, `_bisearch`, parallel 3 | 3 of 12 episodes pass, same failures | $0.27 |

Speed: the 12 tasks that fit finished in 0.3 to 1.4 minutes each (about 1,600 tokens/s), the whole
Run B in about 16 minutes against hours on the B300; cost about $0.04 per passing task.

**The 6 failures are not rate limits.** The proxy absorbed every 429 (31 retries, 0 give-ups).
They are `HTTP 400 context_length_exceeded`: the context-first solver dumps the whole project
(`max_context_bytes` 200 KB) and for `wcwidth` and `_bisearch` that is about 170K tokens (the
Unicode tables tokenize badly) against Cerebras' 131,072-token window. The B300 server had a
larger window. A fix belongs in the solver: a context cap derived from the model's window.

Probe A cannot be compared with the B300 thinking probe (20/23, 22/23): Cerebras caps output at
40,960 tokens and wcswidth's reasoning does not fit.

## Pod (2026-11-07, 09:06 to 09:46 CET): thinking-on probe on `wcswidth`

**Setup.** SGLang with `--reasoning-parser qwen3` and no `enable_thinking=false` default, so
Qwen thinks and the reasoning arrives as `reasoning_content`, apart from `content`. Probe A:
3 trials of the `wcswidth` function task (the one function Qwen could not do without thinking),
`--max-agents 1`, visible-test verifier on, temperature 0.6, `max_tokens` 32768, 30 minute cap,
constrained JSON off. Main a3de91a plus reasoning-token counting (branch `feat/thinking-support`).

**Checked before the run (fake server, both DSPy engines, streamed and not).** The parsed
answer stays clean and `reasoning_content` is kept as a separate field; the server's
`completion_tokens_details.reasoning_tokens` survives on both engines and is now counted per
episode; the loop guard does not abort 100,000 characters of varied reasoning and does abort a
reasoning loop; reasoning that uses the whole `max_tokens` budget is a visible truncation.

**Result: no score.** At 09:34 (28 minutes in) no episode had finished and no file had been
edited (all three trials: 23 of 23 visible tests failing). What the calls looked like
(prompt tokens, completion tokens):

| trial | solve call 1 | solve call 2 (JSON-mode retry) | after the no-edit retry |
|---|---|---|---|
| 1 | 12,903 / 32,768 truncated | 12,892 / 32,768 truncated | planner calls again |
| 2 | 15,962 / 32,768 truncated | 15,951 / 32,768 truncated | planner calls again |
| 3 | 30,752 / 28,745 answered | 30,741 / 17,210 answered | 17,291 / 32,768 truncated |

**Findings.**
1. **Thinking is expensive on this function:** 17,000 to 33,000 tokens of reasoning for one
   solve call, against about 1,500 tokens for the same call without thinking. At about 52
   tokens/s per stream that is 6 to 11 minutes per call, and a call that fails is asked again
   (DSPy's JSON-mode retry, then the single-task retry), so one episode needs 30+ minutes.
2. **32,768 `max_tokens` is too small with thinking on:** 5 of the 8 solve calls used the whole
   budget on reasoning and produced no answer. Every one of them was caught (`LLM RESPONSE
   TRUNCATED`) and failed visibly; none was a quiet empty success.
3. **Trial 3 did get two complete answers** (28,745 and 17,210 tokens) yet nothing was edited;
   whether those answers were wrong or unparseable was not examined before the pod stopped.
4. Probe B (all 18 tasks with thinking) was not run: 18 episodes of 10-minute calls cannot finish
   in the pod's time.

**To try this properly:** `max_tokens` of about 65,000 to 98,000 (the context is 262,144 and the
prompt is 13,000 to 31,000 tokens), a pod of at least 90 minutes, `wcswidth` only with 3 trials,
thinking off for the small planning calls, and a smaller solve prompt (only the relevant files).
Cost of this pod was a few dollars for no result; the experiment answers "does the model fit the
budget", not "does thinking solve `wcswidth`".

## Pod `hsz253jmn1dgrr` (2026-11-07, serving from about 05:42 until 06:16 CET): wcwidth as 6 function tasks

**Setup.** `wcwidth` split into 6 function-level tasks (`commit0_adapter.py split`): the rest of
the library is the reference implementation, one function is `pass`, graded by the tests that
execute it. 3 trials each = 18 episodes in parallel, no planner (`--no-decompose`), temperature
0.7, repetition penalty 1.05, loop guard and constrained JSON on, `max_tokens` 32768, 25 minute
cap. Main was e8bff5e. The pod rate was $8.99/h, $6.88 for the whole pod.

| function | tests | c1: 1 agent | c2: `--max-agents 3` | c3: visible-test verifier | best of 3 (c1) | best of 9 |
|---|---:|---|---|---|---:|---:|
| `_bisearch` | 21 | 21 / 21 / 21 | 21 / 21 / 21 | 21 / 21 / 21 | 21/21 | 21/21 |
| `_wcversion_value` | 9 | 9 / 9 / 9 | 9 / 9 / 9 | 9 / 9 / 9 | 9/9 | 9/9 |
| `list_versions` | 13 | 10 / 13 / 13 | 13 / 13 / 0* | 8 / 13 / 0* | 13/13 | 13/13 |
| `_wcmatch_version` | 12 | 0* / 8 / 12 | 0 / 12 / 12 | 12 / 12 / 12 | 12/12 | 12/12 |
| `wcwidth` | 22 | 22 / 0* / 21 | 22 / 22 / 0* | 22 / 0* / 22 | 22/22 | 22/22 |
| `wcswidth` | 23 | 19 / 19 / 19 | 19 / 19 / 19 | 19 / 16 / 19 | 19/23 | 19/23 |

Hidden tests passed per trial (`*` = the episode changed no file and failed closed).

| cycle | tests passed | episodes fully passing |
|---|---:|---:|
| c1 | 246/300 (82%) | 10/18 |
| c2 | 241/300 (80%) | 12/18 |
| c3 | 245/300 (82%) | 12/18 |

(The denominator is 3 x the tests kept per function = 300. c2's 12 full episodes against c1's 10
is noise, not a signal: both have the same 5 functions fully passing.)

### What the cycles were

| cycle | what changed | wall | approx. cost |
|---|---|---:|---:|
| c1 | baseline, `--max-agents 1` | 9.6 min (swarms finished in 3.7 to 7 min but did not exit; I ended them) | $1.44 |
| c2 | `--max-agents 3`, a reaper ends each swarm 8 s after its result | 3.0 min | $0.45 |
| c3 | visible-test verifier + one retry on a no-edit or unparseable first try (my branch, not merged) | 3.1 min | $0.46 |
| probe 1 | the 3 `wcswidth` trials through the iterative RLM loop (`--single-task-solver rlm`) | 2.3 min | $0.34 |
| probe 2 | same, with the full test command in the task text and a per-step retry | 9.8 min, ended by the pod termination | $1.46 |

Cost is wall time x $8.99/h of each run, about $4.15 of the pod's $6.88; the rest is boot, model
load, idle time and the analysis between runs.

### Findings

1. **Agents per task is not the lever.** With `--no-decompose` there is exactly one sub-task and
   one agent per episode (`is_single_task: true`); `--max-agents` only caps concurrency. c2 is c1
   with different sampling: 241 vs 246 of 300, 12 vs 10 full episodes, the same failure types.
2. **Why every c1 episode shows 9.0 min wall.** An artifact: `memfun swarm` prints its result and
   then never exits (0% CPU), the runner waits for the process, and I ended all 18 at once. The
   swarms' own `wall_seconds` are 220 to 420 s. A reaper fixed this for c2 and c3 (episodes took
   0.7 to 2.7 min). The swarm not exiting is an unfixed bug.
3. **Agents edit once and never run a test (c1, c2).** Every episode: one edit, zero commands. The
   single-task solver's verify-and-fix loop only knows ruff, eslint, go vet and cargo check;
   wcwidth has no ruff config, so nothing was verified and the first answer was final. Typical
   misses a test run would show: `wcwidth` returned -1 for NUL (`test_null_width_0`);
   `_wcmatch_version` got the nearest-version rules wrong.
4. **`wcswidth` fails the same 4 tests in every trial of every cycle** (19/23, once 16/23), all in
   `tests/test_emojis.py`: `test_unicode_9_vs16` (a female sign + U+FE0F must measure 2 cells),
   `test_longer_emoji_zwj_sequence` (a 10-codepoint zero-width-joiner sequence with variation
   selectors must measure 2 cells) and the two `test_recommended_*` tests that check every
   sequence unicode.org publishes. The agent writes the plain "sum the widths" function and
   ignores the emoji rules. The 4 tests ARE in the visible set and in the verify command: this
   is not a data-split problem.
5. **The verifier (c3) gave no gain.** 245/300, 12/18, the same as the baseline. What happened:
   the verify command ran where tests failed (LLM calls: 4 per episode when tests passed first
   time, 6 for each `wcswidth` trial = about 2 fix calls; the verify runs themselves are not
   logged, so this is inferred from the call counts), but the fix step is a single call
   whose shell operations (`ls`, `grep`, `sed`) run without the model seeing their output, and
   the loop stops as soon as the fixer proposes no operation (1 to 2 of the 4 rounds used).
6. **Three bugs found on the way, fixed on the branch:**
   - *Retries replayed DSPy's cache.* The retried solver call returned the identical reply
     (1,222 characters, four times) because the cache key was unchanged; a retry needs a fresh
     `rollout_id`.
   - *The task text hid the failing tests.* It listed the first 8 relevant test ids, which sort
     alphabetically and all come from `test_core.py`; the 4 emoji tests sat behind "and 15 more",
     so in probe 1 the agent ran only `test_core.py`. The text now gives the complete command.
   - *One unparseable model reply ended the whole iterative loop* (9 iterations lost in probe
     1); the loop now re-asks that step up to twice.
7. **Probes (iterative loop, which does see test output).** Probe 1: trials ended 19/23, 19/23
   and a failed run. Probe 2, after the fixes: about 40 loop steps per trial in 9 minutes and
   still 19/23 on the visible tests in all three when the pod was terminated. The loop works
   mechanically (it reads the code and the spec, runs the tests, edits); it does not find the
   emoji rules.
8. **Best of 3 by visible tests = 5 of 6 functions** (21/21, 9/9, 13/13, 12/12, 22/22; `wcswidth`
   19/23), identically in each cycle, and still 5 of 6 pooled over all 54 episodes. Selection
   cannot rescue `wcswidth`: its best trial in nine is 19/23. (Retries resample only at a
   temperature above 0: DSPy ignores `rollout_id` at 0.)
9. **Whole-library check** (`commit0_adapter.py merge-check wcwidth --run ...`): the best trial
   per function, ranked by visible tests, spliced into one stubbed wcwidth tree and run against
   the full suite: **33 of 37 pass**, for the best of each cycle and for the pooled best of nine.
   The only failures are the same 4 `wcswidth` emoji tests, so the pieces compose without
   breaking one another. The splice carries over the agent's new top-level imports, helpers and
   constants as well as the function (an audit found the first version dropped them); nested
   additions, such as a new method elsewhere in a class, are not carried over. `merge-check` runs
   each task's visible-test command from the operator's bundle cache with a shell: use it on your
   own runs only.

### Failure modes seen on this pod

- no-edit (failed closed, 2 per cycle): the solver reply could not be parsed into operations
  ("Truncation detected: N chars of operations but 0 parsed", the real token-truncation counter
  stayed 0), or its operations were only `ls`/`grep` commands. Under `--no-decompose` there is
  nothing to escalate to.
- wrong logic that a test run would catch (NUL width, nearest-version matching).
- the emoji rules of `wcswidth` (above).
- no truncations, loop aborts, stall stops or auditor-JSON rejections in c1 to c3 (there is no
  auditor on the single-task path).

### Next steps

1. **A stronger model for the hard function.** `wcswidth` is the only function the model cannot
   do, with and without test feedback; try a stronger model on just that function (and the
   no-edit cases) and keep Qwen for the rest.
2. **A second library to test generality.** `cachetools` (10 function tasks, mostly decorators
   and keys) or `tinydb`: if the same 5-of-6-style pattern shows up, the harness numbers are
   about the model and not about wcwidth.
3. **Merge the verifier branch** (`feat/visible-test-verifier`): visible-test verifier, complete
   test command in the task text, cache-safe retries, per-step retry, `--single-task-solver
   rlm`, `merge-check`. It did not raise the wcwidth number; it removes real bugs and makes the
   next experiment cleaner.
4. **Make `memfun swarm` exit after its result** instead of relying on a reaper.
5. **Feed command output back in the one-shot fix step**, or route function tasks through the
   iterative loop by default once a model that can use it is in place.

## Pod `9xyaj8g14utwgz` (2026-11-06, 21:50 to 22:52 CET): the first two runs on 6 libraries

Run 1 (21:50 to 22:35, 12 episodes: tinydb, parsel, pyjwt, imapclient, wcwidth, cachetools x 2):
aborted, 0 usable episodes. At temperature 0.0 the decomposer's reply degenerated into an
85,000 character repetition loop and stopped only at `max_tokens` (about 10 minutes per call);
the swarm fell back silently, ran 36 minutes and reported `success True` with no file created.
Run 2 (22:36 to 22:50, temperature 0.7): stopped after 14 minutes, no episode had edited a file;
the decomposer produced read-only "analysis" sub-tasks plus auditor rounds (20 auditor-JSON
rejections), 2 of 12 episodes still truncated (one a 176,166 character loop). Spend $11.78 at the
stop. Lessons that became PRs: repetition penalty and an early loop abort (#68), truncation
that fails visibly (#67), "no edits is failure" (#69), constrained JSON (#70, #72), function-level
tasks with no planner and an incremental progress file (#71).
