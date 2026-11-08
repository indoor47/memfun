# Harness benchmark: does memfun's scaffolding add quality over other coding agents?

Run on 2026-11-08 (UTC; evening in CET). The same model served by SGLang on one B300 (Qwen3.8-27B FP8,
`qwen3.8-27b-fp8`, thinking off), the same tasks, the same hidden-test grader. Only the agent harness differs.

## Result in one paragraph

On 15 function-level tasks (3 trials each, 45 episodes per arm) **mini-swe-agent solved the most (40 of 45
episodes, 89%)**, memfun was close behind (38, 84%), then aider (32, 71%) and a plain single call (26, 58%).
memfun matched mini-swe-agent's pass rate at about **one fourteenth of its input tokens and a third of its wall
time**, but it did not beat it. memfun lost on `methodkey` and `mru_cache` and on `list_versions` (2/3), where
mini-swe-agent was at 0/3 only for `methodkey`; everywhere else memfun was equal or better than aider and the
single call. No arm solved `wcswidth` except mini-swe-agent (3/3), the one function the other three never got.

## Setup

| | |
|---|---|
| Model | Qwen3.8-27B FP8 on one B300 via SGLang, reached through an SSH tunnel (127.0.0.1:18000), thinking off (server default) |
| Tasks | 15 function-level tasks: wcwidth (6: `_bisearch`, `_wcmatch_version`, `_wcversion_value`, `list_versions`, `wcswidth`, `wcwidth`) and cachetools (9: `fifo_cache`, `lfu_cache`, `lru_cache`, `mru_cache`, `rr_cache`, `ttl_cache`, `hashkey`, `methodkey`, `typedkey`; `typedmethodkey` left out, only 4 tests) |
| Trials | 3 per task, all 45 episodes of an arm at once; 25 min cap per episode |
| Task text | byte-identical for every arm (`task.txt`), including the visible test command |
| Agent's view | the episode repo with the visible tests; the hidden grader copy is never reachable |
| Sampling | temperature 0.7, repetition_penalty 1.05 where the harness could send it (single-shot: sent; memfun: sent; mini: sent via `extra_body`; aider: sent via `extra_params`), top_p left at the server default, `max_tokens` 32768 |
| Grading | `run_smoke.py`'s grader for every arm: hidden tests, symlink-safe, a run that changed no file is a failure; `merge-check` per arm |
| Limits | 25 min wall per task, one agent per task |

Arms and their exact setup:

- **single-shot**: one chat call: task text, target file(s) and the visible test files in, the whole new file out,
  written to disk. No tools, no iteration. A lower bound.
- **memfun**: this repo at the rebased `feat/harness-arms` (main plus #77), `memfun swarm --no-decompose`,
  `--max-pool 1`, verifier on (the visible test command runs after the edit and failures go back to the model),
  `[llm] context_window = 262144`.
- **mini-swe-agent**: version 2.4.6, `mini_textbased.yaml` with `litellm_textbased` (text-based bash actions, so
  no server-side tool parser is needed), `agent.mode=yolo`, `cost_limit=0`, no step limit (the 25 min wall is the limit).
- **aider**: aider-chat 0.86.2, `--edit-format diff`, `--yes-always`, `--no-auto-commits`, `--auto-test` with the
  visible test command (the way Commit0's own aider agent iterates), the target file as the editable file, aider's
  default of 3 reflections.
- **claude-code**: not run (it needs an Anthropic-Messages endpoint in front of SGLang; not smoked in time).

## Per arm

| arm | full-pass episodes | partial | failed | no-file-change | wall | input / output tokens | hidden tests passed | whole-library merge-check (wcwidth / cachetools) | pod time cost |
|---|---|---|---|---|---|---|---|---|---|
| single-shot | 26/45 (58%) | 10 | 9 | 7 (6 unparseable replies) | 74 s | 296k / 72k | 569/723 | 35/37, 208/215 | ~$0.2 |
| memfun | 38/45 (84%) | 3 | 4 | 2 | 338 s | 1.24M / 213k | 671/717 | 33/37, 210/215 | ~$0.8 |
| aider | 32/45 (71%) | 2 | 11 | 7 + 4 over the context window | 320 s | ~2.33M / 45k | 583/723 | 35/37, 210/215 | ~$0.8 |
| mini-swe-agent | 40/45 (89%) | 1 | 4 | 4 | 1,020 s | 17.6M / 175k | 715/723 | 37/37, 210/215 | ~$2.6 |

Notes: token numbers are the harnesses' own counts (single-shot, aider, mini) or the server counters (memfun);
aider's are rounded from its log lines. The pod cost is wall time times the pod's hourly rate, so it ignores grading
and idle time; the arms ran one after another. memfun's hidden-test denominator is 717 rather than 723 because one
episode's grade did not count 6 tests (see caveats).

## Per function (full-pass trials out of 3)

| function | single-shot | memfun | aider | mini |
|---|---|---|---|---|
| wcwidth `_bisearch` | 2 | 3 | 2 | 3 |
| wcwidth `_wcmatch_version` | 1 | 3 | 1 | 3 |
| wcwidth `_wcversion_value` | 2 | 3 | 2 | 3 |
| wcwidth `list_versions` | 2 | 2 | 1 | 3 |
| wcwidth `wcswidth` | 0 | 0 | 0 | **3** |
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

## Where memfun loses

- **`wcswidth`:** memfun, aider and the single call fail all three trials; mini-swe-agent solves all three. This is the
  function the B300 iterations doc (`docs/B300-COMMIT0-ITERATIONS.md`) found the model cannot write in one shot. mini-swe-agent
  gets there by running code and reading real output over many steps; memfun's one edit plus a verifier pass does not.
  That is the clearest quality gap and it costs mini-swe-agent about 14 times memfun's input tokens overall.
- **`mru_cache`:** aider 3, mini 2, memfun 1.
- **Overall:** mini-swe-agent is 2 episodes ahead (40 vs 38) and 44 hidden tests ahead (715 vs 671, with the
  denominator caveat); in the whole-library merge-check it is 37/37 on wcwidth against memfun's 33/37.

## Where memfun wins

- Against the single call (+12 episodes) and aider (+6): the verifier loop and the context handling matter.
  Aider lost four episodes to prompts over the context window and seven to ending with no applied edit.
- Cost and time: about 14x fewer input tokens and a third of the wall time of mini-swe-agent for 95% of its score.
- Fewer no-change failures than aider and single-shot (2 vs 7 vs 7).

## Caveats

- Our function split is not official Commit0, and 15 tasks x 3 trials is small: a 2-episode gap (38 vs 40) is inside
  the trial noise. Read it as "memfun and mini-swe-agent are level on pass rate", not as a ranking.
- The arms' prompts can only be equalised so far: each harness adds its own system prompt and templates around the
  identical task text. aider got its default `diff` format (a `whole` run was not made) and a text-based mini config.
- One run per arm; no repeated-run variance. The sampling was not greedy.
- memfun's hidden-test count is 717 not 723 for one episode; the grade for it is worth a look before quoting
  test-level numbers for memfun.
- The agentic arms ran on the Mac with bash against the model on the B300; sandboxed (writes denied outside the
  output directory) after a local mini run once wandered into the working repo.
- claude-code was not run.
