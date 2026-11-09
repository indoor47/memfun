# Concurrency ramp on one H200: how many agents can a GPU serve at once?

Run on 2026-11-09. One H200 SXM on RunPod (141 GB, $5.29/h), Qwen3.8-27B FP8 served by SGLang, thinking off
(the server flag, checked with a one-question test before the run started). The agents are memfun (current
`main`, the new fix loop), one agent per episode, each in its own read-isolated sandbox, driven from a Mac over an SSH
tunnel. The tasks are the 43 function tasks of **tinydb** (MIT licence), a Commit0-lite library split into
one task per function (50 stubs; 7 skipped because no test executes them; 201 tests). The stub fails the library's tests
and the reference passes all of them (every one of the 43 bundles validated).

## The ramp

Step k runs every function with k trials, all episodes at the same time (agents = 43 x k). A sampler read the
server's Prometheus metrics every 10 seconds throughout.

| step | agents | wall | passing episodes | running (median / peak) | queued (peak) | aggregate tok/s (median / peak) | per stream (median) | KV cache (peak) |
|---|---|---|---|---|---|---|---|---|
| 1 | 43 | 263 s | 43 of 43 | 12 / 42 | 0 | 634 / 2,894 | 64 tok/s | 69% |
| 2 | 86 | 383 s | 86 of 86 | 23 / 79 | 21 | 807 / **3,210** | 52 tok/s | 93% |
| 3 | 129 | 441 s | 127 of 129 | 46.5 / 80 | **56** | 1,110 / 2,495 | **33 tok/s** | 93% |

Over the whole ramp the server generated about 725k tokens and read about 7.0M prompt tokens. Mean aggregate
generation over each step's wall time was 495, 617 and 813 tok/s.

**Saturation is between 86 and 129 agents, and the edge is already visible at 86.**

- At 86 agents the KV cache peaks at 93%, a queue appears (21 waiting at the peak), per-stream speed has fallen from 64 to
  52 tok/s, and the peak aggregate throughput grew only 11% for twice the agents.
- At 129 agents the server holds about 80 requests running at once (the KV cache is full) and the rest wait: the queue peaks at 56 and its
  90th percentile stays at 54. Peak aggregate throughput falls to 2,495 tok/s (from 3,210), and per-stream speed halves to 33 tok/s.
- **The capacity answer for this model on one H200: about 80 concurrent streams and roughly 3.2k tok/s in total.**
  Beyond that, extra agents mostly wait.
- Agents spend much of their time outside the model (running tests, applying edits), which is why the median number of
  running requests (12, 23, 46.5) is far below the number of agents. About 2x to 3x more agents than the 80-stream limit can
  be kept busy before the queue dominates.
- Episodes finished per hour still rose with more agents (about 590, 810 and 1,050 per hour for steps 1, 2 and 3),
  because the queue only delays the model-bound part of an episode; each episode just takes longer.
- Quality did not change with load: 43 of 43, 86 of 86, 127 of 129. The two misses at 129 were one partial pass and one reply
  that was cut off before its edits (the runner labelled that one a harness error because of an unrelated LiteLLM warning; it is a
  model failure).

## Whole-library check (merge-check)

`merge-check` assembles the library from the best trial of each function (all three steps, 258 episodes) and runs
the library's full test suite once, to see whether independently written functions work together.

**Result: 197 of 201 tests pass.** The four failures are two tests in their json and memory variants,
`test_insert_with_doc_id` and `test_upsert_by_id`, so one or two functions handling document ids differently from the
library's intent.

The first attempt read "0 of 0": the whole-library starting point had a helper stripped out (`FrozenDict._immutable`, which no task covers
because no test executes it), so the merged library could not even be imported. `merge-check` now starts from the reference library
and replaces only the functions that have a task with the best trial's version, so untested functions stay as the reference's. This
changes how earlier wcwidth and cachetools merge-check numbers would read (functions without tasks would now count as the reference).

## Caveats

- One ramp, one model, one GPU, one library; the 129-agent step was the last because it saturated, so the exact knee
  between 86 and 129 agents is not located.
- The agents ran on a Mac (64 GB) through a tunnel; at 129 processes the Mac was not the bottleneck (it had 30 GB free), but
  this was not measured further.
- The saturation rule (a long queue, aggregate throughput growing under 10% for 1.5x the agents, or per-stream speed under half of
  step 1) is a pragmatic definition.
- The H200 cost about $5.4 for the session (from the account balance before and after).
- Tools: `evals/swarm/run_concurrency_ramp.sh`, `metrics_sampler.py`, `ramp_report.py`.
