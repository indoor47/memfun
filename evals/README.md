# Memfun Evaluation Suite

Evaluate Memfun against standard coding agent benchmarks.

## SWE-bench

[SWE-bench](https://www.swebench.com) is the industry-standard benchmark for
coding agents. It drops your agent into a real GitHub repository with a real
issue and checks whether the agent's patch makes the failing tests pass.

### Prerequisites

```bash
# 1. Docker (required for SWE-bench evaluation harness)
#    macOS: brew install --cask docker  OR  download Docker Desktop
#    Linux: https://docs.docker.com/engine/install/

# 2. SWE-bench
pip install swebench

# 3. HuggingFace datasets
pip install datasets
```

### Running the Evaluation

**Step 1: Generate predictions** -- Memfun solves each SWE-bench instance:

```bash
# Run on SWE-bench Lite (300 instances, recommended to start)
python evals/swebench_harness.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --output evals/predictions.jsonl \
    --max-instances 10           # start small to test
    --max-workers 4              # parallel instances

# Run on SWE-bench Verified (500 instances, the gold standard)
python evals/swebench_harness.py \
    --dataset princeton-nlp/SWE-bench_Verified \
    --output evals/predictions.jsonl
```

**Step 2: Evaluate predictions** -- SWE-bench runs tests in Docker:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path evals/predictions.jsonl \
    --max_workers 8 \
    --run_id memfun-v0.2.0
```

On macOS ARM (M-series), add `--namespace ''` to build images locally.

**Step 3: Check results:**

```bash
# Results appear in evaluation_results/memfun-v0.2.0/
# Score = % of instances where FAIL_TO_PASS tests now pass
#         AND PASS_TO_PASS tests still pass
```

### How It Works

```
For each SWE-bench instance:
  1. Clone the repo at the specified base commit
  2. Feed the issue description to Memfun's ContextFirstSolver
  3. If context-first fails → escalate to WorkflowEngine (multi-agent)
  4. Capture all file modifications as a unified diff
  5. Write prediction to JSONL

SWE-bench evaluation harness then:
  1. Applies your diff in a Docker container
  2. Runs the repo's test suite
  3. Checks FAIL_TO_PASS (issue resolved?) and PASS_TO_PASS (no regressions?)
```

## ACE-Bench

[ACE-Bench](https://openreview.net/forum?id=41xrZ3uGuI) evaluates end-to-end
feature development (not just bug fixes). 212 tasks from 16 repos. The code
is not yet publicly released -- when available, integration will follow the
same pattern as SWE-bench but with multi-file, multi-commit tasks that
specifically test Memfun's WorkflowEngine decomposition.
