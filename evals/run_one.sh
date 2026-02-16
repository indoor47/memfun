#!/usr/bin/env bash
# Quick sanity check: run Memfun on a single SWE-bench instance.
#
# Usage:
#   ./evals/run_one.sh
#
# Prerequisites:
#   pip install swebench datasets
#
# This runs ONE easy instance (sympy__sympy-20590) and shows the result.
# Good for verifying the harness works before running the full benchmark.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: Generate prediction for 1 instance ==="
uv run python evals/swebench_harness.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --output evals/predictions_test.jsonl \
    --instance-ids sympy__sympy-20590 \
    --model-name memfun-test

echo ""
echo "=== Step 2: Show prediction ==="
cat evals/predictions_test.jsonl | python -m json.tool 2>/dev/null || cat evals/predictions_test.jsonl

echo ""
echo "=== Step 3: Evaluate with SWE-bench (requires Docker) ==="
echo "Run this manually:"
echo ""
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Lite \\"
echo "      --predictions_path evals/predictions_test.jsonl \\"
echo "      --max_workers 1 \\"
echo "      --instance_ids sympy__sympy-20590 \\"
echo "      --run_id memfun-test"
echo ""
echo "  # On macOS ARM, add: --namespace ''"
