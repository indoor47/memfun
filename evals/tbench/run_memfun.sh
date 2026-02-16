#!/bin/bash
# Run Terminal-Bench with Memfun agent on Hetzner
#
# Usage:
#   ./run_memfun.sh [model] [run-id] [n-concurrent]
#
# Examples:
#   ./run_memfun.sh                                    # default: opus, 2 concurrent
#   ./run_memfun.sh anthropic/claude-sonnet-4-5-20250929 memfun-sonnet 3

set -euo pipefail

MODEL="${1:-anthropic/claude-opus-4-6-20250929}"
RUN_ID="${2:-memfun-agent-opus}"
N_CONCURRENT="${3:-2}"
OUTPUT_PATH="/root/tbench-runs"

# Ensure PYTHONPATH includes the agent directory
AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${AGENT_DIR}:${PYTHONPATH:-}"

echo "=== Memfun Terminal-Bench Run ==="
echo "Model:      ${MODEL}"
echo "Run ID:     ${RUN_ID}"
echo "Concurrent: ${N_CONCURRENT}"
echo "Agent dir:  ${AGENT_DIR}"
echo "Output:     ${OUTPUT_PATH}"
echo ""

tb run \
    --dataset terminal-bench-core==0.1.1 \
    --agent-import-path memfun_agent:MemfunAgent \
    --model "${MODEL}" \
    --n-concurrent "${N_CONCURRENT}" \
    --run-id "${RUN_ID}" \
    --output-path "${OUTPUT_PATH}"
