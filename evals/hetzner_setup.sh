#!/usr/bin/env bash
# Setup a Hetzner cloud instance for SWE-bench evaluation.
#
# Usage:
#   1. Create a Hetzner server (Ubuntu 24.04, CPX31 or higher recommended)
#   2. SSH in and run this script:
#      curl -sSL <this-file-url> | bash
#   OR copy-paste and run manually.
#
# After setup, copy your predictions.jsonl to the server and run:
#   python -m swebench.harness.run_evaluation \
#       --dataset_name princeton-nlp/SWE-bench_Lite \
#       --predictions_path predictions.jsonl \
#       --max_workers 8 \
#       --run_id memfun-v0.2.0

set -euo pipefail

echo "=== 1. System packages ==="
apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv docker.io

echo "=== 2. Start Docker ==="
systemctl enable docker
systemctl start docker
docker --version

echo "=== 3. Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== 4. Clone memfun ==="
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync

echo "=== 5. Install SWE-bench ==="
pip install swebench datasets

echo "=== 6. Ready ==="
echo ""
echo "Setup complete. Next steps:"
echo ""
echo "  # Set your API key"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  # Generate predictions (runs Memfun on each issue)"
echo "  cd memfun"
echo "  uv run python evals/swebench_harness.py \\"
echo "      --dataset princeton-nlp/SWE-bench_Lite \\"
echo "      --output evals/predictions.jsonl \\"
echo "      --max-instances 10"
echo ""
echo "  # Evaluate (runs repo tests in Docker)"
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Lite \\"
echo "      --predictions_path evals/predictions.jsonl \\"
echo "      --max_workers 8 \\"
echo "      --run_id memfun-v0.2.0"
