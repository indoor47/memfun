#!/bin/sh
# One box's shard of a long run with sealed grading (docs/SWEBENCH-LIVE-LITE-PLAN.md).
#
#   box_run.sh BASE NAME IDS_FILE DATASET SPLIT
#
# BASE      the box's state directory, set up before: BASE/repo (this repository at the frozen
#           commit), BASE/bundle (provision.sh up), BASE/eval (grade.py setup)
# NAME      the shard's name: run directory BASE/runs/NAME, sealed directory BASE/sealed/NAME
# IDS_FILE  the instance ids of this shard, one per line
# DATASET   the split as JSONL (dataset.py fetch)
# SPLIT     lite | verified | full  (what the official evaluator is told)
#
# The agent settings are fixed HERE, in the repository, so that every box runs the same ones
# and the frozen commit names them.  Only where the model is reached and how many instances
# run at once come from the environment:
#   UPSTREAM  (default http://127.0.0.1:18431/e/live/v1: the key-holding proxy, on this box or
#             through an SSH forward)   MODEL (qwen-3.8-27b)
#   WORKERS (3) instances in work at once on this box, of which at most ROLLOUT_SLOTS (1) are
#             in their rollout; the others are being graded.  ROLLOUT_SLOTS times the number
#             of boxes is what the model path must carry: an endpoint limited to 150,000
#             tokens per minute carries about five rollouts with waiting, not fifteen
#   PRICE_IN, PRICE_OUT  dollars per million tokens, for the cost column only (0.99, 1.49)
#
# An instance that ends as an infrastructure error (a failed pull, a step of the runner that
# timed out, a model path that stayed down) is run again from scratch by the next pass; the
# earlier attempt is kept and logged by the runner.  A final instance is never run again.
# The script ends when every instance is final (exit 0) or after PASSES passes (exit 1).
# It does NOT finalize the run: that is a separate, deliberate step.
set -u
BASE=${1:?usage: box_run.sh BASE NAME IDS_FILE DATASET SPLIT}
NAME=${2:?}; IDS=${3:?}; DATASET=${4:?}; SPLIT=${5:?}
R="$BASE/repo/evals/swebench_live"
PASSES=${PASSES:-8}
mkdir -p "$BASE/runs" "$BASE/sealed"
n=0
while :; do
  n=$((n + 1))
  echo "box_run $NAME: pass $n at $(date -u +%H:%M:%S)"
  python3 "$R/runner.py" run --dataset "$DATASET" --ids-file "$IDS" --run-dir "$BASE/runs/$NAME" \
    --upstream "${UPSTREAM:-http://127.0.0.1:18431/e/live/v1}" --model "${MODEL:-qwen-3.8-27b}" \
    --bundle "$BASE/bundle" --solver context-first --temperature 0.7 --reasoning-effort none \
    --llm-max-tokens 32768 --context-window 131072 \
    --instance-timeout 1800 --max-tokens-per-instance 520000 --cpus 2 --memory 8g \
    --price-in "${PRICE_IN:-0.99}" --price-out "${PRICE_OUT:-1.49}" \
    --host-firewall --iptables-record "$BASE/iptables-added-$NAME.txt" --network "mfl-$NAME" \
    --workers "${WORKERS:-3}" --rollout-slots "${ROLLOUT_SLOTS:-1}" \
    --retry-infra --outage-wait 10800 \
    --sealed "$BASE/sealed/$NAME" --evaluator "$BASE/eval" --split "$SPLIT" --gold-check \
    --agent-option discover_tests=true --agent-option reproduce=true \
    --agent-option repo_search=true --agent-option probe_guard=off \
    --agent-option fix_detail=true --agent-option probe_rounds=2 \
    --agent-option max_fix_attempts=4 --agent-option read_rounds=2 \
    --agent-option read_offer=false --agent-option offline=true
  code=$?
  echo "box_run $NAME: pass $n ended with $code at $(date -u +%H:%M:%S)"
  [ "$code" -eq 0 ] && { echo "BOXDONE $NAME: every instance is final"; exit 0; }
  [ "$n" -ge "$PASSES" ] && { echo "BOXSTOP $NAME: not final after $n passes"; exit 1; }
  sleep 120
done
