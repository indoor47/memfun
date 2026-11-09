#!/usr/bin/env bash
# Concurrency ramp: how many one-agent memfun episodes can one GPU serve at once?
#
#   LIB=tinydb OUT=<dir> evals/swarm/run_concurrency_ramp.sh        # steps 1, 2, 3 trials
#   STEPS="1 2 3 4" ...                                              # also 4 trials (4N agents)
#
# Step k runs EVERY function task of $LIB with k trials, all episodes at once (N x k agents), through
# run_harness_benchmark.sh (memfun arm only, read isolation, 25-min task cap).  A 10-second sampler
# records the server's /metrics (running / queued requests, KV usage, aggregate and per-stream tok/s,
# cache hit rate) the whole time; ramp_report.py summarises each step and says when the GPU
# saturates.  The ramp stops at the first saturated step (or when the Mac runs low on memory), then
# merge-checks the whole library from the best trial per function.
#
# Same environment as run_harness_benchmark.sh (PY, MEMFUN, OUT, CACHE, BASE_URL, METRICS_URL ...).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
CACHE=${CACHE:-$HOME/.cache/memfun-commit0}
OUT=${OUT:-$REPO/.memfun/ramp-runs}
LIB=${LIB:-tinydb}
STEPS=${STEPS:-"1 2 3"}
METRICS_URL=${METRICS_URL:-http://127.0.0.1:18000/metrics}
GB_PER_AGENT=${GB_PER_AGENT:-0.2}      # Mac memory one memfun process needs (guard only)
mkdir -p "$OUT"
export LIB OUT CACHE PY REPO METRICS_URL
N=$(ls "$CACHE/bundles" | grep -c "^c0s-${LIB}-")
[ "$N" -gt 0 ] || { echo "no c0s-${LIB}-* bundles in $CACHE/bundles"; exit 2; }
SAMPLES="$OUT/samples.csv"; STEPS_JSON="$OUT/steps.json"; echo "[]" > "$STEPS_JSON"
echo "== concurrency ramp: $LIB, $N function tasks, steps: $STEPS (agents = $N x trials)"
"$PY" "$REPO/evals/swarm/metrics_sampler.py" --url "$METRICS_URL" --out "$SAMPLES" --interval 10 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

free_gb() {  # macOS: free + inactive + speculative pages
  vm_stat 2>/dev/null | awk '/page size of/ {ps=$8} /Pages free|Pages inactive|Pages speculative/ {gsub("\\.","",$NF); n+=$NF} END {printf "%.0f", n*ps/1073741824}'
}

first=1; RUNS=()
for k in $STEPS; do
  agents=$((N * k))
  need=$(awk -v a="$agents" -v g="$GB_PER_AGENT" 'BEGIN {printf "%.0f", a*g}')
  have=$(free_gb); have=${have:-0}
  if [ "$have" -gt 0 ] && [ "$have" -lt "$need" ]; then
    echo "memory guard: step $k needs ~${need} GB for $agents agents, ${have} GB free: stopping the ramp"; break
  fi
  echo; echo "=== step $k: $agents agents ($k trials x $N tasks)  $(date +%H:%M:%S)"
  start=$(date +%s)
  SKIP_SELFTEST=$([ $first = 1 ] && echo 0 || echo 1) ARMS=memfun TRIALS=$k RUN_TAG=ramp$k LIBS=$LIB \
    "$REPO/evals/swarm/run_harness_benchmark.sh"
  end=$(date +%s); first=0; RUNS+=("$OUT/bench-memfun-ramp$k")
  "$PY" - "$STEPS_JSON" "$k" "$agents" "$start" "$end" "$OUT/bench-memfun-ramp$k/results.json" <<'PYEOF'
import json, sys
p, step, agents, start, end, res = sys.argv[1:]
d = json.load(open(p))
d.append({"step": int(step), "trials": int(step), "agents": int(agents), "start": int(start), "end": int(end), "results": res})
json.dump(d, open(p, "w"))
PYEOF
  "$PY" "$REPO/evals/swarm/ramp_report.py" --samples "$SAMPLES" --steps "$STEPS_JSON" --decide
  if [ $? -eq 10 ]; then echo "step $k is saturated: stopping the ramp"; break; fi
  sleep 20   # let the server drain before the next step
done

echo; echo "== agents vs throughput"
"$PY" "$REPO/evals/swarm/ramp_report.py" --samples "$SAMPLES" --steps "$STEPS_JSON" --json "$OUT/ramp-report.json" | tee "$OUT/ramp-report.txt"
if [ "${#RUNS[@]}" -gt 0 ]; then
  echo; echo "== merge-check: $LIB, best trial per function across all steps"
  args=(); for r in "${RUNS[@]}"; do args+=(--run "$r"); done
  "$PY" "$REPO/evals/swarm/commit0_adapter.py" merge-check "$LIB" "${args[@]}" --cache "$CACHE" \
    > "$OUT/merge-check-all.json" 2>&1 || true
  tr -d '\n ' < "$OUT/merge-check-all.json" | cut -c1-300; echo
fi
echo "done: $OUT (samples.csv, steps.json, ramp-report.txt, bench-memfun-ramp<k>/)"
