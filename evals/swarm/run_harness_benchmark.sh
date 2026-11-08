#!/usr/bin/env bash
# Harness benchmark (steer-15): the SAME model, tasks and hidden-test grader, different coding agents.
#
#   evals/swarm/run_harness_benchmark.sh            # run every arm, cheapest first
#   PLAN=1 evals/swarm/run_harness_benchmark.sh     # print the plan + expected wall time, run nothing
#   ARMS="single-shot memfun" ...                   # a subset (order kept)
#
# Runs on the Mac against the B300 through the SSH tunnel (SGLang on 127.0.0.1:18000, model served
# as qwen3.8-27b-fp8, thinking off).  Each arm = ONE run_smoke.py run with ALL tasks in parallel:
# 15 function tasks (wcwidth 6 + cachetools 9) x 3 trials = 45 episodes, 25 min cap per episode.
# Results: $OUT/bench-<arm>/results.json ; progress: .memfun/bench-<arm>-progress.md
#
# Setup the script expects (override with env):
#   PY        python with pytest + the libraries' test deps (runs run_smoke.py and the grader)
#   CACHE     commit0 bundle cache holding c0-wcwidth, c0s-wcwidth-*, c0s-cachetools-*
#             (python evals/swarm/commit0_adapter.py build wcwidth cachetools --split --cache $CACHE)
#   MEMFUN    path of the memfun executable of the checkout under test (main after PR #77)
#   MINI      mini-swe-agent v2 `mini`   (uv venv V && uv pip install --python V/bin/python mini-swe-agent)
#   AIDER     aider-chat `aider`         (uv venv V && uv pip install --python V/bin/python aider-chat)
# Agentic arms (memfun, mini, aider) run bash on THIS Mac, so they run under a sandbox-exec profile
# that denies writes anywhere under $HOME except $OUT (a local mini run once wandered into the repo).
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
CACHE=${CACHE:-$HOME/.cache/memfun-commit0}
OUT=${OUT:-$REPO/.memfun/bench-runs}
MEMFUN=${MEMFUN:-$REPO/.venv/bin/memfun}
MINI=${MINI:-mini}
AIDER=${AIDER:-aider}
BASE_URL=${BASE_URL:-http://127.0.0.1:18000/v1}
METRICS_URL=${METRICS_URL:-http://127.0.0.1:18000/metrics}
TRIALS=${TRIALS:-3}
EPISODE_TIMEOUT=${EPISODE_TIMEOUT:-1500}          # 25 min per task
MAX_TOKENS=${MAX_TOKENS:-32768}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-262144}
AIDER_EDIT_FORMAT=${AIDER_EDIT_FORMAT:-diff}
ARMS=${ARMS:-"single-shot memfun aider mini-swe-agent"}   # cheapest first; claude-code only if smoked

# 15 tasks: wcwidth (6) + cachetools (9; typedmethodkey, 4 tests, is left out as too weak).
BUNDLES=(c0s-wcwidth-_bisearch c0s-wcwidth-_wcmatch_version c0s-wcwidth-_wcversion_value
         c0s-wcwidth-list_versions c0s-wcwidth-wcswidth c0s-wcwidth-wcwidth
         c0s-cachetools-fifo_cache c0s-cachetools-lfu_cache c0s-cachetools-lru_cache
         c0s-cachetools-mru_cache c0s-cachetools-rr_cache c0s-cachetools-ttl_cache
         c0s-cachetools-hashkey c0s-cachetools-methodkey c0s-cachetools-typedkey)
PARALLEL=$(( ${#BUNDLES[@]} * TRIALS ))

# Expected wall per arm (all 45 episodes at once; from the Cerebras smoke + the B300 wcwidth runs).
expect() { case "$1" in
  single-shot)    echo "~3 min  (one call per task)";;
  memfun)         echo "~8 min  (B300 runs: 3-7 min; the result reaper ends the lingering process)";;
  aider)          echo "~8 min  (one edit + auto-test rounds)";;
  mini-swe-agent) echo "~25 min (10-20 bash steps per task; capped at 25 min per task)";;
  claude-code)    echo "~25 min (only if its smoke passed)";;
  *) echo "?";; esac; }

echo "== harness benchmark plan"
echo "model server : $BASE_URL (metrics $METRICS_URL)   tasks: ${#BUNDLES[@]} x $TRIALS trials = $PARALLEL episodes per arm, all at once"
echo "limits       : ${EPISODE_TIMEOUT}s per task, max_tokens $MAX_TOKENS, temperature 0.7, repetition_penalty 1.05, thinking off (server default)"
for a in $ARMS; do echo "  arm $a: $(expect "$a")"; done
echo "expected total: ~45 min of arm time plus ~2 min grading per arm (< 70 min)"
[ "${PLAN:-0}" = 1 ] && exit 0

mkdir -p "$OUT"
SB="$OUT/nowrite.sb"
cat > "$SB" <<EOF
(version 1)
(allow default)
(deny file-write* (subpath "$HOME"))
(allow file-write* (subpath "$OUT"))
(allow file-write* (subpath "$HOME/.cache"))
(allow file-write* (subpath "$HOME/.local"))
EOF
curl -fsS -m 10 "$BASE_URL/models" >/dev/null || { echo "model server not answering on $BASE_URL"; exit 2; }
SANDBOX="sandbox-exec -f $SB"

run_arm() {
  local arm=$1; shift
  echo; echo "=== arm $arm  $(date +%H:%M:%S)  expected $(expect "$arm")"
  "$PY" "$REPO/evals/swarm/run_smoke.py" --suite commit0 --commit0-dir "$CACHE" \
    --base-url "$BASE_URL" --metrics-url "$METRICS_URL" --arm "$arm" \
    --bundles "${BUNDLES[@]}" --trials "$TRIALS" --parallel-tasks "$PARALLEL" \
    --episode-timeout "$EPISODE_TIMEOUT" --task-timeout $((EPISODE_TIMEOUT - 50)) --grader-timeout 900 \
    --temperature 0.7 --repetition-penalty 1.05 --max-tokens "$MAX_TOKENS" \
    --context-window "$CONTEXT_WINDOW" --no-dashboard \
    --out-dir "$OUT" --run-id "bench-$arm" \
    --progress-file "$REPO/.memfun/bench-$arm-progress.md" "$@" || echo "arm $arm exited $?"
}

for arm in $ARMS; do
  case "$arm" in
    single-shot)    run_arm single-shot;;
    memfun)         run_arm memfun --memfun-cmd "$SANDBOX $MEMFUN";;
    aider)          run_arm aider --aider-edit-format "$AIDER_EDIT_FORMAT" --arm-bin "$SANDBOX $AIDER";;
    mini-swe-agent) run_arm mini-swe-agent --arm-bin "$SANDBOX $MINI";;
    claude-code)    run_arm claude-code --arm-bin "$SANDBOX ${CLAUDE:-claude}" --anthropic-base-url "${ANTHROPIC_BASE_URL_FOR_BENCH:?set it}";;
    *) echo "unknown arm $arm"; exit 2;;
  esac
done

echo; echo "== merge-check (whole-library score from the best trial per function), per arm"
for arm in $ARMS; do
  for lib in wcwidth cachetools; do
    "$PY" "$REPO/evals/swarm/commit0_adapter.py" merge-check "$lib" --run "$OUT/bench-$arm" \
      --cache "$CACHE" > "$OUT/bench-$arm/merge-check-$lib.json" 2>&1 || true
    echo "$arm/$lib: $(tr -d '\n ' < "$OUT/bench-$arm/merge-check-$lib.json" | cut -c1-200)"
  done
done
echo "done: results in $OUT/bench-<arm>/results.json"
