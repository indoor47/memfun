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
# Agentic arms (memfun, mini, aider) run on THIS Mac as the user, so each episode is READ-ISOLATED
# (evals/swarm/sandbox/episode.sb via run_in_episode_sandbox.sh): the agent can read only its episode
# repo, its throwaway HOME and its own tool venvs; not the bundle cache (reference solutions, hidden
# tests), the clones cache, other episodes, pip caches or this repo; writes only to its episode;
# network only to loopback (the model tunnel).  A self-test refuses to start if that does not hold,
# and a contamination audit runs after every arm.  (The first run had write-only isolation: one
# mini episode read the reference solution.  See docs/HARNESS-BENCHMARK-2026-11.md.)
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
curl -fsS -m 10 "$BASE_URL/models" >/dev/null || { echo "model server not answering on $BASE_URL"; exit 2; }
SANDBOX="$REPO/evals/swarm/sandbox/run_in_episode_sandbox.sh"
real() { "$PY" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1"; }
PYPFX=$(dirname "$(dirname "$(real "$PY")")")   # interpreter prefix (uv python)
PYVENV=$(dirname "$(dirname "$PY")")            # the venv holding pytest + the test deps
# run_smoke gives every episode a CLEANED environment (an allowlist), so SANDBOX_READ_PATHS cannot be
# exported to the wrapper: it is passed INSIDE the command with env(1).  (Exporting it made every
# sandboxed episode exit 126: the harness could not read its own interpreter.)
sandbox_for() { echo "env SANDBOX_READ_PATHS=$(reads_for "$1") $SANDBOX"; }
# Read-only paths each arm's harness needs besides its episode (max 6, colon separated).
reads_for() { case "$1" in
  memfun) echo "$PYVENV:$PYPFX:$REPO/.venv:$REPO/packages";;
  mini-swe-agent) echo "$PYVENV:$PYPFX:$(dirname "$(dirname "$MINI")")";;
  aider) echo "$PYVENV:$PYPFX:$(dirname "$(dirname "$AIDER")")";;
  *) echo "$PYVENV:$PYPFX";; esac; }

echo "== isolation self-test (the run does not start unless every line passes)"
T="$OUT/isotest"; mkdir -p "$T/repo" "$T/home"
( cd "$T/repo"; export HOME="$T/home" SANDBOX_READ_PATHS="$PYVENV:$PYPFX"
  fail=0
  "$SANDBOX" ls "$CACHE/bundles" >/dev/null 2>&1 && { echo "  FAIL: can read the bundle cache"; fail=1; } || echo "  ok: bundle cache (solutions, hidden tests) unreadable"
  "$SANDBOX" ls "$CACHE" >/dev/null 2>&1 && { echo "  FAIL: can read the commit0 cache"; fail=1; } || echo "  ok: commit0 cache / clones unreadable"
  "$SANDBOX" ls "$REPO" >/dev/null 2>&1 && { echo "  FAIL: can read the memfun repo"; fail=1; } || echo "  ok: memfun repo unreadable"
  "$SANDBOX" ls "$OUT" >/dev/null 2>&1 && { echo "  FAIL: can read other episodes ($OUT)"; fail=1; } || echo "  ok: other episodes unreadable"
  "$SANDBOX" sh -c "echo x > $T/escape" >/dev/null 2>&1 && { echo "  FAIL: can write outside the episode"; fail=1; } || echo "  ok: writes outside the episode denied"
  "$SANDBOX" curl -s -m 6 -o /dev/null https://pypi.org/ >/dev/null 2>&1 && { echo "  FAIL: internet reachable"; fail=1; } || echo "  ok: internet unreachable"
  "$SANDBOX" sh -c 'echo ok > f && cat f' >/dev/null 2>&1 && echo "  ok: the episode repo is readable and writable" || { echo "  FAIL: cannot use the episode repo"; fail=1; }
  "$SANDBOX" "$PY" -c "import wcwidth" >/dev/null 2>&1 && { echo "  FAIL: wcwidth is installed for the agent"; fail=1; } || echo "  ok: no installed copy of wcwidth"
  "$SANDBOX" "$PY" -c "import cachetools" >/dev/null 2>&1 && { echo "  FAIL: cachetools is installed for the agent"; fail=1; } || echo "  ok: no installed copy of cachetools"
  # The way run_smoke launches a harness: cleaned environment, read paths passed via env(1).
  for arm in $ARMS; do
    bin=$PY; [ "$arm" = memfun ] && bin=$MEMFUN
    [ "$arm" = mini-swe-agent ] && bin=$MINI
    [ "$arm" = aider ] && bin=$AIDER
    [ "$arm" = single-shot ] && continue
    env -i PATH="$PATH" HOME="$T/home" sh -c "$(sandbox_for "$arm") $bin --help" >/dev/null 2>&1 \
      && echo "  ok: $arm starts under the sandbox with a cleaned environment" \
      || { echo "  FAIL: $arm cannot start under the sandbox with a cleaned environment"; fail=1; }
  done
  exit $fail ) || { echo "isolation self-test FAILED: not starting"; exit 3; }

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
  echo "-- contamination audit: $arm"
  "$PY" "$REPO/evals/swarm/audit_contamination.py" --run "$OUT/bench-$arm" --bundles "$CACHE/bundles" \
    --json "$OUT/bench-$arm/audit.json" || echo "audit failed for $arm"
}

for arm in $ARMS; do
  case "$arm" in
    single-shot)    run_arm single-shot;;
    memfun)         run_arm memfun --memfun-cmd "$(sandbox_for memfun) $MEMFUN";;
    aider)          run_arm aider --aider-edit-format "$AIDER_EDIT_FORMAT" --arm-bin "$(sandbox_for aider) $AIDER";;
    mini-swe-agent) run_arm mini-swe-agent --arm-bin "$(sandbox_for mini-swe-agent) $MINI";;
    claude-code)    run_arm claude-code --arm-bin "$(sandbox_for claude-code) ${CLAUDE:-claude}" --anthropic-base-url "${ANTHROPIC_BASE_URL_FOR_BENCH:?set it}";;
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
