#!/usr/bin/env bash
# Hard-function arms (steer-16) on a served model: A memfun as is, C memfun with a deeper fix loop,
# D mini-swe-agent.  Six functions x TRIALS (default 3) = 18 one-agent episodes per arm, all at once,
# arms one after another, read isolation on.  Same environment as run_harness_benchmark.sh.
#
#   OUT=<dir> PY=... MEMFUN=... MINI=... evals/swarm/run_hard_functions.sh
#   ARMS="A C D"   which arms (default all);  FIX_ATTEMPTS=12 for arm C;  TRIALS=3
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
export REPO
ARMS_ABC=${ARMS_ABC:-"A C D"}
FIX_ATTEMPTS=${FIX_ATTEMPTS:-12}
export TRIALS=${TRIALS:-3}
# The functions where some arm missed a full pass in the isolated B300 rerun (wcswidth for sure).
export BUNDLES_OVERRIDE=${BUNDLES_OVERRIDE:-"c0s-wcwidth-wcswidth c0s-cachetools-mru_cache c0s-cachetools-hashkey c0s-wcwidth-_wcmatch_version c0s-wcwidth-list_versions c0s-wcwidth-wcwidth"}
export LIBS=${LIBS:-"wcwidth cachetools"}
first=1
for a in $ARMS_ABC; do
  echo; echo "######## arm $a  $(date +%H:%M:%S)"
  case "$a" in
    A) ARMS=memfun RUN_TAG=hardA EXTRA_ARGS="" ;;
    C) ARMS=memfun RUN_TAG=hardC EXTRA_ARGS="--fix-attempts $FIX_ATTEMPTS" ;;
    D) ARMS=mini-swe-agent RUN_TAG=hardD EXTRA_ARGS="" ;;
    *) echo "unknown arm $a"; continue ;;
  esac
  export ARMS RUN_TAG EXTRA_ARGS
  SKIP_SELFTEST=$([ $first = 1 ] && echo 0 || echo 1) "$REPO/evals/swarm/run_harness_benchmark.sh"
  first=0
done
echo; echo "hard-function arms done $(date +%H:%M:%S)"
