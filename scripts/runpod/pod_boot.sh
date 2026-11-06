#!/usr/bin/env bash
# In-pod boot script for memfun swarm pods (shipped base64 in $MEMFUN_BOOT_B64 by
# runpod_kit.py launch; runs as PID 1 via `exec bash /tmp/memfun_boot.sh`).
#
# Order matters: the hard deadline is armed FIRST, before anything that can hang.
#
# Self-termination uses RunPod's in-pod `runpodctl` with the Pod-scoped API key that
# RunPod injects as $RUNPOD_API_KEY (docs.runpod.io/runpodctl/overview: every Pod "comes
# preinstalled with runpodctl and a Pod-scoped API key"; docs.runpod.io/pods/templates/
# environment-variables: RUNPOD_API_KEY = "Pod-scoped API key", RUNPOD_POD_ID = "Unique
# Pod identifier").  The ACCOUNT key is never passed in.  If runpodctl is missing or
# fails, the fallback stops SGLang and exits PID 1: that frees the GPU from work but may
# NOT stop billing, which is why the local watchdog and terminateAfter also exist.
set -u
set -f  # no globbing: $MEMFUN_SGLANG_ARGS is word-split on purpose but must never glob

log() { echo "[memfun-boot] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

MAX_MIN="${MEMFUN_MAX_MINUTES:?MEMFUN_MAX_MINUTES unset}"
DEADLINE_EPOCH="${MEMFUN_DEADLINE_EPOCH:?MEMFUN_DEADLINE_EPOCH unset}"
MODEL_DIR="${MEMFUN_MODEL_DIR:?MEMFUN_MODEL_DIR unset}"
MODEL_REPO="${MEMFUN_MODEL_REPO:-}"
REVISION="${MEMFUN_MODEL_REVISION:-main}"
SERVED_NAME="${MEMFUN_SERVED_NAME:-model}"
PORT="${MEMFUN_SGLANG_PORT:-30000}"
EXTRA_ARGS="${MEMFUN_SGLANG_ARGS:-}"
ALLOW_PULL="${MEMFUN_ALLOW_PULL:-0}"
CRASH_GRACE_MIN="${MEMFUN_CRASH_GRACE_MIN:-10}"
FAIL_GRACE_MIN="${MEMFUN_FAIL_GRACE_MIN:-2}"
[[ "$FAIL_GRACE_MIN" =~ ^[0-9]{1,3}$ ]] || FAIL_GRACE_MIN=2
[[ "$CRASH_GRACE_MIN" =~ ^[0-9]{1,3}$ ]] || CRASH_GRACE_MIN=10
PULL_RETRIES="${MEMFUN_PULL_RETRIES:-3}"
[[ "$PULL_RETRIES" =~ ^[1-9]$ ]] || PULL_RETRIES=3
PULL_RETRY_SLEEP="${MEMFUN_PULL_RETRY_SLEEP:-15}"
[[ "$PULL_RETRY_SLEEP" =~ ^[0-9]{1,4}$ ]] || PULL_RETRY_SLEEP=15
PROGRESS_SEC="${MEMFUN_PROGRESS_SEC:-20}"
[[ "$PROGRESS_SEC" =~ ^[1-9][0-9]{0,3}$ ]] || PROGRESS_SEC=20
RUN_DIR="${MEMFUN_RUN_DIR:-/workspace/runs/${RUNPOD_POD_ID:-local}}"
mkdir -p "$RUN_DIR" 2>/dev/null || RUN_DIR=/tmp
SGLANG_PID=""
MIRROR_PID=""
LOGS_DUMPED=0

# Print the tail of every run log that exists, so the RunPod log stream (stdout) shows
# WHY the pod is going away even after the pod (and its disk) is gone.
dump_logs() {
  [ "$LOGS_DUMPED" = 1 ] && return 0
  LOGS_DUMPED=1
  local f
  for f in pull apt sglang; do
    if [ -s "$RUN_DIR/$f.log" ]; then
      log "---- last 40 lines of $RUN_DIR/$f.log ----"
      tail -n 40 "$RUN_DIR/$f.log" 2>/dev/null
      log "---- end of $f.log ----"
    fi
  done
}

stop_mirror() {
  if [ -n "$MIRROR_PID" ]; then
    sleep 1  # let tail -F flush what SGLang wrote last
    kill "$MIRROR_PID" 2>/dev/null
    MIRROR_PID=""
  fi
}

# One pod-scoped termination attempt.  The in-pod runpodctl is the OLDER CLI (observed
# 2026-11-06: `Error: unknown command "pod"`; `remove pod` worked), so try that first
# and keep the newer `pod delete` as a fallback.  Their stderr noise is not printed.
runpodctl_remove() {
  local out rc
  out=$(runpodctl remove pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    log "terminated via 'runpodctl remove pod': $(printf '%s' "$out" | head -n 1)"
    return 0
  fi
  log "'runpodctl remove pod' failed (rc=$rc); trying 'runpodctl pod delete'"
  out=$(runpodctl pod delete "$RUNPOD_POD_ID" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    log "terminated via 'runpodctl pod delete': $(printf '%s' "$out" | head -n 1)"
    return 0
  fi
  log "'runpodctl pod delete' failed (rc=$rc)"
  return 1
}

self_terminate() {
  trap '' TERM INT
  log "self-terminate: $1"
  dump_logs
  stop_mirror
  local i
  for i in 1 2 3 4 5; do
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      runpodctl_remove && exit 0
    else
      log "runpodctl or RUNPOD_POD_ID unavailable"
      break
    fi
    sleep 10
  done
  log "pod-scoped termination failed: stopping SGLang and exiting PID 1 (billing may continue;"
  log "the local watchdog / terminateAfter must terminate the pod)"
  [ -n "$SGLANG_PID" ] && kill -TERM "$SGLANG_PID" 2>/dev/null
  pkill -TERM -f sglang.launch_server 2>/dev/null
  sleep 15
  pkill -KILL -f sglang.launch_server 2>/dev/null
  # From the deadline subshell, also take down the main script (PID 1) so the container exits.
  [ "${BASHPID:-$$}" != "$$" ] && kill -TERM "$$" 2>/dev/null
  exit 1
}

# A failure (not the deadline, not a signal): explain, keep the pod FAIL_GRACE_MIN minutes
# so an operator can read the logs (the deadline still fires during the grace), then go.
fail() {
  log "FAILED: $1"
  dump_logs
  if [ "$FAIL_GRACE_MIN" -gt 0 ]; then
    log "terminating in ${FAIL_GRACE_MIN}m (MEMFUN_FAIL_GRACE_MIN); run dir: $RUN_DIR"
    sleep $(( FAIL_GRACE_MIN * 60 )) &
    wait $!
  fi
  self_terminate "$1"
}

# ---- 1. hard deadline: min(MAX_MIN from boot, absolute deadline from launch time) ----
now=$(date +%s)
by_epoch=$(( DEADLINE_EPOCH - now ))
by_minutes=$(( MAX_MIN * 60 ))
remaining=$(( by_epoch < by_minutes ? by_epoch : by_minutes ))
[ "$remaining" -lt 0 ] && remaining=0
log "hard stop armed: ${remaining}s from now"
( sleep "$remaining"; self_terminate "deadline reached" ) &
trap 'self_terminate "received SIGTERM"' TERM
trap 'self_terminate "received SIGINT"' INT

# ---- 1b. preflight diagnostics (stdout => RunPod log stream).  Names, never values. ----
with_timeout() { if command -v timeout >/dev/null 2>&1; then timeout 10 "$@"; else "$@"; fi; }
preflight() {
  log "---- preflight ----"
  echo "date: $(date -u)"
  echo "uname: $(uname -a 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu: $(with_timeout nvidia-smi --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader 2>&1 | tr '\n' ';')"
  else
    echo "gpu: nvidia-smi not found"
  fi
  echo "disk:"; df -h / "${MEMFUN_MODELS_ROOT:-/workspace}" 2>&1 | sed 's/^/  /'
  if command -v free >/dev/null 2>&1; then
    echo "memory: $(free -h 2>&1 | sed -n 2p)"
  elif [ -r /proc/meminfo ]; then
    echo "memory: $(head -n 2 /proc/meminfo | tr -s ' ' | tr '\n' ';')"
  else
    echo "memory: unknown"
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3: $(python3 --version 2>&1 | head -n 1)"
    with_timeout python3 -c '
import importlib
for m in ("sglang", "torch", "transformers", "huggingface_hub", "flashinfer", "hf_transfer"):
    try:
        print("  py %s %s" % (m, getattr(importlib.import_module(m), "__version__", "?")))
    except Exception as e:
        print("  py %s MISSING (%s)" % (m, type(e).__name__))
' 2>&1 | head -n 12
  else
    echo "python3: MISSING"
  fi
  local t v
  for t in hf huggingface-cli runpodctl curl wget git apt-get; do
    if command -v "$t" >/dev/null 2>&1; then
      v=""  # runpodctl is never run here: it is only for termination
      [ "$t" != runpodctl ] && v=$(with_timeout "$t" --version 2>&1 | head -n 1)
      echo "tool $t: $(command -v "$t") ${v:0:80}"
    else
      echo "tool $t: MISSING"
    fi
  done
  if command -v curl >/dev/null 2>&1; then
    echo "huggingface.co: $(curl -sI -m 10 https://huggingface.co 2>&1 | head -n 1 | tr -d '\r')"
  fi
  echo "env names: $(compgen -e | grep -E '^(RUNPOD|MEMFUN)_' | sort | tr '\n' ' ')"
  echo "ulimit -n: $(ulimit -n)"
  log "---- end preflight ----"
}
preflight

# ---- 2. sshd for the tunnel (key-only auth) ----
if [ -n "${PUBLIC_KEY:-}" ]; then
  if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then
    log "installing openssh-server"
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server) \
      >"$RUN_DIR/apt.log" 2>&1 || log "openssh-server install failed (see apt.log)"
  fi
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >/root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1
  /usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password \
    && log "sshd started" || log "sshd failed to start"
else
  log "no PUBLIC_KEY: sshd not started (tunnel impossible)"
fi

# ---- 3. weights ----
# Names are validated by the kit too; re-check here so the pod never builds a path from
# an unvalidated string (no "..", no absolute paths).
valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$ ]] && [[ "$1" != *..* ]]
}
valid_name "$MODEL_REPO" || fail "invalid MEMFUN_MODEL_REPO"
valid_name "$SERVED_NAME" || fail "invalid MEMFUN_SERVED_NAME"
[[ "$REVISION" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$REVISION" != *..* ]] \
  || fail "invalid MEMFUN_MODEL_REVISION"
MODELS_ROOT="${MEMFUN_MODELS_ROOT:-/workspace/models}"
case "$MODEL_DIR" in
  "$MODELS_ROOT"/*) ;;
  *) fail "MEMFUN_MODEL_DIR outside $MODELS_ROOT" ;;
esac
[[ "$MODEL_DIR" != *..* ]] || fail "MEMFUN_MODEL_DIR contains .."

MARKER="$MODEL_DIR/.memfun_verified"
marker_field() {  # one-line JSON written by verify_weights.py / this script
  sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" "$MARKER" 2>/dev/null | head -n 1
}
weights_ok() {
  [ -f "$MARKER" ] || { log "no $MARKER"; return 1; }
  [ "$(marker_field repo)" = "$MODEL_REPO" ] || { log "marker repo != $MODEL_REPO"; return 1; }
  if [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then  # pinned commit: must be exactly that one
    [ "$(marker_field sha)" = "$REVISION" ] || { log "marker sha != pinned $REVISION"; return 1; }
  fi
  return 0
}

# Weights download.  The lmsysorg/sglang runtime images ship NEITHER `hf` nor
# `huggingface-cli` (observed 2026-11-06), so fall back to huggingface_hub's Python API
# (a dependency of sglang/transformers; pip-installed as a last resort).  The `python -m`
# CLI module moved between huggingface_hub versions (commands.huggingface_cli in 0.x is
# gone in 2.x), so it is not used: snapshot_download() is the stable interface.
PY_SNAPSHOT='
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                         max_workers=8)
print("snapshot_download finished:", path)
'
DL_METHOD=""
DL_MISSING=""
py_has() { command -v python3 >/dev/null 2>&1 && python3 -c "import $1" >/dev/null 2>&1; }
pick_download_method() {
  if command -v hf >/dev/null 2>&1; then DL_METHOD=hf; return 0; fi
  DL_MISSING="$DL_MISSING hf"
  if command -v huggingface-cli >/dev/null 2>&1; then DL_METHOD=huggingface-cli; return 0; fi
  DL_MISSING="$DL_MISSING huggingface-cli"
  if py_has huggingface_hub; then DL_METHOD=python-api; return 0; fi
  DL_MISSING="$DL_MISSING python3:huggingface_hub"
  if command -v python3 >/dev/null 2>&1; then
    log "huggingface_hub not importable: trying pip install -q huggingface_hub"
    { python3 -m pip install -q huggingface_hub || pip install -q huggingface_hub; } \
      >>"$RUN_DIR/pull.log" 2>&1
    if py_has huggingface_hub; then DL_METHOD=python-api-pip; return 0; fi
    DL_MISSING="$DL_MISSING pip-install-failed"
  else
    DL_MISSING="$DL_MISSING python3"
  fi
  return 1
}
run_download() {
  case "$DL_METHOD" in
    hf) hf download "$MODEL_REPO" --revision "$REVISION" --local-dir "$MODEL_DIR" ;;
    huggingface-cli)
      huggingface-cli download "$MODEL_REPO" --revision "$REVISION" --local-dir "$MODEL_DIR" ;;
    python-api*) python3 -c "$PY_SNAPSHOT" "$MODEL_REPO" "$REVISION" "$MODEL_DIR" ;;
    *) return 1 ;;
  esac
}
download_weights() {
  local pull_log="$RUN_DIR/pull.log" attempt rc=1 start prog
  : >"$pull_log"
  if ! pick_download_method; then
    log "no way to download weights: missing:$DL_MISSING"
    log "---- last 30 lines of $pull_log ----"; tail -n 30 "$pull_log"
    return 1
  fi
  [ -n "$DL_MISSING" ] && log "not available:$DL_MISSING"
  if py_has hf_transfer; then export HF_HUB_ENABLE_HF_TRANSFER=1; log "hf_transfer enabled"; fi
  start=$(date +%s)
  for attempt in $(seq 1 "$PULL_RETRIES"); do
    log "download attempt $attempt/$PULL_RETRIES via $DL_METHOD: $MODEL_REPO@$REVISION"
    ( while sleep "$PROGRESS_SEC"; do
        log "download progress: $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1) in $MODEL_DIR," \
          "$(( $(date +%s) - start ))s elapsed ($DL_METHOD)"
      done ) &
    prog=$!
    run_download >>"$pull_log" 2>&1
    rc=$?
    kill "$prog" 2>/dev/null; wait "$prog" 2>/dev/null
    if [ "$rc" = 0 ]; then
      log "download finished via $DL_METHOD in $(( $(date +%s) - start ))s:" \
        "$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
      return 0
    fi
    log "download attempt $attempt/$PULL_RETRIES failed (rc=$rc); resumable: files kept"
    [ "$attempt" -lt "$PULL_RETRIES" ] && { sleep "$PULL_RETRY_SLEEP" & wait $!; }
  done
  log "all $PULL_RETRIES download attempts via $DL_METHOD failed;" \
    "not available:${DL_MISSING:- (none)}"
  log "---- last 30 lines of $pull_log ----"; tail -n 30 "$pull_log"
  return 1
}

if ! weights_ok; then
  if [ "$ALLOW_PULL" = "1" ] && [ -n "$MODEL_REPO" ]; then
    log "weights not verified at $MODEL_DIR: downloading $MODEL_REPO@$REVISION (GPU time is burning)"
    rm -f "$MARKER"
    mkdir -p "$MODEL_DIR"
    download_weights || fail "weight download failed (see the reason above)"
    # The Hub client checks transfer integrity; full Hub-checksum verification is
    # prepull_weights.sh's job (verify_weights.py is not shipped into the pod).
    sha=""
    [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] && sha="$REVISION"
    printf '{"repo": "%s", "sha": "%s", "revision": "%s", "source": "in-pod pull"}\n' \
      "$MODEL_REPO" "$sha" "$REVISION" >"$MARKER"
  else
    fail "weights missing/unverified at $MODEL_DIR (run prepull_weights.sh first)"
  fi
fi

# ---- 4. SGLang, bound to 127.0.0.1 only (reachable solely through the SSH tunnel) ----
# --enable-metrics: Prometheus text at /metrics on the same port (docs.sglang.io
# references/production_metrics); the memfun dashboard strip polls it via the tunnel.
log "starting SGLang: $MODEL_DIR as $SERVED_NAME on 127.0.0.1:$PORT $EXTRA_ARGS"
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
python3 -m sglang.launch_server --model-path "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" --enable-metrics $EXTRA_ARGS >"$RUN_DIR/sglang.log" 2>&1 &
SGLANG_PID=$!
# Mirror SGLang's log to stdout so the RunPod log stream shows startup and crashes.
tail -n +1 -F "$RUN_DIR/sglang.log" 2>/dev/null &
MIRROR_PID=$!
wait "$SGLANG_PID"
rc=$?
stop_mirror
log "SGLang exited rc=$rc; terminating in ${CRASH_GRACE_MIN}m (ssh in to debug: $RUN_DIR)"
dump_logs
sleep $(( CRASH_GRACE_MIN * 60 )) &
wait $!
self_terminate "SGLang exited rc=$rc"
