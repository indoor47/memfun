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
RUN_DIR="${MEMFUN_RUN_DIR:-/workspace/runs/${RUNPOD_POD_ID:-local}}"
mkdir -p "$RUN_DIR" 2>/dev/null || RUN_DIR=/tmp
SGLANG_PID=""

self_terminate() {
  trap '' TERM INT
  log "self-terminate: $1"
  local i
  for i in 1 2 3 4 5; do
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      # Newer CLI: `pod delete`; older CLI: `remove pod`.  Both documented by RunPod.
      runpodctl pod delete "$RUNPOD_POD_ID" && exit 0
      runpodctl remove pod "$RUNPOD_POD_ID" && exit 0
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
valid_name "$MODEL_REPO" || self_terminate "invalid MEMFUN_MODEL_REPO"
valid_name "$SERVED_NAME" || self_terminate "invalid MEMFUN_SERVED_NAME"
[[ "$REVISION" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$REVISION" != *..* ]] \
  || self_terminate "invalid MEMFUN_MODEL_REVISION"
MODELS_ROOT="${MEMFUN_MODELS_ROOT:-/workspace/models}"
case "$MODEL_DIR" in
  "$MODELS_ROOT"/*) ;;
  *) self_terminate "MEMFUN_MODEL_DIR outside $MODELS_ROOT" ;;
esac
[[ "$MODEL_DIR" != *..* ]] || self_terminate "MEMFUN_MODEL_DIR contains .."

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

if ! weights_ok; then
  if [ "$ALLOW_PULL" = "1" ] && [ -n "$MODEL_REPO" ]; then
    log "weights not verified at $MODEL_DIR: downloading $MODEL_REPO@$REVISION (GPU time is burning)"
    rm -f "$MARKER"
    mkdir -p "$MODEL_DIR"
    if command -v hf >/dev/null 2>&1; then
      hf download "$MODEL_REPO" --revision "$REVISION" --local-dir "$MODEL_DIR" >"$RUN_DIR/pull.log" 2>&1
    else
      huggingface-cli download "$MODEL_REPO" --revision "$REVISION" --local-dir "$MODEL_DIR" \
        >"$RUN_DIR/pull.log" 2>&1
    fi || self_terminate "weight download failed"
    # hf download checks transfer integrity; full Hub-checksum verification is
    # prepull_weights.sh's job (verify_weights.py is not shipped into the pod).
    sha=""
    [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] && sha="$REVISION"
    printf '{"repo": "%s", "sha": "%s", "revision": "%s", "source": "in-pod pull"}\n' \
      "$MODEL_REPO" "$sha" "$REVISION" >"$MARKER"
  else
    self_terminate "weights missing/unverified at $MODEL_DIR (run prepull_weights.sh first)"
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
wait "$SGLANG_PID"
rc=$?
log "SGLang exited rc=$rc; terminating in ${CRASH_GRACE_MIN}m (ssh in to debug: $RUN_DIR)"
sleep $(( CRASH_GRACE_MIN * 60 )) &
wait $!
self_terminate "SGLang exited rc=$rc"
