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
# The container's real stdout/stderr, so a trap that fires while a step's output is
# redirected (e.g. `bounded ... >apt.log`) still reports to the RunPod log stream.
exec 8>&1 9>&2
to_console() { exec 1>&8 2>&9; }

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
num_or() { [[ "$1" =~ ^[0-9]{1,5}$ ]] && echo "$1" || echo "$2"; }
PULL_ATTEMPT_SEC=$(num_or "${MEMFUN_PULL_ATTEMPT_SEC:-1500}" 1500)  # per-attempt cap
STALL_SEC=$(num_or "${MEMFUN_STALL_SEC:-300}" 300)         # no growth this long => retry
PIP_TIMEOUT=$(num_or "${MEMFUN_PIP_TIMEOUT:-300}" 300)
RPC_TIMEOUT=$(num_or "${MEMFUN_RUNPODCTL_TIMEOUT:-60}" 60)  # per runpodctl call
TERM_ATTEMPTS=$(num_or "${MEMFUN_TERM_ATTEMPTS:-5}" 5)
TERM_RETRY_SLEEP=$(num_or "${MEMFUN_TERM_RETRY_SLEEP:-10}" 10)
KILL_GRACE=$(num_or "${MEMFUN_KILL_GRACE:-15}" 15)
# After runpodctl REPORTS success, give RunPod this long to actually stop the container;
# if it is still running, stop it from inside (H2: a 0 exit is not proof).
REMOVE_WAIT=$(num_or "${MEMFUN_REMOVE_WAIT_SEC:-120}" 120)
GROUP_KILL_WAIT=$(num_or "${MEMFUN_GROUP_KILL_WAIT:-10}" 10)  # TERM -> KILL for a group
RUN_DIR="${MEMFUN_RUN_DIR:-/workspace/runs/${RUNPOD_POD_ID:-local}}"
mkdir -p "$RUN_DIR" 2>/dev/null || RUN_DIR=/tmp
SGLANG_PID=""
IN_DEADLINE=0
MIRROR_PID=""
LOGS_DUMPED=0

# Kill a process and all its descendants (children first).  Needs pgrep for the
# descendants; without it only the process itself (callers that start a process in its
# own group kill the whole group instead, which needs no pgrep).
kill_tree() {
  local p=$1 sig=${2:-TERM} c
  if command -v pgrep >/dev/null 2>&1; then
    for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c" "$sig"; done
  fi
  kill -"$sig" "$p" 2>/dev/null
}

# spawn_group CMD...: start CMD in the background in its OWN process group (set -m), so
# the whole tree can be signalled with `kill -- -PGID`.  The pid (= pgid) is $SPAWNED.
SPAWNED=""
spawn_group() {
  set -m
  "$@" &
  SPAWNED=$!
  set +m
}

# kill_group PGID: TERM the group, wait up to GROUP_KILL_WAIT s, then KILL it.
kill_group() {
  local g=$1 i
  kill -TERM -- "-$g" 2>/dev/null || kill_tree "$g" TERM
  for i in $(seq 1 "$GROUP_KILL_WAIT"); do
    kill -0 -- "-$g" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$g" 2>/dev/null || kill_tree "$g" KILL
}

# bounded SECS CMD...: run CMD in the background and `wait` for it, so (a) a hang is cut
# off after SECS (TERM, then KILL 5 s later; returns 124 like coreutils timeout) and (b)
# this shell stays in `wait`, where TERM traps run immediately (bash defers traps while a
# FOREGROUND child runs).  The watcher writes nothing, so $(bounded ...) never blocks on it.
bounded() {
  local secs=$1 pid wd rc flag
  shift
  flag="$RUN_DIR/.timeout.$$.$RANDOM"
  spawn_group "$@"
  pid=$SPAWNED
  ( sleep "$secs"; kill -0 "$pid" 2>/dev/null || exit 0; : >"$flag"
    kill -TERM -- "-$pid" 2>/dev/null || kill_tree "$pid" TERM; sleep 5
    kill -KILL -- "-$pid" 2>/dev/null || kill_tree "$pid" KILL ) >/dev/null 2>&1 &
  wd=$!
  disown "$wd" 2>/dev/null  # no "Killed" job notice when it is stopped below
  wait "$pid"; rc=$?
  kill_tree "$wd" KILL 2>/dev/null
  if [ -f "$flag" ]; then rm -f "$flag"; log "timed out after ${secs}s: $1" >&2; return 124; fi
  return "$rc"
}

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

# Run runpodctl with default TERM/INT handling (self_terminate ignores them, and ignored
# signals are inherited), so a timeout's TERM takes effect at once (L1).
runpodctl_untrapped() { trap - TERM INT; exec runpodctl "$@"; }

# One pod-scoped termination attempt.  The in-pod runpodctl is the OLDER CLI (observed
# 2026-11-06: `Error: unknown command "pod"`; `remove pod` worked), so try that first
# and keep the newer `pod delete` as a fallback.  Their stderr noise is not printed.
runpodctl_remove() {
  local out rc
  out=$(bounded "$RPC_TIMEOUT" runpodctl_untrapped remove pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    log "terminated via 'runpodctl remove pod': $(printf '%s' "$out" | head -n 1)"
    return 0
  fi
  [ "$rc" = 124 ] && log "'runpodctl remove pod' timed out after ${RPC_TIMEOUT}s"
  log "'runpodctl remove pod' failed (rc=$rc); trying 'runpodctl pod delete'"
  out=$(bounded "$RPC_TIMEOUT" runpodctl_untrapped pod delete "$RUNPOD_POD_ID" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    log "terminated via 'runpodctl pod delete': $(printf '%s' "$out" | head -n 1)"
    return 0
  fi
  [ "$rc" = 124 ] && log "'runpodctl pod delete' timed out after ${RPC_TIMEOUT}s"
  log "'runpodctl pod delete' failed (rc=$rc)"
  return 1
}

# Main shell, on SIGUSR1 from the deadline subshell (removal failed, or "succeeded" but
# the container is still running): stop everything and exit PID 1.  USR1 needs no marker
# file (a read-only RUN_DIR cannot break it) and is never ignored by self_terminate.
die_now() {
  trap '' TERM INT USR1
  to_console
  local c j
  log "container shutdown: stopping all processes and exiting PID 1"
  for j in $(jobs -p); do kill -TERM -- "-$j" 2>/dev/null; kill_tree "$j" TERM; done
  if command -v pgrep >/dev/null 2>&1; then
    for c in $(pgrep -P "$$" 2>/dev/null); do kill_tree "$c" TERM; done
  fi
  [ -n "$SGLANG_PID" ] && kill_tree "$SGLANG_PID" TERM
  sleep 2
  for j in $(jobs -p); do kill -KILL -- "-$j" 2>/dev/null; kill_tree "$j" KILL; done
  if command -v pgrep >/dev/null 2>&1; then
    for c in $(pgrep -P "$$" 2>/dev/null); do kill_tree "$c" KILL; done
  fi
  exit 1
}

# Is the main shell (PID 1) still running?  A zombie (exited, not yet reaped) is not.
main_alive() {
  kill -0 "$$" 2>/dev/null || return 1
  case "$(ps -o stat= -p "$$" 2>/dev/null)" in Z*) return 1 ;; esac
  return 0
}

# Deadline subshell only: make PID 1 stop the container.
stop_container() {
  kill -USR1 "$$" 2>/dev/null
  sleep 10
  kill -KILL "$$" 2>/dev/null  # effective only when not PID 1 (PID 1 ignores it)
}

self_terminate() {
  trap '' TERM INT
  to_console
  log "self-terminate: $1"
  dump_logs
  stop_mirror
  local i
  for i in $(seq 1 "$TERM_ATTEMPTS"); do
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      if runpodctl_remove; then
        if [ "$IN_DEADLINE" = 1 ]; then
          # A 0 exit is not proof: if the container still runs after REMOVE_WAIT, stop it.
          local w=0
          while [ "$w" -lt "$REMOVE_WAIT" ] && main_alive; do
            sleep 1; w=$(( w + 1 ))
          done
          if main_alive; then
            log "runpodctl reported success but PID 1 still runs after ${REMOVE_WAIT}s:" \
              "stopping the container"
            stop_container
          fi
        fi
        exit 0
      fi
    else
      log "runpodctl or RUNPOD_POD_ID unavailable"
      break
    fi
    [ "$i" -lt "$TERM_ATTEMPTS" ] && sleep "$TERM_RETRY_SLEEP"
  done
  log "pod-scoped termination failed: stopping SGLang and exiting PID 1 (billing may continue;"
  log "the local watchdog / terminateAfter must terminate the pod)"
  [ -n "$SGLANG_PID" ] && kill -TERM "$SGLANG_PID" 2>/dev/null
  pkill -TERM -f sglang.launch_server 2>/dev/null
  sleep "$KILL_GRACE"
  pkill -KILL -f sglang.launch_server 2>/dev/null
  # From the deadline subshell, also take down the main script (PID 1) so the container
  # exits.  Every long step in the main shell runs under `bounded`/`wait`, so its TERM
  # trap runs at once; the marker makes it skip runpodctl and just stop (die_now).  KILL
  # follows for a non-PID-1 main shell (PID 1 ignores KILL from inside its namespace).
  [ "$IN_DEADLINE" = 1 ] && stop_container
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
DEADLINE_AT=$(( now + remaining ))
# IN_DEADLINE marks the timer subshell explicitly ($BASHPID does not exist in bash 3.2).
( IN_DEADLINE=1; sleep "$remaining"; self_terminate "deadline reached" ) &
trap 'self_terminate "received SIGTERM"' TERM
trap 'self_terminate "received SIGINT"' INT
trap die_now USR1

# ---- 1b. preflight diagnostics (stdout => RunPod log stream).  Names, never values. ----
with_timeout() { bounded 10 "$@"; }
# probe SECS CMD...: first line of CMD's output, bounded, run in THIS shell (not in a
# command substitution, where a pending TERM trap would wait for it).
PROBE_OUT="$RUN_DIR/.probe.$$"
probe() {
  local secs=$1
  shift
  bounded "$secs" "$@" >"$PROBE_OUT" 2>&1 || echo "(failed or timed out)" >>"$PROBE_OUT"
  PROBE=$(head -n 1 "$PROBE_OUT" 2>/dev/null)
}
preflight() {
  log "---- preflight ----"
  echo "date: $(date -u)"
  echo "uname: $(uname -a 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    bounded 10 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
      >"$PROBE_OUT" 2>&1
    echo "gpu: $(tr '\n' ';' <"$PROBE_OUT")"
  else
    echo "gpu: nvidia-smi not found"
  fi
  echo "disk:"
  bounded 10 df -h / "${MEMFUN_MODELS_ROOT:-/workspace}" >"$PROBE_OUT" 2>&1
  sed 's/^/  /' "$PROBE_OUT"
  if command -v free >/dev/null 2>&1; then
    bounded 10 free -h >"$PROBE_OUT" 2>&1
    echo "memory: $(sed -n 2p "$PROBE_OUT")"
  elif [ -r /proc/meminfo ]; then
    echo "memory: $(head -n 2 /proc/meminfo | tr -s ' ' | tr '\n' ';')"
  else
    echo "memory: unknown"
  fi
  if command -v python3 >/dev/null 2>&1; then
    probe 10 python3 --version
    echo "python3: $PROBE"
    bounded 20 python3 -c '
import importlib
for m in ("sglang", "torch", "transformers", "huggingface_hub", "flashinfer", "hf_transfer"):
    try:
        print("  py %s %s" % (m, getattr(importlib.import_module(m), "__version__", "?")))
    except Exception as e:
        print("  py %s MISSING (%s)" % (m, type(e).__name__))
' >"$PROBE_OUT" 2>&1
    head -n 12 "$PROBE_OUT"
  else
    echo "python3: MISSING"
  fi
  local t v
  for t in hf huggingface-cli runpodctl curl wget git apt-get pgrep pkill; do
    if command -v "$t" >/dev/null 2>&1; then
      v=""  # runpodctl is never run here: it is only for termination
      case "$t" in runpodctl|pgrep|pkill) ;; *) probe 10 "$t" --version; v=$PROBE ;; esac
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
    apt_sshd() {
      apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server
    }
    bounded 900 apt_sshd >"$RUN_DIR/apt.log" 2>&1 || log "openssh-server install failed (see apt.log)"
  fi
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >/root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  bounded 60 ssh-keygen -A >/dev/null 2>&1 || log "ssh-keygen -A failed or timed out"
  if bounded 30 /usr/sbin/sshd -o PasswordAuthentication=no \
      -o PermitRootLogin=prohibit-password; then
    log "sshd started"
  else
    log "sshd failed to start (or timed out): skipped"
  fi
else
  log "no PUBLIC_KEY: sshd not started (tunnel impossible)"
fi

# ---- 3. weights ----
# Names are validated by the kit too; re-check here so the pod never builds a path from
# an unvalidated string (no "..", no absolute paths).
valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$ ]] && [[ "$1" != *..* ]] \
    && [[ "$1" != -* ]] && [[ "$1" != */-* ]]  # never an option to hf/pip
}
valid_name "$MODEL_REPO" || fail "invalid MEMFUN_MODEL_REPO"
valid_name "$SERVED_NAME" || fail "invalid MEMFUN_SERVED_NAME"
[[ "$REVISION" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$REVISION" != *..* ]] && [[ "$REVISION" != -* ]] \
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
py_has() {
  command -v python3 >/dev/null 2>&1 && bounded 60 python3 -c "import $1" >/dev/null 2>&1
}
pick_download_method() {
  if command -v hf >/dev/null 2>&1; then DL_METHOD=hf; return 0; fi
  DL_MISSING="$DL_MISSING hf"
  if command -v huggingface-cli >/dev/null 2>&1; then DL_METHOD=huggingface-cli; return 0; fi
  DL_MISSING="$DL_MISSING huggingface-cli"
  if py_has huggingface_hub; then DL_METHOD=python-api; return 0; fi
  DL_MISSING="$DL_MISSING python3:huggingface_hub"
  if command -v python3 >/dev/null 2>&1; then
    log "huggingface_hub not importable: trying pip install -q huggingface_hub"
    { bounded "$PIP_TIMEOUT" python3 -m pip install -q huggingface_hub \
        || bounded "$PIP_TIMEOUT" pip install -q huggingface_hub; } \
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
# Directories a download writes to: the target plus the Hugging Face / xet caches.
dl_dirs() {
  local d
  for d in "$MODEL_DIR" "${HF_HOME:-${HOME:-/root}/.cache/huggingface}" "${HF_XET_CACHE:-}"; do
    [ -n "$d" ] && [ -d "$d" ] && printf '%s\n' "$d"
  done
}
# Total KB of the download dirs (apparent size where du supports it), bounded.
dl_size_kb() {
  local dirs
  dirs=$(dl_dirs)
  [ -z "$dirs" ] && { echo 0; return; }
  # shellcheck disable=SC2086  # dirs are newline-separated paths without spaces by validation
  { bounded 30 du -sk --apparent-size $dirs 2>/dev/null || bounded 30 du -skA $dirs 2>/dev/null \
      || bounded 30 du -sk $dirs 2>/dev/null; } | awk '{s += $1} END {print s + 0}'
}
dl_touched_since() {
  local dirs
  dirs=$(dl_dirs)
  [ -z "$dirs" ] && return 0
  # shellcheck disable=SC2086
  bounded 30 find $dirs -type f -newer "$1" -print 2>/dev/null | head -n 1
}

download_weights() {
  local pull_log="$RUN_DIR/pull.log" attempt rc=1 start prog dl cap
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
    # Per-attempt cap: never past the deadline (it fires anyway), at most PULL_ATTEMPT_SEC.
    cap=$(( DEADLINE_AT - $(date +%s) ))
    [ "$cap" -gt "$PULL_ATTEMPT_SEC" ] && cap=$PULL_ATTEMPT_SEC
    [ "$cap" -lt 1 ] && cap=1
    # Background + wait (not foreground) so the TERM trap / deadline can always act.
    # Own process group: a stall/cap kill reaches every process the downloader started.
    spawn_group run_download >>"$pull_log" 2>&1
    dl=$SPAWNED
    ( t0=$(date +%s); last=-1; still=0; stamp="$RUN_DIR/.dl_stamp"; : >"$stamp"
      while sleep "$PROGRESS_SEC"; do
        kill -0 "$dl" 2>/dev/null || exit 0
        kb=$(dl_size_kb)
        log "download progress: $(( ${kb:-0} / 1024 )) MiB in $MODEL_DIR," \
          "$(( $(date +%s) - start ))s elapsed ($DL_METHOD)"
        # Progress = the size changed OR any file under the model dir / HF caches was
        # modified since the last check (preallocated files keep the size flat).
        if [ "${kb:-0}" != "$last" ] || [ -n "$(dl_touched_since "$stamp")" ]; then
          still=0
        else
          still=$(( still + PROGRESS_SEC ))
        fi
        last=${kb:-0}; : >"$stamp"
        if [ "$STALL_SEC" -gt 0 ] && [ "$still" -ge "$STALL_SEC" ]; then
          log "download stalled: no progress for ${still}s; killing this attempt"
          kill_group "$dl"; exit 0
        fi
        if [ $(( $(date +%s) - t0 )) -ge "$cap" ]; then
          log "download attempt hit its ${cap}s cap; killing it"
          kill_group "$dl"; exit 0
        fi
      done ) &
    prog=$!
    wait "$dl"
    rc=$?
    kill_tree "$prog" KILL; wait "$prog" 2>/dev/null
    kill -KILL -- "-$dl" 2>/dev/null  # no stray writer of this attempt survives into the next
    if [ "$rc" = 0 ]; then
      kb=$(dl_size_kb)
      log "download finished via $DL_METHOD in $(( $(date +%s) - start ))s:" \
        "$(( ${kb:-0} / 1024 )) MiB"
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
    bounded 60 rm -f "$MARKER" || fail "cannot remove $MARKER (volume hung?)"
    bounded 60 mkdir -p "$MODEL_DIR" || fail "cannot create $MODEL_DIR (volume hung?)"
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
