#!/usr/bin/env bash
# Pre-pull model weights onto the RunPod network volume BEFORE renting the GPU.
#
# Run on a cheap CPU pod (or any pod) that has the network volume mounted at /workspace,
# in the SAME data center as the volume.  Resumable: re-running continues a partial
# download (hf keeps per-file progress under <dest>/.cache).  Then every file is checked
# against the Hub's sha256 (LFS) / git blob sha1 and a .memfun_verified marker is written;
# pod_boot.sh refuses to start SGLang without it.
#
# Usage: bash prepull_weights.sh [REPO] [DEST]
#   REPO default Qwen/Qwen3.8-27B-FP8 (~30.9 GB, 81 files; checked 2026-11-06)
#   DEST default /workspace/models/$REPO
#   REVISION=<sha|branch> to pin (default main).  HF_TOKEN only needed for gated repos.
#   --verify-only as $1 skips the download.
set -euo pipefail

VERIFY_ONLY=0
if [ "${1:-}" = "--verify-only" ]; then VERIFY_ONLY=1; shift; fi
REPO="${1:-Qwen/Qwen3.8-27B-FP8}"
DEST="${2:-/workspace/models/$REPO}"
REVISION="${REVISION:-main}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

need_gb=40   # weights ~31 GB + hf cache/temp headroom
mkdir -p "$DEST"
avail_gb=$(df -Pk "$DEST" | awk 'NR==2 {print int($4/1024/1024)}')
echo "destination $DEST: ${avail_gb} GB free (need ~${need_gb} GB)"
if [ "$VERIFY_ONLY" = "0" ] && [ "$avail_gb" -lt "$need_gb" ]; then
  echo "not enough space on the volume; grow it or pick a smaller model" >&2
  exit 1
fi

if [ "$VERIFY_ONLY" = "0" ]; then
  if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    python3 -m pip install -q -U "huggingface_hub[hf_xet]"
  fi
  for attempt in 1 2 3 4 5; do
    echo "download attempt $attempt: $REPO@$REVISION -> $DEST"
    if command -v hf >/dev/null 2>&1; then
      hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" && break
    else
      huggingface-cli download "$REPO" --revision "$REVISION" --local-dir "$DEST" && break
    fi
    [ "$attempt" = 5 ] && { echo "download failed 5 times" >&2; exit 1; }
    sleep 20
  done
fi

python3 "$HERE/verify_weights.py" "$REPO" "$DEST" --revision "$REVISION"
du -sh "$DEST"
