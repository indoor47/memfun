#!/usr/bin/env bash
# Run a command (an agent harness) read-isolated for ONE episode: cwd = the episode repo,
# $HOME = the episode's throwaway home (what run_smoke sets).  Extra read-only paths (the
# harness venv, the interpreter prefix) come from SANDBOX_READ_PATHS (colon separated, max 6).
#   --arm-bin "evals/swarm/sandbox/run_in_episode_sandbox.sh /path/to/mini"
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
args=(-D "REPO=$(pwd -P)" -D "HOME_EP=$(cd "$HOME" && pwd -P)")
IFS=: read -r -a extra <<< "${SANDBOX_READ_PATHS:-}"
for i in 0 1 2 3 4 5; do
  p=${extra[$i]:-/nonexistent}
  [ "$p" = /nonexistent ] || p=$(cd "$p" && pwd -P)
  args+=(-D "P$i=$p")
done
exec sandbox-exec "${args[@]}" -f "$here/episode.sb" "$@"
