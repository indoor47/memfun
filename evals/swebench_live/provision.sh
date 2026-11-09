#!/usr/bin/env bash
# Prepare a Linux x86_64 Docker box for the SWE-bench-Live runner, and take it down again.
#
#   provision.sh up   BASE REPO_DIR     build the agent bundle from the checkout at REPO_DIR
#   provision.sh down BASE              teardown: containers, network, firewall rules, images
#
# BASE is the ONLY directory this script writes to (e.g. /root/memfun-live).  It creates:
#   BASE/bundle/python   a standalone CPython 3.12 (uv-managed)
#   BASE/bundle/venv     memfun installed non-editable; mounted read-only into task containers
#                        at this same absolute path, so nothing needs relocating
#   BASE/bundle/VERSION  the commit the bundle was built from (recorded in every run.json)
# Needs: docker, git, uv.  No API key is ever needed or stored on the box: the model is reached
# through the recording proxy, whose upstream is a tunnel to a key-holding proxy elsewhere.
#
# Teardown checklist (provision.sh down does 1-4; 5-6 are yours):
#   1. no container named mfl-* is running        4. task images (starryzhang/sweb.eval.*) removed
#   2. the docker network mfl-net is gone         5. close the SSH reverse tunnel to the model proxy
#   3. iptables rules in BASE/iptables-added.txt  6. copy BASE/runs off the box before deleting it
#      are removed and the file is gone
set -euo pipefail
cmd=${1:?usage: provision.sh up BASE REPO_DIR | down BASE}
BASE=${2:?BASE directory}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}

case "$cmd" in
up)
  REPO=${3:?REPO_DIR (a memfun checkout)}
  [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || { echo "need Linux x86_64"; exit 2; }
  docker info >/dev/null || { echo "docker is not usable"; exit 2; }
  mkdir -p "$BASE"; BASE=$(cd "$BASE" && pwd -P)
  B="$BASE/bundle"
  rm -rf "$B.new"; mkdir -p "$B.new"
  # Build at the final path: a venv records absolute paths.
  rm -rf "$B"; mv "$B.new" "$B"
  UV_PYTHON_INSTALL_DIR="$B/python" "$UV" python install 3.12 >/dev/null
  PYBIN=$(ls "$B"/python/cpython-3.12*/bin/python3.12 | head -1)
  "$UV" venv --python "$PYBIN" "$B/venv" >/dev/null
  "$UV" pip install --python "$B/venv/bin/python" --link-mode copy "$REPO"/packages/*/ >/dev/null
  git -C "$REPO" rev-parse HEAD > "$B/VERSION"
  [ -z "$(git -C "$REPO" status --porcelain)" ] || echo "dirty" >> "$B/VERSION"
  echo "== self-tests"
  "$B/venv/bin/memfun" --help >/dev/null && echo "  ok: memfun starts from the bundle"
  # everything the venv points at must live inside the bundle (it is the only thing mounted)
  out=$(find "$B/venv" -type l -exec readlink -f {} \; | grep -v "^$B/" | head -3 || true)
  [ -z "$out" ] && echo "  ok: the bundle is self-contained" || { echo "  FAIL: links leave the bundle: $out"; exit 3; }
  sh "$REPO/evals/swebench_live/lockdown.sh" --check "$REPO" >/dev/null 2>&1 \
    && echo "  note: this checkout itself has no future history" || echo "  ok: lockdown self-check runs (and flags a normal checkout)"
  echo "bundle ready: $B ($(cat "$B/VERSION" | tr '\n' ' '))  size $(du -sh "$B" | cut -f1)"
  ;;
down)
  BASE=$(cd "$BASE" && pwd -P)
  ids=$(docker ps -aq --filter name='^mfl-' || true)
  [ -z "$ids" ] || docker rm -f -v $ids >/dev/null
  docker network rm mfl-net >/dev/null 2>&1 || true
  if [ -f "$BASE/iptables-added.txt" ]; then
    grep '^iptables -D ' "$BASE/iptables-added.txt" | while read -r line; do $line 2>/dev/null || true; done
    rm -f "$BASE/iptables-added.txt"
  fi
  imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^starryzhang/sweb\.eval\.' || true)
  [ -z "$imgs" ] || docker rmi $imgs >/dev/null 2>&1 || true
  echo "containers: $(docker ps -aq --filter name='^mfl-' | wc -l)  network: $(docker network ls --format '{{.Name}}' | grep -c '^mfl-net$' || true)  task images: $(docker images --format '{{.Repository}}' | grep -c '^starryzhang/sweb' || true)"
  ;;
*) echo "unknown command $cmd"; exit 2;;
esac
