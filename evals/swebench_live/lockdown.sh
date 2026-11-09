#!/bin/sh
# Remove every trace of history newer than the checked-out commit from a git repository.
#
#   lockdown.sh REPO_DIR          lock down, then self-check
#   lockdown.sh --check REPO_DIR  self-check only (exit 1 and a reason per line when it fails)
#
# SWE-bench-Live rule: the agent may not see commits, issues or PRs newer than the issue.  The
# task image's repository can carry branches, tags, stashes, reflogs or loose objects that hold
# the real fix.  After this script the object store contains only what is reachable from HEAD:
# remotes removed, every ref that is not an ancestor of HEAD deleted (ancestor tags stay:
# setuptools_scm needs them), reflogs expired, unreachable objects pruned.  The working tree
# and the index are not touched.  POSIX sh + git only (it runs inside the task container).
set -eu

check_only=0
if [ "${1:-}" = "--check" ]; then check_only=1; shift; fi
repo=${1:?usage: lockdown.sh [--check] REPO_DIR}
cd "$repo"
git rev-parse --git-dir >/dev/null
gitdir=$(git rev-parse --absolute-git-dir)
head=$(git rev-parse HEAD)
g() { git -c gc.auto=0 -c core.hooksPath=/dev/null "$@"; }

if [ "$check_only" = 0 ]; then
  for r in $(g remote); do g remote remove "$r"; done
  cur=$(g symbolic-ref -q HEAD || true)
  g for-each-ref --format='%(refname)' | while IFS= read -r ref; do
    [ "$ref" = "$cur" ] && continue
    c=$(g rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)
    if [ -z "$c" ] || ! g merge-base --is-ancestor "$c" "$head"; then
      g update-ref -d "$ref"
    fi
  done
  rm -f "$gitdir/FETCH_HEAD" "$gitdir/ORIG_HEAD" "$gitdir/MERGE_HEAD" "$gitdir/CHERRY_PICK_HEAD" \
        "$gitdir/REVERT_HEAD" "$gitdir/BISECT_LOG"
  rm -rf "$gitdir/refs/original" "$gitdir/rebase-merge" "$gitdir/rebase-apply" \
         "$gitdir/worktrees" "$gitdir/lost-found"
  g reflog expire --expire=now --expire-unreachable=now --all
  rm -rf "$gitdir/logs"
  g repack -a -d -q            # drop packs holding unreachable objects (gc keeps "cruft")
  g prune --expire=now
  g gc --prune=now --quiet
  rm -f "$gitdir"/objects/pack/*.mtimes "$gitdir"/objects/pack/tmp_* 2>/dev/null || true
fi

fail=0
say() { echo "lockdown check FAILED: $*"; fail=1; }
[ -z "$(g remote)" ] || say "remotes remain: $(g remote | tr '\n' ' ')"
[ "$(g rev-parse HEAD)" = "$head" ] || say "HEAD moved"
all=$(g rev-list --all --count 2>/dev/null || echo x)
anc=$(g rev-list --count HEAD)
# --all also covers HEAD's own branch and ancestor tags: equal counts = nothing beyond HEAD
[ "$all" = "$anc" ] || [ "$all" = 0 ] || say "refs reach $all commits, HEAD reaches $anc"
g for-each-ref --format='%(refname)' | while IFS= read -r ref; do
  c=$(g rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)
  if [ -z "$c" ] || ! g merge-base --is-ancestor "$c" "$head"; then echo "$ref"; fi
done > "$gitdir/.lockdown-bad-refs"
[ ! -s "$gitdir/.lockdown-bad-refs" ] || say "refs beyond HEAD: $(tr '\n' ' ' < "$gitdir/.lockdown-bad-refs")"
rm -f "$gitdir/.lockdown-bad-refs"
un=$(g fsck --unreachable --no-reflogs --no-progress 2>/dev/null | grep -c '^unreachable' || true)
[ "$un" = 0 ] || say "$un unreachable objects remain in the object store"
[ ! -d "$gitdir/logs" ] || [ -z "$(find "$gitdir/logs" -type f 2>/dev/null | head -1)" ] || say "reflogs remain"
for f in FETCH_HEAD ORIG_HEAD; do [ ! -e "$gitdir/$f" ] || say "$f remains"; done
[ ! -d "$gitdir/modules" ] || say "submodule repositories under .git/modules were not checked"
[ "$fail" = 0 ] && echo "lockdown ok: $anc commits reachable, nothing beyond HEAD $head"
exit "$fail"
