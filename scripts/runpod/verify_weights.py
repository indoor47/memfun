#!/usr/bin/env python3
"""Verify a downloaded Hugging Face snapshot against the Hub's own checksums (stdlib only).

For every file in the repo revision: LFS files are checked by sha256 (``lfs.sha256``),
other files by git blob sha1 (``blobId``).  On success writes ``.memfun_verified``
(revision sha + timestamp), which ``pod_boot.sh`` requires before starting SGLang.

Usage: python3 verify_weights.py <repo_id> <local_dir> [--revision main] [--offline-meta F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/models"
CHUNK = 8 * 1024 * 1024


def fetch_meta(repo: str, revision: str) -> dict[str, Any]:
    url = f"{HF_API}/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as resp:
        meta: dict[str, Any] = json.loads(resp.read())
    return meta


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(meta: dict[str, Any], local: Path) -> list[str]:
    """Return a list of problems (empty = verified)."""
    problems: list[str] = []
    siblings: list[dict[str, Any]] = meta.get("siblings") or []
    if not siblings:
        return ["Hub metadata lists no files"]
    for s in siblings:
        name = str(s["rfilename"])
        path = local / name
        if not path.is_file():
            problems.append(f"missing: {name}")
            continue
        lfs: dict[str, Any] | None = s.get("lfs")
        size = (lfs or {}).get("size", s.get("size"))
        if size is not None and path.stat().st_size != int(size):
            problems.append(f"size mismatch: {name} ({path.stat().st_size} != {size})")
            continue
        if lfs and lfs.get("sha256"):
            if sha256_file(path) != lfs["sha256"]:
                problems.append(f"sha256 mismatch: {name}")
        elif s.get("blobId") and git_blob_sha1(path) != s["blobId"]:
            problems.append(f"blob sha1 mismatch: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("repo")
    ap.add_argument("local_dir", type=Path)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--offline-meta", type=Path, default=None, help="metadata JSON (tests)")
    args = ap.parse_args(argv)

    meta: dict[str, Any]
    if args.offline_meta:
        meta = json.loads(args.offline_meta.read_text(encoding="utf-8"))
    else:
        meta = fetch_meta(args.repo, args.revision)
    siblings: list[dict[str, Any]] = meta.get("siblings") or []
    total = 0
    for s in siblings:
        lfs: dict[str, Any] = s.get("lfs") or {}
        total += int(lfs.get("size", s.get("size") or 0))
    print(f"{args.repo}@{meta.get('sha')}: {len(siblings)} files, {total / 1e9:.1f} GB")
    problems = verify(meta, args.local_dir)
    marker = args.local_dir / ".memfun_verified"
    if problems:
        marker.unlink(missing_ok=True)
        for p in problems[:50]:
            print("FAIL", p)
        print(f"NOT VERIFIED: {len(problems)} problem(s)")
        return 1
    marker.write_text(
        json.dumps({"repo": args.repo, "sha": meta.get("sha"), "verified_at": time.time()}),
        encoding="utf-8",
    )
    print(f"VERIFIED: wrote {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
