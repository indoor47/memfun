"""Check that an agent bundle holds exactly the dependency versions of its lock file.

A bundle built later from the same commit must be the same bundle.  ``provision.sh`` installs
the versions pinned in ``uv.lock`` and keeps a copy of that file in the bundle; this script
compares what is installed in ``BUNDLE/venv`` with it and fails on any difference.  A bundle
that picked up a newer ``dspy`` or ``litellm`` once sent every request in a format the model
server refused: every rollout ended before its first model call.

    python -m evals.swebench_live.bundle_check BUNDLE [--lock uv.lock]

Prints the versions of the packages named in :data:`MUST_MATCH` and a digest of the whole
installed set (recorded with a run as the identity of its bundle).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

#: Missing or different versions of these fail the check even when nothing else differs.
MUST_MATCH = ("dspy", "litellm")
_LIST_INSTALLED = (
    "import importlib.metadata as m, json; "
    "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
)


def canonical(name: str) -> str:
    """A package name as the lock file and the installer both write it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock_text: str) -> dict[str, str]:
    """``{package: version}`` of a ``uv.lock`` (workspace members have no fixed version in
    the bundle and are left out)."""
    out: dict[str, str] = {}
    for pkg in tomllib.loads(lock_text).get("package", []):
        source = pkg.get("source") or {}
        if "editable" in source or "virtual" in source or "workspace" in source:
            continue
        if pkg.get("name") and pkg.get("version"):
            out[canonical(str(pkg["name"]))] = str(pkg["version"])
    return out


def mismatches(installed: dict[str, str], locked: dict[str, str]) -> list[str]:
    """Every way *installed* departs from *locked*: a different version of a locked package,
    or a package of :data:`MUST_MATCH` that is not installed or not locked."""
    have = {canonical(k): v for k, v in installed.items()}
    out = [
        f"{name}: installed {have[name]}, locked {version}"
        for name, version in sorted(locked.items())
        if name in have and have[name] != version
    ]
    for name in MUST_MATCH:
        if name not in have:
            out.append(f"{name}: not installed")
        elif name not in locked:
            out.append(f"{name}: not in the lock file")
    return out


def freeze_digest(installed: dict[str, str]) -> str:
    """A digest of the whole installed set: the identity of a bundle's dependencies."""
    lines = sorted(f"{canonical(k)}=={v}" for k, v in installed.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def installed_versions(bundle: Path) -> dict[str, str]:
    """What the bundle's own interpreter has installed."""
    res = subprocess.run(
        [str(bundle / "venv" / "bin" / "python"), "-c", _LIST_INSTALLED],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    doc = json.loads(res.stdout)
    return {str(k): str(v) for k, v in doc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--lock", type=Path, default=None, help="default: BUNDLE/uv.lock")
    ap.add_argument("--write", action="store_true", help="write BUNDLE/FREEZE.txt and .sha256")
    a = ap.parse_args(argv)
    lock = a.lock or a.bundle / "uv.lock"
    if not lock.is_file():
        print(f"FAIL: no lock file at {lock}")
        return 2
    installed = installed_versions(a.bundle)
    locked = locked_versions(lock.read_text("utf-8"))
    bad = mismatches(installed, locked)
    have = {canonical(k): v for k, v in installed.items()}
    for name in MUST_MATCH:
        print(f"  {name} {have.get(name, '(missing)')} (locked {locked.get(name, '(missing)')})")
    digest = freeze_digest(installed)
    print(f"  {len(installed)} packages installed, digest {digest}")
    if a.write:
        lines = sorted(f"{canonical(k)}=={v}" for k, v in installed.items())
        (a.bundle / "FREEZE.txt").write_text("\n".join(lines) + "\n", "utf-8")
        (a.bundle / "FREEZE.sha256").write_text(digest + "\n", "utf-8")
    if bad:
        print("FAIL: the bundle is not what the lock file says:")
        for line in bad[:20]:
            print(f"  {line}")
        return 3
    print("  ok: every installed package has its locked version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
