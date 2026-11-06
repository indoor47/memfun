"""`minicli hashsum`: file digests.  See the task spec."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TextIO


def register(subparsers) -> None:
    p = subparsers.add_parser("hashsum", help="file digests")
    p.add_argument("--algo", choices=("sha256", "sha1", "md5"), default="sha256")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=run)


def run(args: argparse.Namespace, out: TextIO) -> int:
    rc = 0
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            out.write(f"error: {name}: not found\n")
            rc = 1
            continue
        h = hashlib.new(args.algo)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        out.write(f"{h.hexdigest()}  {name}\n")
    return rc
