"""`minicli hashsum`: file digests.  See the task spec."""

from __future__ import annotations

import argparse
from typing import TextIO


def register(subparsers) -> None:
    p = subparsers.add_parser("hashsum", help="file digests")
    p.set_defaults(func=run)


def run(args: argparse.Namespace, out: TextIO) -> int:
    raise NotImplementedError
