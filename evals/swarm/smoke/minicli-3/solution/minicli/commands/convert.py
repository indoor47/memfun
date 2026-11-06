"""`minicli convert`: convert lengths.  See the task spec."""

from __future__ import annotations

import argparse
from typing import TextIO

METERS = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
}


def register(subparsers) -> None:
    p = subparsers.add_parser("convert", help="convert lengths")
    p.add_argument("value", type=float)
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(func=run)


def run(args: argparse.Namespace, out: TextIO) -> int:
    for unit in (args.src, args.dst):
        if unit not in METERS:
            out.write(f"error: unknown unit {unit}\n")
            return 2
    result = args.value * METERS[args.src] / METERS[args.dst]
    out.write(f"{format(result, '.6g')} {args.dst}\n")
    return 0
