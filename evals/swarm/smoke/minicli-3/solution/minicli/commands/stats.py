"""`minicli stats`: summary statistics.  See the task spec."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import TextIO


def register(subparsers) -> None:
    p = subparsers.add_parser("stats", help="summary statistics")
    p.add_argument("numbers", nargs="+", type=float)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=run)


def run(args: argparse.Namespace, out: TextIO) -> int:
    xs = args.numbers
    result = {
        "count": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }
    if args.json:
        out.write(json.dumps(result) + "\n")
    else:
        for key, value in result.items():
            out.write(f"{key} {format(value, 'g')}\n")
    return 0
