"""Dispatcher.  Do not edit: subcommands plug in via register()/run()."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TextIO

COMMANDS = ("stats", "convert", "hashsum")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minicli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        importlib.import_module(f"minicli.commands.{name}").register(sub)
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args, out or sys.stdout))


if __name__ == "__main__":
    sys.exit(main())
