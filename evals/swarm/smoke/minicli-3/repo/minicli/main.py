"""Dispatcher.  Do not edit: subcommands plug in via register()/run()."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TextIO

COMMANDS = ("stats", "convert", "hashsum")


def build_parser(only: str | None = None) -> argparse.ArgumentParser:
    """All subcommands, or with ``only`` just that one imported (the others are listed but
    their modules are not imported, so one broken command cannot break the rest)."""
    parser = argparse.ArgumentParser(prog="minicli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        if only is None or name == only:
            importlib.import_module(f"minicli.commands.{name}").register(sub)
        else:
            sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    only = argv[0] if argv and argv[0] in COMMANDS else None
    args = build_parser(only).parse_args(argv)
    return int(args.func(args, out or sys.stdout))


if __name__ == "__main__":
    sys.exit(main())
