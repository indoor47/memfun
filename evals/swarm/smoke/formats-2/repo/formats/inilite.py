"""A strict INI subset.  See the task spec."""

from __future__ import annotations


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    raise NotImplementedError


def dump_ini(data: dict[str, dict[str, str]]) -> str:
    raise NotImplementedError
