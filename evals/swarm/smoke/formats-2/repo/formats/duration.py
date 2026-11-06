"""Durations like 1h30m.  See the task spec."""

from __future__ import annotations


def parse_duration(s: str) -> int:
    raise NotImplementedError


def format_duration(seconds: int) -> str:
    raise NotImplementedError
