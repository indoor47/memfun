"""Semantic versions.  See the task spec."""

from __future__ import annotations


def parse_semver(s: str) -> tuple[int, int, int, str | None]:
    raise NotImplementedError


def semver_valid(s: str) -> bool:
    raise NotImplementedError


def compare_semver(a: str, b: str) -> int:
    raise NotImplementedError
