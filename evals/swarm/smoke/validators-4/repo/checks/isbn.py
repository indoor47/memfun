"""ISBN-10 / ISBN-13.  See the task spec."""

from __future__ import annotations


def isbn10_valid(s: str) -> bool:
    raise NotImplementedError


def isbn13_valid(s: str) -> bool:
    raise NotImplementedError


def isbn10_to_13(s: str) -> str:
    raise NotImplementedError
