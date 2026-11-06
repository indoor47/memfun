"""Luhn checksum.  See the task spec."""

from __future__ import annotations


def _sum(digits: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total


def luhn_valid(s: str) -> bool:
    if not isinstance(s, str):
        return False
    digits = s.replace(" ", "")
    if len(digits) < 2 or not digits.isascii() or not digits.isdigit():
        return False
    return _sum(digits) % 10 == 0


def luhn_check_digit(partial: str) -> str:
    digits = partial.replace(" ", "")
    if not digits.isascii() or not digits.isdigit():
        raise ValueError("digits only")
    for d in "0123456789":
        if _sum(digits + d) % 10 == 0:
            return d
    raise AssertionError("unreachable")
