"""ISBN-10 / ISBN-13.  See the task spec."""

from __future__ import annotations


def _clean(s: str) -> str:
    return s.replace("-", "").replace(" ", "") if isinstance(s, str) else ""


def isbn10_valid(s: str) -> bool:
    t = _clean(s).upper()
    if len(t) != 10 or not t[:9].isdigit() or not (t[9].isdigit() or t[9] == "X"):
        return False
    if not t.isascii():
        return False
    total = sum((10 - i) * int(c) for i, c in enumerate(t[:9]))
    total += 10 if t[9] == "X" else int(t[9])
    return total % 11 == 0


def _isbn13_check(first12: str) -> str:
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(first12))
    return str((10 - total % 10) % 10)


def isbn13_valid(s: str) -> bool:
    t = _clean(s)
    if len(t) != 13 or not t.isascii() or not t.isdigit():
        return False
    return _isbn13_check(t[:12]) == t[12]


def isbn10_to_13(s: str) -> str:
    if not isbn10_valid(s):
        raise ValueError(f"not a valid ISBN-10: {s!r}")
    first12 = "978" + _clean(s)[:9]
    return first12 + _isbn13_check(first12)
