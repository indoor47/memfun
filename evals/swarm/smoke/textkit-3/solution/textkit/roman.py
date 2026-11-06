"""Roman numerals, 1..3999."""

from __future__ import annotations

_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]
_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n: int) -> str:
    if not isinstance(n, int) or not 1 <= n <= 3999:
        raise ValueError(f"out of range: {n!r}")
    out = []
    for value, sym in _TABLE:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def from_roman(s: str) -> int:
    t = s.strip().upper()
    if not t or any(c not in _VALUES for c in t):
        raise ValueError(f"not a roman numeral: {s!r}")
    total = 0
    for i, c in enumerate(t):
        v = _VALUES[c]
        if i + 1 < len(t) and _VALUES[t[i + 1]] > v:
            total -= v
        else:
            total += v
    if not 1 <= total <= 3999 or to_roman(total) != t:
        raise ValueError(f"non-canonical roman numeral: {s!r}")
    return total
