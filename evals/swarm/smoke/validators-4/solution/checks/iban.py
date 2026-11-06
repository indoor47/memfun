"""IBAN mod-97.  See the task spec."""

from __future__ import annotations

import re

_LENGTHS = {"DE": 22, "GB": 22, "FR": 27, "NL": 18, "ES": 24}
_SHAPE = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z0-9]+")


def iban_valid(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.replace(" ", "").upper()
    if not t.isascii() or not _SHAPE.fullmatch(t) or not 15 <= len(t) <= 34:
        return False
    expected = _LENGTHS.get(t[:2])
    if expected is not None and len(t) != expected:
        return False
    rearranged = t[4:] + t[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1
