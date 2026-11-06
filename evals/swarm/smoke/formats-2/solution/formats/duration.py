"""Durations like 1h30m.  See the task spec."""

from __future__ import annotations

import re

_UNITS = (("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
_RE = re.compile(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def parse_duration(s: str) -> int:
    m = _RE.fullmatch(s)
    if not s or m is None or not any(m.groups()):
        raise ValueError(f"invalid duration: {s!r}")
    return sum(int(g) * mult for g, (_, mult) in zip(m.groups(), _UNITS) if g)


def format_duration(seconds: int) -> str:
    if seconds < 0:
        raise ValueError("negative duration")
    if seconds == 0:
        return "0s"
    out = []
    for suffix, mult in _UNITS:
        n, seconds = divmod(seconds, mult)
        if n:
            out.append(f"{n}{suffix}")
    return "".join(out)
