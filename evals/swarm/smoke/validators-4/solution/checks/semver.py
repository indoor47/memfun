"""Semantic versions.  See the task spec."""

from __future__ import annotations

import re

_NUM = r"(0|[1-9][0-9]*)"
_IDENT = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_RE = re.compile(
    rf"{_NUM}\.{_NUM}\.{_NUM}(?:-({_IDENT}(?:\.{_IDENT})*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def parse_semver(s: str) -> tuple[int, int, int, str | None]:
    m = _RE.fullmatch(s) if isinstance(s, str) and s.isascii() else None
    if m is None:
        raise ValueError(f"invalid semver: {s!r}")
    return int(m[1]), int(m[2]), int(m[3]), m[4]


def semver_valid(s: str) -> bool:
    try:
        parse_semver(s)
    except ValueError:
        return False
    return True


def _cmp(x: object, y: object) -> int:
    return (x > y) - (x < y)  # type: ignore[operator]


def _cmp_pre(a: str, b: str) -> int:
    pa, pb = a.split("."), b.split(".")
    for x, y in zip(pa, pb):
        if x == y:
            continue
        xn, yn = x.isdigit(), y.isdigit()
        if xn and yn:
            return _cmp(int(x), int(y))
        if xn != yn:
            return -1 if xn else 1
        return _cmp(x, y)
    return _cmp(len(pa), len(pb))


def compare_semver(a: str, b: str) -> int:
    va, vb = parse_semver(a), parse_semver(b)
    core = _cmp(va[:3], vb[:3])
    if core:
        return core
    if va[3] == vb[3]:
        return 0
    if va[3] is None:
        return 1
    if vb[3] is None:
        return -1
    return _cmp_pre(va[3], vb[3])
