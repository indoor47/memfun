"""Tokens: (kind, text, pos).  kind in NUM, NAME, OP, EOF."""

from __future__ import annotations

from typing import NamedTuple


class Token(NamedTuple):
    kind: str
    text: str
    pos: int


_OPS = ("**", "+", "-", "*", "/", "%", "(", ")", ",", ";", "=")


def tokenize(src: str) -> list[Token]:
    from calc import CalcError

    out: list[Token] = []
    i = 0
    while i < len(src):
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < len(src) and src[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < len(src) and (src[j].isdigit() or (src[j] == "." and not seen_dot)):
                seen_dot = seen_dot or src[j] == "."
                j += 1
            out.append(Token("NUM", src[i:j], i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] == "_"):
                j += 1
            out.append(Token("NAME", src[i:j], i))
            i = j
            continue
        for op in _OPS:
            if src.startswith(op, i):
                out.append(Token("OP", op, i))
                i += len(op)
                break
        else:
            raise CalcError(f"unexpected character {c!r}", i)
    out.append(Token("EOF", "", len(src)))
    return out
