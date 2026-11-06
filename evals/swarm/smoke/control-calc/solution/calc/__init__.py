"""Tiny expression language.  See the task spec."""

from __future__ import annotations


class CalcError(Exception):
    def __init__(self, message: str, pos: int | None = None) -> None:
        super().__init__(message)
        self.pos = pos


def evaluate(src: str, env: dict[str, int | float] | None = None) -> int | float:
    from calc.evaluator import run
    from calc.lexer import tokenize
    from calc.parser import Parser

    program = Parser(tokenize(src)).program()
    return run(program, dict(env or {}))
