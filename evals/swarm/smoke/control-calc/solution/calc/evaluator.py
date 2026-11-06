"""Tree-walking evaluator."""

from __future__ import annotations

from typing import Any


def _call(name: str, args: list[Any], pos: int) -> Any:
    from calc import CalcError

    if name in ("min", "max"):
        if not args:
            raise CalcError(f"{name}() needs at least one argument", pos)
        return min(args) if name == "min" else max(args)
    if name == "abs":
        if len(args) != 1:
            raise CalcError("abs() takes exactly one argument", pos)
        return abs(args[0])
    raise CalcError(f"unknown function {name!r}", pos)


def ev(node: Any, env: dict[str, Any]) -> Any:
    from calc import CalcError

    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        if node[1] not in env:
            raise CalcError(f"unknown variable {node[1]!r}", node[2])
        return env[node[1]]
    if kind == "neg":
        return -ev(node[1], env)
    if kind == "assign":
        env[node[1]] = ev(node[2], env)
        return env[node[1]]
    if kind == "call":
        return _call(node[1], [ev(a, env) for a in node[2]], node[3])
    if kind == "bin":
        _, op, left, right, pos = node
        a, b = ev(left, env), ev(right, env)
        try:
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "%":
                return a % b
            return a**b
        except ZeroDivisionError:
            raise CalcError("division by zero", pos) from None
        except (OverflowError, ValueError, TypeError) as exc:
            raise CalcError(str(exc), pos) from None
    raise AssertionError(kind)


def run(program: list[Any], env: dict[str, Any]) -> Any:
    result = None
    for stmt in program:
        result = ev(stmt, env)
    return result
