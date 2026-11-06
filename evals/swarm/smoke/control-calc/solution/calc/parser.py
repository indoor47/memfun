"""Recursive-descent parser producing tuples:
("num", v) ("var", name) ("neg", e) ("bin", op, l, r) ("call", name, args, pos)
("assign", name, e).  Each node also carries the source pos as its last element."""

from __future__ import annotations

from typing import Any

from calc.lexer import Token


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self, offset: int = 0) -> Token:
        return self.toks[min(self.i + offset, len(self.toks) - 1)]

    def take(self) -> Token:
        tok = self.peek()
        self.i += 1
        return tok

    def error(self, msg: str) -> Exception:
        from calc import CalcError

        tok = self.peek()
        return CalcError(f"{msg} at {tok.pos}", tok.pos)

    def expect(self, text: str) -> Token:
        if self.peek().kind == "OP" and self.peek().text == text:
            return self.take()
        raise self.error(f"expected {text!r}")

    def program(self) -> list[Any]:
        stmts: list[Any] = []
        while True:
            if self.peek().kind == "EOF":
                break
            if self.peek().kind == "OP" and self.peek().text == ";":
                self.take()
                continue
            stmts.append(self.statement())
            if self.peek().kind == "OP" and self.peek().text == ";":
                self.take()
            elif self.peek().kind != "EOF":
                raise self.error("expected ';' or end")
        if not stmts:
            raise self.error("empty program")
        return stmts

    def statement(self) -> Any:
        tok = self.peek()
        nxt = self.peek(1)
        if tok.kind == "NAME" and nxt.kind == "OP" and nxt.text == "=":
            self.take()
            self.take()
            return ("assign", tok.text, self.expr(), tok.pos)
        return self.expr()

    def expr(self) -> Any:
        node = self.term()
        while self.peek().kind == "OP" and self.peek().text in ("+", "-"):
            op = self.take()
            node = ("bin", op.text, node, self.term(), op.pos)
        return node

    def term(self) -> Any:
        node = self.unary()
        while self.peek().kind == "OP" and self.peek().text in ("*", "/", "%"):
            op = self.take()
            node = ("bin", op.text, node, self.unary(), op.pos)
        return node

    def unary(self) -> Any:
        tok = self.peek()
        if tok.kind == "OP" and tok.text in ("-", "+"):
            self.take()
            inner = self.unary()
            return ("neg", inner, tok.pos) if tok.text == "-" else inner
        return self.power()

    def power(self) -> Any:
        base = self.atom()
        if self.peek().kind == "OP" and self.peek().text == "**":
            op = self.take()
            return ("bin", "**", base, self.unary(), op.pos)
        return base

    def atom(self) -> Any:
        tok = self.peek()
        if tok.kind == "NUM":
            self.take()
            value = float(tok.text) if "." in tok.text else int(tok.text)
            return ("num", value, tok.pos)
        if tok.kind == "NAME":
            self.take()
            if self.peek().kind == "OP" and self.peek().text == "(":
                self.take()
                args = []
                if not (self.peek().kind == "OP" and self.peek().text == ")"):
                    args.append(self.expr())
                    while self.peek().kind == "OP" and self.peek().text == ",":
                        self.take()
                        args.append(self.expr())
                self.expect(")")
                return ("call", tok.text, args, tok.pos)
            return ("var", tok.text, tok.pos)
        if tok.kind == "OP" and tok.text == "(":
            self.take()
            node = self.expr()
            self.expect(")")
            return node
        raise self.error("unexpected token" if tok.kind != "EOF" else "unexpected end")
