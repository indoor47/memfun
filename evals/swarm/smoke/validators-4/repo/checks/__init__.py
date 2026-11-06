"""Validators.  Do not edit this dispatcher."""

from __future__ import annotations


def validate(kind: str, value: str) -> bool:
    if kind == "luhn":
        from checks.luhn import luhn_valid

        return luhn_valid(value)
    if kind == "isbn":
        from checks.isbn import isbn10_valid, isbn13_valid

        return isbn10_valid(value) or isbn13_valid(value)
    if kind == "iban":
        from checks.iban import iban_valid

        return iban_valid(value)
    if kind == "semver":
        from checks.semver import semver_valid

        return semver_valid(value)
    raise KeyError(kind)
