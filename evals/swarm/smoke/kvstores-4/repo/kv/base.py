"""The KVStore contract shared by every store.  Do not edit."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class KVStore(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> bool: ...

    def keys(self) -> list[str]: ...


def check_key(key: object) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError(f"invalid key: {key!r}")
    return key


def check_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid value: {value!r}")
    return value
