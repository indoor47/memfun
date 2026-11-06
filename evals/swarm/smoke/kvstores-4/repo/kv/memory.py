"""MemoryStore: see task spec."""

from __future__ import annotations


class MemoryStore:
    def __init__(self) -> None:
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError
