"""MemoryStore: see task spec."""

from __future__ import annotations

from kv.base import check_key, check_value


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(check_key(key))

    def set(self, key: str, value: str) -> None:
        self._data[check_key(key)] = check_value(value)

    def delete(self, key: str) -> bool:
        return self._data.pop(check_key(key), None) is not None

    def keys(self) -> list[str]:
        return sorted(self._data)
