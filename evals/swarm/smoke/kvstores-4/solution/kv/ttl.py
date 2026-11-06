"""TTLStore: expiring wrapper around any KVStore."""

from __future__ import annotations

import time
from collections.abc import Callable

from kv.base import KVStore, check_key


class TTLStore:
    def __init__(
        self, inner: KVStore, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._inner = inner
        self._ttl = ttl_seconds
        self._clock = clock
        self._expires: dict[str, float] = {}

    def _expired(self, key: str) -> bool:
        deadline = self._expires.get(key)
        return deadline is not None and self._clock() >= deadline

    def _purge(self, key: str) -> None:
        if self._expired(key):
            self._inner.delete(key)
            del self._expires[key]

    def get(self, key: str) -> str | None:
        self._purge(check_key(key))
        return self._inner.get(key)

    def set(self, key: str, value: str) -> None:
        self._inner.set(key, value)
        self._expires[key] = self._clock() + self._ttl

    def delete(self, key: str) -> bool:
        self._purge(check_key(key))
        self._expires.pop(key, None)
        return self._inner.delete(key)

    def keys(self) -> list[str]:
        for key in list(self._expires):
            self._purge(key)
        return self._inner.keys()
