"""TTLStore: expiring wrapper around any KVStore."""

from __future__ import annotations

import time
from collections.abc import Callable

from kv.base import KVStore


class TTLStore:
    def __init__(
        self, inner: KVStore, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError
