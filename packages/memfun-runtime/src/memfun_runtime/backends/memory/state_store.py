from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from memfun_core.types import StateChange, StateChangeKind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class InProcessStateStore:
    """T0 state store: Python dict with asyncio-based TTL expiration."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._expiry: dict[str, float] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._watchers: dict[str, list[asyncio.Queue[StateChange]]] = {}
        self._revision: int = 0

    def _is_expired(self, key: str) -> bool:
        if key in self._expiry and time.time() >= self._expiry[key]:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def _schedule_expiry(self, key: str, ttl: int) -> None:
        if key in self._timers:
            self._timers[key].cancel()
        loop = asyncio.get_running_loop()
        self._expiry[key] = time.time() + ttl
        self._timers[key] = loop.call_later(ttl, self._expire_key, key)

    def _expire_key(self, key: str) -> None:
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        self._timers.pop(key, None)
        self._revision += 1
        self._notify_watchers(key, StateChangeKind.DELETE, None)

    def _notify_watchers(self, key: str, kind: StateChangeKind, value: bytes | None) -> None:
        for queue in self._watchers.get(key, []):
            change = StateChange(key=key, kind=kind, value=value, revision=self._revision)
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(change)

    async def get(self, key: str) -> bytes | None:
        if self._is_expired(key):
            return None
        return self._data.get(key)

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        self._data[key] = value
        self._revision += 1
        if ttl is not None:
            self._schedule_expiry(key, ttl)
        elif key in self._expiry:
            if key in self._timers:
                self._timers[key].cancel()
            self._expiry.pop(key, None)
            self._timers.pop(key, None)
        self._notify_watchers(key, StateChangeKind.PUT, value)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
        if key in self._timers:
            self._timers[key].cancel()
        self._expiry.pop(key, None)
        self._timers.pop(key, None)
        self._revision += 1
        self._notify_watchers(key, StateChangeKind.DELETE, None)

    async def exists(self, key: str) -> bool:
        if self._is_expired(key):
            return False
        return key in self._data

    async def list_keys(self, prefix: str) -> AsyncIterator[str]:
        for key in list(self._data.keys()):
            if key.startswith(prefix) and not self._is_expired(key):
                yield key

    async def watch(self, key: str) -> AsyncIterator[StateChange]:
        queue: asyncio.Queue[StateChange] = asyncio.Queue(maxsize=100)
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(queue)
        try:
            while True:
                change = await queue.get()
                yield change
        finally:
            self._watchers[key].remove(queue)
            if not self._watchers[key]:
                del self._watchers[key]
