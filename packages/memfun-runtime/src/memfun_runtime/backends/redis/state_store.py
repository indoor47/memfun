from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import StateChange, StateChangeKind

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from redis.asyncio import Redis

logger = get_logger("backend.redis.state_store")


class RedisStateStore:
    """T2 state store: Redis Strings with TTL for key-value storage."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix
        self._revision = 0

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisStateStore:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _key(self, key: str) -> str:
        return f"{self._prefix}state:{key}"

    def _rev_key(self, key: str) -> str:
        return f"{self._prefix}state_rev:{key}"

    async def get(self, key: str) -> bytes | None:
        value = await self._r.get(self._key(key))
        return value  # type: ignore[return-value]

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        rk = self._key(key)
        if ttl is not None:
            await self._r.set(rk, value, ex=ttl)
            await self._r.set(self._rev_key(key), b"0", ex=ttl)
        else:
            await self._r.set(rk, value)
            await self._r.set(self._rev_key(key), b"0")
        await self._r.incr(self._rev_key(key))

    async def delete(self, key: str) -> None:
        await self._r.delete(self._key(key))
        await self._r.delete(self._rev_key(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._r.exists(self._key(key)))

    async def list_keys(self, prefix: str) -> AsyncIterator[str]:
        full_prefix = self._key(prefix)
        cursor: int | bytes = 0
        while True:
            cursor, keys = await self._r.scan(
                cursor=cursor, match=f"{full_prefix}*", count=100,
            )
            state_prefix = f"{self._prefix}state:"
            for key in keys:
                decoded = key.decode() if isinstance(key, bytes) else key
                # Strip the internal prefix to return the logical key.
                yield decoded[len(state_prefix):]
            if cursor == 0:
                break

    async def watch(self, key: str) -> AsyncIterator[StateChange]:
        """Poll-based watch: checks for value changes every 0.5 s."""
        last_rev = 0
        last_existed = False
        while True:
            rev_raw = await self._r.get(self._rev_key(key))
            current_rev = int(rev_raw) if rev_raw else 0
            value = await self._r.get(self._key(key))
            exists_now = value is not None

            if exists_now and current_rev > last_rev:
                last_rev = current_rev
                last_existed = True
                yield StateChange(
                    key=key,
                    kind=StateChangeKind.PUT,
                    value=value,
                    revision=current_rev,
                )
            elif not exists_now and last_existed:
                last_existed = False
                last_rev = 0
                yield StateChange(
                    key=key,
                    kind=StateChangeKind.DELETE,
                    value=None,
                    revision=0,
                )

            await asyncio.sleep(0.5)
