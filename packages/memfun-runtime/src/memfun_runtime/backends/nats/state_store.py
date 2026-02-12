from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import StateChange, StateChangeKind

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.state_store")


class NATSStateStore:
    """T3 state store: NATS KV Store for state."""

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        kv: object,
    ) -> None:
        self._nc = nc
        self._js = js
        self._kv = kv

    @classmethod
    async def create(
        cls,
        nats_url: str,
        *,
        creds_file: str | None = None,
        bucket: str = "memfun_state",
    ) -> NATSStateStore:
        nc, js = await connect(nats_url, creds_file=creds_file)
        kv = await ensure_kv_bucket(js, bucket)
        return cls(nc, js, kv)

    async def get(self, key: str) -> bytes | None:
        try:
            entry = await self._kv.get(key)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return None
            return bytes(entry.value)
        except Exception:
            return None

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        await self._kv.put(key, value)  # type: ignore[union-attr]
        # NATS KV bucket-level TTL is set at creation time.  Per-key TTL
        # is not natively supported in all versions, so we store the value
        # and rely on bucket-level TTL or an external cleanup if needed.
        # For per-key TTL we could use a separate bucket, but that adds
        # complexity.  We accept the bucket-level TTL for now.

    async def delete(self, key: str) -> None:
        with contextlib.suppress(Exception):
            await self._kv.delete(key)  # type: ignore[union-attr]

    async def exists(self, key: str) -> bool:
        try:
            entry = await self._kv.get(key)  # type: ignore[union-attr]
            return entry is not None and entry.value is not None
        except Exception:
            return False

    async def list_keys(self, prefix: str) -> AsyncIterator[str]:
        try:
            keys = await self._kv.keys()  # type: ignore[union-attr]
        except Exception:
            return
        for key in keys:
            decoded = key if isinstance(key, str) else key.decode()
            if decoded.startswith(prefix):
                yield decoded

    async def watch(self, key: str) -> AsyncIterator[StateChange]:
        """Watch a key for changes using NATS KV watcher."""
        try:
            watcher = await self._kv.watch(key)  # type: ignore[union-attr]
            async for entry in watcher:
                if entry is None:
                    continue
                if entry.operation is not None and str(entry.operation).endswith("DELETE"):
                    yield StateChange(
                        key=key,
                        kind=StateChangeKind.DELETE,
                        value=None,
                        revision=entry.revision if hasattr(entry, "revision") else 0,
                    )
                elif entry.value is not None:
                    yield StateChange(
                        key=key,
                        kind=StateChangeKind.PUT,
                        value=bytes(entry.value),
                        revision=entry.revision if hasattr(entry, "revision") else 0,
                    )
        except Exception:
            logger.debug("Watch for key %r ended", key)
