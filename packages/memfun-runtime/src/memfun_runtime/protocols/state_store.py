from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from memfun_core.types import StateChange


@runtime_checkable
class StateStoreAdapter(Protocol):
    """Key-value state storage for agents."""

    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list_keys(self, prefix: str) -> AsyncIterator[str]: ...
    async def watch(self, key: str) -> AsyncIterator[StateChange]: ...
