from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from memfun_core.types import Message, TopicConfig


@runtime_checkable
class EventBusAdapter(Protocol):
    """Publish/subscribe messaging between agents."""

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str: ...
    async def subscribe(self, topic: str, group: str | None = None) -> AsyncIterator[Message]: ...
    async def create_topic(self, topic: str, config: TopicConfig | None = None) -> None: ...
    async def delete_topic(self, topic: str) -> None: ...
    async def ack(self, message_id: str) -> None: ...
    async def nack(self, message_id: str) -> None: ...
