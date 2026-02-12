from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

from memfun_core.types import Message, TopicConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class InProcessEventBus:
    """T0 event bus: asyncio.Queue per topic, dict of subscriber queues."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)
        self._topics: dict[str, TopicConfig] = {}
        self._groups: dict[str, dict[str, asyncio.Queue[Message]]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str:
        msg_id = uuid.uuid4().hex
        msg = Message(id=msg_id, topic=topic, payload=message, key=key, timestamp=time.time())
        async with self._lock:
            for queue in self._subscribers.get(topic, []):
                await queue.put(msg)
            for group_topics in self._groups.values():
                if topic in group_topics:
                    await group_topics[topic].put(msg)
        return msg_id

    async def subscribe(self, topic: str, group: str | None = None) -> AsyncIterator[Message]:
        if group is not None:
            async with self._lock:
                if topic not in self._groups[group]:
                    self._groups[group][topic] = asyncio.Queue()
                queue = self._groups[group][topic]
        else:
            queue: asyncio.Queue[Message] = asyncio.Queue()
            async with self._lock:
                self._subscribers[topic].append(queue)
        try:
            while True:
                msg = await queue.get()
                yield msg
        finally:
            async with self._lock:
                if group is None and queue in self._subscribers.get(topic, []):
                    self._subscribers[topic].remove(queue)

    async def create_topic(self, topic: str, config: TopicConfig | None = None) -> None:
        config = config or TopicConfig()
        if topic in self._topics:
            if self._topics[topic] != config:
                raise ValueError(f"Topic {topic!r} already exists with different config")
            return
        self._topics[topic] = config

    async def delete_topic(self, topic: str) -> None:
        self._topics.pop(topic, None)
        self._subscribers.pop(topic, None)

    async def ack(self, message_id: str) -> None:
        pass

    async def nack(self, message_id: str) -> None:
        pass
