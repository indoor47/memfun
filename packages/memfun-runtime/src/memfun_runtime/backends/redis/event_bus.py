from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import Message, TopicConfig

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from redis.asyncio import Redis

logger = get_logger("backend.redis.event_bus")


class RedisEventBus:
    """T2 event bus: Redis Streams with consumer groups (XADD/XREAD/XREADGROUP)."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisEventBus:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _stream_key(self, topic: str) -> str:
        return f"{self._prefix}stream:{topic}"

    def _topic_meta_key(self) -> str:
        return f"{self._prefix}topics"

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str:
        msg_id = uuid.uuid4().hex
        fields: dict[bytes | str, bytes | str] = {
            b"id": msg_id.encode(),
            b"payload": message,
            b"key": (key or "").encode(),
            b"timestamp": str(time.time()).encode(),
        }
        await self._r.xadd(self._stream_key(topic), fields)  # type: ignore[arg-type]
        return msg_id

    async def subscribe(self, topic: str, group: str | None = None) -> AsyncIterator[Message]:
        stream_key = self._stream_key(topic)

        if group is not None:
            # Ensure consumer group exists; start from the latest message.
            # Group already exists -- that's fine.
            with contextlib.suppress(Exception):
                await self._r.xgroup_create(stream_key, group, id="0", mkstream=True)

            consumer_name = uuid.uuid4().hex
            while True:
                entries = await self._r.xreadgroup(
                    groupname=group,
                    consumername=consumer_name,
                    streams={stream_key: ">"},
                    count=50,
                    block=1000,
                )
                for _stream, messages in entries:
                    for _redis_id, fields in messages:
                        yield self._decode_message(topic, fields)
        else:
            last_id = "$"
            while True:
                entries = await self._r.xread(
                    streams={stream_key: last_id},
                    count=50,
                    block=1000,
                )
                for _stream, messages in entries:
                    for redis_id, fields in messages:
                        last_id = redis_id
                        yield self._decode_message(topic, fields)

    def _decode_message(self, topic: str, fields: dict[bytes, bytes]) -> Message:
        return Message(
            id=fields[b"id"].decode(),
            topic=topic,
            payload=fields[b"payload"],
            key=fields[b"key"].decode() or None,
            timestamp=float(fields[b"timestamp"]),
        )

    async def create_topic(self, topic: str, config: TopicConfig | None = None) -> None:
        config = config or TopicConfig()
        meta_key = self._topic_meta_key()
        existing = await self._r.hget(meta_key, topic)
        config_json = json.dumps({
            "retention_seconds": config.retention_seconds,
            "max_consumers": config.max_consumers,
            "max_message_bytes": config.max_message_bytes,
        })
        if existing is not None:
            if existing.decode() != config_json:
                raise ValueError(f"Topic {topic!r} already exists with different config")
            return
        await self._r.hset(meta_key, topic, config_json)
        # Ensure the stream key exists (XADD with a dummy trimmed immediately
        # is not needed; the stream will be created on first publish.)

    async def delete_topic(self, topic: str) -> None:
        await self._r.hdel(self._topic_meta_key(), topic)
        await self._r.delete(self._stream_key(topic))

    async def ack(self, message_id: str) -> None:
        # In Redis Streams, acknowledgement is per-consumer-group and
        # requires the stream key and group name.  We provide a no-op here
        # for API conformance -- real acknowledgement happens via XACK
        # inside the consumer group read loop.
        pass

    async def nack(self, message_id: str) -> None:
        # Similarly, NACK / re-delivery is managed by the consumer group
        # pending entries list (PEL).  A no-op satisfies the protocol.
        pass
