from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import Message, TopicConfig

from memfun_runtime.backends.nats._connection import connect

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.event_bus")


class NATSEventBus:
    """T3 event bus: NATS JetStream for pub/sub with durable consumers."""

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        *,
        stream_prefix: str = "memfun",
    ) -> None:
        self._nc = nc
        self._js = js
        self._stream_prefix = stream_prefix
        self._topics: dict[str, TopicConfig] = {}

    @classmethod
    async def create(
        cls,
        nats_url: str,
        *,
        creds_file: str | None = None,
        stream_prefix: str = "memfun",
    ) -> NATSEventBus:
        nc, js = await connect(nats_url, creds_file=creds_file)
        return cls(nc, js, stream_prefix=stream_prefix)

    def _stream_name(self, topic: str) -> str:
        return f"{self._stream_prefix}_{topic}"

    def _subject(self, topic: str) -> str:
        return f"{self._stream_prefix}.{topic}"

    async def _ensure_stream(self, topic: str, config: TopicConfig | None = None) -> None:
        """Create the JetStream stream for a topic if it does not exist."""
        from nats.js.api import StreamConfig

        config = config or TopicConfig()
        stream_name = self._stream_name(topic)
        subject = self._subject(topic)

        try:
            await self._js.find_stream_name_by_subject(subject)
        except Exception:
            # Stream does not exist yet -- create it.
            sc = StreamConfig(
                name=stream_name,
                subjects=[subject],
                max_age=config.retention_seconds * 1_000_000_000,  # nanoseconds
                max_msg_size=config.max_message_bytes,
            )
            await self._js.add_stream(sc)

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str:
        msg_id = uuid.uuid4().hex
        headers: dict[str, str] = {"Memfun-Msg-Id": msg_id}
        if key is not None:
            headers["Memfun-Key"] = key
        headers["Memfun-Timestamp"] = str(time.time())

        await self._ensure_stream(topic)

        await self._js.publish(
            self._subject(topic),
            message,
            headers=headers,
        )
        return msg_id

    async def subscribe(self, topic: str, group: str | None = None) -> AsyncIterator[Message]:
        await self._ensure_stream(topic)
        subject = self._subject(topic)

        if group is not None:
            sub = await self._js.pull_subscribe(subject, durable=group)
            while True:
                try:
                    msgs = await sub.fetch(batch=50, timeout=1)
                except Exception:
                    # Timeout or no messages -- loop again.
                    continue
                for nats_msg in msgs:
                    yield self._decode_message(topic, nats_msg)
        else:
            sub = await self._js.subscribe(subject)
            async for nats_msg in sub.messages:
                yield self._decode_message(topic, nats_msg)

    @staticmethod
    def _decode_message(topic: str, nats_msg: object) -> Message:
        headers = {}
        raw_headers = getattr(nats_msg, "headers", None) or {}
        for k, v in raw_headers.items():
            headers[k] = v

        msg_id = headers.get("Memfun-Msg-Id", uuid.uuid4().hex)
        key = headers.get("Memfun-Key")
        ts_str = headers.get("Memfun-Timestamp")
        ts = float(ts_str) if ts_str else time.time()

        return Message(
            id=msg_id,
            topic=topic,
            payload=getattr(nats_msg, "data", b""),
            key=key,
            timestamp=ts,
        )

    async def create_topic(self, topic: str, config: TopicConfig | None = None) -> None:
        config = config or TopicConfig()
        if topic in self._topics:
            if self._topics[topic] != config:
                raise ValueError(f"Topic {topic!r} already exists with different config")
            return
        await self._ensure_stream(topic, config)
        self._topics[topic] = config

    async def delete_topic(self, topic: str) -> None:
        self._topics.pop(topic, None)
        try:
            await self._js.delete_stream(self._stream_name(topic))
        except Exception:
            logger.debug("Stream for topic %r did not exist", topic)

    async def ack(self, message_id: str) -> None:
        # NATS JetStream handles ack via the message object itself.
        # This is a no-op for protocol conformance.
        pass

    async def nack(self, message_id: str) -> None:
        # NATS JetStream handles nack/redelivery via the message object.
        pass
