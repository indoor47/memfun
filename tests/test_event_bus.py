from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def event_bus(request):
    backends = {
        "memory": "memory_event_bus",
        "sqlite": "sqlite_event_bus",
        "redis": "redis_event_bus",
        "nats": "nats_event_bus",
    }
    return request.getfixturevalue(backends[request.param])


class TestEventBusConformance:
    """Conformance tests for EventBusAdapter implementations."""

    async def test_publish_returns_message_id(self, event_bus):
        msg_id = await event_bus.publish("test-topic", b"hello")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    async def test_publish_unique_ids(self, event_bus):
        id1 = await event_bus.publish("test-topic", b"msg1")
        id2 = await event_bus.publish("test-topic", b"msg2")
        assert id1 != id2

    async def test_subscribe_receives_published_messages(self, event_bus):
        received = []

        async def consume():
            async for msg in event_bus.subscribe("test-topic"):
                received.append(msg)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)

        await event_bus.publish("test-topic", b"msg1")
        await event_bus.publish("test-topic", b"msg2")

        await asyncio.wait_for(task, timeout=5.0)
        assert len(received) == 2
        assert received[0].payload == b"msg1"
        assert received[1].payload == b"msg2"

    async def test_create_topic_idempotent(self, event_bus):
        from memfun_core.types import TopicConfig
        config = TopicConfig()
        await event_bus.create_topic("my-topic", config)
        await event_bus.create_topic("my-topic", config)  # Should not raise

    async def test_create_topic_conflict_raises(self, event_bus):
        from memfun_core.types import TopicConfig
        await event_bus.create_topic("conflict-topic", TopicConfig(retention_seconds=100))
        with pytest.raises(ValueError):
            await event_bus.create_topic("conflict-topic", TopicConfig(retention_seconds=200))

    async def test_delete_topic_idempotent(self, event_bus):
        await event_bus.delete_topic("nonexistent")  # Should not raise

    async def test_ack_noop(self, event_bus):
        await event_bus.ack("fake-id")  # Should not raise

    async def test_nack_noop(self, event_bus):
        await event_bus.nack("fake-id")  # Should not raise
