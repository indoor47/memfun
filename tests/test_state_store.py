from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def state_store(request):
    backends = {
        "memory": "memory_state_store",
        "sqlite": "sqlite_state_store",
        "redis": "redis_state_store",
        "nats": "nats_state_store",
    }
    return request.getfixturevalue(backends[request.param])


class TestStateStoreConformance:
    """Conformance tests for StateStoreAdapter implementations."""

    async def test_get_nonexistent_returns_none(self, state_store):
        result = await state_store.get("nonexistent-key")
        assert result is None

    async def test_set_and_get(self, state_store):
        await state_store.set("key1", b"value1")
        result = await state_store.get("key1")
        assert result == b"value1"

    async def test_set_overwrites(self, state_store):
        await state_store.set("key1", b"old")
        await state_store.set("key1", b"new")
        result = await state_store.get("key1")
        assert result == b"new"

    async def test_delete(self, state_store):
        await state_store.set("key1", b"value")
        await state_store.delete("key1")
        result = await state_store.get("key1")
        assert result is None

    async def test_delete_nonexistent(self, state_store):
        await state_store.delete("nonexistent")  # Should not raise

    async def test_exists(self, state_store):
        assert not await state_store.exists("key1")
        await state_store.set("key1", b"value")
        assert await state_store.exists("key1")

    async def test_list_keys(self, state_store):
        await state_store.set("prefix:a", b"1")
        await state_store.set("prefix:b", b"2")
        await state_store.set("other:c", b"3")

        keys = []
        async for key in state_store.list_keys("prefix:"):
            keys.append(key)
        assert sorted(keys) == ["prefix:a", "prefix:b"]

    async def test_ttl_expiration(self, state_store):
        await state_store.set("ttl-key", b"value", ttl=1)
        assert await state_store.get("ttl-key") == b"value"
        await asyncio.sleep(1.5)
        assert await state_store.get("ttl-key") is None
