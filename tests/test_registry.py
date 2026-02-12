from __future__ import annotations

import pytest


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def registry(request):
    backends = {
        "memory": "memory_registry",
        "sqlite": "sqlite_registry",
        "redis": "redis_registry",
        "nats": "nats_registry",
    }
    return request.getfixturevalue(backends[request.param])


class TestRegistryConformance:
    """Conformance tests for RegistryAdapter implementations."""

    async def test_register_and_get(self, registry):
        await registry.register("agent-1", ["code-review"], {"name": "Reviewer", "version": "1.0"})
        info = await registry.get("agent-1")
        assert info is not None
        assert info.agent_id == "agent-1"
        assert "code-review" in info.capabilities

    async def test_get_nonexistent(self, registry):
        assert await registry.get("nonexistent") is None

    async def test_deregister(self, registry):
        await registry.register("agent-1", ["test"], {"name": "Test"})
        await registry.deregister("agent-1")
        assert await registry.get("agent-1") is None

    async def test_deregister_nonexistent(self, registry):
        await registry.deregister("nonexistent")  # Should not raise

    async def test_discover_by_capability(self, registry):
        await registry.register("agent-1", ["code-review", "testing"], {"name": "A1"})
        await registry.register("agent-2", ["testing"], {"name": "A2"})
        await registry.register("agent-3", ["deploy"], {"name": "A3"})

        testers = await registry.discover("testing")
        assert len(testers) == 2
        ids = {a.agent_id for a in testers}
        assert ids == {"agent-1", "agent-2"}

    async def test_discover_no_matches(self, registry):
        result = await registry.discover("nonexistent-capability")
        assert result == []

    async def test_register_idempotent_updates(self, registry):
        await registry.register("agent-1", ["v1"], {"name": "Old"})
        await registry.register("agent-1", ["v2"], {"name": "New"})
        info = await registry.get("agent-1")
        assert info is not None
        assert info.capabilities == ["v2"]
