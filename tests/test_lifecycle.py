from __future__ import annotations

import pytest
from memfun_core.errors import AgentAlreadyRunningError
from memfun_core.types import AgentStatusKind


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def lifecycle(request):
    backends = {
        "memory": "memory_lifecycle",
        "sqlite": "sqlite_lifecycle",
        "redis": "redis_lifecycle",
        "nats": "nats_lifecycle",
    }
    return request.getfixturevalue(backends[request.param])


class TestLifecycleConformance:
    """Conformance tests for AgentLifecycle implementations."""

    async def test_start_agent(self, lifecycle):
        await lifecycle.start("agent-1")
        status = await lifecycle.status("agent-1")
        assert status.status == AgentStatusKind.RUNNING

    async def test_start_already_running_raises(self, lifecycle):
        await lifecycle.start("agent-1")
        with pytest.raises(AgentAlreadyRunningError):
            await lifecycle.start("agent-1")

    async def test_stop_agent(self, lifecycle):
        await lifecycle.start("agent-1")
        await lifecycle.stop("agent-1")
        status = await lifecycle.status("agent-1")
        assert status.status == AgentStatusKind.STOPPED

    async def test_stop_nonexistent_noop(self, lifecycle):
        await lifecycle.stop("nonexistent")  # Should not raise

    async def test_restart_agent(self, lifecycle):
        await lifecycle.start("agent-1")
        await lifecycle.restart("agent-1")
        status = await lifecycle.status("agent-1")
        assert status.status == AgentStatusKind.RUNNING

    async def test_status_unknown_agent(self, lifecycle):
        status = await lifecycle.status("unknown")
        assert status.status == AgentStatusKind.UNKNOWN
