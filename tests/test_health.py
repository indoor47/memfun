from __future__ import annotations

import pytest
from memfun_core.types import HealthState


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def health(request):
    backends = {
        "memory": "memory_health_monitor",
        "sqlite": "sqlite_health_monitor",
        "redis": "redis_health_monitor",
        "nats": "nats_health_monitor",
    }
    return request.getfixturevalue(backends[request.param])


class TestHealthConformance:
    """Conformance tests for HealthMonitor implementations."""

    async def test_check_unknown_agent(self, health):
        status = await health.check("unknown-agent")
        assert status.state == HealthState.UNKNOWN

    async def test_heartbeat_makes_healthy(self, health):
        await health.heartbeat("agent-1", {"cpu": 50.0})
        status = await health.check("agent-1")
        assert status.state == HealthState.HEALTHY
        assert status.metrics.get("cpu") == 50.0

    async def test_subscribe_alerts(self, health):
        alerts = []
        async def on_alert(status):
            alerts.append(status)
        await health.subscribe_alerts(on_alert)
        # Just verify it doesn't raise
