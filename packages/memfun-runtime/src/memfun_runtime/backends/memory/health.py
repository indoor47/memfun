from __future__ import annotations

import time
from typing import TYPE_CHECKING

from memfun_core.types import HealthState, HealthStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class InProcessHealthMonitor:
    """T0 health monitor: in-memory heartbeat tracking."""

    def __init__(self, healthy_timeout: float = 30.0, degraded_timeout: float = 60.0) -> None:
        self._heartbeats: dict[str, float] = {}
        self._metrics: dict[str, dict[str, float]] = {}
        self._callbacks: list[Callable[[HealthStatus], Awaitable[None]]] = []
        self._healthy_timeout = healthy_timeout
        self._degraded_timeout = degraded_timeout

    async def heartbeat(self, agent_id: str, metrics: dict) -> None:
        self._heartbeats[agent_id] = time.time()
        self._metrics[agent_id] = {k: float(v) for k, v in metrics.items()}

    async def check(self, agent_id: str) -> HealthStatus:
        last_hb = self._heartbeats.get(agent_id)
        if last_hb is None:
            return HealthStatus(agent_id=agent_id, state=HealthState.UNKNOWN, last_heartbeat=0.0)

        elapsed = time.time() - last_hb
        if elapsed <= self._healthy_timeout:
            state = HealthState.HEALTHY
        elif elapsed <= self._degraded_timeout:
            state = HealthState.DEGRADED
        else:
            state = HealthState.UNHEALTHY

        return HealthStatus(
            agent_id=agent_id,
            state=state,
            last_heartbeat=last_hb,
            metrics=self._metrics.get(agent_id, {}),
        )

    async def subscribe_alerts(self, callback: Callable[[HealthStatus], Awaitable[None]]) -> None:
        self._callbacks.append(callback)
