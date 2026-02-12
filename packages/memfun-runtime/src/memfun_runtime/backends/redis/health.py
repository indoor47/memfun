from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import HealthState, HealthStatus

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from redis.asyncio import Redis

logger = get_logger("backend.redis.health")


class RedisHealthMonitor:
    """T2 health monitor: Redis key-based heartbeats with TTL."""

    def __init__(
        self,
        client: Redis,
        *,
        prefix: str = "memfun:",
        healthy_timeout: float = 30.0,
        degraded_timeout: float = 60.0,
    ) -> None:
        self._r = client
        self._prefix = prefix
        self._callbacks: list[Callable[[HealthStatus], Awaitable[None]]] = []
        self._healthy_timeout = healthy_timeout
        self._degraded_timeout = degraded_timeout

    @classmethod
    async def create(
        cls,
        redis_url: str,
        *,
        prefix: str = "memfun:",
        healthy_timeout: float = 30.0,
        degraded_timeout: float = 60.0,
    ) -> RedisHealthMonitor:
        client = await create_pool(redis_url)
        return cls(
            client,
            prefix=prefix,
            healthy_timeout=healthy_timeout,
            degraded_timeout=degraded_timeout,
        )

    def _hb_key(self, agent_id: str) -> str:
        return f"{self._prefix}health:{agent_id}"

    async def heartbeat(self, agent_id: str, metrics: dict) -> None:
        payload = json.dumps({
            "last_heartbeat": time.time(),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })
        # Set with TTL slightly beyond the degraded timeout so stale keys
        # are automatically cleaned up.
        ttl = int(self._degraded_timeout * 2)
        await self._r.set(self._hb_key(agent_id), payload.encode(), ex=ttl)

    async def check(self, agent_id: str) -> HealthStatus:
        raw = await self._r.get(self._hb_key(agent_id))
        if raw is None:
            return HealthStatus(
                agent_id=agent_id,
                state=HealthState.UNKNOWN,
                last_heartbeat=0.0,
            )

        data = json.loads(raw)
        last_hb = data["last_heartbeat"]
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
            metrics=data.get("metrics", {}),
        )

    async def subscribe_alerts(
        self, callback: Callable[[HealthStatus], Awaitable[None]],
    ) -> None:
        self._callbacks.append(callback)
