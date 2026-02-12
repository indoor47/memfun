from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import HealthState, HealthStatus

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.health")


class NATSHealthMonitor:
    """T3 health monitor: NATS KV-based heartbeats."""

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        kv: object,
        *,
        healthy_timeout: float = 30.0,
        degraded_timeout: float = 60.0,
    ) -> None:
        self._nc = nc
        self._js = js
        self._kv = kv
        self._callbacks: list[Callable[[HealthStatus], Awaitable[None]]] = []
        self._healthy_timeout = healthy_timeout
        self._degraded_timeout = degraded_timeout

    @classmethod
    async def create(
        cls,
        nats_url: str,
        *,
        creds_file: str | None = None,
        bucket: str = "memfun_health",
        healthy_timeout: float = 30.0,
        degraded_timeout: float = 60.0,
    ) -> NATSHealthMonitor:
        nc, js = await connect(nats_url, creds_file=creds_file)
        # Bucket TTL: auto-expire entries after 2x degraded timeout.
        ttl_ns = int(degraded_timeout * 2) * 1_000_000_000
        kv = await ensure_kv_bucket(js, bucket, ttl=ttl_ns)
        return cls(
            nc, js, kv,
            healthy_timeout=healthy_timeout,
            degraded_timeout=degraded_timeout,
        )

    async def heartbeat(self, agent_id: str, metrics: dict) -> None:
        payload = json.dumps({
            "last_heartbeat": time.time(),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })
        await self._kv.put(agent_id, payload.encode())  # type: ignore[union-attr]

    async def check(self, agent_id: str) -> HealthStatus:
        try:
            entry = await self._kv.get(agent_id)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return HealthStatus(
                    agent_id=agent_id,
                    state=HealthState.UNKNOWN,
                    last_heartbeat=0.0,
                )
        except Exception:
            return HealthStatus(
                agent_id=agent_id,
                state=HealthState.UNKNOWN,
                last_heartbeat=0.0,
            )

        data = json.loads(entry.value)
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
