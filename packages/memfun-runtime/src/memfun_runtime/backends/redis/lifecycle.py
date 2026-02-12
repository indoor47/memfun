from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from memfun_core.errors import AgentAlreadyRunningError
from memfun_core.logging import get_logger
from memfun_core.types import AgentInfo, AgentStatus, AgentStatusKind

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("backend.redis.lifecycle")


class RedisLifecycle:
    """T2 lifecycle: Redis-backed agent lifecycle state."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisLifecycle:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _hash_key(self) -> str:
        return f"{self._prefix}lifecycle"

    async def start(self, agent_id: str) -> None:
        raw = await self._r.hget(self._hash_key(), agent_id)
        if raw is not None:
            data = json.loads(raw)
            if data.get("status") == "running":
                raise AgentAlreadyRunningError(f"Agent {agent_id!r} is already running")

        now = time.time()
        payload = json.dumps({
            "status": "running",
            "uptime_start": now,
            "last_heartbeat": now,
            "error": None,
        })
        await self._r.hset(self._hash_key(), agent_id, payload)

    async def stop(self, agent_id: str) -> None:
        payload = json.dumps({
            "status": "stopped",
            "uptime_start": None,
            "last_heartbeat": time.time(),
            "error": None,
        })
        await self._r.hset(self._hash_key(), agent_id, payload)

    async def restart(self, agent_id: str) -> None:
        await self.stop(agent_id)
        await self.start(agent_id)

    async def status(self, agent_id: str) -> AgentStatus:
        raw = await self._r.hget(self._hash_key(), agent_id)
        if raw is None:
            return AgentStatus(
                agent_id=agent_id,
                status=AgentStatusKind.UNKNOWN,
                uptime_seconds=0.0,
                last_heartbeat=0.0,
            )
        data = json.loads(raw)
        uptime_start = data.get("uptime_start")
        is_running = uptime_start and data["status"] == "running"
        uptime = (time.time() - uptime_start) if is_running else 0.0
        return AgentStatus(
            agent_id=agent_id,
            status=AgentStatusKind(data["status"]),
            uptime_seconds=uptime,
            last_heartbeat=data["last_heartbeat"],
            error=data.get("error"),
        )

    async def list_agents(self) -> list[AgentInfo]:
        results: list[AgentInfo] = []
        cursor: int | bytes = 0
        while True:
            cursor, entries = await self._r.hscan(self._hash_key(), cursor=cursor, count=100)
            for field, _raw in entries.items():
                agent_id = field.decode() if isinstance(field, bytes) else field
                results.append(
                    AgentInfo(
                        agent_id=agent_id,
                        name=agent_id,
                        version="unknown",
                        capabilities=[],
                    )
                )
            if cursor == 0:
                break
        return results
