from __future__ import annotations

import json
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import AgentInfo

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("backend.redis.registry")


class RedisRegistry:
    """T2 registry: Redis Hash for agent discovery (HSET/HGET/HDEL/HSCAN)."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisRegistry:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _hash_key(self) -> str:
        return f"{self._prefix}registry"

    async def register(self, agent_id: str, capabilities: list[str], metadata: dict) -> None:
        info = AgentInfo(
            agent_id=agent_id,
            name=metadata.get("name", agent_id),
            version=metadata.get("version", "0.0.0"),
            capabilities=capabilities,
            metadata={k: str(v) for k, v in metadata.items()},
        )
        data = json.dumps({
            "agent_id": info.agent_id,
            "name": info.name,
            "version": info.version,
            "capabilities": info.capabilities,
            "metadata": info.metadata,
            "endpoint": info.endpoint,
        })
        await self._r.hset(self._hash_key(), agent_id, data)

    async def deregister(self, agent_id: str) -> None:
        await self._r.hdel(self._hash_key(), agent_id)

    async def discover(self, capability: str) -> list[AgentInfo]:
        results: list[AgentInfo] = []
        cursor: int | bytes = 0
        while True:
            cursor, entries = await self._r.hscan(self._hash_key(), cursor=cursor, count=100)
            for _field, raw in entries.items():
                data = json.loads(raw)
                if capability in data["capabilities"]:
                    results.append(self._to_info(data))
            if cursor == 0:
                break
        return results

    async def get(self, agent_id: str) -> AgentInfo | None:
        raw = await self._r.hget(self._hash_key(), agent_id)
        if raw is None:
            return None
        return self._to_info(json.loads(raw))

    @staticmethod
    def _to_info(data: dict) -> AgentInfo:
        return AgentInfo(
            agent_id=data["agent_id"],
            name=data["name"],
            version=data["version"],
            capabilities=data["capabilities"],
            metadata=data.get("metadata", {}),
            endpoint=data.get("endpoint"),
        )
