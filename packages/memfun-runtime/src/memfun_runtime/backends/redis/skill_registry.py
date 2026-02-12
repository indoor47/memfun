from __future__ import annotations

import json
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_runtime.backends.redis._pool import create_pool
from memfun_runtime.protocols.skill_registry import SkillInfo

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("backend.redis.skill_registry")


class RedisSkillRegistry:
    """T2 skill registry: Redis Hash for skill discovery."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisSkillRegistry:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _hash_key(self) -> str:
        return f"{self._prefix}skills"

    async def register_skill(self, skill: SkillInfo) -> None:
        data = json.dumps({
            "name": skill.name,
            "description": skill.description,
            "source_path": skill.source_path,
            "user_invocable": skill.user_invocable,
            "model_invocable": skill.model_invocable,
            "allowed_tools": skill.allowed_tools,
            "metadata": skill.metadata,
        })
        await self._r.hset(self._hash_key(), skill.name, data)

    async def deregister_skill(self, name: str) -> None:
        await self._r.hdel(self._hash_key(), name)

    async def get_skill(self, name: str) -> SkillInfo | None:
        raw = await self._r.hget(self._hash_key(), name)
        if raw is None:
            return None
        return self._to_skill(json.loads(raw))

    async def list_skills(self) -> list[SkillInfo]:
        all_raw = await self._r.hgetall(self._hash_key())
        return [self._to_skill(json.loads(v)) for v in all_raw.values()]

    async def search_skills(self, query: str) -> list[SkillInfo]:
        query_lower = query.lower()
        results: list[SkillInfo] = []
        all_raw = await self._r.hgetall(self._hash_key())
        for v in all_raw.values():
            skill = self._to_skill(json.loads(v))
            if query_lower in skill.name.lower() or query_lower in skill.description.lower():
                results.append(skill)
        return results

    @staticmethod
    def _to_skill(data: dict) -> SkillInfo:
        return SkillInfo(
            name=data["name"],
            description=data["description"],
            source_path=data["source_path"],
            user_invocable=data.get("user_invocable", True),
            model_invocable=data.get("model_invocable", True),
            allowed_tools=data.get("allowed_tools", []),
            metadata=data.get("metadata", {}),
        )
