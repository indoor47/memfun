from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket
from memfun_runtime.protocols.skill_registry import SkillInfo

if TYPE_CHECKING:
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.skill_registry")


class NATSSkillRegistry:
    """T3 skill registry: NATS KV for skill discovery."""

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        kv: object,
    ) -> None:
        self._nc = nc
        self._js = js
        self._kv = kv

    @classmethod
    async def create(
        cls,
        nats_url: str,
        *,
        creds_file: str | None = None,
        bucket: str = "memfun_skills",
    ) -> NATSSkillRegistry:
        nc, js = await connect(nats_url, creds_file=creds_file)
        kv = await ensure_kv_bucket(js, bucket)
        return cls(nc, js, kv)

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
        await self._kv.put(skill.name, data.encode())  # type: ignore[union-attr]

    async def deregister_skill(self, name: str) -> None:
        with contextlib.suppress(Exception):
            await self._kv.delete(name)  # type: ignore[union-attr]

    async def get_skill(self, name: str) -> SkillInfo | None:
        try:
            entry = await self._kv.get(name)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return None
            return self._to_skill(json.loads(entry.value))
        except Exception:
            return None

    async def list_skills(self) -> list[SkillInfo]:
        results: list[SkillInfo] = []
        try:
            keys = await self._kv.keys()  # type: ignore[union-attr]
        except Exception:
            return results

        for key in keys:
            decoded_key = key if isinstance(key, str) else key.decode()
            try:
                entry = await self._kv.get(decoded_key)  # type: ignore[union-attr]
                if entry is not None and entry.value is not None:
                    results.append(self._to_skill(json.loads(entry.value)))
            except Exception:
                continue
        return results

    async def search_skills(self, query: str) -> list[SkillInfo]:
        query_lower = query.lower()
        all_skills = await self.list_skills()
        return [
            skill for skill in all_skills
            if query_lower in skill.name.lower() or query_lower in skill.description.lower()
        ]

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
