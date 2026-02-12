from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import AgentInfo

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket

if TYPE_CHECKING:
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.registry")


class NATSRegistry:
    """T3 registry: NATS KV for agent discovery."""

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
        bucket: str = "memfun_registry",
    ) -> NATSRegistry:
        nc, js = await connect(nats_url, creds_file=creds_file)
        kv = await ensure_kv_bucket(js, bucket)
        return cls(nc, js, kv)

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
        await self._kv.put(agent_id, data.encode())  # type: ignore[union-attr]

    async def deregister(self, agent_id: str) -> None:
        with contextlib.suppress(Exception):
            await self._kv.delete(agent_id)  # type: ignore[union-attr]

    async def discover(self, capability: str) -> list[AgentInfo]:
        results: list[AgentInfo] = []
        try:
            keys = await self._kv.keys()  # type: ignore[union-attr]
        except Exception:
            return results

        for key in keys:
            decoded_key = key if isinstance(key, str) else key.decode()
            try:
                entry = await self._kv.get(decoded_key)  # type: ignore[union-attr]
                if entry is None or entry.value is None:
                    continue
                data = json.loads(entry.value)
                if capability in data["capabilities"]:
                    results.append(self._to_info(data))
            except Exception:
                continue
        return results

    async def get(self, agent_id: str) -> AgentInfo | None:
        try:
            entry = await self._kv.get(agent_id)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return None
            return self._to_info(json.loads(entry.value))
        except Exception:
            return None

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
