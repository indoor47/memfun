from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from memfun_core.errors import AgentAlreadyRunningError
from memfun_core.logging import get_logger
from memfun_core.types import AgentInfo, AgentStatus, AgentStatusKind

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket

if TYPE_CHECKING:
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.lifecycle")


class NATSLifecycle:
    """T3 lifecycle: NATS KV-backed agent lifecycle state."""

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
        bucket: str = "memfun_lifecycle",
    ) -> NATSLifecycle:
        nc, js = await connect(nats_url, creds_file=creds_file)
        kv = await ensure_kv_bucket(js, bucket)
        return cls(nc, js, kv)

    async def start(self, agent_id: str) -> None:
        try:
            entry = await self._kv.get(agent_id)  # type: ignore[union-attr]
            if entry is not None and entry.value is not None:
                data = json.loads(entry.value)
                if data.get("status") == "running":
                    raise AgentAlreadyRunningError(f"Agent {agent_id!r} is already running")
        except AgentAlreadyRunningError:
            raise
        except Exception:
            pass

        now = time.time()
        payload = json.dumps({
            "status": "running",
            "uptime_start": now,
            "last_heartbeat": now,
            "error": None,
        })
        await self._kv.put(agent_id, payload.encode())  # type: ignore[union-attr]

    async def stop(self, agent_id: str) -> None:
        payload = json.dumps({
            "status": "stopped",
            "uptime_start": None,
            "last_heartbeat": time.time(),
            "error": None,
        })
        await self._kv.put(agent_id, payload.encode())  # type: ignore[union-attr]

    async def restart(self, agent_id: str) -> None:
        await self.stop(agent_id)
        await self.start(agent_id)

    async def status(self, agent_id: str) -> AgentStatus:
        try:
            entry = await self._kv.get(agent_id)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return AgentStatus(
                    agent_id=agent_id,
                    status=AgentStatusKind.UNKNOWN,
                    uptime_seconds=0.0,
                    last_heartbeat=0.0,
                )
        except Exception:
            return AgentStatus(
                agent_id=agent_id,
                status=AgentStatusKind.UNKNOWN,
                uptime_seconds=0.0,
                last_heartbeat=0.0,
            )

        data = json.loads(entry.value)
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
        try:
            keys = await self._kv.keys()  # type: ignore[union-attr]
        except Exception:
            return results

        for key in keys:
            agent_id = key if isinstance(key, str) else key.decode()
            results.append(
                AgentInfo(
                    agent_id=agent_id,
                    name=agent_id,
                    version="unknown",
                    capabilities=[],
                )
            )
        return results
