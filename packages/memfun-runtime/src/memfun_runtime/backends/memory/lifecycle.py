from __future__ import annotations

import time

from memfun_core.errors import AgentAlreadyRunningError
from memfun_core.types import AgentInfo, AgentStatus, AgentStatusKind


class InProcessLifecycle:
    """T0 lifecycle: tracks agent status in-memory."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentStatus] = {}
        self._info: dict[str, AgentInfo] = {}

    async def start(self, agent_id: str) -> None:
        if agent_id in self._agents and self._agents[agent_id].status == AgentStatusKind.RUNNING:
            raise AgentAlreadyRunningError(f"Agent {agent_id!r} is already running")
        self._agents[agent_id] = AgentStatus(
            agent_id=agent_id,
            status=AgentStatusKind.RUNNING,
            uptime_seconds=0.0,
            last_heartbeat=time.time(),
        )

    async def stop(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id] = AgentStatus(
                agent_id=agent_id,
                status=AgentStatusKind.STOPPED,
                uptime_seconds=0.0,
                last_heartbeat=time.time(),
            )

    async def restart(self, agent_id: str) -> None:
        await self.stop(agent_id)
        await self.start(agent_id)

    async def status(self, agent_id: str) -> AgentStatus:
        if agent_id not in self._agents:
            return AgentStatus(
                agent_id=agent_id,
                status=AgentStatusKind.UNKNOWN,
                uptime_seconds=0.0,
                last_heartbeat=0.0,
            )
        return self._agents[agent_id]

    async def list_agents(self) -> list[AgentInfo]:
        return list(self._info.values())
