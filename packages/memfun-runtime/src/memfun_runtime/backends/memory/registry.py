from __future__ import annotations

from memfun_core.types import AgentInfo


class InProcessRegistry:
    """T0 registry: in-memory agent discovery."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    async def register(self, agent_id: str, capabilities: list[str], metadata: dict) -> None:
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            name=metadata.get("name", agent_id),
            version=metadata.get("version", "0.0.0"),
            capabilities=capabilities,
            metadata={k: str(v) for k, v in metadata.items()},
        )

    async def deregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    async def discover(self, capability: str) -> list[AgentInfo]:
        return [
            info for info in self._agents.values()
            if capability in info.capabilities
        ]

    async def get(self, agent_id: str) -> AgentInfo | None:
        return self._agents.get(agent_id)
