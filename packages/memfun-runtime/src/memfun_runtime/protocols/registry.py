from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memfun_core.types import AgentInfo


@runtime_checkable
class RegistryAdapter(Protocol):
    """Service discovery for agents."""

    async def register(self, agent_id: str, capabilities: list[str], metadata: dict) -> None: ...
    async def deregister(self, agent_id: str) -> None: ...
    async def discover(self, capability: str) -> list[AgentInfo]: ...
    async def get(self, agent_id: str) -> AgentInfo | None: ...
