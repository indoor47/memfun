from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memfun_core.types import AgentInfo, AgentStatus


@runtime_checkable
class AgentLifecycle(Protocol):
    """Agent process management."""

    async def start(self, agent_id: str) -> None: ...
    async def stop(self, agent_id: str) -> None: ...
    async def restart(self, agent_id: str) -> None: ...
    async def status(self, agent_id: str) -> AgentStatus: ...
    async def list_agents(self) -> list[AgentInfo]: ...
