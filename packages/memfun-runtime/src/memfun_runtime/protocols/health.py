from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from memfun_core.types import HealthStatus


@runtime_checkable
class HealthMonitor(Protocol):
    """Health monitoring for agents."""

    async def heartbeat(self, agent_id: str, metrics: dict) -> None: ...
    async def check(self, agent_id: str) -> HealthStatus: ...
    async def subscribe_alerts(
        self,
        callback: Callable[[HealthStatus], Awaitable[None]],
    ) -> None: ...
