from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult


@runtime_checkable
class OrchestratorAdapter(Protocol):
    """Orchestration patterns for multi-agent task coordination."""

    async def dispatch(self, task: TaskMessage, agent_name: str) -> TaskResult:
        """Send a task directly to a named agent and return the result."""
        ...

    async def fan_out(
        self, tasks: list[TaskMessage], agent_names: list[str]
    ) -> list[TaskResult]:
        """Execute tasks in parallel across the given agents."""
        ...

    async def pipeline(
        self, task: TaskMessage, agent_names: list[str]
    ) -> TaskResult:
        """Execute a task through a sequential pipeline of agents.

        Each agent's result payload is fed as the next agent's task payload.
        """
        ...

    async def route(self, task: TaskMessage) -> TaskResult:
        """Route a task to the most appropriate agent based on capabilities."""
        ...
