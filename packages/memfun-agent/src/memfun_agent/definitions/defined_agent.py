"""DefinedAgent: wraps an AgentDefinition into a runnable agent."""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_runtime.agent import BaseAgent

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult
    from memfun_runtime.context import RuntimeContext
    from memfun_runtime.protocols.orchestrator import OrchestratorAdapter

    from memfun_agent.definitions.types import AgentDefinition

logger = get_logger("agent.defined")


class DefinedAgent(BaseAgent):
    """An agent constructed from an AGENT.md definition.

    Wraps an :class:`AgentDefinition` into a fully runnable
    :class:`BaseAgent` subclass.  The ``handle`` method builds a prompt
    from the definition's instructions combined with the incoming task
    payload, and respects the ``allowed-tools``, ``delegates-to``,
    ``model``, and ``max-turns`` settings from the definition.
    """

    def __init__(
        self,
        context: RuntimeContext,
        definition: AgentDefinition,
        orchestrator: OrchestratorAdapter | None = None,
    ) -> None:
        super().__init__(context)
        self._definition = definition
        self._orchestrator = orchestrator

    @property
    def definition(self) -> AgentDefinition:
        """The underlying agent definition."""
        return self._definition

    @property
    def agent_id(self) -> str:
        return self._definition.name

    @property
    def version(self) -> str:
        return self._definition.version

    async def handle(self, task: TaskMessage) -> TaskResult:
        """Process a task using the agent definition's instructions.

        Builds a prompt from the definition instructions and task payload,
        then returns the result.  Delegation to sub-agents is performed
        via the orchestrator when ``delegates-to`` is configured.
        """
        from memfun_core.types import TaskResult as TaskRes

        start = time.monotonic()

        prompt = self._build_prompt(task)

        # Build the result payload with all definition metadata
        result_payload: dict[str, object] = {
            "prompt": prompt,
            "model": self._definition.model,
            "max_turns": self._definition.max_turns,
            "allowed_tools": list(self._definition.allowed_tools),
            "delegates_to": list(self._definition.delegates_to),
        }

        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            "DefinedAgent '%s' handled task %s (%.1fms)",
            self._definition.name,
            task.task_id,
            duration_ms,
        )

        return TaskRes(
            task_id=task.task_id,
            agent_id=self._definition.name,
            success=True,
            result=result_payload,
            duration_ms=duration_ms,
        )

    async def delegate(self, agent_name: str, payload: dict[str, object]) -> TaskResult:
        """Delegate a subtask to another agent by name.

        Requires an orchestrator to be configured.  The target agent must
        be listed in the definition's ``delegates-to`` field.

        Args:
            agent_name: Name of the agent to delegate to.
            payload: Task payload for the delegated agent.

        Returns:
            The TaskResult from the delegated agent.

        Raises:
            RuntimeError: If no orchestrator is configured or the target
                agent is not in the ``delegates-to`` list.
        """
        from memfun_core.types import TaskMessage as TaskMsg

        if self._orchestrator is None:
            msg = (
                f"Agent '{self._definition.name}' cannot delegate: "
                f"no orchestrator configured"
            )
            raise RuntimeError(msg)

        if agent_name not in self._definition.delegates_to:
            msg = (
                f"Agent '{self._definition.name}' is not allowed to delegate to "
                f"'{agent_name}'. Allowed delegates: {self._definition.delegates_to}"
            )
            raise RuntimeError(msg)

        subtask = TaskMsg(
            task_id=uuid.uuid4().hex,
            agent_id=agent_name,
            payload=payload,
            parent_task_id=None,
        )

        return await self._orchestrator.dispatch(subtask, agent_name)

    def _build_prompt(self, task: TaskMessage) -> str:
        """Build a prompt string from the definition instructions and task payload."""
        parts: list[str] = []

        if self._definition.instructions:
            parts.append(self._definition.instructions)

        parts.append("\n## Task\n")

        # Include all payload fields
        for key, value in task.payload.items():
            parts.append(f"**{key}**: {value}\n")

        return "\n".join(parts)
