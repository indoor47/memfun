from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from memfun_core.types import TaskMessage, TaskResult

    from memfun_runtime.context import RuntimeContext

logger = get_logger("agent")

# Global registry of agent classes
_AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


class BaseAgent(ABC):
    """Base class for all Memfun agents.

    Subclasses must implement the `handle` method. The agent receives
    a RuntimeContext at initialization that provides backend-agnostic
    access to all infrastructure services.
    """

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @property
    def context(self) -> RuntimeContext:
        return self._context

    @property
    def agent_id(self) -> str:
        meta = getattr(self.__class__, "_agent_meta", {})
        return meta.get("name", self.__class__.__name__)

    @property
    def version(self) -> str:
        meta = getattr(self.__class__, "_agent_meta", {})
        return meta.get("version", "0.0.0")

    @abstractmethod
    async def handle(self, task: TaskMessage) -> TaskResult:
        """Process a task and return a result."""
        ...

    async def on_start(self) -> None:
        """Called when the agent starts. Override for initialization."""
        logger.info("Agent %s started", self.agent_id)

    async def on_stop(self) -> None:
        """Called when the agent stops. Override for cleanup."""
        logger.info("Agent %s stopped", self.agent_id)


def agent(
    name: str,
    version: str = "1.0.0",
    capabilities: list[str] | None = None,
) -> Any:
    """Decorator to register an agent class.

    Usage:
        @agent(name="rlm-executor", version="1.0", capabilities=["code-analysis"])
        class RLMExecutor(BaseAgent):
            async def handle(self, task: TaskMessage) -> TaskResult:
                ...
    """
    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        cls._agent_meta = {  # type: ignore[attr-defined]
            "name": name,
            "version": version,
            "capabilities": capabilities or [],
        }
        _AGENT_REGISTRY[name] = cls
        logger.debug("Registered agent: %s v%s", name, version)
        return cls
    return decorator


def get_agent_registry() -> dict[str, type[BaseAgent]]:
    """Return the global agent registry (read-only copy)."""
    return dict(_AGENT_REGISTRY)
