from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfun_core.config import MemfunConfig

    from memfun_runtime.protocols.event_bus import EventBusAdapter
    from memfun_runtime.protocols.health import HealthMonitor
    from memfun_runtime.protocols.lifecycle import AgentLifecycle
    from memfun_runtime.protocols.registry import RegistryAdapter
    from memfun_runtime.protocols.sandbox import SandboxAdapter
    from memfun_runtime.protocols.session import SessionManager
    from memfun_runtime.protocols.skill_registry import SkillRegistryAdapter
    from memfun_runtime.protocols.state_store import StateStoreAdapter


@dataclass(slots=True)
class RuntimeContext:
    """Central context object injected into every agent.

    Provides access to all runtime adapters and project configuration.
    Created once at startup by the RuntimeBuilder.
    """
    event_bus: EventBusAdapter
    state_store: StateStoreAdapter
    sandbox: SandboxAdapter
    lifecycle: AgentLifecycle
    registry: RegistryAdapter
    session: SessionManager
    health: HealthMonitor
    skill_registry: SkillRegistryAdapter
    config: MemfunConfig
