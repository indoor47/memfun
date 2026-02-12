"""Protocol interfaces for the Memfun pluggable runtime."""
from __future__ import annotations

from memfun_runtime.protocols.event_bus import EventBusAdapter
from memfun_runtime.protocols.health import HealthMonitor
from memfun_runtime.protocols.lifecycle import AgentLifecycle
from memfun_runtime.protocols.orchestrator import OrchestratorAdapter
from memfun_runtime.protocols.registry import RegistryAdapter
from memfun_runtime.protocols.sandbox import SandboxAdapter
from memfun_runtime.protocols.session import SessionManager
from memfun_runtime.protocols.skill_registry import SkillInfo, SkillRegistryAdapter
from memfun_runtime.protocols.state_store import StateStoreAdapter

__all__ = [
    "AgentLifecycle",
    "EventBusAdapter",
    "HealthMonitor",
    "OrchestratorAdapter",
    "RegistryAdapter",
    "SandboxAdapter",
    "SessionManager",
    "SkillInfo",
    "SkillRegistryAdapter",
    "StateStoreAdapter",
]
