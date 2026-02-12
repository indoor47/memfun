"""Memfun Runtime: pluggable backend architecture for autonomous agents."""
from __future__ import annotations

from memfun_runtime.agent import BaseAgent, agent, get_agent_registry
from memfun_runtime.builder import RuntimeBuilder
from memfun_runtime.context import RuntimeContext
from memfun_runtime.lifecycle import AgentManager
from memfun_runtime.orchestrator import AgentOrchestrator
from memfun_runtime.protocols import (
    AgentLifecycle,
    EventBusAdapter,
    HealthMonitor,
    OrchestratorAdapter,
    RegistryAdapter,
    SandboxAdapter,
    SessionManager,
    SkillInfo,
    SkillRegistryAdapter,
    StateStoreAdapter,
)

__all__ = [
    "AgentLifecycle",
    "AgentManager",
    "AgentOrchestrator",
    "BaseAgent",
    "EventBusAdapter",
    "HealthMonitor",
    "OrchestratorAdapter",
    "RegistryAdapter",
    "RuntimeBuilder",
    "RuntimeContext",
    "SandboxAdapter",
    "SessionManager",
    "SkillInfo",
    "SkillRegistryAdapter",
    "StateStoreAdapter",
    "agent",
    "get_agent_registry",
]
