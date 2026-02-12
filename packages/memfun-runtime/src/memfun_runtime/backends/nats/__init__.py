"""T3 NATS Backend: distributed, NATS JetStream-backed runtime adapters."""
from __future__ import annotations

from memfun_runtime.backends.nats.event_bus import NATSEventBus
from memfun_runtime.backends.nats.health import NATSHealthMonitor
from memfun_runtime.backends.nats.lifecycle import NATSLifecycle
from memfun_runtime.backends.nats.registry import NATSRegistry
from memfun_runtime.backends.nats.session import NATSSessionManager
from memfun_runtime.backends.nats.skill_registry import NATSSkillRegistry
from memfun_runtime.backends.nats.state_store import NATSStateStore

__all__ = [
    "NATSEventBus",
    "NATSHealthMonitor",
    "NATSLifecycle",
    "NATSRegistry",
    "NATSSessionManager",
    "NATSSkillRegistry",
    "NATSStateStore",
]
