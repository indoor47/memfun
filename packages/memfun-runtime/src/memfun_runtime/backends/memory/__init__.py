"""T0 In-Process Backend: zero dependencies, in-memory only."""
from __future__ import annotations

from memfun_runtime.backends.memory.event_bus import InProcessEventBus
from memfun_runtime.backends.memory.health import InProcessHealthMonitor
from memfun_runtime.backends.memory.lifecycle import InProcessLifecycle
from memfun_runtime.backends.memory.registry import InProcessRegistry
from memfun_runtime.backends.memory.session import InProcessSessionManager
from memfun_runtime.backends.memory.skill_registry import InProcessSkillRegistry
from memfun_runtime.backends.memory.state_store import InProcessStateStore

__all__ = [
    "InProcessEventBus",
    "InProcessHealthMonitor",
    "InProcessLifecycle",
    "InProcessRegistry",
    "InProcessSessionManager",
    "InProcessSkillRegistry",
    "InProcessStateStore",
]
