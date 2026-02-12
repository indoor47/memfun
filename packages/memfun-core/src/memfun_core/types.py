from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

# ── Event Bus Types ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Message:
    """A message received from the event bus."""
    id: str
    topic: str
    payload: bytes
    key: str | None = None
    timestamp: float = field(default_factory=time.time)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TopicConfig:
    """Configuration for creating a topic."""
    retention_seconds: int = 86400
    max_consumers: int | None = None
    max_message_bytes: int = 1_048_576
    dedup_enabled: bool = False
    replay_policy: str = "instant"


# ── State Store Types ────────────────────────────────────────────────

class StateChangeKind(enum.Enum):
    PUT = "put"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class StateChange:
    """Notification of a state change on a watched key."""
    key: str
    kind: StateChangeKind
    value: bytes | None
    revision: int
    timestamp: float = field(default_factory=time.time)


# ── Sandbox Types ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Configuration for creating a sandbox."""
    language: str = "python"
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    network_access: bool = False
    read_paths: list[str] = field(default_factory=list)
    write_paths: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SandboxHandle:
    """Opaque handle to a running sandbox."""
    id: str
    backend: str
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of executing code in a sandbox."""
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    truncated: bool = False


# ── Agent Types ──────────────────────────────────────────────────────

class AgentStatusKind(enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AgentStatus:
    """Current status of a running agent."""
    agent_id: str
    status: AgentStatusKind
    uptime_seconds: float
    last_heartbeat: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """Metadata about a registered agent."""
    agent_id: str
    name: str
    version: str
    capabilities: list[str]
    metadata: dict[str, str] = field(default_factory=dict)
    endpoint: str | None = None


# ── Session Types ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Configuration for creating a user session."""
    ttl_seconds: int = 3600
    max_history: int = 1000
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Session:
    """A user session with mutable data."""
    session_id: str
    user_id: str
    created_at: float
    data: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


# ── Health Types ─────────────────────────────────────────────────────

class HealthState(enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Health check result for an agent."""
    agent_id: str
    state: HealthState
    last_heartbeat: float
    metrics: dict[str, float] = field(default_factory=dict)
    message: str | None = None


# ── Task Types (for agent delegation) ───────────────────────────────

@dataclass(frozen=True, slots=True)
class TaskMessage:
    """A task to be processed by an agent."""
    task_id: str
    agent_id: str
    payload: dict[str, Any]
    correlation_id: str | None = None
    parent_task_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Result of a task processed by an agent."""
    task_id: str
    agent_id: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
