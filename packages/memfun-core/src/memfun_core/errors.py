from __future__ import annotations


class MemfunError(Exception):
    """Base exception for all Memfun errors."""


# ── Runtime Errors ───────────────────────────────────────────────────

class RuntimeBackendError(MemfunError):
    """Base for runtime/infrastructure errors."""


class BackendError(RuntimeBackendError):
    """Error from a backend adapter."""


class BackendUnavailableError(BackendError):
    """Backend is not reachable or not configured."""


# ── Agent Errors ─────────────────────────────────────────────────────

class AgentError(MemfunError):
    """Base for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Agent not found in the registry."""


class AgentAlreadyRunningError(AgentError):
    """Agent is already running."""


# ── Session Errors ───────────────────────────────────────────────────

class SessionError(MemfunError):
    """Base for session-related errors."""


class SessionNotFoundError(SessionError):
    """Session does not exist or has expired."""


# ── Sandbox Errors ───────────────────────────────────────────────────

class SandboxError(MemfunError):
    """Base for sandbox-related errors."""


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded timeout."""


class SandboxUnavailableError(SandboxError):
    """Sandbox backend is not available."""


# ── Config Errors ────────────────────────────────────────────────────

class ConfigError(MemfunError):
    """Invalid or missing configuration."""


# ── Skill Errors ─────────────────────────────────────────────────────

class SkillError(MemfunError):
    """Base for skill-related errors."""


class SkillNotFoundError(SkillError):
    """Skill not found in any skill directory."""


class SkillValidationError(SkillError):
    """Skill definition is invalid."""


# ── Agent Definition Errors ─────────────────────────────────────────

class AgentValidationError(AgentError):
    """Agent definition is invalid."""
