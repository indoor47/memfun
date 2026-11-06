from __future__ import annotations

import dataclasses
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from memfun_core.errors import ConfigError
from memfun_core.logging import get_logger

logger = get_logger("core.config")

#: Roles that may carry their own model in ``[llm.roles.<role>]``.
#:
#: * ``orchestrator`` -- top-level workflow driving (swarm / chat workflow).
#: * ``decomposer``   -- ``TaskDecomposer`` (task -> sub-task DAG).
#: * ``worker``       -- specialist agents that write code / run tools.
#: * ``auditor``      -- per-task ``review-agent`` dispatches.
#: * ``reviewer``     -- the ``pr-reviewer`` agent at the PR layer.
LLM_ROLES: tuple[str, ...] = (
    "orchestrator",
    "decomposer",
    "worker",
    "auditor",
    "reviewer",
)

# An env var NAME (never a value).  Rejecting anything else catches the
# common mistake of pasting the API key itself into ``api_key_env``.
_ENV_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning empty dict if missing."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (tables merge, scalars replace).

    Recursion matters for nested tables such as ``[llm.roles.worker]``:
    a project config that sets only ``[llm.roles.worker].model`` keeps the
    global ``[llm.roles.worker].base_url``.
    """
    merged: dict[str, Any] = dict(base)
    for key, val in override.items():
        cur = merged.get(key)
        if isinstance(cur, dict) and isinstance(val, dict):
            merged[key] = _deep_merge(cast("dict[str, Any]", cur), cast("dict[str, Any]", val))
        else:
            merged[key] = val
    return merged


@dataclass(frozen=True, slots=True)
class LLMRoleConfig:
    """Per-role overrides from ``[llm.roles.<role>]``.

    Every field is optional; ``None`` means "inherit from ``[llm]``".
    For ``base_url`` an explicit empty string in TOML means "no base URL"
    (clears an inherited one) and is stored as ``""``.  ``api_key_env``
    is inherited only when the role keeps ``[llm]``'s provider AND
    base_url; ``api_key_env = ""`` means "no key".
    """

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def _no_roles() -> dict[str, LLMRoleConfig]:
    return {}


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = "anthropic"
    # Default to Sonnet 4.6: fast + smart, good for most tasks. Users
    # can pick Opus (most intelligent) or Haiku (cheapest/fastest)
    # via the wizard or by editing config.toml.
    model: str = "claude-sonnet-4-6"
    sub_model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 128_000
    # Role -> overrides, from ``[llm.roles.<role>]``.  Empty = every role
    # uses this ``[llm]`` config (the pre-roles behaviour).  Excluded from
    # hashing so a resolved (role-free) LLMConfig stays hashable.
    roles: dict[str, LLMRoleConfig] = field(default_factory=lambda: _no_roles(), hash=False)

    def for_role(self, role: str) -> LLMConfig:
        """Resolve *role* to a concrete :class:`LLMConfig`.

        Each field falls back to ``[llm]`` individually when the role
        does not set it.  The result has ``roles={}``.  Raises
        :class:`ConfigError` for a role name not in :data:`LLM_ROLES`.
        """
        if role not in LLM_ROLES:
            msg = f"Unknown LLM role {role!r}; expected one of {', '.join(LLM_ROLES)}"
            raise ConfigError(msg)
        override = self.roles.get(role)
        if override is None:
            return dataclasses.replace(self, roles={})
        updates: dict[str, Any] = {}
        for f in dataclasses.fields(LLMRoleConfig):
            val = getattr(override, f.name)
            if val is None:
                continue
            if f.name == "base_url" and val == "":
                updates["base_url"] = None
                continue
            updates[f.name] = val
        # Never send [llm]'s key to a different endpoint: api_key_env is
        # inherited only when provider AND base_url resolve to [llm]'s.
        if "api_key_env" not in updates:
            same_endpoint = updates.get("provider", self.provider) == self.provider and (
                updates.get("base_url", self.base_url) == self.base_url
            )
            if not same_endpoint:
                updates["api_key_env"] = ""
        return dataclasses.replace(self, roles={}, **updates)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    tier: str = "sqlite"
    sqlite_path: str = ".memfun/memfun.db"
    sqlite_wal: bool = True
    sqlite_poll_interval_ms: int = 100
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "memfun:"
    nats_url: str = "nats://localhost:4222"
    nats_creds_file: str | None = None
    nats_stream_prefix: str = "memfun"


@dataclass(frozen=True, slots=True)
class SandboxBackendConfig:
    backend: str = "local"
    docker_image: str = "python:3.12-slim"
    modal_app_name: str = "memfun-sandbox"
    timeout_seconds: int = 30
    memory_limit_mb: int = 512


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Multi-agent workflow / swarm execution settings.

    Read from the ``[workflow]`` section of ``memfun.toml``.  Drives
    the auto-commit behaviour added in issue #37: when a
    :class:`WorkflowEngine` run succeeds, the produced files can be
    automatically staged and committed (and optionally pushed) on
    behalf of the user so ``memfun swarm`` output lands in git
    history rather than as a heap of untracked files.

    Attributes:
        auto_commit: Commit produced files after a successful workflow.
            Defaults to ``True`` (auto-commit is opt-OUT, not opt-in).
        auto_push: Push the auto-commit to ``origin`` after committing.
            Defaults to ``False`` — pushing is always opt-in.
        auto_branch: Branch strategy applied *before* the workflow
            runs.  ``None`` (default) keeps the user on their current
            branch.  ``"auto"`` creates a new
            ``memfun/swarm-<timestamp>`` branch.  Any other string is
            used verbatim as the new branch name.
        gh_pr: When ``True``, ``memfun swarm`` opens a draft GitHub PR
            per sub-task and routes each PR through the ``pr-reviewer``
            agent (issue #38).  Defaults to ``False`` — opt-in only,
            because the feature requires ``gh auth`` and a remote repo.
    """

    auto_commit: bool = True
    auto_push: bool = False
    auto_branch: str | None = None
    gh_pr: bool = False


@dataclass(frozen=True, slots=True)
class WebToolsConfig:
    fetch_timeout_seconds: int = 30
    fetch_max_size_bytes: int = 5_242_880  # 5 MiB
    fetch_cache_ttl_seconds: int = 900  # 15 min
    search_backend: str = "duckduckgo"  # duckduckgo | brave | tavily | searxng
    search_cache_ttl_seconds: int = 3600  # 1 hour
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 5
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    brave_api_key_env: str = "BRAVE_API_KEY"
    tavily_api_key_env: str = "TAVILY_API_KEY"
    searxng_url: str = "http://localhost:8888"


@dataclass(frozen=True, slots=True)
class MemfunConfig:
    """Top-level configuration, parsed from memfun.toml."""

    project_name: str = "memfun-project"
    llm: LLMConfig = field(default_factory=LLMConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    sandbox: SandboxBackendConfig = field(default_factory=SandboxBackendConfig)
    web: WebToolsConfig = field(default_factory=WebToolsConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def llm_for_role(self, role: str) -> LLMConfig:
        """Return the effective LLM config for *role* (see ``LLMConfig.for_role``)."""
        return self.llm.for_role(role)

    def has_role_override(self, role: str) -> bool:
        """``True`` when ``[llm.roles.<role>]`` is configured."""
        return role in self.llm.roles

    @classmethod
    def from_toml(cls, path: Path | str = "memfun.toml") -> MemfunConfig:
        path = Path(path)
        raw = _load_toml(path)
        return cls._from_raw(raw)

    @classmethod
    def load(cls, project_dir: Path | str | None = None) -> MemfunConfig:
        """Load config with global → project layering.

        Resolution order (later wins):
        1. Built-in defaults
        2. ~/.memfun/config.toml (global)
        3. .memfun/config.toml or memfun.toml (project)
        """
        global_path = Path.home() / ".memfun" / "config.toml"

        project_dir = Path.cwd() if project_dir is None else Path(project_dir)

        # Project config: .memfun/config.toml takes priority
        project_path = project_dir / ".memfun" / "config.toml"
        if not project_path.exists():
            project_path = project_dir / "memfun.toml"

        global_raw = _load_toml(global_path)
        project_raw = _load_toml(project_path)
        merged = _deep_merge(global_raw, project_raw)

        return cls._from_raw(merged)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> MemfunConfig:
        """Build MemfunConfig from a raw TOML dict."""

        def _section(name: str) -> dict[str, Any]:
            val = raw.get(name, {})
            return cast("dict[str, Any]", val) if isinstance(val, dict) else {}

        llm_raw = _section("llm")
        backend_raw = _section("backend")
        sandbox_raw = _section("sandbox")
        web_raw = _section("web")
        workflow_raw = _section("workflow")

        def _pick(section: dict[str, Any], dc: type) -> dict[str, Any]:
            names = {f.name for f in dataclasses.fields(dc)}
            return {k: v for k, v in section.items() if k in names}

        return cls(
            project_name=_section("project").get("name", "memfun-project"),
            llm=LLMConfig(
                **{k: v for k, v in _pick(llm_raw, LLMConfig).items() if k != "roles"},
                roles=_parse_roles(llm_raw.get("roles")),
            ),
            backend=BackendConfig(**_pick(backend_raw, BackendConfig)),
            sandbox=SandboxBackendConfig(**_pick(sandbox_raw, SandboxBackendConfig)),
            web=WebToolsConfig(**_pick(web_raw, WebToolsConfig)),
            workflow=WorkflowConfig(**_pick(workflow_raw, WorkflowConfig)),
        )


def _parse_roles(raw: Any) -> dict[str, LLMRoleConfig]:
    """Parse and validate the ``[llm.roles]`` table.

    Unknown role names and unknown keys are dropped with a warning (a typo
    must not crash startup).  Wrong value types and an ``api_key_env`` that
    is not a valid env var NAME raise :class:`ConfigError`; the offending
    value is never echoed, in case it is a secret.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("[llm.roles] must be a table of [llm.roles.<role>] tables")

    allowed = {f.name for f in dataclasses.fields(LLMRoleConfig)}
    roles: dict[str, LLMRoleConfig] = {}
    for role, table in cast("dict[str, Any]", raw).items():
        if role not in LLM_ROLES:
            logger.warning(
                "Ignoring unknown LLM role [llm.roles.%s]; known roles: %s",
                role,
                ", ".join(LLM_ROLES),
            )
            continue
        if not isinstance(table, dict):
            msg = f"[llm.roles.{role}] must be a table"
            raise ConfigError(msg)
        values: dict[str, Any] = {}
        for key, val in cast("dict[str, Any]", table).items():
            if key not in allowed:
                logger.warning("Ignoring unknown key %r in [llm.roles.%s]", key, role)
                continue
            values[key] = _validate_role_value(role, key, val)
        roles[role] = LLMRoleConfig(**values)
    return roles


def _validate_role_value(role: str, key: str, val: Any) -> Any:
    where = f"[llm.roles.{role}].{key}"
    if key in ("provider", "model", "base_url", "api_key_env"):
        if not isinstance(val, str):
            msg = f"{where} must be a string"
            raise ConfigError(msg)
        if key in ("provider", "model") and not val.strip():
            msg = f"{where} must not be empty"
            raise ConfigError(msg)
        # "" = explicitly no key; otherwise the WHOLE value must be a name
        # (fullmatch: "$" alone would accept a trailing newline).
        if key == "api_key_env" and val and not _ENV_VAR_NAME_RE.fullmatch(val):
            msg = (
                f"{where} must be the NAME of an environment variable "
                "(e.g. ANTHROPIC_API_KEY), not a key value"
            )
            raise ConfigError(msg)
        return val
    if key == "temperature":
        if isinstance(val, bool) or not isinstance(val, int | float):
            msg = f"{where} must be a number"
            raise ConfigError(msg)
        return float(val)
    if key == "max_tokens":
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            msg = f"{where} must be a positive integer"
            raise ConfigError(msg)
        return val
    return val  # pragma: no cover - guarded by the allowed-keys check
