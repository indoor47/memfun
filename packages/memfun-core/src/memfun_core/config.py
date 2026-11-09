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

#: Providers memfun maps itself (provider + model -> LiteLLM route).  For
#: these the endpoint is (provider, base_url).  Anything else -- notably
#: ``custom`` -- passes ``model`` straight to LiteLLM, so the route lives
#: in the model string (``openrouter/x``) and a model change can mean a
#: different provider.
KNOWN_LLM_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "openai-compat", "ollama")
#: Provider spellings accepted without a warning.
_RECOGNIZED_PROVIDERS: tuple[str, ...] = (*KNOWN_LLM_PROVIDERS, "custom")


def _normalize_provider(value: str, where: str) -> str:
    """Lower-case/strip known provider names; warn on unrecognized ones."""
    norm = value.strip().lower()
    if norm in _RECOGNIZED_PROVIDERS:
        return norm
    logger.warning(
        "%s: unrecognized provider %r; it is passed to LiteLLM as a custom route (known: %s)",
        where,
        value,
        ", ".join(_RECOGNIZED_PROVIDERS),
    )
    return value


def _model_route(model: str) -> str:
    """LiteLLM route of a raw model string: the prefix before the first '/'."""
    return model.split("/", 1)[0] if "/" in model else model


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
    constrained_json: bool | None = None
    #: none | low | medium | high: thinking budget for this role only (e.g. worker "medium"
    #: while the planner and auditors stay at ``[llm]``'s "none").
    reasoning_effort: str | None = None


#: Context windows (tokens) by provider.  Cerebras serves Qwen3.8-27B at 131072; the
#: anthropic default is the standard 200k.  Any other OpenAI-compatible endpoint is
#: assumed to be 131072 (the safe smaller size); set ``[llm] context_window`` for a
#: bigger one (the B300 / SGLang pod serves 262144).
_CONTEXT_WINDOW_CEREBRAS = 131_072
_CONTEXT_WINDOW_ANTHROPIC = 200_000
_CONTEXT_WINDOW_DEFAULT = 131_072


def default_context_window(provider: str, base_url: str | None = None) -> int:
    """The provider's default context window in tokens."""
    if provider == "cerebras" or "cerebras" in (base_url or "").lower():
        return _CONTEXT_WINDOW_CEREBRAS
    if provider == "anthropic":
        return _CONTEXT_WINDOW_ANTHROPIC
    return _CONTEXT_WINDOW_DEFAULT


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
    # Sampling penalties against repetition loops, sent to OpenAI-compatible servers
    # (SGLang: frequency_penalty / presence_penalty in [-2, 2]; repetition_penalty in
    # (0, 2], 1.0 = off, sent as ``extra_body``).  0 / None send nothing.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float | None = None
    # Abort a call to an OpenAI-compatible server (``base_url`` set) as soon as its output
    # degenerates into a repetition loop (``memfun_agent.loop_guard``).  Makes the call with
    # streaming on DSPy's LiteLLM engine and disables DSPy's response cache for it.  Off by
    # default; the B300 runner turns it on.
    loop_guard: bool = False
    # Reasoning effort for OpenAI-compatible servers that take ``reasoning_effort``
    # (Cerebras: none = thinking off, low/medium/high).  None sends nothing.
    reasoning_effort: str | None = None
    # Grammar-constrained decoding (structured outputs) for the decomposer's
    # plan and the auditor's findings.  Only effective for provider
    # openai/openai-compat with a base_url (SGLang / vLLM style servers);
    # off by default, so other providers see no change.  Inherited by roles.
    constrained_json: bool = False
    # The model's context window in tokens (prompt + completion).  0 = the provider
    # default (:func:`default_context_window`).  The context-first solver keeps the
    # gathered context well inside it.
    context_window: int = 0
    # Role -> overrides, from ``[llm.roles.<role>]``.  Empty = every role
    # uses this ``[llm]`` config (the pre-roles behaviour).  Excluded from
    # hashing so a resolved (role-free) LLMConfig stays hashable.
    roles: dict[str, LLMRoleConfig] = field(default_factory=lambda: _no_roles(), hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.loop_guard, bool):
            raise ConfigError(f"[llm] loop_guard must be true or false, got {self.loop_guard!r}")
        if self.reasoning_effort is not None and self.reasoning_effort not in (
            "none",
            "low",
            "medium",
            "high",
        ):
            raise ConfigError(
                "[llm] reasoning_effort must be none, low, medium or high, "
                f"got {self.reasoning_effort!r}"
            )
        for name in ("frequency_penalty", "presence_penalty"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int | float) or not -2 <= val <= 2:
                raise ConfigError(f"[llm] {name} must be a number in [-2, 2], got {val!r}")
        rep = self.repetition_penalty
        if rep is not None and (
            isinstance(rep, bool) or not isinstance(rep, int | float) or not 0 < rep <= 2
        ):
            raise ConfigError(f"[llm] repetition_penalty must be a number in (0, 2], got {rep!r}")
        cw = self.context_window
        if isinstance(cw, bool) or not isinstance(cw, int) or cw < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigError(f"[llm] context_window must be an integer >= 0, got {cw!r}")
        if not isinstance(self.constrained_json, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigError(
                f"[llm] constrained_json must be true or false, got {self.constrained_json!r}"
            )

    @property
    def context_window_effective(self) -> int:
        """``context_window``, or the provider default when it is 0."""
        return self.context_window or default_context_window(self.provider, self.base_url)

    @property
    def constrained_json_effective(self) -> bool:
        """``constrained_json`` AND an endpoint that can honour it.

        Grammar-constrained decoding is an OpenAI-compatible-server feature
        (``response_format``); it is never sent to Anthropic/Ollama/custom
        routes, nor to openai without a ``base_url`` (the hosted API).
        """
        return (
            self.constrained_json
            and self.provider in ("openai", "openai-compat")
            and bool(self.base_url)
        )

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
        # A role that switches provider does not inherit [llm]'s base_url:
        # e.g. an anthropic role over an openai tunnel [llm] must go to
        # Anthropic, not send its key to the tunnel.
        if "base_url" not in updates and updates.get("provider", self.provider) != self.provider:
            updates["base_url"] = None
        # Never send [llm]'s key to a different endpoint: api_key_env is
        # inherited only when provider AND base_url resolve to [llm]'s --
        # and, for providers routed by the model string (custom), when
        # the model's LiteLLM route prefix is unchanged too.
        if "api_key_env" not in updates:
            provider = updates.get("provider", self.provider)
            same_endpoint = provider == self.provider and (
                updates.get("base_url", self.base_url) == self.base_url
            )
            if same_endpoint and provider not in KNOWN_LLM_PROVIDERS:
                same_endpoint = _model_route(updates.get("model", self.model)) == _model_route(
                    self.model
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


DEFAULT_STALL_STEPS = 12


def _no_iterations() -> dict[str, int]:
    return {}


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Agent step limits, from ``[limits]`` in ``memfun.toml``.

    Step caps are a backstop, not a budget: the real stop for an agent that is not
    making progress is stall detection, and a run's overall budget is a wall-clock /
    cost limit set outside memfun (e.g. the RunPod kit's pod hard stop).

    Attributes:
        stall_steps: Stop an agent after this many consecutive iterations that
            did nothing new (only re-read files it already read, re-ran a command
            with no edit in between, or ran no operations at all).  ``0`` disables
            stall stopping.
        iterations: Per-agent iteration caps by agent name, e.g.
            ``[limits.iterations]`` with ``coder-agent = 80``.  Agents not listed
            use their built-in default (see ``specialists.py`` / ``coding_agent.py``).
    """

    stall_steps: int = DEFAULT_STALL_STEPS
    iterations: dict[str, int] = field(default_factory=lambda: _no_iterations(), hash=False)


def _parse_limits(raw: dict[str, Any]) -> LimitsConfig:
    stall = raw.get("stall_steps", DEFAULT_STALL_STEPS)
    if isinstance(stall, bool) or not isinstance(stall, int) or stall < 0:
        raise ConfigError(f"[limits] stall_steps must be an integer >= 0, got {stall!r}")
    its_raw = raw.get("iterations", {})
    if not isinstance(its_raw, dict):
        raise ConfigError("[limits.iterations] must be a table of agent name = integer")
    its: dict[str, int] = {}
    for name, val in cast("dict[str, Any]", its_raw).items():
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ConfigError(f"[limits.iterations] {name} must be an integer >= 1, got {val!r}")
        its[str(name)] = val
    return LimitsConfig(stall_steps=stall, iterations=its)


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
    limits: LimitsConfig = field(default_factory=LimitsConfig)

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
        limits_raw = _section("limits")

        def _pick(section: dict[str, Any], dc: type) -> dict[str, Any]:
            names = {f.name for f in dataclasses.fields(dc)}
            return {k: v for k, v in section.items() if k in names}

        return cls(
            project_name=_section("project").get("name", "memfun-project"),
            llm=LLMConfig(
                **_llm_fields(_pick(llm_raw, LLMConfig)),
                roles=_parse_roles(llm_raw.get("roles")),
            ),
            backend=BackendConfig(**_pick(backend_raw, BackendConfig)),
            sandbox=SandboxBackendConfig(**_pick(sandbox_raw, SandboxBackendConfig)),
            web=WebToolsConfig(**_pick(web_raw, WebToolsConfig)),
            workflow=WorkflowConfig(**_pick(workflow_raw, WorkflowConfig)),
            limits=_parse_limits(limits_raw),
        )


def _llm_fields(picked: dict[str, Any]) -> dict[str, Any]:
    """``[llm]`` fields minus ``roles``, with ``provider`` normalized."""
    fields = {k: v for k, v in picked.items() if k != "roles"}
    provider = fields.get("provider")
    if isinstance(provider, str):
        fields["provider"] = _normalize_provider(provider, "[llm]")
    return fields


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
        if key == "provider":
            return _normalize_provider(val, f"[llm.roles.{role}]")
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
    if key == "constrained_json":
        if not isinstance(val, bool):
            msg = f"{where} must be true or false"
            raise ConfigError(msg)
        return val
    if key == "reasoning_effort":
        if val not in ("none", "low", "medium", "high"):
            msg = f"{where} must be none, low, medium or high"
            raise ConfigError(msg)
        return val
    return val  # pragma: no cover - guarded by the allowed-keys check
