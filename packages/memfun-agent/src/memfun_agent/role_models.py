"""Per-role models: resolve ``[llm.roles.<role>]`` to a DSPy LM per call.

The swarm runs different roles on different models -- e.g. the
orchestrator and decomposer on a frontier hosted model, workers and
per-task auditors on a self-hosted OpenAI-compatible endpoint.  This
module is the single place that turns an :class:`LLMConfig` into a
LiteLLM model name + ``dspy.LM`` kwargs, and binds a role's LM to a
region of code.

Binding uses ``dspy.context(lm=...)``, which stores the override in a
:class:`~contextvars.ContextVar`.  Every asyncio task (and every
``asyncio.to_thread`` worker, which copies the caller's context) sees
only its own override, so concurrent agents running different roles
never clobber each other -- unlike ``dspy.configure``, which mutates a
process-wide global.

Binding rules for :func:`role_scope`:

* No ``[llm.roles]`` tables at all: no LM is bound; the globally
  configured ``[llm]`` LM is used exactly as before (the role is still
  recorded for ``llm.call`` telemetry).
* Any ``[llm.roles]`` table present: EVERY scope binds its role's
  resolved LM explicitly -- a role without a table binds the ``[llm]``
  LM.  Nested scopes therefore never inherit an enclosing role's model
  (e.g. a worker dispatched inside the orchestrator scope).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import ipaddress
import os
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from memfun_core.config import LLM_ROLES, LLMConfig, MemfunConfig
from memfun_core.errors import ConfigError
from memfun_core.logging import get_logger

from memfun_agent.llm_telemetry import role_context

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger("agent.role_models")

#: Placeholder key for self-hosted OpenAI-compatible endpoints that run
#: without auth (LiteLLM still requires a non-empty ``api_key``).
LOCAL_PLACEHOLDER_KEY = "sk-local"

#: Agents whose LLM calls run under a role other than ``worker``.
#: Specialists not listed here are workers.
AGENT_ROLE_MAP: dict[str, str] = {
    "review-agent": "auditor",
    "pr-reviewer": "reviewer",
}

# Seconds.  Self-hosted / queued endpoints can take 10-30+ minutes per
# call on large outputs; LiteLLM's 600 s default kills them mid-flight.
_LM_TIMEOUT_SECONDS = 3600


# ── Provider mapping (shared with the CLI's _configure_dspy) ────────


def litellm_model_name(provider: str, model: str) -> str:
    """Map a memfun ``provider`` + ``model`` to a LiteLLM model name."""
    if provider == "anthropic":
        return f"anthropic/{model}"
    if provider in ("openai", "openai-compat"):
        # openai-compat (llama.cpp / vLLM / SGLang / OpenRouter) is
        # LiteLLM's openai client with a custom api_base.
        return f"openai/{model}"
    if provider == "ollama":
        return f"ollama_chat/{model}"
    return model


def is_local_base_url(url: str | None) -> bool:
    """``True`` when *url*'s host is loopback / private / link-local.

    Accepts ``localhost``, ``*.local``, and IP literals in 127.0.0.0/8,
    ::1, 10/8, 172.16/12, 192.168/16 and link-local ranges.  The host is
    taken from the parsed URL (so ``http://127.0.0.1@evil.com`` is
    ``evil.com``); look-alike names such as ``localhost.evil.com`` or
    ``127.0.0.1.evil.com`` are NOT local.
    """
    if not url:
        return False
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def redact_url(url: str | None) -> str:
    """Return *url* as ``scheme://host[:port]/path`` for logging.

    Drops userinfo, query and fragment (they can carry credentials,
    e.g. ``https://user:pass@host/v1?api_key=...``).
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    scheme = f"{parts.scheme}://" if parts.scheme else ""
    return f"{scheme}{netloc}{parts.path}"


def resolve_api_key(llm: LLMConfig) -> str:
    """Read the API key from the env var NAMED by ``llm.api_key_env``.

    Returns ``""`` when unset, except for keyless local servers which get
    a placeholder: ``ollama``, ``openai-compat``, and ``openai`` whose
    ``base_url`` is a loopback / private host (:func:`is_local_base_url`).
    A hosted ``base_url`` without a key returns ``""`` so the missing key
    is reported instead of hidden behind a placeholder.
    """
    if llm.provider == "ollama":
        return "ollama"
    key = os.environ.get(llm.api_key_env, "") if llm.api_key_env else ""
    if key:
        return key
    if llm.provider == "openai-compat":
        return LOCAL_PLACEHOLDER_KEY
    if llm.provider == "openai" and is_local_base_url(llm.base_url):
        return LOCAL_PLACEHOLDER_KEY
    return ""


def lm_settings(llm: LLMConfig) -> tuple[str, dict[str, Any]]:
    """Return ``(litellm_model_name, dspy.LM kwargs)`` for *llm*.

    The kwargs contain the resolved API key value when one is available;
    never log or persist them.
    """
    kwargs: dict[str, Any] = {
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "timeout": _LM_TIMEOUT_SECONDS,
    }
    if llm.frequency_penalty:
        kwargs["frequency_penalty"] = llm.frequency_penalty
    if llm.presence_penalty:
        kwargs["presence_penalty"] = llm.presence_penalty
    if llm.repetition_penalty is not None and llm.repetition_penalty != 1.0:
        # Not an OpenAI field: SGLang reads it from the request body (extra_body is merged).
        kwargs["extra_body"] = {"repetition_penalty": llm.repetition_penalty}
    api_key = resolve_api_key(llm)
    if api_key:
        kwargs["api_key"] = api_key
    if llm.base_url:
        kwargs["api_base"] = llm.base_url
    return litellm_model_name(llm.provider, llm.model), kwargs


def build_lm(llm: LLMConfig) -> Any:
    """Construct a ``dspy.LM`` for *llm* (no caching)."""
    import dspy

    name, kwargs = lm_settings(llm)
    from memfun_agent.loop_guard import enabled, install_loop_guard

    if llm.loop_guard and name.startswith("openai/") and kwargs.get("api_base") and enabled():
        # The loop guard wraps litellm.completion, which DSPy only calls on its LiteLLM
        # engine (the default "auto" picks the native lm15 engine for this request).
        install_loop_guard()
        # Only DSPy >= 3.x has an ``engine`` parameter (the default there picks the native
        # engine, which never reaches litellm); older DSPy always uses litellm and would
        # forward an unknown ``engine`` to the server as a request field.
        if "engine" in inspect.signature(dspy.LM.__init__).parameters:
            kwargs["engine"] = "litellm"
        # No response cache on these sampled, guarded calls: a cached loop-aborted (or
        # otherwise bad) response would be replayed identically on every retry.
        kwargs["cache"] = False
    return dspy.LM(name, **kwargs)


# ── Per-role LM cache ───────────────────────────────────────────────

_LM_CACHE: dict[tuple[Any, ...], Any] = {}
_LM_CACHE_LOCK = threading.Lock()


def _cache_key(llm: LLMConfig) -> tuple[Any, ...]:
    # Env var NAME, not value: the key never contains a secret.
    return (
        llm.provider,
        llm.model,
        llm.base_url,
        llm.api_key_env,
        llm.temperature,
        llm.max_tokens,
        llm.frequency_penalty,
        llm.presence_penalty,
        llm.repetition_penalty,
        llm.loop_guard,
    )


def _cached_lm(llm: LLMConfig) -> Any:
    key = _cache_key(llm)
    with _LM_CACHE_LOCK:
        lm = _LM_CACHE.get(key)
        if lm is None:
            lm = build_lm(llm)
            _LM_CACHE[key] = lm
        return lm


def lm_for_role(config: MemfunConfig, role: str) -> Any:
    """Return the (cached) ``dspy.LM`` for *role* under *config*.

    Roles that resolve to the same effective settings share one LM
    instance.  Raises :class:`ConfigError` for an unknown role.
    """
    return _cached_lm(config.llm_for_role(role))


def default_lm(config: MemfunConfig) -> Any:
    """Return the (cached) ``dspy.LM`` for plain ``[llm]`` (no role)."""
    return _cached_lm(dataclasses.replace(config.llm, roles={}))


def clear_lm_cache() -> None:
    """Drop cached role LMs (tests / config reload)."""
    with _LM_CACHE_LOCK:
        _LM_CACHE.clear()


# ── Active config + per-call binding ────────────────────────────────

# Set once at CLI startup (alongside dspy.configure); read by every
# role_scope.  A plain module global, not a contextvar, so agents
# started in any task or thread see it.
_role_config: MemfunConfig | None = None


def set_role_config(config: MemfunConfig | None) -> None:
    """Register the config whose ``[llm.roles]`` drive :func:`role_scope`.

    Runs a startup preflight: one WARNING per configured role whose API
    key is missing or whose LM cannot be built, so a dead role config is
    visible at startup rather than as a failure mid-swarm.
    """
    global _role_config
    _role_config = config
    if config is not None:
        preflight_roles(config)


def preflight_roles(config: MemfunConfig) -> list[str]:
    """Check every ``[llm.roles.<role>]``; return the roles with problems.

    Never raises.  Logs the env var NAME, never a key value.
    """
    bad: list[str] = []
    for role in LLM_ROLES:
        if not config.has_role_override(role):
            continue
        try:
            resolved = config.llm_for_role(role)
            if not resolve_api_key(resolved):
                bad.append(role)
                logger.warning(
                    "LLM role %r has no API key: %s is %s for provider %r at %s; "
                    "calls on this role will fail. Set [llm.roles.%s].api_key_env.",
                    role,
                    resolved.api_key_env or "(api_key_env not set)",
                    "unset" if resolved.api_key_env else "not configured",
                    resolved.provider,
                    redact_url(resolved.base_url) or "the provider's default endpoint",
                    role,
                )
                continue
            lm_for_role(config, role)
        except Exception:
            bad.append(role)
            logger.warning("LLM role %r: cannot build its LM", role, exc_info=True)
    return bad


def get_role_config() -> MemfunConfig | None:
    """Return the config registered with :func:`set_role_config`."""
    return _role_config


def iteration_limit(agent_name: str, default: int) -> int:
    """Max iterations for *agent_name*: ``[limits.iterations]`` override, else *default*."""
    cfg = get_role_config()
    if cfg is not None:
        return cfg.limits.iterations.get(agent_name, default)
    return default


def stall_steps(default: int = 12) -> int:
    """Consecutive no-progress iterations before an agent is stopped (``[limits]``)."""
    cfg = get_role_config()
    return cfg.limits.stall_steps if cfg is not None else default


def role_for_agent(agent_name: str, default: str | None = "worker") -> str | None:
    """Return the role an agent's LLM calls run under."""
    return AGENT_ROLE_MAP.get(agent_name, default)


@contextlib.contextmanager
def role_scope(role: str, config: MemfunConfig | None = None) -> Generator[None]:
    """Run the ``with`` body under *role*.

    * Telemetry: ``llm.call`` events inside the block report ``role``.
    * Model: when the config has ANY ``[llm.roles]`` table, DSPy calls
      inside the block use this role's resolved LM via ``dspy.context``
      (task-local) -- the ``[llm]`` LM for a role without its own table,
      never an enclosing scope's LM.  With no role tables at all nothing
      is bound and the global ``[llm]`` LM is used as before.

    If the role's LM cannot be built, the plain ``[llm]`` LM is bound
    explicitly (WARNING names the role); if that fails too, the error
    propagates rather than running on some other role's model.

    *config* defaults to the one registered with :func:`set_role_config`.
    Raises :class:`ConfigError` for an unknown role name.
    """
    if role not in LLM_ROLES:
        msg = f"Unknown LLM role {role!r}; expected one of {', '.join(LLM_ROLES)}"
        raise ConfigError(msg)

    cfg = config if config is not None else _role_config
    lm: Any = None
    if cfg is not None and cfg.llm.roles:
        try:
            lm = lm_for_role(cfg, role)
        except Exception:
            logger.warning(
                "Could not build LM for role %r; binding the [llm] LM instead",
                role,
                exc_info=True,
            )
            lm = default_lm(cfg)

    with role_context(role):
        if lm is None:
            yield
            return
        import dspy

        with dspy.context(lm=lm):  # pyright: ignore[reportUnknownMemberType]
            yield
