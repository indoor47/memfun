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

Backward compatibility: :func:`role_scope` only overrides the LM when
the role has an explicit ``[llm.roles.<role>]`` table.  Without one the
globally configured ``[llm]`` LM is used exactly as before; the role is
still recorded for ``llm.call`` telemetry.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING, Any

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


def resolve_api_key(llm: LLMConfig) -> str:
    """Read the API key from the env var NAMED by ``llm.api_key_env``.

    Returns ``""`` when unset, except for keyless local providers which
    get a placeholder: ``ollama``, ``openai-compat``, and ``openai`` with
    an explicit ``base_url`` (a self-hosted OpenAI-compatible server).
    """
    if llm.provider == "ollama":
        return "ollama"
    key = os.environ.get(llm.api_key_env, "") if llm.api_key_env else ""
    if key:
        return key
    if llm.provider == "openai-compat" or (llm.provider == "openai" and llm.base_url):
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
    if "api_key" not in kwargs:
        logger.warning(
            "No API key for %s (set %s); calls with model %s will fail",
            llm.provider,
            llm.api_key_env,
            llm.model,
        )
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
    )


def lm_for_role(config: MemfunConfig, role: str) -> Any:
    """Return the (cached) ``dspy.LM`` for *role* under *config*.

    Roles that resolve to the same effective settings share one LM
    instance.  Raises :class:`ConfigError` for an unknown role.
    """
    resolved = config.llm_for_role(role)
    key = _cache_key(resolved)
    with _LM_CACHE_LOCK:
        lm = _LM_CACHE.get(key)
        if lm is None:
            lm = build_lm(resolved)
            _LM_CACHE[key] = lm
        return lm


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
    """Register the config whose ``[llm.roles]`` drive :func:`role_scope`."""
    global _role_config
    _role_config = config


def get_role_config() -> MemfunConfig | None:
    """Return the config registered with :func:`set_role_config`."""
    return _role_config


def role_for_agent(agent_name: str, default: str | None = "worker") -> str | None:
    """Return the role an agent's LLM calls run under."""
    return AGENT_ROLE_MAP.get(agent_name, default)


@contextlib.contextmanager
def role_scope(role: str, config: MemfunConfig | None = None) -> Generator[None]:
    """Run the ``with`` body under *role*.

    * Telemetry: ``llm.call`` events inside the block report ``role``.
    * Model: when ``[llm.roles.<role>]`` is configured, DSPy calls inside
      the block use that role's LM via ``dspy.context`` (task-local).
      Otherwise the global ``[llm]`` LM is left untouched.

    *config* defaults to the one registered with :func:`set_role_config`.
    Raises :class:`ConfigError` for an unknown role name.
    """
    if role not in LLM_ROLES:
        msg = f"Unknown LLM role {role!r}; expected one of {', '.join(LLM_ROLES)}"
        raise ConfigError(msg)

    cfg = config if config is not None else _role_config
    lm: Any = None
    if cfg is not None and cfg.has_role_override(role):
        try:
            lm = lm_for_role(cfg, role)
        except Exception:
            # Never let a bad role LM take the agent down silently on
            # the wrong model: log loudly and fall back to [llm].
            logger.warning(
                "Could not build LM for role %r; falling back to [llm]",
                role,
                exc_info=True,
            )
            lm = None

    with role_context(role):
        if lm is None:
            yield
            return
        import dspy

        with dspy.context(lm=lm):  # pyright: ignore[reportUnknownMemberType]
            yield
