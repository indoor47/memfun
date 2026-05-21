"""Per-agent LLM telemetry — publish ``llm.call`` events to the EventBus.

Wires DSPy's :class:`BaseCallback` hooks to the memfun runtime event bus so
the dashboard's Logs tab can show *every* LLM call made by *every* specialist
agent (10-30 calls per coder-agent iteration loop), not just the lifecycle
events (``task.published``, ``task.completed`` ...).

The plumbing is intentionally best-effort:

* If no runtime is bound to :data:`_active_runtime`, the callback returns
  silently.  This keeps DSPy calls outside a memfun runtime (eg. unit
  tests, the optimizer playground) zero-cost.
* If the EventBus :meth:`publish` raises, the exception is swallowed.
  **Observability MUST NOT break the agent.**
* Prompt / completion previews are clamped to 200 characters each so
  sensitive data doesn't leak through dashboards.

Two :class:`~contextvars.ContextVar`'s drive attribution:

* :data:`_active_runtime` — set once at chat/swarm startup so the callback
  can reach ``runtime.event_bus``.
* :data:`_active_agent_context` — set by each specialist's ``handle()``
  method so calls are attributed to the right ``agent_name`` /
  ``task_id`` / ``workflow_id``.

Both default to ``None``.  Callers wrap a region with
:func:`agent_context` (a context manager that resets on exit, so nested
calls don't leak attribution to sibling tasks).
"""

from __future__ import annotations

import contextlib
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from memfun_runtime.context import RuntimeContext

logger = get_logger("agent.telemetry")

# Maximum chars for prompt / completion previews stored in the event.
# Keep these tight: dashboards must never spill sensitive data.
_MAX_PREVIEW_CHARS = 200

# ── Context vars (per-task attribution + active runtime) ────────────

_active_runtime: ContextVar[RuntimeContext | None] = ContextVar(
    "memfun_active_runtime",
    default=None,
)
_active_agent_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "memfun_active_agent_context",
    default=None,
)
_call_start_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "memfun_lm_call_state",
    default=None,
)


def set_active_runtime(runtime: RuntimeContext | None) -> None:
    """Bind the memfun :class:`RuntimeContext` for telemetry publishing.

    Called once at chat/swarm/agent-command startup, right after the
    runtime is built.  Setting to ``None`` disables event publishing.
    """
    _active_runtime.set(runtime)


def get_active_runtime() -> RuntimeContext | None:
    """Return the currently-bound runtime (or ``None`` if unset)."""
    return _active_runtime.get()


@contextlib.contextmanager
def agent_context(
    *,
    agent_name: str | None = None,
    worker_id: str | None = None,
    task_id: str | None = None,
    workflow_id: str | None = None,
) -> Iterator[None]:
    """Attribute LLM calls inside the ``with`` block to a specific agent.

    Specialist agents wrap the body of ``handle()`` with this so every
    DSPy call (plan, code, verify, fix, polish …) gets the right
    ``agent_name`` / ``task_id`` / ``workflow_id`` on the resulting
    ``llm.call`` event.

    The previous attribution is restored on exit so nested workflows
    (eg. an orchestrator calling a sub-specialist) don't leak the
    inner attribution back to the outer scope.
    """
    payload = {
        "agent_name": agent_name,
        "worker_id": worker_id,
        "task_id": task_id,
        "workflow_id": workflow_id,
    }
    token = _active_agent_context.set(payload)
    try:
        yield
    finally:
        _active_agent_context.reset(token)


# ── Helpers ────────────────────────────────────────────────────────


def _truncate(text: str | None, limit: int = _MAX_PREVIEW_CHARS) -> str:
    """Clamp *text* to *limit* chars + ``…`` suffix.

    Returns ``""`` for ``None``.  Used for both prompt and completion
    previews to keep events small and avoid leaking secrets.
    """
    if not text:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _extract_prompt_preview(inputs: dict[str, Any] | None) -> str:
    """Return a short preview of the *user* portion of the prompt.

    DSPy's LM ``__call__`` accepts either a raw ``prompt`` string or a
    ``messages`` list.  For messages we prefer the *last user message*
    since that's what an operator skimming a dashboard cares about.
    """
    if not inputs:
        return ""

    messages = inputs.get("messages")
    if isinstance(messages, list) and messages:
        # Walk from the end so we get the most recent user message.
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role in ("user", "system", None):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return _truncate(content)
                # Some providers use a list of content parts.
                if isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and isinstance(part.get("text"), str)
                            and part["text"].strip()
                        ):
                            return _truncate(part["text"])
        return ""

    prompt = inputs.get("prompt")
    if isinstance(prompt, str):
        return _truncate(prompt)
    return ""


def _extract_completion_and_usage(
    instance: Any,
    outputs: Any,
) -> tuple[str, int | None, int | None, str | None]:
    """Return ``(completion_preview, prompt_tokens, completion_tokens, model)``.

    Prefers the structured ``instance.history[-1]`` entry (DSPy stores
    raw response + usage there) and falls back to inspecting *outputs*
    if history is empty (eg. on exception paths).
    """
    completion = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None

    # Path 1: DSPy populates instance.history just before on_lm_end fires.
    history = getattr(instance, "history", None)
    if isinstance(history, list) and history:
        entry = history[-1]
        if isinstance(entry, dict):
            usage = entry.get("usage") or {}
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                if isinstance(pt, int):
                    prompt_tokens = pt
                if isinstance(ct, int):
                    completion_tokens = ct
            model = (
                entry.get("response_model") or entry.get("model")
                if isinstance(entry.get("response_model"), str)
                or isinstance(entry.get("model"), str)
                else None
            )
            outs = entry.get("outputs")
            if isinstance(outs, list) and outs:
                completion = _stringify_output(outs[0])

    # Path 2: fall back to scanning *outputs* directly.
    if not completion and outputs is not None:
        if isinstance(outputs, list) and outputs:
            completion = _stringify_output(outputs[0])
        else:
            completion = _stringify_output(outputs)

    if model is None:
        model = getattr(instance, "model", None)
        if not isinstance(model, str):
            model = None

    return _truncate(completion), prompt_tokens, completion_tokens, model


def _stringify_output(item: Any) -> str:
    """Convert a DSPy LM output element to a previewable string.

    DSPy LM returns either a list of strings (for text completion) or a
    list of dicts (for chat completion w/ ``model_type="chat"`` and
    ``finish_reason``).  We handle both.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # OpenAI chat-completion style: {"text": ..., "logprobs": ...}
        for key in ("text", "content", "message"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                inner = val.get("content")
                if isinstance(inner, str):
                    return inner
        # Last-ditch: stringify the whole dict.
        return str(item)
    return str(item)


# ── DSPy callback ──────────────────────────────────────────────────


class MemfunDSPyCallback:
    """DSPy :class:`BaseCallback` that publishes ``llm.call`` events.

    Subclassing :class:`dspy.utils.callback.BaseCallback` happens lazily
    at instantiation so importing this module doesn't force a DSPy
    import (memfun-runtime, memfun-core, and unit tests don't need it).
    """

    def __init__(self) -> None:
        # No state — we use the call_id contextvar to thread start->end.
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any] | None,
    ) -> None:
        """Record start timestamp + prompt preview for matching on_lm_end."""
        try:
            state = {
                "start": time.perf_counter(),
                "prompt_preview": _extract_prompt_preview(inputs),
                # Stash the instance so on_lm_end can read history/model.
                # DSPy doesn't pass `instance` to on_lm_end so we capture
                # it here.
                "instance": instance,
            }
            _call_start_state.set(state)
        except Exception:
            # Never let observability blow up the LM call.
            logger.debug("on_lm_start failed", exc_info=True)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any,
        exception: Exception | None = None,
    ) -> None:
        """Compute duration + token counts and publish to the event bus."""
        try:
            start_state = _call_start_state.get() or {}
            start = start_state.get("start")
            duration_ms: float | None = None
            if isinstance(start, float):
                duration_ms = (time.perf_counter() - start) * 1000.0

            instance = start_state.get("instance")
            (
                completion_preview,
                prompt_tokens,
                completion_tokens,
                model,
            ) = _extract_completion_and_usage(
                instance=instance,
                outputs=outputs,
            )

            success = exception is None
            error = str(exception) if exception else None

            ctx = _active_agent_context.get() or {}
            event_payload = {
                "event_type": "llm.call",
                "agent_name": ctx.get("agent_name"),
                "worker_id": ctx.get("worker_id"),
                "task_id": ctx.get("task_id"),
                "workflow_id": ctx.get("workflow_id"),
                "model": model,
                "prompt_preview": start_state.get("prompt_preview", ""),
                "completion_preview": completion_preview,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "ts": time.time(),
            }

            publish_llm_call_event(event_payload)
        except Exception:
            logger.debug("on_lm_end failed", exc_info=True)
        finally:
            # Always clear the per-call start state so a leak doesn't
            # carry stale timing into the next call.
            with contextlib.suppress(Exception):
                _call_start_state.set(None)


def make_dspy_callback() -> Any:
    """Return a callback instance that inherits from DSPy's BaseCallback.

    Lazy-importing DSPy keeps this module usable in environments where
    DSPy is absent (eg. unit tests that mock the runtime).  If DSPy is
    unavailable we return a plain :class:`MemfunDSPyCallback` — the
    DSPy ``with_callbacks`` decorator only calls our handler methods
    via ``_get_on_end_handler``, which uses ``getattr``, so the
    inheritance is purely for ``isinstance`` checks DSPy doesn't
    actually perform on callbacks.
    """
    try:
        from dspy.utils.callback import BaseCallback as _BaseCallback

        class _Bound(_BaseCallback, MemfunDSPyCallback):  # type: ignore[misc, valid-type]
            """Memfun callback that satisfies DSPy's BaseCallback type."""

            def __init__(self) -> None:
                _BaseCallback.__init__(self)
                MemfunDSPyCallback.__init__(self)

        return _Bound()
    except Exception:
        logger.debug("DSPy BaseCallback unavailable", exc_info=True)
        return MemfunDSPyCallback()


# ── Publishing ─────────────────────────────────────────────────────


def publish_llm_call_event(event_payload: dict[str, Any]) -> None:
    """Best-effort publish of an ``llm.call`` event to the memfun event bus.

    Public for tests — they can call this directly without setting up
    DSPy.  Production code goes through :class:`MemfunDSPyCallback`.

    The publish runs on the active asyncio loop via ``loop.create_task``
    when called from sync code (DSPy's LM ``__call__`` is sync).  If no
    loop is running we silently drop the event — never raise.
    """
    runtime = _active_runtime.get()
    if runtime is None:
        return

    try:
        from memfun_runtime.distributed import (
            EVENT_TOPIC,
            DistributedEvent,
        )

        event = DistributedEvent(
            event_type=event_payload.get("event_type", "llm.call"),
            task_id=event_payload.get("task_id"),
            agent_name=event_payload.get("agent_name"),
            worker_id=event_payload.get("worker_id"),
            success=event_payload.get("success"),
            duration_ms=event_payload.get("duration_ms"),
            # `detail` lets the lifecycle log row show *something* when
            # an operator hasn't enabled the "LLM calls" filter yet.
            detail=event_payload.get("prompt_preview", "")[:200] or None,
            ts=event_payload.get("ts") or time.time(),
            workflow_id=event_payload.get("workflow_id"),
            model=event_payload.get("model"),
            prompt_preview=event_payload.get("prompt_preview", ""),
            completion_preview=event_payload.get("completion_preview", ""),
            prompt_tokens=event_payload.get("prompt_tokens"),
            completion_tokens=event_payload.get("completion_tokens"),
            error=event_payload.get("error"),
        )
        _publish_bytes_best_effort(runtime, EVENT_TOPIC, event.to_bytes())
    except Exception:
        logger.debug("publish_llm_call_event failed", exc_info=True)


def _publish_bytes_best_effort(
    runtime: RuntimeContext,
    topic: str,
    payload: bytes,
) -> None:
    """Schedule a fire-and-forget publish on the current event loop.

    DSPy LM ``__call__`` is a sync function so we can't ``await`` the
    coroutine directly.  We schedule it on the running loop (if any)
    and immediately return.  Errors during the publish are logged at
    debug level but never raised — observability is never load-bearing.
    """
    import asyncio

    coro = runtime.event_bus.publish(topic, payload)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No loop running → run synchronously in a private loop.  This
        # path is mostly hit by unit tests; in production the chat /
        # swarm / agent commands always have a running loop.
        try:
            asyncio.run(coro)
        except Exception:
            logger.debug("sync publish failed", exc_info=True)
        return

    task = loop.create_task(coro)
    # Attach a noop done callback so unawaited-task warnings don't
    # spam the logs if the publish raises.
    task.add_done_callback(_swallow_task_exception)


def _swallow_task_exception(task: Any) -> None:
    """Drain a task's exception so asyncio doesn't log it as unhandled."""
    with contextlib.suppress(Exception):
        task.exception()


# ── Registration helper ────────────────────────────────────────────


def install_dspy_callback() -> None:
    """Register :class:`MemfunDSPyCallback` as a global DSPy callback.

    Safe to call multiple times — duplicate registrations are filtered
    by class name so the chat REPL / swarm / agent CLI commands can all
    invoke this at startup without piling up redundant handlers.
    """
    try:
        import dspy

        existing = list(getattr(dspy.settings, "callbacks", []) or [])
        for cb in existing:
            if cb.__class__.__name__ == "_Bound" or isinstance(
                cb,
                MemfunDSPyCallback,
            ):
                # Already installed.
                return

        existing.append(make_dspy_callback())
        dspy.configure(callbacks=existing)
        logger.debug("MemfunDSPyCallback installed (%d total)", len(existing))
    except Exception:
        logger.debug("install_dspy_callback failed", exc_info=True)
