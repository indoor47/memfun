"""Tests for :mod:`memfun_agent.llm_telemetry`.

Covers:

1. ``MemfunDSPyCallback`` publishes an ``llm.call`` event with the right
   shape when called via simulated DSPy hooks.
2. ``agent_context`` correctly threads attribution into the event payload.
3. When no runtime is bound, the callback is a no-op (best-effort guarantee).
4. ``install_dspy_callback`` is idempotent.
5. Token counts are extracted from ``instance.history[-1]['usage']``.
6. Prompt / completion previews are truncated to ``_MAX_PREVIEW_CHARS``
   so sensitive data never leaks through dashboards.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

# Module under test.
from memfun_agent.llm_telemetry import (
    _MAX_PREVIEW_CHARS,
    MemfunDSPyCallback,
    _truncate,
    agent_context,
    install_dspy_callback,
    make_dspy_callback,
    publish_llm_call_event,
    set_active_runtime,
)

# ── Fake event bus / runtime (in-process, captures publishes) ────────


class _FakeEventBus:
    """Captures every publish for later inspection."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(
        self,
        topic: str,
        message: bytes,
        key: str | None = None,
    ) -> str:
        self.published.append((topic, message))
        return "ok"


@dataclass
class _FakeRuntime:
    event_bus: _FakeEventBus


@pytest.fixture
def fake_runtime() -> _FakeRuntime:
    """A minimal runtime that records publishes."""
    return _FakeRuntime(event_bus=_FakeEventBus())


@pytest.fixture(autouse=True)
def _reset_runtime_after_test():
    """Always clear the contextvar after each test so state never leaks."""
    yield
    set_active_runtime(None)


# ── Fake DSPy LM instance (mimics history + model) ──────────────────


class _FakeLM:
    """Mimics ``dspy.LM`` enough for the callback's history-scraping path."""

    def __init__(
        self,
        model: str = "openai/qwen3-coder-30b-a3b",
        prompt_tokens: int = 1466,
        completion_tokens: int = 412,
        completion_text: str = "def clamp(x, lo, hi): ...",
    ) -> None:
        self.model = model
        self.history = [
            {
                "model": model,
                "response_model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "outputs": [completion_text],
            }
        ]


# ── _truncate ────────────────────────────────────────────────────────


def test_truncate_clamps_to_limit() -> None:
    s = "x" * (_MAX_PREVIEW_CHARS + 100)
    out = _truncate(s)
    assert len(out) == _MAX_PREVIEW_CHARS + 1  # +1 for the ellipsis
    assert out.endswith("…")


def test_truncate_short_string_unchanged() -> None:
    assert _truncate("hello") == "hello"


def test_truncate_none_returns_empty_string() -> None:
    assert _truncate(None) == ""


# ── Callback event-emission path ────────────────────────────────────


def test_callback_publishes_event_with_expected_shape(
    fake_runtime: _FakeRuntime,
) -> None:
    """End-to-end: on_lm_start + on_lm_end -> single ``llm.call`` event."""
    set_active_runtime(fake_runtime)
    cb = MemfunDSPyCallback()

    instance = _FakeLM()
    inputs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Implement clamp(x, lo, hi)."},
        ]
    }

    with agent_context(
        agent_name="coder-agent",
        task_id="t-abc123",
        workflow_id="wf-1",
        worker_id="w1",
    ):
        cb.on_lm_start(call_id="c1", instance=instance, inputs=inputs)
        cb.on_lm_end(call_id="c1", outputs=["def clamp(...): ..."])

    # Give the loop a tick to drain the create_task() publish.
    # In sync tests there's no loop, so publish runs via asyncio.run().
    assert fake_runtime.event_bus.published, "no events published"
    topic, payload = fake_runtime.event_bus.published[-1]
    from memfun_runtime.distributed import EVENT_TOPIC

    assert topic == EVENT_TOPIC

    wire = json.loads(payload.decode())
    assert wire["event_type"] == "llm.call"
    assert wire["agent_name"] == "coder-agent"
    assert wire["task_id"] == "t-abc123"
    assert wire["workflow_id"] == "wf-1"
    assert wire["worker_id"] == "w1"
    assert wire["model"] == "openai/qwen3-coder-30b-a3b"
    assert wire["prompt_tokens"] == 1466
    assert wire["completion_tokens"] == 412
    assert wire["success"] is True
    assert wire["error"] is None
    # The last *user* message is the prompt preview, not the system one.
    assert "Implement clamp(x, lo, hi)." in wire["prompt_preview"]
    assert wire["completion_preview"]  # any non-empty content


def test_callback_records_exception_path(
    fake_runtime: _FakeRuntime,
) -> None:
    """An ``exception`` in on_lm_end -> ``success=False`` + ``error=<str>``."""
    set_active_runtime(fake_runtime)
    cb = MemfunDSPyCallback()

    instance = _FakeLM()
    cb.on_lm_start(
        call_id="c2",
        instance=instance,
        inputs={"prompt": "go"},
    )
    cb.on_lm_end(
        call_id="c2",
        outputs=None,
        exception=RuntimeError("rate limited"),
    )

    assert fake_runtime.event_bus.published
    wire = json.loads(
        fake_runtime.event_bus.published[-1][1].decode(),
    )
    assert wire["success"] is False
    assert wire["error"] == "rate limited"


def test_callback_truncates_prompt_and_completion_previews(
    fake_runtime: _FakeRuntime,
) -> None:
    """Previews MUST be clipped to ``_MAX_PREVIEW_CHARS`` (no secret leakage)."""
    set_active_runtime(fake_runtime)
    cb = MemfunDSPyCallback()

    long_prompt = "secret-token-" + ("A" * 5000)
    long_output = "B" * 5000
    instance = _FakeLM(completion_text=long_output)

    cb.on_lm_start(
        call_id="c3",
        instance=instance,
        inputs={"messages": [{"role": "user", "content": long_prompt}]},
    )
    cb.on_lm_end(call_id="c3", outputs=[long_output])

    wire = json.loads(
        fake_runtime.event_bus.published[-1][1].decode(),
    )
    # +1 for the trailing ellipsis on truncation.
    assert len(wire["prompt_preview"]) == _MAX_PREVIEW_CHARS + 1
    assert len(wire["completion_preview"]) == _MAX_PREVIEW_CHARS + 1


def test_callback_is_noop_when_no_runtime_bound() -> None:
    """If the active runtime is None the callback never raises and never publishes."""
    set_active_runtime(None)
    cb = MemfunDSPyCallback()

    # Should not raise.
    cb.on_lm_start(call_id="x", instance=_FakeLM(), inputs={"prompt": "?"})
    cb.on_lm_end(call_id="x", outputs=["ok"])

    # If a publish slipped through there's no fake bus to record it,
    # so the assertion is implicit: no exception.


def test_callback_survives_inputs_being_none(
    fake_runtime: _FakeRuntime,
) -> None:
    """``inputs=None`` and weird outputs MUST NOT raise — observability is best-effort."""
    set_active_runtime(fake_runtime)
    cb = MemfunDSPyCallback()
    cb.on_lm_start(call_id="x", instance=_FakeLM(), inputs=None)
    cb.on_lm_end(call_id="x", outputs={"weird": object()})

    # The publish either succeeded or was silently dropped — either way
    # no exception escaped.
    assert True


# ── agent_context propagation ────────────────────────────────────────


def test_agent_context_restores_prior_attribution(
    fake_runtime: _FakeRuntime,
) -> None:
    """Nested ``agent_context`` must not leak attribution to the outer scope."""
    set_active_runtime(fake_runtime)
    cb = MemfunDSPyCallback()
    instance = _FakeLM()

    with agent_context(agent_name="outer-agent", task_id="t-out"):
        # Inner attribution → only the inner call should see it.
        with agent_context(agent_name="inner-agent", task_id="t-in"):
            cb.on_lm_start(
                call_id="i",
                instance=instance,
                inputs={"prompt": "p"},
            )
            cb.on_lm_end(call_id="i", outputs=["o"])

        # Back to outer attribution.
        cb.on_lm_start(
            call_id="o",
            instance=instance,
            inputs={"prompt": "p"},
        )
        cb.on_lm_end(call_id="o", outputs=["o"])

    assert len(fake_runtime.event_bus.published) == 2
    inner = json.loads(
        fake_runtime.event_bus.published[0][1].decode(),
    )
    outer = json.loads(
        fake_runtime.event_bus.published[1][1].decode(),
    )
    assert inner["agent_name"] == "inner-agent"
    assert inner["task_id"] == "t-in"
    assert outer["agent_name"] == "outer-agent"
    assert outer["task_id"] == "t-out"


# ── publish_llm_call_event direct ────────────────────────────────────


def test_publish_llm_call_event_writes_to_event_bus(
    fake_runtime: _FakeRuntime,
) -> None:
    set_active_runtime(fake_runtime)
    publish_llm_call_event(
        {
            "event_type": "llm.call",
            "agent_name": "coder-agent",
            "task_id": "t1",
            "model": "openai/test",
            "prompt_preview": "hello",
            "completion_preview": "world",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "duration_ms": 123.4,
            "success": True,
            "ts": 1000.0,
        }
    )
    assert fake_runtime.event_bus.published
    wire = json.loads(
        fake_runtime.event_bus.published[-1][1].decode(),
    )
    assert wire["event_type"] == "llm.call"
    assert wire["prompt_tokens"] == 10
    assert wire["completion_tokens"] == 20


def test_publish_llm_call_event_swallows_bus_errors() -> None:
    """If the EventBus raises, ``publish_llm_call_event`` MUST NOT propagate."""

    class _BrokenBus:
        async def publish(
            self,
            topic: str,
            message: bytes,
            key: str | None = None,
        ) -> str:
            raise RuntimeError("boom")

    set_active_runtime(_FakeRuntime(event_bus=_BrokenBus()))  # type: ignore[arg-type]
    # Should not raise.
    publish_llm_call_event(
        {
            "event_type": "llm.call",
            "agent_name": "x",
            "prompt_preview": "p",
            "completion_preview": "c",
        }
    )


# ── install_dspy_callback (idempotent) ──────────────────────────────


def test_install_dspy_callback_is_idempotent() -> None:
    """Calling ``install_dspy_callback`` twice must not register twice."""
    try:
        import dspy
    except Exception:
        pytest.skip("DSPy not installed in this environment")

    # Clear any prior callbacks for isolation.
    dspy.configure(callbacks=[])

    install_dspy_callback()
    after_one = list(getattr(dspy.settings, "callbacks", []) or [])
    install_dspy_callback()
    after_two = list(getattr(dspy.settings, "callbacks", []) or [])
    assert len(after_two) == len(after_one)


def test_make_dspy_callback_returns_callable_object() -> None:
    """The returned object has both ``on_lm_start`` and ``on_lm_end``."""
    cb = make_dspy_callback()
    assert hasattr(cb, "on_lm_start")
    assert hasattr(cb, "on_lm_end")


# ── Async publish path ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_uses_running_loop(
    fake_runtime: _FakeRuntime,
) -> None:
    """When called from inside an event loop, the publish is scheduled
    as a task — not run via asyncio.run() (which would error).
    """
    set_active_runtime(fake_runtime)
    publish_llm_call_event(
        {
            "event_type": "llm.call",
            "agent_name": "a",
            "prompt_preview": "p",
            "completion_preview": "c",
        }
    )
    # The publish is dispatched as a task — yield so it runs.
    await asyncio.sleep(0.01)
    assert fake_runtime.event_bus.published


# ── DistributedEvent shape (forward-compat) ──────────────────────────


def test_distributed_event_round_trips_llm_fields() -> None:
    """``DistributedEvent`` MUST preserve all llm.call fields across serialise."""
    from memfun_runtime.distributed import DistributedEvent

    event = DistributedEvent(
        event_type="llm.call",
        agent_name="coder-agent",
        task_id="t1",
        workflow_id="wf-1",
        model="openai/test",
        prompt_preview="hello world",
        completion_preview="response text",
        prompt_tokens=42,
        completion_tokens=24,
        duration_ms=1234.5,
        success=True,
    )
    payload = event.to_bytes()
    restored = DistributedEvent.from_bytes(payload)
    assert restored.event_type == "llm.call"
    assert restored.model == "openai/test"
    assert restored.prompt_preview == "hello world"
    assert restored.completion_preview == "response text"
    assert restored.prompt_tokens == 42
    assert restored.completion_tokens == 24
    assert restored.duration_ms == 1234.5
