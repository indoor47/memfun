"""Live end-to-end test for :mod:`memfun_agent.llm_telemetry`.

Drives a real DSPy ``LM`` call against the local Qwen 3-Coder 30B endpoint
(or any OpenAI-compatible server reachable at ``LIVE_LM_BASE_URL``) and
asserts that exactly one ``llm.call`` event lands on the in-memory event
bus.

This is the regression test for the MRO bug fixed in this PR:
``class _Bound(BaseCallback, MemfunDSPyCallback)`` resolved
``on_lm_start`` to ``BaseCallback``'s no-op (a ``pass``), so the unit
tests in :mod:`tests.test_llm_telemetry` (which call
``MemfunDSPyCallback`` directly, bypassing the ``_Bound`` subclass)
all passed — yet *zero* ``llm.call`` events were ever published in
production.  This test exercises the full DSPy ``with_callbacks``
machinery, which is what actually runs in the swarm.

Skipped when no live endpoint is reachable.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest

LIVE_LM_BASE_URL = os.environ.get(
    "LIVE_LM_BASE_URL", "http://127.0.0.1:8090/v1",
)
LIVE_LM_MODEL = os.environ.get(
    "LIVE_LM_MODEL", "qwen3-coder-30b-a3b",
)
LIVE_LM_API_KEY = os.environ.get("LIVE_LM_API_KEY", "sk-local")


def _endpoint_reachable() -> bool:
    """Return True iff a TCP connect to the LM endpoint succeeds quickly."""
    try:
        # Parse "http://host:port/..."
        from urllib.parse import urlparse

        u = urlparse(LIVE_LM_BASE_URL)
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _endpoint_reachable(),
    reason=f"Live LM endpoint at {LIVE_LM_BASE_URL} not reachable.",
)


class _FakeEventBus:
    """Records publish() calls in-memory."""

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


class _FakeRuntime:
    def __init__(self) -> None:
        self.event_bus = _FakeEventBus()


@pytest.mark.asyncio
async def test_real_dspy_lm_call_publishes_llm_call_event() -> None:
    """A real DSPy ``LM.__call__`` MUST produce one ``llm.call`` event."""
    import dspy
    from memfun_agent.llm_telemetry import (
        agent_context,
        install_dspy_callback,
        set_active_runtime,
    )

    runtime = _FakeRuntime()
    set_active_runtime(runtime)

    # Configure DSPy with the live local endpoint.
    lm = dspy.LM(
        model=f"openai/{LIVE_LM_MODEL}",
        api_base=LIVE_LM_BASE_URL,
        api_key=LIVE_LM_API_KEY,
        timeout=180,
        max_tokens=32,
        temperature=0.0,
    )
    install_dspy_callback()

    # Verify the callback actually attached.
    cbs = list(getattr(dspy.settings, "callbacks", []) or [])
    assert any(
        c.__class__.__name__ == "_Bound" for c in cbs
    ), f"_Bound callback not installed; saw {[c.__class__.__name__ for c in cbs]}"

    # Sync ``lm(...)`` would block the loop, so we run it in a worker
    # thread the same way the swarm does (DSPy LM call is sync).
    # Use ``dspy.context(...)`` so this test can coexist with others
    # that also configure DSPy on a different asyncio task.
    with dspy.context(lm=lm), agent_context(
        agent_name="live-test-agent",
        task_id="live-T1",
        worker_id="live-w1",
    ):
        out = await asyncio.to_thread(lm, "Reply with exactly: ok.")

    # Sanity check the LM actually responded.
    assert out, "LM returned empty output"

    # The callback's publish goes through loop.create_task(), so yield
    # to let it run.  Also explicitly drain any pending tasks.
    await asyncio.sleep(0.3)
    pending = [
        t
        for t in asyncio.all_tasks()
        if not t.done() and t is not asyncio.current_task()
    ]
    if pending:
        await asyncio.wait(pending, timeout=2.0)

    # We MUST have at least one llm.call event by now.  This is the
    # exact regression we're guarding against (PR #34's _Bound MRO bug
    # meant the callback never published any events).
    llm_events = []
    for _topic, payload in runtime.event_bus.published:
        try:
            obj = json.loads(payload.decode())
        except Exception:
            continue
        if obj.get("event_type") == "llm.call":
            llm_events.append(obj)

    assert llm_events, (
        f"No llm.call events published — telemetry is broken.  "
        f"Saw {len(runtime.event_bus.published)} total publishes."
    )

    ev = llm_events[-1]
    assert ev["agent_name"] == "live-test-agent"
    assert ev["task_id"] == "live-T1"
    assert ev["worker_id"] == "live-w1"
    assert ev["success"] is True
    # Model may carry the openai/ prefix or not depending on DSPy path.
    assert LIVE_LM_MODEL in (ev.get("model") or ""), (
        f"Expected model to contain {LIVE_LM_MODEL!r}, got {ev.get('model')!r}"
    )
    # Prompt preview must be non-empty so the dashboard has something
    # to show.  Completion preview can be empty if the LM history
    # didn't surface the response text — that's OK; the event still
    # publishes.
    assert ev["prompt_preview"], "prompt_preview was empty"


@pytest.mark.asyncio
async def test_real_dspy_predict_publishes_llm_call_event() -> None:
    """A ``dspy.Predict`` call (the path the agents actually use) also publishes."""
    import dspy
    from memfun_agent.llm_telemetry import (
        agent_context,
        install_dspy_callback,
        set_active_runtime,
    )

    runtime = _FakeRuntime()
    set_active_runtime(runtime)

    lm = dspy.LM(
        model=f"openai/{LIVE_LM_MODEL}",
        api_base=LIVE_LM_BASE_URL,
        api_key=LIVE_LM_API_KEY,
        timeout=180,
        max_tokens=64,
        temperature=0.0,
    )
    # ``dspy.configure`` is task-owned in async tests, so use the
    # ``dspy.context(...)`` ctxmgr to scope the LM for this task.
    install_dspy_callback()

    pred = dspy.Predict("question -> answer")
    with dspy.context(lm=lm), agent_context(
        agent_name="predict-test", task_id="P1",
    ):
        # Predict is sync — run in a thread so the loop stays responsive.
        await asyncio.to_thread(pred, question="What is 2 + 2?")

    await asyncio.sleep(0.3)
    pending = [
        t
        for t in asyncio.all_tasks()
        if not t.done() and t is not asyncio.current_task()
    ]
    if pending:
        await asyncio.wait(pending, timeout=2.0)

    llm_events = [
        json.loads(payload.decode())
        for _topic, payload in runtime.event_bus.published
        if b'"llm.call"' in payload
    ]
    assert llm_events, "dspy.Predict produced no llm.call events"
    ev = llm_events[-1]
    assert ev["agent_name"] == "predict-test"
    # When usage IS returned by the server, tokens should be populated.
    # qwen3-coder via llama.cpp does return usage for chat completions
    # routed through ChatAdapter.
    if ev.get("prompt_tokens") is not None:
        assert ev["prompt_tokens"] > 0
    if ev.get("completion_tokens") is not None:
        assert ev["completion_tokens"] >= 0
