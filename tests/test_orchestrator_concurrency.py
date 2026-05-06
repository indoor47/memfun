"""Bounded-concurrency tests for AgentOrchestrator and DistributedOrchestrator (issue #10).

These tests verify that ``OrchestratorConfig.max_concurrency`` (and the
analogous ``DistributedConfig.max_concurrency``) actually caps the
number of in-flight agent dispatches, regardless of fan-out width.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from memfun_core.types import TaskMessage, TaskResult
from memfun_runtime.agent import BaseAgent
from memfun_runtime.backends.memory import InProcessEventBus
from memfun_runtime.distributed import DistributedConfig, DistributedOrchestrator
from memfun_runtime.orchestrator import AgentOrchestrator, OrchestratorConfig

# ── Shared test doubles ─────────────────────────────────────────────


class _ConcurrencyProbe:
    """Tracks the number of currently-running ``handle`` calls."""

    def __init__(self) -> None:
        self.currently_running: int = 0
        self.peak: int = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.currently_running += 1
            if self.currently_running > self.peak:
                self.peak = self.currently_running

    async def exit(self) -> None:
        async with self._lock:
            self.currently_running -= 1


class _SleepingAgent(BaseAgent):
    """Test agent: sleeps for ``sleep_seconds`` and updates the probe."""

    def __init__(
        self,
        context: Any,
        probe: _ConcurrencyProbe,
        sleep_seconds: float,
    ) -> None:
        super().__init__(context)
        self._probe = probe
        self._sleep = sleep_seconds

    async def handle(self, task: TaskMessage) -> TaskResult:
        await self._probe.enter()
        try:
            await asyncio.sleep(self._sleep)
        finally:
            await self._probe.exit()
        return TaskResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            success=True,
            result={"ok": True},
        )


class _FakeManager:
    """Minimal AgentManager stand-in used by AgentOrchestrator tests.

    AgentOrchestrator only calls ``is_running`` and ``get_agent`` on the
    manager during dispatch, so a tiny dict-backed double is enough.
    """

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = agents

    def is_running(self, name: str) -> bool:
        return name in self._agents

    def get_agent(self, name: str) -> BaseAgent:
        return self._agents[name]


@pytest_asyncio.fixture
async def runtime_context() -> Any:
    """Minimal context that only exposes ``event_bus``.

    AgentOrchestrator only touches ``context.event_bus`` (for the
    best-effort dispatch event), so a duck-typed namespace is enough
    and avoids constructing the full 9-field ``RuntimeContext``.
    """
    return SimpleNamespace(event_bus=InProcessEventBus())


def _make_task(agent_id: str) -> TaskMessage:
    return TaskMessage(
        task_id=str(uuid.uuid4()),
        agent_id=agent_id,
        payload={"q": "x"},
    )


# ── AgentOrchestrator tests ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [2, 8])
async def test_fan_out_respects_concurrency_cap(
    runtime_context: Any, cap: int
) -> None:
    """Peak in-flight handles must never exceed ``max_concurrency``."""
    probe = _ConcurrencyProbe()
    agent = _SleepingAgent(runtime_context, probe, sleep_seconds=0.05)

    # Use a single shared agent name; the manager returns the same
    # agent for every name.  This isolates the test from any per-agent
    # serialization concerns and proves the cap is global.
    n_tasks = 20
    agents = {f"a{i}": agent for i in range(n_tasks)}
    manager = _FakeManager(agents)

    orch = AgentOrchestrator(
        context=runtime_context,
        manager=manager,  # type: ignore[arg-type]
        config=OrchestratorConfig(max_concurrency=cap, default_timeout_seconds=10.0),
    )

    tasks = [_make_task(name) for name in agents]
    names = list(agents)

    results = await orch.fan_out(tasks, names)

    assert len(results) == n_tasks
    assert all(r.success for r in results)
    # The cap is the only thing standing between this many parallel
    # sleeping agents and a peak of ``n_tasks``.
    assert probe.peak <= cap, (
        f"peak in-flight {probe.peak} exceeded cap {cap}"
    )
    # Sanity: with such a tight workload we should actually saturate
    # the cap (otherwise the test wouldn't prove anything).
    assert probe.peak == cap, (
        f"expected peak == cap={cap} given 20 tasks, got peak={probe.peak}"
    )


@pytest.mark.asyncio
async def test_fan_out_parallel_when_cap_is_high(
    runtime_context: Any,
) -> None:
    """Sanity check: a high cap must still allow full parallelism.

    With 4 tasks each sleeping 0.1s and cap=8, total wall time should
    be roughly 0.1s (one batch), not 0.4s (serialized).
    """
    probe = _ConcurrencyProbe()
    agent = _SleepingAgent(runtime_context, probe, sleep_seconds=0.1)
    agents = {f"p{i}": agent for i in range(4)}
    manager = _FakeManager(agents)

    orch = AgentOrchestrator(
        context=runtime_context,
        manager=manager,  # type: ignore[arg-type]
        config=OrchestratorConfig(max_concurrency=8, default_timeout_seconds=5.0),
    )
    tasks = [_make_task(name) for name in agents]

    start = time.monotonic()
    results = await orch.fan_out(tasks, list(agents))
    elapsed = time.monotonic() - start

    assert len(results) == 4
    assert all(r.success for r in results)
    # Full parallelism: ~0.1s.  Allow generous slack for CI jitter,
    # but be tight enough to fail if execution is serialized (~0.4s).
    assert elapsed < 0.3, f"expected parallel ~0.1s, got {elapsed:.3f}s"
    assert probe.peak == 4


@pytest.mark.asyncio
async def test_semaphore_is_shared_across_calls(
    runtime_context: Any,
) -> None:
    """Cap is per-orchestrator-instance, not per fan_out call.

    Two concurrent fan_out invocations on the same orchestrator must
    share the cap: 2 calls of 4 tasks each with cap=3 should still cap
    total in-flight at 3.
    """
    probe = _ConcurrencyProbe()
    agent = _SleepingAgent(runtime_context, probe, sleep_seconds=0.1)
    agents = {f"s{i}": agent for i in range(8)}
    manager = _FakeManager(agents)

    orch = AgentOrchestrator(
        context=runtime_context,
        manager=manager,  # type: ignore[arg-type]
        config=OrchestratorConfig(max_concurrency=3, default_timeout_seconds=5.0),
    )

    names = list(agents)
    half_a = names[:4]
    half_b = names[4:]
    tasks_a = [_make_task(n) for n in half_a]
    tasks_b = [_make_task(n) for n in half_b]

    results_a, results_b = await asyncio.gather(
        orch.fan_out(tasks_a, half_a),
        orch.fan_out(tasks_b, half_b),
    )

    assert all(r.success for r in [*results_a, *results_b])
    assert probe.peak <= 3, (
        f"peak in-flight {probe.peak} exceeded shared cap 3"
    )


# ── DistributedOrchestrator tests ───────────────────────────────────


class _FakeBus:
    """Minimal EventBusAdapter double for DistributedOrchestrator.

    Simulates a worker pool that takes ``sleep_seconds`` to handle each
    task, then publishes a TaskResult on the per-task reply topic.  The
    probe counts how many simulated workers are running concurrently.
    """

    def __init__(self, probe: _ConcurrencyProbe, sleep_seconds: float) -> None:
        self._probe = probe
        self._sleep = sleep_seconds
        # reply_topic -> queue of payloads
        self._queues: dict[str, asyncio.Queue[bytes]] = {}
        self._tasks: list[asyncio.Task[Any]] = []

    async def publish(self, topic: str, payload: bytes, *, key: str | None = None) -> None:
        # When the orchestrator publishes a task, schedule a fake
        # worker to "process" it and push a TaskResult onto the
        # corresponding reply topic.
        from memfun_runtime.distributed import (
            RESULT_TOPIC_PREFIX,
            TASK_TOPIC,
            bytes_to_task,
        )

        if topic == TASK_TOPIC:
            task = bytes_to_task(payload)
            reply_topic = f"{RESULT_TOPIC_PREFIX}{task.task_id}"
            self._tasks.append(asyncio.create_task(self._fake_worker(task, reply_topic)))

    async def _fake_worker(self, task: TaskMessage, reply_topic: str) -> None:
        from memfun_runtime.distributed import result_to_bytes

        await self._probe.enter()
        try:
            await asyncio.sleep(self._sleep)
        finally:
            await self._probe.exit()
        result = TaskResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            success=True,
            result={"ok": True},
        )
        q = self._queues.setdefault(reply_topic, asyncio.Queue())
        await q.put(result_to_bytes(result))

    async def subscribe(self, topic: str):  # type: ignore[no-untyped-def]
        # Yield a single message from the per-topic queue when it
        # arrives.  Mirrors the contract expected by ``_wait_for_result``.
        q = self._queues.setdefault(topic, asyncio.Queue())
        payload = await q.get()
        msg = type("Msg", (), {"payload": payload})()
        yield msg


@pytest.mark.asyncio
async def test_distributed_fan_out_respects_concurrency_cap() -> None:
    """DistributedOrchestrator must also cap in-flight dispatches."""
    probe = _ConcurrencyProbe()
    bus = _FakeBus(probe, sleep_seconds=0.05)

    orch = DistributedOrchestrator(
        event_bus=bus,  # type: ignore[arg-type]
        config=DistributedConfig(
            max_concurrency=2,
            default_timeout_seconds=10.0,
        ),
    )

    n_tasks = 10
    tasks = [_make_task(f"d{i}") for i in range(n_tasks)]
    names = [f"d{i}" for i in range(n_tasks)]

    results = await orch.fan_out(tasks, names)

    assert len(results) == n_tasks
    assert all(r.success for r in results), [r.error for r in results if not r.success]
    assert probe.peak <= 2, (
        f"distributed peak in-flight {probe.peak} exceeded cap 2"
    )
    assert probe.peak == 2, (
        f"expected distributed peak == cap=2 given {n_tasks} tasks, got {probe.peak}"
    )
