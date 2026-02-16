"""Distributed orchestrator: dispatches tasks via Redis event bus.

Instead of calling ``agent.handle(task)`` locally, the
:class:`DistributedOrchestrator` publishes each task to a Redis stream
and waits for a result on a per-task reply stream.  Remote
:mod:`memfun_runtime.worker` processes pick up tasks and push results
back.

This enables true multi-process, multi-machine agent coordination
while keeping the same :class:`TaskMessage` / :class:`TaskResult` API
that the in-process :class:`AgentOrchestrator` uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage, TaskResult

if TYPE_CHECKING:
    from memfun_runtime.protocols import EventBusAdapter

logger = get_logger("distributed")

# ── Stream topic conventions ─────────────────────────────────────────
TASK_TOPIC = "memfun.distributed.tasks"
RESULT_TOPIC_PREFIX = "memfun.distributed.results."
EVENT_TOPIC = "memfun.distributed.events"


def _result_topic(task_id: str) -> str:
    return f"{RESULT_TOPIC_PREFIX}{task_id}"


# ── Serialization helpers ────────────────────────────────────────────

def task_to_bytes(task: TaskMessage) -> bytes:
    """Serialize a TaskMessage to JSON bytes for the event bus."""
    return json.dumps({
        "task_id": task.task_id,
        "agent_id": task.agent_id,
        "payload": task.payload,
        "correlation_id": task.correlation_id,
        "parent_task_id": task.parent_task_id,
        "timestamp": task.timestamp,
    }).encode()


def bytes_to_task(data: bytes) -> TaskMessage:
    """Deserialize JSON bytes back to a TaskMessage."""
    d = json.loads(data)
    return TaskMessage(
        task_id=d["task_id"],
        agent_id=d["agent_id"],
        payload=d.get("payload", {}),
        correlation_id=d.get("correlation_id"),
        parent_task_id=d.get("parent_task_id"),
        timestamp=d.get("timestamp", time.time()),
    )


def result_to_bytes(result: TaskResult) -> bytes:
    """Serialize a TaskResult to JSON bytes for the event bus."""
    return json.dumps({
        "task_id": result.task_id,
        "agent_id": result.agent_id,
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
        "timestamp": result.timestamp,
    }).encode()


def bytes_to_result(data: bytes) -> TaskResult:
    """Deserialize JSON bytes back to a TaskResult."""
    d = json.loads(data)
    return TaskResult(
        task_id=d["task_id"],
        agent_id=d["agent_id"],
        success=d["success"],
        result=d.get("result", {}),
        error=d.get("error"),
        duration_ms=d.get("duration_ms", 0.0),
        timestamp=d.get("timestamp", time.time()),
    )


# ── Event types for the dashboard ────────────────────────────────────

@dataclass(slots=True)
class DistributedEvent:
    """An event emitted to the dashboard stream."""

    # task.published, task.picked_up, task.completed, worker.online, worker.offline
    event_type: str
    task_id: str | None = None
    agent_name: str | None = None
    worker_id: str | None = None
    success: bool | None = None
    duration_ms: float | None = None
    detail: str | None = None
    ts: float = 0.0

    def to_bytes(self) -> bytes:
        self.ts = self.ts or time.time()
        return json.dumps({
            "event_type": self.event_type,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "worker_id": self.worker_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
            "ts": self.ts,
        }).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> DistributedEvent:
        d = json.loads(data)
        return cls(**d)


# ── Distributed Orchestrator ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configuration for the distributed orchestrator."""

    default_timeout_seconds: float = 300.0
    result_poll_interval_ms: int = 100


class DistributedOrchestrator:
    """Orchestrates agents via Redis streams (distributed, multi-process).

    Publishes tasks to ``TASK_TOPIC`` and polls per-task result topics.
    Worker processes (see :mod:`memfun_runtime.worker`) subscribe to the
    task stream, execute agent.handle(), and push results back.
    """

    def __init__(
        self,
        event_bus: EventBusAdapter,
        config: DistributedConfig | None = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or DistributedConfig()

    async def dispatch(
        self,
        task: TaskMessage,
        agent_name: str,
        *,
        timeout: float | None = None,
    ) -> TaskResult:
        """Publish a task and wait for the remote worker to return a result."""
        timeout = timeout or self._config.default_timeout_seconds

        # Stamp the target agent into the task
        stamped = TaskMessage(
            task_id=task.task_id,
            agent_id=agent_name,
            payload=task.payload,
            correlation_id=task.correlation_id,
            parent_task_id=task.parent_task_id,
        )

        # Emit dashboard event
        await self._emit(DistributedEvent(
            event_type="task.published",
            task_id=task.task_id,
            agent_name=agent_name,
            detail=task.payload.get("query", "")[:200],
        ))

        # Publish task
        await self._bus.publish(TASK_TOPIC, task_to_bytes(stamped), key=task.task_id)
        logger.info("Published task %s -> %s", task.task_id, agent_name)

        # Poll for result on the reply topic
        reply_topic = _result_topic(task.task_id)
        result = await self._wait_for_result(reply_topic, task, agent_name, timeout)

        # Emit completion event
        await self._emit(DistributedEvent(
            event_type="task.completed",
            task_id=task.task_id,
            agent_name=agent_name,
            success=result.success,
            duration_ms=result.duration_ms,
        ))

        return result

    async def fan_out(
        self,
        tasks: list[TaskMessage],
        agent_names: list[str],
        *,
        timeout: float | None = None,
    ) -> list[TaskResult]:
        """Publish N tasks in parallel and wait for all results."""
        if len(tasks) != len(agent_names):
            raise ValueError("tasks and agent_names must have the same length")

        coros = [
            self.dispatch(t, n, timeout=timeout)
            for t, n in zip(tasks, agent_names, strict=False)
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [
            self._exception_to_result(r, tasks[i])
            if isinstance(r, BaseException)
            else r
            for i, r in enumerate(results)
        ]

    async def _wait_for_result(
        self,
        reply_topic: str,
        task: TaskMessage,
        agent_name: str,
        timeout: float,
    ) -> TaskResult:
        """Poll the reply stream until a result arrives or timeout."""
        deadline = time.monotonic() + timeout

        async for msg in self._bus.subscribe(reply_topic):
            try:
                return bytes_to_result(msg.payload)
            except Exception as exc:
                logger.warning("Bad result message: %s", exc)
                continue

            if time.monotonic() > deadline:
                break

        logger.error("Task %s timed out after %.0fs", task.task_id, timeout)
        return TaskResult(
            task_id=task.task_id,
            agent_id=agent_name,
            success=False,
            error=f"Timeout after {timeout}s waiting for worker result",
        )

    async def _emit(self, event: DistributedEvent) -> None:
        """Publish a dashboard event (best-effort)."""
        with contextlib.suppress(Exception):
            await self._bus.publish(EVENT_TOPIC, event.to_bytes())

    @staticmethod
    def _exception_to_result(exc: BaseException, task: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            success=False,
            error=str(exc),
        )
