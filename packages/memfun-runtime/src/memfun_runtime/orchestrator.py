from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage, TaskResult

if TYPE_CHECKING:

    from memfun_runtime.context import RuntimeContext
    from memfun_runtime.lifecycle import AgentManager

logger = get_logger("orchestrator")


class RetryPolicy(Enum):
    """How the orchestrator retries failed dispatches."""

    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Tuning knobs for :class:`AgentOrchestrator`."""

    default_timeout_seconds: float = 120.0
    retry_policy: RetryPolicy = RetryPolicy.NONE
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    pipeline_carry_key: str = "pipeline_payload"


@dataclass(slots=True)
class _RouteEntry:
    """Internal record linking a task type to a target agent."""

    task_type: str
    agent_name: str
    priority: int = 0


class AgentOrchestrator:
    """Routes, fans-out, and pipelines tasks across managed agents.

    The orchestrator does **not** own agent instances -- it delegates
    execution to an :class:`AgentManager` which is responsible for the
    actual process lifecycle.  This separation keeps routing / coordination
    concerns cleanly decoupled from start / stop bookkeeping.

    Supports four dispatch patterns:

    * **direct dispatch** -- send a single task to a named agent.
    * **fan-out** -- send N tasks to N agents in parallel, collecting
      all results.
    * **pipeline** -- pass a task through a chain of agents
      sequentially, forwarding each result as the next task payload.
    * **round-robin** -- distribute incoming tasks evenly across a
      pre-configured agent group.
    """

    def __init__(
        self,
        context: RuntimeContext,
        manager: AgentManager,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._context = context
        self._manager = manager
        self._config = config or OrchestratorConfig()

        # Route table: task_type -> agent_name
        self._routes: dict[str, list[_RouteEntry]] = {}
        # Round-robin iterators keyed by group name
        self._rr_iters: dict[str, itertools.cycle[str]] = {}
        # Capability index: capability -> set of agent names
        self._capability_index: dict[str, set[str]] = {}
        # Pending dependencies: task_id -> (remaining upstream ids, future)
        self._pending: dict[str, tuple[set[str], asyncio.Future[list[TaskResult]]]] = {}

    # ── Route registration ──────────────────────────────────────────

    def register_route(
        self,
        task_type: str,
        agent_name: str,
        *,
        priority: int = 0,
    ) -> None:
        """Register *agent_name* as a handler for *task_type*.

        Higher *priority* values are preferred when multiple agents can
        handle the same type.
        """
        entry = _RouteEntry(
            task_type=task_type, agent_name=agent_name, priority=priority
        )
        self._routes.setdefault(task_type, []).append(entry)
        self._routes[task_type].sort(key=lambda e: e.priority, reverse=True)
        logger.debug(
            "Route registered: %s -> %s (priority %d)",
            task_type,
            agent_name,
            priority,
        )

    def register_capability(self, agent_name: str, capability: str) -> None:
        """Index an agent under a capability tag for capability-based routing."""
        self._capability_index.setdefault(capability, set()).add(agent_name)

    def register_round_robin_group(
        self, group: str, agent_names: list[str]
    ) -> None:
        """Create a named round-robin group from the given agents."""
        if not agent_names:
            raise ValueError("Round-robin group must contain at least one agent")
        self._rr_iters[group] = itertools.cycle(agent_names)
        logger.info("Round-robin group %r created with %s", group, agent_names)

    # ── Core dispatch patterns ──────────────────────────────────────

    async def dispatch(
        self,
        task: TaskMessage,
        agent_name: str,
        *,
        timeout: float | None = None,
    ) -> TaskResult:
        """Send *task* directly to a named agent and return its result.

        Applies the configured :class:`RetryPolicy` on failure.
        """
        timeout = timeout or self._config.default_timeout_seconds
        return await self._execute_with_retry(task, agent_name, timeout)

    async def fan_out(
        self,
        tasks: list[TaskMessage],
        agent_names: list[str],
        *,
        timeout: float | None = None,
    ) -> list[TaskResult]:
        """Execute *tasks* in parallel, one per agent.

        *tasks* and *agent_names* must have the same length.  Returns a
        list of :class:`TaskResult` in corresponding order.
        """
        if len(tasks) != len(agent_names):
            raise ValueError(
                f"tasks ({len(tasks)}) and agent_names ({len(agent_names)}) "
                "must have the same length"
            )

        timeout = timeout or self._config.default_timeout_seconds
        coros = [
            self._execute_with_retry(t, n, timeout)
            for t, n in zip(tasks, agent_names, strict=False)
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [
            self._exception_to_result(r, tasks[i])
            if isinstance(r, BaseException)
            else r
            for i, r in enumerate(results)
        ]

    async def pipeline(
        self,
        task: TaskMessage,
        agent_names: list[str],
        *,
        timeout: float | None = None,
    ) -> TaskResult:
        """Execute *task* through a sequential chain of agents.

        The output payload of each stage becomes the input payload of the
        next stage under the key configured by
        ``OrchestratorConfig.pipeline_carry_key``.
        """
        if not agent_names:
            raise ValueError("Pipeline must contain at least one agent")

        timeout = timeout or self._config.default_timeout_seconds
        carry_key = self._config.pipeline_carry_key
        current_task = task

        for idx, name in enumerate(agent_names):
            result = await self._execute_with_retry(
                current_task, name, timeout
            )
            if not result.success:
                logger.error(
                    "Pipeline stage %d (%s) failed: %s",
                    idx,
                    name,
                    result.error,
                )
                return result

            if idx < len(agent_names) - 1:
                next_payload = dict(current_task.payload)
                next_payload[carry_key] = result.result
                current_task = TaskMessage(
                    task_id=str(uuid.uuid4()),
                    agent_id=agent_names[idx + 1],
                    payload=next_payload,
                    correlation_id=task.correlation_id or task.task_id,
                    parent_task_id=result.task_id,
                )

        return result  # type: ignore[possibly-undefined]

    async def route(
        self,
        task: TaskMessage,
        *,
        timeout: float | None = None,
    ) -> TaskResult:
        """Route *task* to the best agent based on registered routes or capabilities.

        Resolution order:
        1. Exact ``task_type`` match from the route table (highest
           priority, first running agent wins).
        2. Capability match -- if ``task.payload`` contains a
           ``"capability"`` key, an agent registered for that capability
           is selected.
        3. Falls back to ``task.agent_id`` if none of the above match.
        """
        timeout = timeout or self._config.default_timeout_seconds
        agent_name = self._resolve_agent(task)
        return await self._execute_with_retry(task, agent_name, timeout)

    async def round_robin(
        self,
        group: str,
        task: TaskMessage,
        *,
        timeout: float | None = None,
    ) -> TaskResult:
        """Dispatch *task* to the next agent in a round-robin group.

        Raises ``KeyError`` if the group has not been registered.
        """
        rr = self._rr_iters.get(group)
        if rr is None:
            raise KeyError(
                f"Round-robin group {group!r} not registered. "
                f"Available: {', '.join(self._rr_iters) or '(none)'}"
            )
        agent_name = next(rr)

        # Skip agents that are not currently running (up to full cycle)
        attempts = 0
        while not self._manager.is_running(agent_name):
            agent_name = next(rr)
            attempts += 1
            if attempts > 100:
                raise RuntimeError(
                    f"No running agents found in round-robin group {group!r}"
                )

        timeout = timeout or self._config.default_timeout_seconds
        return await self._execute_with_retry(task, agent_name, timeout)

    # ── Dependency tracking ─────────────────────────────────────────

    async def dispatch_with_dependencies(
        self,
        task: TaskMessage,
        agent_name: str,
        *,
        depends_on: list[str],
        timeout: float | None = None,
    ) -> TaskResult:
        """Dispatch *task* only after all upstream *depends_on* task IDs complete.

        This creates an :class:`asyncio.Future` that will be resolved
        when the dependency set empties.  Call :meth:`notify_completion`
        for each upstream result.
        """
        if not depends_on:
            return await self.dispatch(task, agent_name, timeout=timeout)

        loop = asyncio.get_event_loop()
        future: asyncio.Future[list[TaskResult]] = loop.create_future()
        remaining = set(depends_on)
        self._pending[task.task_id] = (remaining, future)

        logger.debug(
            "Task %s waiting on %d upstream task(s): %s",
            task.task_id,
            len(remaining),
            remaining,
        )

        timeout = timeout or self._config.default_timeout_seconds
        try:
            await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(task.task_id, None)
            logger.error(
                "Task %s timed out waiting for dependencies", task.task_id
            )
            return self._timeout_result(task, agent_name)

        self._pending.pop(task.task_id, None)
        return await self.dispatch(task, agent_name, timeout=timeout)

    def notify_completion(self, upstream_task_id: str) -> None:
        """Signal that an upstream task has completed.

        Any pending tasks whose dependency set becomes empty will have
        their future resolved.
        """
        for _tid, (remaining, future) in list(self._pending.items()):
            if upstream_task_id in remaining:
                remaining.discard(upstream_task_id)
                if not remaining and not future.done():
                    future.set_result([])

    # ── Retry & execution helpers ───────────────────────────────────

    async def _execute_with_retry(
        self,
        task: TaskMessage,
        agent_name: str,
        timeout: float,
    ) -> TaskResult:
        """Execute a task against an agent, applying the retry policy."""
        policy = self._config.retry_policy
        max_attempts = 1 if policy is RetryPolicy.NONE else self._config.max_retries + 1

        last_error: str | None = None
        for attempt in range(max_attempts):
            if attempt > 0:
                delay = self._retry_delay(attempt)
                logger.info(
                    "Retrying task %s on %s (attempt %d, delay %.1fs)",
                    task.task_id,
                    agent_name,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                result = await self._execute_single(
                    task, agent_name, timeout
                )
                if result.success:
                    return result
                last_error = result.error
            except TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning(
                    "Task %s on %s timed out (attempt %d/%d)",
                    task.task_id,
                    agent_name,
                    attempt + 1,
                    max_attempts,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception(
                    "Task %s on %s raised (attempt %d/%d)",
                    task.task_id,
                    agent_name,
                    attempt + 1,
                    max_attempts,
                )

        return self._failure_result(task, agent_name, last_error)

    async def _execute_single(
        self,
        task: TaskMessage,
        agent_name: str,
        timeout: float,
    ) -> TaskResult:
        """Run a single task dispatch to an agent with a timeout."""
        if not self._manager.is_running(agent_name):
            raise RuntimeError(f"Agent {agent_name!r} is not running")

        agent = self._manager.get_agent(agent_name)
        start = time.monotonic()
        result = await asyncio.wait_for(agent.handle(task), timeout=timeout)
        elapsed_ms = (time.monotonic() - start) * 1000

        await self._emit_dispatch_event(task, agent_name, result, elapsed_ms)
        return result

    def _retry_delay(self, attempt: int) -> float:
        base = self._config.retry_delay_seconds
        if self._config.retry_policy is RetryPolicy.EXPONENTIAL:
            return base * (2 ** (attempt - 1))
        return base

    # ── Route resolution ────────────────────────────────────────────

    def _resolve_agent(self, task: TaskMessage) -> str:
        """Determine which agent should handle *task*."""
        # 1. Exact task-type match
        task_type = task.payload.get("task_type")
        if task_type and task_type in self._routes:
            for entry in self._routes[task_type]:
                if self._manager.is_running(entry.agent_name):
                    return entry.agent_name

        # 2. Capability match
        capability = task.payload.get("capability")
        if capability and capability in self._capability_index:
            for name in self._capability_index[capability]:
                if self._manager.is_running(name):
                    return name

        # 3. Fallback to agent_id on the task itself
        if self._manager.is_running(task.agent_id):
            return task.agent_id

        raise RuntimeError(
            f"No running agent found for task {task.task_id!r} "
            f"(task_type={task_type!r}, capability={capability!r}, "
            f"agent_id={task.agent_id!r})"
        )

    # ── Result helpers ──────────────────────────────────────────────

    @staticmethod
    def _failure_result(
        task: TaskMessage, agent_name: str, error: str | None
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_id=agent_name,
            success=False,
            error=error or "Unknown error",
        )

    @staticmethod
    def _timeout_result(task: TaskMessage, agent_name: str) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_id=agent_name,
            success=False,
            error="Timed out waiting for upstream dependencies",
        )

    @staticmethod
    def _exception_to_result(
        exc: BaseException, task: TaskMessage
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            success=False,
            error=str(exc),
        )

    # ── Event bus integration ───────────────────────────────────────

    async def _emit_dispatch_event(
        self,
        task: TaskMessage,
        agent_name: str,
        result: TaskResult,
        elapsed_ms: float,
    ) -> None:
        """Publish a dispatch event to the event bus (best-effort)."""
        payload: dict[str, Any] = {
            "event": "task.dispatched",
            "task_id": task.task_id,
            "agent": agent_name,
            "success": result.success,
            "elapsed_ms": round(elapsed_ms, 2),
            "ts": time.time(),
        }
        try:
            await self._context.event_bus.publish(
                "memfun.orchestrator",
                json.dumps(payload).encode(),
                key=task.task_id,
            )
        except Exception:
            logger.debug(
                "Failed to publish dispatch event for task %s",
                task.task_id,
                exc_info=True,
            )
