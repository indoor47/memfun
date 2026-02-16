"""Distributed agent worker: picks up tasks from Redis and executes them.

Each worker process connects to Redis, subscribes to the task stream
as a competing consumer (consumer group), runs the specialist agent's
``handle()`` method, and pushes the result back to a per-task reply stream.

Usage::

    # Start a coder-agent worker
    python -m memfun_runtime.worker --agent coder-agent --redis-url redis://localhost:6379

    # Start multiple workers (each gets distinct tasks via consumer group)
    python -m memfun_runtime.worker --agent coder-agent --redis-url redis://localhost:6379 &
    python -m memfun_runtime.worker --agent test-agent  --redis-url redis://localhost:6379 &
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import time
import uuid

from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage, TaskResult

from memfun_runtime.distributed import (
    EVENT_TOPIC,
    TASK_TOPIC,
    DistributedEvent,
    _result_topic,
    bytes_to_task,
    result_to_bytes,
)

logger = get_logger("worker")


class AgentWorker:
    """A single-agent worker that processes tasks from a Redis stream.

    Multiple workers can run in parallel for the same agent type —
    Redis consumer groups ensure each task is delivered to exactly one
    worker.
    """

    def __init__(
        self,
        agent_name: str,
        redis_url: str = "redis://localhost:6379",
        worker_id: str | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.redis_url = redis_url
        self.worker_id = worker_id or f"{agent_name}-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._agent = None
        self._bus = None
        self._context = None
        self._tasks_processed = 0
        self._tasks_succeeded = 0

    async def start(self) -> None:
        """Connect to Redis, start the agent, and begin processing tasks."""
        from memfun_core.config import MemfunConfig

        from memfun_runtime.builder import RuntimeBuilder
        from memfun_runtime.lifecycle import AgentManager

        logger.info(
            "[%s] Starting worker (agent=%s, redis=%s)",
            self.worker_id, self.agent_name, self.redis_url,
        )

        # Build runtime with Redis backend
        config = MemfunConfig.load()
        config = config.model_copy(update={
            "backend": config.backend.model_copy(update={
                "tier": "redis",
                "redis_url": self.redis_url,
            }),
        }) if hasattr(config, 'model_copy') else config

        # Override backend config for Redis
        os.environ["MEMFUN_REDIS_URL"] = self.redis_url

        self._context = await RuntimeBuilder(config).build()
        self._bus = self._context.event_bus

        # Start the specialist agent
        manager = AgentManager(self._context)

        # Import specialists to register them
        try:
            import memfun_agent.specialists  # noqa: F401
        except ImportError:
            logger.warning("memfun_agent.specialists not importable")

        await manager.start_agent(self.agent_name)
        self._agent = manager.get_agent(self.agent_name)

        # Announce worker online
        await self._emit(DistributedEvent(
            event_type="worker.online",
            agent_name=self.agent_name,
            worker_id=self.worker_id,
        ))

        self._running = True
        logger.info("[%s] Worker ready, listening for tasks...", self.worker_id)

        # Process loop
        try:
            await self._process_loop()
        finally:
            await self._emit(DistributedEvent(
                event_type="worker.offline",
                agent_name=self.agent_name,
                worker_id=self.worker_id,
            ))
            await manager.stop_all()

    async def _process_loop(self) -> None:
        """Main loop: subscribe to task stream and handle each task."""
        group = f"{self.agent_name}-workers"

        async for msg in self._bus.subscribe(TASK_TOPIC, group=group):
            if not self._running:
                break

            try:
                task = bytes_to_task(msg.payload)
            except Exception as exc:
                logger.warning("[%s] Bad task message: %s", self.worker_id, exc)
                continue

            # Only handle tasks targeted at our agent type
            if task.agent_id != self.agent_name:
                continue

            await self._handle_task(task)

    async def _handle_task(self, task: TaskMessage) -> None:
        """Execute a single task and publish the result."""
        logger.info(
            "[%s] Picked up task %s",
            self.worker_id, task.task_id,
        )

        await self._emit(DistributedEvent(
            event_type="task.picked_up",
            task_id=task.task_id,
            agent_name=self.agent_name,
            worker_id=self.worker_id,
        ))

        start = time.monotonic()
        try:
            result = await self._agent.handle(task)
        except Exception as exc:
            logger.error(
                "[%s] Task %s failed: %s",
                self.worker_id, task.task_id, exc,
            )
            result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_name,
                success=False,
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )

        elapsed_ms = (time.monotonic() - start) * 1000
        self._tasks_processed += 1
        if result.success:
            self._tasks_succeeded += 1

        # Publish result to the reply topic
        reply_topic = _result_topic(task.task_id)
        await self._bus.publish(reply_topic, result_to_bytes(result))

        logger.info(
            "[%s] Task %s %s in %.1fs (%d/%d total)",
            self.worker_id,
            task.task_id,
            "succeeded" if result.success else "failed",
            elapsed_ms / 1000,
            self._tasks_succeeded,
            self._tasks_processed,
        )

    async def _emit(self, event: DistributedEvent) -> None:
        """Publish event to the dashboard stream."""
        if self._bus:
            with contextlib.suppress(Exception):
                await self._bus.publish(EVENT_TOPIC, event.to_bytes())

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False


async def run_worker(agent_name: str, redis_url: str) -> None:
    """Entry point for running a single worker."""
    worker = AgentWorker(agent_name, redis_url)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)

    await worker.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Memfun distributed agent worker")
    parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Agent name to run (e.g. coder-agent, test-agent)",
    )
    parser.add_argument(
        "--redis-url", "-r",
        default="redis://localhost:6379",
        help="Redis connection URL",
    )
    args = parser.parse_args()
    asyncio.run(run_worker(args.agent, args.redis_url))


if __name__ == "__main__":
    main()
