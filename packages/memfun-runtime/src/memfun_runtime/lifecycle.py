from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

from memfun_runtime.agent import BaseAgent, get_agent_registry

if TYPE_CHECKING:
    from memfun_runtime.context import RuntimeContext

logger = get_logger("lifecycle")


class RestartPolicy(Enum):
    """Policy applied when an agent becomes unhealthy."""

    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class AgentManagerConfig:
    """Tuning knobs for the :class:`AgentManager`."""

    heartbeat_interval_seconds: float = 5.0
    unhealthy_threshold: int = 3
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    max_restart_attempts: int = 5
    restart_backoff_seconds: float = 1.0
    shutdown_timeout_seconds: float = 10.0


@dataclass(slots=True)
class _ManagedAgent:
    """Internal bookkeeping for a single running agent."""

    instance: BaseAgent
    started_at: float = field(default_factory=time.time)
    missed_heartbeats: int = 0
    restart_count: int = 0
    running: bool = True


class AgentManager:
    """Manages the full lifecycle of a pool of agent instances.

    Responsibilities:
    * Start / stop / restart individual agents or the entire pool.
    * Emit lifecycle events (``agent.started``, ``agent.stopped``,
      ``agent.failed``) to the event bus so other components can react.
    * Run a periodic health-check loop that automatically restarts
      unhealthy agents according to the configured
      :class:`RestartPolicy`.
    * Provide discovery helpers (list running agents, get by name).
    """

    def __init__(
        self,
        context: RuntimeContext,
        config: AgentManagerConfig | None = None,
    ) -> None:
        self._context = context
        self._config = config or AgentManagerConfig()
        self._agents: dict[str, _ManagedAgent] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    # ── Public API ──────────────────────────────────────────────────

    async def start_agent(self, name: str) -> None:
        """Instantiate and start an agent by its registered name.

        Raises ``KeyError`` if *name* is not in the global agent registry
        and ``RuntimeError`` if the agent is already running.
        """
        if name in self._agents and self._agents[name].running:
            raise RuntimeError(f"Agent {name!r} is already running")

        registry = get_agent_registry()
        cls = registry.get(name)
        if cls is None:
            raise KeyError(
                f"Agent {name!r} not found in the registry. "
                f"Available: {', '.join(registry) or '(none)'}"
            )

        instance = cls(self._context)
        await instance.on_start()

        self._agents[name] = _ManagedAgent(instance=instance)
        await self._emit_event("agent.started", name)
        logger.info("Agent %s started", name)

    async def stop_agent(self, name: str) -> None:
        """Gracefully stop a running agent.

        Raises ``KeyError`` if *name* is not currently managed.
        """
        managed = self._get_managed(name)
        managed.running = False

        try:
            await asyncio.wait_for(
                managed.instance.on_stop(),
                timeout=self._config.shutdown_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Agent %s did not stop within %ss, forcing",
                name,
                self._config.shutdown_timeout_seconds,
            )
        except Exception:
            logger.exception("Error stopping agent %s", name)

        del self._agents[name]
        await self._emit_event("agent.stopped", name)
        logger.info("Agent %s stopped", name)

    async def restart_agent(self, name: str) -> None:
        """Stop then start an agent by name."""
        if name in self._agents and self._agents[name].running:
            await self.stop_agent(name)
        await self.start_agent(name)

    async def start_all(self, names: list[str] | None = None) -> None:
        """Start agents by name, or all agents in the registry."""
        targets = names or list(get_agent_registry())
        for name in targets:
            await self.start_agent(name)

    async def stop_all(self) -> None:
        """Stop every managed agent and cancel the health monitor."""
        self._stopped.set()
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        for name in list(self._agents):
            try:
                await self.stop_agent(name)
            except Exception:
                logger.exception("Error during stop_all for %s", name)

    # ── Discovery ───────────────────────────────────────────────────

    def list_running(self) -> list[str]:
        """Return the names of all currently running agents."""
        return [n for n, m in self._agents.items() if m.running]

    def get_agent(self, name: str) -> BaseAgent:
        """Return the :class:`BaseAgent` instance for *name*.

        Raises ``KeyError`` if the agent is not running.
        """
        return self._get_managed(name).instance

    def is_running(self, name: str) -> bool:
        """Check whether a named agent is currently running."""
        return name in self._agents and self._agents[name].running

    # ── Health Monitor Loop ─────────────────────────────────────────

    def start_monitor(self) -> None:
        """Spawn the background health-check coroutine."""
        if self._monitor_task is not None and not self._monitor_task.done():
            logger.warning("Health monitor already running")
            return
        self._stopped.clear()
        self._monitor_task = asyncio.get_event_loop().create_task(
            self._health_loop()
        )
        logger.info("Health monitor started")

    async def _health_loop(self) -> None:
        """Periodically heartbeat and check agent health."""
        health = self._context.health
        interval = self._config.heartbeat_interval_seconds

        while not self._stopped.is_set():
            for name in list(self._agents):
                managed = self._agents.get(name)
                if managed is None or not managed.running:
                    continue

                try:
                    await health.heartbeat(name, {"ts": time.time()})
                    status = await health.check(name)
                except Exception:
                    logger.exception(
                        "Health check error for agent %s", name
                    )
                    status = None

                if status is not None and status.state.value == "healthy":
                    managed.missed_heartbeats = 0
                else:
                    managed.missed_heartbeats += 1
                    logger.warning(
                        "Agent %s missed heartbeat (%d/%d)",
                        name,
                        managed.missed_heartbeats,
                        self._config.unhealthy_threshold,
                    )

                if (
                    managed.missed_heartbeats
                    >= self._config.unhealthy_threshold
                ):
                    await self._handle_unhealthy(name, managed)

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=interval
                )

    async def _handle_unhealthy(
        self, name: str, managed: _ManagedAgent
    ) -> None:
        """React to an unhealthy agent according to the restart policy."""
        policy = self._config.restart_policy
        logger.error("Agent %s is unhealthy (policy=%s)", name, policy.value)

        await self._emit_event("agent.failed", name, error="unhealthy")

        if policy is RestartPolicy.NEVER:
            managed.running = False
            return

        if (
            managed.restart_count >= self._config.max_restart_attempts
        ):
            logger.error(
                "Agent %s exceeded max restart attempts (%d), giving up",
                name,
                self._config.max_restart_attempts,
            )
            managed.running = False
            return

        backoff = self._config.restart_backoff_seconds * (
            2 ** managed.restart_count
        )
        logger.info(
            "Restarting agent %s (attempt %d, backoff %.1fs)",
            name,
            managed.restart_count + 1,
            backoff,
        )
        await asyncio.sleep(backoff)

        try:
            await self.restart_agent(name)
            new_managed = self._agents.get(name)
            if new_managed is not None:
                new_managed.restart_count = managed.restart_count + 1
        except Exception:
            logger.exception("Failed to restart agent %s", name)
            managed.running = False

    # ── Internals ───────────────────────────────────────────────────

    def _get_managed(self, name: str) -> _ManagedAgent:
        managed = self._agents.get(name)
        if managed is None:
            raise KeyError(
                f"Agent {name!r} is not managed. "
                f"Running: {', '.join(self._agents) or '(none)'}"
            )
        return managed

    async def _emit_event(
        self,
        event_type: str,
        agent_name: str,
        *,
        error: str | None = None,
    ) -> None:
        """Publish a lifecycle event to the event bus (best-effort)."""
        payload: dict[str, Any] = {
            "event": event_type,
            "agent": agent_name,
            "ts": time.time(),
        }
        if error is not None:
            payload["error"] = error

        try:
            await self._context.event_bus.publish(
                "memfun.lifecycle",
                json.dumps(payload).encode(),
                key=agent_name,
            )
        except Exception:
            logger.debug(
                "Failed to publish lifecycle event %s for %s",
                event_type,
                agent_name,
                exc_info=True,
            )
