from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
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
    """Internal bookkeeping for a single running agent instance."""

    instance: BaseAgent
    name: str
    instance_id: str
    started_at: float = field(default_factory=time.time)
    missed_heartbeats: int = 0
    restart_count: int = 0
    running: bool = True


def _new_instance_id() -> str:
    """Mint a fresh, collision-resistant instance id (12 hex chars)."""
    return secrets.token_hex(6)


class AgentManager:
    """Manages the full lifecycle of a pool of agent instances.

    Multiple instances of the same agent type may run concurrently; each
    receives a distinct ``instance_id``. Lookups by instance id resolve
    directly; lookups by name return the **first** running instance for
    that name (backward-compat).

    Responsibilities:
    * Start / stop / restart individual instances or the entire pool.
    * Emit lifecycle events (``agent.started``, ``agent.stopped``,
      ``agent.failed``) to the event bus so other components can react.
    * Run a periodic health-check loop that automatically restarts
      unhealthy instances according to the configured
      :class:`RestartPolicy`.
    * Provide discovery helpers (list running instances, get by name or
      instance id).
    """

    def __init__(
        self,
        context: RuntimeContext,
        config: AgentManagerConfig | None = None,
    ) -> None:
        self._context = context
        self._config = config or AgentManagerConfig()
        # Primary table: instance_id -> _ManagedAgent
        self._agents: dict[str, _ManagedAgent] = {}
        # Secondary index: name -> ordered list of instance_ids
        self._instance_by_name: dict[str, list[str]] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    # ── Public API ──────────────────────────────────────────────────

    async def start_agent(self, name: str) -> str:
        """Instantiate and start a new instance of an agent by name.

        Returns the freshly minted ``instance_id``. Multiple calls with
        the same *name* yield distinct instances that coexist.

        Raises ``KeyError`` if *name* is not in the global agent registry.
        """
        registry = get_agent_registry()
        cls = registry.get(name)
        if cls is None:
            raise KeyError(
                f"Agent {name!r} not found in the registry. "
                f"Available: {', '.join(registry) or '(none)'}"
            )

        instance_id = _new_instance_id()
        instance = cls(self._context)
        await instance.on_start()

        self._agents[instance_id] = _ManagedAgent(
            instance=instance, name=name, instance_id=instance_id
        )
        self._instance_by_name.setdefault(name, []).append(instance_id)

        await self._emit_event("agent.started", name, instance_id=instance_id)
        logger.info("Agent %s started (instance %s)", name, instance_id)
        return instance_id

    async def start_pool(self, name: str, count: int) -> list[str]:
        """Start *count* instances of agent *name* in O(N).

        Returns the list of fresh instance_ids in start order. Useful for
        spinning up a worker pool for fan-out workloads where multiple
        sub-tasks of the same agent type must run truly in parallel.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        ids: list[str] = []
        for _ in range(count):
            ids.append(await self.start_agent(name))
        return ids

    async def stop_agent(self, key: str) -> None:
        """Gracefully stop running agent(s) addressed by *key*.

        *key* may be either an ``instance_id`` (stops that one instance)
        or an agent ``name`` (stops **every** running instance of that
        name).

        Raises ``KeyError`` if neither resolves to a managed agent.
        """
        instance_ids = self._resolve_to_instance_ids(key)
        if not instance_ids:
            raise KeyError(
                f"Agent {key!r} is not managed. "
                f"Running: {', '.join(self._agents) or '(none)'}"
            )
        for instance_id in instance_ids:
            await self._stop_instance(instance_id)

    async def restart_agent(self, key: str) -> str | None:
        """Stop then start an agent.

        If *key* is an ``instance_id``, restarts that single instance.
        If *key* is a name, restarts **every** running instance of that
        name (returns the new instance_id of the *first* one started, or
        ``None`` if none were running before).
        """
        instance_ids = self._resolve_to_instance_ids(key)
        if not instance_ids:
            # Nothing running — start a fresh single instance
            return await self.start_agent(key)

        # Capture names before we stop (instance entries are deleted on stop)
        names = [self._agents[iid].name for iid in instance_ids]
        for iid in instance_ids:
            await self._stop_instance(iid)

        first_new: str | None = None
        for n in names:
            new_id = await self.start_agent(n)
            if first_new is None:
                first_new = new_id
        return first_new

    async def start_all(self, names: list[str] | None = None) -> list[str]:
        """Start one instance per registered agent (or the named subset).

        Returns the list of fresh instance_ids in start order.
        """
        targets = names or list(get_agent_registry())
        ids: list[str] = []
        for name in targets:
            ids.append(await self.start_agent(name))
        return ids

    async def stop_all(self) -> None:
        """Stop every managed instance and cancel the health monitor."""
        self._stopped.set()
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        for instance_id in list(self._agents):
            try:
                await self._stop_instance(instance_id)
            except Exception:
                logger.exception(
                    "Error during stop_all for instance %s", instance_id
                )

    # ── Discovery ───────────────────────────────────────────────────

    def list_running(self) -> list[str]:
        """Return the names (with duplicates) of all running instances.

        Names of agents with multiple running instances appear once per
        instance, in start order.
        """
        return [m.name for m in self._agents.values() if m.running]

    def list_running_instances(self) -> list[str]:
        """Return the instance_ids of all currently running instances."""
        return [iid for iid, m in self._agents.items() if m.running]

    def get_agent(self, key: str) -> BaseAgent:
        """Return the :class:`BaseAgent` instance for *key*.

        *key* may be either an ``instance_id`` (direct lookup) or a
        ``name`` (returns the *first* running instance of that name —
        backward-compat for legacy callers that addressed agents by
        name).

        Raises ``KeyError`` if no running instance matches.
        """
        managed = self._resolve_managed(key)
        if managed is None:
            raise KeyError(
                f"Agent {key!r} is not managed. "
                f"Running: {', '.join(self._agents) or '(none)'}"
            )
        return managed.instance

    def get_agent_by_instance(self, instance_id: str) -> BaseAgent:
        """Return the :class:`BaseAgent` for an explicit ``instance_id``.

        Unlike :meth:`get_agent`, this does **not** fall back to name
        resolution — useful when callers must address one specific
        instance and treating the id as a name would mask a bug.

        Raises ``KeyError`` if the id is unknown.
        """
        managed = self._agents.get(instance_id)
        if managed is None:
            raise KeyError(
                f"Instance {instance_id!r} is not managed. "
                f"Running: {', '.join(self._agents) or '(none)'}"
            )
        return managed.instance

    def is_running(self, key: str) -> bool:
        """Whether *key* (instance_id or name) resolves to a running agent."""
        # Direct instance-id hit
        managed = self._agents.get(key)
        if managed is not None:
            return managed.running
        # Name fallback — running iff at least one instance is alive
        for iid in self._instance_by_name.get(key, []):
            entry = self._agents.get(iid)
            if entry is not None and entry.running:
                return True
        return False

    def instance_ids_for(self, name: str) -> list[str]:
        """Return all running instance_ids for an agent name (in start order)."""
        return [
            iid
            for iid in self._instance_by_name.get(name, [])
            if iid in self._agents and self._agents[iid].running
        ]

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
        """Periodically heartbeat and check instance health."""
        health = self._context.health
        interval = self._config.heartbeat_interval_seconds

        while not self._stopped.is_set():
            for instance_id in list(self._agents):
                managed = self._agents.get(instance_id)
                if managed is None or not managed.running:
                    continue

                # Health subsystem is keyed by a stable per-instance id
                # so multiple instances of the same name don't collide.
                heartbeat_key = f"{managed.name}:{instance_id}"
                try:
                    await health.heartbeat(heartbeat_key, {"ts": time.time()})
                    status = await health.check(heartbeat_key)
                except Exception:
                    logger.exception(
                        "Health check error for instance %s", instance_id
                    )
                    status = None

                if status is not None and status.state.value == "healthy":
                    managed.missed_heartbeats = 0
                else:
                    managed.missed_heartbeats += 1
                    logger.warning(
                        "Instance %s (%s) missed heartbeat (%d/%d)",
                        instance_id,
                        managed.name,
                        managed.missed_heartbeats,
                        self._config.unhealthy_threshold,
                    )

                if (
                    managed.missed_heartbeats
                    >= self._config.unhealthy_threshold
                ):
                    await self._handle_unhealthy(instance_id, managed)

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=interval
                )

    async def _handle_unhealthy(
        self, instance_id: str, managed: _ManagedAgent
    ) -> None:
        """React to an unhealthy instance according to the restart policy."""
        policy = self._config.restart_policy
        logger.error(
            "Instance %s (%s) is unhealthy (policy=%s)",
            instance_id,
            managed.name,
            policy.value,
        )

        await self._emit_event(
            "agent.failed",
            managed.name,
            instance_id=instance_id,
            error="unhealthy",
        )

        if policy is RestartPolicy.NEVER:
            managed.running = False
            return

        if managed.restart_count >= self._config.max_restart_attempts:
            logger.error(
                "Instance %s (%s) exceeded max restart attempts (%d), giving up",
                instance_id,
                managed.name,
                self._config.max_restart_attempts,
            )
            managed.running = False
            return

        backoff = self._config.restart_backoff_seconds * (
            2 ** managed.restart_count
        )
        logger.info(
            "Restarting instance %s (%s) (attempt %d, backoff %.1fs)",
            instance_id,
            managed.name,
            managed.restart_count + 1,
            backoff,
        )
        await asyncio.sleep(backoff)

        try:
            old_count = managed.restart_count
            name = managed.name
            await self._stop_instance(instance_id)
            new_id = await self.start_agent(name)
            new_managed = self._agents.get(new_id)
            if new_managed is not None:
                new_managed.restart_count = old_count + 1
        except Exception:
            logger.exception(
                "Failed to restart instance %s (%s)", instance_id, managed.name
            )
            managed.running = False

    # ── Internals ───────────────────────────────────────────────────

    def _resolve_managed(self, key: str) -> _ManagedAgent | None:
        """Resolve *key* (instance_id or name) to a single managed entry.

        Direct instance-id lookup wins. On a name match, returns the
        first **running** instance for that name. Returns ``None`` if no
        running entry matches.
        """
        managed = self._agents.get(key)
        if managed is not None:
            return managed
        for iid in self._instance_by_name.get(key, []):
            entry = self._agents.get(iid)
            if entry is not None and entry.running:
                return entry
        return None

    def _resolve_to_instance_ids(self, key: str) -> list[str]:
        """Resolve *key* to one or more instance_ids.

        - A direct id hit returns ``[key]``.
        - A name match returns every recorded instance for that name.
        - Returns ``[]`` if neither resolves.
        """
        if key in self._agents:
            return [key]
        ids = self._instance_by_name.get(key, [])
        # Filter to ids that still have a managed entry
        return [iid for iid in ids if iid in self._agents]

    async def _stop_instance(self, instance_id: str) -> None:
        """Stop a single instance by its ``instance_id``."""
        managed = self._agents.get(instance_id)
        if managed is None:
            raise KeyError(
                f"Instance {instance_id!r} is not managed. "
                f"Running: {', '.join(self._agents) or '(none)'}"
            )
        managed.running = False

        try:
            await asyncio.wait_for(
                managed.instance.on_stop(),
                timeout=self._config.shutdown_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Instance %s (%s) did not stop within %ss, forcing",
                instance_id,
                managed.name,
                self._config.shutdown_timeout_seconds,
            )
        except Exception:
            logger.exception(
                "Error stopping instance %s (%s)",
                instance_id,
                managed.name,
            )

        # Remove from primary table and the name index
        del self._agents[instance_id]
        ids = self._instance_by_name.get(managed.name)
        if ids is not None:
            with contextlib.suppress(ValueError):
                ids.remove(instance_id)
            if not ids:
                self._instance_by_name.pop(managed.name, None)

        await self._emit_event(
            "agent.stopped", managed.name, instance_id=instance_id
        )
        logger.info(
            "Agent %s stopped (instance %s)", managed.name, instance_id
        )

    async def _emit_event(
        self,
        event_type: str,
        agent_name: str,
        *,
        instance_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Publish a lifecycle event to the event bus (best-effort)."""
        payload: dict[str, Any] = {
            "event": event_type,
            "agent": agent_name,
            "ts": time.time(),
        }
        if instance_id is not None:
            payload["instance_id"] = instance_id
        if error is not None:
            payload["error"] = error

        try:
            await self._context.event_bus.publish(
                "memfun.lifecycle",
                json.dumps(payload).encode(),
                key=instance_id or agent_name,
            )
        except Exception:
            logger.debug(
                "Failed to publish lifecycle event %s for %s",
                event_type,
                agent_name,
                exc_info=True,
            )
