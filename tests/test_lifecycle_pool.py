"""Per-instance pool semantics for AgentManager (issue #12).

Validates that a single agent *type* can have many concurrent Python
instances, each addressable by a distinct ``instance_id``, while
backward-compatible name-based lookup still picks the first running
instance.
"""
from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from memfun_core.config import BackendConfig, MemfunConfig
from memfun_core.types import TaskMessage, TaskResult
from memfun_runtime.agent import BaseAgent, agent
from memfun_runtime.builder import RuntimeBuilder
from memfun_runtime.lifecycle import AgentManager

# ── Test agent definitions ──────────────────────────────────────────────


@agent(name="pool-counter", version="1.0", capabilities=["counting"])
class _CounterAgent(BaseAgent):
    """Per-instance counter — proves each instance is its own object."""

    def __init__(self, context: Any) -> None:  # type: ignore[override]
        super().__init__(context)
        self.count = 0

    async def handle(self, task: TaskMessage) -> TaskResult:  # pragma: no cover
        self.count += 1
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            result={"count": self.count, "obj_id": id(self)},
        )


@agent(name="pool-other", version="1.0")
class _OtherAgent(BaseAgent):
    async def handle(self, task: TaskMessage) -> TaskResult:  # pragma: no cover
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
        )


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def manager() -> AgentManager:
    """Fresh AgentManager backed by an in-process runtime."""
    config = MemfunConfig(backend=BackendConfig(tier="memory"))
    ctx = await RuntimeBuilder(config).build()
    return AgentManager(ctx)


# ── Tests ───────────────────────────────────────────────────────────────


class TestPerInstancePool:
    async def test_start_agent_returns_distinct_ids(
        self, manager: AgentManager
    ) -> None:
        """Five start_agent calls -> five distinct instance ids."""
        ids = [await manager.start_agent("pool-counter") for _ in range(5)]
        assert len(set(ids)) == 5
        assert all(len(i) == 12 for i in ids)  # secrets.token_hex(6)

        # All five coexist as separate Python objects
        instances = [manager.get_agent_by_instance(i) for i in ids]
        assert len({id(inst) for inst in instances}) == 5

        await manager.stop_all()

    async def test_get_agent_by_name_returns_first_instance(
        self, manager: AgentManager
    ) -> None:
        """Backward compat: get_agent('name') returns the first running instance."""
        first_id = await manager.start_agent("pool-counter")
        second_id = await manager.start_agent("pool-counter")
        assert first_id != second_id

        first = manager.get_agent_by_instance(first_id)
        second = manager.get_agent_by_instance(second_id)
        by_name = manager.get_agent("pool-counter")

        assert by_name is first
        assert by_name is not second

        await manager.stop_all()

    async def test_get_agent_by_instance_id(
        self, manager: AgentManager
    ) -> None:
        """get_agent(instance_id) hits the exact instance directly."""
        ids = [await manager.start_agent("pool-counter") for _ in range(3)]

        for iid in ids:
            via_get = manager.get_agent(iid)
            via_explicit = manager.get_agent_by_instance(iid)
            assert via_get is via_explicit

        # All three are independent Python objects
        objs = [manager.get_agent_by_instance(i) for i in ids]
        assert len({id(o) for o in objs}) == 3

        await manager.stop_all()

    async def test_start_pool_returns_n_distinct_ids(
        self, manager: AgentManager
    ) -> None:
        """start_pool(name, 5) -> five fresh instances of the same name."""
        ids = await manager.start_pool("pool-counter", 5)
        assert len(ids) == 5
        assert len(set(ids)) == 5

        # All running and tied to the right name
        for iid in ids:
            assert manager.is_running(iid)

        running_for_name = manager.instance_ids_for("pool-counter")
        assert set(running_for_name) == set(ids)

        # And they are five truly-separate Python objects
        objs = [manager.get_agent_by_instance(i) for i in ids]
        assert len({id(o) for o in objs}) == 5

        await manager.stop_all()

    async def test_start_pool_rejects_non_positive(
        self, manager: AgentManager
    ) -> None:
        with pytest.raises(ValueError, match="count must be positive"):
            await manager.start_pool("pool-counter", 0)
        with pytest.raises(ValueError, match="count must be positive"):
            await manager.start_pool("pool-counter", -1)

    async def test_stop_agent_by_name_stops_all_instances(
        self, manager: AgentManager
    ) -> None:
        """stop_agent('name') tears down every running instance of that name."""
        ids = await manager.start_pool("pool-counter", 4)
        # Spin up an unrelated agent that should NOT be touched
        other_id = await manager.start_agent("pool-other")

        await manager.stop_agent("pool-counter")

        for iid in ids:
            assert not manager.is_running(iid)
        assert manager.instance_ids_for("pool-counter") == []
        assert manager.is_running(other_id)
        assert manager.is_running("pool-other")

        await manager.stop_all()

    async def test_stop_agent_by_instance_id_only_stops_one(
        self, manager: AgentManager
    ) -> None:
        """stop_agent(instance_id) leaves siblings of the same name alive."""
        ids = await manager.start_pool("pool-counter", 3)
        await manager.stop_agent(ids[1])

        assert manager.is_running(ids[0])
        assert not manager.is_running(ids[1])
        assert manager.is_running(ids[2])

        # Name still resolves to the surviving first instance
        survivors = manager.instance_ids_for("pool-counter")
        assert survivors == [ids[0], ids[2]]
        assert manager.get_agent("pool-counter") is manager.get_agent_by_instance(ids[0])

        await manager.stop_all()

    async def test_legacy_start_then_get_by_name_still_works(
        self, manager: AgentManager
    ) -> None:
        """The pre-#12 pattern: start once, look up by name, must keep working."""
        instance_id = await manager.start_agent("pool-counter")
        assert isinstance(instance_id, str) and len(instance_id) > 0

        agent_obj = manager.get_agent("pool-counter")
        assert isinstance(agent_obj, _CounterAgent)
        assert manager.is_running("pool-counter")
        assert manager.is_running(instance_id)

        # Stop by the legacy name path
        await manager.stop_agent("pool-counter")
        assert not manager.is_running("pool-counter")
        assert not manager.is_running(instance_id)

    async def test_unknown_key_raises(self, manager: AgentManager) -> None:
        with pytest.raises(KeyError):
            manager.get_agent("never-started")
        with pytest.raises(KeyError):
            manager.get_agent_by_instance("deadbeefdead")
        with pytest.raises(KeyError):
            await manager.stop_agent("never-started")

    async def test_unregistered_agent_raises_keyerror(
        self, manager: AgentManager
    ) -> None:
        with pytest.raises(KeyError):
            await manager.start_agent("not-in-registry")

    async def test_list_running_reports_per_instance(
        self, manager: AgentManager
    ) -> None:
        """list_running returns the *name* of every instance (with duplicates)."""
        await manager.start_pool("pool-counter", 3)
        await manager.start_agent("pool-other")

        running = manager.list_running()
        assert running.count("pool-counter") == 3
        assert running.count("pool-other") == 1

        instances = manager.list_running_instances()
        assert len(instances) == 4
        assert len(set(instances)) == 4

        await manager.stop_all()

    async def test_restart_by_instance_id(
        self, manager: AgentManager
    ) -> None:
        """Restarting by id stops that one and starts a fresh one."""
        ids = await manager.start_pool("pool-counter", 2)
        new_id = await manager.restart_agent(ids[0])

        # Original id is gone, sibling untouched, a fresh id replaces it
        assert not manager.is_running(ids[0])
        assert manager.is_running(ids[1])
        assert new_id is not None
        assert new_id != ids[0]
        assert manager.is_running(new_id)

        await manager.stop_all()

    async def test_restart_by_name_replaces_all_instances(
        self, manager: AgentManager
    ) -> None:
        """Restart by name stops every running instance and starts the same count back."""
        ids = await manager.start_pool("pool-counter", 3)
        await manager.restart_agent("pool-counter")

        # Old ids are gone, three fresh ones are running
        for iid in ids:
            assert not manager.is_running(iid)
        new_ids = manager.instance_ids_for("pool-counter")
        assert len(new_ids) == 3
        assert set(new_ids).isdisjoint(ids)

        await manager.stop_all()


class TestOrchestratorFanOutToPool:
    """Verify the orchestrator can fan-out same-type tasks to distinct instances."""

    async def test_fan_out_to_pool_runs_each_task_on_own_instance(
        self, manager: AgentManager
    ) -> None:
        """Five same-type tasks -> five distinct Python objects (verifiable via id())."""
        from memfun_runtime.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(manager._context, manager)
        tasks = [
            TaskMessage(
                task_id=f"t-{i}",
                agent_id="pool-counter",
                payload={"i": i},
            )
            for i in range(5)
        ]

        results = await orchestrator.fan_out_to_pool(tasks, "pool-counter")

        # Every task succeeded
        assert len(results) == 5
        assert all(r.success for r in results)

        # The handlers reported each Python obj_id — each must be unique
        # (proving they ran on five truly-isolated BaseAgent instances).
        obj_ids = {r.result["obj_id"] for r in results if r.result is not None}
        assert len(obj_ids) == 5

        # Each instance saw exactly one call (per-instance counter == 1)
        counts = [r.result["count"] for r in results if r.result is not None]
        assert sorted(counts) == [1, 1, 1, 1, 1]

        # And the manager has five running instances of that name
        assert len(manager.instance_ids_for("pool-counter")) == 5

        await manager.stop_all()

    async def test_fan_out_to_pool_reuses_existing_instances(
        self, manager: AgentManager
    ) -> None:
        """If 3 instances already exist and 5 tasks come in, only 2 new ones are spawned."""
        from memfun_runtime.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(manager._context, manager)
        existing = await manager.start_pool("pool-counter", 3)
        tasks = [
            TaskMessage(task_id=f"t-{i}", agent_id="pool-counter", payload={})
            for i in range(5)
        ]

        await orchestrator.fan_out_to_pool(tasks, "pool-counter")

        all_ids = manager.instance_ids_for("pool-counter")
        assert len(all_ids) == 5
        # The original three are still in the pool
        for iid in existing:
            assert iid in all_ids

        await manager.stop_all()

    async def test_fan_out_to_pool_empty_tasks_is_noop(
        self, manager: AgentManager
    ) -> None:
        from memfun_runtime.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(manager._context, manager)
        results = await orchestrator.fan_out_to_pool([], "pool-counter")
        assert results == []
        # No instances spun up
        assert manager.instance_ids_for("pool-counter") == []
