"""Validation tests for the parallel-agent epic — Experiment A re-run.

Background
----------
Experiment A (May 2026, see ``docs/LEARNINGS.md``) showed that with 5 LLM
agents independently rewriting the same shared file, vanilla ``git merge``
produced **6/10 pairwise conflicts** at both Qwen 2.5 Coder 3B and 7B.  A
stronger model did not reduce the conflict rate.

The architectural decision was to **prevent overlapping output assignments
at decomposition time**, rather than try to merge afterwards.  PR #9 added
:func:`memfun_agent.decomposer.detect_output_overlap` and
:func:`memfun_agent.decomposer.resolve_output_overlap` and wired them into
:meth:`TaskDecomposer.adecompose`'s post-process.  PR #11 added hierarchical
decomposition.  PR #8 added :class:`WorktreeManager` for filesystem
isolation.

This module closes the validation loop: it asserts that, when the
decomposer emits 5 sub-tasks all declaring the same output file (the
S2 scenario from Experiment A), the system actually serialises them
through :class:`WorkflowEngine` rather than fanning them out in parallel.

Three test layers
-----------------
* **Test A** — decomposition-only: covered already by
  ``tests/test_decomposer_overlap.py::test_e2e_experiment_a_s2_five_writers_to_middleware``.
  We re-import the helpers here only as a sanity check (see
  :func:`test_decomposer_alone_sequences_five_writers`).
* **Test B** — through :class:`WorkflowEngine`: stub the orchestrator so
  each fan-out call records ``(scheduled, completed)`` timestamps.  We
  then assert (1) the engine schedules five separate fan-out groups (one
  per task), (2) the per-task scheduling intervals don't overlap, and
  (3) the wall time is approximately ``5 x per_task_duration``.  This is
  the load-bearing integration property for the parallel-agent epic.
* **Test C** — with :class:`WorktreeManager`: SKIPPED.  See the docstring
  on :func:`test_e2e_with_worktree_manager_skipped` for the rationale —
  :class:`WorkflowEngine` does not currently merge worktrees back into
  the project root, so a "5-worktrees merge cleanly" assertion would
  test code that does not exist.  The architectural fix being validated
  here is decomposition-time serialisation; merge-back is a separate
  follow-on (issue #18 family).
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import pytest
from memfun_agent.decomposer import (
    DecompositionResult,
    SubTask,
    TaskDecomposer,
    detect_output_overlap,
    resolve_output_overlap,
)
from memfun_agent.workflow import WorkflowEngine
from memfun_core.types import TaskMessage, TaskResult

# ── Fixtures / helpers ────────────────────────────────────────


@dataclass(slots=True)
class ScheduleRecord:
    """Captures when a stub specialist started and finished a task."""

    task_id: str
    sub_task_id: str
    scheduled_at: float
    completed_at: float

    @property
    def duration(self) -> float:
        return self.completed_at - self.scheduled_at


@dataclass(slots=True)
class FanOutCall:
    """Records what was passed to one ``orchestrator.fan_out`` invocation."""

    sub_task_ids: list[str] = field(default_factory=lambda: [])
    scheduled_at: float = 0.0
    completed_at: float = 0.0


def _build_workflow_engine(
    *,
    per_task_duration: float = 0.05,
) -> tuple[WorkflowEngine, list[FanOutCall], list[ScheduleRecord]]:
    """Construct an engine wired to instrumented mock collaborators.

    Each stubbed specialist sleeps ``per_task_duration`` seconds, so two
    tasks running concurrently in the same fan-out call would overlap in
    wall time, while two tasks run in sequential fan-out calls would
    produce strictly non-overlapping intervals.
    """
    fan_out_calls: list[FanOutCall] = []
    schedules: list[ScheduleRecord] = []

    async def fake_fan_out(
        tasks: list[TaskMessage],
        agent_names: list[str],
        *,
        timeout: float = 600.0,
    ) -> list[TaskResult]:
        del agent_names, timeout  # unused
        call = FanOutCall(
            sub_task_ids=[t.payload.get("sub_task_id", "?") for t in tasks],
            scheduled_at=time.perf_counter(),
        )

        async def run_one(t: TaskMessage) -> TaskResult:
            start = time.perf_counter()
            await asyncio.sleep(per_task_duration)
            end = time.perf_counter()
            schedules.append(
                ScheduleRecord(
                    task_id=t.task_id,
                    sub_task_id=t.payload.get("sub_task_id", "?"),
                    scheduled_at=start,
                    completed_at=end,
                ),
            )
            return TaskResult(
                task_id=t.task_id,
                agent_id=t.agent_id,
                success=True,
                result={
                    "answer": f"done {t.payload.get('sub_task_id', '?')}",
                    "files_created": list(t.payload.get("outputs", [])),
                    "ops": [],
                },
                duration_ms=(end - start) * 1000.0,
            )

        # Specialists in the same group run concurrently — exactly what
        # the architectural fix is supposed to prevent for overlapping
        # outputs.  If the decomposer sequences them, ``tasks`` will
        # contain a single sub-task per call.
        results = await asyncio.gather(*(run_one(t) for t in tasks))
        call.completed_at = time.perf_counter()
        fan_out_calls.append(call)
        return list(results)

    review_result = TaskResult(
        task_id="review-x",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: true\n\nLooks fine"},
    )

    orchestrator = MagicMock()
    orchestrator.fan_out = AsyncMock(side_effect=fake_fan_out)
    orchestrator.dispatch = AsyncMock(return_value=review_result)

    manager = MagicMock()
    manager.is_running = MagicMock(return_value=True)
    manager.start_agent = AsyncMock()

    state_store = AsyncMock()
    state_store.get = AsyncMock(return_value=None)
    state_store.set = AsyncMock()

    context = MagicMock()
    context.state_store = state_store
    context.event_bus = MagicMock()
    context.event_bus.publish = AsyncMock()

    engine = WorkflowEngine(
        context=context,
        orchestrator=orchestrator,
        manager=manager,
    )
    return engine, fan_out_calls, schedules


def _five_writers_decomposition() -> DecompositionResult:
    """Mirrors Experiment A scenario S2 — 5 writers to middleware.py.

    Each task ALSO declares its own endpoints/* file (so the per-task
    intent is non-trivial, but the shared output remains the conflict
    site).  All 5 are emitted in one parallelism group.
    """
    sub_tasks = [
        SubTask(
            id=f"T{i}",
            description=f"Add feature {i} (touches src/middleware.py).",
            agent_type="coder",
            outputs=["src/middleware.py", f"src/endpoints/feature_{i}.py"],
        )
        for i in range(1, 6)
    ]
    return DecompositionResult(
        sub_tasks=sub_tasks,
        shared_spec="All 5 tasks add features behind a shared middleware.",
        parallelism_groups=[[st.id for st in sub_tasks]],
        is_single_task=False,
    )


# ── Test A — decomposition-only sanity check ─────────────────


def test_decomposer_alone_sequences_five_writers() -> None:
    """Sanity-check that the decomposer's overlap resolver still works.

    The full LLM-mocked end-to-end flavour lives in
    ``tests/test_decomposer_overlap.py``; this is just the static check
    that, given Experiment A's S2 input shape, the resolver produces
    five sequential singletons.  If this regresses, the rest of
    Experiment A's validation no longer holds, so the failure mode
    ought to surface in this file as well as the unit suite.
    """
    decomp = _five_writers_decomposition()
    overlaps = detect_output_overlap(decomp.sub_tasks, decomp.parallelism_groups)
    # All 5 tasks claim middleware.py — pre-fix this WAS the bug.
    assert overlaps == {"src/middleware.py": ["T1", "T2", "T3", "T4", "T5"]}

    new_groups, log = resolve_output_overlap(
        decomp.sub_tasks,
        decomp.parallelism_groups,
        strategy="sequence",
    )
    assert new_groups == [["T1"], ["T2"], ["T3"], ["T4"], ["T5"]]
    assert any("sequenced" in m for m in log)


# ── Test B — through WorkflowEngine ──────────────────────────


def _patch_decomposer_to_emit(
    engine: WorkflowEngine,
    raw_decomp: DecompositionResult,
) -> Callable[[str, str], Awaitable[DecompositionResult]]:
    """Patch helper that runs the resolver on the way out.

    The integration is the same as production: the decomposer's LLM
    pass is what produces the conflicting groups, and the
    ``adecompose`` post-process is what serialises them.  We mock the
    LLM with the conflicting plan and let the real resolver run.
    """
    real_decomposer = TaskDecomposer()
    mock_llm_result = MagicMock()
    mock_llm_result.sub_tasks = [
        {
            "id": st.id,
            "description": st.description,
            "agent_type": st.agent_type,
            "outputs": list(st.outputs),
            "inputs": list(st.inputs),
            "depends_on": list(st.depends_on),
        }
        for st in raw_decomp.sub_tasks
    ]
    mock_llm_result.shared_spec = raw_decomp.shared_spec
    mock_llm_result.parallelism_groups = [list(g) for g in raw_decomp.parallelism_groups]

    async def patched_adecompose(
        task: str,
        project_context: str,
        **kwargs: Any,
    ) -> DecompositionResult:
        del task, project_context, kwargs

        # Reuse the real decomposer's post-process by patching the
        # threaded LLM call inside it; this exercises the actual
        # production path through detect/resolve_output_overlap.
        async def mock_to_thread(func: Any, *args: Any, **kw: Any) -> Any:
            del func, args, kw
            return mock_llm_result

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            return await real_decomposer.adecompose("ignored", "ignored")

    engine._decomposer.adecompose = patched_adecompose  # type: ignore[method-assign]
    return patched_adecompose


@pytest.mark.asyncio
async def test_workflow_engine_serialises_five_writers_to_one_file() -> None:
    """End-to-end through WorkflowEngine: 5 middleware.py writers serialise.

    This is the load-bearing integration test.  The architectural fix
    promises that:
      (1) the decomposer's sequenced groups make it through to the
          execution loop unchanged;
      (2) the engine fans out one group at a time, so no two
          conflicting writes execute concurrently;
      (3) the resulting wall time is approx N x per-task latency, not 1 x.
    """
    per_task = 0.05
    engine, fan_out_calls, schedules = _build_workflow_engine(
        per_task_duration=per_task,
    )

    raw = _five_writers_decomposition()
    _patch_decomposer_to_emit(engine, raw)

    # Bypass the file-system context gather (it touches ``os.getcwd()``
    # which is the whole memfun repo and balloons test time).
    async def _stub_gather(
        project_context: str,
        decomposition: DecompositionResult,
    ) -> str:
        del decomposition
        return project_context

    with patch.object(
        engine,
        "_gather_shared_context",
        side_effect=_stub_gather,
    ):
        wall_start = time.perf_counter()
        result = await engine.execute_workflow(
            "Add 5 features behind a shared middleware",
            "ctx",
            None,
        )
        wall_elapsed = time.perf_counter() - wall_start

    # ── (0) Workflow succeeded ─────────────────────────────────
    assert result.success, f"workflow failed: {result.error}"
    assert result.decomposition is not None

    # ── (1) Decomposition sequenced the conflict away ─────────
    final_groups = result.decomposition.parallelism_groups
    # Order may differ, but every group is a singleton and all 5 tasks
    # are accounted for.
    assert len(final_groups) == 5, f"expected 5 sequential groups, got {final_groups}"
    flattened = [tid for grp in final_groups for tid in grp]
    assert sorted(flattened) == ["T1", "T2", "T3", "T4", "T5"]
    for grp in final_groups:
        assert len(grp) == 1, f"expected each group to be a singleton (sequenced), got {grp}"

    # No remaining overlap is the structural invariant.
    assert (
        detect_output_overlap(
            result.decomposition.sub_tasks,
            result.decomposition.parallelism_groups,
        )
        == {}
    )

    # ── (2) Engine fanned out 5 separate groups ──────────────
    # Each fan-out call carries exactly one sub_task — the proof that
    # tasks are NOT racing concurrently against middleware.py.
    assert len(fan_out_calls) == 5, (
        f"expected 5 sequential fan_out calls, got {len(fan_out_calls)}"
    )
    for call in fan_out_calls:
        assert len(call.sub_task_ids) == 1, (
            f"fan_out received {len(call.sub_task_ids)} tasks at once;"
            f" the architectural fix requires 1 (got {call.sub_task_ids})"
        )

    # Sanity: all 5 sub-tasks were run exactly once.
    assert sorted(s.sub_task_id for s in schedules) == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    ]

    # ── (3) Schedule intervals do not overlap ────────────────
    # If any two tasks ran concurrently, ``[start_i, end_i]`` and
    # ``[start_j, end_j]`` would intersect.  After the fix, the engine
    # awaits each group fully before launching the next.
    by_start = sorted(schedules, key=lambda s: s.scheduled_at)
    for prev, nxt in itertools.pairwise(by_start):
        assert nxt.scheduled_at >= prev.completed_at - 1e-3, (
            f"tasks {prev.sub_task_id} and {nxt.sub_task_id} ran concurrently: {prev} vs {nxt}"
        )

    # ── (4) Wall time looks sequential ───────────────────────
    # Allow generous slop for CI jitter, but reject "all parallel"
    # which would be approx per_task seconds.  The architectural fix
    # implies wall approx N x per_task; we require >= 0.6 x N x
    # per_task to stay robust under noisy schedulers without trusting
    # microsecond precision.
    expected_lower_bound = 0.6 * 5 * per_task
    assert wall_elapsed >= expected_lower_bound, (
        f"wall time {wall_elapsed:.3f}s suggests parallel execution;"
        f" expected ≥ {expected_lower_bound:.3f}s for sequenced tasks"
    )


# ── Test C — with WorktreeManager: SKIPPED, see docstring ────


@pytest.mark.skip(
    reason=(
        "WorkflowEngine does not currently merge per-task worktrees back"
        " into the project root, so a 'N worktrees merge cleanly' assertion"
        " would test code that does not exist.  The architectural property"
        " under validation here — decomposition-time serialisation —"
        " is fully covered by Test B.  Worktree merge-back is tracked"
        " separately as a follow-on to PR #8 (worktree provisioning) and"
        " issue #18 (cleanup-path scoping).  Re-enable this test once"
        " WorkflowEngine grows a merge step."
    ),
)
@pytest.mark.asyncio
async def test_e2e_with_worktree_manager_skipped() -> None:  # pragma: no cover
    """Placeholder for the future full-merge validation.

    The intended assertion: configure a :class:`WorktreeManager`, run
    five stub specialists that each commit a small edit in their own
    worktree, then verify ``git merge-tree`` between sequential pairs
    is conflict-free.  The serialisation guarantees pairwise mergeability
    because each worktree builds on the previous's HEAD.

    Skipped — see the ``@pytest.mark.skip`` reason above.
    """
    raise AssertionError("unreachable — test is skipped")
