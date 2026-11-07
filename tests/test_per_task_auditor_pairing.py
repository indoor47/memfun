"""Unit tests for the per-task implementer/auditor pairing dyad.

These tests cover the new ``WorkflowEngine`` behaviour where each
sub-task that succeeds is paired with a synchronous ``review-agent``
dispatch inside :meth:`WorkflowEngine._execute_group`, and the coder
is re-dispatched (up to ``max_review_rounds_per_task`` rounds) when
the reviewer requests changes.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.workflow import (
    PerTaskReview,
    WorkflowEngine,
)
from memfun_core.types import TaskResult

# Canned reviewer answers embed the pinned nonce (see conftest).
pytestmark = pytest.mark.usefixtures("fixed_review_id")

# ── Helpers ──────────────────────────────────────────────────


def _coder_result(tid: str, success: bool = True, answer: str = "ok") -> TaskResult:
    return TaskResult(
        task_id=f"wf_{tid}",
        agent_id="coder-agent",
        success=success,
        result={
            "answer": answer,
            "files_created": [],
            "ops": [{"type": "edit", "target": "stub.py"}],
        },
    )


def _review_result(approved: bool, feedback: str = "") -> TaskResult:
    """Per-task review answer in the structured findings format.

    The verdict is computed in code from severities, so "approved"
    means no blocking findings and "rejected" means one MEDIUM finding.
    """
    findings = (
        []
        if approved
        else [
            {
                "severity": "MEDIUM",
                "file": "src/x.py",
                "line": 1,
                "issue": feedback or "fix it",
                "fix": "do it properly",
            }
        ]
    )
    block = json.dumps(
        {"review_id": "test-review-id", "findings": findings, "summary": feedback or "review"}
    )
    body = f"Reviewed.\n```json\n{block}\n```"
    return TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": body},
    )


def _phase3_review(approved: bool = True) -> TaskResult:
    """The Phase 3 cross-task review result (structured findings JSON)."""
    findings = (
        []
        if approved
        else [{"severity": "MEDIUM", "task_id": "T1", "file": "a.py", "issue": "still bad"}]
    )
    body = "All consistent.\n" + json.dumps(
        {"review_id": "test-review-id", "findings": findings, "summary": "cross-task"}
    )
    return TaskResult(
        task_id="wf_phase3_review",
        agent_id="review-agent",
        success=True,
        result={"answer": body},
    )


def _make_engine(
    *,
    pair: bool = True,
    max_rounds: int = 2,
) -> tuple[WorkflowEngine, AsyncMock, MagicMock]:
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(
        mock_context,
        mock_orchestrator,
        mock_manager,
        pair_with_reviewer=pair,
        max_review_rounds_per_task=max_rounds,
    )
    return engine, mock_orchestrator, mock_manager


# ── Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approved_on_first_round_no_redispatch():
    """Reviewer approves on round 1 → no coder re-dispatch."""
    engine, orchestrator, _ = _make_engine()

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Implement feature", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    # 1 coder dispatch via fan_out.
    orchestrator.fan_out.return_value = [_coder_result("T1")]
    # Per-task review + Phase 3 review both return approved.
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True, feedback="Looks good"),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("task", "context", None)

    assert result.success is True
    # 1 fan_out for the coder.
    assert orchestrator.fan_out.await_count == 1
    # 2 dispatches: per-task review + phase 3 review.
    assert orchestrator.dispatch.await_count == 2
    # Per-task review state captured.
    # We can't peek into state directly from outside, but we can
    # check that the test ran the per-task review on a coder task
    # and didn't dispatch any "rev" tasks (no re-dispatch).
    rev_dispatch_calls = [
        c for c in orchestrator.dispatch.await_args_list if "_pair" in c.args[0].task_id
    ]
    assert len(rev_dispatch_calls) == 0


@pytest.mark.asyncio
async def test_changes_requested_then_approved_redispatches_coder():
    """Reviewer rejects round 1, approves round 2.

    Expect 2 dispatches to the coder (1 via fan_out + 1 via dispatch
    for the re-dispatch) and 2 dispatches to the reviewer (one per
    round) plus the Phase 3 review at the end.
    """
    engine, orchestrator, _ = _make_engine()

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Implement feature", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1", answer="v1")]

    # First per-task review rejects; coder re-dispatch (dispatch);
    # second per-task review approves; Phase 3 review approves.
    orchestrator.dispatch.side_effect = [
        _review_result(approved=False, feedback="missing edge case"),
        _coder_result("T1", answer="v2"),  # re-dispatched coder
        _review_result(approved=True, feedback="now correct"),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("task", "context", None)

    assert result.success is True

    # 1 fan_out for initial coder.
    assert orchestrator.fan_out.await_count == 1
    # 4 dispatches: review1 + coder rev1 + review2 + phase3.
    assert orchestrator.dispatch.await_count == 4

    # Verify the re-dispatched coder task carried the feedback.
    coder_redispatch_calls = [
        c for c in orchestrator.dispatch.await_args_list if c.args[1] == "coder-agent"
    ]
    assert len(coder_redispatch_calls) == 1
    redispatch_task = coder_redispatch_calls[0].args[0]
    assert "REVIEWER FEEDBACK" in redispatch_task.payload["query"]
    assert "missing edge case" in redispatch_task.payload["query"]
    assert redispatch_task.payload["sub_task_id"] == "T1"


@pytest.mark.asyncio
async def test_max_rounds_exhausted_leaves_task_unaccepted():
    """Reviewer rejects every round — the last coder output is kept for
    inspection but the sub-task is NOT accepted and the workflow fails.

    (Previously this asserted ``success is True``: rounds exhausted
    silently accepted the last coder output — fail-open.)
    """
    engine, orchestrator, _ = _make_engine(max_rounds=2)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Implement feature", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1", answer="v1")]

    # Round 1: reject → re-dispatch → round 2: reject → re-dispatch
    # → round 3 review: reject again (we've hit max_rounds=2 so we stop
    # re-dispatching and leave T1 unaccepted).  Then Phase 3 review
    # approves.
    #
    # Dispatch sequence:
    #   1. per-task review #1 (reject)
    #   2. coder re-dispatch round 1
    #   3. per-task review #2 (reject)
    #   4. coder re-dispatch round 2
    #   5. per-task review #3 (reject — cap is hit, T1 unaccepted)
    #   6. Phase 3 review (approve)
    orchestrator.dispatch.side_effect = [
        _review_result(approved=False, feedback="round 1 issues"),
        _coder_result("T1", answer="v2"),
        _review_result(approved=False, feedback="round 2 issues"),
        _coder_result("T1", answer="v3"),
        _review_result(approved=False, feedback="round 3 issues"),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("task", "context", None)

    # All per-task reviews rejected → T1 unaccepted → workflow failed.
    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    assert result.error is not None
    assert "T1" in result.error
    assert "rejected" in result.error
    # 6 dispatches total.
    assert orchestrator.dispatch.await_count == 6

    # Verify the final coder result was v3 (the last re-dispatch).
    assert "v3" in result.answer

    # Verify the cap was respected — exactly 2 coder re-dispatches.
    coder_redispatches = [
        c for c in orchestrator.dispatch.await_args_list if c.args[1] == "coder-agent"
    ]
    assert len(coder_redispatches) == 2


@pytest.mark.asyncio
async def test_pair_with_reviewer_false_no_per_task_dispatches():
    """``pair_with_reviewer=False`` keeps backward-compat — no per-task review."""
    engine, orchestrator, _ = _make_engine(pair=False)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Build", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1")]
    # Only the Phase 3 batch review should dispatch.
    orchestrator.dispatch.return_value = _phase3_review(approved=True)

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("task", "context", None)

    assert result.success is True
    # 1 fan_out, 1 dispatch (only Phase 3 review).
    assert orchestrator.fan_out.await_count == 1
    assert orchestrator.dispatch.await_count == 1
    # The single dispatch was the Phase 3 review.
    call = orchestrator.dispatch.await_args
    assert call.args[1] == "review-agent"


@pytest.mark.asyncio
async def test_three_tasks_all_approved_round_one():
    """3 sub-tasks → 3 coder dispatches, 3 reviewer dispatches + Phase 3."""
    engine, orchestrator, _ = _make_engine()

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Build A", agent_type="coder"),
            SubTask(id="T2", description="Build B", agent_type="coder"),
            SubTask(id="T3", description="Build C", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1", "T2", "T3"]],
        is_single_task=False,
    )

    # 3 coders all run in one parallel group through fan_out_to_pool
    # (since they all target "coder-agent").
    orchestrator.fan_out_to_pool.return_value = [
        _coder_result("T1"),
        _coder_result("T2"),
        _coder_result("T3"),
    ]

    # All 3 per-task reviews approve, plus the final Phase 3 review.
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True, feedback="ok"),
        _review_result(approved=True, feedback="ok"),
        _review_result(approved=True, feedback="ok"),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("triple task", "context", None)

    assert result.success is True
    # 1 fan_out_to_pool for the 3 coders.
    assert orchestrator.fan_out_to_pool.await_count == 1
    # 4 dispatches: 3 per-task reviews + 1 Phase 3 review.
    assert orchestrator.dispatch.await_count == 4

    # Verify all 3 reviews were dispatched against review-agent.
    review_calls = [
        c for c in orchestrator.dispatch.await_args_list if c.args[1] == "review-agent"
    ]
    assert len(review_calls) == 4

    # The per-task review tasks carry the sub_task_id in their
    # payload (the Phase 3 cross-task review doesn't).
    per_task_review_calls = [c for c in review_calls if "sub_task_id" in c.args[0].payload]
    assert len(per_task_review_calls) == 3
    sub_task_ids = {c.args[0].payload["sub_task_id"] for c in per_task_review_calls}
    assert sub_task_ids == {"T1", "T2", "T3"}


@pytest.mark.asyncio
async def test_failed_coder_skips_per_task_review():
    """Failed coder result → no per-task review (nothing to review)."""
    engine, orchestrator, _ = _make_engine()

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Build A", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [
        _coder_result("T1", success=False, answer=""),
    ]
    # Only the Phase 3 review should dispatch.
    orchestrator.dispatch.return_value = _phase3_review(approved=True)

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("task", "context", None)

    # The workflow still produces a result, but a failed coder means
    # the sub-task is unaccepted and the workflow is not successful.
    assert result is not None
    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    # Only 1 dispatch — Phase 3 review.  No per-task review because
    # the coder result was a failure.
    assert orchestrator.dispatch.await_count == 1
    # And the single dispatch was the Phase 3 batch review, not a
    # per-task review for T1 (per-task reviews carry sub_task_id
    # in their payload).
    call = orchestrator.dispatch.await_args
    assert "sub_task_id" not in call.args[0].payload


@pytest.mark.asyncio
async def test_per_task_review_uses_dispatch_timeout():
    """Per-task review dispatch uses the same timeout as coder dispatches."""
    engine, orchestrator, _ = _make_engine()

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="x", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1")]
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        await engine.execute_workflow("task", "context", None)

    # The per-task review (first dispatch) should use the engine's
    # configured dispatch timeout (600s by default).
    per_task_call = orchestrator.dispatch.await_args_list[0]
    assert per_task_call.kwargs.get("timeout") == 600.0


@pytest.mark.asyncio
async def test_per_task_review_includes_worktree_path_when_present():
    """Reviewer dispatch carries ``cwd`` pointing at the coder's worktree."""
    from pathlib import Path

    engine, orchestrator, _ = _make_engine()

    # Stub _provision_worktree so it injects a known path into both
    # the coder's payload AND the engine's worktree cache, mimicking
    # what WorktreeManager would do.
    seeded_path = Path("/tmp/wf-test/T1")

    def _fake_provision(workflow_id: str, sub_task_id: str) -> Path:
        engine._worktrees.setdefault(workflow_id, {})[sub_task_id] = seeded_path
        return seeded_path

    engine._provision_worktree = _fake_provision  # type: ignore[assignment]

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="x", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1")]
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),
        _phase3_review(approved=True),
    ]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        await engine.execute_workflow("task", "context", None)

    per_task_call = orchestrator.dispatch.await_args_list[0]
    payload = per_task_call.args[0].payload
    # The per-task review payload should carry the worktree as cwd.
    assert "cwd" in payload
    assert str(seeded_path) == payload["cwd"]


# ── Parser tests ─────────────────────────────────────────────


def test_parse_per_task_review_empty_findings_approves():
    """A findings block with no findings parses approved (computed in code)."""
    result = TaskResult(
        task_id="r",
        agent_id="review-agent",
        success=True,
        result={
            "answer": (
                'Tests look good.\n```json\n{"review_id": "test-review-id", "findings": []}\n```'
            )
        },
    )

    parsed = WorkflowEngine._parse_per_task_review("T1", result, 12.5, review_id="test-review-id")

    assert isinstance(parsed, PerTaskReview)
    assert parsed.task_id == "T1"
    assert parsed.approved is True
    assert parsed.status == "approved"
    assert parsed.duration_ms == 12.5


def test_parse_per_task_review_changes_requested():
    """Blocking findings parse unapproved and become the issue list."""
    answer = json.dumps(
        {
            "review_id": "test-review-id",
            "findings": [
                {"severity": "HIGH", "file": "a.py", "line": 3, "issue": "Missing null check"},
                {"severity": "MEDIUM", "file": "a.py", "line": 9, "issue": "Wrong return type"},
            ],
        }
    )
    result = TaskResult(
        task_id="r",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )

    parsed = WorkflowEngine._parse_per_task_review("T2", result, 0.0, review_id="test-review-id")

    assert parsed.approved is False
    assert parsed.status == "rejected"
    assert any("Missing null check" in i for i in parsed.issues)
    assert any("Wrong return type" in i for i in parsed.issues)


def test_parse_per_task_review_legacy_prose_reject_is_error():
    """Prose-only ``approved: false`` (no findings JSON) is a reviewer
    error — still not approved, but the verdict is not read from prose."""
    result = TaskResult(
        task_id="r",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: false\n[T1] major: bad code"},
    )

    parsed = WorkflowEngine._parse_per_task_review("T1", result, 1.0, review_id="test-review-id")
    assert parsed.approved is False
    assert parsed.status == "error"


def test_parse_per_task_review_failed_dispatch_is_error():
    """Failed reviewer task → status ``error``, NOT approved.

    (Previously: auto-approve "so we don't block the workflow".)
    """
    result = TaskResult(
        task_id="r",
        agent_id="review-agent",
        success=False,
        error="reviewer crashed",
    )

    parsed = WorkflowEngine._parse_per_task_review("T1", result, 5.0, review_id="test-review-id")
    assert parsed.approved is False
    assert parsed.status == "error"
    assert "crashed" in parsed.feedback
