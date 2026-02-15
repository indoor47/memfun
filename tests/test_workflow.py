"""Unit tests for WorkflowEngine.

Tests multi-agent workflow orchestration without full integration.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.workflow import (
    ReviewResult,
    SubTaskStatus,
    WorkflowEngine,
    WorkflowState,
    WorkflowStatus,
)
from memfun_core.types import TaskResult

# ── Helper function tests ─────────────────────────────────────


def test_parse_review_result_approved():
    """Parse approved review with no issues."""
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: true\n\nAll code looks good."},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert review.approved is True
    assert len(review.issues) == 0
    # Summary is first paragraph, which is "approved: true" before the blank line
    assert "approved" in review.summary.lower()


def test_parse_review_result_not_approved():
    """Parse rejected review with issues."""
    answer = (
        "approved: false\n\n"
        "Found several issues:\n"
        "[T1] major: Missing error handling in auth module\n"
        "[T2] critical: SQL injection vulnerability\n"
        "[T3] minor: Inconsistent naming\n"
    )
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert review.approved is False
    assert len(review.issues) == 3
    assert review.issues[0].task_id == "T1"
    assert review.issues[0].severity == "major"
    assert "Missing error handling" in review.issues[0].description
    assert review.issues[1].task_id == "T2"
    assert review.issues[1].severity == "critical"
    assert review.issues[2].task_id == "T3"
    assert review.issues[2].severity == "minor"


def test_parse_review_result_case_insensitive_approved():
    """approved status is case-insensitive."""
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": "Approved:False\n\nProblems found"},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert review.approved is False


def test_parse_review_result_lowercase_task_id():
    """Task IDs are normalized to uppercase."""
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": "[t2] major: Issue description"},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert len(review.issues) == 1
    assert review.issues[0].task_id == "T2"


def test_parse_review_result_failed_review():
    """Failed review agent defaults to approved."""
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=False,
        error="Review agent crashed",
    )

    review = WorkflowEngine._parse_review_result(result)

    assert review.approved is True
    assert "Review agent failed" in review.summary


def test_parse_review_result_summary_extraction():
    """Summary is first paragraph."""
    answer = (
        "First paragraph line 1.\n"
        "First paragraph line 2.\n"
        "\n"
        "Second paragraph.\n"
        "[T1] major: issue"
    )
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert "First paragraph line 1" in review.summary
    assert "First paragraph line 2" in review.summary
    assert "Second paragraph" not in review.summary


def test_parse_review_result_summary_truncation():
    """Summary is truncated to 300 chars."""
    answer = "x" * 500 + "\n[T1] major: issue"
    result = TaskResult(
        task_id="review1",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )

    review = WorkflowEngine._parse_review_result(result)

    assert len(review.summary) <= 300


def test_build_result_single_task():
    """Build result from single sub-task."""
    state = WorkflowState(workflow_id="wf123")
    state.sub_task_statuses["T1"] = SubTaskStatus(
        task_id="wf123_T1",
        sub_task=SubTask(id="T1", description="Task 1", agent_type="coder"),
        result=TaskResult(
            task_id="wf123_T1",
            agent_id="coder-agent",
            success=True,
            result={
                "answer": "Completed task 1",
                "files_created": ["src/main.py"],
                "ops": [{"type": "write", "target": "src/main.py", "detail": ""}],
            },
        ),
    )
    review = ReviewResult(approved=True, issues=[], summary="Good")

    result = WorkflowEngine._build_result(state, review, 1000.0)

    assert result.workflow_id == "wf123"
    assert result.success is True
    assert "Task 1" in result.answer
    assert "Completed task 1" in result.answer
    assert result.files_created == ["src/main.py"]
    assert len(result.ops) == 1
    assert result.total_duration_ms == 1000.0
    assert "Good" in result.answer


def test_build_result_multiple_tasks():
    """Build result from multiple sub-tasks."""
    state = WorkflowState(workflow_id="wf123")
    state.sub_task_statuses["T1"] = SubTaskStatus(
        task_id="wf123_T1",
        sub_task=SubTask(id="T1", description="Task 1", agent_type="file"),
        result=TaskResult(
            task_id="wf123_T1",
            agent_id="file-agent",
            success=True,
            result={"answer": "Analysis complete", "files_created": [], "ops": []},
        ),
    )
    state.sub_task_statuses["T2"] = SubTaskStatus(
        task_id="wf123_T2",
        sub_task=SubTask(id="T2", description="Task 2", agent_type="coder"),
        result=TaskResult(
            task_id="wf123_T2",
            agent_id="coder-agent",
            success=True,
            result={
                "answer": "Code written",
                "files_created": ["src/feature.py", "tests/test.py"],
                "ops": [{"type": "write", "target": "src/feature.py", "detail": ""}],
            },
        ),
    )
    review = ReviewResult(approved=True, issues=[], summary="All good")

    result = WorkflowEngine._build_result(state, review, 2000.0)

    assert result.success is True
    assert "Task 1" in result.answer
    assert "Task 2" in result.answer
    assert "Analysis complete" in result.answer
    assert "Code written" in result.answer
    assert len(result.files_created) == 2
    assert "src/feature.py" in result.files_created
    assert "tests/test.py" in result.files_created
    assert len(result.ops) == 1


def test_build_result_deduplicates_files():
    """Build result deduplicates files_created."""
    state = WorkflowState(workflow_id="wf123")
    state.sub_task_statuses["T1"] = SubTaskStatus(
        task_id="wf123_T1",
        sub_task=SubTask(id="T1", description="Task 1", agent_type="coder"),
        result=TaskResult(
            task_id="wf123_T1",
            agent_id="coder-agent",
            success=True,
            result={"answer": "Done", "files_created": ["src/main.py", "src/utils.py"], "ops": []},
        ),
    )
    state.sub_task_statuses["T2"] = SubTaskStatus(
        task_id="wf123_T2",
        sub_task=SubTask(id="T2", description="Task 2", agent_type="coder"),
        result=TaskResult(
            task_id="wf123_T2",
            agent_id="coder-agent",
            success=True,
            result={"answer": "Done", "files_created": ["src/main.py"], "ops": []},
        ),
    )
    review = ReviewResult(approved=True, issues=[], summary="OK")

    result = WorkflowEngine._build_result(state, review, 1000.0)

    # src/main.py should appear only once
    assert result.files_created == ["src/main.py", "src/utils.py"]


# ── WorkflowStatus enum tests ─────────────────────────────────


def test_workflow_status_values():
    """WorkflowStatus has expected values."""
    assert WorkflowStatus.PENDING.value == "pending"
    assert WorkflowStatus.DECOMPOSING.value == "decomposing"
    assert WorkflowStatus.RUNNING.value == "running"
    assert WorkflowStatus.REVIEWING.value == "reviewing"
    assert WorkflowStatus.REVISING.value == "revising"
    assert WorkflowStatus.COMPLETED.value == "completed"
    assert WorkflowStatus.FAILED.value == "failed"


# ── WorkflowEngine tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_workflow_expanded_pipeline():
    """Even a simple decomposition runs through the full multi-agent pipeline."""
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    # Expanded pipeline: file -> coder -> review (3 groups)
    expanded_decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Analyze code", agent_type="file"),
            SubTask(
                id="T2", description="Implement changes", agent_type="coder",
                depends_on=["T1"],
            ),
            SubTask(
                id="T3", description="Review implementation", agent_type="review",
                depends_on=["T2"],
            ),
        ],
        shared_spec="",
        parallelism_groups=[["T1"], ["T2"], ["T3"]],
        is_single_task=False,
    )

    t1_result = TaskResult(
        task_id="wf_T1", agent_id="file-agent", success=True,
        result={"answer": "Analysis done", "files_created": [], "ops": []},
    )
    t2_result = TaskResult(
        task_id="wf_T2", agent_id="coder-agent", success=True,
        result={
            "answer": "Code written",
            "files_created": ["output.py"],
            "ops": [],
        },
    )
    t3_result = TaskResult(
        task_id="wf_T3", agent_id="review-agent", success=True,
        result={"answer": "Looks good", "files_created": [], "ops": []},
    )
    review_result = TaskResult(
        task_id="wf_review", agent_id="review-agent", success=True,
        result={"answer": "approved: true\n\nAll good"},
    )

    mock_orchestrator.fan_out.side_effect = [
        [t1_result],
        [t2_result],
        [t3_result],
    ]
    mock_orchestrator.dispatch.return_value = review_result

    with patch.object(engine._decomposer, "adecompose", return_value=expanded_decomp):
        result = await engine.execute_workflow("Simple task", "context", None)

    assert result.success is True
    assert "output.py" in result.files_created
    assert mock_orchestrator.fan_out.call_count == 3


@pytest.mark.asyncio
async def test_execute_workflow_multi_task_path():
    """Multi-task decomposition fans out to specialists."""
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    # Mock decomposer to return 2 tasks
    multi_decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Analyze", agent_type="file", inputs=[], outputs=[]),
            SubTask(
                id="T2", description="Implement", agent_type="coder",
                depends_on=["T1"], inputs=[], outputs=[],
            ),
        ],
        shared_spec="Use snake_case",
        parallelism_groups=[["T1"], ["T2"]],
        is_single_task=False,
    )

    # Mock fan_out responses
    t1_result = TaskResult(
        task_id="wf_T1",
        agent_id="file-agent",
        success=True,
        result={"answer": "Analysis done", "files_created": [], "ops": []},
    )
    t2_result = TaskResult(
        task_id="wf_T2",
        agent_id="coder-agent",
        success=True,
        result={"answer": "Code written", "files_created": ["src/main.py"], "ops": []},
    )

    # Mock review response
    review_result = TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: true\n\nLooks good"},
    )

    mock_orchestrator.fan_out.side_effect = [
        [t1_result],  # Group 1
        [t2_result],  # Group 2
    ]
    mock_orchestrator.dispatch.return_value = review_result

    with patch.object(engine._decomposer, "adecompose", return_value=multi_decomp):
        result = await engine.execute_workflow("Complex task", "context", None)

    assert result.success is True
    assert "Analysis done" in result.answer
    assert "Code written" in result.answer
    assert len(result.sub_task_results) == 2
    assert result.files_created == ["src/main.py"]
    assert mock_orchestrator.fan_out.call_count == 2


@pytest.mark.asyncio
async def test_execute_workflow_single_task_bypass():
    """Single-task decomposition bypasses full pipeline (fast path)."""
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    # Single-task decomposition (simple bug fix)
    single_decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Fix the bug", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=True,
    )

    coder_result = TaskResult(
        task_id="wf_single", agent_id="rlm-coder", success=True,
        result={
            "answer": "Fixed the bug",
            "files_created": ["src/main.js"],
            "ops": [{"type": "edit", "target": "src/main.js", "detail": ""}],
        },
    )

    mock_orchestrator.dispatch.return_value = coder_result

    with patch.object(engine._decomposer, "adecompose", return_value=single_decomp):
        result = await engine.execute_workflow("Fix the bug", "context", None)

    assert result.success is True
    assert "Fixed the bug" in result.answer
    # Should use dispatch (single task), NOT fan_out (multi-agent)
    mock_orchestrator.dispatch.assert_called_once()
    mock_orchestrator.fan_out.assert_not_called()


@pytest.mark.asyncio
async def test_execute_workflow_with_revision():
    """Workflow handles revision round on failed review."""
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    # Mock decomposer
    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Code", agent_type="coder", inputs=[], outputs=[]),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    # Mock initial task result
    initial_result = TaskResult(
        task_id="wf_T1",
        agent_id="coder-agent",
        success=True,
        result={"answer": "Code v1", "files_created": ["main.py"], "ops": []},
    )

    # Mock first review (rejected)
    first_review = TaskResult(
        task_id="wf_review_0",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: false\n\n[T1] major: Missing tests"},
    )

    # Mock revision result
    revision_result = TaskResult(
        task_id="wf_T1_rev1",
        agent_id="coder-agent",
        success=True,
        result={
            "answer": "Code v2 with tests",
            "files_created": ["main.py", "test.py"],
            "ops": [],
        },
    )

    # Mock second review (approved)
    second_review = TaskResult(
        task_id="wf_review_1",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: true\n\nNow complete"},
    )

    mock_orchestrator.fan_out.side_effect = [
        [initial_result],  # Initial execution
        [revision_result],  # Revision
    ]
    mock_orchestrator.dispatch.side_effect = [first_review, second_review]

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("Task", "context", None)

    assert result.success is True
    assert result.review_rounds == 1
    assert "Code v2 with tests" in result.answer


@pytest.mark.asyncio
async def test_execute_workflow_max_revision_rounds():
    """Workflow stops after max_review_rounds."""
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    # Mock decomposer
    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Code", agent_type="coder", inputs=[], outputs=[]),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    # Mock task results
    task_result = TaskResult(
        task_id="wf_T1",
        agent_id="coder-agent",
        success=True,
        result={"answer": "Code", "files_created": [], "ops": []},
    )

    # Mock reviews (always rejected)
    rejected_review = TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: false\n\n[T1] major: Still broken"},
    )

    mock_orchestrator.fan_out.return_value = [task_result]
    mock_orchestrator.dispatch.return_value = rejected_review

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        result = await engine.execute_workflow("Task", "context", None)

    # Should stop after 2 revision rounds (max_review_rounds default is 2)
    assert result.success is True
    assert result.review_rounds == 2


@pytest.mark.asyncio
async def test_execute_workflow_exception_handling():
    """Workflow handles exceptions gracefully."""
    mock_context = MagicMock()
    mock_context.state_store = AsyncMock()
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()

    engine = WorkflowEngine(mock_context, mock_orchestrator, mock_manager)

    with patch.object(engine._decomposer, "adecompose", side_effect=RuntimeError("Decomp failed")):
        result = await engine.execute_workflow("Task", "context", None)

    assert result.success is False
    assert "Decomp failed" in result.error


@pytest.mark.asyncio
async def test_workflow_callbacks():
    """Workflow calls status callbacks."""
    mock_context = MagicMock()
    mock_context.state_store = AsyncMock()
    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True

    workflow_statuses = []
    subtask_statuses = []

    def on_workflow(state):
        workflow_statuses.append(state.status)

    def on_subtask(tid, status):
        subtask_statuses.append((tid, status.status))

    engine = WorkflowEngine(
        mock_context,
        mock_orchestrator,
        mock_manager,
        on_workflow_status=on_workflow,
        on_sub_task_status=on_subtask,
    )

    # Multi-task decomposition
    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Analyze", agent_type="file"),
            SubTask(
                id="T2", description="Implement", agent_type="coder",
                depends_on=["T1"],
            ),
        ],
        shared_spec="",
        parallelism_groups=[["T1"], ["T2"]],
        is_single_task=False,
    )

    t1_result = TaskResult(
        task_id="wf_T1", agent_id="file-agent", success=True,
        result={"answer": "Done", "files_created": [], "ops": []},
    )
    t2_result = TaskResult(
        task_id="wf_T2", agent_id="coder-agent", success=True,
        result={"answer": "Done", "files_created": [], "ops": []},
    )
    review_result = TaskResult(
        task_id="wf_review", agent_id="review-agent", success=True,
        result={"answer": "approved: true\n\nLooks good"},
    )

    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store

    mock_orchestrator.fan_out.side_effect = [
        [t1_result],
        [t2_result],
    ]
    mock_orchestrator.dispatch.return_value = review_result

    with patch.object(engine._decomposer, "adecompose", return_value=decomp):
        await engine.execute_workflow("Task", "context", None)

    # Check callbacks were called for full pipeline
    assert WorkflowStatus.DECOMPOSING in workflow_statuses
    assert WorkflowStatus.RUNNING in workflow_statuses
    assert WorkflowStatus.COMPLETED in workflow_statuses
