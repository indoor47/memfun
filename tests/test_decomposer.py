"""Unit tests for TaskDecomposer.

Tests task decomposition into DAG of sub-tasks with dependency validation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from memfun_agent.decomposer import (
    SubTask,
    TaskDecomposer,
    _infer_groups,
    _parse_groups,
    _parse_sub_tasks,
    _validate_dag,
)

# ── Helper function tests ─────────────────────────────────────


def test_parse_sub_tasks_from_dict_list():
    """Parse sub-tasks from list of dicts."""
    raw = [
        {
            "id": "T1",
            "description": "Analyze code",
            "agent_type": "file",
            "inputs": ["src/"],
            "outputs": ["analysis.txt"],
            "depends_on": [],
            "max_iterations": 5,
        },
        {
            "id": "T2",
            "description": "Write tests",
            "agent_type": "test",
            "inputs": ["src/module.py"],
            "outputs": ["tests/test_module.py"],
            "depends_on": ["T1"],
            "max_iterations": 8,
        },
    ]

    tasks = _parse_sub_tasks(raw)

    assert len(tasks) == 2
    assert tasks[0].id == "T1"
    assert tasks[0].agent_type == "file"
    assert tasks[0].max_iterations == 5
    assert tasks[1].id == "T2"
    assert tasks[1].depends_on == ["T1"]


def test_parse_sub_tasks_from_json_strings():
    """Parse sub-tasks from list of JSON strings."""
    raw = [
        json.dumps({"id": "T1", "description": "task1", "agent_type": "coder"}),
        json.dumps({"id": "T2", "description": "task2", "agent_type": "test"}),
    ]

    tasks = _parse_sub_tasks(raw)

    assert len(tasks) == 2
    assert tasks[0].id == "T1"
    assert tasks[1].id == "T2"


def test_parse_sub_tasks_plain_string_fallback():
    """Plain strings become simple coder tasks."""
    raw = ["Implement user authentication", "Write API endpoints"]

    tasks = _parse_sub_tasks(raw)

    assert len(tasks) == 2
    assert tasks[0].description == "Implement user authentication"
    assert tasks[0].agent_type == "coder"
    assert tasks[0].id == "T1"
    assert tasks[1].id == "T2"


def test_parse_sub_tasks_invalid_agent_type():
    """Invalid agent_type defaults to 'coder'."""
    raw = [{"id": "T1", "description": "test", "agent_type": "invalid"}]

    tasks = _parse_sub_tasks(raw)

    assert tasks[0].agent_type == "coder"


def test_parse_sub_tasks_max_iterations_clamped():
    """max_iterations is clamped to [3, 25]."""
    raw = [
        {"id": "T1", "description": "test", "max_iterations": 1},
        {"id": "T2", "description": "test", "max_iterations": 50},
    ]

    tasks = _parse_sub_tasks(raw)

    assert tasks[0].max_iterations == 3  # clamped to min
    assert tasks[1].max_iterations == 25  # clamped to max


def test_parse_groups_from_list_of_lists():
    """Parse parallelism groups from list of lists."""
    raw = [["T1", "T2"], ["T3"], ["T4", "T5"]]

    groups = _parse_groups(raw)

    assert groups == [["T1", "T2"], ["T3"], ["T4", "T5"]]


def test_parse_groups_from_json_strings():
    """Parse groups from JSON string elements."""
    raw = ['["T1", "T2"]', '["T3"]']

    groups = _parse_groups(raw)

    assert groups == [["T1", "T2"], ["T3"]]


def test_parse_groups_comma_separated_fallback():
    """Comma-separated strings are parsed as groups."""
    raw = ["T1, T2", "T3"]

    groups = _parse_groups(raw)

    assert groups == [["T1", "T2"], ["T3"]]


def test_validate_dag_no_cycle():
    """DAG validation passes for acyclic graph."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder"),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T3", description="c", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T4", description="d", agent_type="coder", depends_on=["T2", "T3"]),
    ]

    # Should not raise
    _validate_dag(tasks)


def test_validate_dag_detects_cycle():
    """DAG validation raises on cycle."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", depends_on=["T3"]),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T3", description="c", agent_type="coder", depends_on=["T2"]),
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        _validate_dag(tasks)


def test_validate_dag_self_cycle():
    """DAG validation handles self-references."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", depends_on=["T1"]),
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        _validate_dag(tasks)


def test_validate_dag_ignores_unknown_deps():
    """DAG validation ignores dependencies to non-existent tasks."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", depends_on=["T99"]),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
    ]

    # Should not raise (T99 is ignored)
    _validate_dag(tasks)


def test_infer_groups_parallel_start():
    """Tasks with no deps form first parallel group."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder"),
        SubTask(id="T2", description="b", agent_type="coder"),
        SubTask(id="T3", description="c", agent_type="coder", depends_on=["T1", "T2"]),
    ]

    groups = _infer_groups(tasks)

    assert groups == [["T1", "T2"], ["T3"]]


def test_infer_groups_sequential():
    """Linear dependency chain creates sequential groups."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder"),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T3", description="c", agent_type="coder", depends_on=["T2"]),
    ]

    groups = _infer_groups(tasks)

    assert groups == [["T1"], ["T2"], ["T3"]]


def test_infer_groups_diamond():
    """Diamond dependency pattern."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder"),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T3", description="c", agent_type="coder", depends_on=["T1"]),
        SubTask(id="T4", description="d", agent_type="coder", depends_on=["T2", "T3"]),
    ]

    groups = _infer_groups(tasks)

    assert groups == [["T1"], ["T2", "T3"], ["T4"]]


# ── TaskDecomposer tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_adecompose_with_mock():
    """Decomposer calls DSPy and parses result."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": "T1",
            "description": "Analyze codebase",
            "agent_type": "file",
            "inputs": ["src/"],
            "outputs": [],
            "depends_on": [],
            "max_iterations": 5,
        },
        {
            "id": "T2",
            "description": "Implement feature",
            "agent_type": "coder",
            "inputs": [],
            "outputs": ["src/feature.py"],
            "depends_on": ["T1"],
            "max_iterations": 12,
        },
    ]
    mock_result.shared_spec = "Use snake_case for all functions."
    mock_result.parallelism_groups = [["T1"], ["T2"]]

    async def mock_to_thread(func, *args, **kwargs):
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("Build a feature", "Project context")

    assert len(result.sub_tasks) == 2
    assert result.sub_tasks[0].id == "T1"
    assert result.sub_tasks[1].id == "T2"
    assert result.shared_spec == "Use snake_case for all functions."
    assert result.parallelism_groups == [["T1"], ["T2"]]
    assert result.is_single_task is False


@pytest.mark.asyncio
async def test_adecompose_single_task_detection():
    """Single task result is flagged as single_task."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {"id": "T1", "description": "Simple task", "agent_type": "coder"},
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = []

    with patch.object(decomposer.decomposer, "__call__", return_value=mock_result):
        result = await decomposer.adecompose("Simple task", "context")

    assert len(result.sub_tasks) == 1
    assert result.is_single_task is True


@pytest.mark.asyncio
async def test_adecompose_infers_groups_when_missing():
    """Groups are inferred from dependencies if LLM doesn't provide them."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {"id": "T1", "description": "a", "agent_type": "coder"},
        {"id": "T2", "description": "b", "agent_type": "coder", "depends_on": ["T1"]},
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = []  # Empty

    async def mock_to_thread(func, *args, **kwargs):
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("task", "context")

    # Groups should be inferred
    assert result.parallelism_groups == [["T1"], ["T2"]]


@pytest.mark.asyncio
async def test_adecompose_cycle_detection_fallback():
    """Cycle in dependencies triggers single-task fallback."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {"id": "T1", "description": "a", "agent_type": "coder", "depends_on": ["T2"]},
        {"id": "T2", "description": "b", "agent_type": "coder", "depends_on": ["T1"]},
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = []

    with patch.object(decomposer.decomposer, "__call__", return_value=mock_result):
        result = await decomposer.adecompose("task", "context")

    # Should fall back to single task
    assert result.is_single_task is True
    assert result.sub_tasks[0].id == "T1"
    assert result.sub_tasks[0].description == "task"


@pytest.mark.asyncio
async def test_adecompose_empty_tasks_fallback():
    """Empty sub_tasks from LLM triggers fallback."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = []
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = []

    with patch.object(decomposer.decomposer, "__call__", return_value=mock_result):
        result = await decomposer.adecompose("Original task", "context")

    assert result.is_single_task is True
    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].description == "Original task"


@pytest.mark.asyncio
async def test_adecompose_exception_fallback():
    """LLM exception triggers single-task fallback."""
    decomposer = TaskDecomposer()

    with patch.object(decomposer.decomposer, "__call__", side_effect=RuntimeError("LLM error")):
        result = await decomposer.adecompose("task", "context")

    assert result.is_single_task is True
    assert result.sub_tasks[0].description == "task"


@pytest.mark.asyncio
async def test_adecompose_context_truncation():
    """Project context is truncated to 4000 chars."""
    decomposer = TaskDecomposer()
    long_context = "x" * 5000

    mock_result = MagicMock()
    mock_result.sub_tasks = [{"id": "T1", "description": "test", "agent_type": "coder"}]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = []

    captured_kwargs = {}

    async def mock_to_thread(func, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        await decomposer.adecompose("task", long_context)

        # Check context was truncated
        assert len(captured_kwargs["project_context"]) == 4000


def test_single_fallback_structure():
    """Single-task fallback has correct structure."""
    result = TaskDecomposer._single_fallback("Test description")

    assert result.is_single_task is True
    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].id == "T1"
    assert result.sub_tasks[0].description == "Test description"
    assert result.sub_tasks[0].agent_type == "coder"
    assert result.sub_tasks[0].max_iterations == 25
    assert result.shared_spec == ""
    assert result.parallelism_groups == [["T1"]]
