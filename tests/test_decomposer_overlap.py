"""Tests for decomposition-time output overlap detection.

Issue #9 — prevent two parallel sub-tasks from writing the same file
before fan-out, instead of trying to merge divergent diffs after the
fact.  Experiment A (docs/LEARNINGS.md) showed that even at Qwen 2.5
Coder 7B, two agents independently rewriting the same shared file
produce textually-divergent edits 6/10 times.  Fixing the merge
problem is unreliable; preventing it at decomposition time is a
deterministic, model-agnostic structural fix.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from memfun_agent.decomposer import (
    DecompositionConfig,
    DecompositionError,
    SubTask,
    TaskDecomposer,
    _backfill_outputs_with_heuristic,
    _heuristic_paths_from_text,
    _normalize_path,
    detect_output_overlap,
    resolve_output_overlap,
)

# ── Detection ────────────────────────────────────────────────


def test_detect_no_overlap_returns_empty() -> None:
    """Disjoint outputs across parallel tasks → no conflicts."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["src/a.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/b.py"]),
        SubTask(id="T3", description="c", agent_type="coder", outputs=["src/c.py"]),
    ]
    groups = [["T1", "T2", "T3"]]

    assert detect_output_overlap(tasks, groups) == {}


def test_detect_pairwise_overlap_in_one_group() -> None:
    """Two parallel tasks declaring the same output is reported."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["src/models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/models.py"]),
        SubTask(id="T3", description="c", agent_type="coder", outputs=["src/views.py"]),
    ]
    groups = [["T1", "T2", "T3"]]

    overlaps = detect_output_overlap(tasks, groups)
    assert overlaps == {"src/models.py": ["T1", "T2"]}


def test_detect_three_way_overlap_in_one_group() -> None:
    """Transitive overlap (3+ writers) is captured as a single key."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["src/middleware.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/middleware.py"]),
        SubTask(id="T3", description="c", agent_type="coder", outputs=["src/middleware.py"]),
    ]
    groups = [["T1", "T2", "T3"]]

    overlaps = detect_output_overlap(tasks, groups)
    assert overlaps == {"src/middleware.py": ["T1", "T2", "T3"]}


def test_detect_ignores_overlap_across_groups() -> None:
    """Tasks in DIFFERENT groups are sequential by construction — no conflict."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["src/models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/models.py"]),
    ]
    # Distinct groups => no overlap reported.
    groups = [["T1"], ["T2"]]

    assert detect_output_overlap(tasks, groups) == {}


def test_detect_normalises_leading_dot_slash() -> None:
    """`./src/foo.py` and `src/foo.py` are the same file."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["./src/foo.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/foo.py"]),
    ]
    groups = [["T1", "T2"]]

    overlaps = detect_output_overlap(tasks, groups)
    assert overlaps == {"src/foo.py": ["T1", "T2"]}


def test_detect_case_sensitive() -> None:
    """Paths are case-sensitive (POSIX semantics)."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["src/Models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["src/models.py"]),
    ]
    groups = [["T1", "T2"]]

    assert detect_output_overlap(tasks, groups) == {}


def test_detect_handles_singleton_groups() -> None:
    """A group with 1 task can't conflict with itself."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["x.py", "x.py"]),
    ]
    groups = [["T1"]]

    assert detect_output_overlap(tasks, groups) == {}


# ── Sequencing resolver ───────────────────────────────────────


def test_resolve_sequence_three_writers_one_group() -> None:
    """3 tasks all writing models.py in one group → 3 separate groups."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T3", description="c", agent_type="coder", outputs=["models.py"]),
    ]
    groups = [["T1", "T2", "T3"]]

    new_groups, log = resolve_output_overlap(tasks, groups, strategy="sequence")

    # All three must end up sequential (each in its own group), in
    # deterministic ID order.
    assert new_groups == [["T1"], ["T2"], ["T3"]]
    assert any("sequenced" in m for m in log)


def test_resolve_sequence_preserves_safe_tasks_in_group() -> None:
    """Tasks not part of the conflict stay parallel."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T3", description="c", agent_type="coder", outputs=["views.py"]),
        SubTask(id="T4", description="d", agent_type="coder", outputs=["urls.py"]),
    ]
    groups = [["T1", "T2", "T3", "T4"]]

    new_groups, _ = resolve_output_overlap(tasks, groups, strategy="sequence")

    # T3 and T4 stay together; T1 and T2 are split out sequentially.
    assert new_groups == [["T3", "T4"], ["T1"], ["T2"]]


def test_resolve_sequence_no_conflict_is_passthrough() -> None:
    """No overlap → groups unchanged, empty log."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["a.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["b.py"]),
    ]
    groups = [["T1", "T2"]]

    new_groups, log = resolve_output_overlap(tasks, groups, strategy="sequence")

    assert new_groups == groups
    assert log == []


def test_resolve_reject_raises_with_details() -> None:
    """`reject` strategy fails fast with the conflicting paths listed."""
    tasks = [
        SubTask(id="T1", description="a", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["models.py"]),
    ]
    groups = [["T1", "T2"]]

    with pytest.raises(DecompositionError) as exc_info:
        resolve_output_overlap(tasks, groups, strategy="reject")

    msg = str(exc_info.value)
    assert "models.py" in msg
    assert "T1" in msg
    assert "T2" in msg


def test_resolve_sequence_ordering_is_deterministic() -> None:
    """Conflicting tasks are chained sorted by id, not by input order."""
    tasks = [
        SubTask(id="T9", description="a", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T2", description="b", agent_type="coder", outputs=["models.py"]),
        SubTask(id="T5", description="c", agent_type="coder", outputs=["models.py"]),
    ]
    groups = [["T9", "T2", "T5"]]

    new_groups, _ = resolve_output_overlap(tasks, groups, strategy="sequence")

    assert new_groups == [["T2"], ["T5"], ["T9"]]


# ── Path normalisation helpers ────────────────────────────────


def test_normalize_path_strips_dot_slash() -> None:
    assert _normalize_path("./src/foo.py") == "src/foo.py"


def test_normalize_path_collapses_double_slash() -> None:
    assert _normalize_path("src//foo.py") == "src/foo.py"


def test_normalize_path_strips_whitespace() -> None:
    assert _normalize_path("  src/foo.py  ") == "src/foo.py"


def test_heuristic_paths_finds_path_tokens() -> None:
    text = "Update src/auth/middleware.py and tests/test_auth.py to add a JWT check."
    paths = _heuristic_paths_from_text(text)
    assert "src/auth/middleware.py" in paths
    assert "tests/test_auth.py" in paths


def test_heuristic_paths_skips_bare_words() -> None:
    """Bare words like 'models' or 'middleware' must NOT match."""
    text = "Refactor the models and middleware classes."
    assert _heuristic_paths_from_text(text) == []


def test_backfill_outputs_only_for_writer_agents_with_empty_outputs() -> None:
    """Heuristic only fills when outputs is empty AND agent writes files."""
    tasks = [
        # Writer with empty outputs — should be filled.
        SubTask(
            id="T1",
            description="Edit src/auth/middleware.py to add a check.",
            agent_type="coder",
            outputs=[],
        ),
        # Writer with declared outputs — must NOT be overwritten.
        SubTask(
            id="T2",
            description="Edit pkg/x.py.",
            agent_type="coder",
            outputs=["explicit.py"],
        ),
        # Read-only agent — must NOT be filled even if the description
        # mentions a path.
        SubTask(
            id="T3",
            description="Read src/foo.py and report findings.",
            agent_type="file",
            outputs=[],
        ),
    ]

    backfilled = _backfill_outputs_with_heuristic(tasks)

    assert backfilled[0].outputs == ["src/auth/middleware.py"]
    assert backfilled[1].outputs == ["explicit.py"]
    assert backfilled[2].outputs == []


# ── Integration with TaskDecomposer ──────────────────────────


@pytest.mark.asyncio
async def test_adecompose_resolves_overlap_via_sequencing() -> None:
    """End-to-end: LLM emits an overlap, the decomposer sequences it."""
    decomposer = TaskDecomposer(DecompositionConfig(overlap_strategy="sequence"))

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": "T1",
            "description": "Edit models.py for X",
            "agent_type": "coder",
            "outputs": ["models.py"],
        },
        {
            "id": "T2",
            "description": "Edit models.py for Y",
            "agent_type": "coder",
            "outputs": ["models.py"],
        },
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = [["T1", "T2"]]

    async def mock_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("Refactor models", "ctx")

    assert result.parallelism_groups == [["T1"], ["T2"]]
    assert len(result.sub_tasks) == 2


@pytest.mark.asyncio
async def test_adecompose_reject_strategy_propagates_error() -> None:
    """`reject` mode surfaces the DecompositionError to the caller."""
    decomposer = TaskDecomposer(DecompositionConfig(overlap_strategy="reject"))

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": "T1",
            "description": "Edit models.py",
            "agent_type": "coder",
            "outputs": ["models.py"],
        },
        {
            "id": "T2",
            "description": "Edit models.py",
            "agent_type": "coder",
            "outputs": ["models.py"],
        },
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = [["T1", "T2"]]

    async def mock_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_result

    with (
        patch("asyncio.to_thread", side_effect=mock_to_thread),
        pytest.raises(DecompositionError, match=r"models\.py"),
    ):
        await decomposer.adecompose("Refactor models", "ctx")


@pytest.mark.asyncio
async def test_adecompose_passthrough_when_no_overlap() -> None:
    """When the LLM emits a clean plan, groups are unchanged."""
    decomposer = TaskDecomposer()  # default = sequence

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": "T1",
            "description": "Edit a.py",
            "agent_type": "coder",
            "outputs": ["a.py"],
        },
        {
            "id": "T2",
            "description": "Edit b.py",
            "agent_type": "coder",
            "outputs": ["b.py"],
        },
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = [["T1", "T2"]]

    async def mock_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("ok", "ctx")

    assert result.parallelism_groups == [["T1", "T2"]]


@pytest.mark.asyncio
async def test_adecompose_uses_heuristic_when_outputs_empty() -> None:
    """LLM emits empty outputs but a path-bearing description → heuristic backfills."""
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": "T1",
            "description": "Update src/models.py: add a new field.",
            "agent_type": "coder",
            "outputs": [],
        },
        {
            "id": "T2",
            "description": "Update src/models.py: add a new method.",
            "agent_type": "coder",
            "outputs": [],
        },
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = [["T1", "T2"]]

    async def mock_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("update models", "ctx")

    # Heuristic finds src/models.py in both descriptions, then the
    # overlap detector sequences the tasks.
    assert result.parallelism_groups == [["T1"], ["T2"]]


# ── End-to-end smoke (Experiment A's S2 scenario) ────────────


@pytest.mark.asyncio
async def test_e2e_experiment_a_s2_five_writers_to_middleware() -> None:
    """Mirrors Experiment A S2: 5 tasks all touching middleware.py.

    Without the overlap detector, all 5 would run in parallel and each
    would emit a divergent rewrite of the same file (6/10 conflicts at
    Qwen 2.5 Coder 7B).  With the detector, they get sequenced.
    """
    decomposer = TaskDecomposer()

    mock_result = MagicMock()
    mock_result.sub_tasks = [
        {
            "id": f"T{i}",
            "description": f"Add feature {i} to middleware.",
            "agent_type": "coder",
            "outputs": ["src/middleware.py"],
        }
        for i in range(1, 6)
    ]
    mock_result.shared_spec = ""
    mock_result.parallelism_groups = [["T1", "T2", "T3", "T4", "T5"]]

    async def mock_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        result = await decomposer.adecompose("Add 5 features", "ctx")

    # All 5 must end up in their own sequential group — zero parallel
    # writers to middleware.py.
    assert result.parallelism_groups == [["T1"], ["T2"], ["T3"], ["T4"], ["T5"]]
    # Sanity: detector confirms no remaining overlap.
    assert detect_output_overlap(result.sub_tasks, result.parallelism_groups) == {}
