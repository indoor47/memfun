"""Tests for hierarchical (recursive) task decomposition.

Issue #11 — single-LLM-call decomposition cannot reliably emit 100
well-scoped sub-tasks.  The fix: top-level decomposer emits 5-10
high-level branches, each branch is recursively decomposed, capped at
``DecompositionConfig.max_depth`` levels.

These tests exercise the recursion machinery with canned LLM
``Predict(TaskDecomposition)`` responses so the assertions are
deterministic — no actual LLM calls.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from memfun_agent.decomposer import (
    DecompositionConfig,
    DecompositionResult,
    SubTask,
    TaskDecomposer,
    _estimate_complexity,
    _renumber_children,
    detect_output_overlap,
)

# ── _estimate_complexity ─────────────────────────────────────


def test_complexity_small_task_is_below_threshold() -> None:
    """A short single-file edit scores below the default threshold."""
    task = SubTask(
        id="T1",
        description="Fix typo in foo",
        agent_type="coder",
        inputs=["src/foo.py"],
        outputs=["src/foo.py"],
    )
    # Default threshold is 12 — a small task scores well below.
    assert _estimate_complexity(task) < 12


def test_complexity_large_task_exceeds_threshold() -> None:
    """A many-input/output task scores above the default threshold."""
    task = SubTask(
        id="T1",
        description=(
            "Refactor the entire authentication subsystem: rewrite "
            "src/auth/middleware.py, src/auth/jwt.py, src/auth/session.py, "
            "tests/test_auth.py, and config/auth.toml. Ensure all callers "
            "in src/api/handlers.py are updated. Migrate the user model "
            "in src/models/user.py to use the new sessions backend, then "
            "wire up tests/test_user.py to cover the new shape."
        ),
        agent_type="coder",
        inputs=["src/a.py", "src/b.py", "src/c.py", "src/d.py"],
        outputs=["src/e.py", "src/f.py", "src/g.py"],
    )
    assert _estimate_complexity(task) > 12


def test_complexity_explicit_field_wins() -> None:
    """An explicit ``complexity`` attribute is preferred over the heuristic."""
    # Mocked subtask-like object with the optional attribute.
    class FakeTask:
        id = "T1"
        description = "trivial"
        agent_type = "coder"
        inputs: tuple[str, ...] = ()
        outputs: tuple[str, ...] = ()
        complexity = 99

    assert _estimate_complexity(FakeTask()) == 99


# ── _renumber_children ───────────────────────────────────────


def test_renumber_children_namespaces_under_parent() -> None:
    """Children get IDs like ``T1.1``, ``T1.2``, deps are rewired."""
    children = [
        SubTask(id="T1", description="a", agent_type="coder"),
        SubTask(id="T2", description="b", agent_type="coder", depends_on=["T1"]),
    ]
    renamed = _renumber_children("X", children)
    assert renamed[0].id == "X.1"
    assert renamed[1].id == "X.2"
    assert renamed[1].depends_on == ["X.1"]


# ── Helpers for mock LLM responses ───────────────────────────


def _mock_decomposition(
    sub_tasks: list[dict[str, Any]],
    parallelism_groups: list[list[str]] | None = None,
    shared_spec: str = "",
) -> MagicMock:
    """Build a MagicMock that mimics ``Predict(TaskDecomposition)`` output."""
    m = MagicMock()
    m.sub_tasks = sub_tasks
    m.shared_spec = shared_spec
    m.parallelism_groups = (
        parallelism_groups
        if parallelism_groups is not None
        else [[t["id"] for t in sub_tasks]]
    )
    return m


class _SequencedMockPredict:
    """Returns a list of canned responses in order, then keeps last on overflow.

    The decomposer calls ``self.decomposer(...)`` once per recursion node.
    """

    def __init__(self, responses: list[MagicMock]) -> None:
        self._responses = responses
        self._iter: Iterator[MagicMock] = iter(responses)
        self._last = responses[-1] if responses else MagicMock()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> MagicMock:
        self.calls.append(dict(kwargs))
        try:
            return next(self._iter)
        except StopIteration:
            return self._last


def _patch_to_thread_with(
    mock: _SequencedMockPredict,
) -> Any:
    """Wrap the sequenced predictor so ``asyncio.to_thread`` returns its output."""
    async def _to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return mock(**kwargs)
    return patch("asyncio.to_thread", side_effect=_to_thread)


# ── Single-pass: small task does NOT recurse ─────────────────


@pytest.mark.asyncio
async def test_small_task_does_not_recurse() -> None:
    """One-input/one-output task → no recursion, leaves are LLM's first cut."""
    decomposer = TaskDecomposer(DecompositionConfig(enable_hierarchical=True))

    plan = _mock_decomposition([
        {
            "id": "T1",
            "description": "Fix bug in foo",
            "agent_type": "coder",
            "inputs": ["src/foo.py"],
            "outputs": ["src/foo.py"],
        },
    ])
    pred = _SequencedMockPredict([plan])

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("Fix bug", "ctx")

    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].id == "T1"
    # Only one LLM call — no recursion was triggered.
    assert len(pred.calls) == 1


# ── Recursion: large branches expand ─────────────────────────


@pytest.mark.asyncio
async def test_large_branches_each_recurse() -> None:
    """8 branches of complexity > threshold → each recurses to >=4 leaves.

    Resulting leaf count: >= 8 * 4 = 32.
    """
    config = DecompositionConfig(
        enable_hierarchical=True,
        max_depth=2,
        recurse_threshold=12,
    )
    decomposer = TaskDecomposer(config)

    # Top level: 8 coarse branches, each declaring many outputs so
    # _estimate_complexity puts them above 12.
    top = _mock_decomposition([
        {
            "id": f"T{i}",
            "description": (
                f"Branch {i}: rewrite many files: "
                + ", ".join(f"src/branch{i}/mod{j}.py" for j in range(6))
            ),
            "agent_type": "coder",
            "inputs": [f"src/branch{i}/mod{j}.py" for j in range(6)],
            "outputs": [f"src/branch{i}/mod{j}.py" for j in range(6)],
        }
        for i in range(1, 9)
    ])

    # Each branch's recursion: 5 small leaves (complexity 1-2 each, no
    # further recursion).
    def _child_plan(branch_idx: int) -> MagicMock:
        return _mock_decomposition([
            {
                "id": f"T{j}",
                "description": f"Edit src/branch{branch_idx}/leaf{j}.py",
                "agent_type": "coder",
                "inputs": [f"src/branch{branch_idx}/leaf{j}.py"],
                "outputs": [f"src/branch{branch_idx}/leaf{j}.py"],
            }
            for j in range(1, 6)
        ])

    responses = [top] + [_child_plan(i) for i in range(1, 9)]
    pred = _SequencedMockPredict(responses)

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("Big refactor", "ctx")

    # 8 parents replaced by 5 leaves each = 40 leaves total.
    assert len(result.sub_tasks) >= 30
    # All leaf IDs are namespaced under their parent (T1.1, T1.2, ...).
    namespaced = [t for t in result.sub_tasks if "." in t.id]
    assert len(namespaced) >= 30
    # Sanity: the LLM was called 1 (top) + 8 (children) = 9 times.
    assert len(pred.calls) == 9


# ── Synthetic 50-file refactor: 30-50 leaves across >= 2 levels ─


@pytest.mark.asyncio
async def test_50_file_refactor_decomposes_to_many_leaves() -> None:
    """Synthetic 50-file refactor: top emits 8 branches, each emits 5-8 leaves.

    Acceptance criterion from issue #11: ≥30 leaf sub-tasks across ≥2
    levels, every leaf has non-empty inputs/outputs.
    """
    decomposer = TaskDecomposer(DecompositionConfig(
        enable_hierarchical=True,
        max_depth=2,
        recurse_threshold=8,  # lower so all 8 branches recurse
    ))

    top_branches = [
        {
            "id": f"T{i}",
            "description": (
                f"Module {i}: refactor 6 files in src/mod{i}/. "
                + ", ".join(f"src/mod{i}/f{j}.py" for j in range(6))
            ),
            "agent_type": "coder",
            "inputs": [f"src/mod{i}/f{j}.py" for j in range(6)],
            "outputs": [f"src/mod{i}/f{j}.py" for j in range(6)],
        }
        for i in range(1, 9)
    ]
    top = _mock_decomposition(top_branches)

    # Variable child fan-out: branches alternate 5..8 leaves.
    def _child_plan(branch_idx: int) -> MagicMock:
        n = 5 + (branch_idx % 4)  # 5, 6, 7, 8
        leaves = [
            {
                "id": f"T{j}",
                "description": f"Edit src/mod{branch_idx}/f{j}.py",
                "agent_type": "coder",
                "inputs": [f"src/mod{branch_idx}/f{j}.py"],
                "outputs": [f"src/mod{branch_idx}/f{j}.py"],
            }
            for j in range(1, n + 1)
        ]
        return _mock_decomposition(leaves)

    responses = [top] + [_child_plan(i) for i in range(1, 9)]
    pred = _SequencedMockPredict(responses)

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("Refactor 50 files", "ctx")

    leaves = result.sub_tasks
    assert 30 <= len(leaves) <= 60, f"got {len(leaves)} leaves"
    # Acceptance: all leaves have non-empty inputs and outputs (so the
    # #9 overlap detector can act on them).
    for t in leaves:
        assert t.inputs, f"leaf {t.id} has empty inputs"
        assert t.outputs, f"leaf {t.id} has empty outputs"
    # ≥2 levels: leaves are namespaced (parent.child).
    levels = {t.id.count(".") + 1 for t in leaves}
    assert max(levels) >= 2

    # Cross-check: detector has no remaining overlaps (because branches
    # write to disjoint subtrees).
    assert detect_output_overlap(leaves, result.parallelism_groups) == {}


# ── max_depth caps recursion ─────────────────────────────────


@pytest.mark.asyncio
async def test_max_depth_caps_recursion() -> None:
    """``max_depth`` is the structural termination invariant.

    Semantics: at recursion ``_depth``, the decomposer recurses only
    when ``_depth < max_depth``.  So ``max_depth=1`` allows one
    recursion (top → children but children stay leaves).  Even if
    every emitted sub-task has high complexity, the LLM is not
    invoked past that limit.
    """
    config = DecompositionConfig(
        enable_hierarchical=True,
        max_depth=1,  # exactly one recursion level allowed
        recurse_threshold=4,
    )
    decomposer = TaskDecomposer(config)

    # Top: 2 super-coarse branches.
    top = _mock_decomposition([
        {
            "id": "T1",
            "description": (
                "Branch 1: rewrite src/a/x.py, src/a/y.py, "
                "src/a/z.py, src/a/w.py, src/a/v.py"
            ),
            "agent_type": "coder",
            "inputs": [f"src/a/{c}.py" for c in "xyzwv"],
            "outputs": [f"src/a/{c}.py" for c in "xyzwv"],
        },
        {
            "id": "T2",
            "description": (
                "Branch 2: rewrite src/b/x.py, src/b/y.py, "
                "src/b/z.py, src/b/w.py, src/b/v.py"
            ),
            "agent_type": "coder",
            "inputs": [f"src/b/{c}.py" for c in "xyzwv"],
            "outputs": [f"src/b/{c}.py" for c in "xyzwv"],
        },
    ])

    # Depth-1 children — also coarse so they would recurse if allowed.
    coarse_child = _mock_decomposition([
        {
            "id": "T1",
            "description": (
                "Still coarse: rewrite src/x.py, src/y.py, src/z.py, "
                "src/w.py, src/v.py, src/u.py"
            ),
            "agent_type": "coder",
            "inputs": [f"src/{c}.py" for c in "xyzwvu"],
            "outputs": [f"src/{c}.py" for c in "xyzwvu"],
        },
        {
            "id": "T2",
            "description": (
                "Still coarse: rewrite src/aa.py, src/bb.py, src/cc.py, "
                "src/dd.py, src/ee.py, src/ff.py"
            ),
            "agent_type": "coder",
            "inputs": [f"src/{c}{c}.py" for c in "abcdef"],
            "outputs": [f"src/{c}{c}.py" for c in "abcdef"],
        },
    ])

    # Plenty of extras — if depth-2 recursion happens, we'd consume them.
    extras = [coarse_child] * 6
    responses = [top, coarse_child, coarse_child, *extras]
    pred = _SequencedMockPredict(responses)

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("Big refactor", "ctx")

    # Top (1) + 2 children (depth 1).  No depth-2 LLM call.
    assert len(pred.calls) == 3, (
        f"expected 3 LLM calls (max_depth=1), got {len(pred.calls)}"
    )

    # IDs are namespaced 1 level deep at most: ``T1.1``, ``T2.2`` (one dot).
    assert len(result.sub_tasks) >= 4
    for t in result.sub_tasks:
        depth = t.id.count(".")
        assert depth <= 1, (
            f"task {t.id} sits at recursion depth {depth} > max_depth=1"
        )


@pytest.mark.asyncio
async def test_max_depth_zero_disables_recursion() -> None:
    """``max_depth=0`` is a hard "no recursion" mode."""
    decomposer = TaskDecomposer(DecompositionConfig(
        enable_hierarchical=True, max_depth=0,
    ))
    plan = _mock_decomposition([
        {
            "id": "T1",
            "description": (
                "rewrite src/a.py, src/b.py, src/c.py, src/d.py, "
                "src/e.py, src/f.py, src/g.py"
            ),
            "agent_type": "coder",
            "inputs": [f"src/{c}.py" for c in "abcdefg"],
            "outputs": [f"src/{c}.py" for c in "abcdefg"],
        },
    ])
    pred = _SequencedMockPredict([plan])

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("x", "ctx")

    assert len(pred.calls) == 1
    assert len(result.sub_tasks) == 1


# ── enable_hierarchical=False disables recursion ─────────────


@pytest.mark.asyncio
async def test_disable_hierarchical_keeps_single_pass_behavior() -> None:
    """``enable_hierarchical=False`` → exactly one LLM call, no recursion."""
    decomposer = TaskDecomposer(DecompositionConfig(
        enable_hierarchical=False,
    ))

    # Even though this branch is "coarse" (would recurse if enabled), it
    # must stay as-is.
    top = _mock_decomposition([
        {
            "id": "T1",
            "description": (
                "Rewrite src/a.py, src/b.py, src/c.py, src/d.py, src/e.py, "
                "src/f.py, src/g.py — many files."
            ),
            "agent_type": "coder",
            "inputs": [f"src/{c}.py" for c in "abcdefg"],
            "outputs": [f"src/{c}.py" for c in "abcdefg"],
        },
    ])
    pred = _SequencedMockPredict([top])

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("ctx", "ctx")

    assert len(pred.calls) == 1
    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].id == "T1"


# ── Termination invariant (the non-negotiable) ───────────────


@pytest.mark.asyncio
async def test_recursion_assert_terminates_at_max_depth() -> None:
    """Direct call with ``_depth > max_depth`` MUST trip the assertion.

    This is the structural termination guarantee — even if the LLM
    output were pathological, the assert at the top of ``adecompose``
    cannot be bypassed.
    """
    decomposer = TaskDecomposer(DecompositionConfig(max_depth=2))

    # Any plan — we never get past the assert.
    plan = _mock_decomposition([
        {"id": "T1", "description": "x", "agent_type": "coder"},
    ])
    pred = _SequencedMockPredict([plan])

    with _patch_to_thread_with(pred), pytest.raises(AssertionError):
        await decomposer.adecompose("x", "ctx", _depth=3)


# ── Recursion + overlap detector integration ─────────────────


@pytest.mark.asyncio
async def test_recursion_then_overlap_detector_runs_on_leaves() -> None:
    """After hierarchical expansion, the #9 overlap detector still acts.

    Leaves all writing the same file → detector serialises them.
    """
    config = DecompositionConfig(
        enable_hierarchical=True,
        max_depth=2,
        recurse_threshold=4,
        overlap_strategy="sequence",
    )
    decomposer = TaskDecomposer(config)

    # Top: one coarse branch that recurses.
    top = _mock_decomposition([
        {
            "id": "T1",
            "description": (
                "Add 3 features to src/middleware.py, src/util.py, "
                "src/handler.py and src/x.py and src/y.py"
            ),
            "agent_type": "coder",
            "inputs": ["src/middleware.py"] * 5,
            "outputs": ["src/middleware.py"] * 5,
        },
    ])

    # Children: 3 leaves all writing to middleware.py — overlap.
    children = _mock_decomposition([
        {
            "id": "T1",
            "description": "Feature 1 for middleware",
            "agent_type": "coder",
            "inputs": ["src/middleware.py"],
            "outputs": ["src/middleware.py"],
        },
        {
            "id": "T2",
            "description": "Feature 2 for middleware",
            "agent_type": "coder",
            "inputs": ["src/middleware.py"],
            "outputs": ["src/middleware.py"],
        },
        {
            "id": "T3",
            "description": "Feature 3 for middleware",
            "agent_type": "coder",
            "inputs": ["src/middleware.py"],
            "outputs": ["src/middleware.py"],
        },
    ], parallelism_groups=[["T1", "T2", "T3"]])

    pred = _SequencedMockPredict([top, children])

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("Add features", "ctx")

    # All three children write the same file, so each gets its own group.
    assert len(result.sub_tasks) == 3
    # Each group has at most 1 task that writes to middleware.py.
    for grp in result.parallelism_groups:
        if len(grp) <= 1:
            continue
        outputs_in_group: list[str] = []
        for tid in grp:
            task = next(t for t in result.sub_tasks if t.id == tid)
            outputs_in_group.extend(task.outputs)
        # Overlap inside a parallel group is forbidden.
        assert len(outputs_in_group) == len(set(outputs_in_group))


# ── Recursion result type stability ──────────────────────────


@pytest.mark.asyncio
async def test_recursion_returns_decomposition_result_type() -> None:
    """The recursive pass must always return ``DecompositionResult``."""
    decomposer = TaskDecomposer(DecompositionConfig(enable_hierarchical=True))
    plan = _mock_decomposition([
        {"id": "T1", "description": "small", "agent_type": "coder",
         "inputs": ["a.py"], "outputs": ["a.py"]},
    ])
    pred = _SequencedMockPredict([plan])

    with _patch_to_thread_with(pred):
        result = await decomposer.adecompose("x", "y")

    assert isinstance(result, DecompositionResult)
