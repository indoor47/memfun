"""Integration test for the ``WorkflowEngine`` gh_pr flag (issue #38).

These tests assert that:

1. ``WorkflowEngine(..., gh_pr=True)`` invokes
   :func:`memfun_agent.gh_pr_flow.open_subtask_pr` once per successful
   sub-task with the right arguments (branch derived from the
   worktree manager, base = ``main``, draft = True).
2. The pr-reviewer agent is dispatched via the orchestrator with the
   PR number and the spec in its payload.
3. Approved PRs end up in the merge queue and
   :func:`memfun_agent.gh_pr_flow.merge_pr` is called with the right
   merge strategy.
4. With ``gh_pr=False`` (the default) NONE of the gh_pr_flow helpers
   are called — the flag is fully opt-in.

We patch all the gh_pr_flow helpers at the boundary so no
``subprocess.run`` or real ``gh`` command ever fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.gh_pr_flow import (
    MergeResult,
    OpenPrResult,
    PushResult,
)
from memfun_agent.workflow import WorkflowEngine
from memfun_core.types import TaskResult

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────


def _coder_result(
    tid: str,
    *,
    files: list[str] | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=f"wf_{tid}",
        agent_id="coder-agent",
        success=True,
        result={
            "answer": "done",
            "files_created": files or [],
            "ops": [],
        },
    )


def _review_result(approved: bool) -> TaskResult:
    """Per-task review verdict (the in-loop ``review-agent`` agent)."""
    body = "APPROVED" if approved else "CHANGES_REQUESTED: nope"
    return TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": body},
    )


def _pr_review_result(verdict: str) -> TaskResult:
    """pr-reviewer agent verdict — emits the AGENT.md JSON block."""
    body = (
        f'Reviewed PR.\n\n{{"verdict": "{verdict}",'
        f' "findings": [], "merge_action_taken": "skipped",'
        f' "rationale": "..."}}'
    )
    return TaskResult(
        task_id="wf_pr_review",
        agent_id="pr-reviewer",
        success=True,
        result={"answer": body},
    )


def _phase3_review() -> TaskResult:
    return TaskResult(
        task_id="wf_phase3",
        agent_id="review-agent",
        success=True,
        result={"answer": "approved: true\n\nLGTM."},
    )


class _FakeWorktreeManager:
    """Stand-in for ``WorktreeManager``.

    The engine only ever calls ``make_worktree`` and reads
    ``BRANCH_PREFIX``; we record every make_worktree call and
    return a fresh per-task directory under tmp.
    """

    BRANCH_PREFIX = "memfun"

    def __init__(self, base: Path) -> None:
        self.base = base
        self.calls: list[tuple[str, str]] = []

    def make_worktree(
        self,
        workflow_id: str,
        task_id: str,
        base_sha: str | None = None,
    ) -> Path:
        self.calls.append((workflow_id, task_id))
        path = self.base / workflow_id / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _make_engine(
    tmp_path: Path,
    *,
    gh_pr: bool,
) -> tuple[WorkflowEngine, AsyncMock, MagicMock, _FakeWorktreeManager]:
    mock_context = MagicMock()
    mock_state_store = AsyncMock()
    mock_state_store.get.return_value = None
    mock_state_store.set = AsyncMock()
    mock_context.state_store = mock_state_store

    mock_orchestrator = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.is_running.return_value = True
    mock_manager.start_agent = AsyncMock()

    wt_manager = _FakeWorktreeManager(tmp_path / "worktrees")

    engine = WorkflowEngine(
        mock_context,
        mock_orchestrator,
        mock_manager,
        worktree_manager=wt_manager,  # type: ignore[arg-type]
        pair_with_reviewer=True,
        max_review_rounds_per_task=1,
        project_root=tmp_path,
        gh_pr=gh_pr,
    )
    return engine, mock_orchestrator, mock_manager, wt_manager


# ── Tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gh_pr_off_skips_pr_flow(
    tmp_path: Path,
) -> None:
    """With ``gh_pr=False`` none of the gh_pr_flow helpers fire."""
    engine, orchestrator, _, _ = _make_engine(tmp_path, gh_pr=False)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Do thing", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1")]
    # 2 reviewer dispatches: per-task review + phase 3 review.
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),
        _phase3_review(),
    ]

    # gh_pr_flow helpers raise if accidentally called.
    with (
        patch(
            "memfun_agent.workflow.open_subtask_pr",
            side_effect=AssertionError("open_subtask_pr should not be called"),
        ),
        patch(
            "memfun_agent.workflow.push_subtask_branch",
            side_effect=AssertionError("push should not be called"),
        ),
        patch(
            "memfun_agent.workflow.merge_pr",
            side_effect=AssertionError("merge should not be called"),
        ),
        patch(
            "memfun_agent.workflow.current_repo_slug",
            side_effect=AssertionError("slug lookup should not happen"),
        ),
        patch.object(
            engine._decomposer,
            "adecompose",
            return_value=decomp,
        ),
        # The context-gather phase invokes os.getcwd + file IO — skip.
        patch.object(
            engine,
            "_gather_shared_context",
            new=AsyncMock(return_value="ctx"),
        ),
    ):
        result = await engine.execute_workflow(
            "build a thing",
            "ctx",
            None,
        )

    assert result.success is True


@pytest.mark.asyncio
async def test_gh_pr_on_opens_pr_per_subtask(
    tmp_path: Path,
) -> None:
    """``gh_pr=True`` triggers one open_subtask_pr call per successful task,
    using the worktree branch and base=main."""
    engine, orchestrator, _, _wt_manager = _make_engine(
        tmp_path,
        gh_pr=True,
    )

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Implement areas.py", agent_type="coder"),
            SubTask(id="T2", description="Implement angles.py", agent_type="coder"),
        ],
        shared_spec="",
        # Use ONE task per group to avoid hitting the pool-dispatch
        # path (which routes through fan_out_to_pool); the gh_pr
        # behaviour we're testing is per-task and runs identically
        # whether the group has 1 or N members.
        parallelism_groups=[["T1"], ["T2"]],
        is_single_task=False,
    )

    # fan_out is called once per group of size 1 → return one result each.
    orchestrator.fan_out.side_effect = [
        [_coder_result("T1", files=["areas.py"])],
        [_coder_result("T2", files=["angles.py"])],
    ]
    # T1: per-task review approves → pr-reviewer approves
    # T2: per-task review approves → pr-reviewer approves
    # then the phase-3 review at the end.
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),  # T1 per-task review
        _pr_review_result("approve"),  # T1 pr-reviewer
        _review_result(approved=True),  # T2 per-task review
        _pr_review_result("approve"),  # T2 pr-reviewer
        _phase3_review(),
    ]

    open_pr = MagicMock(
        side_effect=[
            OpenPrResult(pr_number=101, url="https://x/101", draft=True),
            OpenPrResult(pr_number=102, url="https://x/102", draft=True),
        ],
    )
    push = MagicMock(return_value=PushResult(pushed=True, branch="b"))
    commit = MagicMock(
        return_value=type(
            "CR",
            (),
            {
                "committed": True,
                "skipped": False,
                "error": None,
                "commit_sha": "deadbeef",
                "branch": "memfun/wf/T1",
                "pushed": False,
                "push_error": None,
                "files_committed": ("areas.py",),
            },
        ),
    )
    merge = MagicMock(
        side_effect=[
            MergeResult(merged=True, pr_number=101, strategy="squash"),
            MergeResult(merged=True, pr_number=102, strategy="squash"),
        ],
    )
    mark_ready = MagicMock(return_value=True)

    with (
        patch(
            "memfun_agent.workflow.current_repo_slug",
            return_value="owner/name",
        ),
        patch("memfun_agent.workflow.commit_workflow_output", commit),
        patch("memfun_agent.workflow.push_subtask_branch", push),
        patch("memfun_agent.workflow.open_subtask_pr", open_pr),
        patch("memfun_agent.workflow.mark_pr_ready", mark_ready),
        patch("memfun_agent.workflow.merge_pr", merge),
        patch.object(
            engine._decomposer,
            "adecompose",
            return_value=decomp,
        ),
        patch.object(
            engine,
            "_gather_shared_context",
            new=AsyncMock(return_value="ctx"),
        ),
        # Disable end-of-workflow auto-commit — we already test that
        # path in test_swarm_autocommit; here we only care about
        # the gh_pr_flow plumbing.
    ):
        result = await engine.execute_workflow(
            "build areas + angles",
            "ctx",
            None,
        )

    assert result.success is True

    # ── Each sub-task got its own PR opened. ────────────────────
    assert open_pr.call_count == 2
    # Both calls used base=main and the worktree branch shape.
    calls_kwargs = [c.kwargs for c in open_pr.call_args_list]
    branches = sorted(c["head"] for c in calls_kwargs)
    assert branches == ["memfun/wf-dummy/T1", "memfun/wf-dummy/T2"] or all(
        b.startswith("memfun/") and b.endswith(("T1", "T2")) for b in branches
    )
    for kwargs in calls_kwargs:
        assert kwargs["repo"] == "owner/name"
        assert kwargs["base"] == "main"
        assert kwargs["draft"] is True
        # Body includes spec.
        assert "areas" in kwargs["body"] or "angles" in kwargs["body"]

    # ── pr-reviewer was dispatched twice with pr_number + spec ─
    pr_reviewer_dispatches = [
        c for c in orchestrator.dispatch.await_args_list if c.args[1] == "pr-reviewer"
    ]
    assert len(pr_reviewer_dispatches) == 2
    for call in pr_reviewer_dispatches:
        task = call.args[0]
        assert task.payload["pr_number"] in (101, 102)
        assert task.payload["base_repo"] == "owner/name"
        assert task.payload["merge_authority"] == "approve-only"
        # Spec is forwarded so the reviewer can verify acceptance criteria.
        assert "spec" in task.payload

    # ── Approved PRs were marked ready and merged. ──────────────
    assert mark_ready.call_count == 2
    assert merge.call_count == 2
    merged_numbers = sorted(c.args[1] for c in merge.call_args_list)
    assert merged_numbers == [101, 102]


@pytest.mark.asyncio
async def test_gh_pr_changes_requested_skips_merge(
    tmp_path: Path,
) -> None:
    """When the pr-reviewer requests changes, the PR is NOT merged."""
    engine, orchestrator, _, _ = _make_engine(tmp_path, gh_pr=True)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Do thing", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),  # per-task review approves
        _pr_review_result("request-changes"),  # pr-reviewer rejects
        _phase3_review(),
    ]

    open_pr = MagicMock(
        return_value=OpenPrResult(
            pr_number=303,
            url="https://x/303",
            draft=True,
        )
    )
    push = MagicMock(return_value=PushResult(pushed=True, branch="b"))
    commit_cr = MagicMock(
        return_value=type(
            "CR",
            (),
            {
                "committed": True,
                "skipped": False,
                "error": None,
                "commit_sha": "deadbeef",
                "branch": "memfun/wf/T1",
                "pushed": False,
                "push_error": None,
                "files_committed": ("a.py",),
            },
        ),
    )
    merge = MagicMock(
        return_value=MergeResult(
            merged=False,
            pr_number=303,
            strategy="squash",
            error="should not be called",
        ),
    )
    mark_ready = MagicMock(return_value=True)
    comment = MagicMock(return_value=True)

    with (
        patch(
            "memfun_agent.workflow.current_repo_slug",
            return_value="owner/name",
        ),
        patch("memfun_agent.workflow.commit_workflow_output", commit_cr),
        patch("memfun_agent.workflow.push_subtask_branch", push),
        patch("memfun_agent.workflow.open_subtask_pr", open_pr),
        patch("memfun_agent.workflow.mark_pr_ready", mark_ready),
        patch("memfun_agent.workflow.merge_pr", merge),
        patch("memfun_agent.workflow.add_pr_review_comment", comment),
        patch.object(
            engine._decomposer,
            "adecompose",
            return_value=decomp,
        ),
        patch.object(
            engine,
            "_gather_shared_context",
            new=AsyncMock(return_value="ctx"),
        ),
    ):
        result = await engine.execute_workflow("do thing", "ctx", None)

    assert result.success is True
    # Rejected → never marked ready, never merged.
    assert mark_ready.call_count == 0
    assert merge.call_count == 0
    # A comment is posted explaining the verdict.
    assert comment.call_count >= 1


@pytest.mark.asyncio
async def test_gh_pr_no_repo_slug_skips_silently(
    tmp_path: Path,
) -> None:
    """When the project is not gh-tracked, the gh_pr layer no-ops."""
    engine, orchestrator, _, _ = _make_engine(tmp_path, gh_pr=True)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Do thing", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),
        _phase3_review(),
    ]

    open_pr = MagicMock(
        side_effect=AssertionError("open_subtask_pr must not be called"),
    )
    with (
        patch(
            "memfun_agent.workflow.current_repo_slug",
            return_value=None,
        ),
        patch("memfun_agent.workflow.open_subtask_pr", open_pr),
        patch.object(
            engine._decomposer,
            "adecompose",
            return_value=decomp,
        ),
        patch.object(
            engine,
            "_gather_shared_context",
            new=AsyncMock(return_value="ctx"),
        ),
    ):
        result = await engine.execute_workflow("do thing", "ctx", None)

    assert result.success is True
    assert open_pr.call_count == 0


@pytest.mark.asyncio
async def test_gh_pr_push_failure_does_not_open_pr(
    tmp_path: Path,
) -> None:
    """If the branch can't be pushed, we never call open_subtask_pr."""
    engine, orchestrator, _, _ = _make_engine(tmp_path, gh_pr=True)

    decomp = DecompositionResult(
        sub_tasks=[
            SubTask(id="T1", description="Do thing", agent_type="coder"),
        ],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=False,
    )

    orchestrator.fan_out.return_value = [_coder_result("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [
        _review_result(approved=True),
        _phase3_review(),
    ]

    push = MagicMock(
        return_value=PushResult(
            pushed=False,
            branch="b",
            error="no remote",
        )
    )
    open_pr = MagicMock(
        side_effect=AssertionError("open should NOT happen after push failure"),
    )
    commit_cr = MagicMock(
        return_value=type(
            "CR",
            (),
            {
                "committed": True,
                "skipped": False,
                "error": None,
                "commit_sha": "deadbeef",
                "branch": "b",
                "pushed": False,
                "push_error": None,
                "files_committed": ("a.py",),
            },
        ),
    )

    with (
        patch(
            "memfun_agent.workflow.current_repo_slug",
            return_value="owner/name",
        ),
        patch("memfun_agent.workflow.commit_workflow_output", commit_cr),
        patch("memfun_agent.workflow.push_subtask_branch", push),
        patch("memfun_agent.workflow.open_subtask_pr", open_pr),
        patch.object(
            engine._decomposer,
            "adecompose",
            return_value=decomp,
        ),
        patch.object(
            engine,
            "_gather_shared_context",
            new=AsyncMock(return_value="ctx"),
        ),
    ):
        result = await engine.execute_workflow("do thing", "ctx", None)

    assert result.success is True
    assert push.call_count == 1
    assert open_pr.call_count == 0
