"""Auditors fail closed (swarm-eval PR 2).

Covers the gate semantics introduced so that unverified output can
never be reported, committed, PR'd or merged as approved:

* structured reviewer findings with the verdict computed in code
  (any HIGH/MEDIUM -> changes requested; prose is ignored);
* reviewer unavailable / dispatch raised / task failed / unparseable
  output -> status ``error`` (one review retry, coder NOT re-dispatched);
* unaccepted sub-tasks -> ``WorkflowResult.success is False``;
* unaccepted output is not auto-committed and not lifted to a PR;
* PR merge git traps: mergeable check (CONFLICTING/DIRTY/UNKNOWN),
  ``--match-head-commit`` pinning and post-merge MERGED verification.

Everything is hermetic: the orchestrator/manager are mocks and every
``gh``/``git`` call goes through a monkeypatched ``subprocess.run``.
"""

# Tests exercise the engine's private gate methods directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import subprocess
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from memfun_agent.decomposer import DecompositionResult, SubTask
from memfun_agent.gh_pr_flow import (
    MergeabilityResult,
    MergeResult,
    OpenPrResult,
    PushResult,
    check_pr_mergeable,
    merge_pr,
)
from memfun_agent.review_findings import (
    compute_verdict,
    normalize_severity,
    parse_review_output,
)
from memfun_agent.workflow import (
    PerTaskReview,
    SubTaskPRState,
    WorkflowEngine,
    WorkflowState,
    WorkflowStatus,
)
from memfun_core.types import TaskResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────


def _coder(tid: str = "T1", *, success: bool = True, files: list[str] | None = None) -> TaskResult:
    return TaskResult(
        task_id=f"wf_{tid}",
        agent_id="coder-agent",
        success=success,
        result={"answer": f"coded {tid}", "files_created": files or [], "ops": []},
        error=None if success else "coder blew up",
    )


def _review(findings: list[dict[str, Any]], prose: str = "Reviewed.") -> TaskResult:
    block = json.dumps({"findings": findings, "summary": "s"})
    return TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": f"{prose}\n```json\n{block}\n```"},
    )


def _approve() -> TaskResult:
    return _review([])


def _reject(sev: str = "MEDIUM") -> TaskResult:
    return _review([{"severity": sev, "file": "a.py", "line": 1, "issue": "bad", "fix": "fix"}])


def _prose(answer: str) -> TaskResult:
    return TaskResult(
        task_id="wf_review",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )


def _decomp(*tids: str) -> DecompositionResult:
    return DecompositionResult(
        sub_tasks=[SubTask(id=t, description=f"Do {t}", agent_type="coder") for t in tids],
        shared_spec="",
        parallelism_groups=[[t] for t in tids],
        is_single_task=False,
    )


def _engine(**kwargs: Any) -> tuple[WorkflowEngine, AsyncMock, MagicMock]:
    ctx = MagicMock()
    store = AsyncMock()
    store.get.return_value = None
    ctx.state_store = store
    orchestrator = AsyncMock()
    manager = MagicMock()
    manager.is_running.return_value = True
    manager.start_agent = AsyncMock()
    kwargs.setdefault("max_review_rounds_per_task", 2)
    engine = WorkflowEngine(ctx, orchestrator, manager, **kwargs)
    return engine, orchestrator, manager


async def _run(engine: WorkflowEngine, decomp: DecompositionResult) -> Any:
    with (
        patch.object(engine._decomposer, "adecompose", return_value=decomp),
        patch.object(engine, "_gather_shared_context", new=AsyncMock(return_value="ctx")),
    ):
        return await engine.execute_workflow("task", "ctx", None)


def _calls_to(orchestrator: AsyncMock, agent: str) -> list[Any]:
    return [c for c in orchestrator.dispatch.await_args_list if c.args[1] == agent]


def _per_task_review_calls(orchestrator: AsyncMock) -> list[Any]:
    return [
        c for c in _calls_to(orchestrator, "review-agent") if "sub_task_id" in c.args[0].payload
    ]


# ── Verdict computed in code ─────────────────────────────────


def test_prose_approved_without_findings_json_is_not_approval() -> None:
    parsed = WorkflowEngine._parse_per_task_review("T1", _prose("APPROVED\nLooks great."), 1.0)
    assert parsed.approved is False
    assert parsed.status == "error"
    assert "no parseable findings JSON" in parsed.feedback


def test_blocking_finding_overrides_reviewer_claimed_approval() -> None:
    """Reviewer says APPROVED in prose AND in its own verdict field,
    but lists a MEDIUM finding → CHANGES_REQUESTED (computed in code)."""
    block = json.dumps(
        {
            "verdict": "APPROVED",
            "findings": [{"severity": "MEDIUM", "file": "a.py", "line": 2, "issue": "no test"}],
        }
    )
    parsed = WorkflowEngine._parse_per_task_review("T1", _prose(f"APPROVED\n{block}"), 1.0)
    assert parsed.status == "rejected"
    assert parsed.approved is False
    assert any("no test" in i for i in parsed.issues)


@pytest.mark.parametrize("sev", ["HIGH", "MEDIUM", "critical", "major"])
def test_high_or_medium_requests_changes(sev: str) -> None:
    parsed = WorkflowEngine._parse_per_task_review("T1", _reject(sev), 1.0)
    assert parsed.status == "rejected"


def test_only_low_findings_approve() -> None:
    parsed = WorkflowEngine._parse_per_task_review(
        "T1",
        _review([{"severity": "LOW", "file": "a.py", "issue": "nit"}, {"severity": "minor"}]),
        1.0,
    )
    assert parsed.status == "approved"
    assert parsed.approved is True
    assert len(parsed.findings) == 2


def test_legacy_severity_aliases() -> None:
    assert normalize_severity("critical") == "HIGH"
    assert normalize_severity("major") == "MEDIUM"
    assert normalize_severity("minor") == "LOW"
    assert normalize_severity("CRITICAL") == "HIGH"
    assert normalize_severity("warning") == "MEDIUM"
    assert normalize_severity("info") == "LOW"
    # Unknown / missing severity fails closed to blocking.
    assert normalize_severity("bogus") == "MEDIUM"
    assert normalize_severity(None) == "MEDIUM"


def test_parse_takes_last_findings_block_and_tolerates_fences() -> None:
    """A quoted findings block earlier in the answer (e.g. from a diff)
    must not win over the reviewer's real final block."""
    answer = (
        'The test file contains `{"findings": []}` literally.\n\n'
        "```json\n"
        '{"findings": [{"severity": "HIGH", "file": "x.py", "line": 7, "issue": "crash"}]}\n'
        "```\n"
    )
    parsed = parse_review_output(answer)
    assert parsed is not None
    assert compute_verdict(parsed.findings) == "CHANGES_REQUESTED"
    assert parsed.findings[0].line == 7


def test_malformed_findings_entry_is_unparseable() -> None:
    assert parse_review_output('{"findings": ["not an object"]}') is None
    assert parse_review_output('{"findings": "none"}') is None
    assert parse_review_output('{"findings": [{"severity": "LOW"') is None


def test_per_task_review_status_and_approved_stay_consistent() -> None:
    r = PerTaskReview(task_id="T1", approved=True, feedback="", status="error")
    assert r.approved is False
    legacy = PerTaskReview(task_id="T1", approved=False, feedback="")
    assert legacy.status == "rejected"


# ── Fail closed in the pairing loop ──────────────────────────


@pytest.mark.asyncio
async def test_reviewer_cannot_start_means_unapproved_and_workflow_fails() -> None:
    engine, orchestrator, manager = _engine()

    def _is_running(name: str) -> bool:
        return name != "review-agent"

    manager.is_running.side_effect = _is_running

    async def _start(name: str) -> None:
        if name == "review-agent":
            raise RuntimeError("no reviewer binary")

    manager.start_agent.side_effect = _start
    orchestrator.fan_out.return_value = [_coder("T1")]

    result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    # Per-task review error + cross-task review error.
    assert result.review_errors == 2
    assert "reviewer error" in (result.error or "")
    assert "cross-task review error" in (result.error or "")
    # Nothing was dispatched: no reviewer, and the coder was not re-run.
    assert orchestrator.dispatch.await_count == 0


@pytest.mark.asyncio
async def test_dispatch_raises_retries_review_once_then_error_without_coder_redispatch() -> None:
    engine, orchestrator, _ = _engine()
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [
        RuntimeError("reviewer timeout"),
        RuntimeError("reviewer timeout again"),
        _approve(),  # cross-task review
    ]

    result = await _run(engine, _decomp("T1"))

    per_task = _per_task_review_calls(orchestrator)
    assert len(per_task) == 2  # original + exactly one retry
    assert per_task[1].args[0].task_id.endswith("_retry1")
    assert _calls_to(orchestrator, "coder-agent") == []
    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    assert result.review_errors == 1


@pytest.mark.asyncio
async def test_review_error_then_successful_retry_is_accepted() -> None:
    engine, orchestrator, _ = _engine()
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [
        _prose("APPROVED"),  # unparseable -> error
        _approve(),  # retry ok
        _approve(),  # cross-task
    ]

    result = await _run(engine, _decomp("T1"))

    assert result.success is True
    assert result.unaccepted_tasks == []
    assert _calls_to(orchestrator, "coder-agent") == []


@pytest.mark.asyncio
async def test_prose_only_reviewer_never_approves_workflow() -> None:
    engine, orchestrator, _ = _engine()
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [
        _prose("APPROVED — great work"),
        _prose("APPROVED"),
        _approve(),
    ]

    result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    assert _calls_to(orchestrator, "coder-agent") == []


@pytest.mark.asyncio
async def test_rejected_redispatches_coder_with_hypothesis_instructions() -> None:
    engine, orchestrator, _ = _engine(max_review_rounds_per_task=1)
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [
        _reject("HIGH"),
        _coder("T1"),  # re-dispatched coder
        _approve(),
        _approve(),  # cross-task
    ]

    result = await _run(engine, _decomp("T1"))

    assert result.success is True
    coder_calls = _calls_to(orchestrator, "coder-agent")
    assert len(coder_calls) == 1
    query = coder_calls[0].args[0].payload["query"]
    assert "HYPOTHESIS" in query
    assert "rebut" in query
    assert "file:line" in query
    assert "[HIGH] a.py:1" in query


@pytest.mark.asyncio
async def test_rounds_exhausted_with_zero_budget_is_unaccepted() -> None:
    engine, orchestrator, _ = _engine(max_review_rounds_per_task=0)
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [_reject(), _approve()]

    result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert result.unaccepted_tasks == ["T1"]
    assert result.review_errors == 0
    assert _calls_to(orchestrator, "coder-agent") == []


@pytest.mark.asyncio
async def test_cross_task_review_unparseable_fails_workflow() -> None:
    engine, orchestrator, _ = _engine()
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.side_effect = [
        _approve(),  # per-task
        _prose("approved: true"),  # cross-task, unparseable
        _prose("approved: true"),  # cross-task retry, unparseable
    ]

    result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert result.unaccepted_tasks == []
    assert result.review_errors == 1
    assert "cross-task review error" in (result.error or "")
    # No revision was attempted for a reviewer error.
    assert orchestrator.fan_out.await_count == 1


@pytest.mark.asyncio
async def test_all_approved_happy_path_succeeds() -> None:
    engine, orchestrator, _ = _engine()
    orchestrator.fan_out.side_effect = [[_coder("T1")], [_coder("T2")]]
    orchestrator.dispatch.side_effect = [
        _approve(),
        _review([{"severity": "LOW", "file": "b.py", "issue": "nit"}]),
        _approve(),
    ]

    result = await _run(engine, _decomp("T1", "T2"))

    assert result.success is True
    assert result.error is None
    assert result.unaccepted_tasks == []
    assert result.review_errors == 0


@pytest.mark.asyncio
async def test_pair_with_reviewer_false_has_no_per_task_gate() -> None:
    engine, orchestrator, _ = _engine(pair_with_reviewer=False)
    orchestrator.fan_out.return_value = [_coder("T1")]
    orchestrator.dispatch.return_value = _approve()

    result = await _run(engine, _decomp("T1"))

    assert result.success is True
    assert result.unaccepted_tasks == []
    assert _per_task_review_calls(orchestrator) == []


# ── Unaccepted output is never committed or PR'd ─────────────


@pytest.mark.asyncio
async def test_unaccepted_task_is_not_auto_committed(tmp_path: Path) -> None:
    engine, orchestrator, _ = _engine(
        auto_commit=True,
        project_root=tmp_path,
        max_review_rounds_per_task=0,
    )
    orchestrator.fan_out.return_value = [_coder("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [_reject(), _approve()]

    commit = MagicMock(side_effect=AssertionError("must not commit unaccepted output"))
    with patch("memfun_agent.workflow.commit_workflow_output", commit):
        result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert result.commit_result is None
    assert commit.call_count == 0


class _FakeWorktreeManager:
    BRANCH_PREFIX = "memfun"

    def __init__(self, base: Path) -> None:
        self.base = base

    def make_worktree(self, workflow_id: str, task_id: str, base_sha: str | None = None) -> Path:
        path = self.base / workflow_id / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _pr_review(verdict: str, findings: list[dict[str, Any]] | None = None) -> TaskResult:
    block = json.dumps({"verdict": verdict, "findings": findings or []})
    return TaskResult(
        task_id="wf_pr_review",
        agent_id="pr-reviewer",
        success=True,
        result={"answer": f"Reviewed PR.\n{block}"},
    )


def _gh_patches(
    *,
    open_pr: MagicMock,
    merge: MagicMock,
    mergeable: MagicMock | None = None,
    push_sha: str | None = "abc123",
) -> list[Any]:
    commit = MagicMock(
        return_value=MagicMock(committed=True, skipped=False, error=None),
    )
    return [
        patch("memfun_agent.workflow.current_repo_slug", return_value="owner/name"),
        patch("memfun_agent.workflow.commit_workflow_output", commit),
        patch(
            "memfun_agent.workflow.push_subtask_branch",
            MagicMock(return_value=PushResult(pushed=True, branch="b", head_sha=push_sha)),
        ),
        patch("memfun_agent.workflow.open_subtask_pr", open_pr),
        patch("memfun_agent.workflow.mark_pr_ready", MagicMock(return_value=True)),
        patch("memfun_agent.workflow.add_pr_review_comment", MagicMock(return_value=True)),
        patch("memfun_agent.workflow.merge_pr", merge),
        patch(
            "memfun_agent.workflow.check_pr_mergeable",
            mergeable or MagicMock(return_value=MergeabilityResult(mergeable=True)),
        ),
    ]


@pytest.mark.asyncio
async def test_unaccepted_task_is_not_opened_as_pr(tmp_path: Path) -> None:
    engine, orchestrator, _ = _engine(
        worktree_manager=_FakeWorktreeManager(tmp_path / "wt"),
        project_root=tmp_path,
        gh_pr=True,
        max_review_rounds_per_task=0,
    )
    orchestrator.fan_out.return_value = [_coder("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [_reject("HIGH"), _approve()]

    open_pr = MagicMock(side_effect=AssertionError("must not open a PR for unaccepted output"))
    merge = MagicMock(side_effect=AssertionError("must not merge"))
    with ExitStack() as stack:
        for p in _gh_patches(open_pr=open_pr, merge=merge):
            stack.enter_context(p)
        result = await _run(engine, _decomp("T1"))

    assert result.success is False
    assert open_pr.call_count == 0
    assert merge.call_count == 0
    assert _calls_to(orchestrator, "pr-reviewer") == []


@pytest.mark.asyncio
async def test_approved_pr_not_merged_when_workflow_fails(tmp_path: Path) -> None:
    """T1 is accepted and its PR approved, but the cross-task review
    errors → workflow not successful → nothing is merged."""
    engine, orchestrator, _ = _engine(
        worktree_manager=_FakeWorktreeManager(tmp_path / "wt"),
        project_root=tmp_path,
        gh_pr=True,
    )
    orchestrator.fan_out.return_value = [_coder("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [
        _approve(),  # per-task
        _pr_review("approve"),
        RuntimeError("cross-task reviewer down"),
        RuntimeError("cross-task reviewer down"),
    ]
    open_pr = MagicMock(return_value=OpenPrResult(pr_number=7, url="u", draft=True))
    merge = MagicMock(side_effect=AssertionError("must not merge"))
    with ExitStack() as stack:
        for p in _gh_patches(open_pr=open_pr, merge=merge):
            stack.enter_context(p)
        result = await _run(engine, _decomp("T1"))

    assert open_pr.call_count == 1
    assert merge.call_count == 0
    assert result.success is False


@pytest.mark.asyncio
async def test_conflicting_pr_is_not_marked_ready(tmp_path: Path) -> None:
    engine, orchestrator, _ = _engine(
        worktree_manager=_FakeWorktreeManager(tmp_path / "wt"),
        project_root=tmp_path,
        gh_pr=True,
    )
    orchestrator.fan_out.return_value = [_coder("T1", files=["a.py"])]
    orchestrator.dispatch.side_effect = [_approve(), _pr_review("approve"), _approve()]
    open_pr = MagicMock(return_value=OpenPrResult(pr_number=7, url="u", draft=True))
    merge = MagicMock(side_effect=AssertionError("must not merge a conflicting PR"))
    mergeable = MagicMock(
        return_value=MergeabilityResult(mergeable=False, error="merge conflict"),
    )
    with ExitStack() as stack:
        entered = [
            stack.enter_context(p)
            for p in _gh_patches(open_pr=open_pr, merge=merge, mergeable=mergeable)
        ]
        ready = entered[4]  # mark_pr_ready
        result = await _run(engine, _decomp("T1"))

    assert ready.call_count == 0
    assert merge.call_count == 0
    # Approved-but-conflicting PR surfaces as a merge failure.
    assert result.success is False
    assert "not mergeable" in (result.error or "")


# ── _merge_approved_prs git traps ────────────────────────────


def _bare_engine() -> WorkflowEngine:
    engine = object.__new__(WorkflowEngine)
    engine._gh_pr = True
    engine._gh_pr_merge_strategy = "squash"
    engine._emit_dashboard = AsyncMock()  # type: ignore[method-assign]
    engine._resolve_repo_slug = MagicMock(return_value="owner/name")  # type: ignore[method-assign]
    return engine


@pytest.mark.asyncio
async def test_merge_refused_without_recorded_head_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    state = WorkflowState(workflow_id="wf", status=WorkflowStatus.COMPLETED)
    state.pr_states["T1"] = SubTaskPRState(task_id="T1", branch="b", approved=True, pr_number=5)
    merge = MagicMock(side_effect=AssertionError("unpinned merge"))
    monkeypatch.setattr("memfun_agent.workflow.merge_pr", merge)
    monkeypatch.setattr(
        "memfun_agent.workflow.check_pr_mergeable",
        MagicMock(side_effect=AssertionError("not reached")),
    )

    merged, failed = await engine._merge_approved_prs(state=state, groups=[["T1"]])

    assert merged == 0
    assert failed and failed[0][0] == 5
    assert "head SHA" in failed[0][1]


@pytest.mark.asyncio
async def test_merge_skips_conflicting_pr_and_pins_head(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    state = WorkflowState(workflow_id="wf", status=WorkflowStatus.COMPLETED)
    state.pr_states["T1"] = SubTaskPRState(
        task_id="T1", branch="b1", approved=True, pr_number=1, head_sha="sha1"
    )
    state.pr_states["T2"] = SubTaskPRState(
        task_id="T2", branch="b2", approved=True, pr_number=2, head_sha="sha2"
    )

    def fake_check(repo: str, n: int, *, expected_head_sha: str | None) -> MergeabilityResult:
        if n == 2:
            return MergeabilityResult(mergeable=False, error="merge conflict (CONFLICTING)")
        return MergeabilityResult(mergeable=True, head_sha=expected_head_sha)

    merge = MagicMock(return_value=MergeResult(merged=True, pr_number=1, strategy="squash"))
    monkeypatch.setattr("memfun_agent.workflow.check_pr_mergeable", fake_check)
    monkeypatch.setattr("memfun_agent.workflow.merge_pr", merge)

    merged, failed = await engine._merge_approved_prs(state=state, groups=[["T1"], ["T2"]])

    assert merged == 1
    assert merge.call_count == 1
    assert merge.call_args.args[1] == 1
    assert merge.call_args.kwargs["match_head_commit"] == "sha1"
    assert failed == [(2, "not mergeable: merge conflict (CONFLICTING)")]
    assert state.pr_states["T2"].merged is False
    assert "not mergeable" in (state.pr_states["T2"].error or "")


# ── gh_pr_flow: mergeable / merge (subprocess mocked) ────────


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=""
    )


def _runner(
    responses: list[subprocess.CompletedProcess[str]],
    capture: list[tuple[str, ...]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def _run(cmd: tuple[str, ...] | list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        capture.append(tuple(cmd))
        if not responses:
            raise AssertionError(f"unexpected subprocess call: {cmd}")
        return responses.pop(0)

    return _run


def _view(
    mergeable: str, state: str = "CLEAN", head: str = "sha1"
) -> subprocess.CompletedProcess[str]:
    return _completed(
        json.dumps({"mergeable": mergeable, "mergeStateStatus": state, "headRefOid": head})
    )


_GH_VERSION = "gh version 2.50.0\n"


def _check(
    monkeypatch: pytest.MonkeyPatch,
    views: list[subprocess.CompletedProcess[str]],
    **kwargs: Any,
) -> tuple[MergeabilityResult, list[tuple[str, ...]], list[float]]:
    capture: list[tuple[str, ...]] = []
    sleeps: list[float] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _runner([_completed(_GH_VERSION), *views], capture),
    )
    res = check_pr_mergeable("owner/name", 9, sleep_fn=sleeps.append, **kwargs)
    return res, capture, sleeps


def test_mergeable_conflicting_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    res, capture, sleeps = _check(monkeypatch, [_view("CONFLICTING", "DIRTY")])
    assert res.mergeable is False
    assert "conflict" in (res.error or "")
    assert sleeps == []
    assert capture[1][:3] == ("gh", "pr", "view")
    assert "mergeable,mergeStateStatus,headRefOid" in capture[1]


def test_merge_state_dirty_fails_even_if_mergeable_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    res, _, _ = _check(monkeypatch, [_view("MERGEABLE", "DIRTY")])
    assert res.mergeable is False


def test_mergeable_unknown_then_conflicting(monkeypatch: pytest.MonkeyPatch) -> None:
    res, capture, sleeps = _check(
        monkeypatch,
        [_view("UNKNOWN", "UNKNOWN"), _view("CONFLICTING", "DIRTY")],
    )
    assert res.mergeable is False
    assert "conflict" in (res.error or "")
    assert len(sleeps) == 1
    assert len(capture) == 3


def test_mergeable_unknown_then_mergeable(monkeypatch: pytest.MonkeyPatch) -> None:
    res, _, sleeps = _check(
        monkeypatch,
        [_view("UNKNOWN", "UNKNOWN"), _view("UNKNOWN", "UNKNOWN"), _view("MERGEABLE", "DRAFT")],
        expected_head_sha="sha1",
    )
    assert res.mergeable is True
    assert res.error is None
    assert len(sleeps) == 2


def test_mergeable_unknown_forever_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    res, capture, sleeps = _check(
        monkeypatch,
        [_view("UNKNOWN", "UNKNOWN") for _ in range(3)],
        unknown_retries=2,
    )
    assert res.mergeable is False
    assert "UNKNOWN" in (res.error or "")
    assert len(sleeps) == 2
    assert len(capture) == 4  # version probe + 3 views


def test_mergeable_head_changed_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    res, _, _ = _check(monkeypatch, [_view("MERGEABLE", head="other")], expected_head_sha="sha1")
    assert res.mergeable is False
    assert "head changed" in (res.error or "")


def test_mergeable_gh_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    res, _, _ = _check(monkeypatch, [_completed("", returncode=1)])
    assert res.mergeable is False
    res, _, _ = _check(monkeypatch, [_completed("not json")])
    assert res.mergeable is False


def test_merge_pr_passes_match_head_commit_and_verifies_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _runner(
            [
                _completed(_GH_VERSION),
                _completed("Merged\n"),
                _completed(json.dumps({"state": "MERGED"})),
            ],
            capture,
        ),
    )
    res = merge_pr("owner/name", 9, "squash", match_head_commit="abc123")
    assert res.merged is True
    merge_cmd = capture[1]
    assert merge_cmd[:4] == ("gh", "pr", "merge", "9")
    idx = merge_cmd.index("--match-head-commit")
    assert merge_cmd[idx + 1] == "abc123"
    assert capture[2][:3] == ("gh", "pr", "view")


def test_merge_pr_exit_zero_but_not_merged_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _runner(
            [
                _completed(_GH_VERSION),
                _completed("queued\n"),
                _completed(json.dumps({"state": "OPEN"})),
            ],
            capture,
        ),
    )
    res = merge_pr("owner/name", 9, "squash", match_head_commit="abc123")
    assert res.merged is False
    assert "OPEN" in (res.error or "")


def test_merge_pr_nonzero_exit_is_failure_without_state_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "memfun_agent.gh_pr_flow.subprocess.run",
        _runner(
            [_completed(_GH_VERSION), _completed("head mismatch", returncode=1)],
            capture,
        ),
    )
    res = merge_pr("owner/name", 9, "squash", match_head_commit="abc123")
    assert res.merged is False
    assert len(capture) == 2


# ── PR-layer verdict: stricter-only, nested findings ─────────


def test_pr_verdict_approve_with_warning_becomes_request_changes() -> None:
    answer = json.dumps(
        {
            "verdict": "approve",
            "findings": [{"severity": "WARNING", "title": "no tests", "file": "a.py", "line": 1}],
        }
    )
    assert WorkflowEngine._parse_pr_verdict(answer) == "request-changes"


def test_pr_verdict_approve_with_info_findings_stays_approve() -> None:
    """Nested findings objects no longer defeat the parser."""
    answer = "LGTM\n" + json.dumps(
        {"verdict": "approve", "findings": [{"severity": "INFO", "title": "nit"}]}
    )
    assert WorkflowEngine._parse_pr_verdict(answer) == "approve"


def test_pr_verdict_approve_with_malformed_findings_is_comment() -> None:
    answer = json.dumps({"verdict": "approve", "findings": "none"})
    assert WorkflowEngine._parse_pr_verdict(answer) == "comment"
