"""WorkflowEngine: multi-agent task orchestration.

Manages the full lifecycle of a multi-agent workflow:
decompose -> fan-out parallel tasks -> collect -> review -> revise -> merge.

Uses the existing :class:`AgentOrchestrator` for dispatch/fan-out and
:class:`AgentManager` for agent lifecycle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage, TaskResult

from memfun_agent.decomposer import DecompositionResult, SubTask, TaskDecomposer
from memfun_agent.gh_pr_flow import (
    add_pr_review_comment,
    current_repo_slug,
    mark_pr_ready,
    merge_pr,
    open_subtask_pr,
    push_subtask_branch,
)
from memfun_agent.git_committer import (
    CommitResult,
    SubTaskReview,
    commit_workflow_output,
)
from memfun_agent.shared_spec import SharedSpec, SharedSpecStore
from memfun_agent.specialists import agent_name_for_type

if TYPE_CHECKING:
    from memfun_runtime.context import RuntimeContext
    from memfun_runtime.lifecycle import AgentManager
    from memfun_runtime.orchestrator import AgentOrchestrator
    from memfun_runtime.worktree import WorktreeManager

logger = get_logger("agent.workflow")


# ── Status / result types ─────────────────────────────────────


class WorkflowStatus(Enum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class SubTaskStatus:
    task_id: str
    sub_task: SubTask
    status: str = "pending"  # pending/running/completed/failed/revision
    result: TaskResult | None = None
    agent_name: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass(slots=True)
class WorkflowState:
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    decomposition: DecompositionResult | None = None
    sub_task_statuses: dict[str, SubTaskStatus] = field(default_factory=dict)
    review_rounds: int = 0
    max_review_rounds: int = 2
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error: str | None = None
    shared_context: str = ""
    project: str = ""
    # Per-sub-task auditor outcomes from the pairing loop in
    # :meth:`WorkflowEngine._execute_group`.  Indexed by sub-task ID
    # (the short ``Tn`` form, not ``workflow_id_Tn``).
    per_task_reviews: dict[str, PerTaskReview] = field(default_factory=dict)
    # Per-sub-task PR bookkeeping when running with ``gh_pr=True``.
    # Populated by :meth:`WorkflowEngine._open_pr_for_subtask` and
    # consumed by :meth:`WorkflowEngine._merge_approved_prs`.
    pr_states: dict[str, SubTaskPRState] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    task_id: str
    description: str
    detail: str
    severity: str = "major"  # critical/major/minor


@dataclass(frozen=True, slots=True)
class ReviewResult:
    approved: bool
    issues: list[ReviewIssue]
    summary: str


@dataclass(frozen=True, slots=True)
class PerTaskReview:
    """Result of a per-sub-task auditor review.

    Emitted by :meth:`WorkflowEngine._review_sub_task` once for each
    completed coder result while the pairing loop is active.
    """

    task_id: str
    approved: bool
    feedback: str
    issues: list[str] = field(default_factory=list)
    rounds: int = 1
    duration_ms: float = 0.0


@dataclass(slots=True)
class SubTaskPRState:
    """Per-sub-task GitHub PR bookkeeping (issue #38).

    Populated when :class:`WorkflowEngine` runs with ``gh_pr=True``.
    Tracks what we did with each sub-task's PR so the final merge
    pass can decide what's mergeable.

    Attributes:
        task_id: Short sub-task ID (``T1``, ``T2``, …).
        branch: Worktree branch name we pushed.
        pr_number: Opened PR number, or ``None`` if open failed.
        url: PR URL when known.
        approved: ``True`` when the pr-reviewer agent's verdict was
            ``approve``.  ``None`` while still pending.
        verdict: Raw verdict string from the reviewer
            (``approve`` / ``request-changes`` / ``comment`` / error).
        merged: ``True`` after a successful merge_pr call.
        error: Last error message that stopped progress on this PR
            (push failure, open failure, merge failure, …).
    """

    task_id: str
    branch: str
    pr_number: int | None = None
    url: str | None = None
    approved: bool | None = None
    verdict: str = ""
    merged: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    workflow_id: str
    success: bool
    answer: str
    sub_task_results: dict[str, TaskResult] = field(default_factory=dict)
    review_summary: str = ""
    review_rounds: int = 0
    total_duration_ms: float = 0.0
    files_created: list[str] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    decomposition: DecompositionResult | None = None
    # Populated by :meth:`WorkflowEngine._run_auto_commit` when
    # ``auto_commit=True`` and the workflow succeeded.  ``None`` when
    # auto-commit was disabled or the workflow failed before the
    # commit step ran.
    commit_result: CommitResult | None = None


# ── Callback type aliases ─────────────────────────────────────

OnWorkflowStatus = Callable[[WorkflowState], None]
OnSubTaskStatus = Callable[[str, SubTaskStatus], None]


# ── Workflow engine ───────────────────────────────────────────


class WorkflowEngine:
    """Orchestrates multi-agent task execution.

    Lifecycle:
    1. DECOMPOSE -- TaskDecomposer splits the task into sub-tasks
    2. EXECUTE  -- fan_out parallel groups to specialist agents
    3. REVIEW   -- ReviewAgent checks all outputs
    4. REVISE   -- re-run failing agents (max 2 rounds)
    5. MERGE    -- collect all results into WorkflowResult
    """

    def __init__(
        self,
        context: RuntimeContext,
        orchestrator: AgentOrchestrator,
        manager: AgentManager,
        *,
        on_workflow_status: OnWorkflowStatus | None = None,
        on_sub_task_status: OnSubTaskStatus | None = None,
        worktree_manager: WorktreeManager | None = None,
        pair_with_reviewer: bool = True,
        max_review_rounds_per_task: int = 2,
        dispatch_timeout_seconds: float = 600.0,
        auto_commit: bool = False,
        auto_push: bool = False,
        project_root: Path | None = None,
        gh_pr: bool = False,
        gh_pr_base: str = "main",
        gh_pr_merge_strategy: str = "squash",
    ) -> None:
        self._context = context
        self._orchestrator = orchestrator
        self._manager = manager
        self._decomposer = TaskDecomposer()
        self._spec_store = SharedSpecStore(context.state_store)
        self._on_workflow_status = on_workflow_status
        self._on_sub_task_status = on_sub_task_status
        # Auto-commit / auto-push (issue #37).  Disabled by default at
        # the engine level so unit tests don't unexpectedly touch git;
        # the CLI explicitly flips this on (the CLI default is ON).
        self._auto_commit = auto_commit
        self._auto_push = auto_push
        # Project root used as the cwd for the auto-commit step.
        # Falls back to process cwd when not passed in.
        self._project_root = project_root
        # Optional per-task git worktree provisioning.  When set, each
        # sub-task in a parallel group gets an isolated checkout under
        # ``.memfun/worktrees/<workflow_id>/<task_id>`` and its cwd is
        # injected into ``TaskMessage.payload["cwd"]``.  When None,
        # fan-out behaves exactly as before.
        self._worktree_manager = worktree_manager
        # workflow_id -> {sub_task_id -> Path}.
        self._worktrees: dict[str, dict[str, Path]] = {}
        # Implementer/auditor dyad: when True (default), each completed
        # sub-task is paired with a ``review-agent`` dispatch INSIDE
        # the fan-out loop, and the coder is re-dispatched (up to
        # ``max_review_rounds_per_task`` times) with the reviewer's
        # specific feedback when changes are requested.  The legacy
        # Phase 3 batch review still runs at the end, but as a
        # cross-task consistency check rather than the primary gate.
        self._pair_with_reviewer = pair_with_reviewer
        self._max_review_rounds_per_task = max(0, max_review_rounds_per_task)
        # Same timeout the existing coder dispatches use — keep the
        # reviewer-leg consistent with the coder-leg so a slow review
        # doesn't outlive a slow code generation.
        self._dispatch_timeout_seconds = dispatch_timeout_seconds
        # GitHub PR-per-sub-task flow (issue #38).  When enabled the
        # engine: commits the worktree, pushes the branch, opens a
        # draft PR, dispatches the ``pr-reviewer`` agent, then merges
        # approved PRs in dependency order at the end of ``_run``.
        # All steps are no-ops when ``gh`` isn't installed/authorised.
        self._gh_pr = gh_pr
        self._gh_pr_base = gh_pr_base
        self._gh_pr_merge_strategy = gh_pr_merge_strategy
        # Cached ``owner/name`` resolved on first use so we don't
        # shell out to ``gh repo view`` once per sub-task.  ``None``
        # value means "not yet resolved"; an empty string means
        # "resolved but no gh-tracked repo here" (disable feature).
        self._gh_repo_slug: str | None = None

    # ── Public API ────────────────────────────────────────────

    async def execute_workflow(
        self,
        task_description: str,
        project_context: str,
        conversation_history: list[dict[str, Any]] | None = None,
        project: str = "",
    ) -> WorkflowResult:
        """Execute a full multi-agent workflow."""
        workflow_id = uuid.uuid4().hex[:12]
        state = WorkflowState(workflow_id=workflow_id, project=project)
        start = time.perf_counter()

        try:
            return await self._run(
                state,
                task_description,
                project_context,
                conversation_history,
            )
        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.error = str(exc)
            self._emit_status(state)
            logger.error("Workflow %s failed: %s", workflow_id, exc)
            elapsed = (time.perf_counter() - start) * 1000.0
            return WorkflowResult(
                workflow_id=workflow_id,
                success=False,
                answer="",
                error=str(exc),
                total_duration_ms=elapsed,
                decomposition=state.decomposition,
            )

    # ── Internal pipeline ─────────────────────────────────────

    async def _run(
        self,
        state: WorkflowState,
        task_description: str,
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> WorkflowResult:
        start = time.perf_counter()

        # Emit workflow.started for dashboard
        await self._emit_dashboard(
            "workflow.started",
            detail=task_description[:200],
            workflow_id=state.workflow_id,
            project=state.project,
        )

        # ── Phase 1: DECOMPOSE ────────────────────────────────
        state.status = WorkflowStatus.DECOMPOSING
        self._emit_status(state)

        decomposition = await self._decomposer.adecompose(
            task_description,
            project_context,
        )
        state.decomposition = decomposition

        logger.info(
            "Decomposed into %d sub-tasks (%d groups, single=%s)",
            len(decomposition.sub_tasks),
            len(decomposition.parallelism_groups),
            decomposition.is_single_task,
        )

        # Fast path: single-task bypass — skip full pipeline.
        if decomposition.is_single_task:
            return await self._execute_single_task(
                state,
                task_description,
                project_context,
                history,
                start,
            )

        # Create shared spec.
        spec = SharedSpec(
            workflow_id=state.workflow_id,
            spec_text=decomposition.shared_spec,
        )
        for st in decomposition.sub_tasks:
            for output in st.outputs:
                spec.register_file(output, st.id)
        await self._spec_store.save(spec)

        # Ensure specialist agents are running.
        await self._ensure_agents_running(decomposition.sub_tasks, state)

        # Initialise sub-task statuses.
        for st in decomposition.sub_tasks:
            aname = agent_name_for_type(st.agent_type)
            state.sub_task_statuses[st.id] = SubTaskStatus(
                task_id=f"{state.workflow_id}_{st.id}",
                sub_task=st,
                agent_name=aname,
            )

        # ── Phase 1.5: CONTEXT GATHER ────────────────────────
        # Gather project file context ONCE and share with all
        # specialists, so they don't waste iterations rediscovering
        # the same files independently.
        state.shared_context = await self._gather_shared_context(
            project_context,
            decomposition,
        )

        # ── Phase 2: EXECUTE ──────────────────────────────────
        state.status = WorkflowStatus.RUNNING
        self._emit_status(state)

        # Final safety: ensure group IDs match sub_task_statuses.
        known_ids = set(state.sub_task_statuses.keys())
        execution_groups = decomposition.parallelism_groups
        all_group_ids = {tid for grp in execution_groups for tid in grp}
        if not all_group_ids.issubset(known_ids):
            logger.warning(
                "Parallelism group IDs %s don't match known sub-task IDs %s — rebuilding groups",
                all_group_ids - known_ids,
                known_ids,
            )
            from memfun_agent.decomposer import _infer_groups

            execution_groups = _infer_groups(
                decomposition.sub_tasks,
            )
            logger.info(
                "Rebuilt %d execution groups from deps",
                len(execution_groups),
            )

        base_context = state.shared_context or project_context
        for group in execution_groups:
            # Rebuild enriched context from base + all completed so far
            # (avoids O(n²) re-appending of prior results).
            enriched_context = self._enrich_with_prior_results(
                base_context,
                state,
            )
            await self._execute_group(
                state,
                group,
                enriched_context,
                history,
            )

        # ── Phase 3: REVIEW ───────────────────────────────────
        state.status = WorkflowStatus.REVIEWING
        self._emit_status(state)

        review = await self._run_review(state, project_context, history)

        # ── Phase 4: REVISE (up to max rounds) ────────────────
        while not review.approved and state.review_rounds < state.max_review_rounds:
            state.status = WorkflowStatus.REVISING
            state.review_rounds += 1
            self._emit_status(state)
            logger.info(
                "Revision round %d (issues: %d)",
                state.review_rounds,
                len(review.issues),
            )

            await self._run_revision(
                state,
                review,
                state.shared_context or project_context,
                history,
            )

            state.status = WorkflowStatus.REVIEWING
            self._emit_status(state)
            review = await self._run_review(state, project_context, history)

        if not review.approved:
            logger.warning(
                "Accepting results after %d review rounds (still unapproved)",
                state.review_rounds,
            )

        # ── Phase 5: MERGE ────────────────────────────────────
        state.status = WorkflowStatus.COMPLETED
        state.completed_at = time.time()
        self._emit_status(state)

        elapsed = (time.perf_counter() - start) * 1000.0
        result = self._build_result(state, review, elapsed)

        # ── Phase 6: AUTO-COMMIT (issue #37) ─────────────────
        # Only when explicitly enabled AND the workflow succeeded —
        # we never commit partial/rejected output (spec requirement).
        commit_result = (
            self._run_auto_commit(
                state=state,
                spec=task_description,
                files_written=result.files_created,
                wall_seconds=elapsed / 1000.0,
            )
            if result.success
            else None
        )
        if commit_result is not None:
            result = dataclasses.replace(
                result,
                commit_result=commit_result,
            )
            # A hard commit failure (eg. broken pre-commit hook) MUST
            # propagate as a workflow failure so callers exit non-zero.
            if not commit_result.committed and not commit_result.skipped and commit_result.error:
                result = dataclasses.replace(
                    result,
                    success=False,
                    error=f"auto-commit failed: {commit_result.error}",
                )

        # ── Phase 7: GH-PR MERGE (issue #38) ─────────────────
        # Merge approved per-sub-task PRs in dependency order.
        # Only when the gh_pr flow ran and at least one PR was
        # approved by the pr-reviewer agent earlier.
        if self._gh_pr and state.pr_states:
            merged_count, merge_failures = await self._merge_approved_prs(
                state=state,
                groups=(
                    state.decomposition.parallelism_groups
                    if state.decomposition is not None
                    else []
                ),
            )
            # Surface merge failures as a workflow failure so the
            # caller can exit non-zero — a silent `success=True` with
            # un-merged PRs hides the most common real-world failure
            # mode (the second PR conflicting with the first one's
            # merge).  Approved PRs that conflict are left OPEN on
            # GitHub for human inspection.
            if merge_failures:
                summary = ", ".join(
                    f"#{pr} ({err})" for pr, err in merge_failures
                )
                result = dataclasses.replace(
                    result,
                    success=False,
                    error=(
                        f"{merged_count}/{merged_count + len(merge_failures)} "
                        f"PRs merged; failures: {summary}"
                    ),
                )
        return result

    # ── Execution helpers ─────────────────────────────────────

    async def _execute_single_task(
        self,
        state: WorkflowState,
        task_description: str,
        project_context: str,
        history: list[dict[str, Any]] | None,
        start: float,
    ) -> WorkflowResult:
        """Fallback: dispatch to rlm-coder via orchestrator."""
        if not self._manager.is_running("rlm-coder"):
            try:
                await self._manager.start_agent("rlm-coder")
            except Exception as exc:
                logger.error("Cannot start rlm-coder: %s", exc)
                elapsed = (time.perf_counter() - start) * 1000.0
                return WorkflowResult(
                    workflow_id=state.workflow_id,
                    success=False,
                    answer="",
                    error=f"Cannot start rlm-coder: {exc}",
                    total_duration_ms=elapsed,
                    decomposition=state.decomposition,
                )

        task_msg = TaskMessage(
            task_id=f"{state.workflow_id}_single",
            agent_id="rlm-coder",
            payload={
                "type": "ask",
                "query": task_description,
                "context": project_context,
                "conversation_history": history,
            },
            correlation_id=state.workflow_id,
        )

        result = await self._orchestrator.dispatch(
            task_msg,
            "rlm-coder",
        )

        state.status = WorkflowStatus.COMPLETED
        state.completed_at = time.time()
        self._emit_status(state)

        elapsed = (time.perf_counter() - start) * 1000.0
        data = result.result or {}
        wf_result = WorkflowResult(
            workflow_id=state.workflow_id,
            success=result.success,
            answer=data.get("answer", ""),
            sub_task_results={"T1": result},
            total_duration_ms=elapsed,
            files_created=data.get("files_created", []),
            ops=data.get("ops", []),
            error=result.error,
            decomposition=state.decomposition,
        )
        # Auto-commit single-task results too (issue #37), but only
        # when the task actually succeeded — never commit a failed
        # run's partial output.
        commit_result = (
            self._run_auto_commit(
                state=state,
                spec=task_description,
                files_written=wf_result.files_created,
                wall_seconds=elapsed / 1000.0,
            )
            if wf_result.success
            else None
        )
        if commit_result is not None:
            wf_result = dataclasses.replace(
                wf_result,
                commit_result=commit_result,
            )
            if not commit_result.committed and not commit_result.skipped and commit_result.error:
                wf_result = dataclasses.replace(
                    wf_result,
                    success=False,
                    error=f"auto-commit failed: {commit_result.error}",
                )
        return wf_result

    async def _execute_group(
        self,
        state: WorkflowState,
        group: list[str],
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Execute a parallelism group, dispatching to per-instance pools.

        When the same ``agent_type`` appears multiple times in *group*,
        we delegate to :meth:`AgentOrchestrator.fan_out_to_pool` so each
        sub-task runs on its **own** ``BaseAgent`` instance (sidesteps
        the single-instance-per-name parallelism cap — see issue #12).

        For mixed-type groups (e.g. one ``file`` + one ``coder``) the
        engine still routes through a pool per agent name, but each
        pool only has the one task it needs.

        Falls back to the bare ``fan_out`` path when ``fan_out_to_pool``
        is unavailable (older orchestrators) so this method remains a
        drop-in replacement.
        """
        tasks: list[TaskMessage] = []
        agent_names: list[str] = []
        matched_tids: list[str] = []

        for tid in group:
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None:
                logger.warning(
                    "Group task ID %r not found in sub_task_statuses (known: %s)",
                    tid,
                    list(state.sub_task_statuses.keys()),
                )
                continue

            task_cwd = self._provision_worktree(state.workflow_id, tid)

            payload: dict[str, Any] = {
                "type": "ask",
                "query": sub_status.sub_task.description,
                "context": project_context,
                "workflow_id": state.workflow_id,
                "sub_task_id": tid,
                "inputs": sub_status.sub_task.inputs,
                "outputs": sub_status.sub_task.outputs,
                "conversation_history": history,
            }
            if task_cwd is not None:
                payload["cwd"] = str(task_cwd)

            task_msg = TaskMessage(
                task_id=sub_status.task_id,
                agent_id=sub_status.agent_name,
                payload=payload,
                correlation_id=state.workflow_id,
            )
            tasks.append(task_msg)
            agent_names.append(sub_status.agent_name)
            matched_tids.append(tid)

            sub_status.status = "running"
            sub_status.started_at = time.time()
            self._emit_sub_task(tid, sub_status)
            await self._emit_dashboard(
                "task.published",
                task_id=sub_status.task_id,
                agent_name=sub_status.agent_name,
                detail=sub_status.sub_task.description[:80],
                workflow_id=state.workflow_id,
                project=state.project,
            )

        if not tasks:
            logger.warning(
                "No tasks to execute in group %s — all IDs"
                " unmatched. This likely means the decomposer"
                " produced mismatched group/task IDs.",
                group,
            )
            return

        logger.info(
            "Executing group of %d tasks: %s",
            len(tasks),
            matched_tids,
        )
        for tid in matched_tids:
            ss = state.sub_task_statuses.get(tid)
            if ss:
                await self._emit_dashboard(
                    "task.picked_up",
                    task_id=ss.task_id,
                    agent_name=ss.agent_name,
                    workflow_id=state.workflow_id,
                    project=state.project,
                )
        results = await self._dispatch_group(
            tasks,
            agent_names,
            matched_tids,
        )

        # Zip results back using matched_tids (not group)
        # to ensure correct alignment when some group IDs
        # were skipped.
        for tid, result in zip(matched_tids, results, strict=False):
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None:
                continue
            sub_status.result = result
            sub_status.completed_at = time.time()
            sub_status.status = "completed" if result.success else "failed"
            self._emit_sub_task(tid, sub_status)

            dur = (
                (sub_status.completed_at - sub_status.started_at) * 1000
                if sub_status.started_at
                else result.duration_ms
            )
            await self._emit_dashboard(
                "task.completed",
                task_id=sub_status.task_id,
                agent_name=sub_status.agent_name,
                success=result.success,
                duration_ms=dur,
                workflow_id=state.workflow_id,
                project=state.project,
            )

            data = result.result or {}
            logger.info(
                "Sub-task %s (%s): success=%s, answer_len=%d, ops=%d, files=%d, error=%s",
                tid,
                sub_status.agent_name,
                result.success,
                len(data.get("answer", "")),
                len(data.get("ops", [])),
                len(data.get("files_created", [])),
                result.error,
            )

        # ── Per-task auditor pairing loop ────────────────────
        # Implementer/auditor dyad: for each sub-task that
        # succeeded, dispatch the review-agent against the
        # same worktree, and re-dispatch the coder if the
        # reviewer requests changes (capped by
        # ``max_review_rounds_per_task``).
        if self._pair_with_reviewer:
            await self._pair_review_group(
                state,
                matched_tids,
                project_context,
                history,
            )

    async def _dispatch_group(
        self,
        tasks: list[TaskMessage],
        agent_names: list[str],
        matched_tids: list[str],
    ) -> list[TaskResult]:
        """Dispatch tasks, using per-instance pools when names repeat.

        When two or more sub-tasks in the same parallel group target the
        same agent **name** (i.e. the same specialist type), the single
        already-running instance cannot truly run them in parallel —
        ``BaseAgent.handle`` serialises on the one Python object.  This
        method groups sub-tasks by agent name and routes each repeated
        group through :meth:`AgentOrchestrator.fan_out_to_pool`, which
        lazily starts additional instances.

        Singleton groups (one task → one name) take the original
        :meth:`AgentOrchestrator.fan_out` path so the legacy contract
        (and existing test mocks) is preserved.

        Returns results aligned with the input *tasks* order.
        """
        # Group indices by agent name.
        by_name: dict[str, list[int]] = {}
        for idx, name in enumerate(agent_names):
            by_name.setdefault(name, []).append(idx)

        results: list[TaskResult | None] = [None] * len(tasks)

        # If every name appears exactly once, the legacy fan_out path is
        # the simplest correct behaviour — one task per running instance.
        if all(len(v) == 1 for v in by_name.values()):
            raw = await self._orchestrator.fan_out(
                tasks,
                agent_names,
            )
            return list(raw)

        # Repeats present — dispatch one pool-call per repeated name and
        # one direct fan_out for singletons (keeps behaviour predictable
        # when other tests mock fan_out specifically).
        import asyncio

        async def _pool_dispatch(
            name: str,
            indices: list[int],
        ) -> list[TaskResult]:
            sub_tasks = [tasks[i] for i in indices]
            pool_fn = getattr(
                self._orchestrator,
                "fan_out_to_pool",
                None,
            )
            if pool_fn is None or len(indices) == 1:
                # Backward-compat or single-task: use plain fan_out.
                return await self._orchestrator.fan_out(
                    sub_tasks,
                    [name] * len(sub_tasks),
                )
            try:
                return await pool_fn(
                    sub_tasks,
                    name,
                )
            except Exception as exc:
                logger.warning(
                    "fan_out_to_pool(%s, %d) failed: %s; falling back to fan_out",
                    name,
                    len(sub_tasks),
                    exc,
                )
                return await self._orchestrator.fan_out(
                    sub_tasks,
                    [name] * len(sub_tasks),
                )

        coros = [_pool_dispatch(name, indices) for name, indices in by_name.items()]
        # Run all pool dispatches concurrently so distinct agent types
        # still execute in parallel.
        gathered = await asyncio.gather(*coros, return_exceptions=False)

        for (name, indices), sub_results in zip(
            by_name.items(),
            gathered,
            strict=False,
        ):
            for local_idx, global_idx in enumerate(indices):
                if local_idx < len(sub_results):
                    results[global_idx] = sub_results[local_idx]
                else:
                    # Pool returned fewer results than requested —
                    # synthesise a failure rather than leaving None.
                    results[global_idx] = TaskResult(
                        task_id=tasks[global_idx].task_id,
                        agent_id=name,
                        success=False,
                        error="No result returned from pool dispatch",
                    )
            _ = matched_tids  # kept for future logging hooks

        # Replace any leftover Nones with explicit failures.
        return [
            r
            if r is not None
            else TaskResult(
                task_id=tasks[i].task_id,
                agent_id=agent_names[i],
                success=False,
                error="Sub-task produced no result",
            )
            for i, r in enumerate(results)
        ]

    # ── Per-task pairing helpers ─────────────────────────────

    async def _pair_review_group(
        self,
        state: WorkflowState,
        matched_tids: list[str],
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Run the implementer/auditor dyad over every successful task.

        Each sub-task in *matched_tids* whose coder result succeeded
        is paired with a synchronous ``review-agent`` dispatch.  When
        the reviewer requests changes, the coder is re-dispatched
        (using the SAME worktree) with the feedback appended.  The
        loop is capped per task by
        ``self._max_review_rounds_per_task``.

        Reviews happen concurrently across the group via
        :func:`asyncio.gather` so a 5-coder swarm gets 5 parallel
        reviewer dispatches rather than serialising them.
        """
        import asyncio

        # Snapshot the worktree map for this workflow so reviewer
        # dispatches can point at the SAME checkout the coder used.
        wt_map = self._worktrees.get(state.workflow_id, {})

        async def _one_pair(tid: str) -> None:
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None or sub_status.result is None:
                return
            # Don't pair against a failed coder — there's nothing
            # to review.  The state already records the failure.
            if not sub_status.result.success:
                return
            await self._pair_single_task(
                state=state,
                tid=tid,
                sub_status=sub_status,
                worktree_path=wt_map.get(tid),
                project_context=project_context,
                history=history,
            )

        await asyncio.gather(
            *(_one_pair(tid) for tid in matched_tids),
            return_exceptions=False,
        )

        # ── gh_pr layer (issue #38) ──────────────────────────
        # Once the per-task in-process pairing has settled, lift
        # each successful sub-task to a real GitHub PR and route
        # it through the ``pr-reviewer`` agent.  Synchronous to
        # avoid multiplying the gh API call rate when N sub-tasks
        # finish at the same time.
        if self._gh_pr:
            await self._run_gh_pr_layer(
                state=state,
                matched_tids=matched_tids,
                wt_map=wt_map,
                spec=project_context,
                history=history,
            )

    async def _pair_single_task(
        self,
        *,
        state: WorkflowState,
        tid: str,
        sub_status: SubTaskStatus,
        worktree_path: Path | None,
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Drive the coder/reviewer dyad for one sub-task.

        Loops:
            review → if approved return → else re-dispatch coder
            with feedback → review again, up to
            ``max_review_rounds_per_task`` rounds.
        """
        rounds = 0
        while True:
            review = await self._review_sub_task(
                state=state,
                tid=tid,
                sub_status=sub_status,
                worktree_path=worktree_path,
                history=history,
            )
            rounds += 1

            # Persist the latest review for the round.
            state.per_task_reviews[tid] = PerTaskReview(
                task_id=tid,
                approved=review.approved,
                feedback=review.feedback,
                issues=review.issues,
                rounds=rounds,
                duration_ms=review.duration_ms,
            )

            if review.approved:
                logger.info(
                    "Per-task review approved %s on round %d",
                    tid,
                    rounds,
                )
                return

            # Not approved — should we re-dispatch?
            if rounds > self._max_review_rounds_per_task:
                logger.warning(
                    "Per-task review for %s exhausted %d round(s);"
                    " accepting last coder output. Feedback: %s",
                    tid,
                    self._max_review_rounds_per_task,
                    review.feedback[:120],
                )
                # Stamp the final flag so callers can see we
                # gave up — but keep the coder result so the
                # workflow can still produce output.
                return

            logger.info(
                "Per-task review for %s requested changes (round %d/%d): %s",
                tid,
                rounds,
                self._max_review_rounds_per_task,
                review.feedback[:200],
            )
            # Re-dispatch the SAME coder with the feedback.
            new_result = await self._redispatch_with_feedback(
                state=state,
                tid=tid,
                sub_status=sub_status,
                worktree_path=worktree_path,
                feedback=review.feedback,
                round_idx=rounds,
                history=history,
            )
            if new_result is None:
                # Re-dispatch failed catastrophically — give up
                # but keep the last successful coder result.
                return
            sub_status.result = new_result
            sub_status.completed_at = time.time()
            sub_status.status = "completed" if new_result.success else "failed"
            self._emit_sub_task(tid, sub_status)

            if not new_result.success:
                # Failed re-dispatch — don't keep auditing a
                # broken coder, just bail.
                logger.warning(
                    "Coder re-dispatch for %s failed: %s",
                    tid,
                    new_result.error,
                )
                return

    async def _review_sub_task(
        self,
        *,
        state: WorkflowState,
        tid: str,
        sub_status: SubTaskStatus,
        worktree_path: Path | None,
        history: list[dict[str, Any]] | None,
    ) -> PerTaskReview:
        """Dispatch the review-agent against a single sub-task.

        The reviewer is pointed at the same worktree the coder used
        (when present) so it can read actual file contents rather
        than relying on the coder's brief answer text.

        Emits ``task.reviewed`` on the dashboard event stream.
        """
        coder_result = sub_status.result
        if coder_result is None:
            # Shouldn't happen — caller guards on this — but stay
            # defensive.  Return an approved review so we don't
            # block the workflow on missing data.
            return PerTaskReview(
                task_id=tid,
                approved=True,
                feedback="No coder result to review",
                issues=[],
                rounds=1,
                duration_ms=0.0,
            )

        # Ensure the reviewer is running.  Mirror the failure
        # handling in ``_run_review`` — if we can't start the
        # reviewer, we approve by default rather than block the
        # entire workflow.
        if not self._manager.is_running("review-agent"):
            try:
                await self._manager.start_agent("review-agent")
            except Exception as exc:
                logger.warning(
                    "Cannot start review-agent for per-task review %s: %s — auto-approving",
                    tid,
                    exc,
                )
                return PerTaskReview(
                    task_id=tid,
                    approved=True,
                    feedback=f"Reviewer unavailable: {exc}",
                    issues=[],
                    rounds=1,
                    duration_ms=0.0,
                )

        data = coder_result.result or {}
        coder_answer = data.get("answer", "")
        files_created = data.get("files_created", [])

        review_query = self._build_per_task_review_query(
            sub_task=sub_status.sub_task,
            coder_answer=coder_answer,
            files_created=files_created,
            worktree_path=worktree_path,
        )

        review_round = state.per_task_reviews.get(tid)
        round_idx = (review_round.rounds + 1) if review_round else 1
        review_task = TaskMessage(
            task_id=(f"{state.workflow_id}_{tid}_review_{round_idx}"),
            agent_id="review-agent",
            payload={
                "type": "ask",
                "query": review_query,
                "context": state.shared_context or "",
                "workflow_id": state.workflow_id,
                "sub_task_id": tid,
                # Point the reviewer at the SAME worktree as the
                # coder so it reads the actual file changes.
                # Fall back to None — the agent's own handler
                # treats missing cwd as the process cwd.
                **({"cwd": str(worktree_path)} if worktree_path is not None else {}),
                "conversation_history": history,
            },
            correlation_id=state.workflow_id,
        )

        await self._emit_dashboard(
            "task.reviewed",
            task_id=review_task.task_id,
            agent_name="review-agent",
            detail=f"Reviewing {tid}",
            workflow_id=state.workflow_id,
            project=state.project,
        )

        start = time.perf_counter()
        try:
            result = await self._orchestrator.dispatch(
                review_task,
                "review-agent",
                timeout=self._dispatch_timeout_seconds,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "Per-task review dispatch for %s failed: %s",
                tid,
                exc,
            )
            return PerTaskReview(
                task_id=tid,
                approved=True,
                feedback=f"Reviewer dispatch failed: {exc}",
                issues=[],
                rounds=1,
                duration_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        review = self._parse_per_task_review(tid, result, elapsed_ms)

        # Telemetry on the dashboard event stream — show the
        # per-task pair in the Events tab and Logs tab.
        await self._emit_dashboard(
            "task.reviewed",
            task_id=review_task.task_id,
            agent_name="review-agent",
            success=review.approved,
            duration_ms=elapsed_ms,
            detail=review.feedback[:200]
            if review.feedback
            else ("APPROVED" if review.approved else "CHANGES_REQUESTED"),
            workflow_id=state.workflow_id,
            project=state.project,
        )

        return review

    @staticmethod
    def _build_per_task_review_query(
        sub_task: SubTask,
        coder_answer: str,
        files_created: list[str],
        worktree_path: Path | None,
    ) -> str:
        """Build the prompt for a per-task review."""
        location_line = (
            f"Working directory: {worktree_path}\n"
            if worktree_path is not None
            else "Working directory: <shared process cwd>\n"
        )
        files_line = (
            f"Files the coder reports having created/modified: "
            f"{', '.join(files_created) if files_created else '(none reported)'}"
        )
        inputs_line = (
            f"Declared inputs: {', '.join(sub_task.inputs) if sub_task.inputs else '(none)'}"
        )
        outputs_line = (
            f"Declared outputs: {', '.join(sub_task.outputs) if sub_task.outputs else '(none)'}"
        )

        return (
            "You are auditing a single sub-task in a multi-agent"
            " workflow. Read the ACTUAL files in the working"
            " directory below to verify the implementation —"
            " do NOT rely only on the coder's summary text.\n\n"
            f"=== SUB-TASK SPEC ({sub_task.id}) ===\n"
            f"Description: {sub_task.description}\n"
            f"{inputs_line}\n"
            f"{outputs_line}\n"
            f"{location_line}\n"
            f"{files_line}\n\n"
            "=== CODER'S SUMMARY ===\n"
            f"{coder_answer[:3000]}\n\n"
            "=== YOUR JOB ===\n"
            "Read the files at the working directory and decide"
            " whether the implementation meets the spec. Check"
            " for: missing edge cases, broken imports, type"
            " errors, missing tests, security issues, and any"
            " other quality problems.\n\n"
            "Reply with EXACTLY one of:\n"
            "  - `APPROVED` (followed by an optional brief note)\n"
            "  - `CHANGES_REQUESTED: <specific actionable feedback>`\n\n"
            "Be terse. Be specific. If you list issues, prefix"
            " each with a dash on its own line so the implementer"
            " can act on them one by one."
        )

    @staticmethod
    def _parse_per_task_review(
        tid: str,
        result: TaskResult,
        duration_ms: float,
    ) -> PerTaskReview:
        """Parse a per-task review result into a PerTaskReview.

        We deliberately favour approval on ambiguous output so the
        per-task loop doesn't block a workflow on a flaky reviewer.
        The Phase 3 cross-task review is still there to catch
        anything we let through.
        """
        if not result.success:
            return PerTaskReview(
                task_id=tid,
                approved=True,
                feedback=(
                    f"Reviewer task failed ({result.error or 'unknown'}) — defaulting to approved"
                ),
                issues=[],
                rounds=1,
                duration_ms=duration_ms,
            )

        data = result.result or {}
        answer = (data.get("answer", "") or "").strip()
        low = answer.lower()

        # Default approved=True on parse ambiguity — explicit
        # CHANGES_REQUESTED is the only "no" signal.
        approved = True
        # Be liberal about how the reviewer might phrase a
        # rejection — match the canonical token and the older
        # ``approved: false`` shape too.
        explicit_reject = (
            "changes_requested" in low
            or "changes requested" in low
            or "approved: false" in low
            or "approved:false" in low
        )
        if explicit_reject:
            approved = False
        elif low.startswith("approved"):
            # Explicit APPROVED token at start of answer is a
            # positive signal — used as a tiebreaker only when
            # there's no explicit reject marker above.
            approved = True

        # Pull issue bullets out of the body when the reviewer
        # listed them — useful as targeted feedback for the
        # re-dispatched coder.
        issues = [
            line.lstrip("-* \t").strip()
            for line in answer.splitlines()
            if line.lstrip().startswith(("-", "*")) and len(line.strip()) > 2
        ]

        return PerTaskReview(
            task_id=tid,
            approved=approved,
            feedback=answer[:4000],
            issues=issues,
            rounds=1,
            duration_ms=duration_ms,
        )

    async def _redispatch_with_feedback(
        self,
        *,
        state: WorkflowState,
        tid: str,
        sub_status: SubTaskStatus,
        worktree_path: Path | None,
        feedback: str,
        round_idx: int,
        history: list[dict[str, Any]] | None,
    ) -> TaskResult | None:
        """Re-dispatch the same coder with reviewer feedback."""
        sub_task = sub_status.sub_task
        revised_description = (
            f"{sub_task.description}\n\n"
            "=== REVIEWER FEEDBACK (round "
            f"{round_idx}/{self._max_review_rounds_per_task}) ===\n"
            "Your previous output was rejected by the auditor."
            " Address the following BEFORE re-submitting — do"
            " NOT start over from scratch; modify the existing"
            " files in the working directory.\n\n"
            f"{feedback}"
        )

        payload: dict[str, Any] = {
            "type": "ask",
            "query": revised_description,
            "context": state.shared_context or "",
            "workflow_id": state.workflow_id,
            "sub_task_id": tid,
            "inputs": sub_task.inputs,
            "outputs": sub_task.outputs,
            "conversation_history": history,
        }
        if worktree_path is not None:
            payload["cwd"] = str(worktree_path)

        revised_task = TaskMessage(
            task_id=f"{state.workflow_id}_{tid}_pair{round_idx}",
            agent_id=sub_status.agent_name,
            payload=payload,
            correlation_id=state.workflow_id,
        )

        sub_status.status = "revision"
        sub_status.started_at = time.time()
        self._emit_sub_task(tid, sub_status)

        await self._emit_dashboard(
            "task.published",
            task_id=revised_task.task_id,
            agent_name=sub_status.agent_name,
            detail=f"Pair re-dispatch round {round_idx}: {tid}",
            workflow_id=state.workflow_id,
            project=state.project,
        )

        try:
            return await self._orchestrator.dispatch(
                revised_task,
                sub_status.agent_name,
                timeout=self._dispatch_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Coder re-dispatch for %s round %d failed: %s",
                tid,
                round_idx,
                exc,
            )
            return None

    async def _run_review(
        self,
        state: WorkflowState,
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> ReviewResult:
        """Cross-task consistency review (post-pair-loop).

        With per-task pairing enabled (the default), every sub-task
        has already been individually reviewed and either approved
        or sent back to the coder for revision.  This Phase 3
        review is a final pass over the COMBINED outputs that looks
        only for **inter-task** issues — duplication across files,
        inconsistent naming/conventions between modules, missing
        ``__init__`` exports, type mismatches across module
        boundaries, etc.

        When ``pair_with_reviewer`` is disabled this method
        retains its original behaviour as the primary quality
        gate.
        """
        # Build review context from ALL sub-task results.
        review_parts: list[str] = []
        for tid, sub_status in state.sub_task_statuses.items():
            if sub_status.result is None:
                continue
            data = sub_status.result.result or {}
            answer = data.get("answer", "")[:2000]
            files = ", ".join(data.get("files_created", []))
            status_label = (
                "SUCCESS"
                if sub_status.result.success
                else f"FAILED: {sub_status.result.error or 'unknown'}"
            )
            review_parts.append(
                f"=== Sub-task {tid} ({sub_status.agent_name})"
                f" [{status_label}] ===\n"
                f"Description: {sub_status.sub_task.description}\n"
                f"Answer: {answer}\n"
                f"Files: {files}\n"
            )

        if not review_parts:
            logger.warning("No sub-task results found for review (all results are None)")
            return ReviewResult(
                approved=True,
                issues=[],
                summary="No sub-task results available",
            )

        if self._pair_with_reviewer and state.per_task_reviews:
            review_query = (
                "All sub-tasks have ALREADY been individually reviewed"
                " and approved by a per-task auditor pair. Look only"
                " for INTER-task issues that the per-task reviews"
                " could not see:\n"
                "  - Duplicated logic across files\n"
                "  - Inconsistent naming/conventions between modules\n"
                "  - Missing ``__init__`` exports\n"
                "  - Type mismatches across module boundaries\n"
                "  - Conflicting file edits or import cycles\n\n"
                "If everything is consistent, reply with"
                " `approved: true`.\n\n" + "\n".join(review_parts)
            )
        else:
            review_query = (
                "Review the outputs of the following sub-tasks for"
                " consistency, quality, and adherence to the shared"
                " specification. Check for file conflicts, import"
                " issues, and integration problems.\n\n" + "\n".join(review_parts)
            )

        # Ensure review agent running.
        if not self._manager.is_running("review-agent"):
            try:
                await self._manager.start_agent("review-agent")
            except Exception as exc:
                logger.warning("Cannot start review-agent: %s", exc)
                return ReviewResult(
                    approved=True,
                    issues=[],
                    summary=f"Review skipped: {exc}",
                )

        review_task = TaskMessage(
            task_id=f"{state.workflow_id}_review_{state.review_rounds}",
            agent_id="review-agent",
            payload={
                "type": "ask",
                "query": review_query,
                "context": project_context,
                "workflow_id": state.workflow_id,
                "conversation_history": history,
            },
            correlation_id=state.workflow_id,
        )

        result = await self._orchestrator.dispatch(
            review_task,
            "review-agent",
            timeout=300.0,
        )

        return self._parse_review_result(result)

    async def _run_revision(
        self,
        state: WorkflowState,
        review: ReviewResult,
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Re-run specific agents based on review issues."""
        revision_tasks: list[TaskMessage] = []
        revision_agents: list[str] = []
        revision_tids: list[str] = []

        for issue in review.issues:
            tid = issue.task_id
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None:
                continue

            revision_query = (
                f"REVISION REQUIRED: {issue.description}\n\n"
                f"Your original output had the following issue:\n"
                f"{issue.detail}\n\n"
                f"Fix this issue while maintaining compatibility with "
                f"other agents' outputs."
            )

            rev_payload: dict[str, Any] = {
                "type": "ask",
                "query": revision_query,
                "context": project_context,
                "workflow_id": state.workflow_id,
                "sub_task_id": tid,
                "conversation_history": history,
            }
            # Reuse the existing worktree if one was provisioned for
            # this sub_task during the original fan-out.
            existing_path = self._worktrees.get(
                state.workflow_id,
                {},
            ).get(tid)
            if existing_path is not None:
                rev_payload["cwd"] = str(existing_path)

            rev_task = TaskMessage(
                task_id=f"{state.workflow_id}_{tid}_rev{state.review_rounds}",
                agent_id=sub_status.agent_name,
                payload=rev_payload,
                correlation_id=state.workflow_id,
            )
            revision_tasks.append(rev_task)
            revision_agents.append(sub_status.agent_name)
            revision_tids.append(tid)

            sub_status.status = "revision"
            self._emit_sub_task(tid, sub_status)

        if not revision_tasks:
            return

        results = await self._orchestrator.fan_out(
            revision_tasks,
            revision_agents,
        )

        for tid, result in zip(revision_tids, results, strict=False):
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is not None:
                sub_status.result = result
                sub_status.completed_at = time.time()
                sub_status.status = "completed" if result.success else "failed"
                self._emit_sub_task(tid, sub_status)

    # ── Context gathering ─────────────────────────────────────

    async def _gather_shared_context(
        self,
        project_context: str,
        decomposition: DecompositionResult,
    ) -> str:
        """Gather project file context ONCE for all specialists.

        Uses :class:`ContextGatherer` to read relevant files so
        specialist agents don't each waste iterations re-discovering
        the same codebase.  For small projects (< 200 KB), reads
        ALL files with zero LLM calls.  For larger projects, uses
        :class:`ContextPlanner` with the combined sub-task descriptions
        to select the most relevant files.

        Returns the enriched context string.
        """
        from memfun_agent.code_map import build_code_map, code_map_to_string
        from memfun_agent.context_first import (
            ContextFirstConfig,
            ContextGatherer,
            ContextPlanner,
            build_file_manifest,
        )

        project_root = os.getcwd()
        config = ContextFirstConfig()
        manifest = build_file_manifest(project_root)

        if not manifest:
            logger.info("No source files found, using raw project context")
            return project_context

        total_size = sum(size for _, size in manifest)
        gatherer = ContextGatherer(
            max_bytes=config.max_gather_bytes,
            max_files=config.max_files,
        )

        if total_size <= config.max_context_bytes:
            # Small project: read everything (0 LLM calls).
            logger.info(
                "Reading all %d files (%d KB) for shared context",
                len(manifest),
                total_size // 1024,
            )
            return gatherer.read_all_files(manifest, project_root)

        # Large project: use planner with combined descriptions.
        all_descriptions = "\n".join(f"- {st.description}" for st in decomposition.sub_tasks)
        combined_query = (
            f"The following sub-tasks need to be solved:\n"
            f"{all_descriptions}\n\n"
            f"Gather the files needed for ALL of these sub-tasks."
        )
        logger.info(
            "Using ContextPlanner for %d files (%d KB)",
            len(manifest),
            total_size // 1024,
        )

        try:
            planner = ContextPlanner()
            plan = await planner.aplan(
                query=combined_query,
                file_manifest=code_map_to_string(
                    build_code_map(project_root, manifest=manifest),
                    max_tokens=2000,
                ),
                project_summary=project_context[:2000],
            )
            return await gatherer.agather(
                files=plan.files_to_read,
                project_root=project_root,
                search_patterns=plan.search_patterns or None,
            )
        except Exception as exc:
            logger.warning("Context planner failed: %s, falling back", exc)
            return project_context

    # ── Result forwarding ─────────────────────────────────────

    @staticmethod
    def _enrich_with_prior_results(
        base_context: str,
        state: WorkflowState,
    ) -> str:
        """Append completed sub-task answers to the context.

        This ensures subsequent dependency groups can see what prior
        agents discovered or produced.
        """
        parts: list[str] = []
        for tid in sorted(state.sub_task_statuses):
            ss = state.sub_task_statuses[tid]
            if ss.status not in ("completed", "failed"):
                continue
            if ss.result is None:
                continue
            data = ss.result.result or {}
            answer = data.get("answer", "")
            files = data.get("files_created", [])
            ops = data.get("ops", [])
            if not answer and not files:
                continue
            section = f"=== PRIOR AGENT RESULT: {tid} ({ss.agent_name}) ===\n"
            if answer:
                section += f"{answer[:3000]}\n"
            if files:
                section += f"Files created: {', '.join(files)}\n"
            if ops:
                writes = [o for o in ops if isinstance(o, dict) and o.get("type") == "write"]
                edits = [o for o in ops if isinstance(o, dict) and o.get("type") == "edit"]
                if writes or edits:
                    section += f"Operations: {len(writes)} writes, {len(edits)} edits\n"
            parts.append(section)

        if not parts:
            return base_context

        prior_block = "\n\n=== COMPLETED AGENT RESULTS ===\n" + "\n".join(parts)
        return base_context + prior_block

    # ── Worktree provisioning ─────────────────────────────────

    def _provision_worktree(
        self,
        workflow_id: str,
        sub_task_id: str,
    ) -> Path | None:
        """Create (or return cached) worktree path for a sub-task.

        Returns ``None`` when no :class:`WorktreeManager` is
        configured (legacy behaviour: all specialists share the
        process cwd) or when worktree creation fails — in that
        case the engine logs and falls back to the shared cwd
        rather than aborting the whole workflow.
        """
        if self._worktree_manager is None:
            return None

        cache = self._worktrees.setdefault(workflow_id, {})
        cached = cache.get(sub_task_id)
        if cached is not None:
            return cached

        try:
            path = self._worktree_manager.make_worktree(
                workflow_id,
                sub_task_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to provision worktree for %s/%s: %s",
                workflow_id,
                sub_task_id,
                exc,
            )
            return None
        cache[sub_task_id] = path
        return path

    def cleanup_worktrees(self, workflow_id: str) -> None:
        """Remove every worktree provisioned for *workflow_id*.

        Safe to call when no manager is configured or no worktrees
        were created — it is a no-op in those cases.  The caller
        decides when (if ever) to invoke this — keeping the
        worktrees around after a failed run is often useful for
        debugging.
        """
        if self._worktree_manager is None:
            return
        paths = self._worktrees.pop(workflow_id, {})
        for tid, path in paths.items():
            try:
                self._worktree_manager.cleanup_worktree(path)
            except Exception as exc:
                logger.warning(
                    "Failed to clean up worktree for %s/%s: %s",
                    workflow_id,
                    tid,
                    exc,
                )

    # ── GitHub PR flow (issue #38) ────────────────────────────

    def _resolve_repo_slug(self) -> str | None:
        """Resolve ``owner/name`` once per workflow and cache it.

        Returns the slug on success, ``None`` when this project is
        not gh-tracked — in which case the gh_pr layer silently
        no-ops and the rest of the workflow proceeds normally.
        """
        if self._gh_repo_slug is None:
            project_root = self._project_root or Path.cwd()
            slug = current_repo_slug(project_root)
            # Cache as empty string for "resolved → not available"
            # so we don't shell out again on each sub-task.
            self._gh_repo_slug = slug or ""
        return self._gh_repo_slug or None

    async def _run_gh_pr_layer(
        self,
        *,
        state: WorkflowState,
        matched_tids: list[str],
        wt_map: dict[str, Path],
        spec: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Open one PR per successful sub-task and route it for review.

        For each *tid* in *matched_tids* whose coder result succeeded:

        1. Resolve the worktree branch (``memfun/<workflow>/<task>``).
        2. Commit the worktree's changes via
           :func:`commit_workflow_output` so the push has something
           to deliver.
        3. Push the branch via :func:`push_subtask_branch`.
        4. Open a draft PR via :func:`open_subtask_pr`.
        5. Dispatch the ``pr-reviewer`` agent.
        6. Based on the reviewer verdict, ``mark_pr_ready`` (queue
           for merge) or ``close_pr`` (give up; keep the PR open
           for a human if review couldn't be completed).

        Runs synchronously across the group — issuing N gh API calls
        in parallel would multiply rate limits with no benefit.
        """
        repo = self._resolve_repo_slug()
        if not repo:
            logger.info(
                "gh_pr: no gh-tracked repo at %s — skipping PR flow",
                self._project_root or Path.cwd(),
            )
            return
        if self._worktree_manager is None:
            logger.info(
                "gh_pr: no WorktreeManager configured — skipping"
                " PR flow (per-task PRs require isolated worktrees)",
            )
            return

        # Ensure the pr-reviewer agent is running before we open
        # any PRs.  If we can't start it, fall back to leaving the
        # PRs in draft for human review rather than racing to open
        # them with no automated reviewer behind them.
        if not self._manager.is_running("pr-reviewer"):
            try:
                await self._manager.start_agent("pr-reviewer")
            except Exception as exc:
                logger.warning(
                    "gh_pr: cannot start pr-reviewer agent: %s —"
                    " PRs will be opened in draft and left for a"
                    " human reviewer",
                    exc,
                )

        for tid in matched_tids:
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None or sub_status.result is None:
                continue
            if not sub_status.result.success:
                continue
            worktree_path = wt_map.get(tid)
            if worktree_path is None:
                logger.debug(
                    "gh_pr: no worktree for %s, skipping",
                    tid,
                )
                continue

            # Resolve the auto-generated branch name from the
            # worktree manager — this is the SAME convention
            # used by ``WorktreeManager.make_worktree``.
            branch = f"{self._worktree_manager.BRANCH_PREFIX}/{state.workflow_id}/{tid}"

            await self._open_pr_for_subtask(
                state=state,
                tid=tid,
                sub_status=sub_status,
                worktree_path=worktree_path,
                branch=branch,
                repo=repo,
                spec=spec,
                history=history,
            )

    async def _open_pr_for_subtask(
        self,
        *,
        state: WorkflowState,
        tid: str,
        sub_status: SubTaskStatus,
        worktree_path: Path,
        branch: str,
        repo: str,
        spec: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Commit → push → open PR → dispatch reviewer for one task.

        Records progress in ``state.pr_states[tid]``.  Every failure
        path is logged and stored on the SubTaskPRState, never raised
        — a single bad PR shouldn't blow up a 10-task swarm.
        """
        pr_state = SubTaskPRState(task_id=tid, branch=branch)
        state.pr_states[tid] = pr_state

        # ── Step 1: commit ──────────────────────────────────
        data = sub_status.result.result or {} if sub_status.result else {}
        files_written = list(data.get("files_created", []))
        commit_res = commit_workflow_output(
            project_root=worktree_path,
            files_written=files_written,
            spec=sub_status.sub_task.description,
            sub_tasks=[],
            stats=None,
            push=False,
        )
        if not commit_res.committed and not commit_res.skipped:
            pr_state.error = f"commit failed: {commit_res.error or 'unknown'}"
            logger.warning(
                "gh_pr[%s]: commit failed: %s",
                tid,
                commit_res.error,
            )
            return
        # An empty / skipped commit is OK only if HEAD already
        # advanced past the base — the worktree may have been
        # committed externally.  We just continue to the push.

        # ── Step 2: push ────────────────────────────────────
        push_res = push_subtask_branch(worktree_path, branch)
        if not push_res.pushed:
            pr_state.error = f"push failed: {push_res.error or 'unknown'}"
            logger.warning(
                "gh_pr[%s]: push failed: %s",
                tid,
                push_res.error,
            )
            return

        # ── Step 3: open PR ─────────────────────────────────
        title = sub_status.sub_task.description.strip().split("\n")[0]
        if len(title) > 100:
            title = title[:97] + "..."
        body = self._build_pr_body(
            tid=tid,
            sub_status=sub_status,
            spec=spec,
            worktree_path=worktree_path,
        )
        open_res = open_subtask_pr(
            repo=repo,
            base=self._gh_pr_base,
            head=branch,
            title=title,
            body=body,
            draft=True,
        )
        if open_res.pr_number is None:
            pr_state.error = f"open PR failed: {open_res.error or 'unknown'}"
            logger.warning(
                "gh_pr[%s]: open PR failed: %s",
                tid,
                open_res.error,
            )
            return
        pr_state.pr_number = open_res.pr_number
        pr_state.url = open_res.url
        logger.info(
            "gh_pr[%s]: opened draft PR #%d (%s)",
            tid,
            open_res.pr_number,
            open_res.url or "?",
        )
        await self._emit_dashboard(
            "pr.opened",
            task_id=sub_status.task_id,
            agent_name=sub_status.agent_name,
            detail=f"PR #{open_res.pr_number}: {title[:60]}",
            workflow_id=state.workflow_id,
            project=state.project,
        )

        # ── Step 4: dispatch pr-reviewer ────────────────────
        verdict = await self._dispatch_pr_reviewer(
            state=state,
            tid=tid,
            pr_number=open_res.pr_number,
            spec=spec,
            repo=repo,
            history=history,
        )
        pr_state.verdict = verdict
        pr_state.approved = verdict == "approve"

        # ── Step 5: mark ready / close ──────────────────────
        if pr_state.approved:
            if mark_pr_ready(repo, open_res.pr_number):
                logger.info(
                    "gh_pr[%s]: PR #%d marked ready (approved)",
                    tid,
                    open_res.pr_number,
                )
            else:
                logger.warning(
                    "gh_pr[%s]: PR #%d approved but could not be marked ready",
                    tid,
                    open_res.pr_number,
                )
        elif verdict == "request-changes":
            # Leave the PR open in draft.  A human (or follow-up
            # coder dispatch) will iterate on the branch.
            comment_body = (
                f"pr-reviewer agent requested changes for sub-task"
                f" {tid}. Leaving PR in draft for follow-up.\n\n"
                "The orchestrator will NOT auto-merge this PR."
            )
            add_pr_review_comment(repo, open_res.pr_number, comment_body)
            logger.info(
                "gh_pr[%s]: PR #%d left in draft (changes-requested)",
                tid,
                open_res.pr_number,
            )
        else:
            # ``comment`` or error verdict — leave in draft, surface
            # the reason on the PR.
            add_pr_review_comment(
                repo,
                open_res.pr_number,
                f"pr-reviewer verdict: {verdict or 'unknown'}.\n"
                "Leaving PR in draft for human review.",
            )

    @staticmethod
    def _build_pr_body(
        *,
        tid: str,
        sub_status: SubTaskStatus,
        spec: str,
        worktree_path: Path,
    ) -> str:
        """Render the PR body for a sub-task.

        Includes the workflow spec, sub-task description, declared
        inputs/outputs, per-agent LLM stats (when available), and
        the worktree path so a human reviewer landing on the PR
        can reproduce locally.
        """
        st = sub_status.sub_task
        result_data = sub_status.result.result or {} if sub_status.result is not None else {}
        llm = result_data.get("llm") or {}
        files_created = result_data.get("files_created", [])
        ops = result_data.get("ops", [])

        lines: list[str] = [
            f"## Sub-task {tid}: {st.description}",
            "",
            "Opened automatically by `memfun swarm --gh-pr` (issue #38).",
            "",
            "### Workflow spec",
            "",
            "```",
            (spec or "(no spec provided)").strip(),
            "```",
            "",
            "### Sub-task",
            "",
            f"- **ID**: `{tid}`",
            f"- **Agent**: `{sub_status.agent_name}`",
            # Only the basename — never leak the absolute filesystem
            # path on the developer's machine into a public PR body.
            f"- **Branch**: `{worktree_path.name}`",
            f"- **Inputs**: {', '.join(st.inputs) if st.inputs else '(none)'}",
            f"- **Outputs**: {', '.join(st.outputs) if st.outputs else '(none)'}",
        ]

        if files_created:
            lines.extend(
                [
                    "",
                    "### Files touched",
                    "",
                    *(f"- `{p}`" for p in files_created[:50]),
                ]
            )

        if llm:
            calls = llm.get("calls")
            tokens_in = llm.get("tokens_in")
            tokens_out = llm.get("tokens_out")
            stats_bits: list[str] = []
            if isinstance(calls, int):
                stats_bits.append(f"{calls} calls")
            if isinstance(tokens_in, int):
                stats_bits.append(f"{tokens_in} in")
            if isinstance(tokens_out, int):
                stats_bits.append(f"{tokens_out} out")
            if stats_bits:
                lines.extend(
                    [
                        "",
                        "### LLM telemetry",
                        "",
                        "- " + " · ".join(stats_bits),
                    ]
                )

        if ops:
            lines.extend(
                [
                    "",
                    "### Operations",
                    "",
                    f"- {len(ops)} operation(s) recorded by the coder agent.",
                ]
            )

        return "\n".join(lines) + "\n"

    async def _dispatch_pr_reviewer(
        self,
        *,
        state: WorkflowState,
        tid: str,
        pr_number: int,
        spec: str,
        repo: str,
        history: list[dict[str, Any]] | None,
    ) -> str:
        """Dispatch the ``pr-reviewer`` agent against a single PR.

        Returns the verdict string (``approve`` / ``request-changes``
        / ``comment``).  Defaults to ``comment`` on parse failure —
        we never auto-merge an ambiguous review.
        """
        if not self._manager.is_running("pr-reviewer"):
            logger.warning(
                "gh_pr[%s]: pr-reviewer not running — defaulting to 'comment' verdict",
                tid,
            )
            return "comment"

        review_task = TaskMessage(
            task_id=f"{state.workflow_id}_{tid}_pr_review",
            agent_id="pr-reviewer",
            payload={
                "type": "ask",
                "query": (
                    f"Review PR #{pr_number} against the spec below.\n\n"
                    f"=== SPEC ===\n{spec}\n\n"
                    "=== INSTRUCTIONS ===\n"
                    "Follow the pr-reviewer AGENT.md protocol. Read"
                    f" the PR via `gh pr view {pr_number} --repo"
                    f" {repo}` and `gh pr diff {pr_number} --repo"
                    f" {repo}`. Post your review and emit the JSON"
                    " verdict block as your final answer."
                ),
                "context": state.shared_context or "",
                "workflow_id": state.workflow_id,
                "sub_task_id": tid,
                "pr_number": pr_number,
                "spec": spec,
                "base_repo": repo,
                "merge_authority": "approve-only",
                "conversation_history": history,
            },
            correlation_id=state.workflow_id,
        )

        try:
            result = await self._orchestrator.dispatch(
                review_task,
                "pr-reviewer",
                timeout=self._dispatch_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "gh_pr[%s]: pr-reviewer dispatch failed: %s",
                tid,
                exc,
            )
            return "comment"

        if not result.success:
            logger.warning(
                "gh_pr[%s]: pr-reviewer returned failure: %s",
                tid,
                result.error,
            )
            return "comment"

        data = result.result or {}
        answer = data.get("answer", "") or ""
        return self._parse_pr_verdict(answer)

    @staticmethod
    def _parse_pr_verdict(answer: str) -> str:
        """Extract the verdict from the pr-reviewer's final answer.

        The AGENT.md protocol asks the reviewer to emit a JSON block
        with a ``verdict`` field as its final output.  We parse the
        LAST valid JSON object containing a ``verdict`` key rather
        than substring-matching the whole text — a substring match
        can be fooled by a diff fragment containing the literal
        ``"verdict": "approve"`` string (e.g. when reviewing tests
        for this very function) and would lead to unsafe auto-approve.

        Returns one of ``approve`` / ``request-changes`` / ``comment``.
        Defaults to ``comment`` on parse failure — we never auto-merge
        an ambiguous review.
        """
        import json
        import re

        # Match flat JSON objects containing a verdict key.  Non-
        # recursive on purpose: the verdict block emitted by the
        # AGENT.md protocol has no nested ``verdict`` objects, so a
        # plain bracket-match is enough and avoids parsing the entire
        # answer with a recursive descent grammar.
        candidates = re.findall(
            r'\{[^{}]*"verdict"[^{}]*\}', answer,
        )
        for blob in reversed(candidates):
            try:
                obj = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                continue
            verdict = obj.get("verdict")
            if verdict in {"approve", "request-changes", "comment"}:
                return verdict

        # Loose fallback ONLY when no JSON block exists at all — so a
        # diff fragment cannot win against a real reviewer verdict.
        lower = answer.lower()
        if "lgtm" in lower:
            return "approve"
        if "changes-requested" in lower or "changes requested" in lower:
            return "request-changes"
        return "comment"

    async def _merge_approved_prs(
        self,
        *,
        state: WorkflowState,
        groups: list[list[str]],
    ) -> tuple[int, list[tuple[int, str]]]:
        """Merge approved PRs in dependency-group order.

        *groups* comes straight from
        ``DecompositionResult.parallelism_groups`` — group ``N`` is
        merged before group ``N+1`` so the base branch stays in
        sync with the workflow's dependency DAG.

        Within a single group merges run sequentially (one PR at a
        time) to avoid races where two PRs target ``main``
        simultaneously and the second one needs a rebase.

        Returns:
            A ``(merged_count, failed_list)`` tuple.  ``failed_list``
            is ``[(pr_number, error), ...]`` for every approved PR
            whose ``gh pr merge`` call did not succeed — typically a
            merge conflict introduced by the previously-merged PR.
            Caller MUST surface these so the workflow does not return
            ``success=True`` with broken merges left behind.
        """
        repo = self._resolve_repo_slug()
        if not repo:
            return 0, []

        # Build an ordered list of (group_idx, tid) honouring the
        # decomposer's groups.  Any PR not in groups (defensive) is
        # appended to the end.
        ordered: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for tid in group:
                if tid in state.pr_states and tid not in seen:
                    ordered.append(tid)
                    seen.add(tid)
        for tid in state.pr_states:
            if tid not in seen:
                ordered.append(tid)

        merged = 0
        failed: list[tuple[int, str]] = []
        for tid in ordered:
            pr_state = state.pr_states[tid]
            if not pr_state.approved or pr_state.pr_number is None:
                continue
            if pr_state.merged:
                continue
            merge_res = merge_pr(
                repo,
                pr_state.pr_number,
                self._gh_pr_merge_strategy,
            )
            pr_state.merged = merge_res.merged
            if not merge_res.merged:
                err = merge_res.error or "unknown"
                pr_state.error = f"merge failed: {err}"
                failed.append((pr_state.pr_number, err))
                logger.warning(
                    "gh_pr[%s]: merge PR #%d failed: %s",
                    tid,
                    pr_state.pr_number,
                    err,
                )
                await self._emit_dashboard(
                    "pr.merge_failed",
                    detail=f"PR #{pr_state.pr_number}: {err}",
                    workflow_id=state.workflow_id,
                    project=state.project,
                )
                continue
            merged += 1
            logger.info(
                "gh_pr[%s]: merged PR #%d (%s)",
                tid,
                pr_state.pr_number,
                self._gh_pr_merge_strategy,
            )
            await self._emit_dashboard(
                "pr.merged",
                detail=f"PR #{pr_state.pr_number} ({tid})",
                workflow_id=state.workflow_id,
                project=state.project,
            )
        return merged, failed

    # ── Agent management ──────────────────────────────────────

    async def _ensure_agents_running(
        self,
        sub_tasks: list[SubTask],
        state: WorkflowState | None = None,
    ) -> None:
        """Start needed specialist agents."""
        needed = {agent_name_for_type(st.agent_type) for st in sub_tasks}
        needed.add("review-agent")
        if self._gh_pr:
            needed.add("pr-reviewer")
        started = []
        failed = []

        for name in sorted(needed):
            if self._manager.is_running(name):
                started.append(name)
                continue
            try:
                await self._manager.start_agent(name)
                started.append(name)
                logger.info("Started agent %s", name)
                await self._emit_dashboard(
                    "worker.online",
                    agent_name=name,
                    workflow_id=state.workflow_id if state else None,
                    project=state.project if state else None,
                )
            except Exception as exc:
                failed.append(name)
                logger.error("Failed to start %s: %s", name, exc)

        logger.info(
            "Agent readiness: %d started (%s), %d failed (%s)",
            len(started),
            ", ".join(started),
            len(failed),
            ", ".join(failed) or "none",
        )

    # ── Review parsing ────────────────────────────────────────

    @staticmethod
    def _parse_review_result(result: TaskResult) -> ReviewResult:
        """Extract structured ReviewResult from review agent output."""
        if not result.success:
            return ReviewResult(
                approved=True,
                issues=[],
                summary=f"Review agent failed: {result.error}",
            )

        data = result.result or {}
        answer = data.get("answer", "")

        # Parse approved status.
        approved = True
        low = answer.lower()
        if "approved: false" in low or "approved:false" in low:
            approved = False

        # Parse issues: [T2] major: description
        issues: list[ReviewIssue] = []
        issue_pattern = re.compile(
            r"\[([Tt]\d+)\]\s*(critical|major|minor):\s*(.+)",
        )
        for match in issue_pattern.finditer(answer):
            tid = match.group(1).upper()
            severity = match.group(2).lower()
            desc = match.group(3).strip()
            issues.append(
                ReviewIssue(
                    task_id=tid,
                    description=desc,
                    detail=desc,
                    severity=severity,
                )
            )

        # Extract summary (first paragraph or first 200 chars).
        summary_lines = []
        for line in answer.split("\n"):
            line = line.strip()
            if not line:
                if summary_lines:
                    break
                continue
            summary_lines.append(line)
        summary = " ".join(summary_lines)[:300]

        return ReviewResult(approved=approved, issues=issues, summary=summary)

    # ── Result building ───────────────────────────────────────

    @staticmethod
    def _build_result(
        state: WorkflowState,
        review: ReviewResult,
        elapsed_ms: float,
    ) -> WorkflowResult:
        """Aggregate all sub-task results into WorkflowResult."""
        all_files: list[str] = []
        all_ops: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        sub_results: dict[str, TaskResult] = {}

        none_count = 0
        for tid, sub_status in state.sub_task_statuses.items():
            if sub_status.result is None:
                # Agent never ran or result was lost.
                none_count += 1
                logger.warning(
                    "Sub-task %s (%s) has no result — agent never ran or result was lost",
                    tid,
                    sub_status.agent_name,
                )
                answer_parts.append(
                    f"## {tid}: {sub_status.sub_task.description}\n"
                    f"*No result (agent: {sub_status.agent_name})*"
                )
                continue

            sub_results[tid] = sub_status.result
            data = sub_status.result.result or {}

            files = data.get("files_created", [])
            all_files.extend(files)

            ops = data.get("ops", [])
            all_ops.extend(ops)

            answer = data.get("answer", "")
            if answer:
                answer_parts.append(f"## {tid}: {sub_status.sub_task.description}\n{answer}")
            elif not sub_status.result.success:
                answer_parts.append(
                    f"## {tid}: {sub_status.sub_task.description}\n"
                    f"*Failed: {sub_status.result.error or 'unknown error'}*"
                )

        if none_count:
            logger.error(
                "%d/%d sub-tasks produced no result",
                none_count,
                len(state.sub_task_statuses),
            )

        merged_answer = "\n\n".join(answer_parts)
        if review.summary:
            merged_answer += f"\n\n## Review Summary\n{review.summary}"

        return WorkflowResult(
            workflow_id=state.workflow_id,
            success=True,
            answer=merged_answer,
            sub_task_results=sub_results,
            review_summary=review.summary,
            review_rounds=state.review_rounds,
            total_duration_ms=elapsed_ms,
            files_created=list(dict.fromkeys(all_files)),  # dedupe
            ops=all_ops,
            decomposition=state.decomposition,
        )

    # ── Auto-commit (issue #37) ──────────────────────────────

    def _run_auto_commit(
        self,
        *,
        state: WorkflowState,
        spec: str,
        files_written: list[str],
        wall_seconds: float,
    ) -> CommitResult | None:
        """Commit produced files to the project root after a successful run.

        Returns ``None`` when auto-commit is disabled at the engine
        level — the dashboard / callers treat that as "feature off".
        When auto-commit IS enabled, ALWAYS returns a CommitResult so
        the caller can render exactly what happened (committed / clean
        skip / hook failure).
        """
        if not self._auto_commit:
            return None
        if not files_written:
            # Workflow ran but produced no file artifacts — nothing
            # to commit.  Return a clean skip so the CLI can still
            # show "no files to commit" in the summary.
            return CommitResult(
                committed=False,
                skipped=True,
                error="no files to commit",
            )

        project_root = self._project_root or Path.cwd()

        sub_tasks = self._build_sub_task_reviews(state)
        stats = self._build_llm_stats(state, wall_seconds)

        return commit_workflow_output(
            project_root=project_root,
            files_written=files_written,
            spec=spec,
            sub_tasks=sub_tasks,
            stats=stats,
            push=self._auto_push,
        )

    @staticmethod
    def _build_sub_task_reviews(
        state: WorkflowState,
    ) -> list[SubTaskReview]:
        """Per-sub-task summaries for the commit message body."""
        out: list[SubTaskReview] = []
        for tid, sub_status in state.sub_task_statuses.items():
            data = (sub_status.result.result or {}) if sub_status.result else {}
            files = tuple(data.get("files_created", []))
            review = state.per_task_reviews.get(tid)
            approved = (
                review.approved
                if review is not None
                else bool(sub_status.result and sub_status.result.success)
            )
            out.append(
                SubTaskReview(
                    task_id=tid,
                    files=files,
                    agent_name=sub_status.agent_name,
                    approved=approved,
                )
            )
        return out

    @staticmethod
    def _build_llm_stats(
        state: WorkflowState,
        wall_seconds: float,
    ) -> dict[str, object]:
        """Aggregate LLM-call telemetry from per-sub-task results.

        Sub-task results may expose a ``llm`` dict with ``calls``,
        ``tokens_in`` and ``tokens_out`` (RLM-style agents do).
        Missing keys are silently treated as zero so callers don't
        need to special-case telemetry-off setups.
        """
        calls = 0
        tokens_in = 0
        tokens_out = 0
        for sub_status in state.sub_task_statuses.values():
            data = (sub_status.result.result or {}) if sub_status.result else {}
            llm = data.get("llm") if isinstance(data, dict) else None
            if not isinstance(llm, dict):
                continue
            c = llm.get("calls")
            if isinstance(c, int) and c >= 0:
                calls += c
            ti = llm.get("tokens_in")
            if isinstance(ti, int) and ti >= 0:
                tokens_in += ti
            to_ = llm.get("tokens_out")
            if isinstance(to_, int) and to_ >= 0:
                tokens_out += to_
        return {
            "calls": calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "wall_seconds": wall_seconds,
        }

    # ── Callback helpers ──────────────────────────────────────

    def _emit_status(self, state: WorkflowState) -> None:
        if self._on_workflow_status:
            with contextlib.suppress(Exception):
                self._on_workflow_status(state)

    def _emit_sub_task(self, tid: str, status: SubTaskStatus) -> None:
        if self._on_sub_task_status:
            with contextlib.suppress(Exception):
                self._on_sub_task_status(tid, status)

    async def _emit_dashboard(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        agent_name: str | None = None,
        success: bool | None = None,
        duration_ms: float | None = None,
        detail: str | None = None,
        workflow_id: str | None = None,
        project: str | None = None,
    ) -> None:
        """Best-effort publish event to dashboard stream."""
        try:
            from memfun_runtime.distributed import (
                EVENT_TOPIC,
                DistributedEvent,
            )

            event = DistributedEvent(
                event_type=event_type,
                task_id=task_id,
                agent_name=agent_name,
                worker_id=agent_name or "workflow-engine",
                success=success,
                duration_ms=duration_ms,
                detail=detail,
                project=project,
                workflow_id=workflow_id,
            )
            await self._context.event_bus.publish(
                EVENT_TOPIC,
                event.to_bytes(),
            )
            logger.debug("Dashboard event: %s", event_type)
        except Exception:
            logger.warning(
                "Failed to emit dashboard event %s",
                event_type,
                exc_info=True,
            )
