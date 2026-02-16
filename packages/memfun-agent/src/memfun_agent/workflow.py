"""WorkflowEngine: multi-agent task orchestration.

Manages the full lifecycle of a multi-agent workflow:
decompose -> fan-out parallel tasks -> collect -> review -> revise -> merge.

Uses the existing :class:`AgentOrchestrator` for dispatch/fan-out and
:class:`AgentManager` for agent lifecycle.
"""
from __future__ import annotations

import contextlib
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger
from memfun_core.types import TaskMessage, TaskResult

from memfun_agent.decomposer import DecompositionResult, SubTask, TaskDecomposer
from memfun_agent.shared_spec import SharedSpec, SharedSpecStore
from memfun_agent.specialists import agent_name_for_type

if TYPE_CHECKING:
    from memfun_runtime.context import RuntimeContext
    from memfun_runtime.lifecycle import AgentManager
    from memfun_runtime.orchestrator import AgentOrchestrator

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
    ) -> None:
        self._context = context
        self._orchestrator = orchestrator
        self._manager = manager
        self._decomposer = TaskDecomposer()
        self._spec_store = SharedSpecStore(context.state_store)
        self._on_workflow_status = on_workflow_status
        self._on_sub_task_status = on_sub_task_status

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
                state, task_description, project_context, conversation_history,
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
            task_description, project_context,
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
            project_context, decomposition,
        )

        # ── Phase 2: EXECUTE ──────────────────────────────────
        state.status = WorkflowStatus.RUNNING
        self._emit_status(state)

        # Final safety: ensure group IDs match sub_task_statuses.
        known_ids = set(state.sub_task_statuses.keys())
        execution_groups = decomposition.parallelism_groups
        all_group_ids = {
            tid for grp in execution_groups for tid in grp
        }
        if not all_group_ids.issubset(known_ids):
            logger.warning(
                "Parallelism group IDs %s don't match known"
                " sub-task IDs %s — rebuilding groups",
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
                base_context, state,
            )
            await self._execute_group(
                state, group, enriched_context, history,
            )

        # ── Phase 3: REVIEW ───────────────────────────────────
        state.status = WorkflowStatus.REVIEWING
        self._emit_status(state)

        review = await self._run_review(state, project_context, history)

        # ── Phase 4: REVISE (up to max rounds) ────────────────
        while (
            not review.approved
            and state.review_rounds < state.max_review_rounds
        ):
            state.status = WorkflowStatus.REVISING
            state.review_rounds += 1
            self._emit_status(state)
            logger.info(
                "Revision round %d (issues: %d)",
                state.review_rounds,
                len(review.issues),
            )

            await self._run_revision(
                state, review, state.shared_context or project_context, history,
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
        return self._build_result(state, review, elapsed)

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
            task_msg, "rlm-coder", timeout=600.0,
        )

        state.status = WorkflowStatus.COMPLETED
        state.completed_at = time.time()
        self._emit_status(state)

        elapsed = (time.perf_counter() - start) * 1000.0
        data = result.result or {}
        return WorkflowResult(
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

    async def _execute_group(
        self,
        state: WorkflowState,
        group: list[str],
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> None:
        """Execute a parallelism group via orchestrator.fan_out()."""
        tasks: list[TaskMessage] = []
        agent_names: list[str] = []
        matched_tids: list[str] = []

        for tid in group:
            sub_status = state.sub_task_statuses.get(tid)
            if sub_status is None:
                logger.warning(
                    "Group task ID %r not found in sub_task_statuses"
                    " (known: %s)",
                    tid,
                    list(state.sub_task_statuses.keys()),
                )
                continue

            task_msg = TaskMessage(
                task_id=sub_status.task_id,
                agent_id=sub_status.agent_name,
                payload={
                    "type": "ask",
                    "query": sub_status.sub_task.description,
                    "context": project_context,
                    "workflow_id": state.workflow_id,
                    "sub_task_id": tid,
                    "inputs": sub_status.sub_task.inputs,
                    "outputs": sub_status.sub_task.outputs,
                    "conversation_history": history,
                },
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
        results = await self._orchestrator.fan_out(
            tasks, agent_names, timeout=600.0,
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
                (sub_status.completed_at - sub_status.started_at)
                * 1000
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
                "Sub-task %s (%s): success=%s, answer_len=%d,"
                " ops=%d, files=%d, error=%s",
                tid,
                sub_status.agent_name,
                result.success,
                len(data.get("answer", "")),
                len(data.get("ops", [])),
                len(data.get("files_created", [])),
                result.error,
            )

    async def _run_review(
        self,
        state: WorkflowState,
        project_context: str,
        history: list[dict[str, Any]] | None,
    ) -> ReviewResult:
        """Run ReviewAgent on all completed outputs."""
        # Build review context from ALL sub-task results.
        review_parts: list[str] = []
        for tid, sub_status in state.sub_task_statuses.items():
            if sub_status.result is None:
                continue
            data = sub_status.result.result or {}
            answer = data.get("answer", "")[:2000]
            files = ", ".join(data.get("files_created", []))
            status_label = (
                "SUCCESS" if sub_status.result.success
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
            logger.warning(
                "No sub-task results found for review"
                " (all results are None)"
            )
            return ReviewResult(
                approved=True, issues=[],
                summary="No sub-task results available",
            )

        review_query = (
            "Review the outputs of the following sub-tasks for consistency, "
            "quality, and adherence to the shared specification. "
            "Check for file conflicts, import issues, and integration problems.\n\n"
            + "\n".join(review_parts)
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
            review_task, "review-agent", timeout=300.0,
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

            rev_task = TaskMessage(
                task_id=f"{state.workflow_id}_{tid}_rev{state.review_rounds}",
                agent_id=sub_status.agent_name,
                payload={
                    "type": "ask",
                    "query": revision_query,
                    "context": project_context,
                    "workflow_id": state.workflow_id,
                    "sub_task_id": tid,
                    "conversation_history": history,
                },
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
            revision_tasks, revision_agents, timeout=600.0,
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
            logger.info(
                "No source files found, using raw project context"
            )
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
        all_descriptions = "\n".join(
            f"- {st.description}"
            for st in decomposition.sub_tasks
        )
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
            logger.warning(
                "Context planner failed: %s, falling back", exc
            )
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
            section = (
                f"=== PRIOR AGENT RESULT: {tid}"
                f" ({ss.agent_name}) ===\n"
            )
            if answer:
                section += f"{answer[:3000]}\n"
            if files:
                section += (
                    f"Files created: {', '.join(files)}\n"
                )
            if ops:
                writes = [
                    o for o in ops
                    if isinstance(o, dict) and o.get("type") == "write"
                ]
                edits = [
                    o for o in ops
                    if isinstance(o, dict) and o.get("type") == "edit"
                ]
                if writes or edits:
                    section += (
                        f"Operations: {len(writes)} writes,"
                        f" {len(edits)} edits\n"
                    )
            parts.append(section)

        if not parts:
            return base_context

        prior_block = (
            "\n\n=== COMPLETED AGENT RESULTS ===\n"
            + "\n".join(parts)
        )
        return base_context + prior_block

    # ── Agent management ──────────────────────────────────────

    async def _ensure_agents_running(
        self,
        sub_tasks: list[SubTask],
        state: WorkflowState | None = None,
    ) -> None:
        """Start needed specialist agents."""
        needed = {agent_name_for_type(st.agent_type) for st in sub_tasks}
        needed.add("review-agent")
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
                    "worker.online", agent_name=name,
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
            issues.append(ReviewIssue(
                task_id=tid,
                description=desc,
                detail=desc,
                severity=severity,
            ))

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
                    "Sub-task %s (%s) has no result — agent"
                    " never ran or result was lost",
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
                answer_parts.append(
                    f"## {tid}: {sub_status.sub_task.description}\n{answer}"
                )
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
                EVENT_TOPIC, event.to_bytes(),
            )
            logger.debug("Dashboard event: %s", event_type)
        except Exception:
            logger.warning(
                "Failed to emit dashboard event %s", event_type,
                exc_info=True,
            )
