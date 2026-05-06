"""TaskDecomposer: split complex tasks into a DAG of sub-tasks.

Uses the ``TaskDecomposition`` DSPy signature to produce structured
sub-tasks with dependency information and shared specifications.
Falls back to a single-task result when decomposition fails or
the task is too simple to decompose.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import dspy
from memfun_core.logging import get_logger

from memfun_agent.signatures import TaskDecomposition

logger = get_logger("agent.decomposer")


# ── Errors ────────────────────────────────────────────────────


class DecompositionError(ValueError):
    """Raised when a decomposition is structurally invalid.

    Used for hard-fail conditions such as overlapping outputs in the same
    parallelism group when ``overlap_strategy='reject'`` is configured.
    """


# ── Data types ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SubTask:
    """A single sub-task in a decomposed workflow."""

    id: str
    description: str
    agent_type: str  # file/coder/test/review/web_search/web_fetch/planner/debug/security
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    max_iterations: int = 10


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Result of task decomposition."""

    sub_tasks: list[SubTask]
    shared_spec: str
    parallelism_groups: list[list[str]]
    is_single_task: bool


@dataclass(frozen=True, slots=True)
class DecompositionConfig:
    """Configuration knobs for :class:`TaskDecomposer`.

    Parameters
    ----------
    overlap_strategy:
        How to handle the case where two parallel-group sub-tasks declare
        the same output path.  ``"sequence"`` (default) splits the
        overlapping tasks into separate parallelism groups so they run
        sequentially.  ``"reject"`` raises a :class:`DecompositionError`.
    """

    overlap_strategy: Literal["reject", "sequence"] = "sequence"


# ── Helpers ───────────────────────────────────────────────────

_VALID_AGENT_TYPES = frozenset({
    "file", "coder", "test", "review",
    "web_search", "web_fetch", "planner", "debug", "security",
})


def _normalize_list(value: Any) -> list[Any]:
    """Coerce an LLM output to a Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        # Try newline-separated
        if "\n" in text:
            return [line.strip() for line in text.split("\n") if line.strip()]
        if text:
            return [text]
    return []


def _parse_sub_tasks(raw: Any) -> list[SubTask]:
    """Defensively parse sub-tasks from LLM output."""
    items = _normalize_list(raw)
    tasks: list[SubTask] = []
    for item in items:
        if isinstance(item, str):
            # Try parsing as JSON object
            try:
                item = json.loads(item)
            except (json.JSONDecodeError, ValueError):
                # Treat as plain description
                tid = f"T{len(tasks) + 1}"
                tasks.append(SubTask(id=tid, description=item, agent_type="coder"))
                continue

        if isinstance(item, dict):
            tid = str(item.get("id", f"T{len(tasks) + 1}"))
            desc = str(item.get("description", ""))
            agent_type = str(item.get("agent_type", "coder"))
            if agent_type not in _VALID_AGENT_TYPES:
                agent_type = "coder"
            inputs = [str(x) for x in (item.get("inputs") or [])]
            outputs = [str(x) for x in (item.get("outputs") or [])]
            depends = [str(x) for x in (item.get("depends_on") or [])]
            max_iter = int(item.get("max_iterations", 10))
            max_iter = max(3, min(max_iter, 25))
            tasks.append(SubTask(
                id=tid,
                description=desc,
                agent_type=agent_type,
                inputs=inputs,
                outputs=outputs,
                depends_on=depends,
                max_iterations=max_iter,
            ))

    return tasks


def _parse_groups(raw: Any) -> list[list[str]]:
    """Defensively parse parallelism groups from LLM output.

    Handles various LLM output formats:
    - Proper nested lists: ``[["T1", "T2"], ["T3"]]``
    - JSON strings: ``'["T1", "T2"]'``
    - Python list repr with single quotes: ``"['T1', 'T2']"``
    - Comma-separated: ``"T1, T2"``
    """
    items = _normalize_list(raw)
    groups: list[list[str]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            # Try JSON first (double quotes).
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    groups.append([str(x) for x in parsed])
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Try Python list repr (single quotes) by
            # replacing ' with ".
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text.replace("'", '"'))
                    if isinstance(parsed, list):
                        groups.append([str(x) for x in parsed])
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            # Try comma-separated
            ids = [x.strip() for x in re.split(r"[,\s]+", text) if x.strip()]
            if ids:
                groups.append(ids)
        elif isinstance(item, list):
            groups.append([str(x) for x in item])
    return groups


def _validate_dag(tasks: list[SubTask]) -> None:
    """Validate no cycles exist in the dependency graph (topological sort)."""
    task_ids = {t.id for t in tasks}
    # Build adjacency for in-degree counting.
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    children: dict[str, list[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep in t.depends_on:
            if dep not in task_ids:
                continue  # ignore unknown deps
            in_degree[t.id] += 1
            children[dep].append(t.id)

    queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited < len(tasks):
        raise ValueError(
            f"Cycle detected in task dependency graph "
            f"(visited {visited}/{len(tasks)} tasks)"
        )


def _infer_groups(tasks: list[SubTask]) -> list[list[str]]:
    """Build parallelism groups from dependency information."""
    task_ids = {t.id for t in tasks}
    deps_map = {t.id: [d for d in t.depends_on if d in task_ids] for t in tasks}
    assigned: set[str] = set()
    groups: list[list[str]] = []

    remaining = set(task_ids)
    while remaining:
        # Tasks whose deps are all assigned.
        ready = [tid for tid in remaining if all(d in assigned for d in deps_map[tid])]
        if not ready:
            # Safety: break infinite loop if deps can't resolve.
            ready = sorted(remaining)
        groups.append(sorted(ready))
        for tid in ready:
            assigned.add(tid)
            remaining.discard(tid)

    return groups


# ── Output overlap detection ──────────────────────────────────


_PATH_TOKEN_RE = re.compile(
    r"(?<![\w/.-])(?:\.{0,2}/)?[\w.-]+(?:/[\w.-]+)+(?:\.[A-Za-z0-9]+)?",
)
"""Heuristic path matcher.

Matches tokens like ``src/foo/bar.py``, ``./pkg/module.ts``,
``../shared/util.go``.  Avoids matching bare words ("models") and
domain-style strings ("example.com") by requiring at least one ``/``.
"""


def _normalize_path(path: str) -> str:
    """Normalise an output path for overlap comparison.

    Strips whitespace and a leading ``./``.  Collapses repeated slashes.
    Comparisons are case-sensitive (POSIX semantics) — paths that differ
    only by case are treated as distinct.
    """
    cleaned = path.strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned


def _heuristic_paths_from_text(text: str) -> list[str]:
    """Extract probable file path tokens from free-form spec text.

    Used as a fallback when the LLM emits an empty ``outputs`` list for a
    code-modification sub-task.  Conservative — only returns tokens that
    contain at least one ``/`` to avoid matching bare words.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _PATH_TOKEN_RE.findall(text):
        norm = _normalize_path(match)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def detect_output_overlap(
    sub_tasks: list[SubTask],
    parallelism_groups: list[list[str]],
) -> dict[str, list[str]]:
    """Detect declared output collisions inside a parallelism group.

    Returns a mapping ``{output_path: [task_ids that write to it]}`` for
    every output path declared by 2+ tasks within the *same* parallelism
    group.  Tasks in different groups are sequential by construction —
    they cannot conflict, so they are excluded.

    The output paths are normalised (``./`` stripped, repeated slashes
    collapsed) but otherwise treated as opaque strings — no globbing,
    no case folding.

    Parameters
    ----------
    sub_tasks:
        All sub-tasks produced by the decomposer.  Tasks not referenced
        by any group are ignored.
    parallelism_groups:
        Each inner list is a set of task IDs that would run in parallel.

    Returns
    -------
    dict[str, list[str]]
        Empty when no overlap exists.  Otherwise ``output_path`` ->
        sorted list of conflicting ``task_id`` values.  Determinism is
        preserved so the result can be logged or asserted on directly.
    """
    by_id: dict[str, SubTask] = {t.id: t for t in sub_tasks}
    conflicts: dict[str, set[str]] = {}

    for group in parallelism_groups:
        if len(group) < 2:
            continue
        # Build path -> writers for this group only.
        writers: dict[str, list[str]] = {}
        for tid in group:
            task = by_id.get(tid)
            if task is None:
                continue
            for raw_path in task.outputs:
                norm = _normalize_path(raw_path)
                if not norm:
                    continue
                writers.setdefault(norm, []).append(tid)

        for path, ids in writers.items():
            unique_ids = set(ids)
            if len(unique_ids) >= 2:
                conflicts.setdefault(path, set()).update(unique_ids)

    return {path: sorted(ids) for path, ids in conflicts.items()}


def resolve_output_overlap(
    sub_tasks: list[SubTask],
    parallelism_groups: list[list[str]],
    strategy: Literal["reject", "sequence"] = "sequence",
) -> tuple[list[list[str]], list[str]]:
    """Resolve output overlaps by re-shaping the parallelism groups.

    With ``strategy="sequence"``: any task that conflicts with another
    task in the same group is moved out into its own newly-created
    sequential group, chained immediately after the original group.
    Order within the original group is preserved by ``task_id`` (sorted)
    for reproducibility.

    With ``strategy="reject"``: raises :class:`DecompositionError`
    listing each conflicting output path and the task IDs that declared
    it.

    Returns ``(new_parallelism_groups, log_messages)``.  When there is
    no overlap, returns the input groups unchanged with an empty log.
    """
    overlaps = detect_output_overlap(sub_tasks, parallelism_groups)
    if not overlaps:
        return parallelism_groups, []

    if strategy == "reject":
        details = "; ".join(
            f"{path}: {', '.join(ids)}" for path, ids in sorted(overlaps.items())
        )
        raise DecompositionError(
            f"Decomposition has overlapping outputs in parallel groups "
            f"({len(overlaps)} conflict(s)): {details}"
        )

    # strategy == "sequence"
    # Set of task IDs that participate in any conflict.
    conflicting_ids: set[str] = set()
    for ids in overlaps.values():
        conflicting_ids.update(ids)

    new_groups: list[list[str]] = []
    log_messages: list[str] = []

    for group in parallelism_groups:
        # Tasks in this group that are not part of any conflict stay
        # together as the (possibly reduced) parallel group.
        group_conflicts = [tid for tid in group if tid in conflicting_ids]
        group_safe = [tid for tid in group if tid not in conflicting_ids]

        if not group_conflicts:
            new_groups.append(list(group))
            continue

        # Append the safe tasks (if any) as one shrunken parallel group.
        if group_safe:
            new_groups.append(group_safe)

        # Sort conflicting IDs deterministically and chain each in its
        # own group so the writes are serialised.
        for tid in sorted(group_conflicts):
            new_groups.append([tid])

        log_messages.append(
            f"sequenced {len(group_conflicts)} overlapping task(s) "
            f"out of group {group}: {sorted(group_conflicts)}"
        )

    return new_groups, log_messages


def _backfill_outputs_with_heuristic(tasks: list[SubTask]) -> list[SubTask]:
    """Populate empty ``outputs`` from the description for code tasks.

    Many decomposition LLMs omit ``outputs`` even for code-modification
    tasks.  Without it, :func:`detect_output_overlap` cannot do its job.
    This best-effort fallback parses the sub-task description for
    path-like tokens and uses them only when ``outputs`` is empty *and*
    the task's agent type is one that writes files.
    """
    writer_agents = {"coder", "test", "debug"}
    out: list[SubTask] = []
    for task in tasks:
        if task.outputs or task.agent_type not in writer_agents:
            out.append(task)
            continue
        guessed = _heuristic_paths_from_text(task.description)
        if not guessed:
            out.append(task)
            continue
        logger.debug(
            "Backfilled outputs for %s (%s) from description: %s",
            task.id, task.agent_type, guessed,
        )
        out.append(SubTask(
            id=task.id,
            description=task.description,
            agent_type=task.agent_type,
            inputs=task.inputs,
            outputs=guessed,
            depends_on=task.depends_on,
            max_iterations=task.max_iterations,
        ))
    return out


# ── Main module ───────────────────────────────────────────────


class TaskDecomposer(dspy.Module):
    """Decomposes complex tasks into a DAG of sub-tasks."""

    def __init__(self, config: DecompositionConfig | None = None) -> None:
        super().__init__()
        self.decomposer = dspy.Predict(TaskDecomposition)
        self.config = config or DecompositionConfig()

    async def adecompose(
        self,
        task_description: str,
        project_context: str,
    ) -> DecompositionResult:
        """Decompose *task_description* into sub-tasks.

        Returns a :class:`DecompositionResult` with sub-tasks,
        shared spec, and parallelism groups.  Falls back to a
        single-task result on failure.
        """
        try:
            result = await asyncio.to_thread(
                self.decomposer,
                task_description=task_description,
                project_context=project_context[:4000],
            )

            raw_tasks = getattr(result, "sub_tasks", [])
            sub_tasks = _parse_sub_tasks(raw_tasks)

            shared_spec = str(getattr(result, "shared_spec", ""))

            raw_groups = getattr(result, "parallelism_groups", [])
            groups = _parse_groups(raw_groups)

            # If LLM returned nothing, fall back to a single coder task.
            if not sub_tasks:
                return self._single_fallback(task_description)

            _validate_dag(sub_tasks)

            # Validate group IDs match actual sub_task IDs.
            # LLMs frequently produce mismatched IDs (e.g. "1"
            # instead of "T1"), causing _execute_group to skip
            # all tasks silently.
            known_ids = {t.id for t in sub_tasks}
            if groups:
                all_group_ids = {
                    tid for grp in groups for tid in grp
                }
                if not all_group_ids.issubset(known_ids):
                    logger.warning(
                        "Group IDs %s don't match sub_task IDs %s"
                        " — rebuilding groups from dependencies",
                        all_group_ids - known_ids,
                        known_ids,
                    )
                    groups = []  # force rebuild below

            # If LLM didn't produce groups (or they were
            # invalid), infer from deps.
            if not groups:
                groups = _infer_groups(sub_tasks)

            # Backfill empty `outputs` for code-writing tasks via a
            # heuristic on the description.  See Experiment A
            # (docs/LEARNINGS.md): even strong models silently emit
            # empty outputs, defeating the overlap detector.
            sub_tasks = _backfill_outputs_with_heuristic(sub_tasks)

            # Detect and resolve declared output overlaps inside any
            # parallelism group.  See issue #9.
            overlaps = detect_output_overlap(sub_tasks, groups)
            if overlaps:
                groups, messages = resolve_output_overlap(
                    sub_tasks,
                    groups,
                    strategy=self.config.overlap_strategy,
                )
                logger.info(
                    "resolved %d output overlap(s) in decomposition by %s: %s",
                    len(overlaps),
                    self.config.overlap_strategy,
                    "; ".join(messages) if messages else "(none)",
                )

            return DecompositionResult(
                sub_tasks=sub_tasks,
                shared_spec=shared_spec,
                parallelism_groups=groups,
                is_single_task=len(sub_tasks) <= 1,
            )

        except DecompositionError:
            # `reject` strategy intentionally surfaces the conflict.
            raise
        except Exception as exc:
            logger.warning(
                "Task decomposition failed: %s — using single-task fallback",
                exc,
            )
            return self._single_fallback(task_description)

    @staticmethod
    def _single_fallback(description: str) -> DecompositionResult:
        sub_tasks = [
            SubTask(
                id="T1",
                description=description,
                agent_type="coder",
                max_iterations=10,
            ),
        ]
        return DecompositionResult(
            sub_tasks=sub_tasks,
            shared_spec="",
            parallelism_groups=[["T1"]],
            is_single_task=True,
        )
