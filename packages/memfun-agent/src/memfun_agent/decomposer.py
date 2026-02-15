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
from typing import Any

import dspy
from memfun_core.logging import get_logger

from memfun_agent.signatures import TaskDecomposition

logger = get_logger("agent.decomposer")


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
    """Defensively parse parallelism groups from LLM output."""
    items = _normalize_list(raw)
    groups: list[list[str]] = []
    for item in items:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, list):
                    groups.append([str(x) for x in parsed])
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Try comma-separated
            ids = [x.strip() for x in re.split(r"[,\s]+", item) if x.strip()]
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


# ── Main module ───────────────────────────────────────────────


class TaskDecomposer(dspy.Module):
    """Decomposes complex tasks into a DAG of sub-tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.decomposer = dspy.Predict(TaskDecomposition)

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

            # If LLM didn't produce groups, infer from deps.
            if not groups:
                groups = _infer_groups(sub_tasks)

            return DecompositionResult(
                sub_tasks=sub_tasks,
                shared_spec=shared_spec,
                parallelism_groups=groups,
                is_single_task=len(sub_tasks) <= 1,
            )

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
