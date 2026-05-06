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
    enable_hierarchical:
        When ``True`` (default), the decomposer recursively re-invokes
        itself on each sub-task whose heuristic complexity exceeds
        ``recurse_threshold`` (and depth is below ``max_depth``).  When
        ``False``, the decomposer behaves like the pre-issue-#11
        single-pass version — useful for backward compatibility tests
        and to disable LLM-fan-out cost amplification.
    max_depth:
        Hard cap on recursion levels.  Recursion is gated by
        ``current_depth < max_depth``, so:

        * ``max_depth=0`` — never recurse (single-pass behavior).
        * ``max_depth=1`` — top-level may recurse once into children;
          children stay leaves regardless of complexity.
        * ``max_depth=3`` (default) — three levels of recursion below
          the top, fan-out up to ``branch_factor**4`` leaves in the
          worst case.

        This is the **non-negotiable** termination invariant — even
        pathological LLM output stops here, enforced by an ``assert``
        at the top of :meth:`TaskDecomposer.adecompose`.
    branch_factor:
        Soft target for the number of sub-tasks per level.  Used by
        downstream coordinators (orchestrator AGENT.md) as the parallel-
        fan-out hint.  The decomposer itself does not enforce this — it
        accepts whatever the LLM emits — but ``branch_factor`` is the
        documented "do not exceed" number for any single layer.
    recurse_threshold:
        Numeric heuristic threshold above which a sub-task is
        considered too coarse-grained and is recursively re-decomposed.
        See :func:`_estimate_complexity` for the exact heuristic
        (counts of ``inputs``+``outputs``, description length, file-
        path mentions).  Default ``12`` was chosen so a sub-task that
        touches 6+ files with a long description recurses, while a
        small "edit one file" task does not.
    """

    overlap_strategy: Literal["reject", "sequence"] = "sequence"
    enable_hierarchical: bool = True
    max_depth: int = 3
    branch_factor: int = 8
    recurse_threshold: int = 12


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


# ── Hierarchical complexity heuristic ─────────────────────────


def _estimate_complexity(task: SubTask | Any) -> int:
    """Heuristically score how "big" a sub-task is.

    Used by :class:`TaskDecomposer` to decide whether a sub-task is
    coarse-grained enough to warrant recursive decomposition.  The
    score combines four cheap signals:

    1. ``len(inputs)`` — read-set cardinality.
    2. ``len(outputs)`` — write-set cardinality (weighted 2x because
       writes drive overlap conflicts and parallelism is gated by
       output disjointness).
    3. Number of distinct path-like tokens in the description (caught
       by :func:`_heuristic_paths_from_text`).
    4. ``len(description) // 80`` — long descriptions are usually
       compound asks ("do X, then Y, also Z").

    If the parsed ``SubTask`` carries an explicit numeric ``complexity``
    attribute (future-proofing — the dataclass currently doesn't
    declare one, but raw LLM dicts may), it is preferred over the
    heuristic.

    Returns a non-negative integer.  Compared against
    ``DecompositionConfig.recurse_threshold``.
    """
    explicit = getattr(task, "complexity", None)
    if isinstance(explicit, (int, float)) and explicit >= 0:
        return int(explicit)

    inputs = getattr(task, "inputs", []) or []
    outputs = getattr(task, "outputs", []) or []
    description = str(getattr(task, "description", "") or "")

    paths_in_desc = _heuristic_paths_from_text(description)
    desc_chunks = len(description) // 80

    return len(inputs) + 2 * len(outputs) + len(paths_in_desc) + desc_chunks


def _renumber_children(parent_id: str, children: list[SubTask]) -> list[SubTask]:
    """Rename children of *parent_id* to ``<parent_id>.<i>`` to avoid ID clashes.

    When the recursive call returns sub-tasks named ``T1, T2, ...`` they
    can collide with siblings of the parent.  Re-namespacing under the
    parent's ID keeps the global ID space unique without forcing the
    LLM to know what other tasks exist.

    The original ``depends_on`` references between children are rewritten
    to use the new namespaced IDs so the DAG stays consistent.
    """
    if not children:
        return []
    old_to_new = {child.id: f"{parent_id}.{i + 1}" for i, child in enumerate(children)}
    out: list[SubTask] = []
    for child in children:
        new_deps = [old_to_new.get(d, d) for d in child.depends_on]
        out.append(SubTask(
            id=old_to_new[child.id],
            description=child.description,
            agent_type=child.agent_type,
            inputs=child.inputs,
            outputs=child.outputs,
            depends_on=new_deps,
            max_iterations=child.max_iterations,
        ))
    return out


def _splice_children_into_groups(
    groups: list[list[str]],
    parent_id: str,
    child_ids: list[str],
    child_groups: list[list[str]],
) -> list[list[str]]:
    """Replace ``parent_id`` in *groups* with the flattened ``child_groups``.

    The simple model: parent's slot becomes a sequence of groups, where
    each child group runs in parallel internally and the child groups
    run sequentially relative to each other.  Other tasks in the parent's
    original group remain parallel to the *first* child group only —
    after that the children are sequential, so the rest of the parent's
    group siblings stay parallel only with the first child wave.

    To keep semantics straightforward and conservative, we adopt the
    following policy:

    * Sibling tasks that shared the parent's group remain in that
      group (unchanged) — except the parent itself is replaced by the
      first wave of children.
    * Subsequent waves of children are appended as their own groups
      immediately after, sequenced before any later groups in the
      original plan.

    If ``child_groups`` is empty (LLM returned nothing usable for the
    recursion), we leave the parent in place — caller should not call
    this when child_groups is empty, but we defend.
    """
    if not child_ids or not child_groups:
        return groups

    # Validate that every child id appears in some child_group.
    seen_in_groups: set[str] = set()
    for grp in child_groups:
        seen_in_groups.update(grp)
    missing = [cid for cid in child_ids if cid not in seen_in_groups]
    if missing:
        # Fall back: just put the missing ids in a final group so they run.
        child_groups = [*child_groups, missing]

    new_groups: list[list[str]] = []
    for grp in groups:
        if parent_id not in grp:
            new_groups.append(list(grp))
            continue
        # Replace parent_id with the first wave of children, preserve siblings.
        first_wave = list(child_groups[0])
        rest = grp[:]
        rest.remove(parent_id)
        merged = list(dict.fromkeys(rest + first_wave))  # de-dup, preserve order
        new_groups.append(merged)
        # Append any remaining waves immediately after.
        for wave in child_groups[1:]:
            new_groups.append(list(wave))
    return new_groups


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
        _depth: int = 0,
    ) -> DecompositionResult:
        """Decompose *task_description* into sub-tasks.

        Returns a :class:`DecompositionResult` with sub-tasks,
        shared spec, and parallelism groups.  Falls back to a
        single-task result on failure.

        Hierarchical mode (``DecompositionConfig.enable_hierarchical=True``,
        default) walks the result and recursively re-decomposes any
        sub-task whose heuristic complexity exceeds
        ``recurse_threshold``, up to ``max_depth`` levels.  Children are
        re-namespaced under their parent's ID so the global ID space
        stays unique.

        Termination: ``_depth`` is bounded by ``max_depth`` and asserted
        — even pathological LLM output cannot recurse forever.

        Parameters
        ----------
        task_description:
            Resolved task description.
        project_context:
            Truncated to 4000 chars before sending to the LLM.
        _depth:
            Current recursion depth.  Caller passes 0; recursive calls
            pass ``_depth + 1``.  Capped at ``self.config.max_depth``.
        """
        # Hard termination invariant — assert before *any* LLM call.
        assert _depth <= self.config.max_depth, (
            f"hierarchical decomposer recursed past max_depth="
            f"{self.config.max_depth} (got {_depth})"
        )

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

            # ── Hierarchical recursion (issue #11) ────────────────
            # If enabled, walk each sub-task and recursively decompose
            # the coarse-grained ones.  This must happen BEFORE the
            # overlap detector so the (now-finer) leaves are what the
            # detector sees.
            if (
                self.config.enable_hierarchical
                and _depth < self.config.max_depth
            ):
                sub_tasks, groups = await self._recurse_into_subtasks(
                    sub_tasks,
                    groups,
                    project_context,
                    _depth=_depth,
                )

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
        except AssertionError:
            # Termination invariant — re-raise so the test catches it.
            raise
        except Exception as exc:
            logger.warning(
                "Task decomposition failed: %s — using single-task fallback",
                exc,
            )
            return self._single_fallback(task_description)

    async def _recurse_into_subtasks(
        self,
        sub_tasks: list[SubTask],
        groups: list[list[str]],
        project_context: str,
        _depth: int,
    ) -> tuple[list[SubTask], list[list[str]]]:
        """For each sub-task whose complexity exceeds the threshold,
        recursively re-decompose it and splice the children in.

        Cheap path: most sub-tasks score below the threshold and are
        emitted as leaves — only the LLM call that spawned them is
        billed.  Only coarse-grained sub-tasks pay an extra LLM call.

        Termination is guaranteed by ``_depth < self.config.max_depth``
        gating the recursive call.
        """
        threshold = self.config.recurse_threshold
        new_tasks: list[SubTask] = []
        new_groups = [list(g) for g in groups]

        for task in sub_tasks:
            score = _estimate_complexity(task)
            if score <= threshold:
                new_tasks.append(task)
                continue

            # Recurse — note ``_depth + 1`` and the assert at the top of
            # adecompose enforce termination.
            try:
                child_result = await self.adecompose(
                    task.description,
                    project_context,
                    _depth=_depth + 1,
                )
            except Exception as exc:
                logger.warning(
                    "Recursive decomposition of %s failed (%s) — "
                    "keeping as leaf",
                    task.id,
                    exc,
                )
                new_tasks.append(task)
                continue

            if not child_result.sub_tasks or child_result.is_single_task:
                # Either nothing came back or the recursive call gave
                # us a single-task fallback — splicing that in adds no
                # value, so keep the original.
                new_tasks.append(task)
                continue

            children = _renumber_children(task.id, child_result.sub_tasks)
            child_id_map = {
                old.id: new.id
                for old, new in zip(child_result.sub_tasks, children, strict=True)
            }
            child_groups = [
                [child_id_map[cid] for cid in grp if cid in child_id_map]
                for grp in child_result.parallelism_groups
            ]
            child_groups = [g for g in child_groups if g]
            child_ids = [c.id for c in children]

            # Anything that depended on the parent now depends on every
            # leaf of the parent's expansion (conservative — guarantees
            # the dependency invariant).
            new_tasks_after_rewrite: list[SubTask] = []
            for existing in new_tasks:
                if task.id in existing.depends_on:
                    rewritten_deps = [
                        d for d in existing.depends_on if d != task.id
                    ] + child_ids
                    new_tasks_after_rewrite.append(SubTask(
                        id=existing.id,
                        description=existing.description,
                        agent_type=existing.agent_type,
                        inputs=existing.inputs,
                        outputs=existing.outputs,
                        depends_on=rewritten_deps,
                        max_iterations=existing.max_iterations,
                    ))
                else:
                    new_tasks_after_rewrite.append(existing)
            new_tasks = new_tasks_after_rewrite

            new_tasks.extend(children)
            new_groups = _splice_children_into_groups(
                new_groups, task.id, child_ids, child_groups,
            )

        # Same dependency rewrite for siblings later in the list — any
        # task in the original ``sub_tasks`` list past this point that
        # depends on a now-expanded parent must be patched too.  The
        # loop above only patched tasks already added to ``new_tasks``
        # at the time the parent was expanded.  Run a second pass.
        expanded_ids = {t.id for t in sub_tasks} - {t.id for t in new_tasks}
        if expanded_ids:
            children_per_parent: dict[str, list[str]] = {}
            for t in new_tasks:
                # Children of ``parent_id`` are renamed to
                # ``parent_id.<i>`` — recover the parent prefix.
                if "." in t.id:
                    parent = t.id.rsplit(".", 1)[0]
                    children_per_parent.setdefault(parent, []).append(t.id)

            patched: list[SubTask] = []
            for t in new_tasks:
                new_deps: list[str] = []
                changed = False
                for d in t.depends_on:
                    if d in expanded_ids and d in children_per_parent:
                        new_deps.extend(children_per_parent[d])
                        changed = True
                    else:
                        new_deps.append(d)
                if changed:
                    patched.append(SubTask(
                        id=t.id,
                        description=t.description,
                        agent_type=t.agent_type,
                        inputs=t.inputs,
                        outputs=t.outputs,
                        depends_on=new_deps,
                        max_iterations=t.max_iterations,
                    ))
                else:
                    patched.append(t)
            new_tasks = patched

        # Drop any group entries that no longer correspond to a known task
        # (defensive — shouldn't happen, but keeps the result internally
        # consistent).
        valid_ids = {t.id for t in new_tasks}
        cleaned_groups: list[list[str]] = []
        for grp in new_groups:
            kept = [tid for tid in grp if tid in valid_ids]
            if kept:
                cleaned_groups.append(kept)

        return new_tasks, cleaned_groups

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
