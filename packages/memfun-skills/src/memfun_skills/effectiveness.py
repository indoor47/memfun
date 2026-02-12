"""Skill effectiveness tracking: metrics collection and analysis.

Tracks execution outcomes, success rates, durations, and user ratings
for each skill to identify underperforming skills and opportunities
for optimization or synthesis.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from memfun_runtime.protocols.state_store import StateStoreAdapter

logger = get_logger("skills.effectiveness")

# Key prefix for skill stats storage
_STATS_PREFIX = "memfun:skill_stats:"

# Safety limits
_MAX_RECORDS_PER_SKILL = 10_000
_MAX_SKILLS_IN_MEMORY = 1_000
_MAX_SKILL_NAME_LENGTH = 128
_SAFE_SKILL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _validate_skill_name(name: str) -> None:
    """Validate a skill name to prevent key injection in the state store.

    Raises:
        ValueError: If the name is invalid.
    """
    if not name or len(name) > _MAX_SKILL_NAME_LENGTH:
        msg = f"Invalid skill name length: {len(name) if name else 0}"
        raise ValueError(msg)
    if not _SAFE_SKILL_NAME_RE.match(name):
        msg = f"Skill name contains invalid characters: {name!r}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SkillStats:
    """Aggregated statistics for a skill's execution history."""

    skill_name: str
    total_executions: int
    successful_executions: int
    success_rate: float
    avg_duration_ms: float
    avg_user_rating: float | None
    last_execution: float  # timestamp
    error_count: int
    common_errors: list[str]


class SkillEffectivenessTracker:
    """Tracks and aggregates skill execution effectiveness metrics.

    Uses StateStoreAdapter for persistent storage when available,
    falls back to in-memory tracking otherwise.

    Example usage::

        tracker = SkillEffectivenessTracker(state_store)
        await tracker.record_execution(
            "analyze-code",
            success=True,
            duration_ms=1234.5,
            user_rating=4.5,
        )
        stats = await tracker.get_stats("analyze-code")
        underperforming = await tracker.get_underperforming()
    """

    def __init__(self, state_store: StateStoreAdapter | None = None) -> None:
        """Initialize the effectiveness tracker.

        Args:
            state_store: Optional persistent storage backend. If None,
                metrics are stored in-memory only.
        """
        self._state_store = state_store
        self._in_memory: dict[str, list[dict[str, Any]]] = {}

    async def record_execution(
        self,
        skill_name: str,
        *,
        success: bool,
        duration_ms: float,
        user_rating: float | None = None,
        error: str | None = None,
    ) -> None:
        """Record a single skill execution outcome.

        Args:
            skill_name: Name of the skill that was executed.
            success: Whether the execution succeeded.
            duration_ms: Execution duration in milliseconds.
            user_rating: Optional user rating (0.0-5.0 scale).
            error: Optional error message if execution failed.
        """
        # Validate skill name to prevent key injection
        _validate_skill_name(skill_name)

        record = {
            "timestamp": time.time(),
            "success": success,
            "duration_ms": duration_ms,
            "user_rating": user_rating,
            "error": error[:500] if error else None,  # Cap error message length
        }

        if self._state_store is not None:
            # Load existing records
            key = f"{_STATS_PREFIX}{skill_name}"
            raw = await self._state_store.get(key)
            records = json.loads(raw.decode("utf-8")) if raw is not None else []

            records.append(record)

            # Cap records to prevent unbounded growth; keep most recent
            if len(records) > _MAX_RECORDS_PER_SKILL:
                records = records[-_MAX_RECORDS_PER_SKILL:]

            # Save updated records
            serialized = json.dumps(records)
            await self._state_store.set(key, serialized.encode("utf-8"))
            logger.debug(
                "Recorded execution for skill %s (success=%s)", skill_name, success
            )
        else:
            # Guard against unbounded in-memory growth
            if skill_name not in self._in_memory:
                if len(self._in_memory) >= _MAX_SKILLS_IN_MEMORY:
                    logger.warning(
                        "In-memory skill tracker at capacity (%d skills)",
                        _MAX_SKILLS_IN_MEMORY,
                    )
                    return
                self._in_memory[skill_name] = []
            self._in_memory[skill_name].append(record)

            # Cap in-memory records per skill
            if len(self._in_memory[skill_name]) > _MAX_RECORDS_PER_SKILL:
                self._in_memory[skill_name] = self._in_memory[skill_name][-_MAX_RECORDS_PER_SKILL:]

            logger.debug(
                "Recorded execution for skill %s in-memory (success=%s)",
                skill_name,
                success,
            )

    async def get_stats(self, skill_name: str) -> SkillStats | None:
        """Get aggregated statistics for a specific skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Aggregated SkillStats, or None if no execution data exists.
        """
        records = await self._load_records(skill_name)
        if not records:
            return None

        return self._aggregate_stats(skill_name, records)

    async def get_all_stats(self) -> list[SkillStats]:
        """Get statistics for all tracked skills.

        Returns:
            List of SkillStats for all skills with execution data.
        """
        all_stats: list[SkillStats] = []

        if self._state_store is not None:
            # Load from persistent storage
            async for key in self._state_store.list_keys(_STATS_PREFIX):
                skill_name = key.removeprefix(_STATS_PREFIX)
                records = await self._load_records(skill_name)
                if records:
                    stats = self._aggregate_stats(skill_name, records)
                    all_stats.append(stats)
        else:
            # Load from in-memory storage
            for skill_name, records in self._in_memory.items():
                if records:
                    stats = self._aggregate_stats(skill_name, records)
                    all_stats.append(stats)

        return all_stats

    async def get_underperforming(
        self,
        *,
        min_executions: int = 5,
        max_success_rate: float = 0.7,
    ) -> list[SkillStats]:
        """Find skills with low success rates.

        Args:
            min_executions: Minimum number of executions required to evaluate.
            max_success_rate: Maximum success rate to consider underperforming.

        Returns:
            List of underperforming skills, sorted by success rate (worst first).
        """
        all_stats = await self.get_all_stats()

        underperforming = [
            stats
            for stats in all_stats
            if stats.total_executions >= min_executions
            and stats.success_rate <= max_success_rate
        ]

        # Sort by success rate (worst first)
        underperforming.sort(key=lambda s: s.success_rate)

        logger.info(
            "Found %d underperforming skills (min_executions=%d, max_success_rate=%.2f)",
            len(underperforming),
            min_executions,
            max_success_rate,
        )

        return underperforming

    async def _load_records(self, skill_name: str) -> list[dict[str, Any]]:
        """Load raw execution records for a skill."""
        if self._state_store is not None:
            key = f"{_STATS_PREFIX}{skill_name}"
            raw = await self._state_store.get(key)
            if raw is not None:
                return json.loads(raw.decode("utf-8"))
            return []

        return self._in_memory.get(skill_name, [])

    def _aggregate_stats(
        self, skill_name: str, records: list[dict[str, Any]]
    ) -> SkillStats:
        """Aggregate raw execution records into SkillStats."""
        total = len(records)
        successful = sum(1 for r in records if r["success"])
        success_rate = successful / total if total > 0 else 0.0

        durations = [r["duration_ms"] for r in records]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        ratings = [r["user_rating"] for r in records if r["user_rating"] is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        last_execution = max((r["timestamp"] for r in records), default=0.0)

        errors = [r["error"] for r in records if r["error"] is not None]
        error_count = len(errors)

        # Find most common errors (top 5)
        error_freq: dict[str, int] = {}
        for error in errors:
            error_freq[error] = error_freq.get(error, 0) + 1
        common_errors = sorted(error_freq.keys(), key=error_freq.get, reverse=True)[:5]  # type: ignore[arg-type]

        return SkillStats(
            skill_name=skill_name,
            total_executions=total,
            successful_executions=successful,
            success_rate=success_rate,
            avg_duration_ms=avg_duration,
            avg_user_rating=avg_rating,
            last_execution=last_execution,
            error_count=error_count,
            common_errors=common_errors,
        )
