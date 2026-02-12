"""Persistent learning memory for the Memfun agent.

After each conversation turn, extracts key learnings (user preferences,
technical details, project patterns) and stores them in both:

1. MemoryStore (SQLite-backed TF-IDF searchable database)
2. MEMORY.md (visible, user-editable markdown file)

Before each turn, retrieves relevant learnings from MEMORY.md
(always loaded in full) plus TF-IDF search from the database,
and injects them as highest-priority context.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import dspy
from memfun_agent.signatures import LearningExtraction
from memfun_core.logging import get_logger
from memfun_optimizer.memory.store import MemoryStore
from memfun_optimizer.memory.types import MemoryQuery

if TYPE_CHECKING:
    from memfun_optimizer.memory.types import MemoryEntry
    from memfun_runtime.protocols.state_store import StateStoreAdapter

logger = get_logger("cli.learning")

# ── Constants ────────────────────────────────────────────────

_MAX_LEARNINGS_IN_CONTEXT = 15
_MAX_RESPONSE_FOR_EXTRACTION = 2_000
_MAX_EXISTING_FOR_CONFLICT = 50
_LEARNING_SOURCE = "conversation"


# ── LearningManager ─────────────────────────────────────────


class LearningManager:
    """Manages persistent learning extraction, storage, and retrieval.

    Uses a dual strategy:

    - **MEMORY.md** (primary): Always loaded in full. Visible, editable.
    - **MemoryStore** (secondary): TF-IDF searchable database.

    Both are populated after each turn by the LLM extraction pipeline.

    Lifecycle::

        manager = LearningManager(state_store)

        # Before each turn: get learnings for context
        section = await manager.get_relevant_learnings(user_query)

        # After each turn: extract and store learnings
        await manager.extract_and_store(user_input, agent_response)
    """

    def __init__(
        self,
        state_store: StateStoreAdapter | None = None,
    ) -> None:
        self._memory = MemoryStore(state_store)
        self._extractor = dspy.Predict(LearningExtraction)

    # ── Retrieval ──────────────────────────────────────

    async def get_relevant_learnings(
        self, query: str
    ) -> str:
        """Retrieve learnings relevant to the current query.

        Uses a dual strategy:
        1. MEMORY.md files (always loaded in full).
        2. TF-IDF search from database for additional matches.
        3. High-confidence global preferences from database.

        Args:
            query: The user's current message.

        Returns:
            Formatted context section string, or empty string
            if no relevant learnings exist.
        """
        if not query.strip():
            return ""

        # Primary: always load MEMORY.md
        from memfun_cli.memory import load_memory_context

        file_memory = load_memory_context()

        # Secondary: TF-IDF search from database
        db_lines: list[str] = []
        try:
            seen_ids: set[str] = set()

            search_query = MemoryQuery(
                query=query,
                tags=["learning"],
                min_confidence=0.3,
                limit=_MAX_LEARNINGS_IN_CONTEXT,
            )
            results = await self._memory.search(
                search_query
            )

            for result in results:
                if result.entry.id not in seen_ids:
                    seen_ids.add(result.entry.id)
                    db_lines.append(
                        f"- {result.entry.content}"
                    )

            prefs = await self._memory.list_entries(
                topic="learning:preference",
                limit=20,
            )
            for entry in prefs:
                if (
                    entry.confidence >= 0.9
                    and entry.id not in seen_ids
                ):
                    seen_ids.add(entry.id)
                    db_lines.append(
                        f"- {entry.content}"
                    )
        except Exception:
            logger.debug(
                "Failed to search database learnings",
                exc_info=True,
            )

        # Combine: file memory first, then database extras
        if file_memory and db_lines:
            return (
                file_memory
                + "\n\n=== DATABASE LEARNINGS ===\n"
                + "\n".join(db_lines)
            )
        if file_memory:
            return file_memory
        if db_lines:
            return (
                "=== LEARNED PREFERENCES & PATTERNS ===\n"
                "Apply these to your response:\n"
                + "\n".join(db_lines)
            )
        return ""

    # ── Extraction ─────────────────────────────────────

    async def extract_and_store(
        self,
        user_input: str,
        agent_response: str,
    ) -> list[str]:
        """Extract learnings from a conversation turn and persist.

        Stores in both MemoryStore (database) and MEMORY.md (file).

        Args:
            user_input: The user's message.
            agent_response: The agent's response (will be truncated).

        Returns:
            List of learning content strings that were stored.
        """
        if not user_input.strip() or not agent_response.strip():
            return []

        try:
            truncated = agent_response[
                :_MAX_RESPONSE_FOR_EXTRACTION
            ]

            # Load existing learnings for conflict detection
            # from both MEMORY.md and database
            from memfun_cli.memory import load_memory_context

            file_context = (
                load_memory_context()
                or "(no file learnings)"
            )

            existing = await self._memory.list_entries(
                limit=_MAX_EXISTING_FOR_CONFLICT,
            )
            existing = [
                e
                for e in existing
                if e.topic.startswith("learning:")
            ]

            db_text = (
                "\n".join(
                    f"- [{e.topic}] {e.content}"
                    for e in existing
                )
                if existing
                else ""
            )

            existing_text = file_context
            if db_text:
                existing_text += "\n" + db_text

            logger.info(
                "Starting learning extraction"
                " (input: %d chars, response: %d chars)",
                len(user_input),
                len(truncated),
            )

            # Run extraction via LLM
            result = await asyncio.to_thread(
                self._extractor,
                user_message=user_input,
                agent_response=truncated,
                existing_learnings=existing_text,
            )

            raw_learnings = getattr(
                result, "learnings", []
            )
            raw_updates = getattr(result, "updates", [])

            logger.info(
                "Extraction result: learnings=%r,"
                " updates=%r",
                raw_learnings,
                raw_updates,
            )

            # Normalize lists
            learnings = _normalize_list(raw_learnings)
            updates = _normalize_list(raw_updates)

            if not learnings:
                logger.info("No learnings extracted")
                return []

            logger.info(
                "Normalized: %d learnings, %d updates",
                len(learnings),
                len(updates),
            )

            stored: list[str] = []

            for text in learnings[:3]:
                topic, content = _parse_learning(text)

                # Store in MemoryStore (database)
                updated = await self._try_update_existing(
                    content, topic, existing, updates
                )
                if not updated:
                    await self._memory.add(
                        topic=f"learning:{topic}",
                        content=content,
                        source=_LEARNING_SOURCE,
                        confidence=0.8,
                        tags=("learning", topic),
                    )

                # ALSO store in MEMORY.md (visible file)
                from memfun_cli.memory import append_learning

                append_learning(content)

                stored.append(content)
                logger.info(
                    "Stored learning [%s]: %s",
                    topic,
                    content[:80],
                )

            return stored

        except Exception:
            logger.warning(
                "Learning extraction failed",
                exc_info=True,
            )
            return []

    # ── Internal helpers ───────────────────────────────

    async def _try_update_existing(
        self,
        new_content: str,
        new_topic: str,
        existing: list[MemoryEntry],
        updates: list[str],
    ) -> bool:
        """Try to update an existing entry that conflicts.

        The LLM's ``updates`` output lists the text of outdated
        entries. We match by substring and update.

        Returns:
            True if an existing entry was updated.
        """
        if not updates:
            return False

        for update_text in updates:
            lowered = update_text.strip().lower()
            if not lowered:
                continue
            for entry in existing:
                entry_lower = entry.content.lower()
                if (
                    lowered in entry_lower
                    or entry_lower in lowered
                ):
                    await self._memory.update(
                        entry.id,
                        content=new_content,
                        confidence=0.85,
                        tags=("learning", new_topic),
                    )
                    logger.info(
                        "Updated learning %s: %s -> %s",
                        entry.id,
                        entry.content[:40],
                        new_content[:40],
                    )
                    return True

        return False


# ── Module-level utilities ───────────────────────────────────


def _parse_learning(text: str) -> tuple[str, str]:
    """Parse ``[topic] content`` format.

    Returns ``("general", text)`` if format doesn't match.
    """
    text = text.strip()
    if text.startswith("[") and "]" in text:
        bracket_end = text.index("]")
        topic = text[1:bracket_end].strip().lower()
        content = text[bracket_end + 1 :].strip()
        valid = (
            "preference",
            "pattern",
            "technical",
            "workflow",
        )
        if topic in valid and content:
            return topic, content
    return "general", text


def _normalize_list(value: Any) -> list[str]:
    """Coerce an LLM output to a list of non-empty strings.

    Handles multiple formats that DSPy might return:
    - Python list of strings
    - JSON array string
    - Newline-separated string
    - Single string
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str) and value.strip():
        # Try parsing as JSON array
        if value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [
                        str(v).strip()
                        for v in parsed
                        if v
                    ]
            except (json.JSONDecodeError, ValueError):
                pass
        # Try splitting by newlines
        if "\n" in value:
            lines = [
                line.strip().lstrip("- ").strip()
                for line in value.split("\n")
                if line.strip()
            ]
            if lines:
                return lines
        return [value.strip()]
    return []
