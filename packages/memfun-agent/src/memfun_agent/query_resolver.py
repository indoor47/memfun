"""QueryResolver: resolve deictic references in user queries.

Placed between raw user input and the triage/RLM pipeline.
Expands short/ambiguous queries like "fix this", "2", "do it"
into fully explicit task descriptions using conversation history.
"""
from __future__ import annotations

import asyncio
from typing import Any

import dspy
from memfun_core.logging import get_logger

from memfun_agent.signatures import QueryResolution

logger = get_logger("agent.resolver")

# Queries longer than this are assumed self-contained.
_MAX_SHORT_QUERY = 150

# Minimum history entries needed for resolution (1 user + 1 assistant).
_MIN_HISTORY = 2

# Maximum chars to include per history entry.
_HISTORY_TRUNCATE = 2000


class QueryResolver(dspy.Module):
    """Resolves deictic references in user queries before triage.

    Short-circuits for long queries or missing history.
    """

    def __init__(self) -> None:
        super().__init__()
        self.resolver = dspy.Predict(QueryResolution)

    async def aresolve(
        self,
        query: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """Resolve *query* against *history*.

        Returns:
            Tuple of ``(resolved_query, was_resolved)``.
        """
        stripped = query.strip()

        # Long queries are assumed explicit.
        if len(stripped) > _MAX_SHORT_QUERY:
            return query, False

        # No history -> nothing to resolve against.
        if len(history) < _MIN_HISTORY:
            return query, False

        # Build conversation context from last 2 turns (up to 4 entries).
        recent = history[-4:]
        parts: list[str] = []
        for entry in recent:
            role = entry.get("role", "user")
            content = str(entry.get("content", ""))[:_HISTORY_TRUNCATE]
            tag = "User" if role == "user" else "Agent"
            parts.append(f"{tag}: {content}")

            if role == "assistant":
                files = entry.get("files_created", [])
                if files:
                    parts.append(f"  [files: {', '.join(str(f) for f in files[:10])}]")
                ops = entry.get("ops", [])
                if ops:
                    parts.append(f"  [ops: {len(ops)} operations]")

        conversation_context = "\n".join(parts)

        try:
            result = await asyncio.to_thread(
                self.resolver,
                query=query,
                conversation_context=conversation_context,
            )
        except Exception:
            logger.warning("QueryResolver DSPy call failed", exc_info=True)
            return query, False

        resolved = str(getattr(result, "resolved_query", query)).strip()
        was_resolved = bool(getattr(result, "was_resolved", False))

        # Sanity: if resolved is empty, fall back.
        if not resolved:
            return query, False

        if was_resolved:
            logger.info(
                "Resolved: '%s' -> '%s'",
                stripped[:80],
                resolved[:80],
            )

        return resolved, was_resolved
