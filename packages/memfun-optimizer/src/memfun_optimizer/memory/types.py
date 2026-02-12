"""Memory data types for persistent knowledge storage.

The memory system stores facts, learned patterns, and optimized configurations
across sessions. These types define the core data structures for memory entries,
queries, and search results.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single fact or learned piece of knowledge.

    Memory entries are the fundamental unit of persistent knowledge storage.
    They capture specific facts, patterns, or insights learned during agent
    execution, trace analysis, or user interaction.

    Attributes:
        id: Unique identifier for this entry
        topic: High-level category (e.g., "bug_patterns", "optimization")
        content: The actual knowledge/fact being stored
        source: Where this knowledge came from
        confidence: How confident we are in this knowledge (0.0-1.0)
        tags: Additional categorization tags
        created_at: Unix timestamp when entry was created
        updated_at: Unix timestamp when entry was last updated
        metadata: Additional structured data
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    topic: str = ""
    content: str = ""
    source: str = "agent"  # "trace_analysis" | "user" | "agent" | "optimization"
    confidence: float = 0.8
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "topic": self.topic,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create a MemoryEntry from a dictionary."""
        return cls(
            id=data["id"],
            topic=data.get("topic", ""),
            content=data.get("content", ""),
            source=data.get("source", "agent"),
            confidence=data.get("confidence", 0.8),
            tags=tuple(data.get("tags", [])),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Search parameters for querying the memory store.

    Defines the criteria for searching stored knowledge. Supports
    keyword search, topic filtering, tag filtering, and confidence
    thresholds.

    Attributes:
        query: Free-text search query
        topics: Restrict to specific topics (OR logic)
        tags: Restrict to entries with specific tags (OR logic)
        min_confidence: Minimum confidence threshold
        limit: Maximum number of results to return
    """

    query: str = ""
    topics: list[str] | None = None
    tags: list[str] | None = None
    min_confidence: float = 0.0
    limit: int = 20


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A single ranked search result.

    Wraps a MemoryEntry with its computed relevance score for
    the given search query.

    Attributes:
        entry: The memory entry that matched
        relevance_score: Computed relevance (higher = better match)
    """

    entry: MemoryEntry
    relevance_score: float


@dataclass(frozen=True, slots=True)
class DailySummary:
    """Summary of recent memory activity.

    Provides a high-level overview of what knowledge was added or
    updated on a given date. Useful for daily standup reports or
    continuous learning dashboards.

    Attributes:
        date: ISO date string (YYYY-MM-DD)
        entries_added: Count of new entries
        entries_updated: Count of updated entries
        top_topics: Most active topics
        summary_text: Human-readable summary
    """

    date: str
    entries_added: int = 0
    entries_updated: int = 0
    top_topics: list[str] = field(default_factory=list)
    summary_text: str = ""
