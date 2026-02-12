"""Persistent memory system for agent knowledge storage.

This package provides event-sourced storage for learned facts, patterns,
and optimizations. Memory entries are persisted via the StateStoreAdapter
protocol and can be searched using a lightweight TF-IDF-based search engine.

Core components:

- MemoryStore: Event-sourced persistent storage with CRUD operations
- MemorySearch: Keyword/TF-IDF search engine for ranking entries
- MemoryEntry: A single stored fact or pattern
- MemoryQuery: Search parameters for querying stored knowledge
- DailySummary: Summary of recent memory activity

Example usage::

    from memfun_optimizer.memory import MemoryStore, MemoryQuery

    store = MemoryStore(state_store)

    # Add knowledge
    await store.add(
        topic="optimization",
        content="Using batch size 32 improved throughput by 40%",
        source="trace_analysis",
        confidence=0.95,
        tags=("performance", "batching"),
    )

    # Search for related knowledge
    query = MemoryQuery(query="performance improvements", min_confidence=0.8)
    results = await store.search(query)
    for result in results:
        print(f"[{result.relevance_score:.2f}] {result.entry.content}")
"""
from __future__ import annotations

from memfun_optimizer.memory.search import MemorySearch
from memfun_optimizer.memory.store import MemoryStore
from memfun_optimizer.memory.types import (
    DailySummary,
    MemoryEntry,
    MemoryQuery,
    MemorySearchResult,
)

__all__ = [
    "DailySummary",
    "MemoryEntry",
    "MemoryQuery",
    "MemorySearch",
    "MemorySearchResult",
    "MemoryStore",
]
