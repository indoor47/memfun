"""Lightweight keyword/TF-IDF search engine for memory entries.

This module provides a pure-Python search implementation with no external
dependencies. It uses a simple inverted index and TF-IDF-like scoring to
rank memory entries by relevance.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfun_optimizer.memory.types import (
        MemoryEntry,
        MemoryQuery,
        MemorySearchResult,
    )

# Common English stopwords to filter out
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "this", "but", "they", "have",
    "had", "what", "when", "where", "who", "which", "why", "how",
})


class MemorySearch:
    """Lightweight search engine for memory entries.

    Uses keyword matching and TF-IDF-like scoring to rank entries by
    relevance. Supports topic and tag filtering, confidence weighting,
    and configurable result limits.

    The scoring algorithm:
    - Exact topic match: +10 points
    - Tag match: +5 points per matching tag
    - Content keyword match: +1 per term, weighted by inverse document frequency
    - Final score multiplied by entry confidence

    Example::

        search = MemorySearch()
        search.index(all_entries)
        results = search.search(query, all_entries)
        for result in results:
            print(f"{result.relevance_score:.2f}: {result.entry.content}")
    """

    def __init__(self) -> None:
        # Inverted index: term -> set of entry IDs containing that term
        self._index: dict[str, set[str]] = defaultdict(set)
        # Document frequency: term -> number of entries containing it
        self._doc_freq: Counter[str] = Counter()
        # Total number of indexed entries
        self._total_docs = 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into searchable terms.

        Converts to lowercase, splits on non-alphanumeric characters,
        and filters out stopwords.

        Args:
            text: Raw text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\w+", text.lower())
        # Filter stopwords and short tokens
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    def index(self, entries: list[MemoryEntry]) -> None:
        """Build an inverted index from a list of entries.

        This must be called before search() with the full set of entries
        to be searched. It indexes content, topic, and tags.

        Args:
            entries: All memory entries to index
        """
        self._index.clear()
        self._doc_freq.clear()
        self._total_docs = len(entries)

        for entry in entries:
            # Combine all searchable text
            searchable_text = " ".join([
                entry.topic,
                entry.content,
                *entry.tags,
            ])
            terms = set(self._tokenize(searchable_text))

            for term in terms:
                self._index[term].add(entry.id)
                self._doc_freq[term] += 1

    def search(
        self,
        query: MemoryQuery,
        entries: list[MemoryEntry],
    ) -> list[MemorySearchResult]:
        """Search entries using the given query.

        Scores each entry based on keyword matches, topic/tag matches,
        and confidence. Applies filters and returns top-k results.

        Args:
            query: Search parameters
            entries: All entries to search (must match indexed entries)

        Returns:
            Ranked list of search results, sorted by relevance descending
        """
        from memfun_optimizer.memory.types import MemorySearchResult

        # Tokenize the query
        query_terms = self._tokenize(query.query)

        # Score each entry
        scored_results: list[tuple[float, MemoryEntry]] = []

        for entry in entries:
            # Apply filters
            if entry.confidence < query.min_confidence:
                continue

            if query.topics and entry.topic not in query.topics:
                continue

            if query.tags and not any(tag in entry.tags for tag in query.tags):
                continue

            # Calculate relevance score
            score = 0.0

            # Exact topic match bonus
            if query.query.lower() in entry.topic.lower():
                score += 10.0

            # Tag match bonus
            if query.tags:
                matching_tags = set(query.tags) & set(entry.tags)
                score += len(matching_tags) * 5.0

            # Content keyword matching with IDF weighting
            entry_terms = set(self._tokenize(
                f"{entry.topic} {entry.content} {' '.join(entry.tags)}"
            ))

            for term in query_terms:
                if term in entry_terms:
                    # TF-IDF-like scoring: weight by inverse document frequency
                    doc_freq = self._doc_freq.get(term, 0)
                    if doc_freq > 0 and self._total_docs > 0:
                        # IDF = log(N / df)
                        idf = math.log(self._total_docs / doc_freq)
                        score += idf
                    else:
                        score += 1.0

            # Weight by confidence
            score *= entry.confidence

            if score > 0:
                scored_results.append((score, entry))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Limit results
        scored_results = scored_results[:query.limit]

        # Convert to MemorySearchResult objects
        return [
            MemorySearchResult(entry=entry, relevance_score=score)
            for score, entry in scored_results
        ]
