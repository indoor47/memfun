"""Pattern frequency analysis for strategy signatures.

The PatternAnalyzer clusters similar strategy signatures to identify
recurring execution patterns, then ranks them by frequency, confidence,
and success rate.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_optimizer.types import (
    StepKind,
    StrategyPattern,
)

if TYPE_CHECKING:
    from memfun_optimizer.types import StrategySignature, StrategyStep

logger = get_logger("optimizer.patterns")


class PatternAnalyzer:
    """Analyzes strategy signatures to discover recurring patterns.

    Groups signatures by step sequence similarity, computes aggregate
    metrics, and ranks patterns by composite score (frequency * confidence
    * success rate).

    Usage::

        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze(signatures)
        dominant = analyzer.find_dominant(patterns, min_frequency=3)
        ranked = analyzer.rank(dominant)
    """

    def __init__(self, *, similarity_threshold: float = 0.7) -> None:
        """Initialize the pattern analyzer.

        Args:
            similarity_threshold: Minimum similarity score (0-1) for clustering
        """
        self._threshold = similarity_threshold

    def analyze(
        self, signatures: list[StrategySignature]
    ) -> list[StrategyPattern]:
        """Analyze signatures to discover patterns.

        Args:
            signatures: List of strategy signatures to analyze

        Returns:
            List of discovered patterns (unfiltered)
        """
        logger.info("Analyzing %d signatures for patterns", len(signatures))

        if not signatures:
            return []

        # Cluster signatures by similarity
        clusters = self._cluster(signatures, threshold=self._threshold)
        logger.info("Found %d clusters", len(clusters))

        # Convert each cluster to a pattern
        patterns = [self._cluster_to_pattern(cluster) for cluster in clusters]

        return patterns

    def find_dominant(
        self,
        patterns: list[StrategyPattern],
        *,
        min_frequency: int = 2,
        min_confidence: float = 0.5,
    ) -> list[StrategyPattern]:
        """Filter patterns to find dominant ones.

        Args:
            patterns: List of patterns to filter
            min_frequency: Minimum number of traces exhibiting the pattern
            min_confidence: Minimum average classifier confidence

        Returns:
            Filtered list of dominant patterns
        """
        dominant = [
            p
            for p in patterns
            if p.frequency >= min_frequency and p.confidence >= min_confidence
        ]
        logger.info(
            "Found %d dominant patterns (from %d total)",
            len(dominant),
            len(patterns),
        )
        return dominant

    def rank(self, patterns: list[StrategyPattern]) -> list[StrategyPattern]:
        """Rank patterns by composite score.

        Score = frequency * confidence * success_rate

        Args:
            patterns: List of patterns to rank

        Returns:
            Patterns sorted by descending composite score
        """
        ranked = sorted(
            patterns,
            key=lambda p: p.frequency * p.confidence * p.success_rate,
            reverse=True,
        )
        logger.info("Ranked %d patterns", len(ranked))
        return ranked

    # ── Internal Methods ───────────────────────────────────────────

    def _normalize_sequence(self, steps: list[StrategyStep]) -> list[StepKind]:
        """Extract just the step kinds from a list of strategy steps.

        Args:
            steps: List of strategy steps

        Returns:
            List of step kinds in order
        """
        return [step.kind for step in steps]

    def _similarity(
        self, seq1: list[StepKind], seq2: list[StepKind]
    ) -> float:
        """Compute similarity between two step sequences.

        Uses longest common subsequence (LCS) ratio as similarity metric.

        Args:
            seq1: First step kind sequence
            seq2: Second step kind sequence

        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        if not seq1 or not seq2:
            return 0.0

        # Compute LCS length
        lcs_len = self._lcs_length(seq1, seq2)

        # Normalize by the length of the longer sequence
        max_len = max(len(seq1), len(seq2))
        return lcs_len / max_len if max_len > 0 else 0.0

    def _lcs_length(
        self, seq1: list[StepKind], seq2: list[StepKind]
    ) -> int:
        """Compute the length of the longest common subsequence.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of the LCS
        """
        m, n = len(seq1), len(seq2)
        # DP table: dp[i][j] = LCS length of seq1[:i] and seq2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _cluster(
        self,
        signatures: list[StrategySignature],
        *,
        threshold: float,
    ) -> list[list[StrategySignature]]:
        """Cluster signatures by step sequence similarity.

        Uses a greedy approach: start with each signature, merge similar
        ones into the same cluster.

        Args:
            signatures: List of signatures to cluster
            threshold: Minimum similarity for merging

        Returns:
            List of clusters (each cluster is a list of signatures)
        """
        clusters: list[list[StrategySignature]] = []

        for sig in signatures:
            seq = self._normalize_sequence(sig.steps)

            # Find the best matching cluster
            best_cluster = None
            best_similarity = 0.0

            for cluster in clusters:
                # Compare against the first signature in the cluster
                cluster_seq = self._normalize_sequence(cluster[0].steps)
                sim = self._similarity(seq, cluster_seq)
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster

            # Add to best cluster if above threshold, else create new cluster
            if best_cluster is not None and best_similarity >= threshold:
                best_cluster.append(sig)
            else:
                clusters.append([sig])

        return clusters

    def _cluster_to_pattern(
        self, cluster: list[StrategySignature]
    ) -> StrategyPattern:
        """Convert a cluster of signatures to a pattern.

        Args:
            cluster: List of signatures in the cluster

        Returns:
            A strategy pattern representing the cluster
        """
        # Use the first signature's step sequence as canonical
        canonical_seq = self._normalize_sequence(cluster[0].steps)

        # Compute aggregate metrics
        frequency = len(cluster)
        avg_confidence = sum(
            sum(step.confidence for step in sig.steps) / len(sig.steps)
            for sig in cluster
        ) / frequency
        avg_duration_ms = sum(sig.duration_ms for sig in cluster) / frequency
        avg_token_cost = sum(sig.token_cost for sig in cluster) / frequency
        success_count = sum(1 for sig in cluster if sig.success)
        success_rate = success_count / frequency if frequency > 0 else 0.0

        # Generate a stable pattern ID from the canonical sequence
        seq_str = "-".join(sk.value for sk in canonical_seq)
        pattern_id = hashlib.sha256(seq_str.encode()).hexdigest()[:12]

        # Sample up to 5 example trace IDs
        example_trace_ids = [sig.trace_id for sig in cluster[:5]]

        return StrategyPattern(
            pattern_id=pattern_id,
            step_sequence=canonical_seq,
            frequency=frequency,
            confidence=avg_confidence,
            avg_duration_ms=avg_duration_ms,
            avg_token_cost=avg_token_cost,
            success_rate=success_rate,
            example_trace_ids=example_trace_ids,
        )
