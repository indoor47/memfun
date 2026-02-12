"""Skill routing: matches a query to the best-fitting skill(s)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfun_skills.types import SkillDefinition

logger = logging.getLogger("memfun.skills.router")

# ── Scoring weights ───────────────────────────────────────────

_SCORE_EXACT_NAME = 100
_SCORE_TAG_MATCH = 60
_SCORE_DESCRIPTION_KEYWORD = 30


# ── Scored result ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoredSkill:
    """A skill paired with a routing relevance score."""

    skill: SkillDefinition
    score: int
    match_reasons: list[str] = field(default_factory=list)


# ── Router ────────────────────────────────────────────────────


class SkillRouter:
    """Routes a query string to matching skill definitions.

    Matching strategies (applied in priority order):
    1. **Exact name** -- the query matches the skill name exactly.
    2. **Tag match** -- the query matches one of the skill's tags.
    3. **Description keyword** -- the query appears as a
       case-insensitive substring in the skill description.
    """

    def route(
        self,
        query: str,
        skills: list[SkillDefinition],
    ) -> SkillDefinition | None:
        """Return the single best-matching skill, or *None*.

        Args:
            query: The search query (skill name, tag, or keyword).
            skills: The pool of available skills to search.

        Returns:
            The highest-scoring SkillDefinition, or *None* if nothing
            matches.
        """
        ranked = self.route_all(query, skills)
        if not ranked:
            logger.debug("No skill matched query: %r", query)
            return None
        best = ranked[0]
        logger.debug(
            "Best match for %r: %s (score=%d)",
            query,
            best.skill.name,
            best.score,
        )
        return best.skill

    def route_all(
        self,
        query: str,
        skills: list[SkillDefinition],
    ) -> list[ScoredSkill]:
        """Return all matching skills, scored and sorted (highest first).

        Skills that do not match any strategy are excluded from the
        result.

        Args:
            query: The search query (skill name, tag, or keyword).
            skills: The pool of available skills to search.

        Returns:
            A list of ScoredSkill, sorted by descending score.
        """
        if not query or not skills:
            return []

        query_lower = query.strip().lower()
        results: list[ScoredSkill] = []

        for skill in skills:
            score, reasons = self._score_skill(query_lower, skill)
            if score > 0:
                results.append(
                    ScoredSkill(
                        skill=skill,
                        score=score,
                        match_reasons=reasons,
                    )
                )

        # Sort by score descending, then by name ascending for stability
        results.sort(key=lambda s: (-s.score, s.skill.name))
        return results

    @staticmethod
    def _score_skill(
        query_lower: str,
        skill: SkillDefinition,
    ) -> tuple[int, list[str]]:
        """Compute a relevance score for a single skill.

        Returns:
            A (score, reasons) tuple.  A score of 0 means no match.
        """
        score = 0
        reasons: list[str] = []

        # Strategy 1: exact name match
        if query_lower == skill.name.lower():
            score += _SCORE_EXACT_NAME
            reasons.append(f"exact name: {skill.name}")

        # Strategy 2: tag match
        for tag in skill.tags:
            if query_lower == tag.lower():
                score += _SCORE_TAG_MATCH
                reasons.append(f"tag: {tag}")
                break  # one tag match is enough

        # Strategy 3: description keyword (substring)
        if query_lower in skill.description.lower():
            score += _SCORE_DESCRIPTION_KEYWORD
            reasons.append("description keyword")

        return score, reasons
