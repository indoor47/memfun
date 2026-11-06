"""Structured reviewer findings with a verdict computed in code.

Auditors (the per-task ``review-agent`` pairing, the cross-task review
and the PR-layer ``pr-reviewer``) emit a JSON block of findings::

    {"findings": [{"severity": "HIGH", "file": "a.py", "line": 3,
                   "issue": "...", "fix": "..."}]}

The verdict is NEVER taken from the reviewer's prose or from a
``verdict`` field alone: it is computed here from the severities.
Any ``HIGH`` or ``MEDIUM`` finding means changes are requested;
only ``LOW`` findings (or none) means approval.

Parsing fails closed: when no valid findings JSON can be extracted,
:func:`parse_review_output` returns ``None`` and callers must treat
the review as an *error* (not approved), never as an approval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

Severity = Literal["HIGH", "MEDIUM", "LOW"]
Verdict = Literal["APPROVED", "CHANGES_REQUESTED"]

# Canonical severities plus legacy aliases.  ``critical/major/minor``
# come from the old review-agent issue-line format; ``CRITICAL/
# WARNING/INFO`` from the pr-reviewer AGENT.md.
_SEVERITY_ALIASES: dict[str, Severity] = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "critical": "HIGH",
    "major": "MEDIUM",
    "minor": "LOW",
    "warning": "MEDIUM",
    "info": "LOW",
}

BLOCKING_SEVERITIES: frozenset[str] = frozenset({"HIGH", "MEDIUM"})


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """One reviewer finding.

    ``task_id`` is only meaningful for the cross-task review, where a
    finding must be routed back to the sub-task that caused it.
    """

    severity: Severity
    issue: str
    file: str = ""
    line: int | None = None
    fix: str = ""
    task_id: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity in BLOCKING_SEVERITIES

    def render(self) -> str:
        """One-line human/LLM-readable rendering for feedback text."""
        loc = self.file or "?"
        if self.line is not None:
            loc = f"{loc}:{self.line}"
        prefix = f"[{self.task_id}] " if self.task_id else ""
        text = f"- {prefix}[{self.severity}] {loc} — {self.issue}"
        if self.fix:
            text += f" (fix: {self.fix})"
        return text


@dataclass(frozen=True, slots=True)
class ParsedReview:
    """Findings extracted from a reviewer answer plus the computed verdict."""

    findings: tuple[ReviewFinding, ...]
    summary: str = ""
    # The reviewer's own ``verdict`` field, if any.  Informational only;
    # never used to decide approval.
    claimed_verdict: str = ""
    raw: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def verdict(self) -> Verdict:
        return compute_verdict(self.findings)

    @property
    def blocking(self) -> tuple[ReviewFinding, ...]:
        return tuple(f for f in self.findings if f.blocking)


def normalize_severity(raw: object) -> Severity:
    """Map a reviewer-supplied severity to ``HIGH``/``MEDIUM``/``LOW``.

    Unknown or missing severities fail closed to ``MEDIUM`` (blocking):
    a finding the reviewer bothered to emit is not silently dropped
    to advisory just because it used an unexpected label.
    """
    if isinstance(raw, str):
        mapped = _SEVERITY_ALIASES.get(raw.strip().lower())
        if mapped is not None:
            return mapped
    return "MEDIUM"


def compute_verdict(findings: tuple[ReviewFinding, ...] | list[ReviewFinding]) -> Verdict:
    """Any HIGH or MEDIUM finding -> CHANGES_REQUESTED, else APPROVED."""
    if any(f.blocking for f in findings):
        return "CHANGES_REQUESTED"
    return "APPROVED"


def iter_json_objects(text: str) -> list[dict[str, Any]]:
    """Return top-level JSON objects embedded in *text*, in order.

    Scans for ``{`` and attempts :meth:`json.JSONDecoder.raw_decode`
    at each candidate.  Works for objects inside fenced code blocks
    and for nested structures (unlike a flat bracket regex).  After a
    successful decode the scan resumes past the decoded object, so
    inner objects are not reported separately.
    """
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, ValueError):
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            out.append(cast("dict[str, Any]", obj))
        idx = text.find("{", end)
    return out


def _coerce_line(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _str_field(obj: dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _parse_finding(raw: object) -> ReviewFinding | None:
    if not isinstance(raw, dict):
        return None
    obj = cast("dict[str, Any]", raw)
    return ReviewFinding(
        severity=normalize_severity(obj.get("severity")),
        issue=_str_field(obj, "issue", "title", "description"),
        file=_str_field(obj, "file", "path"),
        line=_coerce_line(obj.get("line")),
        fix=_str_field(obj, "fix", "suggestion"),
        task_id=_str_field(obj, "task_id").upper(),
    )


def findings_from_object(obj: dict[str, Any]) -> tuple[ReviewFinding, ...] | None:
    """Parse ``obj["findings"]``; ``None`` if absent or malformed."""
    raw_findings: object = obj.get("findings")
    if not isinstance(raw_findings, list):
        return None
    parsed: list[ReviewFinding] = []
    for item in cast("list[object]", raw_findings):
        finding = _parse_finding(item)
        if finding is None:
            # A malformed entry makes the whole block untrustworthy.
            return None
        parsed.append(finding)
    return tuple(parsed)


def parse_review_output(answer: str) -> ParsedReview | None:
    """Extract the LAST valid JSON object with a ``findings`` list.

    Returns ``None`` when no such object exists (prose-only answer,
    truncated JSON, malformed findings).  Callers MUST treat ``None``
    as a reviewer error — never as an approval.
    """
    for obj in reversed(iter_json_objects(answer or "")):
        if "findings" not in obj:
            continue
        findings = findings_from_object(obj)
        if findings is None:
            # The last findings-bearing block is malformed: fail closed
            # rather than falling back to an earlier (possibly quoted)
            # block.
            return None
        summary = obj.get("summary")
        claimed = obj.get("verdict")
        return ParsedReview(
            findings=findings,
            summary=summary.strip() if isinstance(summary, str) else "",
            claimed_verdict=claimed if isinstance(claimed, str) else "",
            raw=obj,
        )
    return None
