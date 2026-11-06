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


class _DuplicateKeyError(ValueError):
    """Raised for a JSON object with a repeated key (ambiguous verdict)."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            msg = f"duplicate key {key!r}"
            raise _DuplicateKeyError(msg)
        out[key] = value
    return out


_DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys)


def _looks_like_json_object(text: str, idx: int) -> bool:
    """``{`` followed (after whitespace) by ``"`` or ``}``."""
    j = idx + 1
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    return j < len(text) and text[j] in '"}'


def _balanced_end(text: str, idx: int) -> int:
    """Index just past the ``}`` matching ``text[idx]``; ``-1`` if unbalanced.

    String-aware (quotes and backslash escapes), so braces inside JSON
    strings do not count.
    """
    depth = 0
    in_string = False
    escaped = False
    for j in range(idx, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


@dataclass(frozen=True, slots=True)
class JsonScan:
    """Top-level JSON objects found in a text, plus undecodable ones.

    ``malformed`` holds the raw text of every JSON-looking object that
    failed to decode (syntax error, duplicate key, or truncated — in
    which case it runs to the end of the text).  Their contents are
    NEVER searched for nested objects.
    """

    objects: tuple[dict[str, Any], ...]
    malformed: tuple[str, ...]


def scan_json_objects(text: str) -> JsonScan:
    """Scan *text* for top-level JSON objects, in order.

    * A decodable object is reported and the scan resumes past its end,
      so inner objects are never reported separately.
    * A JSON-looking object (``{"`` or ``{}``) that fails to decode is
      recorded in ``malformed`` and skipped as a whole (balanced-brace
      extent); when it is truncated the rest of the text is consumed.
      This prevents falling back to nested objects of a broken outer
      object (e.g. an inner ``"evidence": {"findings": []}``).
    * Any other ``{`` (prose) is skipped.
    """
    objects: list[dict[str, Any]] = []
    malformed: list[str] = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = _DECODER.raw_decode(text, idx)
        except (json.JSONDecodeError, ValueError):
            if not _looks_like_json_object(text, idx):
                idx = text.find("{", idx + 1)
                continue
            end = _balanced_end(text, idx)
            if end == -1:
                malformed.append(text[idx:])
                break
            malformed.append(text[idx:end])
            idx = text.find("{", end)
            continue
        if isinstance(obj, dict):
            objects.append(cast("dict[str, Any]", obj))
        idx = text.find("{", end)
    return JsonScan(objects=tuple(objects), malformed=tuple(malformed))


def iter_json_objects(text: str) -> list[dict[str, Any]]:
    """Decodable top-level JSON objects in *text* (see :func:`scan_json_objects`)."""
    return list(scan_json_objects(text).objects)


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
    """Aggregate findings from EVERY findings-bearing JSON object.

    Verdict aggregation rule: **most severe wins**.  All top-level JSON
    objects with a ``findings`` key are collected and their findings
    merged; the verdict is computed over the union, so one blocking
    finding anywhere means changes requested.  Text appended after the
    reviewer's real block (a quoted diff line, an echoed fixture, a
    LOW-only object) can therefore only ADD findings, never cancel a
    blocking one — injection can make a review stricter, never laxer.

    Returns ``None`` (reviewer error, never approval) when:

    * no findings-bearing object exists (prose-only answer);
    * any findings-bearing object has a malformed ``findings`` value;
    * any JSON-looking object that mentions ``findings`` fails to decode
      (truncated, syntax error, duplicate keys such as
      ``"findings": [HIGH], "findings": []``).
    """
    scan = scan_json_objects(answer or "")
    if any('"findings"' in blob for blob in scan.malformed):
        return None
    bearing = [obj for obj in scan.objects if "findings" in obj]
    if not bearing:
        return None
    merged: list[ReviewFinding] = []
    for obj in bearing:
        findings = findings_from_object(obj)
        if findings is None:
            return None
        merged.extend(findings)
    last = bearing[-1]
    summary = last.get("summary")
    claimed = last.get("verdict")
    return ParsedReview(
        # Dedupe identical findings (e.g. the same block quoted twice).
        findings=tuple(dict.fromkeys(merged)),
        summary=summary.strip() if isinstance(summary, str) else "",
        claimed_verdict=claimed if isinstance(claimed, str) else "",
        raw=last,
    )


PrVerdict = Literal["approve", "request-changes", "comment"]
_PR_VERDICTS = ("approve", "request-changes", "comment")


def compute_pr_verdict(answer: str) -> PrVerdict:
    """Fail-closed PR-layer verdict over EVERY verdict/findings object.

    Most severe wins (``request-changes`` > ``comment`` > ``approve``):

    * any object with ``"verdict": "request-changes"``, or any blocking
      (HIGH/MEDIUM, incl. CRITICAL/WARNING) finding in any object, ->
      ``request-changes``;
    * otherwise any ``comment`` verdict, any unknown verdict value, any
      malformed ``findings`` value, any undecodable JSON-looking object
      mentioning ``verdict``/``findings``, or an ``approve`` WITHOUT a
      ``findings`` key -> ``comment``;
    * ``approve`` only when every verdict object approves with a
      well-formed, non-blocking ``findings`` list;
    * no verdict object at all (prose such as "LGTM") -> ``comment``.
    """
    scan = scan_json_objects(answer or "")
    relevant = [o for o in scan.objects if "verdict" in o or "findings" in o]
    verdict_objs = [o for o in relevant if "verdict" in o]
    downgrade = any('"verdict"' in b or '"findings"' in b for b in scan.malformed)
    if not verdict_objs:
        return "comment"
    for obj in relevant:
        if obj.get("verdict") == "request-changes":
            return "request-changes"
        if "findings" in obj:
            findings = findings_from_object(obj)
            if findings is None:
                downgrade = True
            elif compute_verdict(findings) == "CHANGES_REQUESTED":
                return "request-changes"
    for obj in verdict_objs:
        verdict = obj.get("verdict")
        if verdict not in _PR_VERDICTS or verdict == "comment" or "findings" not in obj:
            downgrade = True
    return "comment" if downgrade else "approve"
