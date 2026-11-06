"""Structured reviewer findings with a verdict computed in code.

Auditors (the per-task ``review-agent`` pairing, the cross-task review
and the PR-layer ``pr-reviewer``) emit a JSON block of findings::

    {"review_id": "<nonce>",
     "findings": [{"severity": "HIGH", "file": "a.py", "line": 3,
                   "issue": "...", "fix": "..."}]}

The verdict is NEVER taken from the reviewer's prose or from a
``verdict`` field alone: it is computed here from the severities.
Any ``HIGH`` or ``MEDIUM`` finding means changes are requested;
only ``LOW`` findings (or none) means approval.

Two rules make the parse robust against text the reviewer quotes
(diffs, fixtures, coder output):

* **Nonce.** Every reviewer dispatch carries a fresh random
  ``review_id`` (uuid4 hex) that only the reviewer prompt contains.
  An object counts as the reviewer's own block only if it is a
  top-level JSON object carrying that exact ``review_id``.  Approval
  REQUIRES at least one such block; without one the review is an
  error (per-task / cross-task) or ``comment`` (PR layer).  The coder
  never sees the reviewer prompt, so it cannot forge the nonce, and a
  nonce from an earlier dispatch or retry does not match.
* **Most severe wins.** Blocking findings (and a PR ``request-changes``
  verdict) in ANY object still block, nonce or not — quoted text can
  only make a review stricter, never more lenient.

Parsing fails closed: answers over :data:`MAX_REVIEW_ANSWER_CHARS`,
truncated/malformed/duplicate-key or too-deeply-nested JSON that
mentions ``findings``/``verdict``, and answers without a valid
nonce-bearing block are errors, never approvals.  The scan is linear
in the input size.

Known limitations (documented, accepted):

* Quoted text can still steer WHICH sub-task a cross-task revision
  targets (a quoted blocking finding with another ``task_id`` adds a
  revision target).  This only ever makes the gate stricter; the nonce
  does not close it, because blocking findings count from any object.
* Weak local models may omit the ``review_id`` echo: that is an error,
  retried once, and then the sub-task is unaccepted — the intended
  fail-closed behaviour, at the cost of false failures.
* Phase 6 auto-commit (workflow.py) runs before PR-layer failures flip
  ``success``; output revised by the Phase 4 cross-task loop is not
  re-reviewed per task (its PR is refused as stale instead).
"""

from __future__ import annotations

import json
import re
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


# Reviewer answers above this size are rejected outright (error /
# ``comment``): bounds parse time and memory on the event loop.
MAX_REVIEW_ANSWER_CHARS = 256 * 1024

_LEADING_WS = " \t\r\n"
# Characters that matter when brace-matching a JSON object.
_JSON_SPECIAL = re.compile(r'["{}\\]')


def _looks_like_json_object(text: str, idx: int) -> bool:
    """``{`` followed (after whitespace) by ``"`` or ``}``."""
    j = idx + 1
    n = len(text)
    while j < n and text[j] in _LEADING_WS:
        j += 1
    return j < n and text[j] in '"}'


def _balanced_end(text: str, idx: int) -> int:
    """Index just past the ``}`` matching ``text[idx]``; ``-1`` if unbalanced.

    String-aware (quotes and backslash escapes), so braces inside JSON
    strings do not count.  Linear: jumps between special characters
    with a compiled regex instead of looping over every character.
    """
    depth = 0
    in_string = False
    escaped_pos = -1
    for m in _JSON_SPECIAL.finditer(text, idx):
        pos = m.start()
        if pos == escaped_pos:
            # Character right after a backslash inside a string.
            continue
        ch = m.group()
        if in_string:
            if ch == "\\":
                escaped_pos = pos + 1
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
                return pos + 1
    return -1


@dataclass(frozen=True, slots=True)
class JsonScan:
    """Top-level JSON objects found in a text, plus undecodable ones.

    ``malformed`` holds the raw text of every JSON-looking object that
    failed to decode (syntax error, duplicate key, nesting too deep, or
    truncated — in which case it runs to the end of the text).  Their
    contents are NEVER searched for nested objects.
    """

    objects: tuple[dict[str, Any], ...]
    malformed: tuple[str, ...]


def scan_json_objects(text: str) -> JsonScan:
    """Scan *text* for top-level JSON objects, in order.

    * Only JSON-looking candidates (``{"`` or ``{}``) are considered;
      any other ``{`` (prose) is skipped without decoding.
    * The brace-matched extent is found first and ONLY that slice is
      decoded, so a failed candidate costs O(its extent), never
      O(remaining text); the scan resumes past the extent, so inner
      objects are never reported separately.  Total work is linear.
    * A candidate that fails to decode (syntax error, duplicate key,
      ``RecursionError`` from deep nesting) is recorded in
      ``malformed``; a truncated one consumes the rest of the text.
      Nested objects of a broken outer object are never examined.
    """
    objects: list[dict[str, Any]] = []
    malformed: list[str] = []
    idx = text.find("{")
    while idx != -1:
        if not _looks_like_json_object(text, idx):
            idx = text.find("{", idx + 1)
            continue
        end = _balanced_end(text, idx)
        if end == -1:
            malformed.append(text[idx:])
            break
        blob = text[idx:end]
        try:
            obj: object = _DECODER.decode(blob)
        except (ValueError, RecursionError):
            malformed.append(blob)
        else:
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


def _carries_nonce(obj: dict[str, Any], review_id: str) -> bool:
    return bool(review_id) and obj.get("review_id") == review_id


def parse_review_output(answer: str, *, review_id: str) -> ParsedReview | None:
    """Parse a reviewer answer for the dispatch identified by *review_id*.

    Rules (see the module docstring):

    * The reviewer's own blocks are the top-level objects whose
      ``review_id`` equals *review_id*.  At least one is REQUIRED, and
      every one of them must carry a well-formed ``findings`` list.
    * **Most severe wins**: findings from EVERY findings-bearing object
      (nonce or not) are merged, so a quoted blocking finding still
      blocks, while a quoted ``{"findings": []}`` can never approve.

    Returns ``None`` (reviewer error, never approval) when the answer
    is over :data:`MAX_REVIEW_ANSWER_CHARS`; when any JSON-looking
    object mentioning ``findings`` fails to decode; when any
    findings-bearing object has a malformed ``findings`` value; when a
    nonce-bearing object lacks a valid ``findings`` list; or when no
    nonce-bearing block exists.
    """
    text = answer or ""
    if len(text) > MAX_REVIEW_ANSWER_CHARS:
        return None
    scan = scan_json_objects(text)
    if any('"findings"' in blob or '"review_id"' in blob for blob in scan.malformed):
        return None
    own = [obj for obj in scan.objects if _carries_nonce(obj, review_id)]
    if not own:
        return None
    merged: list[ReviewFinding] = []
    for obj in scan.objects:
        if "findings" not in obj and obj not in own:
            continue
        findings = findings_from_object(obj)
        if findings is None:
            return None
        merged.extend(findings)
    last = own[-1]
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


def compute_pr_verdict(answer: str, *, review_id: str) -> PrVerdict:
    """Fail-closed PR-layer verdict for the dispatch *review_id*.

    * ``request-changes`` when ANY object (nonce or not) has
      ``"verdict": "request-changes"`` or any blocking (HIGH/MEDIUM,
      incl. CRITICAL/WARNING) finding — most severe wins.
    * ``approve`` only when at least one top-level object carries the
      matching ``review_id``, and every such object has
      ``"verdict": "approve"`` with a well-formed, non-blocking
      ``findings`` list.
    * ``comment`` otherwise: no nonce-bearing verdict block (prose such
      as "LGTM", or only quoted ``{"verdict": "approve", ...}`` from a
      diff), a nonce block with ``comment``/unknown verdict or without
      ``findings``, malformed findings anywhere, undecodable JSON that
      mentions ``verdict``/``findings``, or an oversized answer.
    """
    text = answer or ""
    if len(text) > MAX_REVIEW_ANSWER_CHARS:
        return "comment"
    scan = scan_json_objects(text)
    downgrade = any('"verdict"' in b or '"findings"' in b for b in scan.malformed)
    for obj in scan.objects:
        if obj.get("verdict") == "request-changes":
            return "request-changes"
        if "findings" in obj:
            findings = findings_from_object(obj)
            if findings is None:
                downgrade = True
            elif compute_verdict(findings) == "CHANGES_REQUESTED":
                return "request-changes"
    own = [obj for obj in scan.objects if _carries_nonce(obj, review_id)]
    if not own:
        return "comment"
    for obj in own:
        verdict = obj.get("verdict")
        if verdict != "approve" or "findings" not in obj:
            downgrade = True
    return "comment" if downgrade else "approve"
