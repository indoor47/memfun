"""Context-First Solver: gather context, then solve in one shot.

Instead of iterating through the RLM loop (which wastes calls on context
discovery), this module:

1. **Plans** what context is needed (1 LLM call, or skip for small projects)
2. **Gathers** files and search results (pure I/O, no LLM)
3. **Solves** in a single pass with full context (1 LLM call)
4. **Executes** the resulting file operations mechanically

Falls back to the existing RLM loop when the single-shot approach fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import time
import uuid

import dspy
from memfun_core.config import LLMConfig
from memfun_core.logging import get_logger

from memfun_agent import fixloop
from memfun_agent.signatures import (
    ConsistencyReview,
    ContextPlanning,
    ExperimentProposal,
    SingleShotSolving,
    VerificationFix,
)

logger = get_logger("agent.context_first")


# ── Fuzzy edit matching helpers ──────────────────────────────


def _normalize_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _apply_ws_normalized_replace(content: str, old: str, new: str) -> str | None:
    """Replace *old* in *content* using whitespace-normalized matching.

    Strips trailing whitespace per line from both *content* and *old*
    to find the match location, then replaces the corresponding span
    in the **original** content with *new*.

    Returns the new content, or ``None`` if no match is found.
    """
    norm_content = _normalize_whitespace(content)
    norm_old = _normalize_whitespace(old)

    idx = norm_content.find(norm_old)
    if idx == -1:
        return None

    # Map the normalized index back to the original content.
    # Walk the original content, tracking the normalized position.
    orig_start = _map_norm_index_to_orig(content, idx)
    orig_end = _map_norm_index_to_orig(content, idx + len(norm_old))

    return content[:orig_start] + new + content[orig_end:]


def _map_norm_index_to_orig(content: str, norm_idx: int) -> int:
    """Map a character index in the normalized text back to the original."""
    norm_pos = 0
    orig_pos = 0
    lines = content.split("\n")

    for i, line in enumerate(lines):
        stripped = line.rstrip()

        if norm_pos + len(stripped) >= norm_idx:
            # Target is within this line's non-trailing content.
            offset = norm_idx - norm_pos
            return orig_pos + offset

        # Account for the stripped line + the newline character.
        norm_pos += len(stripped)
        orig_pos += len(line)

        if i < len(lines) - 1:
            # The newline between lines.
            norm_pos += 1  # \n in normalized text
            orig_pos += 1  # \n in original text

    return orig_pos


def _fuzzy_find_and_replace(
    content: str,
    old: str,
    new: str,
    min_ratio: float = 0.80,
) -> tuple[str | None, float, str, str]:
    """Progressively try to match *old* in *content* and replace with *new*.

    Strategies (in order):
    1. **Exact** — ``old in content``
    2. **Whitespace-normalized** — strip trailing whitespace per line
    3. **Line fuzzy** — sliding window with ``difflib.SequenceMatcher``

    Returns:
        ``(new_content | None, best_ratio, best_snippet, strategy)``
        where *strategy* is one of ``"exact"``, ``"whitespace"``,
        ``"fuzzy"``, or ``"none"`` (if no match found).
    """
    # 1. Exact match.
    if old in content:
        result = content.replace(old, new, 1)
        return (result, 1.0, old[:200], "exact")

    # 2. Whitespace-normalized match.
    ws_result = _apply_ws_normalized_replace(content, old, new)
    if ws_result is not None:
        return (ws_result, 0.99, old[:200], "whitespace")

    # 3. Line-level fuzzy match with sliding window.
    old_lines = old.split("\n")
    content_lines = content.split("\n")
    n_old = len(old_lines)

    best_ratio = 0.0
    best_start = -1
    best_end = -1
    best_snippet = ""

    # Try windows of size n_old-1, n_old, n_old+1.
    for window_size in (n_old, n_old - 1, n_old + 1):
        if window_size < 1 or window_size > len(content_lines):
            continue
        for start in range(len(content_lines) - window_size + 1):
            end = start + window_size
            candidate = "\n".join(content_lines[start:end])
            ratio = difflib.SequenceMatcher(None, old, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = end
                best_snippet = candidate[:200]

    if best_ratio >= min_ratio and best_start >= 0:
        # Replace the matched window with *new*.
        result_lines = content_lines[:best_start] + new.split("\n") + content_lines[best_end:]
        return ("\n".join(result_lines), best_ratio, best_snippet, "fuzzy")

    return (None, best_ratio, best_snippet, "none")


def _get_dspy_token_usage() -> int:
    """Read cumulative token usage from the DSPy LM history.

    DSPy stores per-call history on the thread-local LM.  We read
    the total length of the history list — each entry corresponds
    to one API call.  The actual token counts are in
    ``entry['usage']`` when available.
    """
    try:
        lm = dspy.settings.lm
        if lm is None:
            return 0
        history = getattr(lm, "history", None)
        if not history:
            return 0
        total = 0
        for entry in history:
            usage = entry.get("usage", {}) if isinstance(entry, dict) else {}
            total += usage.get("total_tokens", 0)
        return total
    except Exception:
        return 0


# ── Configuration ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextFirstConfig:
    """Configuration for the Context-First Solver.

    Attributes:
        max_context_bytes: Skip planner and dump ALL files if the total
            project source is below this threshold.  200 KB ≈ 57K tokens.
        max_gather_bytes: Maximum bytes to gather (for large projects
            where the planner selects files).
        max_files: Maximum number of files to read.
        enable_planner: When False, always dump everything (fast path).
        verify_commands: Shell commands to run after operations execute.
            Each command is run in order; if any returns non-zero, the
            errors are fed back to the LLM for a fix attempt.
            Defaults to auto-detecting the project's linter.
        max_fix_attempts: Maximum number of verify → fix cycles.
    """

    max_context_bytes: int = 200_000
    max_gather_bytes: int = 400_000
    max_files: int = 50
    #: A file larger than this (or recognisably generated data) is shown as a short
    #: outline instead of whole.  0 disables the trimming.
    max_file_bytes: int = 50_000
    #: Hard cap on the assembled context (bytes); 0 = no cap.  ``for_llm`` sets it from
    #: the model's context window.
    max_prompt_bytes: int = 0
    enable_planner: bool = True
    verify_commands: tuple[str, ...] = ()
    max_fix_attempts: int = 2
    #: Past ``max_fix_attempts`` the loop keeps going only while a round improves the score
    #: (fewer failing tests), up to this many verify rounds in total.
    fix_hard_cap: int = 10
    #: Stop (or restart) after this many verify rounds in a row without a new best score.
    fix_patience: int = 3
    #: Clean-slate rewrites from the best state when the patches stall (the second loop).
    fix_restarts: int = 1
    #: Probe rounds: when a round makes no progress the model proposes up to 3 small python
    #: experiments, they run, and their real output goes into the next fix.  Off (0) by default:
    #: round 2 of the Cerebras A/B showed no gain, so it is opt-in.
    fix_probe_rounds: int = 0
    enable_edit_retry: bool = True
    enable_consistency_review: bool = True

    @classmethod
    def for_llm(cls, llm: LLMConfig | None = None) -> ContextFirstConfig:
        """A config whose context budget fits *llm*'s context window.

        ``llm`` defaults to the active role config's ``[llm]`` section (else the
        provider defaults).  Gathered context is held to :data:`PROMPT_WINDOW_FRACTION`
        of the window at the worst-case :data:`BYTES_PER_TOKEN`; the rest is left for
        the instructions, the project context and the model's own output.
        """
        if llm is None:
            from memfun_agent.role_models import get_role_config

            cfg = get_role_config()
            llm = cfg.llm if cfg is not None else LLMConfig()
        budget = context_budget_bytes(llm.context_window_effective)
        defaults = cls()
        return cls(
            max_context_bytes=min(defaults.max_context_bytes, budget),
            max_gather_bytes=min(defaults.max_gather_bytes, budget),
            max_prompt_bytes=budget,
        )


#: Worst-case bytes per token (Unicode tables tokenize near 2.4 B/token; code ~3.5).
BYTES_PER_TOKEN = 2.5
#: Share of the context window the gathered context may use.
PROMPT_WINDOW_FRACTION = 0.5


def context_budget_bytes(window_tokens: int) -> int:
    """Bytes of gathered context that safely fit a *window_tokens* context window."""
    return int(window_tokens * PROMPT_WINDOW_FRACTION * BYTES_PER_TOKEN)


# ── Context-length failures ───────────────────────────────────

#: ``method`` of a result whose model call was rejected for exceeding the context window.
CONTEXT_LENGTH_METHOD = "context_length_exceeded"
_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
)


def is_context_length_error(exc: BaseException) -> bool:
    """Whether *exc* is a provider rejection for an over-long prompt (HTTP 400)."""
    text = f"{type(exc).__name__} {exc}".lower()
    if type(exc).__name__ == "ContextWindowExceededError":  # litellm's own class
        return True
    return any(m in text for m in _CONTEXT_LENGTH_MARKERS)


# ── Result types ──────────────────────────────────────────────


@dataclass
class ContextFirstResult:
    """Result from the Context-First Solver."""

    answer: str
    reasoning: str
    ops: list[tuple[str, str, Any]]
    files_created: list[str]
    success: bool
    method: str  # "context_first_fast" or "context_first_planned"
    total_tokens: int = 0

    def to_result_dict(self) -> dict[str, Any]:
        """Convert to the same dict format ``_handle_rlm()`` returns."""
        return {
            "answer": self.answer,
            "task_type": "ask",
            "method": self.method,
            "iterations": 1,
            "trajectory_length": 0,
            "final_reasoning": self.reasoning,
            "total_tokens": self.total_tokens,
            "ops": [{"type": o[0], "target": o[1], "detail": o[2]} for o in self.ops],
            "files_created": self.files_created,
        }


@dataclass
class PlanResult:
    """Output of the ContextPlanner."""

    files_to_read: list[str]
    search_patterns: list[str]
    reasoning: str
    web_searches: list[str] = field(default_factory=list)


@dataclass
class SolveResult:
    """Output of the SingleShotSolver."""

    reasoning: str
    answer: str
    operations: list[dict[str, Any]]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Result of a post-solve consistency review."""

    has_issues: bool
    issues: list[str]
    reasoning: str


@dataclass(frozen=True, slots=True)
class EditDiagnostic:
    """Diagnostic information for a failed ``edit_file`` operation.

    Created when fuzzy matching cannot find the ``old`` text in the
    target file.  Carries enough context for a retry LLM call to
    produce a corrected edit.
    """

    path: str
    old_text: str  # truncated to 500 chars
    new_text: str  # truncated to 500 chars
    best_match_ratio: float
    best_match_snippet: str
    strategy_tried: str
    file_excerpt: str  # first 2000 chars of the file


# ── File manifest helpers ─────────────────────────────────────

# Extensions to include in the manifest (source code only).
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".env",
        ".md",
        ".txt",
        ".rst",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".graphql",
        ".proto",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".swift",
        ".r",
        ".jl",
        ".vue",
        ".svelte",
        ".astro",
        ".dockerfile",
        ".xml",
        ".csv",
    }
)

# Directories to always skip.
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        ".next",
        ".nuxt",
        ".output",
    }
)


def build_file_manifest(
    project_root: str | Path,
) -> list[tuple[str, int]]:
    """Scan *project_root* and return ``(relative_path, size)`` pairs.

    Only includes source-code files, skipping binary blobs,
    ``node_modules``, ``__pycache__``, ``.git``, etc.
    """
    root = Path(project_root).resolve()
    manifest: list[tuple[str, int]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info")]

        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            suffix = fpath.suffix.lower()

            # Dockerfile has no extension
            if fname.lower() in ("dockerfile", "makefile", "rakefile"):
                suffix = ".dockerfile"

            if suffix not in _SOURCE_EXTENSIONS:
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            # Skip very large single files (> 1 MB likely generated).
            if size > 1_000_000:
                continue

            rel = str(fpath.relative_to(root))
            manifest.append((rel, size))

    return sorted(manifest, key=lambda x: x[0])


def manifest_to_string(manifest: list[tuple[str, int]]) -> str:
    """Format a manifest as a newline-separated string for the LLM."""
    lines = [f"{path} ({size} bytes)" for path, size in manifest]
    return "\n".join(lines)


# ── Context Planner ───────────────────────────────────────────


def _normalize_list(value: Any) -> list[Any]:
    """Coerce an LLM output to a Python list (defensive parsing)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        if "\n" in text:
            return [line.strip() for line in text.split("\n") if line.strip()]
        if text:
            return [text]
    return []


class ContextPlanner(dspy.Module):
    """Decide what files need to be read to solve a task (1 LLM call)."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ContextPlanning)

    async def aplan(
        self,
        query: str,
        file_manifest: str,
        project_summary: str,
    ) -> PlanResult:
        """Plan which files to gather."""
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            file_manifest=file_manifest,
            project_summary=project_summary,
        )

        files = _normalize_list(getattr(result, "files_to_read", []))
        patterns = _normalize_list(getattr(result, "search_patterns", []))
        web_searches = _normalize_list(getattr(result, "web_searches", []))
        reasoning = str(getattr(result, "reasoning", ""))

        return PlanResult(
            files_to_read=[str(f) for f in files],
            search_patterns=[str(p) for p in patterns],
            reasoning=reasoning,
            web_searches=[str(q) for q in web_searches],
        )


# ── Context Gatherer ──────────────────────────────────────────


_GENERATED_NAME_RE = re.compile(r"(table|tables|unicode|generated|_pb2|\.min\.|vendor|data)", re.I)
_GENERATED_HEAD_RE = re.compile(
    r"(auto-?generated|generated by|do not edit|this file was generated)", re.I
)
_OUTLINE_HEAD_BYTES = 2_000
_OUTLINE_MAX_LINES = 120


def _python_outline(content: str) -> str:
    """Top-level classes/functions (and class methods) of Python *content*, one per line."""
    import ast

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"{node.lineno}: class {node.name}")
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    lines.append(f"{sub.lineno}:     def {sub.name}")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            lines.append(f"{node.lineno}: def {node.name}")
        if len(lines) >= _OUTLINE_MAX_LINES:
            lines.append("...")
            break
    return "\n".join(lines)


def trim_large_file(content: str, rel_path: str, max_file_bytes: int) -> str:
    """Replace a huge or generated file by a short outline; small files pass unchanged.

    A file is trimmed when it is over *max_file_bytes*, or when it is over a quarter of
    that and looks like generated data (unicode tables, ``# generated`` headers).  The
    outline keeps the head and the Python symbol list, and says how to read the rest.
    """
    if max_file_bytes <= 0:
        return content
    size = len(content)
    generated = size > max_file_bytes // 4 and bool(
        _GENERATED_NAME_RE.search(Path(rel_path).name) or _GENERATED_HEAD_RE.search(content[:600])
    )
    if size <= max_file_bytes and not generated:
        return content
    outline = _python_outline(content) if rel_path.endswith(".py") else ""
    head = content[:_OUTLINE_HEAD_BYTES]
    note = (
        f"... [{rel_path}: {size} bytes, {'generated data' if generated else 'too large'}; "
        "only the head and an outline are shown. Do not rewrite this file whole; "
        "read or edit just the lines you need.]"
    )
    parts = [head.rstrip(), note]
    if outline:
        parts.append("OUTLINE (line: symbol):\n" + outline)
    return "\n".join(parts)


class ContextGatherer:
    """Read files and assemble context (pure I/O, no LLM)."""

    def __init__(
        self,
        max_bytes: int = 400_000,
        max_files: int = 50,
        on_status: Callable[[str], None] | None = None,
        max_file_bytes: int = 0,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self._on_status = on_status

    def _status(self, msg: str) -> None:
        """Emit a per-file status update."""
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(msg)

    async def agather(
        self,
        files: list[str],
        project_root: str | Path,
        search_patterns: list[str] | None = None,
        web_searches: list[str] | None = None,
    ) -> str:
        """Read *files* and return assembled context string.

        Each file is prefixed with ``=== FILE: <path> ===`` so the
        LLM can identify boundaries.  Stops reading when
        ``max_bytes`` is reached.
        """
        root = Path(project_root).resolve()
        parts: list[str] = []
        total = 0
        read_count = 0

        for rel_path in files:
            if read_count >= self.max_files:
                break
            if total >= self.max_bytes:
                break

            fpath = root / rel_path
            if not fpath.is_file():
                # Try as absolute path
                fpath = Path(rel_path)
                if not fpath.is_file():
                    continue

            try:
                content = await asyncio.to_thread(fpath.read_text, "utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            content = trim_large_file(content, rel_path, self.max_file_bytes)

            # Budget check.
            if total + len(content) > self.max_bytes:
                # Include a truncated version.
                remaining = self.max_bytes - total
                content = content[:remaining] + "\n... (truncated)"

            header = f"=== FILE: {rel_path} ===\n"
            parts.append(header + content)
            total += len(content)
            read_count += 1
            self._status(f"Reading {rel_path}")

        # Search patterns — match against already-read content.
        if search_patterns:
            combined = "\n".join(parts)
            matches: list[str] = []
            for pat in search_patterns[:5]:
                try:
                    for m in re.finditer(pat, combined):
                        # Show match with surrounding context.
                        start = max(0, m.start() - 100)
                        end = min(len(combined), m.end() + 100)
                        snippet = combined[start:end]
                        matches.append(f"Pattern '{pat}' match: ...{snippet}...")
                        if len(matches) > 10:
                            break
                except re.error:
                    continue
            if matches:
                parts.append("=== SEARCH RESULTS ===\n" + "\n".join(matches[:10]))

        # Web searches — run after file reads.
        if web_searches:
            self._status("Searching the web...")
            web_results: list[str] = []
            for query in web_searches[:5]:
                self._status(f"Search Web: {query}")
                result = await self._web_search(query)
                web_results.append(f"### Search: {query}\n{result}")
            if web_results:
                parts.append("=== WEB SEARCH RESULTS ===\n" + "\n\n".join(web_results))

        return "\n\n".join(parts)

    async def _web_search(self, query: str) -> str:
        """Run a web search via ddgs and return formatted results."""
        try:
            from ddgs import DDGS
        except ImportError:
            return "[web search unavailable: ddgs not installed]"

        try:
            results = await asyncio.to_thread(lambda: DDGS().text(query, max_results=3))
            if not results:
                return f"No results for: {query}"
            parts: list[str] = []
            for r in results:
                parts.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}")
            return "\n\n".join(parts)
        except Exception as exc:
            logger.warning("Web search failed for %r: %s", query, exc)
            return f"[web search error: {exc}]"

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL and return markdown content (truncated)."""
        try:
            import httpx
            from markdownify import markdownify
        except ImportError:
            return "[web fetch unavailable: missing dependencies]"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                md: str = markdownify(resp.text)
                return md[:8000]
        except Exception as exc:
            logger.warning("Web fetch failed for %s: %s", url, exc)
            return f"[web fetch error: {exc}]"

    def read_all_files(
        self,
        manifest: list[tuple[str, int]],
        project_root: str | Path,
    ) -> str:
        """Read ALL files from *manifest* (fast path for small projects)."""
        root = Path(project_root).resolve()
        parts: list[str] = []
        total = 0

        for rel_path, _size in manifest:
            if total >= self.max_bytes:
                break

            fpath = root / rel_path
            try:
                content = fpath.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            content = trim_large_file(content, rel_path, self.max_file_bytes)
            if total + len(content) > self.max_bytes:
                break  # whole files only: drop the rest rather than cut one mid-file

            header = f"=== FILE: {rel_path} ===\n"
            parts.append(header + content)
            total += len(content)
            self._status(f"Reading {rel_path}")

        return "\n\n".join(parts)


# ── Single-Shot Solver ────────────────────────────────────────

#: Extra attempts when the solver's reply parses to no operations at all.
UNPARSEABLE_RETRIES = 1

# A retry must not be answered from DSPy's response cache (it would replay the same bad
# reply byte for byte).  ``rollout_id`` is part of DSPy's cache key and only matters when
# the temperature is non-zero.  The workflow bumps the salt before re-dispatching the task.
_ATTEMPT_SALT = 0


def set_attempt_salt(value: int) -> None:
    """Set the process-wide retry salt (0 = first attempt, no rollout id)."""
    global _ATTEMPT_SALT
    _ATTEMPT_SALT = max(0, int(value))


def _rollout_config(attempt: int) -> dict[str, Any]:
    rollout_id = _ATTEMPT_SALT * 10 + (attempt - 1)
    return {"rollout_id": rollout_id} if rollout_id else {}


class SingleShotSolver(dspy.Module):
    """Solve a coding task in one LLM call given full context."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(SingleShotSolving)

    async def asolve(
        self,
        query: str,
        full_context: str,
    ) -> SolveResult:
        """Solve the task and return structured operations.

        An unparseable reply (substantial text, 0 operations) is retried ONCE with a fresh
        sample before it is reported as truncated: at temperature > 0 the second attempt
        usually parses, and with no planner there is nothing to escalate to.
        """
        for attempt in range(1, UNPARSEABLE_RETRIES + 2):
            cfg = _rollout_config(attempt)
            result = await asyncio.to_thread(
                self.predictor,
                query=query,
                full_context=full_context,
                **({"config": cfg} if cfg else {}),
            )

            reasoning = str(getattr(result, "reasoning", ""))
            answer = str(getattr(result, "answer", ""))
            raw_ops = str(getattr(result, "operations", "[]"))

            operations = _parse_operations(raw_ops)

            # Detect truncation: if the raw operations string doesn't
            # close the JSON array, the LLM response was likely cut off.
            truncated = _detect_truncation(raw_ops, operations)
            if not truncated or operations or attempt > UNPARSEABLE_RETRIES:
                break
            logger.warning("Solver reply had no parseable operations; retrying once")

        return SolveResult(
            reasoning=reasoning,
            answer=answer,
            operations=operations,
            truncated=truncated,
        )


def _parse_operations(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON operations string from LLM output."""
    text = raw.strip()

    # Try direct JSON parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [op for op in parsed if isinstance(op, dict) and "op" in op]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON array from markdown code block.
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return [op for op in parsed if isinstance(op, dict) and "op" in op]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try line-by-line JSON objects.
    ops: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "op" in obj:
                    ops.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
    return ops


def _detect_truncation(
    raw_ops: str,
    parsed_ops: list[dict[str, Any]],
) -> bool:
    """Detect whether the LLM response was likely truncated.

    Signals:
    - The raw operations string doesn't close the JSON array (no ``]``)
    - Operations were expected (non-empty raw) but parsing yielded nothing
    - The raw string ends mid-JSON (unclosed braces/brackets)
    """
    text = raw_ops.strip()
    if not text or text == "[]":
        return False

    # Strip markdown code fences before checking endings.
    # LLMs often wrap JSON in ```json ... ``` which makes the
    # raw string end with ``` instead of ], causing false positives.
    cleaned = text
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    # If the raw text contains an opening bracket but no closing
    # one, the JSON array was cut off.
    if "[" in cleaned:
        stripped = cleaned.rstrip()
        if stripped and stripped[-1] not in ("]", "}", ")"):
            logger.warning(
                "Truncation detected: operations JSON not closed (ends with %r)",
                stripped[-20:],
            )
            return True

    # If raw text looks substantial but parsed to nothing,
    # likely a truncated JSON that couldn't be parsed.
    if len(text) > 100 and not parsed_ops:
        logger.warning(
            "Truncation detected: %d chars of operations but 0 parsed",
            len(text),
        )
        return True

    return False


# ── Operation Executor ────────────────────────────────────────


class OperationExecutor:
    """Execute file operations produced by the SingleShotSolver.

    Tracks operations in the same ``(type, target, detail)`` tuple
    format used by the RLM so the chat UI can display them.
    """

    def __init__(
        self,
        project_root: str | Path,
        on_status: Callable[[str], None] | None = None,
        *,
        edit_only: bool = False,
        base_dir: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._on_status = on_status
        self._edit_only = edit_only
        # When set, every write/edit path is resolved against base_dir
        # (relative paths) and rejected if it would escape base_dir
        # (absolute or `..`-traversing paths).  Used to confine
        # specialists to their per-task git worktree.
        self.base_dir: Path | None = Path(base_dir).resolve() if base_dir is not None else None
        self.ops: list[tuple[str, str, Any]] = []
        self.files_created: list[str] = []
        self.attempted: int = 0  # total operations attempted
        self.failed: int = 0  # operations that silently failed
        self.edit_diagnostics: list[EditDiagnostic] = []

    def _resolve_path(self, path_str: str) -> Path | None:
        """Resolve *path_str* and (when ``base_dir`` is set) confine it.

        Returns ``None`` if the path would escape ``base_dir``; the
        caller should treat that as a soft failure (increment
        ``self.failed`` and continue).  When ``base_dir`` is unset
        the legacy behaviour — ``Path(os.path.abspath(path_str))`` —
        is preserved so existing tests / callers see no change.
        """
        if self.base_dir is None:
            return Path(os.path.abspath(path_str))

        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.base_dir):
            logger.warning(
                "Refusing path outside base_dir: %s (base=%s)",
                resolved,
                self.base_dir,
            )
            self._status(f"Blocked path outside worktree: {path_str}")
            return None
        return resolved

    def _status(self, msg: str) -> None:
        """Emit a per-operation status update."""
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(msg)

    async def execute(self, operations: list[dict[str, Any]]) -> None:
        """Execute all operations sequentially."""
        for op in operations:
            op_type = op.get("op", "")
            if op_type in ("write_file", "edit_file"):
                self.attempted += 1
            try:
                if op_type == "write_file":
                    await self._write_file(op)
                elif op_type == "edit_file":
                    await self._edit_file(op)
                elif op_type == "run_cmd":
                    await self._run_cmd(op)
                elif op_type == "read_file":
                    # LLM sometimes emits read_file — skip silently.
                    pass
                else:
                    logger.warning("Unknown operation: %s", op_type)
            except Exception as exc:
                self.failed += 1
                logger.warning("Operation %s failed: %s", op_type, exc)

    async def _write_file(self, op: dict[str, Any]) -> None:
        path_str = op.get("path", "")
        content = op.get("content", "")
        if not path_str:
            self.failed += 1
            return

        path = self._resolve_path(path_str)
        if path is None:
            self.failed += 1
            return

        # edit_only mode: block write_file for existing files entirely.
        # Used by polish/fix steps that should only make targeted edits.
        if self._edit_only and path.is_file():
            logger.warning(
                "edit_only: blocked write_file on existing %s (%d chars) — use edit_file instead",
                path,
                len(content),
            )
            self._status(f"Blocked rewrite of {path.name} (edit_only mode)")
            self.failed += 1
            return

        # Guard: detect destructive overwrites of existing files.
        if path.is_file():
            try:
                existing = await asyncio.to_thread(path.read_text, "utf-8")
            except (OSError, UnicodeDecodeError):
                existing = ""

            old_lines = len(existing.splitlines())
            new_lines = len(content.splitlines())

            if old_lines > 10 and new_lines < old_lines * 0.7:
                loss_pct = (1 - new_lines / old_lines) * 100
                logger.warning(
                    "Blocked destructive write_file on %s: %d lines -> %d lines (%.0f%% lost)",
                    path,
                    old_lines,
                    new_lines,
                    loss_pct,
                )
                self._status(
                    f"Blocked rewrite of {path.name} "
                    f"({old_lines}->{new_lines} lines, "
                    f"{loss_pct:.0f}% lost)"
                )
                self.failed += 1
                self.edit_diagnostics.append(
                    EditDiagnostic(
                        path=str(path),
                        old_text=f"(entire file, {old_lines} lines)",
                        new_text=f"(rewrite attempt, {new_lines} lines)",
                        best_match_ratio=(new_lines / old_lines if old_lines else 0),
                        best_match_snippet=("write_file blocked: destructive rewrite"),
                        strategy_tried="write_file_guard",
                        file_excerpt=existing[:2000],
                    )
                )
                return

        path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(path.write_text, content, "utf-8")

        self.ops.append(("write", str(path), len(content)))
        self.files_created.append(str(path))
        logger.info("Wrote %s (%d chars)", path, len(content))
        self._status(f"Wrote {path} ({len(content)} chars)")

    async def _edit_file(self, op: dict[str, Any]) -> None:
        path_str = op.get("path", "")
        old_text = op.get("old", "")
        new_text = op.get("new", "")
        if not path_str or not old_text:
            return

        path = self._resolve_path(path_str)
        if path is None:
            self.failed += 1
            return
        if not path.is_file():
            logger.warning("edit_file: %s does not exist", path)
            self.failed += 1
            return

        content = await asyncio.to_thread(path.read_text, "utf-8")

        new_content, ratio, snippet, strategy = _fuzzy_find_and_replace(
            content, old_text, new_text
        )

        if new_content is None:
            logger.warning(
                "edit_file: no match in %s (best ratio=%.2f, strategy=%s)",
                path,
                ratio,
                strategy,
            )
            self._status(f"Edit failed: no match in {path} (best ratio={ratio:.0%})")
            self.failed += 1
            self.edit_diagnostics.append(
                EditDiagnostic(
                    path=str(path),
                    old_text=old_text[:500],
                    new_text=new_text[:500],
                    best_match_ratio=ratio,
                    best_match_snippet=snippet,
                    strategy_tried=strategy,
                    file_excerpt=content[:2000],
                )
            )
            return

        if strategy != "exact":
            logger.info(
                "edit_file: fuzzy match in %s (ratio=%.2f, strategy=%s)",
                path,
                ratio,
                strategy,
            )

        await asyncio.to_thread(path.write_text, new_content, "utf-8")

        self.ops.append(("edit", str(path), len(new_content)))
        self.files_created.append(str(path))
        logger.info("Edited %s", path)
        if strategy == "exact":
            self._status(f"Edited {path}")
        else:
            self._status(f"Edited {path} ({strategy} match, ratio={ratio:.0%})")

    async def _run_cmd(self, op: dict[str, Any]) -> None:
        import subprocess

        cmd = op.get("cmd", "")
        if not cmd:
            return

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(self.project_root),
        )

        self.ops.append(("cmd", cmd, result.returncode))
        if result.returncode != 0:
            logger.warning(
                "Command '%s' returned %d: %s",
                cmd,
                result.returncode,
                result.stderr[:500],
            )
            self._status(f"Command failed: {cmd} (exit {result.returncode})")
        else:
            self._status(f"Ran: {cmd}")


# ── Shared helpers (importable by specialists) ───────────────


async def read_affected_files(
    files: list[str],
    project_root: Path | None = None,
) -> str:
    """Re-read files that were created/modified for fix context.

    Returns a context string with ``=== FILE: path ===`` headers.
    """
    parts: list[str] = []
    for fpath_str in files:
        fpath = Path(fpath_str)
        if not fpath.is_file():
            continue
        try:
            content = await asyncio.to_thread(fpath.read_text, "utf-8")
            rel = str(fpath.relative_to(project_root)) if project_root else fpath_str
        except (OSError, ValueError):
            rel = fpath_str
            try:
                content = await asyncio.to_thread(fpath.read_text, "utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        parts.append(f"=== FILE: {rel} ===\n{content}")
    return "\n\n".join(parts)


# ── Verification ──────────────────────────────────────────────


def _detect_verify_commands(
    project_root: Path,
) -> list[str]:
    """Auto-detect verification commands for the project.

    Looks for common linter config files and returns appropriate
    commands.  Returns an empty list if nothing is detected.
    """
    cmds: list[str] = []

    # Python: ruff (fast, preferred)
    has_python = (project_root / "pyproject.toml").is_file() or any(project_root.glob("*.py"))
    has_ruff_config = (project_root / ".ruff.toml").is_file() or (
        project_root / "pyproject.toml"
    ).is_file()
    if has_python and has_ruff_config:
        cmds.append("ruff check .")

    # JavaScript/TypeScript: eslint
    has_eslint = (
        (project_root / ".eslintrc.json").is_file()
        or (project_root / ".eslintrc.js").is_file()
        or (project_root / "eslint.config.js").is_file()
    )
    if (project_root / "package.json").is_file() and has_eslint:
        cmds.append("npx eslint .")

    # Go
    if (project_root / "go.mod").is_file():
        cmds.append("go vet ./...")

    # Rust
    if (project_root / "Cargo.toml").is_file():
        cmds.append("cargo check")

    return cmds


#: A project may carry its own verification (e.g. the visible tests of the function to
#: implement): ``.memfun/verify.json`` = ``{"commands": ["..."], "max_fix_attempts": 4}``.
PROJECT_VERIFY_FILE = ".memfun/verify.json"
#: The file's commands run through a shell, so a repository must not be able to make memfun
#: execute them on its own: they are read only when this variable is truthy (the benchmark
#: runner sets it for its episodes; a user opts in explicitly).
TRUST_PROJECT_VERIFY_ENV = "MEMFUN_TRUST_PROJECT_VERIFY"


def read_project_verify(project_root: Path) -> tuple[list[str], int | None]:
    """``(commands, max_fix_attempts)`` from ``.memfun/verify.json``.

    Empty unless ``MEMFUN_TRUST_PROJECT_VERIFY`` is set, and when the file is absent/invalid.
    """
    if os.environ.get(TRUST_PROJECT_VERIFY_ENV, "").lower() not in ("1", "true", "yes"):
        return [], None
    try:
        data = json.loads((project_root / PROJECT_VERIFY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], None
    if not isinstance(data, dict):
        return [], None
    raw_cmds = data.get("commands")
    cmds = (
        [c for c in raw_cmds if isinstance(c, str) and c.strip()]
        if isinstance(raw_cmds, list)
        else []
    )
    fixes = data.get("max_fix_attempts")
    ok = isinstance(fixes, int) and not isinstance(fixes, bool) and 1 <= fixes <= 10
    return cmds, (fixes if ok else None)


@dataclass
class VerifyResult:
    """Result of running verification commands."""

    passed: bool
    errors: str  # Combined stderr/stdout from failing commands
    commands_run: list[str]


class Verifier:
    """Run verification commands (lint, type check, tests) on a project."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    async def averify(
        self,
        commands: list[str],
    ) -> VerifyResult:
        """Run *commands* and collect any errors.

        Returns a :class:`VerifyResult` with ``passed=True`` if all
        commands succeed (exit code 0), or ``passed=False`` with the
        combined error output.
        """
        import subprocess

        errors: list[str] = []
        commands_run: list[str] = []

        for cmd in commands:
            commands_run.append(cmd)
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self.project_root),
                )
                if result.returncode != 0:
                    output = ""
                    if result.stdout:
                        # head + tail: pytest prints its failure summary LAST
                        output += fixloop.clip(result.stdout, 12_000)
                    if result.stderr:
                        output += "\n" + fixloop.clip(result.stderr, 4_000)
                    errors.append(f"$ {cmd} (exit {result.returncode})\n{output.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"$ {cmd} (timed out after 120s)")
            except Exception as exc:
                errors.append(f"$ {cmd} (error: {exc})")

        return VerifyResult(
            passed=len(errors) == 0,
            errors="\n\n".join(errors),
            commands_run=commands_run,
        )


class FixSolver(dspy.Module):
    """Fix verification errors in one LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(VerificationFix)

    async def afix(
        self,
        query: str,
        full_context: str,
        verification_errors: str,
    ) -> list[dict[str, Any]]:
        """Produce fix operations for the given verification errors."""
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            full_context=full_context,
            verification_errors=verification_errors,
        )

        raw_ops = str(getattr(result, "operations", "[]"))
        return _parse_operations(raw_ops)


class ProbeSolver(dspy.Module):
    """Propose small python experiments when patching has stalled (one LLM call)."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ExperimentProposal)

    async def apropose(
        self,
        query: str,
        full_context: str,
        verification_errors: str,
        history: str,
    ) -> list[str]:
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            full_context=full_context,
            verification_errors=verification_errors,
            history=history,
        )
        return fixloop.parse_probes(str(getattr(result, "probes", "[]")))


class ConsistencyReviewer(dspy.Module):
    """Check whether executed operations actually fulfil the user request.

    Runs a single LLM call using :class:`ConsistencyReview` to compare
    intended changes against the actual file contents after execution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ConsistencyReview)

    async def areview(
        self,
        user_request: str,
        intended_changes: str,
        actual_file_contents: str,
    ) -> ConsistencyResult:
        """Run the consistency review (1 LLM call)."""
        result = await asyncio.to_thread(
            self.predictor,
            user_request=user_request,
            intended_changes=intended_changes,
            actual_file_contents=actual_file_contents,
        )

        # Coerce has_issues — DSPy may return string "True"/"False".
        raw_has = getattr(result, "has_issues", False)
        if isinstance(raw_has, str):
            has_issues = raw_has.strip().lower() in ("true", "yes", "1")
        else:
            has_issues = bool(raw_has)

        issues = _normalize_list(getattr(result, "issues", []))
        issues = [str(i) for i in issues if str(i).strip()]
        reasoning = str(getattr(result, "reasoning", ""))

        return ConsistencyResult(
            has_issues=has_issues,
            issues=issues,
            reasoning=reasoning,
        )


# ── Main Orchestrator ─────────────────────────────────────────


class ContextFirstSolver:
    """Orchestrates the context-first solving pipeline.

    Usage::

        solver = ContextFirstSolver(project_root="/path/to/project")
        result = await solver.asolve(query="Fix the bug in auth.py")
        if result.success:
            print(result.answer)
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        config: ContextFirstConfig | None = None,
        on_status: Callable[[str], None] | None = None,
        on_dashboard: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = config or ContextFirstConfig.for_llm()
        self._on_status = on_status
        self._on_dashboard = on_dashboard

        self.planner = ContextPlanner()
        self.gatherer = ContextGatherer(
            max_bytes=self.config.max_gather_bytes,
            max_files=self.config.max_files,
            on_status=on_status,
            max_file_bytes=self.config.max_file_bytes,
        )
        self.solver = SingleShotSolver()
        self.fix_solver = FixSolver()
        self.probe_solver = ProbeSolver()
        self.consistency_reviewer = ConsistencyReviewer()
        self.verifier = Verifier(self.project_root)

    def _status(self, msg: str) -> None:
        """Emit a status update."""
        logger.info(msg)
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(msg)

    def _dashboard(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        agent_name: str = "context-first-solver",
        success: bool | None = None,
        duration_ms: float | None = None,
        detail: str | None = None,
        workflow_id: str | None = None,
        project: str | None = None,
    ) -> None:
        """Emit a dashboard event (best-effort, sync callback)."""
        if self._on_dashboard is None:
            return
        with contextlib.suppress(Exception):
            self._on_dashboard(
                {
                    "event_type": event_type,
                    "task_id": task_id,
                    "agent_name": agent_name,
                    "worker_id": agent_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    "detail": detail,
                    "workflow_id": workflow_id,
                    "project": project,
                    "ts": time.time(),
                }
            )

    async def asolve(
        self,
        query: str,
        raw_query: str = "",
        context: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        *,
        category: str = "",
    ) -> ContextFirstResult:
        """Run the context-first pipeline.

        Args:
            query: The user's task/question (already enriched by
                ``_build_rlm_query``).
            raw_query: The original, un-enriched user message.
                Used for consistency review to check against the
                actual request (not the noise-laden enriched version).
                Falls back to *query* if empty.
            context: The assembled context from chat.py (project state,
                history summary, etc.).  Used for project_root discovery
                and as fallback context.
            conversation_history: Optional conversation history entries.
            category: Query category from triage (e.g. "web").
                When "web", forces the planner path so web searches
                are always planned.

        Returns:
            A :class:`ContextFirstResult`.  Check ``success`` to decide
            whether to fall back to the RLM.
        """
        from memfun_agent.llm_telemetry import (
            ContextFirstTruncatedError,
            current_role,
            truncation_advice,
            truncation_counts,
        )

        role = current_role()
        truncated_before = truncation_counts().get(role, 0)

        def _truncated() -> bool:
            return truncation_counts().get(role, 0) > truncated_before

        try:
            result = await self._solve_internal(
                query,
                context,
                conversation_history,
                category=category,
                raw_query=raw_query or query,
            )
        except Exception as exc:
            if _truncated():
                # Cut off by max_tokens: fail visibly, do not fall back.
                raise ContextFirstTruncatedError(
                    f"context-first solver LLM response truncated; {truncation_advice(role)} "
                    f"(error: {exc})"
                ) from exc
            logger.warning("Context-first solver failed: %s", exc)
            return ContextFirstResult(
                answer="",
                reasoning=f"Context-first failed: {exc}",
                ops=[],
                files_created=[],
                success=False,
                method="context_first_error",
            )
        if not result.success and _truncated():
            raise ContextFirstTruncatedError(
                f"context-first solver LLM response truncated; {truncation_advice(role)}"
            )
        return result

    async def _solve_internal(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, Any]] | None,
        *,
        category: str = "",
        raw_query: str = "",
    ) -> ContextFirstResult:
        tokens_before = _get_dspy_token_usage()
        workflow_id = uuid.uuid4().hex[:12]
        project = self.project_root.name

        # Emit solver started (acts like workflow.started for dashboard)
        self._dashboard(
            "workflow.started",
            detail=(raw_query or query)[:200],
            workflow_id=workflow_id,
            project=project,
        )
        self._dashboard(
            "worker.online",
            agent_name="context-first-solver",
            workflow_id=workflow_id,
            project=project,
        )

        # 1. Build file manifest.
        self._status("Scanning project files...")
        manifest = build_file_manifest(self.project_root)

        if not manifest and category != "web":
            logger.info("No source files found, skipping context-first")
            return ContextFirstResult(
                answer="",
                reasoning="No source files found in project",
                ops=[],
                files_created=[],
                success=False,
                method="context_first_empty",
            )

        total_size = sum(size for _, size in manifest)

        # Build code map (classes/functions/methods) instead of
        # plain path+size manifest for better planner file selection.
        from memfun_agent.code_map import build_code_map, code_map_to_string

        file_maps = build_code_map(self.project_root, manifest=manifest)
        manifest_str = code_map_to_string(file_maps, max_tokens=2000)

        # Inject conversation context as extra info for the solver.
        extra_context = ""
        if context:
            # Include the chat-assembled context (learnings, recent turn).
            extra_context = f"\n\n=== PROJECT CONTEXT ===\n{context[:12000]}"

        # 2. Decide strategy: fast path vs planner.
        #    "web" category always goes through planner so web searches
        #    are planned and executed.
        gather_id = f"gather-{workflow_id}"
        t_gather = time.monotonic()
        self._dashboard(
            "task.published",
            task_id=gather_id,
            agent_name="context-first-solver",
            detail="Gathering context",
            workflow_id=workflow_id,
            project=project,
        )
        self._dashboard(
            "task.picked_up",
            task_id=gather_id,
            agent_name="context-first-solver",
            workflow_id=workflow_id,
            project=project,
        )
        if category != "web" and (
            total_size <= self.config.max_context_bytes or not self.config.enable_planner
        ):
            # Fast path: project is small enough to dump everything.
            self._status(f"Reading all {len(manifest)} files ({total_size // 1024} KB)...")
            full_context = self.gatherer.read_all_files(manifest, self.project_root)
            full_context += extra_context
            method = "context_first_fast"
        else:
            # Large project: use planner to select files.
            self._status("Planning context gathering...")
            project_summary = (
                f"Project at {self.project_root}\n"
                f"{len(manifest)} source files, "
                f"{total_size // 1024} KB total"
            )
            plan = await self.planner.aplan(
                query=query,
                file_manifest=manifest_str,
                project_summary=project_summary,
            )
            self._status(f"Reading {len(plan.files_to_read)} selected files...")
            full_context = await self.gatherer.agather(
                files=plan.files_to_read,
                project_root=self.project_root,
                search_patterns=plan.search_patterns or None,
                web_searches=plan.web_searches or None,
            )
            full_context += extra_context
            method = "context_first_planned"
        self._dashboard(
            "task.completed",
            task_id=gather_id,
            agent_name="context-first-solver",
            success=True,
            duration_ms=(time.monotonic() - t_gather) * 1000,
            workflow_id=workflow_id,
            project=project,
        )

        cap = self.config.max_prompt_bytes
        if cap and len(full_context) > cap:
            logger.warning(
                "gathered context %d bytes > budget %d; trimming", len(full_context), cap
            )
            full_context = full_context[:cap] + "\n... (context trimmed to fit the model's window)"

        if not full_context.strip():
            return ContextFirstResult(
                answer="",
                reasoning="No context could be gathered",
                ops=[],
                files_created=[],
                success=False,
                method=method,
            )

        # 3. Single-shot solve.
        solve_id = f"solve-{workflow_id}"
        t_solve = time.monotonic()
        self._dashboard(
            "task.published",
            task_id=solve_id,
            agent_name="context-first-solver",
            detail="Solving in single shot",
            workflow_id=workflow_id,
            project=project,
        )
        self._dashboard(
            "task.picked_up",
            task_id=solve_id,
            agent_name="context-first-solver",
            workflow_id=workflow_id,
            project=project,
        )
        self._status("Solving in single shot...")
        solver_query = raw_query or query
        solve_result = await self.solver.asolve(
            query=solver_query,
            full_context=full_context,
        )
        self._dashboard(
            "task.completed",
            task_id=solve_id,
            agent_name="context-first-solver",
            success=True,
            duration_ms=(time.monotonic() - t_solve) * 1000,
            workflow_id=workflow_id,
            project=project,
        )

        # 3a. Truncation detection — if the LLM response was cut off,
        # the operations are incomplete.  Bail out so the caller can
        # escalate to the multi-agent workflow.
        if solve_result.truncated:
            self._status("Response truncated — escalating to workflow...")
            logger.warning("Solver output truncated, returning failure for workflow escalation")
            tokens_after = _get_dspy_token_usage()
            return ContextFirstResult(
                answer=solve_result.answer,
                reasoning=(
                    "Solver output was truncated (operations "
                    "incomplete). Task too complex for single-shot."
                ),
                ops=[],
                files_created=[],
                success=False,
                method="context_first_truncated",
                total_tokens=max(0, tokens_after - tokens_before),
            )

        # 4. Execute operations.
        exec_id = f"exec-{workflow_id}"
        t_exec = time.monotonic()
        executor = OperationExecutor(self.project_root, on_status=self._on_status)
        if solve_result.operations:
            self._dashboard(
                "task.published",
                task_id=exec_id,
                agent_name="context-first-solver",
                detail=f"Executing {len(solve_result.operations)} operations",
                workflow_id=workflow_id,
                project=project,
            )
            self._dashboard(
                "task.picked_up",
                task_id=exec_id,
                agent_name="context-first-solver",
                workflow_id=workflow_id,
                project=project,
            )
            self._status(f"Executing {len(solve_result.operations)} operations...")
            await executor.execute(solve_result.operations)
            self._dashboard(
                "task.completed",
                task_id=exec_id,
                agent_name="context-first-solver",
                success=True,
                duration_ms=(time.monotonic() - t_exec) * 1000,
                workflow_id=workflow_id,
                project=project,
            )

        # 4a. Retry failed edits with diagnostic feedback.
        if self.config.enable_edit_retry and executor.edit_diagnostics:
            await self._retry_failed_edits(
                query=solver_query,
                full_context=full_context,
                executor=executor,
            )

        # 4b. Consistency review & polish (semantic check).
        if self.config.enable_consistency_review and executor.files_created:
            review_id = f"review-{workflow_id}"
            t_review = time.monotonic()
            self._dashboard(
                "task.published",
                task_id=review_id,
                agent_name="context-first-solver",
                detail="Consistency review",
                workflow_id=workflow_id,
                project=project,
            )
            self._dashboard(
                "task.picked_up",
                task_id=review_id,
                agent_name="context-first-solver",
                workflow_id=workflow_id,
                project=project,
            )
            await self._consistency_review_and_polish(
                query=raw_query or query,
                solve_result=solve_result,
                executor=executor,
            )
            self._dashboard(
                "task.completed",
                task_id=review_id,
                agent_name="context-first-solver",
                success=True,
                duration_ms=(time.monotonic() - t_review) * 1000,
                workflow_id=workflow_id,
                project=project,
            )

        # 5. Verify & fix loop.
        verify_cmds = self._get_verify_commands()
        if verify_cmds and executor.files_created:
            verify_id = f"verify-{workflow_id}"
            t_verify = time.monotonic()
            self._dashboard(
                "task.published",
                task_id=verify_id,
                agent_name="context-first-solver",
                detail="Verifying",
                workflow_id=workflow_id,
                project=project,
            )
            self._dashboard(
                "task.picked_up",
                task_id=verify_id,
                agent_name="context-first-solver",
                workflow_id=workflow_id,
                project=project,
            )
            await self._verify_and_fix(
                query=solver_query,
                full_context=full_context,
                executor=executor,
                verify_cmds=verify_cmds,
            )
            self._dashboard(
                "task.completed",
                task_id=verify_id,
                agent_name="context-first-solver",
                success=True,
                duration_ms=(time.monotonic() - t_verify) * 1000,
                workflow_id=workflow_id,
                project=project,
            )

        tokens_after = _get_dspy_token_usage()
        total_tokens = max(0, tokens_after - tokens_before)

        # Determine success: if operations were attempted but ALL
        # failed (e.g. every edit_file had "old text not found"),
        # report failure so the caller can escalate.
        success = True
        if executor.attempted > 0 and executor.failed >= executor.attempted:
            logger.warning(
                "All %d file operations failed — marking as unsuccessful",
                executor.attempted,
            )
            success = False

        return ContextFirstResult(
            answer=solve_result.answer,
            reasoning=solve_result.reasoning,
            ops=executor.ops,
            files_created=executor.files_created,
            success=success,
            method=method,
            total_tokens=total_tokens,
        )

    def _get_verify_commands(self) -> list[str]:
        """Get verification commands: explicit config or auto-detected."""
        if self.config.verify_commands:
            return list(self.config.verify_commands)
        project_cmds, _ = read_project_verify(self.project_root)
        if project_cmds:
            return project_cmds  # the project's own checks (e.g. its visible tests)
        return _detect_verify_commands(self.project_root)

    def _max_fix_attempts(self) -> int:
        _, fixes = read_project_verify(self.project_root)
        return fixes or self.config.max_fix_attempts

    async def _verify_and_fix(
        self,
        query: str,
        full_context: str,
        executor: OperationExecutor,
        verify_cmds: list[str],
    ) -> None:
        """Verify, then patch while it helps; never finish worse than the best state seen.

        Each round runs the verifier and scores it (:func:`fixloop.score`).  The owned files
        are snapshotted whenever a round beats the best score and restored at the end, so a
        late bad edit cannot leave the module broken.  The fix step is shown compact failures,
        the failing tests' source and what earlier rounds tried.  Past ``max_fix_attempts`` the
        loop continues only while rounds improve (up to ``fix_hard_cap``); after
        ``fix_patience`` rounds without a new best it either restarts from the best state with
        a clean-slate rewrite (``fix_restarts``, the second loop) or stops.

        Mutates *executor* by appending fix operations and files.
        """
        base = self._max_fix_attempts()
        cap = max(base, self.config.fix_hard_cap)
        patience = max(1, self.config.fix_patience)
        restarts_left = max(0, self.config.fix_restarts)
        probes_left = max(0, self.config.fix_probe_rounds)
        experiments: list[str] = []  # real outputs of probe snippets, newest last
        paths = [Path(f) for f in dict.fromkeys(executor.files_created)]
        best = fixloop.BestSoFar(paths)
        history: list[str] = []
        seen_states: dict[str, tuple[int, tuple[int, int, int]]] = {}  # state -> (round, score)
        stagnant = 0
        sc: tuple[int, int, int] = (1, 1, 0)
        for attempt in range(1, cap + 1):
            self._status(f"Verifying (round {attempt}, base {base}, cap {cap})...")
            vr = await self.verifier.averify(verify_cmds)
            if vr.passed:
                self._status("Verification passed")
                return
            sc = fixloop.score(vr.errors, False)
            state = fixloop.state_key(fixloop.read_texts(paths))
            revisit = seen_states.get(state)
            seen_states.setdefault(state, (attempt, sc))
            if revisit is not None and attempt > 1:
                history.append(
                    f"round {attempt}: the code is IDENTICAL to round {revisit[0]} (same failures "
                    "again). That change was undone or repeated: it is a dead end, try something "
                    "structurally different."
                )
            improved = best.observe(attempt, sc)
            stagnant = 0 if improved else stagnant + 1
            summary = fixloop.summarize(vr.errors)
            logger.info(
                "Verification round %d: score %s (%d failed, %d passed%s)",
                attempt,
                sc,
                summary.failed,
                summary.passed,
                ", BROKEN" if summary.broken else "",
            )
            if attempt >= cap:
                self._status("Verification failed, round cap reached")
                break
            if stagnant >= patience:
                if restarts_left <= 0:
                    self._status("Verification failed, patching has stalled")
                    break
                restarts_left -= 1
                self._status("Patches stalled: rewriting from the best state...")
                best.restore()
                if await self._restart_from_best(
                    query, paths, executor, vr.errors, summary, history, experiments
                ):
                    stagnant = 0
                    history.append(
                        f"round {attempt}: stalled, rewrote the solution from the best state"
                    )
                    continue
                break

            if attempt >= base and not improved:
                self._status("Verification failed, no progress past the base budget")
                break

            self._status("Fixing verification errors...")
            current_context = await self._read_affected_files(executor.files_created)
            if not current_context:
                current_context = full_context
            tests_src = fixloop.failing_tests_source(self.project_root, summary.failed_ids)
            if tests_src:
                current_context += (
                    "\n\n=== FAILING TESTS (read-only; do not edit) ===\n" + tests_src
                )
            if probes_left > 0 and not improved:
                # A round without progress: stop guessing, look at real behaviour.
                probes_left -= 1
                self._status("No progress: running experiments...")
                try:
                    snippets = await self.probe_solver.apropose(
                        query=query,
                        full_context=current_context,
                        verification_errors=fixloop.compact_errors(vr.errors),
                        history="\n".join(history[-6:]),
                    )
                    logger.info("Probe round: %d experiments proposed", len(snippets))
                    if snippets:
                        experiments.append(await fixloop.run_probes(self.project_root, snippets))
                        history.append(
                            f"round {attempt}: ran {len(snippets)} experiments (results below)"
                        )
                except Exception as exc:
                    logger.warning("Probe step failed: %s", exc)
            if experiments:
                current_context += (
                    "\n\n=== EXPERIMENT RESULTS (real output of snippets run here) ===\n"
                    + "\n\n".join(experiments[-2:])
                )
            fix_query = query
            if history:
                fix_query += (
                    "\n\nPREVIOUS FIX ATTEMPTS (do not repeat what did not work):\n"
                    + "\n".join(history[-6:])
                )
            try:
                fix_ops = await self.fix_solver.afix(
                    query=fix_query,
                    full_context=current_context,
                    verification_errors=fixloop.compact_errors(vr.errors),
                )
            except Exception as exc:
                logger.warning("Fix solver failed: %s", exc)
                break

            if not fix_ops:
                logger.info("Fix solver produced no operations")
                history.append(f"round {attempt}: the fix step proposed nothing")
                stagnant += 1
                continue

            self._status(f"Applying {len(fix_ops)} fix operations...")
            before_texts = fixloop.read_texts(paths)
            # Use edit_only executor: fixes should be targeted edits, never full rewrites.
            fix_executor = OperationExecutor(
                self.project_root,
                on_status=self._on_status,
                edit_only=True,
            )
            await fix_executor.execute(fix_ops)
            executor.ops.extend(fix_executor.ops)
            executor.files_created.extend(fix_executor.files_created)
            executor.attempted += fix_executor.attempted
            executor.failed += fix_executor.failed
            for f in fix_executor.files_created:
                if Path(f) not in paths:
                    paths.append(Path(f))  # BestSoFar shares this list
            ids = ", ".join(summary.failed_ids[:4]) or "lint/other errors"
            change = fixloop.short_diff(before_texts, fixloop.read_texts(paths))
            history.append(
                f"round {attempt}: {summary.failed} failing ({ids}); applied "
                f"{fix_executor.attempted - fix_executor.failed}/{fix_executor.attempted} edits"
                + (", some edits did not match the file" if fix_executor.failed else "")
                + (f"; the change was:\n{change}" if change else "; the file did not change")
            )

        # Never finish worse than the best state this loop saw.
        if best.best_score is not None and sc > best.best_score:
            changed = best.restore()
            if changed:
                logger.info(
                    "Restored the best verified state (round %d, score %s) over the final %s",
                    best.best_round,
                    best.best_score,
                    sc,
                )
                self._status(f"Restored the best state (round {best.best_round})")

    async def _restart_from_best(
        self,
        query: str,
        paths: list[Path],
        executor: OperationExecutor,
        errors: str,
        summary: fixloop.TestSummary,
        history: list[str],
        experiments: list[str] | None = None,
    ) -> bool:
        """The second loop: ask the one-shot solver for a corrected solution given the failures.

        The files are already back at the best state.  Returns True when new operations
        were applied (the caller verifies them next; a worse result is rolled back).
        """
        current = await self._read_affected_files([str(p) for p in paths])
        tests_src = fixloop.failing_tests_source(self.project_root, summary.failed_ids)
        rewrite_query = (
            query
            + "\n\nA first solution was applied and patched several times but still fails the "
            "tests below. Study the failing tests and the errors, work out what the code must "
            "do, and produce a corrected implementation (rewrite what is wrong; you may replace "
            "the whole function).\n"
            + ("\nTRIED SO FAR:\n" + "\n".join(history[-6:]) if history else "")
            + "\n\nFAILURES:\n"
            + fixloop.compact_errors(errors)
        )
        ctx = current + (
            "\n\n=== FAILING TESTS (read-only; do not edit) ===\n" + tests_src if tests_src else ""
        )
        if experiments:
            ctx += "\n\n=== EXPERIMENT RESULTS (real output) ===\n" + "\n\n".join(experiments[-2:])
        try:
            result = await self.solver.asolve(query=rewrite_query, full_context=ctx)
        except Exception as exc:
            logger.warning("Restart solve failed: %s", exc)
            return False
        if not result.operations:
            return False
        rerun = OperationExecutor(self.project_root, on_status=self._on_status)
        await rerun.execute(result.operations)
        executor.ops.extend(rerun.ops)
        executor.files_created.extend(rerun.files_created)
        executor.attempted += rerun.attempted
        executor.failed += rerun.failed
        for f in rerun.files_created:
            if Path(f) not in paths:
                paths.append(Path(f))
        return rerun.attempted > rerun.failed

    async def _read_affected_files(self, files: list[str]) -> str:
        """Re-read files that were created/modified for fix context."""
        return await read_affected_files(files, self.project_root)

    async def _retry_failed_edits(
        self,
        query: str,
        full_context: str,
        executor: OperationExecutor,
    ) -> None:
        """Retry failed edits by feeding diagnostics to the FixSolver.

        1. Re-read affected files to get current content.
        2. Format :class:`EditDiagnostic` list as descriptive error text.
        3. Call :meth:`FixSolver.afix` (1 LLM call) with diagnostics.
        4. Execute corrected operations.
        5. Adjust ``executor.failed`` for any recovered edits.
        """
        diagnostics = executor.edit_diagnostics
        if not diagnostics:
            return

        self._status(f"Retrying {len(diagnostics)} failed edit(s) with diagnostic feedback...")

        # Re-read affected files.
        affected_paths = list({d.path for d in diagnostics})
        current_context = await self._read_affected_files(affected_paths)
        if not current_context:
            current_context = full_context

        # Format diagnostics as descriptive error text.
        diag_parts: list[str] = []
        for i, d in enumerate(diagnostics, 1):
            diag_parts.append(
                f"FAILED EDIT #{i}:\n"
                f"  File: {d.path}\n"
                f"  Attempted to find:\n"
                f"    {d.old_text}\n"
                f"  Intended replacement:\n"
                f"    {d.new_text}\n"
                f"  Best match ratio: {d.best_match_ratio:.2f} "
                f"(strategy: {d.strategy_tried})\n"
                f"  Closest snippet found:\n"
                f"    {d.best_match_snippet}\n"
                f"  File excerpt (first 2000 chars):\n"
                f"    {d.file_excerpt}"
            )
        error_text = "\n\n".join(diag_parts)

        try:
            fix_ops = await self.fix_solver.afix(
                query=query,
                full_context=current_context,
                verification_errors=(
                    "The following edit_file operations failed because "
                    "the 'old' text was not found in the file. Please "
                    "produce corrected edit_file or write_file operations "
                    "that accomplish the same changes using the actual "
                    "file content shown below.\n\n" + error_text
                ),
            )
        except Exception as exc:
            logger.warning("Edit retry fix solver failed: %s", exc)
            return

        if not fix_ops:
            logger.info("Edit retry solver produced no operations")
            return

        # Track how many edits succeed to adjust failure count.
        failed_before = executor.failed
        self._status(f"Applying {len(fix_ops)} retry operations...")
        await executor.execute(fix_ops)

        # Any operations that succeeded in this round recover
        # previously-failed edits.
        recovered = max(
            0,
            len(fix_ops) - (executor.failed - failed_before),
        )
        if recovered > 0:
            executor.failed = max(0, executor.failed - recovered)
            logger.info("Edit retry recovered %d edit(s)", recovered)

    @staticmethod
    def _build_intended_changes(
        solve_result: SolveResult,
    ) -> str:
        """Build a text summary of what the solver intended to produce."""
        parts: list[str] = []
        if solve_result.answer:
            parts.append(f"Solver answer:\n{solve_result.answer}")
        if solve_result.operations:
            ops_summary: list[str] = []
            for op in solve_result.operations:
                op_type = op.get("op", "?")
                path = op.get("path", "?")
                if op_type == "write_file":
                    size = len(op.get("content", ""))
                    ops_summary.append(f"  write_file {path} ({size} chars)")
                elif op_type == "edit_file":
                    old_preview = op.get("old", "")[:80]
                    ops_summary.append(f"  edit_file {path}: replace '{old_preview}...'")
                elif op_type == "run_cmd":
                    ops_summary.append(f"  run_cmd: {op.get('cmd', '?')}")
                else:
                    ops_summary.append(f"  {op_type} {path}")
            parts.append("Intended operations:\n" + "\n".join(ops_summary))
        return "\n\n".join(parts) if parts else "(no changes intended)"

    async def _consistency_review_and_polish(
        self,
        query: str,
        solve_result: SolveResult,
        executor: OperationExecutor,
    ) -> None:
        """Run a post-solve consistency review and polish if needed.

        1. Re-read modified files.
        2. Ask the ConsistencyReviewer (1 LLM call).
        3. If no issues → return.
        4. If issues → format as verification_errors, call FixSolver
           (1 LLM call), execute polish operations.
        """
        try:
            self._status("Reviewing consistency...")

            actual_contents = await self._read_affected_files(executor.files_created)
            if not actual_contents:
                logger.info("No file contents to review, skipping consistency")
                return

            intended = self._build_intended_changes(solve_result)

            review = await self.consistency_reviewer.areview(
                user_request=query,
                intended_changes=intended,
                actual_file_contents=actual_contents,
            )

            if not review.has_issues or not review.issues:
                self._status("Consistency check passed")
                logger.info("Consistency review: no issues found")
                return

            self._status(f"Found {len(review.issues)} consistency issues, polishing...")
            logger.info(
                "Consistency issues: %s",
                "; ".join(review.issues[:5]),
            )

            # Format issues as "verification errors" so FixSolver
            # can produce targeted operations.
            issues_text = (
                "CONSISTENCY REVIEW ISSUES:\n"
                + "\n".join(f"- {issue}" for issue in review.issues)
                + f"\n\nReviewer reasoning: {review.reasoning}"
            )

            polish_ops = await self.fix_solver.afix(
                query=query,
                full_context=actual_contents,
                verification_errors=issues_text,
            )

            if polish_ops:
                self._status(f"Applying {len(polish_ops)} polish operations...")
                # Use edit_only executor: polish must not rewrite
                # entire files — only targeted edit_file ops.
                polish_executor = OperationExecutor(
                    self.project_root,
                    on_status=self._on_status,
                    edit_only=True,
                )
                await polish_executor.execute(polish_ops)
                # Merge results back into the main executor.
                executor.ops.extend(polish_executor.ops)
                executor.files_created.extend(polish_executor.files_created)
                executor.attempted += polish_executor.attempted
                executor.failed += polish_executor.failed
            else:
                logger.info("Polish solver produced no operations")

        except Exception as exc:
            # Consistency review is best-effort — never block the pipeline.
            logger.warning("Consistency review failed (non-fatal): %s", exc)
