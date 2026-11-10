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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    #: Answers the read requests of one reply: ``(operations, round number, rounds left)``
    #: to the text that is appended to the context.
    ReadAnswerer = Callable[[list[dict[str, Any]], int, int], Awaitable[str]]

import time
import uuid

import dspy
from memfun_core.config import LLMConfig
from memfun_core.logging import get_logger

from memfun_agent import fixloop, read_requests, repo_search, repo_tests
from memfun_agent.signatures import (
    ConsistencyReview,
    ContextPlanning,
    ExperimentProposal,
    IssueReproduction,
    SingleShotSolving,
    TestFailureTriage,
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
    #: Differential probes tried on ONE stalled round while each produces new output.
    fix_probe_iterations: int = 3
    #: Re-run the failing pytest tests with full assertion output (``-vv``) and show that detail
    #: to the fix step instead of pytest's one-line truncated repr.
    fix_detail: bool = False
    #: How probe scripts are guarded: ``"project"`` (default: the project's interpreter, read-only
    #: audit hook), ``"strict"`` (``python -I -S -B``: stdlib + the project only, for benchmarks
    #: where an installed copy of the library is a leak) or ``"off"`` (no hook, for disposable
    #: sandboxes only).
    fix_probe_guard: str = "project"
    #: Search the whole repository for the names in the task when the project is too large
    #: to read whole (the planner path): the planner sees the files that define or mention
    #: them, and its own search patterns are searched in every file, not only the ones read.
    #: Opt-in (it changes which files the planner is offered and what is read):
    #: ``repo_search`` in ``.memfun/verify.json`` or this field.
    repo_search: bool = False
    #: Verify with the project's OWN tests: find how it runs them, select the tests relevant
    #: to the changed files, leave out failures that were already there, feed the rest to the
    #: fix loop (see ``repo_tests``).  Opt-in (it runs parts of the project's test suite):
    #: ``discover_tests`` in ``.memfun/verify.json`` or this field.
    verify_tests: bool = False
    #: Seconds one run of the selected tests may take.
    test_timeout: float = 300.0
    #: Before solving, have the model write a script that shows the problem, run it on the
    #: unchanged code and, when it does show the problem, use it as a check after the change
    #: (``reproduce`` in ``.memfun/verify.json``).  Guarded like a probe (``fix_probe_guard``).
    reproduce: bool = False
    #: Rounds in which the solver may ask to read more before it edits (lines of a file, a
    #: search, a listing): when a reply holds only read requests they are answered by memfun's
    #: own code (``read_requests``: no shell, confined to the project) and the solver is asked
    #: again.  Opt-in (it changes the solver's prompt): ``read_rounds`` in
    #: ``.memfun/verify.json`` or this field.  With it on, a solve that edits no file is a
    #: failure (``context_first_no_edits``), never a success.
    read_rounds: int = 0
    #: With the read step on: tell the solver up front that it may ask to read (and how).
    #: ``False`` answers read requests all the same but does not invite them, so a solver
    #: that can edit from what it is shown gets the prompt it always got (``read_offer``).
    read_offer: bool = True
    #: No web step: the planner is not offered web searches and nothing is fetched.  For runs
    #: that must not touch the network (``offline`` in ``.memfun/verify.json``).
    offline: bool = False
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
    #: The raw answer when it could not be parsed into the fields above (else empty).
    unparsed: str = ""


@dataclass
class SolveResult:
    """Output of the SingleShotSolver."""

    reasoning: str
    answer: str
    operations: list[dict[str, Any]]
    truncated: bool = False
    #: Rounds of reads that were answered before this reply (the read step).
    read_rounds: int = 0


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
        # memfun's own state: credentials, verify.json, probe logs, memory. Never prompt content.
        ".memfun",
        ".ssh",
        ".aws",
        ".gnupg",
    }
)

#: Source-code suffixes: a module named ``credentials.py`` or ``secrets.py`` is code, not a secret.
_CODE_SUFFIXES = frozenset(
    {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".h",
        ".hpp", ".rb", ".php", ".swift", ".r", ".jl", ".vue", ".svelte", ".astro", ".sql",
        ".sh", ".bash", ".zsh", ".graphql", ".proto", ".html", ".css", ".scss",
    }
)  # fmt: skip

#: Data/config files that hold secrets, by name: never read into a prompt.
_SECRET_FILE = re.compile(
    r"(^\.env(\..*)?$|\.env$|^credentials?(\..*)?$|^secrets?(\..*)?$|\.pem$|\.key$|"
    r"^id_(rsa|dsa|ecdsa|ed25519)|^\.netrc$|^\.npmrc$|^\.pypirc$)",
    re.IGNORECASE,
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
            if _SECRET_FILE.search(fname) and suffix not in _CODE_SUFFIXES:
                continue  # data/config with a secret-looking name; source modules stay visible

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


_PATHISH = re.compile(r"[\w./\-]+")


def salvage_planned_files(raw: str, known_files: list[str] | None, limit: int = 50) -> list[str]:
    """The files a planner answer names when the answer itself could not be parsed.

    A model sometimes answers with a bare JSON list of paths, or with prose, instead of the
    fields it was asked for.  The paths are still there: a JSON list (or an object with
    ``files_to_read``) is taken as written, and otherwise every project file the text names
    is taken in the order it appears.  With *known_files* only real project files count.
    """
    known = set(known_files) if known_files is not None else None
    found: list[str] = []

    def add(item: object) -> None:
        text = item.strip().removeprefix("./") if isinstance(item, str) else ""
        if known is not None and text not in known:
            text = text.rstrip(".,;:")  # a path at the end of a sentence
        if text and text not in found and (known is None or text in known):
            found.append(text)

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for candidate in (text, fenced.group(1) if fenced else ""):
        if not candidate.startswith(("[", "{")):
            continue
        try:
            parsed: object = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            parsed = cast("dict[str, object]", parsed).get("files_to_read")
        if isinstance(parsed, list):
            for item in cast("list[object]", parsed):
                add(item)
    if not found and known is not None:
        for m in _PATHISH.finditer(text):
            add(m.group(0))
    return found[:limit]


class ContextPlanner(dspy.Module):
    """Decide what files need to be read to solve a task (1 LLM call)."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ContextPlanning)
        self._offline_predictor: dspy.Predict | None = None

    async def aplan(
        self,
        query: str,
        file_manifest: str,
        project_summary: str,
        *,
        offline: bool = False,
    ) -> PlanResult:
        """Plan which files to gather.

        With *offline* the planner is not offered the web step at all (the signature has no
        ``web_searches`` field).  An answer that cannot be parsed does not end the solve: it
        comes back as ``PlanResult.unparsed`` and the caller takes the files it names
        (:func:`salvage_planned_files`) or goes on with what its own search found.
        """
        predictor = self.predictor
        if offline:
            if self._offline_predictor is None:
                self._offline_predictor = dspy.Predict(ContextPlanning.delete("web_searches"))
            predictor = self._offline_predictor
        # Non-empty only on the workflow's retry of the whole task: a fresh sample then,
        # instead of the cached answer of the attempt that just failed.
        cfg = _rollout_config(1)
        try:
            result = await asyncio.to_thread(
                predictor,
                query=query,
                file_manifest=file_manifest,
                project_summary=project_summary,
                **({"config": cfg} if cfg else {}),
            )
        except Exception as exc:
            raw = getattr(exc, "lm_response", None)
            if not isinstance(raw, str):
                raise  # not an unparseable answer (no model, a timeout): nothing to salvage
            return PlanResult(
                files_to_read=[],
                search_patterns=[],
                reasoning="(the planner's answer could not be parsed)",
                web_searches=[],
                unparsed=raw or " ",
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


def trim_large_file(
    content: str, rel_path: str, max_file_bytes: int, focus: list[str] | None = None
) -> str:
    """Replace a huge or generated file by a short outline; small files pass unchanged.

    A file is trimmed when it is over *max_file_bytes*, or when it is over a quarter of
    that and looks like generated data (unicode tables, ``# generated`` headers).  The
    outline keeps the head and the Python symbol list, and says how to read the rest.
    With *focus* (names the task is about) the full source of the functions and classes
    that define or mention them is added, so the part that matters is not lost.
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
    if focus and rel_path.endswith(".py") and not generated:
        excerpt = repo_search.focused_excerpt(
            content, focus, budget=max(4_000, max_file_bytes // 2)
        )
        if excerpt:
            parts.append(
                "DEFINITIONS THAT MATCH THE TASK (complete source; edit them with edit_file "
                "using this exact text):\n" + excerpt
            )
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
        #: Names the task is about: a file too large to show whole keeps their definitions.
        self.focus_terms: list[str] = []

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
            content = trim_large_file(content, rel_path, self.max_file_bytes, self.focus_terms)

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
#: Files read when the planner's answer names none: the best of a search for the task's names.
PLAN_FALLBACK_FILES = 10
#: ``ContextFirstResult.method`` of a run that was there to change code and edited no file.
NO_EDITS_METHOD = "context_first_no_edits"

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


#: Appended to the task on the retry after a reply was cut off before its operations.
TRUNCATION_NUDGE = (
    "\n\nIMPORTANT: your previous reply was cut off before the operations. Keep the reasoning "
    "to at most 8 short lines, then write the operations JSON immediately. Do not think aloud "
    "at length."
)
#: Appended on the retry after a reply held only commands and no edit.
NO_EDIT_NUDGE = (
    "\n\nIMPORTANT: your previous reply contained no edit_file or write_file operation, so no "
    "file changed and the task is not done. Reply with the edit_file operation(s) that "
    "implement the task in the file(s) named above. Commands alone do not count."
)


#: Operations that only read (the read step answers them; none of them changes a file).
READ_ONLY_OPS = frozenset(
    {
        "run_cmd", "read_file", "read", "read_lines", "view", "cat", "search", "grep",
        "search_files", "search_code", "rg", "find_files", "find", "glob", "find_file",
        "list_dir", "ls", "list_files", "list_directory", "list",
    }
)  # fmt: skip


def _has_file_change(operations: list[dict[str, Any]], *, reads: bool = False) -> bool:
    """False only when every operation is a pure command or read (nothing would change).
    With *reads* (the read step is on) only an edit or a write is a change: a reply without
    one is answered as reads, whatever else it holds, and nothing in it is ever run."""
    if reads:
        return any(str(o.get("op")) in ("edit_file", "write_file") for o in operations)
    return any(str(o.get("op")) not in ("run_cmd", "read_file") for o in operations)


#: Shown once at the end of the context when the read step is on.
READ_OFFER = (
    "\n\n=== READING MORE BEFORE YOU EDIT ===\n"
    "A file that is too large is shown above only as its head, an outline and the definitions "
    "that match the task. If the code you have to change, or have to see to change it "
    "correctly, is not shown, do not guess its text: reply with read operations ONLY (no "
    "edit_file), in the usual reply format with a one-line answer. You are then shown what "
    "you asked for and asked again, "
    "at most {rounds} time(s), so ask for everything you need at once. The read operations:\n"
    '{{"op":"read_file","path":"pkg/mod.py","start_line":120,"end_line":260}}  lines of a file '
    "(leave the line numbers out for a whole small file)\n"
    '{{"op":"search","pattern":"def name|ClassName","path":"pkg"}}  a regular expression, '
    "searched in every file under the path\n"
    '{{"op":"find_files","name":"*part_of_a_name*","path":"."}}  files by name\n'
    '{{"op":"list_dir","path":"pkg"}}  the entries of a directory\n'
    "Nothing is executed in this step: a command that is not a read is refused. When the code "
    'is shown, reply with the edit_file operation(s); the "old" text of an edit must be copied '
    "exactly from text you were shown."
)
#: After a round of reads, while more rounds are possible.
READ_MORE = (
    "Now reply with the edit_file operation(s) that resolve the task. Only if something you "
    "need is still not shown, ask for more reads instead ({left} round(s) left)."
)
#: After the last round of reads.
READ_LAST = (
    "This was the last round of reads: further read requests are not answered. Reply now with "
    "the edit_file operation(s) that resolve the task, using the text shown above."
)


class SingleShotSolver(dspy.Module):
    """Solve a coding task in one LLM call given full context."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(SingleShotSolving)

    async def asolve(
        self,
        query: str,
        full_context: str,
        *,
        reader: ReadAnswerer | None = None,
        read_rounds: int = 0,
    ) -> SolveResult:
        """Solve the task and return structured operations.

        An unparseable reply (substantial text, 0 operations) is retried ONCE with a fresh
        sample before it is reported as truncated: at temperature > 0 the second attempt
        usually parses, and with no planner there is nothing to escalate to.

        With a *reader* and *read_rounds* > 0, a reply that holds only read requests is
        answered (the reader's text is appended to the context) and the solver is asked
        again, at most *read_rounds* times; these rounds are not retries.  The task text
        stays the same and the context only grows at its end.
        """
        ask = query
        attempt = 1
        reads_left = read_rounds if reader is not None else 0
        rounds_used = 0
        while True:
            cfg = _rollout_config(attempt)
            history = _lm_history()
            marker = history[-1] if history else None
            try:
                result = await asyncio.to_thread(
                    self.predictor,
                    query=ask,
                    full_context=full_context,
                    **({"config": cfg} if cfg else {}),
                )
            except Exception as exc:
                # A reply that could not be split into its fields (a field left out, the
                # fields written as one JSON object, tool-call markup around the operations)
                # still says what to do.  Its operations are read from the raw replies of
                # this call; with none found it is an unparseable reply like any other (one
                # retry), and only then the failure it always was.
                error_text = getattr(exc, "lm_response", None)
                replies = _replies_since(marker)
                if isinstance(error_text, str) and error_text.strip():
                    replies.append(error_text)
                if not replies or attempt > UNPARSEABLE_RETRIES + 1:
                    raise
                salvaged = next((o for o in map(salvage_operations, replies) if o), [])
                if not salvaged and attempt > UNPARSEABLE_RETRIES:
                    raise
                logger.warning(
                    "Solver reply could not be parsed into its fields; %d operation(s) read "
                    "from its text",
                    len(salvaged),
                )
                result = None
            if result is None:
                reasoning, answer = "", ""
                raw_ops = json.dumps(salvaged) if salvaged else replies[-1]
                operations, truncated = salvaged, not salvaged
            else:
                reasoning = str(getattr(result, "reasoning", ""))
                answer = str(getattr(result, "answer", ""))
                raw_ops = str(getattr(result, "operations", "[]"))

                operations = _parse_operations(raw_ops)

                # Detect truncation: if the raw operations string doesn't
                # close the JSON array, the LLM response was likely cut off.
                truncated = _detect_truncation(raw_ops, operations)
            only_reads = bool(operations) and not _has_file_change(
                operations, reads=reader is not None
            )
            if only_reads and reader is not None and reads_left > 0:
                rounds_used += 1
                reads_left -= 1
                full_context += await reader(operations, rounds_used, reads_left)
                continue
            if attempt > UNPARSEABLE_RETRIES:
                break
            if not operations and looks_like_tool_call(raw_ops, answer, reasoning):
                # Tool calls written as text: this step has no tools, so nothing happened.
                logger.warning("Solver reply was written as tool calls; retrying once")
                ask = query + TOOL_CALL_NUDGE
                attempt += 1
                continue
            if truncated and not operations:
                # Cut off before the operations (the reasoning ate the budget): a plain
                # re-sample tends to ramble again, so say what to do differently.
                logger.warning("Solver reply was cut off before its operations; retrying once")
                ask = query + TRUNCATION_NUDGE
                attempt += 1
                continue
            if only_reads:
                # Only commands, no edit: nothing would change.  Ask for the edit.
                logger.warning("Solver reply had no edit/write operation; retrying once")
                ask = query + NO_EDIT_NUDGE
                attempt += 1
                continue
            break

        if reader is not None and only_reads:
            # Reads are answered, never executed: a final reply of reads alone does nothing.
            operations = []
        return SolveResult(
            reasoning=reasoning,
            answer=answer,
            operations=operations,
            truncated=truncated,
            read_rounds=rounds_used,
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


def _lm_history() -> list[Any]:
    """The call history of the active DSPy language model (empty when there is none)."""
    try:
        lm = dspy.settings.lm
        history = getattr(lm, "history", None) if lm is not None else None
    except Exception:
        return []
    return cast("list[Any]", history) if isinstance(history, list) else []


def _replies_since(marker: object) -> list[str]:
    """The raw replies of the model calls made after the history entry *marker* (at most the
    last four): what the model really wrote, before an adapter cut or rejected it."""
    out: list[str] = []
    for entry in reversed(_lm_history()[-4:]):
        if entry is marker:
            break
        outputs = cast("dict[str, Any]", entry).get("outputs") if isinstance(entry, dict) else None
        first: object = cast("list[object]", outputs)[0] if outputs else ""
        text = cast("dict[str, Any]", first).get("text") if isinstance(first, dict) else first
        if isinstance(text, str) and text.strip():
            out.append(text)
    return list(reversed(out))


_TOOL_FUNCTION = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", re.DOTALL)
_TOOL_PARAMETER = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
#: Characters of a reply that are searched for operations.
_SALVAGE_CHARS = 200_000
#: Operation names a tool call written as text may stand for.
_KNOWN_OPS = READ_ONLY_OPS | {"edit_file", "write_file"}


def _operations_from_tool_calls(raw: str) -> list[dict[str, Any]]:
    """Tool calls written as text (``<function=search><parameter=pattern>x</parameter>``, or
    ``{"name": "search", "arguments": {...}}``) as operations, when they name one."""
    ops: list[dict[str, Any]] = []
    for name, body in _TOOL_FUNCTION.findall(raw):
        if name in _KNOWN_OPS:
            params = {k: v.strip() for k, v in _TOOL_PARAMETER.findall(body)}
            ops.append({"op": name, **params})
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{\s*\"name\"", raw):
        try:
            obj, _ = decoder.raw_decode(raw, m.start())
        except ValueError:
            continue
        call = cast("dict[str, Any]", obj) if isinstance(obj, dict) else {}
        args = call.get("arguments")
        if call.get("name") in _KNOWN_OPS and isinstance(args, dict):
            ops.append({"op": call["name"], **cast("dict[str, Any]", args)})
    return ops


def salvage_operations(raw: str) -> list[dict[str, Any]]:
    """The operations in a reply that could not be parsed into its fields: a JSON array
    anywhere in the text (the last one that holds operations), else operation objects written
    one after another, else tool calls written as text that name an operation."""
    raw = raw[:_SALVAGE_CHARS]
    ops = _parse_operations(raw)
    if ops:
        return ops
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[\s*\{", raw):
        try:
            parsed, _ = decoder.raw_decode(raw, m.start())
        except ValueError:
            continue
        if isinstance(parsed, list):
            found = [
                cast("dict[str, Any]", op)
                for op in cast("list[object]", parsed)
                if isinstance(op, dict) and "op" in op
            ]
            ops = found or ops
    if ops:
        return ops
    for m in re.finditer(r"\{\s*\"op\"", raw):
        try:
            obj, _ = decoder.raw_decode(raw, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "op" in obj:
            ops.append(cast("dict[str, Any]", obj))
    return ops or _operations_from_tool_calls(raw)


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


def repair_foreign_path(path_str: str, project_root: Path, *, for_write: bool) -> Path | None:
    """Map an absolute path that does not exist onto the project, or return ``None``.

    A model sometimes answers with paths of a place it imagines (``/workspace/pkg/mod.py``)
    instead of the project it was shown.  An absolute path that exists (or, for a new file,
    whose directory exists) is taken as given.  Otherwise the longest tail of the path that
    names an existing project file (for a new file: an existing project directory) is used,
    and when there is none the operation is refused: memfun never creates directories outside
    the project for a path nobody has.
    """
    path = Path(path_str)
    if path.is_file() or (for_write and path.parent.is_dir()):
        return Path(os.path.abspath(path_str))
    parts = path.parts[1:]
    for i in range(1, len(parts)):
        candidate = project_root.joinpath(*parts[i:])
        if candidate.is_file() or (for_write and candidate.parent.is_dir() and i < len(parts) - 1):
            return candidate
    return None


_CD_ABSOLUTE = re.compile(r"^\s*cd\s+(/[^\s;&|]*)")
#: Markup of a tool call written as text: the solver steps have no tools, so such a reply is
#: not an answer (seen with chat models that fall into their tool-calling habit).
_TOOL_CALL_TEXT = re.compile(
    r"<tool_call>|</tool_call>|<function=|<parameter=|<function_calls>|<invoke\s|\[TOOL_CALLS\]"
)
TOOL_CALL_NUDGE = (
    "\n\nIMPORTANT: your previous reply was written as tool calls. You have NO tools in this "
    "step and nothing will run them. Work only from the files shown above; their paths are "
    "relative to the project root. Reply with the operations JSON and nothing else after it."
)


def looks_like_tool_call(*texts: str) -> bool:
    """Whether any of *texts* is written as a tool call instead of an answer."""
    return any(_TOOL_CALL_TEXT.search(t or "") for t in texts)


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
        originals: dict[Path, str | None] | None = None,
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
        #: The content of every file as it was before this executor (or one sharing the dict)
        #: first changed it; ``None`` = the file did not exist.
        self.originals: dict[Path, str | None] = {} if originals is None else originals

    def _resolve_path(self, path_str: str, *, for_write: bool = False) -> Path | None:
        """Resolve *path_str* and (when ``base_dir`` is set) confine it.

        Returns ``None`` if the path would escape ``base_dir``; the
        caller should treat that as a soft failure (increment
        ``self.failed`` and continue).  When ``base_dir`` is unset a
        relative path resolves against the working directory as before;
        an absolute path that does not exist is mapped onto the project
        or refused (:func:`repair_foreign_path`).
        """
        if self.base_dir is None:
            if os.path.isabs(path_str):
                fixed = repair_foreign_path(path_str, self.project_root, for_write=for_write)
                if fixed is None:
                    logger.warning("Refusing a path that is not in the project: %s", path_str)
                    self._status(f"Blocked path outside the project: {path_str}")
                elif str(fixed) != os.path.abspath(path_str):
                    logger.info("Mapped %s onto the project as %s", path_str, fixed)
                return self._not_own_state(fixed, path_str)
            return self._not_own_state(Path(os.path.abspath(path_str)), path_str)

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

    def _not_own_state(self, path: Path | None, path_str: str) -> Path | None:
        """*path*, unless it is inside the project's ``.memfun`` directory: memfun's own state
        (settings, the reproduction script, logs) is never something a task edits."""
        if path is not None and path.is_relative_to(self.project_root / ".memfun"):
            logger.warning("Refusing to change memfun's own state: %s", path_str)
            self._status(f"Blocked change to memfun's own files: {path_str}")
            return None
        return path

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

        path = self._resolve_path(path_str, for_write=True)
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

        if path not in self.originals:
            with contextlib.suppress(OSError, UnicodeDecodeError):
                self.originals[path] = path.read_text("utf-8") if path.is_file() else None
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

        if new_content == content:
            # An edit whose result is the text that was already there changes nothing: it is
            # a failed edit (the model is told and may send a real one), not a file "edited"
            # that then yields an empty patch reported as a success.
            logger.warning("edit_file: the edit changes nothing in %s", path)
            self._status(f"Edit failed: it changes nothing in {path}")
            self.failed += 1
            self.edit_diagnostics.append(
                EditDiagnostic(
                    path=str(path),
                    old_text=old_text[:500],
                    new_text=new_text[:500],
                    best_match_ratio=ratio,
                    best_match_snippet=snippet,
                    strategy_tried="no-op (the new text is the same as the old text)",
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

        self.originals.setdefault(path, content)
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
        m = _CD_ABSOLUTE.match(cmd)
        if m and not Path(m.group(1)).is_dir():
            # A command for a place that does not exist (an imagined /workspace): not run.
            logger.warning("Skipped a command for a directory that does not exist: %s", cmd[:200])
            self._status(f"Skipped command (no such directory {m.group(1)})")
            self.ops.append(("cmd", cmd, "skipped: no such directory"))
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
    import shutil

    def installed(tool: str) -> bool:
        # A linter that is not installed reports nothing about the code: asking for it only
        # sends "command not found" into the fix loop.
        return shutil.which(tool) is not None

    # Python: ruff (fast, preferred)
    has_python = (project_root / "pyproject.toml").is_file() or any(project_root.glob("*.py"))
    has_ruff_config = (project_root / ".ruff.toml").is_file() or (
        project_root / "pyproject.toml"
    ).is_file()
    if has_python and has_ruff_config and installed("ruff"):
        cmds.append("ruff check .")

    # JavaScript/TypeScript: eslint
    has_eslint = (
        (project_root / ".eslintrc.json").is_file()
        or (project_root / ".eslintrc.js").is_file()
        or (project_root / "eslint.config.js").is_file()
    )
    if (project_root / "package.json").is_file() and has_eslint and installed("npx"):
        cmds.append("npx eslint .")

    # Go
    if (project_root / "go.mod").is_file() and installed("go"):
        cmds.append("go vet ./...")

    # Rust
    if (project_root / "Cargo.toml").is_file() and installed("cargo"):
        cmds.append("cargo check")

    return cmds


#: A project may carry its own verification (e.g. the visible tests of the function to
#: implement): ``.memfun/verify.json`` = ``{"commands": ["..."], "max_fix_attempts": 4}``.
#: Upper bound for ``max_fix_attempts`` in ``.memfun/verify.json``.
MAX_FIX_ATTEMPTS = 50
PROJECT_VERIFY_FILE = ".memfun/verify.json"
#: Upper bound for ``probe_rounds`` in ``.memfun/verify.json``.
MAX_PROBE_ROUNDS = 20
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
    ok = isinstance(fixes, int) and not isinstance(fixes, bool) and 1 <= fixes <= MAX_FIX_ATTEMPTS
    if fixes is not None and not ok:
        # Silently ignoring it once made a "12 rounds" experiment run with the default of 2.
        logger.warning(
            "ignoring max_fix_attempts=%r in %s (must be an integer 1..%d)",
            fixes,
            PROJECT_VERIFY_FILE,
            MAX_FIX_ATTEMPTS,
        )
    return cmds, (fixes if ok else None)


def read_project_verify_options(project_root: Path) -> dict[str, Any]:
    """The switches of ``.memfun/verify.json`` (same trust rule as the commands):
    ``fix_detail`` (bool: re-run failing pytest tests with full assertion output),
    ``probe_rounds`` (int >= 0: stalled rounds on which to run differential probes),
    ``probe_guard``, ``discover_tests`` (bool: verify with the project's own relevant tests),
    ``reproduce`` (bool: a reproduction script before the change, a check after it),
    ``repo_search`` (bool: plan a large project from a search of every file),
    ``read_rounds`` (int >= 0: rounds in which the solver may ask to read more before it
    edits), ``read_offer`` (bool: say so in the prompt), ``offline`` (bool: no web step) and
    ``test_timeout`` (seconds for one run of the selected tests)."""
    if os.environ.get(TRUST_PROJECT_VERIFY_ENV, "").lower() not in ("1", "true", "yes"):
        return {}
    try:
        data = json.loads((project_root / PROJECT_VERIFY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        if "fix_detail" in data:
            if isinstance(data["fix_detail"], bool):
                out["fix_detail"] = data["fix_detail"]
            else:
                logger.warning(
                    "ignoring fix_detail=%r in %s (must be true or false)",
                    data["fix_detail"],
                    PROJECT_VERIFY_FILE,
                )
        if "probe_guard" in data:
            if data["probe_guard"] in fixloop.PROBE_GUARDS:
                out["probe_guard"] = data["probe_guard"]
            else:
                logger.warning(
                    "ignoring probe_guard=%r in %s (must be one of %s); the default guard stays",
                    data["probe_guard"],
                    PROJECT_VERIFY_FILE,
                    ", ".join(fixloop.PROBE_GUARDS),
                )
        for key in ("discover_tests", "reproduce", "repo_search", "offline", "read_offer"):
            if key in data:
                if isinstance(data[key], bool):
                    out[key] = data[key]
                else:
                    logger.warning(
                        "ignoring %s=%r in %s (must be true or false)",
                        key,
                        data[key],
                        PROJECT_VERIFY_FILE,
                    )
        if "test_timeout" in data:
            tt = data["test_timeout"]
            if isinstance(tt, int) and not isinstance(tt, bool) and 10 <= tt <= 3600:
                out["test_timeout"] = tt
            else:
                logger.warning(
                    "ignoring test_timeout=%r in %s (must be an integer 10..3600 seconds)",
                    tt,
                    PROJECT_VERIFY_FILE,
                )
        if "read_rounds" in data:
            rr = data["read_rounds"]
            if (
                isinstance(rr, int)
                and not isinstance(rr, bool)
                and 0 <= rr <= read_requests.MAX_READ_ROUNDS
            ):
                out["read_rounds"] = rr
            else:
                logger.warning(
                    "ignoring read_rounds=%r in %s (must be an integer 0..%d); the read step "
                    "stays OFF",
                    rr,
                    PROJECT_VERIFY_FILE,
                    read_requests.MAX_READ_ROUNDS,
                )
        if "probe_rounds" in data:
            pr = data["probe_rounds"]
            if isinstance(pr, int) and not isinstance(pr, bool) and 0 <= pr <= MAX_PROBE_ROUNDS:
                out["probe_rounds"] = pr
            else:
                logger.warning(
                    "ignoring probe_rounds=%r in %s (must be an integer 0..%d); probes stay OFF",
                    pr,
                    PROJECT_VERIFY_FILE,
                    MAX_PROBE_ROUNDS,
                )
    return out


@dataclass
class VerifyResult:
    """Result of running verification commands."""

    passed: bool
    errors: str  # Combined stderr/stdout from failing commands
    commands_run: list[str]
    #: Commands that could not start at all (the program is not installed): no evidence
    #: about the code either way, so they are neither a pass nor a failure.
    skipped: list[str] = field(default_factory=lambda: [])
    #: The error text of each failing command, in order (``errors`` is these joined).
    blocks: list[str] = field(default_factory=lambda: [])
    #: The failing commands themselves, parallel to ``blocks``.
    failed_cmds: list[str] = field(default_factory=lambda: [])


#: Seconds one verification command may run (the whole process group is killed after that).
VERIFY_TIMEOUT_S = 120.0
_NOT_FOUND = re.compile(r"not found|No such file or directory", re.I)


class Verifier:
    """Run verification commands (lint, type check, tests) on a project."""

    def __init__(self, project_root: str | Path, timeout: float = VERIFY_TIMEOUT_S) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        #: Shown instead of the command line in the error text (for long generated commands).
        self.labels: dict[str, str] = {}
        #: Exit codes other than 0 that are not a failure, per command (pytest ends with 5
        #: when every selected test was left out: nothing ran, nothing failed).
        self.ok_exits: dict[str, tuple[int, ...]] = {}

    async def averify(
        self,
        commands: list[str],
    ) -> VerifyResult:
        """Run *commands* and collect any errors.

        Returns a :class:`VerifyResult` with ``passed=True`` if no command failed, or
        ``passed=False`` with the combined error output.  Each command runs in its own
        process group with a time limit, so a hung test run cannot outlive it.  A command
        whose program is not installed (exit 127, "not found") says nothing about the code:
        it is listed in ``skipped`` and is not an error.
        """
        import subprocess

        errors: list[str] = []
        failed_cmds: list[str] = []
        commands_run: list[str] = []
        skipped: list[str] = []
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

        for cmd in commands:
            commands_run.append(cmd)
            before = len(errors)
            try:
                result = await asyncio.to_thread(
                    fixloop._run_in_group,  # pyright: ignore[reportPrivateUsage]
                    ["/bin/sh", "-c", cmd],
                    str(self.project_root),
                    env,
                    self.timeout,
                )
                if result.returncode == 127 and _NOT_FOUND.search(result.stderr or ""):
                    logger.warning(
                        "Verification command could not start (not installed), skipped: %s", cmd
                    )
                    skipped.append(cmd)
                    continue
                if result.returncode != 0 and result.returncode not in self.ok_exits.get(cmd, ()):
                    output = ""
                    if result.stdout:
                        # head + tail: pytest prints its failure summary LAST
                        output += fixloop.clip(result.stdout, 12_000)
                    if result.stderr:
                        output += "\n" + fixloop.clip(result.stderr, 4_000)
                    shown = self.labels.get(cmd, cmd)
                    errors.append(f"$ {shown} (exit {result.returncode})\n{output.strip()}")
            except subprocess.TimeoutExpired:
                shown = self.labels.get(cmd, cmd)
                errors.append(f"$ {shown} (timed out after {self.timeout:.0f}s)")
            except Exception as exc:
                errors.append(f"$ {self.labels.get(cmd, cmd)} (error: {exc})")
            if len(errors) > before:
                failed_cmds.append(cmd)

        return VerifyResult(
            passed=len(errors) == 0,
            errors="\n\n".join(errors),
            commands_run=commands_run,
            skipped=skipped,
            blocks=errors,
            failed_cmds=failed_cmds,
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
        """Produce fix operations for the given verification errors.

        A reply written as tool calls (the step has no tools), or one the adapter cannot
        parse at all, is not an answer: the model is asked once more, told so.
        """
        ask = query
        for attempt in (1, 2):
            try:
                result = await asyncio.to_thread(
                    self.predictor,
                    query=ask,
                    full_context=full_context,
                    verification_errors=verification_errors,
                )
            except Exception as exc:
                unparseable = any(
                    c.__name__ in ("AdapterParseError", "JSONDecodeError")
                    for c in type(exc).__mro__
                )
                if attempt == 1 and unparseable:
                    logger.warning("Fix reply could not be parsed (%s); asking once more", exc)
                    ask = query + TOOL_CALL_NUDGE
                    continue
                raise
            raw_ops = str(getattr(result, "operations", "[]"))
            ops = _parse_operations(raw_ops)
            if ops or attempt == 2:
                return ops
            if not looks_like_tool_call(raw_ops, str(getattr(result, "reasoning", ""))):
                return ops
            logger.warning("Fix reply was written as tool calls; asking once more")
            ask = query + TOOL_CALL_NUDGE
        return []


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


# ── Reproduction: show the problem before the change, check it after ──────────────────────

#: Where the script is while it runs (guard ``off``).  It is written right before each run and
#: removed right after: memfun's scratch is never there for the project's linters or tests to
#: pick up, and a script that deletes its own directory cannot break a later run.  The text is
#: kept in memory and in ``.memfun/checks.jsonl``.
REPRO_FILE = ".memfun/repro/issue_repro.py"
REPRO_TIMEOUT_S = 90.0
REPRO_LABEL = "reproduction script"
#: Runs the script like a test: from the project root, the project's own packages first.
_REPRO_LAUNCH = (
    "import os, runpy, sys; r = sys.argv[1]; s = sys.argv[2]; "
    "sys.path[:0] = [r, os.path.join(r, 'src')]; os.chdir(r); sys.argv = [s]; "
    "runpy.run_path(s, run_name='__main__')"
)
_FRAME_RE = re.compile(r'File "([^"\n]+)", line (\d+)')


@dataclass
class Reproduction:
    """A script that showed the problem on the unchanged code."""

    script: str
    argv: list[str]
    output: str  # what it printed on the unchanged code
    #: What it reported on the unchanged code, comparable between runs (:func:`repro_signature`).
    signature: str = ""


def repro_argv(python: str, project_root: Path, script: str, guard: str) -> list[str]:
    """The command that runs the reproduction *script*.  ``off`` (a disposable sandbox) runs
    the saved file plainly; the guarded modes run it under the probe's read-only audit hook."""
    if guard == "off":
        return [
            python,
            "-B",
            "-c",
            _REPRO_LAUNCH,
            str(project_root),
            str(project_root / REPRO_FILE),
        ]
    flags = ["-I", "-S", "-B"] if guard == "strict" else ["-I", "-B"]
    owned = ",".join(fixloop.owned_packages(project_root))
    return [python, *flags, "-c", fixloop.PROBE_PREAMBLE, str(project_root), script, guard, owned]


def classify_repro(code: int | None, output: str) -> str:
    """``reproduced`` (the script reports the problem), ``passes`` (it reports the wanted
    behaviour) or ``broken`` (it crashed, timed out or did not follow its own protocol)."""
    failed = re.search(r"^REPRO: FAIL", output, re.M) is not None
    passed = re.search(r"^REPRO: PASS", output, re.M) is not None
    if code not in (0, None) and failed:
        return "reproduced"
    if code == 0 and passed and not failed:
        return "passes"
    return "broken"


def repro_report(output: str) -> str:
    """The script's own report of the problem: its ``REPRO: FAIL`` line and what follows."""
    m = re.search(r"^REPRO: FAIL.*", output, re.M | re.S)
    return m.group(0).strip() if m else ""


def repro_signature(output: str) -> str:
    """What a failing reproduction reported, comparable between two runs: its first
    ``REPRO: FAIL`` line without the values that differ from run to run (quoted text, numbers,
    bracketed lists).  Empty when there is no such line."""
    m = re.search(r"^REPRO: FAIL.*$", output, re.M)
    if not m:
        return ""
    line = m.group(0).lower()
    line = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", "''", line)
    line = re.sub(r"0x[0-9a-f]+|\d+(?:\.\d+)?", "0", line)
    line = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "..", line)
    return " ".join(line.split())


#: Words with which a report says the script ran into an ERROR, as opposed to observing a wrong
#: result.  Such a report is the task's problem only when the task itself speaks of that error.
_ERROR_WORDS = re.compile(
    r"\b[A-Z]\w*(?:Error|Exception)\b|\b(?:unexpected|exception|traceback)\b|is not callable|"
    r"has no attribute|no module named|is not defined|cannot import|unexpected keyword|"
    r"positional argument",
    re.I,
)
_ERROR_CLASS = re.compile(r"\b[A-Z]\w*(?:Error|Exception)\b")


def repro_shows_the_task(task: str, output: str) -> bool:
    """Whether a failing reproduction plausibly shows the problem *task* describes, and not a
    mistake of its own (a wrong attribute, toy objects that do not fit together, a missing
    optional dependency: all seen on real repositories, each reported as ``REPRO: FAIL``).

    A report of a wrong RESULT is taken as it is.  A report of an ERROR (an exception class,
    "unexpected", "has no attribute", "No module named", ...) counts only when the task speaks
    of that error: it names the exception class, or shares four consecutive words with the
    report."""
    report = repro_report(output)
    if not _ERROR_WORDS.search(report):
        return True
    task_words = " ".join(re.findall(r"[a-z0-9_]+", task.lower()))
    if any(name.lower() in task_words for name in _ERROR_CLASS.findall(report)):
        return True
    words = re.findall(r"[a-z0-9_]+", report.lower())[2:]  # after "repro fail"
    return any(" ".join(words[i : i + 4]) in task_words for i in range(len(words) - 3))


def traceback_context(output: str, project_root: Path, have: str, *, radius: int = 40) -> str:
    """The code around each project line named in a traceback of *output*, for files that are
    not in the context *have* yet: where a reproduction fails is where to look."""
    parts: list[str] = []
    seen: set[tuple[str, int]] = set()
    for raw, line_s in _FRAME_RE.findall(output):
        path = Path(raw)
        try:
            rel = str(path.resolve().relative_to(project_root))
        except (OSError, ValueError):
            continue
        if rel.startswith(".memfun") or f"=== FILE: {rel} ===" in have or not path.is_file():
            continue
        line = int(line_s)
        if any(r == rel and abs(line - n) <= radius for r, n in seen):
            continue
        seen.add((rel, line))
        try:
            lines = path.read_text("utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        lo, hi = max(1, line - radius), min(len(lines), line + radius)
        body = "\n".join(lines[lo - 1 : hi])
        parts.append(
            f"=== FILE: {rel} (lines {lo}-{hi}, from the reproduction traceback) ===\n{body}"
        )
        if len(parts) >= 5:
            break
    return "\n\n".join(parts)


class ReproSolver(dspy.Module):
    """Write the reproduction script (one LLM call)."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(IssueReproduction)

    async def awrite(self, query: str, full_context: str, feedback: str) -> str:
        result = await asyncio.to_thread(
            self.predictor, query=query, full_context=full_context, feedback=feedback
        )
        script = str(getattr(result, "script", "") or "").strip()
        script = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", script).strip()
        return script if "REPRO:" in script else ""


@dataclass
class OwnTests:
    """The project's tests selected for one solve, and how to run them."""

    setup: repo_tests.TestSetup
    files: list[str]
    #: Failures that were already there before the change: never counted.
    failing_before: list[str]
    #: The current command (rebuilt when a test is left out).
    command: str = ""
    #: Tests that fail because they assert the behaviour the task changes: left out.
    old_behaviour: list[str] = field(default_factory=lambda: [])
    #: Test id -> the failure it was last classified with (a test is asked about again only
    #: when it fails differently).
    asked: dict[str, str] = field(default_factory=lambda: {})
    triage_calls: int = 0


#: At most this many failing tests are classified in one call, and this many calls per solve.
TRIAGE_MAX_TESTS = 8
TRIAGE_MAX_CALLS = 3
#: Said to the fix step whenever the solver's own checks are on.
FIX_NOTE_OLD_TESTS = (
    "\n\nNOTE: an existing test can assert behaviour that the task asks to change. Never undo "
    "the task's change to make such a test pass: correct the change if it is wrong, and leave "
    "a test that only checks the old behaviour failing."
)


def failure_sections(block: str) -> dict[str, str]:
    """pytest's failure sections of *block* by title (``Class.test[param]``)."""
    parts = re.split(r"\n_{3,}\s+(.+?)\s+_{3,}\n", fixloop.strip_ansi(block))
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        body = re.split(r"\n=+ [^\n]*=+\n", parts[i + 1])[0]
        out.setdefault(parts[i], body.strip())
    return out


class TestTriage(dspy.Module):
    """Classify failing existing tests against the task (one LLM call)."""

    __test__ = False  # not a pytest test class

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(TestFailureTriage)

    async def aclassify(self, query: str, change: str, failing_tests: str) -> dict[int, str]:
        """``{test number: "old_behaviour" | "regression"}``; a test the answer does not name,
        or names with anything else, is a regression."""
        result = await asyncio.to_thread(
            self.predictor, query=query, change=change, failing_tests=failing_tests
        )
        raw = str(getattr(result, "verdicts", "") or "")
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except ValueError:
            items = []
        out: dict[int, str] = {}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            number, verdict = item.get("test"), str(item.get("verdict", "")).strip().lower()
            if isinstance(number, str) and number.strip().lstrip("Tt").isdigit():
                number = int(number.strip().lstrip("Tt"))
            if isinstance(number, int) and not isinstance(number, bool):
                out[number] = "old_behaviour" if verdict == "old_behaviour" else "regression"
        return out


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

    #: Set per solve.  ``_log_checks``: write ``.memfun/checks.jsonl`` (own checks are on).
    #: ``_first_on_tie``: on a tie between verify rounds keep the EARLIEST best state (a
    #: model-written check is in play, so later edits made to satisfy it are not trusted over
    #: the first solution).
    _log_checks: bool = False
    _first_on_tie: bool = False
    #: The kept reproduction (run after every change as EVIDENCE, see :meth:`_run_repro`).
    _repro: Reproduction | None = None
    #: The project's own tests as selected for this solve (to leave out, later, a test that
    #: asserts the behaviour the task changes).
    _tests_run: OwnTests | None = None

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
        self.repro_solver = ReproSolver()
        self.test_triage = TestTriage()
        #: Per solve: what each of the solver's own check commands is (syntax, import, tests).
        self._check_kinds: dict[str, str] = {}
        #: Per solve: bytes the read step may still add to the prompt.
        self._read_budget = 0
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
        opts = read_project_verify_options(self.project_root)
        use_search = bool(opts.get("repo_search", self.config.repo_search))
        read_rounds = int(opts.get("read_rounds", self.config.read_rounds))
        offline = bool(opts.get("offline", self.config.offline))
        if offline and category == "web":
            category = ""  # no web step in this run: an ordinary task
        self.gatherer.focus_terms = []
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
            all_files = [path for path, _ in manifest]
            terms: list[str] = []
            ranked: list[repo_search.FileHit] = []
            if use_search:
                terms = repo_search.issue_terms(raw_query or query)
                if terms:
                    self._status(f"Searching {len(all_files)} files for {len(terms)} names...")
                    ranked = await asyncio.to_thread(
                        repo_search.rank_files, self.project_root, all_files, terms
                    )
                    manifest_str = self._planner_manifest(file_maps, all_files, ranked)
            plan = await self.planner.aplan(
                query=query,
                file_manifest=manifest_str,
                project_summary=project_summary,
                **({"offline": True} if offline else {}),
            )
            files_to_read = list(plan.files_to_read)
            if plan.unparsed:
                # Not the end of the solve: the files the answer names, else the search's.
                files_to_read = salvage_planned_files(plan.unparsed, all_files)
                logger.warning(
                    "The planner's answer could not be parsed; %d file(s) taken from its text",
                    len(files_to_read),
                )
                if not files_to_read:
                    fallback = ranked or await asyncio.to_thread(
                        repo_search.rank_files,
                        self.project_root,
                        all_files,
                        repo_search.issue_terms(raw_query or query),
                    )
                    files_to_read = [h.path for h in fallback[:PLAN_FALLBACK_FILES]]
            grep_block = ""
            if use_search:
                hits, counts = await asyncio.to_thread(
                    repo_search.grep, self.project_root, all_files, plan.search_patterns
                )
                files_to_read = self._files_to_read(files_to_read, ranked, counts)
                if hits:
                    grep_block = (
                        "\n\n=== SEARCH RESULTS (whole repository, path:line: text) ===\n"
                        + repo_search.format_grep(hits)
                    )
                self.gatherer.focus_terms = terms + [
                    p for p in plan.search_patterns if re.fullmatch(r"\w{3,}", p)
                ]
            self._status(f"Reading {len(files_to_read)} selected files...")
            full_context = await self.gatherer.agather(
                files=files_to_read,
                project_root=self.project_root,
                search_patterns=None if use_search else plan.search_patterns or None,
                web_searches=None if offline else plan.web_searches or None,
            )
            full_context += grep_block + extra_context
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

        # 2b. Reproduction (opt-in): show the problem on the unchanged code first.
        repro: Reproduction | None = None
        self._repro, self._tests_run, self._check_kinds = None, None, {}
        wants_repro = bool(opts.get("reproduce", self.config.reproduce))
        self._log_checks = wants_repro or bool(
            opts.get("discover_tests", self.config.verify_tests)
        )
        self._first_on_tie = False
        if self._log_checks:
            pythons = repo_tests.candidate_pythons(self.project_root)
            version = (
                await asyncio.to_thread(repo_tests.python_version, pythons[0], self.project_root)
                if pythons
                else ""
            )
            if version:
                full_context += (
                    f"\n\n=== ENVIRONMENT ===\nThe project runs on Python {version}: "
                    "everything you write must work on that version."
                )
        if wants_repro:
            self._status("Writing a reproduction of the problem...")
            try:
                repro = await self._reproduce(raw_query or query, full_context, opts)
            except Exception as exc:  # the reproduction must never end the solve
                logger.warning("Reproduction step failed: %s", exc)
            if repro is not None:
                self._repro = repro
                self._first_on_tie = True
                full_context += (
                    "\n\n=== REPRODUCTION (this script was run on the current code and reports "
                    "the problem; it is run again after your change. It was written from the "
                    "task text and is evidence, not the specification: where it and the task "
                    "disagree, the task is right. Change the project's code, not the script) "
                    "===\n"
                    + repro.script
                    + "\n--- its output on the current code ---\n"
                    + fixloop.clip(repro.output, 3_000)
                )
                where = traceback_context(repro.output, self.project_root, full_context)
                if where:
                    full_context += "\n\n" + where

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
        if read_rounds > 0:
            # The read step (opt-in): the solver may ask to see more before it edits.
            self._read_budget = max(
                read_requests.ROUND_BYTES,
                min(read_requests.ROUND_BYTES * read_rounds, (cap or 160_000) // 2),
            )
            offer = bool(opts.get("read_offer", self.config.read_offer))
            solve_result = await self.solver.asolve(
                query=solver_query,
                full_context=full_context
                + (READ_OFFER.format(rounds=read_rounds) if offer else ""),
                reader=self._answer_reads,
                read_rounds=read_rounds,
            )
        else:
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
        if executor.files_created:
            try:
                own_cmds = await self._own_checks(executor, opts, raw_query or query)
                if self._log_checks and verify_cmds and self._verify_commands_are_detected():
                    verify_cmds = await self._failing_only_after_the_change(verify_cmds, executor)
                verify_cmds = own_cmds + verify_cmds
            except Exception as exc:  # no check is better than a crash after the edits
                logger.warning("Could not set up the project's own tests: %s", exc)
        if (verify_cmds or self._repro is not None) and executor.files_created:
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
        elif (self._log_checks or read_rounds > 0) and not executor.files_created:
            # A run that is there to change code (its own checks or the read step are on)
            # and edited no file has failed, whatever commands it ran: reported as its own
            # class, never as a success with nothing to show.
            logger.warning("No file was edited — reported as the failure class no_edits")
            self._check_log({"kind": "no_edits", "read_rounds": solve_result.read_rounds})
            success = False
            method = NO_EDITS_METHOD

        return ContextFirstResult(
            answer=solve_result.answer,
            reasoning=solve_result.reasoning,
            ops=executor.ops,
            files_created=executor.files_created,
            success=success,
            method=method,
            total_tokens=total_tokens,
        )

    async def _answer_reads(
        self, operations: list[dict[str, Any]], round_no: int, rounds_left: int
    ) -> str:
        """The read step: answer the read requests of one solver reply (memfun's own code,
        confined to the project, nothing executed) and return the text for the context."""
        items = read_requests.requests_of(operations)
        served = await asyncio.to_thread(
            read_requests.serve,
            self.project_root,
            items,
            budget=max(0, min(read_requests.ROUND_BYTES, self._read_budget)),
        )
        used = sum(len(s.output.encode("utf-8", "replace")) for s in served if s.ok)
        self._read_budget -= used
        read_requests.log_reads(self.project_root, round_no, served)
        n_ok = sum(1 for s in served if s.ok)
        logger.info(
            "Read step, round %d: %d of %d requests served (%d bytes)",
            round_no,
            n_ok,
            len(served),
            used,
        )
        self._status(f"Reading what the solver asked for ({n_ok} of {len(served)} requests)...")
        tail = READ_MORE.format(left=rounds_left) if rounds_left > 0 else READ_LAST
        return (
            f"\n\n=== WHAT YOU ASKED TO READ (round {round_no}) ===\n"
            + (read_requests.format_served(served) or "(the reply asked for nothing readable)")
            + "\n\n"
            + tail
        )

    def _planner_manifest(
        self, file_maps: list[Any], all_files: list[str], ranked: list[repo_search.FileHit]
    ) -> str:
        """What the planner chooses from on a large project: the files that define or mention
        the names in the task (found by searching every file), the definitions of the best of
        them, and the directory layout."""
        from memfun_agent.code_map import code_map_to_string

        sources = [h for h in ranked if not h.is_test]
        tests = [h for h in ranked if h.is_test]
        if not sources and not tests:
            return code_map_to_string(file_maps, max_tokens=2000)
        parts = [
            f"FILES THAT DEFINE OR MENTION NAMES FROM THE TASK (all {len(all_files)} source "
            "files were searched; most relevant first):\n"
            + repo_search.format_ranked(sources, max_files=20, max_chars=6_000)
        ]
        if tests:
            parts.append(
                "TESTS THAT MENTION THE SAME NAMES:\n"
                + repo_search.format_ranked(tests, max_files=6, max_chars=1_500)
            )
        top = {h.path for h in sources[:15]}
        maps = [fm for fm in file_maps if fm.path in top]
        if maps:
            parts.append(
                "DEFINITIONS IN THE MOST RELEVANT FILES:\n"
                + code_map_to_string(maps, max_tokens=2500)
            )
        parts.append(
            "DIRECTORIES OF THE PROJECT (any file may be named, not only the ones above):\n"
            + repo_search.directory_overview(all_files)
        )
        return "\n\n".join(parts)

    @staticmethod
    def _files_to_read(
        planned: list[str], ranked: list[repo_search.FileHit], grep_counts: dict[str, int]
    ) -> list[str]:
        """The planner's files first, then a safety net it may have missed: the two best
        search-ranked source files, the two source files its own patterns hit most, and the
        best-ranked test file (to see how the project tests this code)."""
        files = list(dict.fromkeys(f.strip() for f in planned if f.strip()))
        extra = [h.path for h in ranked if not h.is_test][:2]
        by_hits = sorted(grep_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        extra += [f for f, _ in by_hits if not repo_search.is_test_path(f)][:2]
        extra += [h.path for h in ranked if h.is_test][:1]
        return files + [f for f in dict.fromkeys(extra) if f not in files]

    async def _reproduce(
        self, query: str, full_context: str, opts: dict[str, Any]
    ) -> Reproduction | None:
        """Have the model write a script that shows the problem and keep it only if, run on
        the unchanged code, it reports the problem the task describes (not an error of its
        own, see :func:`repro_shows_the_task`).  One rewrite with the script's own output when
        it does not; after that there is no reproduction."""
        guard = str(opts.get("probe_guard", self.config.fix_probe_guard))
        if guard not in fixloop.PROBE_GUARDS:
            guard = "project"
        candidates = repo_tests.candidate_pythons(self.project_root)
        if not candidates:
            logger.info("Reproduction skipped: no project interpreter found")
            return None
        python = candidates[0]
        feedback = ""
        for attempt in (1, 2):
            script = await self.repro_solver.awrite(query, full_context, feedback)
            if not script:
                logger.info("Reproduction: the model gave no script (attempt %d)", attempt)
                self._check_log({"event": "reproduction", "attempt": attempt, "verdict": "none"})
                break
            argv = repro_argv(python, self.project_root, script, guard)
            code, out = await asyncio.to_thread(self._run_script, script, argv)
            verdict = classify_repro(code, out)
            if verdict == "reproduced" and not repro_shows_the_task(query, out):
                verdict = "own_error"
            logger.info("Reproduction attempt %d: %s (exit %s)", attempt, verdict, code)
            self._check_log(
                {
                    "event": "reproduction",
                    "attempt": attempt,
                    "verdict": verdict,
                    "exit": code,
                    "output": fixloop.clip(out, 1_500),
                    "script": fixloop.clip(script, 12_000),
                }
            )
            if verdict == "reproduced":
                self._status("The problem is reproduced")
                return Reproduction(
                    script=script, argv=argv, output=out, signature=repro_signature(out)
                )
            why = {
                "passes": "it printed REPRO: PASS on the UNCHANGED code, so it does not detect "
                "the problem",
                "own_error": "it reported an error the task does not speak of. That is most "
                "likely a mistake in the script itself (a wrong name or call, objects that do "
                "not fit together, something that is not installed), not the problem of the "
                "task: use the project's API the way its own tests do, and report only what "
                "the task describes",
            }.get(
                verdict,
                "it did not finish with a REPRO: FAIL or REPRO: PASS line (it crashed, timed "
                "out or let an exception escape)",
            )
            feedback = (
                f"Your previous script was run on the unchanged code and {why}.\n"
                f"--- the script ---\n{script}\n--- its output (exit {code}) ---\n"
                + fixloop.clip(out, 3_000)
            )
        self._status("No reproduction of the problem")
        return None

    def _run_script(self, script: str, argv: list[str]) -> tuple[int | None, str]:
        """Run a reproduction *script*.  Its file exists only for the run: it is (re)written
        right before and removed right after."""
        path = self.project_root / REPRO_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script + "\n", "utf-8")
            return repo_tests.run(argv, self.project_root, REPRO_TIMEOUT_S)
        except OSError as exc:
            return None, f"(could not run: {exc})"
        finally:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                path.parent.rmdir()

    async def _run_repro(self) -> tuple[str, str]:
        """Run the kept reproduction on the current code.  ``("passes", out)``: it now reports
        the wanted behaviour.  ``("same", out)``: it reports what it reported on the unchanged
        code, so the change did not touch the problem it shows.  ``("other", out)``: anything
        else (a different report, a crash, a timeout), which is no evidence either way: the
        script checks its own idea of the task, and a change can make that idea fail."""
        repro = self._repro
        if repro is None:
            return "", ""
        code, out = await asyncio.to_thread(self._run_script, repro.script, repro.argv)
        verdict = classify_repro(code, out)
        if verdict == "passes":
            return "passes", out
        if verdict == "reproduced" and repro.signature and repro_signature(out) == repro.signature:
            return "same", out
        return "other", out

    async def _own_checks(
        self,
        executor: OperationExecutor,
        opts: dict[str, Any],
        task: str = "",
    ) -> list[str]:
        """Verify commands from the project itself: a syntax check and an import check of the
        changed Python files, and the project's tests relevant to the changed files with the
        failures that were already there left out.  Empty unless opted in.  (The reproduction
        script is not one of them: it is run next to them, as evidence.)"""
        use_tests = bool(opts.get("discover_tests", self.config.verify_tests))
        if not use_tests:
            return []
        timeout = float(opts.get("test_timeout", self.config.test_timeout))
        self.verifier.timeout = max(self.verifier.timeout, timeout)
        cmds: list[str] = []
        changed: list[str] = []
        for f in dict.fromkeys(executor.files_created):
            with contextlib.suppress(ValueError):
                changed.append(str(Path(f).resolve().relative_to(self.project_root)))
        setup = await asyncio.to_thread(repo_tests.discover, self.project_root)
        if not setup.python:
            self._status("No project interpreter found: the project's tests are not run")
            return cmds
        py_files = [f for f in changed if f.endswith(".py") and (self.project_root / f).is_file()]
        originals = dict(executor.originals)
        if py_files:
            cmd = repo_tests.shell(repo_tests.compile_command(setup.python, py_files))
            self.verifier.labels[cmd] = "syntax check of the changed files"
            self._check_kinds[cmd] = "syntax"
            cmds.insert(0, cmd)
            # What imported before the change must still import after it (a name that does
            # not exist, a circular import: no syntax check sees those).

            def importable() -> list[str]:
                with repo_tests.original_files(originals):
                    return repo_tests.importable_modules(setup.python or "", root, py_files)

            root = self.project_root
            modules = await asyncio.to_thread(importable)
            if modules:
                cmd = repo_tests.shell(repo_tests.import_command(setup.python, modules))
                self.verifier.labels[cmd] = "import check of the changed modules: " + ", ".join(
                    modules
                )
                self._check_kinds[cmd] = "import"
                cmds.insert(1, cmd)
        if not setup.usable:
            self._status("No tests found in the project")
            return cmds
        symbols: list[str] = []
        for path, before in originals.items():
            if path.suffix == ".py" and path.is_file():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    symbols += repo_tests.changed_symbols(before or "", path.read_text("utf-8"))
        selected = repo_tests.select_tests(self.project_root, changed, setup, symbols=symbols)
        files = [s.path for s in selected]
        if len(files) < 2 and task:
            # Few or no tests are tied to the changed files by name or import (common when a
            # project tests through its public interface): add the tests that mention the
            # names the task itself uses.
            terms = repo_search.issue_terms(task)
            by_task = await asyncio.to_thread(
                repo_search.rank_files, self.project_root, list(setup.test_files), terms
            )
            files += [h.path for h in by_task if h.path not in files][: 2 - len(files)]
        if not files:
            self._status("No existing test is related to the changed files")
            self._check_log({"event": "tests_selected", "files": [], "changed": changed})
            return cmds
        new_files = {
            str(p.relative_to(self.project_root))
            for p, text in originals.items()
            if text is None and p.is_relative_to(self.project_root)
        }
        old_files = [f for f in files if f not in new_files]
        failing: list[str] = []
        if old_files:
            self._status(f"Running {len(old_files)} related test files on the original code...")
            failing, broken, note = await asyncio.to_thread(
                repo_tests.baseline_failures,
                self.project_root,
                setup,
                old_files,
                originals,
                timeout,
            )
            noisy = {f for f in old_files if sum(i.startswith(f + "::") for i in failing) > 20}
            files = [f for f in files if f not in set(broken) | noisy]
            failing = [i for i in failing if i.split("::")[0] in files]
            if note:
                logger.info("Related tests on the original code: %s", note)
        if not files:
            self._status("The related tests do not run on the original code: not used")
            return cmds
        argv = repo_tests.test_command(setup, files, failing)
        self._check_log(
            {
                "event": "tests_selected",
                "files": files,
                "by_file": {s.path: list(s.reasons) for s in selected},
                "left_out_failing_before": failing,
                "python": setup.python,
                "runner": setup.runner,
            }
        )
        logger.info(
            "Verifying with %d test files (%s); %d failures that were already there left out",
            len(files),
            ", ".join(files),
            len(failing),
        )
        self._status(f"Verifying with {len(files)} related test files")
        tests_cmd = repo_tests.shell(argv)
        self._check_kinds[tests_cmd] = "tests"
        if setup.runner == "pytest":
            self.verifier.ok_exits[tests_cmd] = (repo_tests.PYTEST_NO_TESTS_RAN,)
        self._tests_run = OwnTests(
            setup=setup, files=files, failing_before=failing, command=tests_cmd
        )
        return [*cmds, tests_cmd]

    def _check_log(self, record: dict[str, Any]) -> None:
        if self._log_checks:
            fixloop.log_check(self.project_root, record)

    def _get_verify_commands(self) -> list[str]:
        """Get verification commands: explicit config or auto-detected."""
        if self.config.verify_commands:
            return list(self.config.verify_commands)
        project_cmds, _ = read_project_verify(self.project_root)
        if project_cmds:
            return project_cmds  # the project's own checks (e.g. its visible tests)
        return _detect_verify_commands(self.project_root)

    def _verify_commands_are_detected(self) -> bool:
        """Whether :meth:`_get_verify_commands` returns auto-detected linters (nobody
        configured a command)."""
        return not self.config.verify_commands and not read_project_verify(self.project_root)[0]

    async def _failing_only_after_the_change(
        self, cmds: list[str], executor: OperationExecutor
    ) -> list[str]:
        """*cmds* without those that already fail on the unchanged code.  A detected linter
        that reports problems before the change says nothing about the change (the same rule
        as for the tests that failed before): asking the fix step to clean them up sends it
        into files the task never touched."""
        if not executor.originals:
            return cmds

        with repo_tests.original_files(executor.originals):
            vr = await self.verifier.averify(cmds)
        for cmd in vr.failed_cmds:
            logger.info("Left out, it already fails on the unchanged code: %s", cmd)
        if vr.failed_cmds:
            self._check_log({"event": "checks_failing_before", "left_out": vr.failed_cmds})
        return [c for c in cmds if c not in vr.failed_cmds]

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
        paths = [Path(f) for f in dict.fromkeys(executor.files_created)]
        best = fixloop.BestSoFar(paths)
        last: list[tuple[int, int, int] | None] = [(1, 1, 0)]  # last failing score; None: passed
        try:
            await self._fix_rounds(query, full_context, executor, verify_cmds, paths, best, last)
        finally:
            # Never finish worse than the best state this loop saw, whatever ended the loop
            # (a cap, a stall, an exception, a cancellation).
            sc = last[0]
            worse = sc is not None and best.best_score is not None and sc > best.best_score
            tie = sc is not None and sc == best.best_score and self._first_on_tie
            # Nor on the starting tree when a candidate existed: an undone change is no answer.
            undone = self._log_checks and bool(best.best_files) and self._unchanged(executor)
            if worse or tie or undone:
                changed = best.restore()
                if changed:
                    logger.info(
                        "Restored the best verified state (round %d, score %s) over the final %s",
                        best.best_round,
                        best.best_score,
                        sc,
                    )
                    self._status(f"Restored the best state (round {best.best_round})")

    async def _fix_rounds(
        self,
        query: str,
        full_context: str,
        executor: OperationExecutor,
        verify_cmds: list[str],
        paths: list[Path],
        best: fixloop.BestSoFar,
        last: list[tuple[int, int, int] | None],
    ) -> None:
        """The rounds of :meth:`_verify_and_fix`; keeps ``last[0]`` at the latest failing score."""
        base = self._max_fix_attempts()
        cap = max(base, self.config.fix_hard_cap)
        patience = max(1, self.config.fix_patience)
        restarts_left = max(0, self.config.fix_restarts)
        opts = read_project_verify_options(self.project_root)
        probes_left = max(0, int(opts.get("probe_rounds", self.config.fix_probe_rounds)))
        fix_detail = bool(opts.get("fix_detail", self.config.fix_detail))
        guard = str(opts.get("probe_guard", self.config.fix_probe_guard))
        if guard not in fixloop.PROBE_GUARDS:
            logger.warning("unknown fix_probe_guard %r; using 'project'", guard)
            guard = "project"
        if probes_left > 0:
            fixloop.init_probe_log(self.project_root)
        experiments: list[str] = []  # real outputs of probe snippets, newest last
        history: list[str] = []
        seen_states: dict[str, tuple[int, tuple[int, int, int]]] = {}  # state -> (round, score)
        stagnant = 0
        repro_only_rounds = 0  # fix rounds spent on nothing but the reproduction
        for attempt in range(1, cap + 1):
            self._status(f"Verifying (round {attempt}, base {base}, cap {cap})...")
            vr = await self.verifier.averify(verify_cmds)
            left_out: list[str] = []
            if self._log_checks and not vr.passed:
                # An existing test that asserts the behaviour the task changes is not a
                # regression: it is left out, and the round is judged without it.
                left_out = await self._leave_out_old_behaviour(query, executor, verify_cmds, vr)
                if left_out:
                    vr = await self.verifier.averify(verify_cmds)
            # The reproduction is evidence next to the checks, never a check of its own: it
            # counts only while it reports exactly what it reported on the unchanged code.
            repro_state, repro_out = "", ""
            if self._repro is not None and not self._does_not_import(vr):
                repro_state, repro_out = await self._run_repro()
            repro_same = repro_state == "same"
            self._check_log(
                {
                    "event": "verify",
                    "round": attempt,
                    "passed": vr.passed and not repro_same,
                    "commands": [self.verifier.labels.get(c, c)[:300] for c in verify_cmds],
                    "skipped": [self.verifier.labels.get(c, c)[:300] for c in vr.skipped],
                    "failed": [b[:1_500] for b in vr.blocks],
                    **(
                        {"reproduction": repro_state, "reproduction_output": repro_out[-1_500:]}
                        if repro_state
                        else {}
                    ),
                    **({"left_out_old_behaviour": left_out} if left_out else {}),
                }
            )
            undone = self._log_checks and self._unchanged(executor)
            if vr.passed and not repro_same:
                if undone and best.best_files:
                    # The edits were taken back until the checks passed: the starting tree is
                    # not a solution.  The best candidate is put back and the loop ends.
                    logger.info("The change was undone; keeping the best candidate instead")
                    self._status("The change was undone: keeping the best candidate")
                    best.restore()
                    last[0] = best.best_score
                    return
                last[0] = None
                self._status("Verification passed")
                return
            sc = self._round_score(vr, repro_same)
            # With the solver's own checks on, several commands can fail in one round: each
            # is shown to the fix step.
            by_command = self._log_checks and len(vr.blocks) > 1
            last[0] = sc
            state = fixloop.state_key(fixloop.read_texts(paths))
            revisit = seen_states.get(state)
            seen_states.setdefault(state, (attempt, sc))
            if revisit is not None and attempt > 1:
                history.append(
                    f"round {attempt}: the code is IDENTICAL to round {revisit[0]} (same failures "
                    "again). That change was undone or repeated: it is a dead end, try something "
                    "structurally different."
                )
            # The starting tree is never the best state (it is what the task wants changed).
            improved = False if undone else best.observe(attempt, sc)
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
            if vr.passed:  # only the reproduction still reports the problem
                if repro_only_rounds >= 1:
                    self._status("Only the reproduction still fails: no further fix round")
                    break
                repro_only_rounds += 1
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
            errors_for_fix = (
                fixloop.compact_blocks(vr.blocks)
                if by_command
                else fixloop.compact_errors(vr.errors)
            )
            if repro_same:
                errors_for_fix = (
                    errors_for_fix
                    + f"\n\n$ {REPRO_LABEL}\n"
                    + fixloop.clip(repro_out, 3_000)
                    + "\n(The reproduction script still reports what it reported before your "
                    "change. It was written from the task text and can itself be wrong: fix the "
                    "project's code where the report is right, and change nothing only to "
                    "silence the script.)"
                ).strip()
            if fix_detail and summary.failed_ids:
                try:
                    detail = await fixloop.run_detail(
                        self.project_root, verify_cmds, list(summary.failed_ids)
                    )
                except Exception as exc:  # the look step must never end the fix loop
                    logger.warning("Failure-detail step failed: %s", exc)
                    detail = ""
                if detail:  # one prompt field only (it used to be sent twice)
                    errors_for_fix += "\n\nFull assertion detail of the failing tests:\n" + detail
            if probes_left > 0 and not improved:
                # A round without progress: stop guessing, look at real behaviour.
                probes_left -= 1
                self._status("No progress: running differential experiments...")
                try:
                    new_runs = await self._look(
                        query,
                        current_context,
                        errors_for_fix,
                        history,
                        experiments,
                        verify_cmds,
                        guard,
                    )
                except Exception as exc:  # the look step must never end the fix loop
                    logger.warning("Look step failed: %s", exc)
                    new_runs = []
                if new_runs:
                    experiments.extend(new_runs)
                    history.append(f"round {attempt}: ran {len(new_runs)} experiments (see below)")
            if experiments:
                current_context += (
                    "\n\n=== EXPERIMENT RESULTS (real output of scripts run in this project) ===\n"
                    + "\n\n".join(experiments[-3:])
                )
            fix_query = query
            if self._log_checks and summary.failed_ids:
                fix_query += FIX_NOTE_OLD_TESTS
            if history:
                fix_query += (
                    "\n\nPREVIOUS FIX ATTEMPTS (do not repeat what did not work):\n"
                    + "\n".join(history[-6:])
                )
            try:
                fix_ops = await self.fix_solver.afix(
                    query=fix_query,
                    full_context=current_context,
                    verification_errors=errors_for_fix,
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
                originals=executor.originals,
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

    def _round_score(self, vr: VerifyResult, repro_same: bool) -> tuple[int, int, int]:
        """A failing round's score, lower is better: ``(broken, failing, -passing)``.

        Without the solver's own checks this is :func:`fixloop.score` of the joined output,
        as it always was.  With them each failing command is scored for what it is: a state
        whose changed files do not compile or import, or whose tests do not collect or run, is
        BROKEN and always worse than one that is not, whatever else fails; then failing tests
        count most, any other failing command (a linter) less, and a reproduction that still
        reports the original problem least."""
        if not self._log_checks:
            return fixloop.score(vr.errors, False)
        broken, tests, other, passing = False, 0, 0, 0
        for cmd, block in zip(vr.failed_cmds, vr.blocks, strict=False):
            kind = self._check_kinds.get(cmd, "other")
            s = fixloop.summarize(block)
            if kind in ("syntax", "import"):
                broken = True
            elif s.found:
                broken = broken or s.broken
                tests += s.failed + s.errors
                passing += s.passed
            elif kind == "tests":  # they did not run at all (time limit, crash at start)
                broken = True
            else:
                other += 1
        return (1 if broken else 0, 4 * tests + 2 * other + (1 if repro_same else 0), -passing)

    def _does_not_import(self, vr: VerifyResult) -> bool:
        """Whether the syntax or import check of the changed files failed this round."""
        return any(self._check_kinds.get(c) in ("syntax", "import") for c in vr.failed_cmds)

    @staticmethod
    def _unchanged(executor: OperationExecutor) -> bool:
        """Whether every file the solve has touched is back at its original content: the tree
        is the starting tree.  False when nothing was touched."""
        if not executor.originals:
            return False
        for path, before in executor.originals.items():
            try:
                now = path.read_text("utf-8") if path.is_file() else None
            except (OSError, UnicodeDecodeError):
                return False
            if now != before:
                return False
        return True

    async def _leave_out_old_behaviour(
        self, query: str, executor: OperationExecutor, verify_cmds: list[str], vr: VerifyResult
    ) -> list[str]:
        """Ask the model, once per new failure, whether each failing existing test asserts the
        behaviour the task changes.  Those tests are left out of the run from now on
        (*verify_cmds* is updated in place); the others are regressions for the fix step.
        Returns the ids left out by this call."""
        own = self._tests_run
        if own is None or own.command not in vr.failed_cmds or self._does_not_import(vr):
            return []  # code that does not import fails tests for that reason alone
        block = vr.blocks[vr.failed_cmds.index(own.command)]
        summary = fixloop.summarize(block)
        if summary.broken or not summary.failed_ids or own.triage_calls >= TRIAGE_MAX_CALLS:
            return []
        sections = failure_sections(block)
        ask: list[tuple[str, str]] = []
        for tid in summary.failed_ids:
            detail = sections.get(".".join(tid.split("::")[1:]), "")
            if own.asked.get(tid) != detail:  # not asked yet, or it fails differently now
                ask.append((tid, detail))
        ask = ask[:TRIAGE_MAX_TESTS]
        if not ask:
            return []
        own.triage_calls += 1
        parts: list[str] = []
        for n, (tid, detail) in enumerate(ask, 1):
            src = fixloop.failing_tests_source(self.project_root, [tid], max_chars=2_500)
            parts.append(
                f"TEST {n}: {tid}\n--- source ---\n{src or '(not found)'}\n--- failure ---\n"
                + fixloop.clip(detail or "(no detail)", 1_500)
            )
        before = {p: (t or "") for p, t in executor.originals.items()}
        change = fixloop.short_diff(before, fixloop.read_texts(list(before)), limit=4_000)
        try:
            verdicts = await self.test_triage.aclassify(
                query=query, change=change, failing_tests="\n\n".join(parts)
            )
        except Exception as exc:  # no classification: every failure stays a regression
            logger.warning("Classifying the failing tests failed: %s", exc)
            verdicts = {}
        left_out: list[str] = []
        for n, (tid, detail) in enumerate(ask, 1):
            own.asked[tid] = detail
            if verdicts.get(n) == "old_behaviour":
                left_out.append(tid)
        self._check_log(
            {
                "event": "failing_tests_classified",
                "asked": [tid for tid, _ in ask],
                "old_behaviour": left_out,
            }
        )
        if not left_out:
            return []
        logger.info(
            "Tests that assert the behaviour the task changes, left out: %s", ", ".join(left_out)
        )
        self._status(f"{len(left_out)} existing tests assert the old behaviour: left out")
        own.old_behaviour += left_out
        cmd = repo_tests.shell(
            repo_tests.test_command(
                own.setup, own.files, [*own.failing_before, *own.old_behaviour]
            )
        )
        self._check_kinds[cmd] = "tests"
        if own.command in self.verifier.ok_exits:
            self.verifier.ok_exits[cmd] = self.verifier.ok_exits[own.command]
        verify_cmds[verify_cmds.index(own.command)] = cmd
        own.command = cmd
        return left_out

    async def _look(
        self,
        query: str,
        context: str,
        errors: str,
        history: list[str],
        experiments: list[str],
        verify_cmds: list[str],
        guard: str,
    ) -> list[str]:
        """Differential probes for one stalled round: go on while a probe prints something new.

        Each iteration the model sees this stall's earlier experiments and writes a new script.
        It stops when the model proposes nothing, the output is empty, refused or timed out, or
        it repeats an earlier output (no new information).  Read-only; at most
        ``fix_probe_iterations`` per stalled round.
        """
        outs: list[str] = []
        seen: set[str] = set()
        empties = 0
        for _ in range(max(1, self.config.fix_probe_iterations)):
            ctx = context
            shown = (experiments + outs)[-3:]
            if shown:
                ctx += "\n\n=== EXPERIMENTS SO FAR (do not repeat) ===\n" + "\n\n".join(shown)
            try:
                snippets = await self.probe_solver.apropose(
                    query=query,
                    full_context=ctx,
                    verification_errors=errors,
                    history="\n".join(history[-6:]),
                )
            except Exception as exc:
                logger.warning("Probe proposal failed: %s", exc)
                break
            logger.info("Look step: %d experiments proposed", len(snippets))
            if not snippets:
                break
            out = await fixloop.run_probes(
                self.project_root,
                snippets,
                python=None
                if guard == "strict"
                else fixloop.probe_python(verify_cmds, self.project_root),
                guard=guard,
            )
            fixloop.log_probe(self.project_root, snippets, out)
            key = fixloop.probe_fingerprint(out)
            if not key:
                outs.append(out)  # a refusal or an empty result is information; try another probe
                empties += 1
                if empties >= 2:
                    break
                continue
            if key in seen:
                break  # the same output again: nothing new to learn
            seen.add(key)
            outs.append(out)
        return outs

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
        rerun = OperationExecutor(
            self.project_root, on_status=self._on_status, originals=executor.originals
        )
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
                    originals=executor.originals,
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
