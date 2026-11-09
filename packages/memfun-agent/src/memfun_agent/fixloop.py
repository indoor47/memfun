"""Progress-aware verify-and-fix helpers for the context-first solver.

The plain loop ran N verify rounds, always kept the LAST edit, and showed the model only
the first 3000 characters of the verifier output.  On the B300 harness benchmark that
ended runs worse than they had been (a late edit broke the module's import: 0 of 23 tests
ran) and starved the model of what it needed (the failing tests' source, the earlier
attempts).  These helpers add:

* :func:`summarize` / :func:`score` -- read a pytest run into ``(broken, failing, -passing)``
  so rounds can be compared (lower is better);
* :class:`BestSoFar` -- snapshot the owned files after every verified round and restore the
  best one at the end, so the loop can never finish worse than its best state;
* :func:`failing_tests_source` / :func:`compact_errors` -- what the fix step should read;
* :func:`clip` -- head + tail clipping of verifier output (pytest's summary is at the END).

Pure functions and small classes: no LLM, no network, hermetic tests.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$", re.M)
_COUNT = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_BROKEN_HINTS = (
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
    "NameError: name",
    "no tests ran",
    "collected 0 items",
    "errors during collection",
    "Interrupted:",
)

if TYPE_CHECKING:
    from pathlib import Path


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@dataclass(frozen=True)
class TestSummary:
    """A parsed pytest run (``found`` is False when the output is not pytest's)."""

    __test__ = False  # not a pytest test class

    found: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failed_ids: tuple[str, ...] = ()
    broken: bool = False


def summarize(output: str) -> TestSummary:
    """Parse a pytest run's output: counts, failing test ids, and whether the code is broken
    (does not import, does not collect) as opposed to merely failing some tests."""
    text = strip_ansi(output)
    counts = {"passed": 0, "failed": 0, "errors": 0}
    last = ""
    for line in text.splitlines():
        if re.search(r"\b(passed|failed|error|errors)\b.*\bin [\d.]+s", line):
            last = line
    for n, word in _COUNT.findall(last):
        key = "errors" if word.startswith("error") else word
        if key in counts:
            counts[key] = int(n)
    ids = tuple(dict.fromkeys(m.group(1) for m in _FAILED_LINE.finditer(text)))
    found = bool(last) or bool(ids)
    # Broken = the code does not import / collect: nothing ran green at all.
    broken = counts["passed"] == 0 and (
        counts["errors"] > 0 or any(h in text for h in _BROKEN_HINTS)
    )
    return TestSummary(
        found=found,
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["errors"],
        failed_ids=ids,
        broken=broken,
    )


def score(output: str, passed: bool) -> tuple[int, int, int]:
    """Comparable progress score, lower is better: ``(broken, failing, -passing)``.

    A run that does not import/collect is the worst; otherwise fewer failing tests win,
    then more passing tests.  A non-pytest verifier (a linter) scores by its error lines."""
    if passed:
        return (0, 0, 0)
    s = summarize(output)
    if s.found:
        return (1 if s.broken else 0, s.failed + s.errors, -s.passed)
    lint = len(re.findall(r"^\S+:\d+:\d+:", strip_ansi(output), re.M))
    return (0, lint or 1, 0)


def clip(text: str, limit: int = 10_000) -> str:
    """Head + tail of *text* (pytest prints its failure summary last)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters clipped] ...\n" + text[-half:]


def _test_function_source(path: Path, qualname: list[str]) -> str:
    try:
        src = path.read_text("utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""
    nodes: list[ast.stmt] = list(tree.body)
    found: ast.AST | None = None
    for name in qualname:
        found = next(
            (
                n
                for n in nodes
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and n.name == name
            ),
            None,
        )
        if found is None:
            return ""
        nodes = list(getattr(found, "body", []))
    return (ast.get_source_segment(src, found) or "") if found is not None else ""


def failing_tests_source(
    project_root: Path, failed_ids: tuple[str, ...] | list[str], max_chars: int = 9_000
) -> str:
    """The source of each failing test (``path::Class::test``), read-only context for the fix."""
    parts: list[str] = []
    total = 0
    for tid in failed_ids:
        path_s, *rest = tid.split("::")
        base = re.sub(r"\[.*\]$", "", rest[-1]) if rest else ""
        qual = [*rest[:-1], base] if rest else []
        p = project_root / path_s
        body = _test_function_source(p, qual) if p.is_file() and qual else ""
        if not body:
            continue
        chunk = f"# {tid}\n{body}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def compact_errors(output: str, per_failure: int = 900, max_failures: int = 12) -> str:
    """The failure summary lines first, then a short excerpt per failing test section."""
    text = strip_ansi(output)
    s = summarize(text)
    head = [f"{s.passed} passed, {s.failed} failed, {s.errors} errors"] if s.found else []
    summary_lines = [ln for ln in text.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    head += summary_lines[: max_failures * 2]
    sections = re.split(r"\n_{3,}\s+(.+?)\s+_{3,}\n", text)
    excerpts: list[str] = []
    for i in range(1, len(sections) - 1, 2):
        if len(excerpts) >= max_failures:
            break
        excerpts.append(f"--- {sections[i]} ---\n{sections[i + 1].strip()[:per_failure]}")
    body = "\n\n".join(excerpts) if excerpts else clip(text, 6_000)
    return ("\n".join(head) + "\n\n" + body).strip()


@dataclass
class BestSoFar:
    """Snapshot the owned files whenever a round beats the best score; restore at the end."""

    paths: list[Path]
    best_score: tuple[int, int, int] | None = None
    best_files: dict[Path, str] = field(default_factory=lambda: {})
    best_round: int = 0

    def observe(self, round_no: int, sc: tuple[int, int, int]) -> bool:
        """Record round *round_no*'s score; True when it is a new best (files snapshotted)."""
        if self.best_score is not None and sc >= self.best_score:
            return False
        self.best_score, self.best_round = sc, round_no
        self.best_files = {}
        for p in self.paths:
            try:
                self.best_files[p] = p.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        return True

    def restore(self) -> int:
        """Write the best snapshot back; returns how many files changed."""
        changed = 0
        for p, text in self.best_files.items():
            try:
                if p.read_text("utf-8") != text:
                    p.write_text(text, "utf-8")
                    changed += 1
            except (OSError, UnicodeDecodeError):
                continue
        return changed


# ── Probes: let the model look at real behaviour when patching stalls ─────────

MAX_PROBES = 2
MAX_PROBE_CHARS = 2_000
PROBE_TIMEOUT_S = 30.0
PROBE_OUTPUT_CHARS = 3_000
#: Snippets touching these are refused (probes observe; they never change or fetch anything).
_PROBE_FORBIDDEN = re.compile(
    r"\b(os\.(remove|unlink|rmdir|system|popen|rename|replace)|shutil\.|subprocess|socket|"
    r"urllib|requests|httpx|http\.client|open\([^)]*['\"][wax+]|write_text|write_bytes|"
    r"__import__|eval\(|exec\()"
)


def parse_probes(raw: str) -> list[str]:
    """The snippets of a model's ``probes`` answer: a JSON array of strings, tolerant of fences
    and of a single bare snippet.  At most :data:`MAX_PROBES`, each at most
    :data:`MAX_PROBE_CHARS`."""
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
    items: list[object]
    try:
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
    except ValueError:
        items = [text] if text and "print" in text else []
    out = [
        x.strip() for x in items if isinstance(x, str) and x.strip() and len(x) <= MAX_PROBE_CHARS
    ]
    return out[:MAX_PROBES]


def probe_refusal(snippet: str) -> str:
    """Why *snippet* may not run ("" when it may)."""
    m = _PROBE_FORBIDDEN.search(snippet)
    return f"refused (uses {m.group(0)!r})" if m else ""


async def run_probes(project_root: Path, snippets: list[str]) -> str:
    """Run each snippet with ``python -c`` from *project_root* (timeout, clipped output) and
    return a text block of snippet + real output for the next fix round."""
    blocks: list[str] = []
    for i, snippet in enumerate(snippets[:MAX_PROBES], 1):
        why = probe_refusal(snippet)
        if why:
            blocks.append(f"--- experiment {i} ---\n{snippet}\n=> {why}")
            continue
        try:
            res = await asyncio.to_thread(
                subprocess.run,
                [shutil.which("python") or sys.executable, "-c", snippet],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_S,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                check=False,
            )
            out = (res.stdout + (("\n" + res.stderr) if res.stderr else "")).strip()
            body = clip(out or "(no output)", PROBE_OUTPUT_CHARS)
        except subprocess.TimeoutExpired:
            body = f"(timed out after {PROBE_TIMEOUT_S:.0f}s)"
        except OSError as exc:
            body = f"(could not run: {exc})"
        blocks.append(f"--- experiment {i} ---\n{snippet}\n=> {body}")
    return "\n\n".join(blocks)


# ── Round memory: what each round changed, and whether we have been here before ──


def read_texts(paths: list[Path]) -> dict[Path, str]:
    """Current text of every readable file in *paths*."""
    out: dict[Path, str] = {}
    for p in paths:
        try:
            out[p] = p.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def state_key(texts: dict[Path, str]) -> str:
    """A hash of the file contents: two rounds with the same key left the code identical."""
    h = hashlib.sha256()
    for p in sorted(texts, key=str):
        h.update(str(p).encode())
        h.update(b"\0")
        h.update(texts[p].encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def short_diff(before: dict[Path, str], after: dict[Path, str], limit: int = 700) -> str:
    """A compact unified diff of what a round changed (changed lines only), clipped."""
    lines: list[str] = []
    for p in sorted(after, key=str):
        a, b = before.get(p, ""), after[p]
        if a == b:
            continue
        for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
            if ln.startswith(("---", "+++", "@@")):
                continue
            lines.append(ln)
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n... [diff clipped]"


# ── The LOOK step: untruncated failures and differential probes ────────────────────────────


def pytest_detail_command(
    verify_cmds: list[str], failed_ids: list[str] | tuple[str, ...]
) -> list[str] | None:
    """A pytest command that re-runs only *failed_ids* with full assertion detail, or None.

    pytest prints an assertion on a list or dict as one truncated line
    (``assert [{'a...'}, ...] == []``) unless verbosity is 2 or more.  This takes the project's own
    pytest verify command, drops its test selectors and ``-q``, adds ``-vv`` and the failing ids.
    Other verify commands (linters) give None.
    """
    for cmd in verify_cmds:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue
        if "pytest" not in tokens and not any(t.endswith("pytest") for t in tokens):
            continue
        idx = next(i for i, t in enumerate(tokens) if t == "pytest" or t.endswith("/pytest"))
        head, rest = tokens[: idx + 1], tokens[idx + 1 :]
        opts: list[str] = []
        skip_next = False
        for i, t in enumerate(rest):
            if skip_next:
                opts.append(t)
                skip_next = False
                continue
            if "::" in t or t.endswith(".py") or t in {"tests", "test"}:
                continue  # a selector: replaced by the failing ids
            if t in {"-q", "--quiet", "-qq"}:
                continue
            if t in {"-p", "-k", "-m", "-o", "--tb", "-c", "--rootdir"} and i + 1 < len(rest):
                opts.append(t)
                skip_next = True
                continue
            opts.append(t)
        if not any(o.startswith("--tb") for o in opts):
            opts.append("--tb=short")
        return [*head, "-vv", *opts, *failed_ids]
    return None


def extract_failure_detail(output: str, per_test: int = 2_500, total: int = 9_000) -> str:
    """The assertion detail of each failing test: its traceback head and its ``E`` lines.

    Per-test and total byte budgets; the head and the first mismatching items are kept.
    """
    text = strip_ansi(output)
    sections = re.split(r"\n_{3,}\s+(.+?)\s+_{3,}\n", text)
    parts: list[str] = []
    used = 0
    for i in range(1, len(sections) - 1, 2):
        name, body = sections[i], re.split(r"\n={3,}", sections[i + 1], maxsplit=1)[0]
        lines = body.splitlines()
        e_lines = [ln for ln in lines if ln.startswith("E ")]
        head = [ln for ln in lines if not ln.startswith("E ")][:14]
        keep = head + (["..."] if e_lines else []) + e_lines
        chunk = clip(f"--- {name} ---\n" + "\n".join(keep), per_test)
        if used + len(chunk) > total:
            parts.append(clip(chunk, max(400, total - used)))
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)


async def run_detail(
    project_root: Path,
    verify_cmds: list[str],
    failed_ids: list[str] | tuple[str, ...],
    timeout: float = 120.0,
) -> str:
    """Re-run the failing tests with full assertion output and return the extracted detail."""
    cmd = pytest_detail_command(verify_cmds, list(failed_ids)[:12])
    if cmd is None:
        return ""
    try:
        res = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return extract_failure_detail(res.stdout + "\n" + res.stderr)


def probe_fingerprint(output: str) -> str:
    """A short key for the OUTPUT of a probe run (not the script that produced it), empty when it
    holds no information (nothing printed, only refusals / timeouts).  Two runs with the same key
    told the model the same thing."""
    bodies = re.findall(r"\n=> (.*?)(?=\n\n--- experiment|\Z)", output, re.S)
    useful: list[str] = []
    for body in bodies:
        b = body.strip()
        if (
            not b
            or b == "(no output)"
            or b.startswith(("refused (", "(timed out", "(could not run"))
        ):
            continue
        useful.append(b)
    if not useful:
        return ""
    return hashlib.sha256("\n".join(useful).encode()).hexdigest()[:16]


PROBE_LOG = ".memfun/probes.jsonl"


def log_probe(project_root: Path, snippets: list[str], output: str) -> None:
    """Append one probe run (the scripts and their output) to ``.memfun/probes.jsonl``.

    A record of what the model ran, for audits (e.g. did a script look outside the project).
    Best effort: a logging failure never affects the fix loop.
    """
    try:
        path = project_root / PROBE_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "scripts": snippets, "output": output}) + "\n")
    except OSError:
        pass
