"""Repository-wide search for file localization.

A task on a real repository names things (functions, classes, options, error messages) long
before anyone knows which of a thousand files to read.  These helpers find the files by
searching ALL source files, with general rules only:

* :func:`issue_terms` -- the code-like names and quoted messages in a task text;
* :func:`rank_files` -- every source file scored by which of those names it defines or
  mentions (rare names count more than common ones), tests kept apart from sources;
* :func:`grep` -- regular-expression search over the whole repository (the planner's own
  search patterns);
* :func:`focused_excerpt` -- for a file too large to show whole: the functions and classes
  that match, with line numbers, instead of only the head of the file.

Pure functions over file contents: no LLM, no subprocess, no network.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Words that look like identifiers but say nothing about where the code is.
_STOP = frozenset(keyword.kwlist) | frozenset(
    [
        "self",
        "cls",
        "none",
        "true",
        "false",
        "null",
        "args",
        "kwargs",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "tuple",
        "set",
        "bytes",
        "object",
        "type",
        "len",
        "print",
        "repr",
        "isinstance",
        "super",
        "init",
        "main",
        "name",
        "file",
        "path",
        "value",
        "values",
        "key",
        "keys",
        "item",
        "items",
        "data",
        "result",
        "results",
        "error",
        "errors",
        "exception",
        "warning",
        "test",
        "tests",
        "example",
        "code",
        "function",
        "method",
        "module",
        "package",
        "python",
        "version",
        "issue",
        "bug",
        "fix",
        "expected",
        "actual",
        "output",
        "input",
        "traceback",
        "line",
        "most",
        "recent",
        "call",
        "last",
        "http",
        "https",
        "www",
        "com",
        "org",
        "html",
        "json",
        "yaml",
        "toml",
        "the",
        "and",
        "for",
        "not",
        "with",
        "this",
        "that",
        "from",
        "when",
        "then",
        "have",
        "has",
        "should",
        "would",
        "could",
        "does",
        "want",
        "need",
        "get",
        "use",
    ]
)
_BACKTICK = re.compile(r"`([^`\n]{2,200})`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_PATHISH = re.compile(r"[\w./-]*[\w-]+\.(?:py|pyi|pyx|toml|cfg|ini|json|ya?ml|md|rst|txt|js|ts)\b")
_QUOTED = re.compile(r"""(?:"([^"\n]{8,80})"|'([^'\n]{8,80})')""")
_FRAME = re.compile(r'File "([^"\n]+)", line \d+, in (\w+)')
_TEST_NAME = re.compile(r"(^test_.*\.py$|.*_test\.py$|^conftest\.py$|^tests?\.py$)")
#: A source file larger than this is not searched (generated data, vendored bundles).
MAX_SEARCH_BYTES = 1_000_000
MAX_TERMS = 40


def _codeish(word: str) -> bool:
    """Whether *word* has the shape of a code name: snake_case, CamelCase, CONSTANT or an
    ``...Error`` / ``...Exception`` / ``...Warning`` class."""
    if "_" in word.strip("_"):
        return True
    if re.search(r"[a-z][A-Z]", word) or re.fullmatch(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+", word):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9]{3,}", word):
        return True
    return bool(re.search(r"(Error|Exception|Warning)$", word))


def issue_terms(text: str, limit: int = MAX_TERMS) -> list[str]:
    """The names worth searching a repository for, most specific first.

    Taken from *text* by shape, never by meaning: everything inside backticks, names in
    traceback frames, file paths, dotted names, identifiers that look like code, and quoted
    messages (searched as phrases).  Plain English words are left out.
    """
    strong: dict[str, None] = {}
    weak: dict[str, None] = {}

    def add(word: str, bucket: dict[str, None]) -> None:
        w = word.strip("_")
        if len(w) >= 3 and w.lower() not in _STOP and not w.isdigit():
            bucket.setdefault(word, None)

    for path, func in _FRAME.findall(text):
        add(Path(path).stem, strong)
        add(func, strong)
    for span in _BACKTICK.findall(text):
        for dotted in _DOTTED.findall(span):
            for part in dotted.split("."):
                add(part, strong)
        for word in _IDENT.findall(span):
            add(word, strong)
    for match in _PATHISH.findall(text):
        stem = Path(match).stem
        if "/" in match or _codeish(stem):
            add(stem, strong)
    for dotted in _DOTTED.findall(text):
        parts = dotted.split(".")
        if any(_codeish(p) for p in parts) or len(parts) >= 3:
            for part in parts:
                add(part, weak)
    for word in _IDENT.findall(text):
        if _codeish(word):
            add(word, strong if word in strong else weak)
    phrases = [a or b for a, b in _QUOTED.findall(text)]
    phrases = [p.strip() for p in phrases if " " in p.strip() and not p.strip().startswith("http")]
    names = list(strong) + [w for w in weak if w not in strong]
    out = names[: max(0, limit - min(len(phrases), 6))] + phrases[:6]
    return list(dict.fromkeys(out))[:limit]


def is_test_path(rel_path: str) -> bool:
    p = Path(rel_path)
    return bool(_TEST_NAME.match(p.name)) or any(
        part in ("test", "tests", "testing") for part in p.parts[:-1]
    )


@dataclass(slots=True)
class FileHit:
    """One file's matches for a set of terms."""

    path: str
    is_test: bool
    score: float = 0.0
    counts: dict[str, int] = field(default_factory=lambda: {})
    defines: list[str] = field(default_factory=lambda: [])
    lines: list[tuple[int, str]] = field(default_factory=lambda: [])


def _read(root: Path, rel: str) -> str:
    try:
        path = root / rel
        if path.stat().st_size > MAX_SEARCH_BYTES:
            return ""
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return ""


def rank_files(
    project_root: str | Path, files: list[str], terms: list[str], *, lines_per_file: int = 4
) -> list[FileHit]:
    """Every file of *files* (relative paths) that mentions a term, best first.

    A term counts by rarity (``log`` of the share of files that do NOT have it), a definition
    (``def`` / ``class`` / top-level assignment of the name) counts three times a mention, and a
    file whose own name is a term gets a bonus.  Deterministic for equal scores (by path).
    """
    root = Path(project_root).resolve()
    names = [t for t in terms if re.fullmatch(r"\w+", t)]
    phrases = [t for t in terms if t not in names]
    if not names and not phrases:
        return []
    word_re = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b") if names else None
    def_re = (
        re.compile(
            r"^[ \t]*(?:(?:async[ \t]+)?def|class)[ \t]+(" + "|".join(map(re.escape, names))
            + r")\b|^(" + "|".join(map(re.escape, names)) + r")[ \t]*(?::[^=\n]+)?=[^=]",
            re.M,
        )
        if names
        else None
    )  # fmt: skip
    hits: list[FileHit] = []
    doc_freq: dict[str, int] = {}
    for rel in files:
        text = _read(root, rel)
        if not text:
            continue
        hit = FileHit(path=rel, is_test=is_test_path(rel))
        if word_re is not None:
            for m in word_re.finditer(text):
                hit.counts[m.group(1)] = hit.counts.get(m.group(1), 0) + 1
        for phrase in phrases:
            n = text.count(phrase)
            if n:
                hit.counts[phrase] = n
        stem = Path(rel).stem
        if stem in names and stem not in hit.counts:
            hit.counts[stem] = 0  # the file is named after a term
        if not hit.counts:
            continue
        if def_re is not None:
            hit.defines = list(dict.fromkeys(a or b for a, b in def_re.findall(text)))
        for term in hit.counts:
            doc_freq[term] = doc_freq.get(term, 0) + 1
        hit.lines = _sample_lines(text, hit, word_re, phrases, lines_per_file)
        hits.append(hit)
    total = max(len(files), 1)
    for hit in hits:
        stem = Path(hit.path).stem
        score = 0.0
        for term, n in hit.counts.items():
            idf = math.log(1 + total / doc_freq[term])
            weight = min(n, 3) + (3 if term in hit.defines else 0) + (2 if term == stem else 0)
            score += idf * weight * (1.5 if " " in term else 1.0)
        hit.score = round(score, 3)
    return sorted(hits, key=lambda h: (-h.score, h.path))


def _sample_lines(
    text: str, hit: FileHit, word_re: re.Pattern[str] | None, phrases: list[str], limit: int
) -> list[tuple[int, str]]:
    """A few matching lines: definitions first, then the first mention of each term."""
    found: list[tuple[int, int, str]] = []
    seen_terms: set[str] = set()
    for no, line in enumerate(text.splitlines(), 1):
        m = word_re.search(line) if word_re is not None else None
        term = m.group(1) if m else next((p for p in phrases if p in line), "")
        if not term:
            continue
        is_def = term in hit.defines and re.match(
            rf"\s*(?:(?:async\s+)?def|class)\s+{re.escape(term)}\b|{re.escape(term)}\s*[:=]", line
        )
        if is_def or term not in seen_terms:
            found.append((0 if is_def else 1, no, line.strip()[:160]))
            seen_terms.add(term)
    found.sort()
    return sorted((no, line) for _, no, line in found[:limit])


def format_ranked(hits: list[FileHit], *, max_files: int = 20, max_chars: int = 6_000) -> str:
    """The ranking as text for a planner: path, what it defines and mentions, sample lines."""
    parts: list[str] = []
    used = 0
    for hit in hits[:max_files]:
        mentions = ", ".join(
            f"{t} x{n}" if n else t
            for t, n in sorted(hit.counts.items(), key=lambda kv: -kv[1])[:6]
            if t not in hit.defines
        )
        head = hit.path
        if hit.defines:
            head += "  [defines: " + ", ".join(hit.defines[:6]) + "]"
        if mentions:
            head += "  [mentions: " + mentions + "]"
        block = "\n".join([head, *[f"  {no}: {line}" for no, line in hit.lines]])
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 1
    return "\n".join(parts)


def grep(
    project_root: str | Path,
    files: list[str],
    patterns: list[str],
    *,
    max_hits: int = 40,
    per_pattern: int = 12,
    per_file: int = 3,
) -> tuple[list[tuple[str, int, str]], dict[str, int]]:
    """Search *files* for each regular expression of *patterns* (an invalid one is searched as
    literal text).  Returns ``(hits, files_by_hit_count)`` where a hit is
    ``(path, line number, line)``; source files are reported before tests."""
    root = Path(project_root).resolve()
    compiled: list[re.Pattern[str]] = []
    for pat in patterns[:8]:
        if not pat or len(pat) > 200:
            continue
        try:
            compiled.append(re.compile(pat))
        except re.error:
            compiled.append(re.compile(re.escape(pat)))
    hits: list[tuple[str, int, str]] = []
    file_counts: dict[str, int] = {}
    per_pat = [0] * len(compiled)
    if not compiled:
        return hits, file_counts
    ordered = sorted(files, key=lambda f: (is_test_path(f), f))
    for rel in ordered:
        text = _read(root, rel)
        if not text or not any(rx.search(text) for rx in compiled):
            continue
        in_file = 0
        for no, line in enumerate(text.splitlines(), 1):
            for i, rx in enumerate(compiled):
                if rx.search(line):
                    file_counts[rel] = file_counts.get(rel, 0) + 1
                    if in_file < per_file and per_pat[i] < per_pattern and len(hits) < max_hits:
                        hits.append((rel, no, line.strip()[:200]))
                        in_file += 1
                        per_pat[i] += 1
                    break
    return hits, file_counts


def format_grep(hits: list[tuple[str, int, str]]) -> str:
    return "\n".join(f"{path}:{no}: {line}" for path, no, line in hits)


def directory_overview(files: list[str], *, max_chars: int = 1_500) -> str:
    """Directories with their file counts (two levels), so a planner sees the layout."""
    counts: dict[str, int] = {}
    for rel in files:
        parts = Path(rel).parts[:-1]
        key = "/".join(parts[:2]) if parts else "."
        counts[key] = counts.get(key, 0) + 1
    lines = [
        f"{d}/ ({n} files)" for d, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    out: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) > max_chars:
            out.append(f"... and {len(lines) - len(out)} more directories")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def focused_excerpt(content: str, terms: list[str], *, budget: int = 24_000) -> str:
    """The functions and classes of Python *content* that define or mention a term, each with
    its line range, within *budget* characters.  Empty when nothing matches or the file does
    not parse.  The innermost matching definitions are shown (a method, not its whole class).
    """
    names = [t for t in terms if re.fullmatch(r"\w+", t)]
    if not names:
        return ""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return ""
    word_re = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    src_lines = content.splitlines()
    spans: list[tuple[int, int, int, str]] = []  # (-priority, start, end, qualified name)

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            start = min([child.lineno, *[d.lineno for d in child.decorator_list]])
            end = getattr(child, "end_lineno", None) or child.lineno
            qual = f"{prefix}{child.name}"
            body = "\n".join(src_lines[start - 1 : end])
            if isinstance(child, ast.ClassDef):
                before = len(spans)
                visit(child, qual + ".")
                if len(spans) == before and (child.name in names or word_re.search(body)):
                    spans.append((-(2 if child.name in names else 1), start, end, qual))
            elif child.name in names:
                spans.append((-2, start, end, qual))
            elif word_re.search(body):
                spans.append((-1, start, end, qual))

    visit(tree, "")
    parts: list[str] = []
    used = 0
    chosen: list[tuple[int, int, str]] = []
    for _prio, start, end, qual in sorted(spans):
        size = sum(len(ln) + 1 for ln in src_lines[start - 1 : end])
        if used + size > budget:
            continue
        chosen.append((start, end, qual))
        used += size
    for start, end, qual in sorted(chosen):
        parts.append(
            f"# --- {qual} (lines {start}-{end}) ---\n" + "\n".join(src_lines[start - 1 : end])
        )
    return "\n\n".join(parts)
