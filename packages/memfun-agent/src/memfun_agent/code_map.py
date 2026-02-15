"""Code map: extract top-level definitions from source files.

Produces an Aider-style compact index of classes, functions, and methods
with their signatures.  The planner uses this to select relevant files
without reading every file in the project.

Extraction strategies:
- Python: stdlib ``ast`` module (zero dependencies, accurate)
- JS/TS: regex patterns (covers ~80-90% of common patterns)
- Go/Rust/Java: simple regex (basic coverage)
- Other: path + size only (no definitions)
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from memfun_core.logging import get_logger

logger = get_logger("agent.code_map")

# ── Data Structures ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Definition:
    """A single code definition (class, function, or method)."""

    name: str
    kind: str  # "class", "function", "method"
    signature: str  # e.g. "def foo(x: int) -> bool"
    line: int


@dataclass(frozen=True, slots=True)
class FileMap:
    """All definitions extracted from a single file."""

    path: str  # relative path
    definitions: tuple[Definition, ...] = ()
    size: int = 0


# ── Python Extraction (stdlib ast) ────────────────────────────

_MAX_ANNOTATION_LEN = 60


def _truncate_annotation(text: str) -> str:
    """Cap annotation string length."""
    if len(text) <= _MAX_ANNOTATION_LEN:
        return text
    return text[: _MAX_ANNOTATION_LEN - 3] + "..."


def _format_arg(arg: ast.arg) -> str:
    """Format a single function argument."""
    name = arg.arg
    if arg.annotation is not None:
        try:
            ann = ast.unparse(arg.annotation)
            return f"{name}: {_truncate_annotation(ann)}"
        except Exception:
            pass
    return name


def _format_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a function/method signature compactly."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    args_parts: list[str] = []
    all_args = node.args

    # positional args
    for arg in all_args.posonlyargs:
        args_parts.append(_format_arg(arg))
    for arg in all_args.args:
        args_parts.append(_format_arg(arg))

    # *args
    if all_args.vararg:
        args_parts.append(f"*{_format_arg(all_args.vararg)}")

    # keyword-only args
    for arg in all_args.kwonlyargs:
        args_parts.append(_format_arg(arg))

    # **kwargs
    if all_args.kwarg:
        args_parts.append(f"**{_format_arg(all_args.kwarg)}")

    args_str = ", ".join(args_parts)

    # Return annotation
    ret = ""
    if node.returns is not None:
        try:
            ret_text = ast.unparse(node.returns)
            ret = f" -> {_truncate_annotation(ret_text)}"
        except Exception:
            pass

    return f"{prefix} {node.name}({args_str}){ret}"


def _format_class_sig(node: ast.ClassDef) -> str:
    """Format a class signature with base classes."""
    bases: list[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            bases.append("?")
    if bases:
        return f"class {node.name}({', '.join(bases)})"
    return f"class {node.name}"


def _extract_python(source: str, path: str, size: int) -> FileMap:
    """Extract definitions from Python source using stdlib ast."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileMap(path=path, size=size)

    defs: list[Definition] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            sig = _format_class_sig(node)
            defs.append(Definition(
                name=node.name,
                kind="class",
                signature=sig,
                line=node.lineno,
            ))
            # Extract methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip private dunders except __init__
                    if (
                        item.name.startswith("__")
                        and item.name.endswith("__")
                        and item.name != "__init__"
                    ):
                        continue
                    msig = _format_func_sig(item)
                    defs.append(Definition(
                        name=f"{node.name}.{item.name}",
                        kind="method",
                        signature=msig,
                        line=item.lineno,
                    ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_func_sig(node)
            defs.append(Definition(
                name=node.name,
                kind="function",
                signature=sig,
                line=node.lineno,
            ))

    return FileMap(path=path, definitions=tuple(defs), size=size)


# ── JavaScript / TypeScript Extraction (regex) ───────────────

_JS_FUNCTION = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_JS_CLASS = re.compile(
    r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+([\w.]+))?",
    re.MULTILINE,
)
_JS_ARROW = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
    re.MULTILINE,
)
_TS_INTERFACE = re.compile(
    r"^(?:export\s+)?(?:interface|type)\s+(\w+)",
    re.MULTILINE,
)


def _extract_javascript(source: str, path: str, size: int) -> FileMap:
    """Extract definitions from JS/TS source using regex."""
    defs: list[Definition] = []

    for m in _JS_CLASS.finditer(source):
        name = m.group(1)
        base = m.group(2)
        sig = f"class {name}({base})" if base else f"class {name}"
        line = source[: m.start()].count("\n") + 1
        defs.append(Definition(name=name, kind="class", signature=sig, line=line))

    for m in _JS_FUNCTION.finditer(source):
        name = m.group(1)
        params = m.group(2).strip()
        sig = f"function {name}({params})"
        line = source[: m.start()].count("\n") + 1
        defs.append(Definition(name=name, kind="function", signature=sig, line=line))

    for m in _JS_ARROW.finditer(source):
        name = m.group(1)
        sig = f"const {name} = (...) =>"
        line = source[: m.start()].count("\n") + 1
        defs.append(Definition(name=name, kind="function", signature=sig, line=line))

    for m in _TS_INTERFACE.finditer(source):
        name = m.group(1)
        sig = f"interface {name}"
        line = source[: m.start()].count("\n") + 1
        defs.append(Definition(name=name, kind="class", signature=sig, line=line))

    # Deduplicate by name (arrow + function can overlap)
    seen: set[str] = set()
    unique: list[Definition] = []
    for d in defs:
        if d.name not in seen:
            seen.add(d.name)
            unique.append(d)

    return FileMap(path=path, definitions=tuple(unique), size=size)


# ── Generic Language Extraction (regex) ───────────────────────

_GO_FUNC = re.compile(
    r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_RUST_FN = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
    re.MULTILINE,
)
_JAVA_CLASS = re.compile(
    r"^(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)",
    re.MULTILINE,
)

_LANG_PATTERNS: dict[str, list[tuple[re.Pattern[str], str, str]]] = {
    # (pattern, kind, sig_template)
    "go": [
        (_GO_FUNC, "function", "func {name}({params})"),
    ],
    "rust": [
        (_RUST_FN, "function", "fn {name}({params})"),
    ],
    "java": [
        (_JAVA_CLASS, "class", "class {name}"),
    ],
}

# Map file extensions to language keys
_EXT_TO_LANG: dict[str, str] = {
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "java",  # Kotlin uses similar class patterns
    ".scala": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".pyi": "python",
}


def _extract_generic(
    source: str,
    path: str,
    size: int,
    lang: str,
) -> FileMap:
    """Extract definitions using language-specific regex patterns."""
    patterns = _LANG_PATTERNS.get(lang)
    if not patterns:
        return FileMap(path=path, size=size)

    defs: list[Definition] = []
    for pattern, kind, template in patterns:
        for m in pattern.finditer(source):
            name = m.group(1)
            params = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            sig = template.format(name=name, params=params)
            line = source[: m.start()].count("\n") + 1
            defs.append(Definition(name=name, kind=kind, signature=sig, line=line))

    return FileMap(path=path, definitions=tuple(defs), size=size)


# ── Main API ──────────────────────────────────────────────────


def _extract_file(
    file_path: Path,
    rel_path: str,
    size: int,
) -> FileMap:
    """Extract definitions from a single file."""
    ext = file_path.suffix.lower()
    lang = _EXT_TO_LANG.get(ext)

    if lang is None:
        return FileMap(path=rel_path, size=size)

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return FileMap(path=rel_path, size=size)

    if lang == "python":
        return _extract_python(source, rel_path, size)
    if lang == "javascript":
        return _extract_javascript(source, rel_path, size)
    return _extract_generic(source, rel_path, size, lang)


def build_code_map(
    project_root: str | Path,
    manifest: list[tuple[str, int]] | None = None,
    max_files: int = 500,
) -> list[FileMap]:
    """Build a code map from the project.

    Parameters
    ----------
    project_root:
        Absolute path to the project directory.
    manifest:
        Pre-built ``(relative_path, size)`` pairs.  If *None*,
        calls :func:`build_file_manifest` internally.
    max_files:
        Maximum number of files to process.
    """
    root = Path(project_root).resolve()

    if manifest is None:
        from memfun_agent.context_first import build_file_manifest
        manifest = build_file_manifest(root)

    file_maps: list[FileMap] = []
    for rel_path, size in manifest[:max_files]:
        fpath = root / rel_path
        fm = _extract_file(fpath, rel_path, size)
        file_maps.append(fm)

    return file_maps


# ── Formatting ────────────────────────────────────────────────


def _fmt_size(size: int) -> str:
    """Format byte size compactly."""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _file_importance(fm: FileMap) -> tuple[int, int]:
    """Score for sorting: (has_classes, definition_count). Higher = more important."""
    has_class = any(d.kind == "class" for d in fm.definitions)
    return (1 if has_class else 0, len(fm.definitions))


def _format_file(fm: FileMap, indent: str = "  ") -> str:
    """Format a single FileMap as compact text."""
    lines = [f"{fm.path} ({_fmt_size(fm.size)})"]
    for d in fm.definitions:
        if d.kind == "method":
            lines.append(f"{indent}{indent}{d.signature}")
        else:
            lines.append(f"{indent}{d.signature}")
    return "\n".join(lines)


def code_map_to_string(
    file_maps: list[FileMap],
    max_tokens: int = 2000,
) -> str:
    """Format the code map as a compact string for LLM context.

    Parameters
    ----------
    file_maps:
        Output from :func:`build_code_map`.
    max_tokens:
        Approximate token budget (1 token ~ 4 chars).

    Truncation strategy:
    1. Sort by importance (files with classes first, then by def count)
    2. Format file + definitions until char budget reached
    3. Remaining files: append just ``path (size)`` one-liners
    4. If still over budget: ``... and N more files``
    """
    if not file_maps:
        return ""

    max_chars = max_tokens * 4

    # Sort: files with definitions first (by importance), then alphabetically
    sorted_maps = sorted(
        file_maps,
        key=lambda fm: (_file_importance(fm), fm.path),
        reverse=True,
    )

    detailed_parts: list[str] = []
    remaining: list[FileMap] = []
    chars_used = 0

    for fm in sorted_maps:
        if fm.definitions:
            formatted = _format_file(fm)
            cost = len(formatted) + 1  # +1 for newline separator
            if chars_used + cost <= max_chars:
                detailed_parts.append(formatted)
                chars_used += cost
            else:
                remaining.append(fm)
        else:
            remaining.append(fm)

    # Add remaining files as compact one-liners
    compact_parts: list[str] = []
    for fm in remaining:
        line = f"{fm.path} ({_fmt_size(fm.size)})"
        cost = len(line) + 1
        if chars_used + cost <= max_chars:
            compact_parts.append(line)
            chars_used += cost
        else:
            # Over budget — add summary and stop
            leftover = len(remaining) - len(compact_parts)
            if leftover > 0:
                compact_parts.append(f"... and {leftover} more files")
            break

    result_parts = detailed_parts
    if compact_parts:
        result_parts = detailed_parts + compact_parts

    return "\n".join(result_parts)
