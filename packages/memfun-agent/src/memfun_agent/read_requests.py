"""Reads a solver may ask for before it edits, served by memfun's own code.

The one-shot solver sees a file that is too large only as its head, an outline and the
definitions that match the task.  When the part it must change is not shown, a careful model
does not guess: it answers with a request to read (``read_file``, or a command such as
``sed -n '760,770p' pkg/mod.py`` or ``grep -rn "name" pkg``).  This module turns such a
request into one of a few plain reads and answers it:

* lines of a file, a search of a regular expression under a path, a search for file names,
  a directory listing, a line count;
* **nothing is passed to a shell**: a command is only parsed (``shlex``) and, when it has the
  shape of a read, answered by the functions below;
* every path is confined to the project: no path outside it (absolute, ``..`` or through a
  symbolic link), nothing under ``.git`` or ``.memfun``, no file whose name says it holds a
  secret;
* the output is capped per request and per round;
* anything else is refused with a reason the model can act on.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import stat
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Most rounds of reads one solve may be configured with.
MAX_READ_ROUNDS = 6
#: Bytes one answered request may add to the prompt.
PER_REQUEST_BYTES = 16_000
#: Bytes all the requests of one round may add.
ROUND_BYTES = 40_000
#: Requests answered per round; the rest are refused (ask again).
MAX_REQUESTS = 12
#: Lines one search or listing may print.
MAX_HITS = 200
#: A file larger than this is not read or searched.
MAX_FILE_BYTES = 5_000_000
#: Seconds one search may take.
SEARCH_SECONDS = 15.0
#: Where the served reads are recorded (next to the solver's other evidence).
READ_LOG = ".memfun/reads.jsonl"

#: Never readable: version-control data (history is not the task's input) and memfun's own
#: state (settings, logs, the reproduction script).
FORBIDDEN_PARTS = frozenset({".git", ".memfun", ".hg", ".svn", ".ssh", ".aws", ".gnupg"})
#: Longest pattern that is compiled; seconds one match may take before the search is stopped.
MAX_PATTERN = 1_000
MATCH_SECONDS = 0.25
#: Requests of one reply that are looked at at all.
MAX_PARSED = 60
_MAX_NUMBER = 10**9
#: Not searched or listed by a walk (they can still be named explicitly, except the
#: forbidden ones).
_SKIP_WALK = FORBIDDEN_PARTS | frozenset(
    {
        "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".eggs", ".ssh", ".aws", ".gnupg",
    }
)  # fmt: skip
_SECRET_NAME = re.compile(
    r"(^\.env(\..*)?$|\.env$|^\.env[-_].*|^\.envrc$|^env\.local$|^credentials?(\..*)?$|"
    r"^secrets?(\..*)?$|\.pem$|\.key$|\.p12$|\.pfx$|\.jks$|\.keystore$|"
    r"^id_(rsa|dsa|ecdsa|ed25519)|^\.netrc$|^\.npmrc$|^\.pypirc$|^\.git-credentials$|"
    r"^\.htpasswd$|^\.pgpass$|^\.dockercfg$|^kubeconfig$|^service[-_]account.*\.json$)",
    re.IGNORECASE,
)
_CODE_SUFFIX = re.compile(
    r"\.(py|pyi|js|jsx|ts|tsx|go|rs|java|kt|c|cc|cpp|h|hpp|rb|php|swift|sh|sql|html|css)$", re.I
)

#: What a refusal tells the model it may ask for instead.
READ_OPS_HINT = (
    'only reading is possible in this step: {"op":"read_file","path":"...","start_line":N,'
    '"end_line":M}, {"op":"search","pattern":"regex","path":"dir or file"}, '
    '{"op":"find_files","name":"*glob*","path":"dir"}, {"op":"list_dir","path":"dir"}'
)


@dataclass(frozen=True, slots=True)
class Post:
    """One later stage of a pipeline, applied to the lines a read printed."""

    kind: str  # "head" | "tail" | "slice" | "grep" | "sort" | "uniq" | "count" | "xargs"
    n: int = 0
    m: int = 0
    pattern: str = ""
    invert: bool = False
    ignore_case: bool = False
    reverse: bool = False
    #: ``xargs grep``: the search to run in each file the earlier stages printed.
    search: Read | None = None


@dataclass(frozen=True, slots=True)
class Read:
    """One read request, already reduced to what memfun itself can answer."""

    kind: str  # "lines" | "search" | "find" | "list" | "count"
    asked: str
    path: str = "."
    #: lines: first and last line (1-based, inclusive); 0 = from the top / to the end.
    start: int = 0
    end: int = 0
    #: lines: the last N lines instead (``tail``).
    tail: int = 0
    #: search: a Python regular expression.
    pattern: str = ""
    ignore_case: bool = False
    files_only: bool = False
    #: search: lines shown before and after each matching line.
    before: int = 0
    after: int = 0
    #: search: file-name globs a file must match; find: file-name globs (any of them).
    include: tuple[str, ...] = ()
    #: find: globs on the path relative to the project; ``exclude`` drops matching paths.
    path_globs: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    ftype: str = ""  # find: "f", "d" or "" (both)
    maxdepth: int = 0
    post: tuple[Post, ...] = ()


@dataclass(frozen=True, slots=True)
class Refusal:
    """A request that is not a read memfun can serve."""

    asked: str
    why: str


@dataclass(frozen=True, slots=True)
class Served:
    """The answer to one request: what goes into the prompt, and what the log records."""

    asked: str
    kind: str  # the read's kind, or "refused"
    heading: str
    output: str
    ok: bool
    paths: tuple[str, ...] = field(default=())


# ── confinement ──────────────────────────────────────────────────────────────────────────────


def is_secret_name(name: str) -> bool:
    """A data or config file whose name says it holds a secret (source modules are code)."""
    return bool(_SECRET_NAME.search(name)) and not _CODE_SUFFIX.search(name)


def confine(root: Path, path_str: str, base: str = ".") -> Path | str:
    """*path_str* as a path inside *root*, or the reason (a ``str``) it is refused.

    Relative paths are taken from *base* (itself relative to the root).  The path is resolved
    with its symbolic links, so a link that leaves the project is refused like ``..``.
    """
    root = root.resolve()
    text = (path_str or ".").strip()
    if text.startswith("~") or "\x00" in text:
        return "is not a path in the project"
    raw = Path(text)
    candidate = raw if raw.is_absolute() else root / base / raw
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return "cannot be resolved"
    if not resolved.is_relative_to(root):
        return "is outside the project (paths are relative to the project root)"
    parts = resolved.relative_to(root).parts
    if any(p.casefold() in FORBIDDEN_PARTS for p in parts):  # .Git on a case-blind disk
        return "is version-control data or memfun's own state, which is not readable"
    if parts and is_secret_name(parts[-1]):
        return "looks like a file of secrets, which is never read"
    return resolved


def _rel(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root.resolve())
    except ValueError:
        return str(path)
    return str(rel) if rel.parts else "."


def _walk(root: Path, base: Path, deadline: float) -> Iterator[tuple[Path, list[str], list[str]]]:
    """``(directory, sub-directories, files)`` under *base* inside the project: skipped
    directories pruned, symbolic links never followed out of the project."""
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        if time.monotonic() > deadline:
            return
        here = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d.casefold() not in _SKIP_WALK
            and not d.endswith(".egg-info")
            and not (here / d).is_symlink()
        )
        yield here, dirnames, sorted(filenames)


def _inside(root: Path, path: Path) -> bool:
    """Whether a file met in a walk or a listing may be read.  A symbolic link is judged by
    where it points: inside the project, and not at version-control data, memfun's own
    state or a file of secrets (a link ``notes.txt -> ../.env`` is not readable)."""
    return not path.is_symlink() or isinstance(confine(root, str(path)), Path)


def _read_text(path: Path) -> str | None:
    """The text of *path*, or ``None`` for a binary, unreadable or oversized file."""
    try:
        st = path.stat()
        # regular files only: a named pipe or a device would block the read for ever
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", "replace")


# ── parsing: JSON operations ─────────────────────────────────────────────────────────────────


def _int(value: Any) -> int:
    """A count or line number from model text: never negative, never absurd, never raising."""
    try:
        return max(0, min(int(value), _MAX_NUMBER))
    except (TypeError, ValueError, OverflowError):
        return 0


def _strs(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list | tuple):
        return tuple(str(v) for v in cast("Sequence[object]", value) if v)
    return ()


def parse_operation(op: dict[str, Any]) -> list[Read | Refusal]:
    """The reads one operation of a solver reply asks for (an edit or a write asks for none)."""
    kind = str(op.get("op", ""))
    asked = json.dumps(op, ensure_ascii=False, sort_keys=True)[:400]
    path = str(op.get("path") or op.get("file") or op.get("dir") or op.get("directory") or ".")
    if kind in ("read_file", "read", "read_lines", "view", "cat"):
        if path == ".":
            return [Refusal(asked, "read_file needs a path")]
        start = _int(op.get("start_line") or op.get("start") or op.get("line_start"))
        end = _int(op.get("end_line") or op.get("end") or op.get("line_end"))
        lines: object = op.get("lines")
        if isinstance(lines, str) and (m := re.fullmatch(r"\s*(\d+)\s*[-,:]\s*(\d+)\s*", lines)):
            start, end = _int(m.group(1)), _int(m.group(2))
        elif isinstance(lines, list | tuple):
            pair = cast("Sequence[object]", lines)
            if len(pair) == 2:
                start, end = _int(pair[0]), _int(pair[1])
        if not start and op.get("offset") is not None:
            start = _int(op.get("offset")) or 1
        if not end and op.get("limit") is not None and _int(op.get("limit")):
            end = (start or 1) + _int(op.get("limit")) - 1
        return [Read("lines", asked, path, start=start, end=end)]
    if kind in ("search", "grep", "search_files", "search_code", "rg"):
        pattern = str(op.get("pattern") or op.get("regex") or op.get("query") or "")
        if not pattern:
            return [Refusal(asked, "search needs a pattern")]
        return [
            Read(
                "search",
                asked,
                path,
                pattern=pattern,
                ignore_case=bool(op.get("ignore_case") or op.get("case_insensitive")),
                files_only=bool(op.get("files_only") or op.get("files_with_matches")),
                before=min(_int(op.get("context")), 20),
                after=min(_int(op.get("context")), 20),
                include=_strs(op.get("include") or op.get("glob")),
            )
        ]
    if kind in ("find_files", "find", "glob", "find_file"):
        names = _strs(op.get("name") or op.get("names") or op.get("pattern") or op.get("glob"))
        if not names:
            return [Refusal(asked, "find_files needs a name glob")]
        return [Read("find", asked, path, include=names)]
    if kind in ("list_dir", "ls", "list_files", "list_directory", "list"):
        return [Read("list", asked, path)]
    if kind == "run_cmd":
        return parse_command(str(op.get("cmd") or op.get("command") or ""))
    return []


# ── parsing: a command that has the shape of a read ──────────────────────────────────────────

_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+|\$|\+\d+))?p$")
_SPLITTERS = frozenset({"&&", ";"})
_NOT_A_READ = (
    "python", "python3", "pytest", "pip", "pip3", "uv", "node", "npm", "make", "bash", "sh",
    "tox", "nox", "cargo", "go", "java", "ruby", "perl", "awk", "curl", "wget", "docker",
)  # fmt: skip


def _tokens(cmd: str) -> list[str] | None:
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError:
        return None


def _strip_redirects(tokens: list[str]) -> list[str] | str:
    """*tokens* without ``2>/dev/null``, ``2>&1`` and ``>/dev/null``; a ``str`` (the reason)
    when the command redirects anywhere else."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t == ">&" and nxt in ("1", "2"):
            if out and out[-1] in ("1", "2"):
                out.pop()
            i += 2
            continue
        if t in (">", ">>") and nxt == "/dev/null":
            if out and out[-1] in ("1", "2"):
                out.pop()
            i += 2
            continue
        if t in (">", ">>", "<", "<<", ">&", "&>", "<<<"):
            return "it redirects to or from a file"
        if t == "&":
            return "it starts a background job"
        out.append(t)
        i += 1
    return out


def _bre(pattern: str) -> str:
    """A basic regular expression (plain ``grep``) as a Python one: there ``( ) { } | + ?``
    are literal and their backslashed forms are the operators."""
    out: list[str] = []
    i, in_class = 0, False
    while i < len(pattern):
        c = pattern[i]
        if in_class:
            out.append(c)
            in_class = c != "]"
        elif c == "[":
            out.append(c)
            in_class = True
            if pattern[i + 1 : i + 2] == "]":  # "[]...]": a leading "]" is literal
                out.append("]")
                i += 1
        elif c == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            out.append(nxt if nxt in "(){}|+?" else c + nxt)
            i += 1
        elif c in "(){}|+?":
            out.append("\\" + c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _number(text: str) -> int | None:
    return int(text) if text.isdecimal() and len(text) <= 9 else None


def _parse_grep(
    args: list[str], asked: str, base: str, *, tool: str, stage: bool = False
) -> list[Read | Refusal] | Post:
    """``grep``/``rg``/``git grep`` as searches; with *stage* a ``| grep`` filter (a Post)."""
    extended = tool != "grep"
    fixed = word = ignore_case = files_only = invert = count = False
    before = after = 0
    patterns: list[str] = []
    include: list[str] = []
    positional: list[str] = []

    def refuse(why: str) -> list[Read | Refusal]:
        return [Refusal(asked, why)]

    i = 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if a == "--":
            positional.extend(args[i + 1 :])
            break
        if a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            takes = name in (
                "include", "glob", "exclude", "exclude-dir", "regexp", "type",
                "after-context", "before-context", "context", "max-count", "max-depth",
            )  # fmt: skip
            if takes and not eq:
                val, i = nxt, i + 1
            if name in ("include", "glob"):
                include.append(val)
            elif name == "type":
                include.append(f"*.{val}")
            elif name == "regexp":
                patterns.append(val)
            elif name in ("after-context", "before-context", "context"):
                if name != "after-context":
                    before = max(before, _number(val) or 0)
                if name != "before-context":
                    after = max(after, _number(val) or 0)
            elif name in ("ignore-case",):
                ignore_case = True
            elif name in ("files-with-matches",):
                files_only = True
            elif name in ("extended-regexp", "perl-regexp", "pcre2"):
                extended = True
            elif name == "fixed-strings":
                fixed = True
            elif name == "word-regexp":
                word = True
            elif name == "invert-match":
                invert = True
            elif name == "count":
                count = True
            elif name not in (
                "recursive", "line-number", "with-filename", "no-filename", "no-messages",
                "color", "colour", "no-heading", "heading", "text", "binary-files", "exclude",
                "exclude-dir", "max-count", "max-depth", "hidden", "no-ignore", "smart-case",
            ):  # fmt: skip
                return refuse(f"option --{name} is not supported; {READ_OPS_HINT}")
        elif a.startswith("-") and len(a) > 1:
            cluster = a[1:]
            j = 0
            while j < len(cluster):
                c = cluster[j]
                rest = cluster[j + 1 :]
                if c in "ABCem" or (tool == "rg" and c in "gt"):
                    val = rest
                    if not val:
                        val, i = nxt, i + 1
                    if c in "BC":
                        before = max(before, _number(val) or 0)
                    if c in "AC":
                        after = max(after, _number(val) or 0)
                    elif c == "e":
                        patterns.append(val)
                    elif c == "g":
                        include.append(val)
                    elif c == "t":
                        include.append(f"*.{val}")
                    break
                if c == "i":
                    ignore_case = True
                elif c == "l":
                    files_only = True
                elif c in "EP":
                    extended = True
                elif c == "F":
                    fixed = True
                elif c == "w":
                    word = True
                elif c == "v":
                    invert = True
                elif c == "c":
                    count = True
                elif c not in "nrRHIshaSNU.":
                    return refuse(f"option -{c} is not supported; {READ_OPS_HINT}")
                j += 1
        else:
            positional.append(a)
        i += 1
    if not patterns:
        if not positional:
            return refuse("the search has no pattern")
        patterns.append(positional.pop(0))
    parts = [re.escape(p) if fixed else (p if extended else _bre(p)) for p in patterns]
    pattern = parts[0] if len(parts) == 1 else "|".join(f"(?:{p})" for p in parts)
    if word:
        pattern = rf"\b(?:{pattern})\b"
    if stage:
        if positional:
            return refuse("a filter after | takes no file")
        if count:
            return refuse(f"option -c after | is not supported; {READ_OPS_HINT}")
        return Post("grep", pattern=pattern, invert=invert, ignore_case=ignore_case)
    if invert or count:
        return refuse(f"only plain searches are served; {READ_OPS_HINT}")
    return [
        Read(
            "search",
            asked,
            os.path.join(base, p) if base != "." and not os.path.isabs(p) else p,
            pattern=pattern,
            ignore_case=ignore_case,
            files_only=files_only,
            before=min(before, 20),
            after=min(after, 20),
            include=tuple(include),
        )
        for p in (positional or ["."])
    ]


def _under(base: str, path: str) -> str:
    return os.path.join(base, path) if base != "." and not os.path.isabs(path) else path


def _parse_lines_tool(name: str, args: list[str], asked: str, base: str) -> list[Read | Refusal]:
    """``cat``, ``head``, ``tail``, ``sed -n 'A,Bp'``, ``nl``, ``less``, ``more``, ``bat``."""
    files: list[str] = []
    start = end = tail = 0
    count: int | None = None
    from_line = 0
    script = ""
    i = 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if name == "sed":
            if a in ("-n", "--quiet", "--silent", "-E", "-r"):
                pass
            elif a == "-e":
                script, i = nxt, i + 1
            elif a.startswith("-"):
                return [Refusal(asked, f"only sed -n 'A,Bp' FILE is served; {READ_OPS_HINT}")]
            elif not script:
                script = a
            else:
                files.append(a)
        elif name in ("head", "tail") and a in ("-n", "--lines"):
            value, i = nxt, i + 1
            if name == "tail" and value.startswith("+"):
                from_line = _number(value[1:]) or 1
            else:
                count = _number(value.lstrip("-"))
        elif name in ("head", "tail") and a.startswith("--lines="):
            count = _number(a.partition("=")[2])
        elif name in ("head", "tail") and re.fullmatch(r"-n?\d+", a):
            count = _int(a.lstrip("-n"))
        elif name == "tail" and re.fullmatch(r"-n\+\d+", a):
            from_line = _int(a[3:])
        elif a.startswith("-") and len(a) > 1:
            if name in ("head", "tail") and a in ("-c", "--bytes", "-f", "--follow"):
                return [Refusal(asked, f"option {a} is not supported; {READ_OPS_HINT}")]
        else:
            files.append(a)
        i += 1
    if name == "sed":
        m = _SED_RANGE.match(script.replace(" ", ""))
        quiet = "-n" in args or "--quiet" in args or "--silent" in args
        if not m or not quiet:
            return [Refusal(asked, f"only sed -n 'A,Bp' FILE is served; {READ_OPS_HINT}")]
        start = _int(m.group(1))
        second = m.group(2)
        if second is None:
            end = start
        elif second == "$":
            end = 0
        elif second.startswith("+"):
            end = start + _int(second[1:])
        else:
            end = _int(second)
    elif name == "head":
        start, end = 1, count if count is not None else 10
    elif name == "tail":
        if from_line:
            start = from_line
        else:
            tail = count if count is not None else 10
    if not files:
        return [Refusal(asked, f"{name} needs a file here")]
    return [Read("lines", asked, _under(base, f), start=start, end=end, tail=tail) for f in files]


def _parse_find(args: list[str], asked: str, base: str) -> list[Read | Refusal]:
    paths: list[str] = []
    names: list[str] = []
    path_globs: list[str] = []
    exclude: list[str] = []
    ftype = ""
    maxdepth = 0
    negate = False
    i = 0
    while i < len(args) and not args[i].startswith("-") and args[i] not in ("(", ")", "!"):
        paths.append(args[i])
        i += 1
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if a in ("-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename"):
            fold = "(?i)" if a.startswith("-i") else ""
            by_name = a in ("-name", "-iname")
            if negate:
                exclude.append(fold + (f"*/{nxt}" if by_name else nxt))
            elif by_name:
                names.append(fold + nxt)
            else:
                path_globs.append(fold + nxt)
            negate = False
            i += 1
        elif a == "-type":
            if nxt not in ("f", "d"):
                return [Refusal(asked, f"find -type {nxt} is not supported; {READ_OPS_HINT}")]
            ftype = nxt
            i += 1
        elif a in ("-maxdepth", "-mindepth"):
            if a == "-maxdepth":
                maxdepth = _number(nxt) or 0
            i += 1
        elif a in ("-not", "!"):
            negate = True
        elif a not in ("-o", "-or", "-a", "-and", "-print", "(", ")", "-L", "-follow"):
            return [Refusal(asked, f"find {a} is not supported; {READ_OPS_HINT}")]
        i += 1
    return [
        Read(
            "find",
            asked,
            _under(base, p),
            include=tuple(names),
            path_globs=tuple(path_globs),
            exclude=tuple(exclude),
            ftype=ftype,
            maxdepth=maxdepth,
        )
        for p in (paths or ["."])
    ]


def _parse_first(stage: list[str], asked: str, base: str) -> list[Read | Refusal]:
    name = os.path.basename(stage[0])
    args = stage[1:]
    if "=" in stage[0] and not stage[0].startswith("-"):
        return [Refusal(asked, f"setting variables is not a read; {READ_OPS_HINT}")]
    if name in ("cat", "head", "tail", "sed", "nl", "less", "more", "bat"):
        return _parse_lines_tool(name, args, asked, base)
    if name in ("grep", "egrep", "fgrep", "rg", "ag", "ack"):
        if name == "fgrep":
            args = ["-F", *args]
        tool = "grep" if name in ("grep", "fgrep") else "rg"
        got = _parse_grep(args, asked, base, tool=tool)
        return got if isinstance(got, list) else [Refusal(asked, "not a search")]
    if name == "git" and args[:1] == ["grep"]:
        got = _parse_grep(args[1:], asked, base, tool="grep")
        return got if isinstance(got, list) else [Refusal(asked, "not a search")]
    if name == "git":
        return [
            Refusal(asked, f"the repository's history is not part of the task; {READ_OPS_HINT}")
        ]
    if name == "find":
        return _parse_find(args, asked, base)
    if name in ("ls", "tree", "dir"):
        paths = [a for a in args if not a.startswith("-")] or ["."]
        flags = "".join(a for a in args if a.startswith("-"))
        kind = "find" if "R" in flags or name == "tree" else "list"
        return [Read(kind, asked, _under(base, p)) for p in paths]
    if name == "wc":
        files = [a for a in args if not a.startswith("-")]
        if not files:
            return [Refusal(asked, "wc needs a file here")]
        return [Read("count", asked, _under(base, f)) for f in files]
    if name in _NOT_A_READ or name.startswith("python"):
        return [
            Refusal(
                asked,
                "nothing is executed before the edit (your change is checked by the project's "
                f"tests afterwards); {READ_OPS_HINT}",
            )
        ]
    return [Refusal(asked, f"{name} is not a read memfun serves; {READ_OPS_HINT}")]


def _parse_post(stage: list[str], asked: str) -> Post | Refusal:
    name = os.path.basename(stage[0])
    args = stage[1:]
    if name in ("head", "tail"):
        n = 10
        for i, a in enumerate(args):
            if re.fullmatch(r"-n?\d+", a):
                n = _int(a.lstrip("-n"))
            elif a in ("-n", "--lines") and i + 1 < len(args) and args[i + 1].isdecimal():
                n = _int(args[i + 1])
            elif not a.isdecimal():
                return Refusal(asked, f"{name} {a} after | is not supported; {READ_OPS_HINT}")
        return Post(name, n=n)
    if name in ("grep", "egrep", "fgrep", "rg"):
        if name == "fgrep":
            args = ["-F", *args]
        got = _parse_grep(
            args, asked, ".", tool="grep" if name in ("grep", "fgrep") else "rg", stage=True
        )
        if isinstance(got, Post):
            return got
        first = got[0]
        return first if isinstance(first, Refusal) else Refusal(asked, "not a filter")
    if name == "sort":
        if any(a.startswith("-") and set(a[1:]) - set("urn") for a in args):
            return Refusal(asked, f"sort with these options is not supported; {READ_OPS_HINT}")
        flags = "".join(a[1:] for a in args if a.startswith("-"))
        return Post("sort", reverse="r" in flags, invert="u" in flags)
    if name == "uniq" and not args:
        return Post("uniq")
    if name == "wc" and args in (["-l"], ["--lines"]):
        return Post("count")
    if name == "sed" and len(args) == 2 and args[0] == "-n":
        m = _SED_RANGE.match(args[1].replace(" ", ""))
        if m and (m.group(2) or "").isdigit():
            return Post("slice", n=_int(m.group(1)), m=_int(m.group(2)))
    if name == "xargs":
        rest = [a for a in args if a not in ("-r", "-0", "--no-run-if-empty", "-d", "\\n")]
        if rest and os.path.basename(rest[0]) in ("grep", "egrep", "rg"):
            tool = "grep" if os.path.basename(rest[0]) == "grep" else "rg"
            # The files come from the earlier stages, so every argument here is an option or
            # the pattern: parse it as a search of "." and keep only the pattern and switches.
            got = _parse_grep(rest[1:], asked, ".", tool=tool)
            if isinstance(got, list) and len(got) == 1 and isinstance(got[0], Read):
                return Post("xargs", search=got[0])
        return Refusal(asked, f"only xargs grep is served after |; {READ_OPS_HINT}")
    return Refusal(asked, f"{name} after | is not supported; {READ_OPS_HINT}")


def parse_command(cmd: str) -> list[Read | Refusal]:
    """The reads a shell command asks for.  The command is parsed, never run: statements
    joined by ``&&`` or ``;`` are separate requests, a ``cd`` moves the base directory of
    the ones after it, and later pipeline stages become filters on the printed lines."""
    asked = cmd.strip()[:400]
    if not asked:
        return []
    if "`" in cmd or "$(" in cmd or "<(" in cmd:
        return [Refusal(asked, f"it runs a command inside the command; {READ_OPS_HINT}")]
    tokens = _tokens(cmd)
    if tokens is None:
        return [Refusal(asked, "its quotes do not close")]
    clean = _strip_redirects(tokens)
    if isinstance(clean, str):
        return [Refusal(asked, f"{clean}; {READ_OPS_HINT}")]
    statements: list[list[str]] = [[]]
    for t in clean:
        if t in _SPLITTERS:
            statements.append([])
        else:
            statements[-1].append(t)
    out: list[Read | Refusal] = []
    base = "."
    for st in statements:
        if "||" in st:  # "cmd || echo not found": the fallback is not a read
            st = st[: st.index("||")]
        if not st:
            continue
        head = os.path.basename(st[0])
        if head == "cd":
            if len(st) != 2:
                out.append(Refusal(asked, "cd wants one directory"))
                break
            base = st[1] if os.path.isabs(st[1]) or base == "." else os.path.join(base, st[1])
            continue
        if head in ("echo", "true", "printf", "pwd"):
            continue
        stages: list[list[str]] = [[]]
        for t in st:
            if t == "|":
                stages.append([])
            else:
                stages[-1].append(t)
        if any(not s for s in stages) or any(t in ("(", ")") for t in stages[0][:1]):
            out.append(Refusal(asked, f"it is not a plain command; {READ_OPS_HINT}"))
            continue
        label = " ".join(shlex.quote(t) if t != "|" else t for t in st)[:400]
        first = _parse_first(stages[0], label, base)
        posts: list[Post] = []
        refusal: Refusal | None = None
        for stage in stages[1:]:
            got = _parse_post(stage, label)
            if isinstance(got, Refusal):
                refusal = got
                break
            posts.append(got)
        if refusal is not None:
            out.append(refusal)
            continue
        out.extend(
            replace(item, post=tuple(posts)) if isinstance(item, Read) and posts else item
            for item in first
        )
    return out


# ── serving ──────────────────────────────────────────────────────────────────────────────────


def _clip_lines(lines: list[str], limit: int) -> tuple[list[str], int]:
    """As many whole *lines* as fit in *limit* bytes, and how many that is."""
    used = 0
    for i, line in enumerate(lines):
        used += len(line.encode("utf-8", "replace")) + 1
        if used > limit:
            return lines[:i], i
    return lines, len(lines)


class PatternTooSlowError(Exception):
    """A match did not finish in time: the pattern backtracks without end on this text."""


class _Matcher:
    """A pattern from model text that cannot run away.

    It is compiled by the ``regex`` engine, whose every search takes a time limit (the
    standard ``re`` has none, and ``(a+)+$`` on thirty characters would hold the whole
    process).  A pattern that does not compile, is too long, or meets an installation
    without ``regex`` is searched as fixed text.
    """

    def __init__(self, pattern: str, ignore_case: bool) -> None:
        self._rx: Any = None
        self._fold = ignore_case
        self._needle = pattern.casefold() if ignore_case else pattern
        if len(pattern) > MAX_PATTERN:
            return
        try:
            import regex

            self._rx = regex.compile(pattern, regex.IGNORECASE if ignore_case else 0)
        except Exception:  # not installed, not a pattern, too deeply nested, too large a count
            self._rx = None

    def search(self, text: str, deadline: float) -> bool:
        if self._rx is None:
            return self._needle in (text.casefold() if self._fold else text)
        left = deadline - time.monotonic()
        if left <= 0:
            raise PatternTooSlowError
        try:
            return self._rx.search(text, timeout=min(MATCH_SECONDS, left)) is not None
        except TimeoutError:
            raise PatternTooSlowError from None


def _compile(pattern: str, ignore_case: bool) -> _Matcher:
    return _Matcher(pattern, ignore_case)


def _glob(name: str, glob: str) -> bool:
    if glob.startswith("(?i)"):
        return fnmatch.fnmatchcase(name.lower(), glob[4:].lower())
    return fnmatch.fnmatchcase(name, glob)


def _serve_lines(root: Path, r: Read, limit: int) -> Served:
    path = confine(root, r.path)
    if isinstance(path, str):
        return _refused(r.asked, f"{r.path} {path}")
    rel = _rel(root, path)
    if path.is_dir():
        return _refused(r.asked, f"{rel} is a directory: list it with list_dir")
    if not path.is_file():
        return _refused(r.asked, f"{rel} does not exist (find it with find_files or search)")
    text = _read_text(path)
    if text is None:
        return _refused(r.asked, f"{rel} is binary or too large to read")
    lines = text.splitlines()
    n = len(lines)
    start = max(1, n - r.tail + 1) if r.tail else max(1, r.start or 1)
    end = n if r.tail else max(start, min(n, r.end or n))
    if start > n:
        return _refused(r.asked, f"{rel} has only {n} lines")
    chosen, kept = _clip_lines(lines[start - 1 : end], limit)
    last = start + kept - 1
    body = "\n".join(chosen)
    if last < end:
        body += (
            f"\n... (cut after line {last} of {n} to keep the reply small: ask for lines "
            f"{last + 1}-{end} if you need them)"
        )
    heading = (
        f"{rel} (whole file, {n} lines)"
        if start == 1 and last >= n
        else f"lines {start}-{last} of {rel} ({n} lines in the file)"
    )
    return Served(r.asked, "lines", heading, body, True, (rel,))


def _search_file(
    rel: str,
    text: str,
    rx: _Matcher,
    around: tuple[int, int],
    hits: list[str],
    room: int,
    deadline: float,
) -> int:
    """Append the matches of one file (``path:line: text``; context lines with ``-``)."""
    lines = text.splitlines()
    matched = [i for i, line in enumerate(lines) if rx.search(line[:2_000], deadline)]
    shown: set[int] = set()
    for i in matched[:room]:
        for j in range(max(0, i - around[0]), min(len(lines), i + around[1] + 1)):
            if j in shown:
                continue
            shown.add(j)
            sep = ":" if j == i or j in matched else "-"
            hits.append(f"{rel}:{j + 1}{sep} {lines[j][:240]}")
    return len(matched)


def _serve_search(root: Path, r: Read, files: list[Path] | None = None) -> Served:
    rx = _compile(r.pattern, r.ignore_case)
    deadline = time.monotonic() + SEARCH_SECONDS
    if files is None:
        base = confine(root, r.path)
        if isinstance(base, str):
            return _refused(r.asked, f"{r.path} {base}")
        if not base.exists():
            return _refused(r.asked, f"{_rel(root, base)} does not exist")
        if base.is_file():
            files = [base]
        else:
            files = [
                here / name
                for here, _dirs, names in _walk(root, base, deadline)
                for name in names
                if (not r.include or any(_glob(name, g) for g in r.include))
                and not is_secret_name(name)
                and _inside(root, here / name)
            ]
    hits: list[str] = []
    total = n_files = 0
    cut = slow = False
    for path in files:
        if time.monotonic() > deadline or len(hits) >= MAX_HITS:
            cut = True
            break
        text = _read_text(path)
        if text is None:
            continue
        rel = _rel(root, path)
        try:
            if r.files_only:
                if any(rx.search(line[:2_000], deadline) for line in text.splitlines()):
                    n_files += 1
                    total += 1
                    hits.append(rel)
                continue
            room = MAX_HITS - len(hits)
            found = _search_file(rel, text, rx, (r.before, r.after), hits, room, deadline)
        except PatternTooSlowError:
            slow = True
            break
        if found:
            n_files += 1
            total += found
            cut = cut or found > room
    where = (
        _rel(root, files[0]) if len(files) == 1 else (r.path if r.path != "." else "the project")
    )
    heading = f'search "{r.pattern[:200]}" in {where}: ' + (
        f"{n_files} files" if r.files_only else f"{total} matches in {n_files} files"
    )
    if slow:
        heading += " (stopped: the pattern takes too long to match here, use a simpler one)"
    elif cut:
        heading += f" (stopped at {MAX_HITS} lines: narrow the pattern or the path)"
    return Served(r.asked, "search", heading, "\n".join(hits) or "(no match)", True)


def _serve_find(root: Path, r: Read) -> Served:
    base = confine(root, r.path)
    if isinstance(base, str):
        return _refused(r.asked, f"{r.path} {base}")
    if not base.is_dir():
        return _refused(r.asked, f"{_rel(root, base)} is not a directory")
    deadline = time.monotonic() + SEARCH_SECONDS
    depth0 = len(base.parts)
    found: list[str] = []
    for here, dirs, names in _walk(root, base, deadline):
        level = len(here.parts) - depth0 + 1
        entries = [(n, False) for n in names] if r.ftype != "d" else []
        if r.ftype != "f":
            entries += [(d, True) for d in dirs]
        for name, is_dir in sorted(entries):
            rel = _rel(root, here / name)
            if r.include and not any(_glob(name, g) for g in r.include):
                continue
            if r.path_globs and not any(
                _glob(rel, g) or _glob("./" + rel, g) for g in r.path_globs
            ):
                continue
            if any(_glob(rel, g) or _glob("./" + rel, g) for g in r.exclude):
                continue
            if not is_dir and (is_secret_name(name) or not _inside(root, here / name)):
                continue
            found.append(rel + ("/" if is_dir else ""))
        if r.maxdepth and level >= r.maxdepth:
            dirs[:] = []
        if len(found) > MAX_HITS * 5:
            break
    what = (" or ".join(r.include + r.path_globs) or "everything")[:200]
    heading = f"files named {what} under {_rel(root, base)}: {len(found)}"
    return Served(r.asked, "find", heading, "\n".join(sorted(found)) or "(none)", True)


def _serve_list(root: Path, r: Read) -> Served:
    base = confine(root, r.path)
    if isinstance(base, str):
        return _refused(r.asked, f"{r.path} {base}")
    rel = _rel(root, base)
    if base.is_file():
        return Served(
            r.asked, "list", f"{rel} is a file", f"{rel} ({base.stat().st_size} bytes)", True
        )
    if not base.is_dir():
        return _refused(r.asked, f"{rel} does not exist")
    rows: list[str] = []
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return _refused(r.asked, f"{rel} cannot be listed")
    for entry in entries:
        if entry.name.casefold() in FORBIDDEN_PARTS or is_secret_name(entry.name):
            continue
        if entry.is_dir() and not entry.is_symlink():
            rows.append(entry.name + "/")
        elif _inside(root, entry) and entry.is_file():
            rows.append(f"{entry.name} ({entry.stat().st_size} bytes)")
    return Served(r.asked, "list", f"directory {rel}: {len(rows)} entries", "\n".join(rows), True)


def _serve_count(root: Path, r: Read) -> Served:
    path = confine(root, r.path)
    if isinstance(path, str):
        return _refused(r.asked, f"{r.path} {path}")
    text = _read_text(path) if path.is_file() else None
    if text is None:
        return _refused(r.asked, f"{_rel(root, path)} is not a readable text file")
    rel = _rel(root, path)
    return Served(
        r.asked, "count", f"line count of {rel}", f"{len(text.splitlines())} {rel}", True
    )


def _refused(asked: str, why: str) -> Served:
    return Served(asked, "refused", f"not served: {asked[:300]}", why, False)


def _apply_post(root: Path, served: Served, posts: tuple[Post, ...]) -> Served:
    lines = served.output.splitlines()
    for p in posts:
        if p.kind == "head":
            lines = lines[: p.n]
        elif p.kind == "tail":
            lines = lines[-p.n :] if p.n else []
        elif p.kind == "slice":
            lines = lines[max(0, p.n - 1) : p.m]
        elif p.kind == "grep":
            rx = _compile(p.pattern, p.ignore_case)
            until = time.monotonic() + SEARCH_SECONDS
            lines = [ln for ln in lines if rx.search(ln[:2_000], until) != p.invert]
        elif p.kind == "sort":
            lines = sorted(set(lines) if p.invert else lines, reverse=p.reverse)
        elif p.kind == "uniq":
            lines = [ln for i, ln in enumerate(lines) if i == 0 or ln != lines[i - 1]]
        elif p.kind == "count":
            lines = [str(len(lines))]
        elif p.kind == "xargs" and p.search is not None:
            files: list[Path] = []
            for ln in lines:
                name = ln.split(":", 1)[0].strip().rstrip("/")
                got = confine(root, name) if 0 < len(name) < 1_000 else ""
                if isinstance(got, Path) and got.is_file() and got not in files:
                    files.append(got)
                if len(files) >= 2_000:
                    break
            if not files:
                return replace(served, output="(no file to search)")
            again = _serve_search(root, replace(p.search, asked=served.asked), files)
            served = replace(served, kind=again.kind, heading=again.heading)
            lines = again.output.splitlines()
    return replace(served, output="\n".join(lines) or "(nothing left after the filters)")


def serve_one(root: Path, item: Read | Refusal, limit: int = PER_REQUEST_BYTES) -> Served:
    """Answer one request inside *root*; a refusal is an answer too (its reason)."""
    root = root.resolve()
    if isinstance(item, Refusal):
        return _refused(item.asked, item.why)
    try:
        if item.kind == "lines":
            served = _serve_lines(root, item, limit)
        elif item.kind == "search":
            served = _serve_search(root, item)
        elif item.kind == "find":
            served = _serve_find(root, item)
        elif item.kind == "list":
            served = _serve_list(root, item)
        elif item.kind == "count":
            served = _serve_count(root, item)
        else:
            return _refused(item.asked, f"unknown read; {READ_OPS_HINT}")
        if served.ok and item.post:
            served = _apply_post(root, served, item.post)
    except PatternTooSlowError:
        return _refused(item.asked, "the pattern takes too long to match: use a simpler one")
    except Exception as exc:  # whatever model text can provoke is a refusal, never a crash
        return _refused(item.asked, f"could not be read ({type(exc).__name__})")
    if served.ok and (item.kind != "lines" or item.post):
        lines, kept = _clip_lines(served.output.splitlines(), limit)
        if kept < len(served.output.splitlines()):
            lines.append(f"... (cut at {kept} lines to keep the reply small: narrow the request)")
            served = replace(served, output="\n".join(lines))
    return served


def serve(
    root: Path,
    items: list[Read | Refusal],
    *,
    per_request: int = PER_REQUEST_BYTES,
    budget: int = ROUND_BYTES,
) -> list[Served]:
    """Answer the requests of one round within *budget* bytes."""
    out: list[Served] = []
    left = budget
    seen: set[tuple[Any, ...]] = set()
    for n, item in enumerate(items):
        key = (item,)
        if key in seen:
            continue
        seen.add(key)
        if n >= MAX_REQUESTS or left < 400:
            # one line for all the rest: the refusals must not outgrow the budget themselves
            out.append(
                _refused(
                    f"{len(items) - n} more request(s), starting with {item.asked[:120]}",
                    "this round's reading budget is used up: ask again for what you still need",
                )
            )
            break
        served = serve_one(root, item, min(per_request, left))
        left -= len(served.output.encode("utf-8", "replace")) + len(served.heading) + 16
        out.append(served)
    return out


def requests_of(operations: list[dict[str, Any]]) -> list[Read | Refusal]:
    """Every read the *operations* of one reply ask for (the first :data:`MAX_PARSED`).
    Model text that cannot be read as a request is a refusal, never an exception."""
    out: list[Read | Refusal] = []
    for op in operations:
        try:
            out.extend(parse_operation(op))
        except Exception as exc:
            out.append(Refusal(str(op)[:200], f"is not a readable request ({type(exc).__name__})"))
        if len(out) >= MAX_PARSED:
            return out[:MAX_PARSED]
    return out


def _printable(text: str) -> str:
    """*text* without what cannot be encoded (a lone surrogate would break the next call)."""
    return text.encode("utf-8", "replace").decode("utf-8")


def format_served(served: list[Served]) -> str:
    """The answers as prompt text: one headed block per request."""
    return _printable(
        "\n\n".join(
            f"--- {s.heading[:400]} ---\n{s.output}"
            if s.ok
            else f"--- {s.heading[:400]} ---\nREFUSED: {s.output}"
            for s in served
        )
    )


def log_reads(project_root: Path, round_no: int, served: list[Served]) -> None:
    """Append what was asked and served to ``.memfun/reads.jsonl`` (an audit record: every
    read of the step, with the refusals).  Best effort: logging never affects the solve."""
    try:
        path = project_root / READ_LOG
        if path.parent.is_symlink() or path.is_symlink():
            return  # a log that would be written through a link somebody put there: not written
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="backslashreplace") as fh:
            for s in served:
                fh.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "round": round_no,
                            "asked": s.asked[:400],
                            "kind": s.kind,
                            "served": s.ok,
                            "answer": (s.heading if s.ok else s.output)[:400],
                            "bytes": len(s.output.encode("utf-8", "replace")) if s.ok else 0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    except Exception:  # best effort: whatever goes wrong here must not reach the solve
        return
