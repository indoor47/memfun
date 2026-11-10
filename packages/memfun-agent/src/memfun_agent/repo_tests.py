"""Find and run a repository's own tests.

General rules only, no knowledge of any particular repository:

* **where the tests are** comes from the project's own pytest configuration
  (``pyproject.toml``, ``pytest.ini``, ``tox.ini``, ``setup.cfg``) and from pytest's naming
  conventions (:func:`discover`);
* **which interpreter runs them** is the project's (``.venv`` or the ``python`` on ``PATH``),
  never the one memfun itself runs in, and pytest is used only if that interpreter can
  import it;
* **which tests are relevant to a change** comes from file names, from imports of the changed
  module, and from mentions of the changed functions and classes (:func:`select_tests`);
* **failures that were already there** before the change are found by running the selected
  tests on the original files (:func:`baseline_failures`) and are left out, so a fix loop only
  ever sees failures the change caused.

Pure helpers plus one bounded subprocess runner: no LLM, no network.
"""

from __future__ import annotations

import ast
import configparser
import contextlib
import difflib
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from memfun_agent import fixloop

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Directories never searched for tests.
SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", ".tox",
        ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".eggs",
        ".memfun", "site-packages",
    }
)  # fmt: skip
#: pytest's own defaults for ``python_files``.
DEFAULT_PATTERNS = ("test_*.py", "*_test.py")
#: Stop walking a huge tree after this many test files (selection only needs candidates).
MAX_TEST_FILES = 20_000
#: A test file larger than this is not read for import / symbol matching.
MAX_TEST_FILE_BYTES = 400_000
#: Seconds for the "can this interpreter import pytest" check.
IMPORT_CHECK_TIMEOUT_S = 30.0
#: Never run more than this many test files for one change.
MAX_SELECTED = 6


@dataclass(frozen=True, slots=True)
class TestSetup:
    """How a repository runs its tests (``runner`` is ``""`` when nothing usable was found)."""

    __test__ = False  # not a pytest test class

    python: str | None
    runner: str  # "pytest", "unittest" or ""
    test_files: tuple[str, ...]  # relative paths, files under the configured testpaths first
    testpaths: tuple[str, ...] = ()
    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    config_file: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.python and self.runner and self.test_files)


@dataclass(frozen=True, slots=True)
class Selected:
    """One selected test file and why it was picked."""

    path: str
    score: float
    reasons: tuple[str, ...]


# ── configuration ────────────────────────────────────────────────────────────────────────────


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str | int | float)]
    return []


def read_pytest_config(project_root: Path) -> tuple[list[str], list[str], str]:
    """``(testpaths, python_files, config_file)`` from the first pytest configuration found,
    in pytest's own order of precedence.  Empty lists when nothing is configured."""
    ini_sources = (("pytest.ini", "pytest"), ("tox.ini", "pytest"), ("setup.cfg", "tool:pytest"))
    name, section = ini_sources[0]
    path = project_root / name
    if path.is_file():
        found = _read_ini(path, section)
        if found is not None:
            return (*found, name)
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            data = {}
        opts = data.get("tool", {}).get("pytest", {}).get("ini_options")
        if isinstance(opts, dict):
            return (
                _as_list(opts.get("testpaths")),
                _as_list(opts.get("python_files")),
                "pyproject.toml",
            )
    for name, section in ini_sources[1:]:
        path = project_root / name
        if path.is_file():
            found = _read_ini(path, section)
            if found is not None:
                return (*found, name)
    return [], [], ""


def _read_ini(path: Path, section: str) -> tuple[list[str], list[str]] | None:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(path.read_text("utf-8"))
    except (OSError, configparser.Error, UnicodeDecodeError):
        return None
    if not parser.has_section(section):
        return None
    sec = parser[section]
    return _as_list(sec.get("testpaths", "")), _as_list(sec.get("python_files", ""))


# ── discovery ────────────────────────────────────────────────────────────────────────────────


def find_test_files(project_root: Path, patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> list[str]:
    """Relative paths of every file whose name matches one of pytest's *patterns*."""
    out: list[str] = []
    root = project_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            if any(fnmatch.fnmatch(fname, pat) for pat in patterns):
                out.append(str((Path(dirpath) / fname).relative_to(root)))
                if len(out) >= MAX_TEST_FILES:
                    return out
    return out


def candidate_pythons(project_root: Path) -> list[str]:
    """Interpreters that may be the project's own, most specific first; never memfun's own."""
    mine = os.path.realpath(sys.executable)
    seen: set[str] = set()
    out: list[str] = []
    local = [project_root / ".venv" / "bin" / "python", project_root / "venv" / "bin" / "python"]
    found = [str(p) for p in local if p.exists()]
    found += [w for w in (shutil.which("python"), shutil.which("python3")) if w]
    for cand in found:
        real = os.path.realpath(cand)
        if real == mine or real in seen:
            continue
        seen.add(real)
        out.append(cand)
    return out


def can_import(python: str, module: str, cwd: Path) -> bool:
    """Whether *python* can import *module* (bounded; False on any problem)."""
    try:
        res = subprocess.run(
            [python, "-c", f"import {module}"],
            cwd=str(cwd),
            capture_output=True,
            timeout=IMPORT_CHECK_TIMEOUT_S,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


def discover(project_root: str | Path, *, python: str | None = None) -> TestSetup:
    """How *project_root* runs its tests.  *python* overrides the interpreter search."""
    root = Path(project_root).resolve()
    testpaths, python_files, config_file = read_pytest_config(root)
    patterns = tuple(p for p in python_files if p.endswith(".py")) or DEFAULT_PATTERNS
    files = find_test_files(root, patterns)
    live_paths = tuple(t.rstrip("/") for t in testpaths if (root / t).exists())
    if live_paths:  # the configured test directories first
        inside = [f for f in files if any(f == t or f.startswith(t + "/") for t in live_paths)]
        files = inside + [f for f in files if f not in set(inside)]
    candidates = [python] if python else candidate_pythons(root)
    chosen, runner = None, ""
    for cand in candidates:
        if can_import(cand, "pytest", root):
            chosen, runner = cand, "pytest"
            break
    if chosen is None and candidates:
        chosen, runner = candidates[0], "unittest"
    return TestSetup(
        python=chosen,
        runner=runner if files else "",
        test_files=tuple(files),
        testpaths=live_paths,
        patterns=patterns,
        config_file=config_file,
    )


# ── what a change touched ────────────────────────────────────────────────────────────────────


def _enclosing_names(source: str, lines: set[int]) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if any(node.lineno <= ln <= end for ln in lines):
            names.add(node.name)
    return names


def changed_symbols(before: str, after: str) -> list[str]:
    """Names of the functions and classes whose body differs between *before* and *after*."""
    a, b = before.splitlines(), after.splitlines()
    old_lines: set[int] = set()
    new_lines: set[int] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old_lines.update(range(i1 + 1, i2 + 1))
        new_lines.update(range(j1 + 1, max(j2, j1 + 1) + 1))
    names = _enclosing_names(after, new_lines) | _enclosing_names(before, old_lines)
    return sorted(
        n for n in names if len(n) >= 3 and not (n.startswith("__") and n.endswith("__"))
    )


def module_names(rel_path: str) -> list[str]:
    """Dotted import names a source file may be imported under (``src/`` layouts included)."""
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    out: list[str] = []
    for skip in range(min(len(parts), 3)):
        tail = parts[skip:]
        if tail and all(p.isidentifier() for p in tail):
            out.append(".".join(tail))
    return out


# ── selection ────────────────────────────────────────────────────────────────────────────────


def _import_regex(dotted: str) -> re.Pattern[str]:
    mod = re.escape(dotted)
    head, _, leaf = dotted.rpartition(".")
    alts = [rf"(?:from|import)\s+{mod}(?![\w])"]
    if head:
        alts.append(rf"from\s+{re.escape(head)}\s+import\s+[^\n#]*\b{re.escape(leaf)}\b")
    return re.compile(r"^\s*(?:" + "|".join(alts) + ")", re.M)


def select_tests(
    project_root: str | Path,
    changed_files: list[str],
    setup: TestSetup,
    *,
    symbols: list[str] | None = None,
    max_files: int = MAX_SELECTED,
) -> list[Selected]:
    """The test files most relevant to *changed_files* (relative paths), best first.

    Three general signals, added up per test file: its **name** matches the changed module
    (``test_<module>.py``, ``<module>_test.py``); it **imports** the changed module; it
    **mentions** a changed function or class (*symbols*).  A changed file that is itself a
    test file is always selected.  Nothing is selected on a weak signal alone.
    """
    root = Path(project_root).resolve()
    tests = list(setup.test_files)
    test_set = set(tests)
    picked: dict[str, Selected] = {}
    sources: list[tuple[str, str, list[re.Pattern[str]]]] = []
    for rel in dict.fromkeys(changed_files):
        if rel in test_set:
            picked[rel] = Selected(rel, 100.0, ("is a changed test file",))
            continue
        if not rel.endswith(".py"):
            continue
        stem = Path(rel).stem
        if stem == "__init__":
            stem = Path(rel).parent.name
        if stem:
            sources.append((rel, stem, [_import_regex(m) for m in module_names(rel)]))
    sym_res = [re.compile(rf"\b{re.escape(s)}\b") for s in (symbols or [])[:12]]
    if not sources:
        return sorted(picked.values(), key=lambda s: (-s.score, s.path))[:max_files]

    for test in tests:
        if test in picked:
            continue
        base = Path(test).stem
        score = 0.0
        reasons: list[str] = []
        for rel, stem, _ in sources:
            if base in (f"test_{stem}", f"{stem}_test", f"tests_{stem}", f"{stem}_tests"):
                score += 5
                reasons.append(f"named after {rel}")
            elif re.search(rf"(?:^|_){re.escape(stem)}(?:_|$)", base):
                score += 2
                reasons.append(f"name mentions {stem}")
            shared = set(Path(rel).parent.parts) & set(Path(test).parent.parts)
            score += 0.5 * len(shared - {"src", "lib", "tests", "test"})
        text = ""
        try:
            path = root / test
            if path.stat().st_size <= MAX_TEST_FILE_BYTES:
                text = path.read_text("utf-8", errors="replace")
        except OSError:
            text = ""
        if text:
            for rel, _, regexes in sources:
                if any(rx.search(text) for rx in regexes):
                    score += 4
                    reasons.append(f"imports {rel}")
            hits = [rx.pattern[2:-2].replace("\\", "") for rx in sym_res if rx.search(text)]
            if hits:
                score += min(6, 2 * len(hits))
                reasons.append("mentions " + ", ".join(hits[:4]))
        if reasons:
            picked[test] = Selected(test, score, tuple(dict.fromkeys(reasons)))

    ranked = sorted(picked.values(), key=lambda s: (-s.score, s.path))
    strong = [s for s in ranked if s.score >= 4]
    return (strong or ranked[:2])[:max_files]


# ── running ──────────────────────────────────────────────────────────────────────────────────


#: pytest's exit code when it collected no test to run (here: every selected test is left out).
PYTEST_NO_TESTS_RAN = 5


def test_command(
    setup: TestSetup, test_files: list[str], deselect: list[str] | tuple[str, ...] = ()
) -> list[str]:
    """The argv that runs *test_files* with the project's interpreter (``[]`` if unusable)."""
    if not setup.python or not test_files:
        return []
    if setup.runner == "pytest":
        return [
            setup.python, "-m", "pytest", "-q", "--no-header", "-rfE", "--tb=short",
            "-p", "no:cacheprovider", "--continue-on-collection-errors",
            *test_files, *[f"--deselect={d}" for d in deselect],
        ]  # fmt: skip
    if setup.runner == "unittest":
        return [setup.python, "-m", "unittest", "-v", *test_files]
    return []


test_command.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def compile_command(python: str, files: list[str]) -> list[str]:
    """The argv that byte-compiles *files* in memory (a syntax check that writes nothing)."""
    code = (
        "import sys\n"
        "bad = 0\n"
        "for p in sys.argv[1:]:\n"
        "    try:\n"
        "        compile(open(p, 'rb').read(), p, 'exec')\n"
        "    except SyntaxError as e:\n"
        "        bad += 1\n"
        "        print(f'{p}:{e.lineno}:{e.offset}: SyntaxError: {e.msg}')\n"
        "sys.exit(1 if bad else 0)\n"
    )
    return [python, "-c", code, *files]


_IMPORT_PROBE = (
    "import importlib, json, os, sys\n"
    "root = os.getcwd()\n"
    "sys.path[:0] = [root, os.path.join(root, 'src')]\n"
    "out = []\n"
    "for rel, names in json.loads(sys.argv[1]):\n"
    "    want = os.path.realpath(os.path.join(root, rel))\n"
    "    for name in names:\n"
    "        try:\n"
    "            mod = importlib.import_module(name)\n"
    "        except BaseException:\n"
    "            continue\n"
    "        if os.path.realpath(getattr(mod, '__file__', None) or '') == want:\n"
    "            out.append(name)\n"
    "            break\n"
    "print('IMPORTABLE ' + json.dumps(out))\n"
)
_IMPORT_CHECK = (
    "import importlib, os, sys\n"
    "sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'src')]\n"
    "for name in sys.argv[1:]:\n"
    "    importlib.import_module(name)\n"
)


def python_version(python: str, cwd: str | Path) -> str:
    """``"3.10.4"`` for *python*, or ``""`` when it cannot be asked."""
    code, out = run([python, "-c", "import sys; print(sys.version.split()[0])"], cwd, 30.0)
    return out.strip().splitlines()[-1] if code == 0 and out.strip() else ""


def importable_modules(
    python: str, project_root: str | Path, rel_paths: list[str], timeout: float = 120.0
) -> list[str]:
    """The dotted names under which *python* imports each of *rel_paths* right now.

    Only modules inside a package (a directory with ``__init__.py``) and not test or setup
    files are tried, so nothing with top-level side effects is imported; a file that does not
    import (already broken, optional dependency missing) is left out.  Call it with the
    ORIGINAL files in place: what imported before a change must still import after it.
    """
    import json

    root = Path(project_root)
    wanted: list[tuple[str, list[str]]] = []
    for rel in dict.fromkeys(rel_paths):
        path = Path(rel)
        if path.suffix != ".py" or path.name in ("setup.py", "conftest.py", "__main__.py"):
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in DEFAULT_PATTERNS):
            continue
        if not (root / path.parent / "__init__.py").is_file():
            continue
        names = [n for n in module_names(rel) if "." in n or path.name == "__init__.py"]
        # ``src/pkg/mod.py`` is ``pkg.mod``; ``src.pkg.mod`` also imports (a namespace package)
        # but is not the module the project uses, so it is tried last.
        names.sort(key=lambda n: n.split(".")[0] in ("src", "lib"))
        if names:
            wanted.append((rel, names))
    if not wanted:
        return []
    code, out = run([python, "-c", _IMPORT_PROBE, json.dumps(wanted)], root, timeout)
    if code != 0:
        return []
    for line in reversed(out.splitlines()):
        if line.startswith("IMPORTABLE "):
            try:
                found = json.loads(line[len("IMPORTABLE ") :])
            except ValueError:
                return []
            return [str(n) for n in found] if isinstance(found, list) else []
    return []


def import_command(python: str, modules: list[str]) -> list[str]:
    """The argv that imports *modules* with the project's interpreter (fails on the first
    one that does not import, with its traceback)."""
    return [python, "-c", _IMPORT_CHECK, *modules]


def run(argv: list[str], cwd: str | Path, timeout: float) -> tuple[int | None, str]:
    """Run *argv* in its own process group; ``(exit code, output)``, or ``(None, note)`` when
    it timed out (the whole group is killed) or could not start."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        res = fixloop._run_in_group(argv, str(cwd), env, timeout)  # pyright: ignore[reportPrivateUsage]
    except subprocess.TimeoutExpired:
        return None, f"(timed out after {timeout:.0f}s)"
    except OSError as exc:
        return None, f"(could not run: {exc})"
    return res.returncode, (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()


@contextlib.contextmanager
def original_files(originals: dict[Path, str | None]) -> Iterator[None]:
    """Temporarily put back the ORIGINAL contents of the changed files (``None`` = the file did
    not exist), restoring the current contents on exit whatever happens."""
    current: dict[Path, str | None] = {}
    for path in originals:
        try:
            current[path] = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            current[path] = None
    try:
        for path, text in originals.items():
            _put(path, text)
        yield
    finally:
        for path, text in current.items():
            _put(path, text)


def _put(path: Path, text: str | None) -> None:
    with contextlib.suppress(OSError):
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(text, "utf-8")


def baseline_failures(
    project_root: str | Path,
    setup: TestSetup,
    test_files: list[str],
    originals: dict[Path, str | None],
    timeout: float,
) -> tuple[list[str], list[str], str]:
    """Run *test_files* on the ORIGINAL code: ``(failing test ids, unusable files, note)``.

    A test that fails before the change says nothing about the change, and a file that does
    not even collect (or the whole run timing out) cannot be used to check it.
    """
    argv = test_command(setup, test_files)
    if not argv:
        return [], list(test_files), "no test command"
    with original_files(originals):
        code, out = run(argv, project_root, timeout)
    if code is None:
        return [], list(test_files), f"the tests did not finish on the original code {out}"
    if code == 0:
        return [], [], ""
    summary = fixloop.summarize(out)
    failing = [i for i in summary.failed_ids if "::" in i]
    broken = [f for f in test_files if any(i == f for i in summary.failed_ids)]
    if not summary.found or (summary.broken and not failing and not broken):
        # No readable pytest result at all (a crash, a usage error): nothing here can be trusted.
        return [], list(test_files), "the tests do not run on the original code"
    return failing, broken, ""


def shell(argv: list[str]) -> str:
    """*argv* as one shell command string."""
    return shlex.join(argv)
