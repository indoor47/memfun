"""Built-in grep / glob / ranged read_file for RLM agents.

These are the Claude-Code-style core navigation tools. They live in the
REPL itself so they cannot silently disappear when the optional MCP
gateway breaks (a fastmcp upgrade once removed every gateway tool).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from memfun_agent import rlm
from memfun_agent.rlm import (
    PathOutsideRootError,
    _canonical_root,
    _make_glob,
    _make_grep,
    _make_read_file,
    build_context_metadata,
)


def _ns() -> dict[str, Any]:
    return {"state": {"_ops": []}}


@pytest.fixture
def task_root(tmp_path: Path) -> str:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "alpha.py").write_text("def handler():\n    return 'TODO'\n")
    (tmp_path / "pkg" / "beta.py").write_text("x = 1\n# TODO later\n")
    (tmp_path / "notes.md").write_text("todo in docs\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("TODO must not be found\n")
    os.utime(tmp_path / "pkg" / "beta.py", (2_000_000_000, 2_000_000_000))
    return str(_canonical_root(str(tmp_path)))


def test_grep_finds_matches_relative_to_the_task_root(task_root: str) -> None:
    out = _make_grep(_ns(), root=task_root)("TODO")
    lines = sorted(out.splitlines())
    assert lines == ["pkg/alpha.py:2:    return 'TODO'", "pkg/beta.py:2:# TODO later"]


def test_grep_glob_and_ignore_case(task_root: str) -> None:
    grep = _make_grep(_ns(), root=task_root)
    assert grep("todo", glob="*.md") == "notes.md:1:todo in docs"
    assert "notes.md" in grep("TODO", ignore_case=True)


def test_grep_reports_no_matches_and_truncation(task_root: str) -> None:
    grep = _make_grep(_ns(), root=task_root)
    assert grep("does-not-exist").startswith("No matches")
    out = grep("TODO", max_results=1)
    assert len(out.splitlines()) == 2 and "[truncated: 1 more lines" in out


def test_grep_refuses_paths_outside_the_root(task_root: str) -> None:
    with pytest.raises(PathOutsideRootError):
        _make_grep(_ns(), root=task_root)("TODO", path="..")


def test_grep_falls_back_without_ripgrep(task_root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    out = _make_grep(_ns(), root=task_root)("TODO")
    assert sorted(out.splitlines()) == [
        "pkg/alpha.py:2:    return 'TODO'",
        "pkg/beta.py:2:# TODO later",
    ]
    # Hidden dirs (.git) are skipped by the fallback too.
    assert ".git" not in out


def test_grep_pattern_is_an_argument_not_shell(task_root: str) -> None:
    # A pattern full of shell metacharacters must be searched for, not run.
    out = _make_grep(_ns(), root=task_root)("$(touch pwned); rm -rf /")
    assert out.startswith("No matches")
    assert not os.path.exists(os.path.join(task_root, "pwned"))


def test_glob_newest_first_and_skips_hidden(task_root: str) -> None:
    files = _make_glob(root=task_root)("**/*.py")
    assert files == ["pkg/beta.py", "pkg/alpha.py"]  # beta has the newer mtime
    assert all(".git" not in f for f in _make_glob(root=task_root)("**/*"))


def test_glob_refuses_escaping_patterns(task_root: str) -> None:
    g = _make_glob(root=task_root)
    with pytest.raises(PathOutsideRootError):
        g("../*")
    with pytest.raises(PathOutsideRootError):
        g("/etc/*")


def test_glob_drops_symlinks_that_point_outside(
    task_root: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.py").write_text("x")
    os.symlink(outside / "secret.py", os.path.join(task_root, "pkg", "link.py"))
    assert "pkg/link.py" not in _make_glob(root=task_root)("**/*.py")


def test_read_file_slice_is_line_numbered(task_root: str) -> None:
    read = _make_read_file(_ns(), root=task_root)
    out = read("pkg/alpha.py", offset=2, limit=1)
    assert out.splitlines()[1] == "     2\t    return 'TODO'"
    assert "lines 2-2 of 2" in out.splitlines()[0]
    # Default behaviour (whole file) is unchanged.
    assert read("pkg/beta.py") == "x = 1\n# TODO later\n"


def test_the_llm_is_told_about_the_new_tools() -> None:
    meta = build_context_metadata("ctx", 100)
    for name in ("grep(pattern", "glob(pattern", "read_file(path: str, offset"):
        assert name in meta


def test_helpers_are_registered_in_the_repl_namespace() -> None:
    src = Path(rlm.__file__).read_text()
    assert 'repl.namespace["grep"] = _make_grep(' in src
    assert 'repl.namespace["glob"] = _make_glob(' in src
