"""Built-in grep / find_files / ranged read_file for RLM agents.

These are the Claude-Code-style core navigation tools. They live in the
REPL itself so they cannot silently disappear when the optional MCP
gateway breaks (a fastmcp upgrade once removed every gateway tool).
Every grep test runs against both engines: real ripgrep (skipped when not
installed) and the pure-Python fallback.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest
from memfun_agent import rlm
from memfun_agent.rlm import (
    PathOutsideRootError,
    _canonical_root,
    _make_find_files,
    _make_grep,
    _make_read_file,
    build_context_metadata,
)


def _ns() -> dict[str, Any]:
    return {"state": {"_ops": []}}


@pytest.fixture(params=["rg", "python"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "rg":
        if shutil.which("rg") is None:
            pytest.skip("ripgrep not installed")
    else:
        monkeypatch.setattr("shutil.which", lambda _name: None)
    return str(request.param)


@pytest.fixture
def task_root(tmp_path: Path) -> str:
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "alpha.py").write_text("def handler():\n    return 'TODO'\n")
    (root / "pkg" / "beta.py").write_text("x = 1\n# TODO later\n")
    (root / "notes.md").write_text("todo in docs\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("TODO must not be found\n")
    os.utime(root / "pkg" / "beta.py", (2_000_000_000, 2_000_000_000))
    return str(_canonical_root(str(root)))


def test_grep_finds_matches_relative_to_the_task_root(task_root: str, engine: str) -> None:
    out = _make_grep(_ns(), root=task_root)("TODO")
    assert sorted(out.splitlines()) == [
        "pkg/alpha.py:2:    return 'TODO'",
        "pkg/beta.py:2:# TODO later",
    ]


def test_grep_glob_and_ignore_case(task_root: str, engine: str) -> None:
    grep = _make_grep(_ns(), root=task_root)
    assert grep("todo", glob="*.md") == "notes.md:1:todo in docs"
    assert "notes.md" in grep("TODO", ignore_case=True)


def test_grep_single_file_keeps_the_filename(task_root: str, engine: str) -> None:
    assert _make_grep(_ns(), root=task_root)("TODO", path="pkg/beta.py") == (
        "pkg/beta.py:2:# TODO later"
    )


def test_grep_reports_no_matches_and_truncation(task_root: str, engine: str) -> None:
    grep = _make_grep(_ns(), root=task_root)
    assert grep("does-not-exist").startswith("No matches")
    out = grep("TODO", max_results=1)
    assert len(out.splitlines()) == 2 and "[truncated:" in out


def test_grep_tolerates_garbage_numbers(task_root: str, engine: str) -> None:
    grep = _make_grep(_ns(), root=task_root)
    assert "TODO" in grep("TODO", max_results=None, context="x")  # type: ignore[arg-type]


def test_grep_refuses_paths_outside_the_root(task_root: str, engine: str) -> None:
    with pytest.raises(PathOutsideRootError):
        _make_grep(_ns(), root=task_root)("TODO", path="..")


def test_grep_never_follows_symlinks_out_of_the_root(
    task_root: str, engine: str, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET\n")
    os.symlink(outside / "secret.txt", Path(task_root) / "link.txt")
    assert _make_grep(_ns(), root=task_root)("TOPSECRET").startswith("No matches")


def test_grep_survives_non_utf8_content(task_root: str, engine: str) -> None:
    (Path(task_root) / "latin1.txt").write_bytes(b"caf\xe9 TOKEN\nsecond TOKEN\n")
    out = _make_grep(_ns(), root=task_root)("TOKEN")
    assert out.count("latin1.txt:") == 2  # both lines, nothing lost after the bad byte


def test_unreadable_file_does_not_hide_other_matches(task_root: str, engine: str) -> None:
    locked = Path(task_root) / "locked.txt"
    locked.write_text("TODO hidden\n")
    locked.chmod(0)
    try:
        out = _make_grep(_ns(), root=task_root)("TODO")
        assert "pkg/alpha.py:2:" in out
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_grep_pattern_is_an_argument_not_shell(task_root: str, engine: str) -> None:
    out = _make_grep(_ns(), root=task_root)("$(touch pwned); rm -rf /")
    assert out.startswith("No matches")
    assert not os.path.exists(os.path.join(task_root, "pwned"))


def test_bad_regex_is_an_error_not_a_match(task_root: str, engine: str) -> None:
    out = _make_grep(_ns(), root=task_root)("(unclosed")
    assert out.startswith("[grep error")


def test_grep_line_numbers_match_read_file(task_root: str, engine: str) -> None:
    # Form feeds & co. must not shift read_file's numbering away from grep's.
    (Path(task_root) / "ff.py").write_text("a\n\x0cb\nTARGET\n")
    hit = _make_grep(_ns(), root=task_root)("TARGET", path="ff.py")
    line_no = int(hit.split(":")[1])
    sliced = _make_read_file(_ns(), root=task_root)("ff.py", offset=line_no, limit=1)
    assert sliced.splitlines()[1].endswith("\tTARGET")


def test_find_files_newest_first_and_skips_hidden(task_root: str) -> None:
    find = _make_find_files(_ns(), root=task_root)
    assert find("**/*.py") == ["pkg/beta.py", "pkg/alpha.py"]  # beta is newer
    assert all(".git" not in f for f in find("**/*"))
    assert find("") == []


def test_find_files_refuses_escaping_patterns(task_root: str) -> None:
    find = _make_find_files(_ns(), root=task_root)
    with pytest.raises(PathOutsideRootError):
        find("../*")
    with pytest.raises(PathOutsideRootError):
        find("/etc/*")


def test_find_files_drops_symlinks_that_point_outside(task_root: str, tmp_path: Path) -> None:
    outside = tmp_path / "outside2"
    outside.mkdir()
    (outside / "secret.py").write_text("x")
    os.symlink(outside / "secret.py", Path(task_root) / "pkg" / "link.py")
    assert "pkg/link.py" not in _make_find_files(_ns(), root=task_root)("**/*.py")


def test_read_file_slice_is_line_numbered(task_root: str) -> None:
    read = _make_read_file(_ns(), root=task_root)
    out = read("pkg/alpha.py", offset=2, limit=1)
    assert out.splitlines()[1] == "     2\t    return 'TODO'"
    assert "lines 2-2 of 2" in out.splitlines()[0]
    assert "past the end" in read("pkg/alpha.py", offset=50)
    # Default behaviour (whole file) is unchanged.
    assert read("pkg/beta.py") == "x = 1\n# TODO later\n"


def test_the_llm_is_told_about_the_new_tools() -> None:
    meta = build_context_metadata("ctx", 100)
    for name in ("grep(pattern", "find_files(pattern", "read_file(path: str, offset"):
        assert name in meta


def test_helpers_are_registered_without_shadowing_stdlib_glob() -> None:
    src = Path(rlm.__file__).read_text()
    assert 'repl.namespace["grep"] = _make_grep(' in src
    assert 'repl.namespace["find_files"] = _make_find_files(' in src
    assert 'repl.namespace["glob"]' not in src
