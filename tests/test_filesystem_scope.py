"""Tests for ``MEMFUN_PROJECT_ROOT`` scoping on the FastMCP filesystem
tools (issue #13).

Each filesystem tool must:
* honour ``MEMFUN_PROJECT_ROOT`` when set — paths that resolve outside
  the root must raise ``ValueError`` (or surface a ``ValueError`` from
  the tool wrapper);
* follow symlinks before the containment check, so a symlink planted
  inside the root pointing outside cannot bypass the guard;
* preserve pre-#13 behaviour when ``MEMFUN_PROJECT_ROOT`` is unset
  (backward compatible).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from memfun_tools.code import filesystem as fs

# The FastMCP ``@fs_server.tool()`` decorator wraps the coroutine in
# a ``FunctionTool`` object that is no longer directly callable, so
# the tests reach for the underlying ``.fn`` callable instead.  This
# exercises the exact same code path the MCP server invokes.
read_file = fs.read_file.fn
write_file = fs.write_file.fn
edit_file = fs.edit_file.fn
list_directory = fs.list_directory.fn
glob_files = fs.glob_files.fn
_check_in_root = fs._check_in_root

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Set ``MEMFUN_PROJECT_ROOT`` to *tmp_path* for the duration of the test."""
    monkeypatch.setenv("MEMFUN_PROJECT_ROOT", str(tmp_path))
    yield tmp_path


@pytest.fixture
def unset_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure ``MEMFUN_PROJECT_ROOT`` is unset for the test."""
    monkeypatch.delenv("MEMFUN_PROJECT_ROOT", raising=False)
    yield


# ---------------------------------------------------------------------------
# _check_in_root primitive
# ---------------------------------------------------------------------------


class TestCheckInRoot:
    def test_inside_root_allowed(self, project_root: Path) -> None:
        target = project_root / "foo.txt"
        target.write_text("hi")
        resolved = _check_in_root(target)
        assert resolved == target.resolve()

    def test_nested_inside_root_allowed(self, project_root: Path) -> None:
        nested = project_root / "a" / "b" / "c.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("hi")
        resolved = _check_in_root(nested)
        assert resolved == nested.resolve()

    def test_relative_dotdot_escape_rejected(self, project_root: Path) -> None:
        # MEMFUN_PROJECT_ROOT=tmp_path, attempt to escape via ../
        outside = project_root / ".." / "escape.txt"
        with pytest.raises(ValueError, match="path outside project root"):
            _check_in_root(outside)

    def test_absolute_path_outside_rejected(
        self,
        project_root: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        # tmp_path_factory mints a sibling temp dir that is *not* the
        # project root.
        sibling_dir = tmp_path_factory.mktemp("sibling")
        outside = sibling_dir / "secret.txt"
        outside.write_text("nope")
        with pytest.raises(ValueError, match="path outside project root"):
            _check_in_root(outside)

    def test_etc_passwd_blocked_even_inside_check(self, project_root: Path) -> None:
        # Sensitive system paths are always rejected.  The blocked
        # prefix check examines both the unresolved and resolved path
        # strings, so the macOS ``/etc`` -> ``/private/etc`` symlink
        # does not let ``/etc/passwd`` slip past.
        with pytest.raises(PermissionError):
            _check_in_root(Path("/etc/passwd"))

    def test_blocked_name_rejected(self, project_root: Path) -> None:
        cred = project_root / "credentials.json"
        cred.write_text("{}")
        with pytest.raises(PermissionError):
            _check_in_root(cred)

    def test_symlink_inside_to_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside_dir = tmp_path_factory.mktemp("outside_dir")
        outside_target = outside_dir / "secret.txt"
        outside_target.write_text("super secret")

        # Plant a symlink inside the project root that points outside.
        link = project_root / "shortcut.txt"
        link.symlink_to(outside_target)

        with pytest.raises(ValueError, match="path outside project root"):
            _check_in_root(link)

    def test_symlink_inside_to_inside_allowed(self, project_root: Path) -> None:
        real = project_root / "real.txt"
        real.write_text("ok")
        link = project_root / "link.txt"
        link.symlink_to(real)

        resolved = _check_in_root(link)
        assert resolved == real.resolve()

    def test_unset_root_allows_anything(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        # Even an absolute path far from any project root works
        # (preserving pre-#13 behaviour).
        target = tmp_path / "anywhere.txt"
        target.write_text("ok")
        # No exception expected.
        _check_in_root(target)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    async def test_inside_root_allowed(self, project_root: Path) -> None:
        f = project_root / "hello.txt"
        f.write_text("line1\nline2\n")
        out = await read_file(str(f))
        assert "line1" in out
        assert "line2" in out

    async def test_dotdot_escape_rejected(self, project_root: Path) -> None:
        with pytest.raises(ValueError, match="path outside project root"):
            await read_file(str(project_root / ".." / "escape.txt"))

    async def test_absolute_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        sibling = tmp_path_factory.mktemp("sib_read")
        target = sibling / "x.txt"
        target.write_text("nope")
        with pytest.raises(ValueError, match="path outside project root"):
            await read_file(str(target))

    async def test_etc_path_rejected(self, project_root: Path) -> None:
        # Even when MEMFUN_PROJECT_ROOT is set, blocked prefixes
        # are still rejected (with PermissionError).
        with pytest.raises((PermissionError, ValueError)):
            await read_file("/etc/passwd")

    async def test_symlink_to_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("read_outside") / "secret.txt"
        outside.write_text("super secret")
        link = project_root / "shortcut.txt"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="path outside project root"):
            await read_file(str(link))

    async def test_unset_root_backward_compat(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        f = tmp_path / "any.txt"
        f.write_text("hi")
        # No project root set; this should still work.
        out = await read_file(str(f))
        assert "hi" in out


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    async def test_inside_root_allowed(self, project_root: Path) -> None:
        target = project_root / "sub" / "out.py"
        msg = await write_file(str(target), "print(1)\n")
        assert "Written" in msg
        assert target.read_text() == "print(1)\n"

    async def test_dotdot_escape_rejected(self, project_root: Path) -> None:
        with pytest.raises(ValueError, match="path outside project root"):
            await write_file(
                str(project_root / ".." / "escape.txt"), "x"
            )

    async def test_absolute_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        sibling = tmp_path_factory.mktemp("sib_write")
        with pytest.raises(ValueError, match="path outside project root"):
            await write_file(str(sibling / "x.txt"), "x")

    async def test_symlink_to_outside_rejected_before_write(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside_dir = tmp_path_factory.mktemp("write_outside")
        outside_target = outside_dir / "victim.txt"
        outside_target.write_text("original")
        link = project_root / "evil_link.txt"
        link.symlink_to(outside_target)

        with pytest.raises(ValueError, match="path outside project root"):
            await write_file(str(link), "pwned")

        # The outside file must NOT have been overwritten.
        assert outside_target.read_text() == "original"

    async def test_unset_root_backward_compat(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        target = tmp_path / "any.py"
        msg = await write_file(str(target), "x = 1\n")
        assert "Written" in msg


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    async def test_inside_root_replaces(self, project_root: Path) -> None:
        f = project_root / "code.py"
        f.write_text("def add(a, b): return a + b\n")
        msg = await edit_file(str(f), "a + b", "a + b + 0")
        assert "Edited" in msg
        assert f.read_text() == "def add(a, b): return a + b + 0\n"

    async def test_missing_old_raises(self, project_root: Path) -> None:
        f = project_root / "code.py"
        f.write_text("hello\n")
        with pytest.raises(ValueError, match="not found"):
            await edit_file(str(f), "world", "x")

    async def test_ambiguous_old_raises(self, project_root: Path) -> None:
        f = project_root / "code.py"
        f.write_text("foo foo\n")
        with pytest.raises(ValueError, match="ambiguous"):
            await edit_file(str(f), "foo", "bar")

    async def test_empty_old_raises(self, project_root: Path) -> None:
        f = project_root / "code.py"
        f.write_text("hi\n")
        with pytest.raises(ValueError, match="non-empty"):
            await edit_file(str(f), "", "x")

    async def test_dotdot_escape_rejected(self, project_root: Path) -> None:
        with pytest.raises(ValueError, match="path outside project root"):
            await edit_file(
                str(project_root / ".." / "escape.txt"), "a", "b"
            )

    async def test_absolute_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        sibling = tmp_path_factory.mktemp("sib_edit")
        target = sibling / "x.txt"
        target.write_text("hello world")
        with pytest.raises(ValueError, match="path outside project root"):
            await edit_file(str(target), "hello", "bye")
        # Outside file untouched.
        assert target.read_text() == "hello world"

    async def test_symlink_to_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside_dir = tmp_path_factory.mktemp("edit_outside")
        outside_target = outside_dir / "victim.txt"
        outside_target.write_text("hello world")
        link = project_root / "linked.txt"
        link.symlink_to(outside_target)

        with pytest.raises(ValueError, match="path outside project root"):
            await edit_file(str(link), "hello", "bye")

        # Outside file untouched.
        assert outside_target.read_text() == "hello world"

    async def test_unset_root_backward_compat(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        f = tmp_path / "any.txt"
        f.write_text("alpha")
        msg = await edit_file(str(f), "alpha", "beta")
        assert "Edited" in msg
        assert f.read_text() == "beta"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    async def test_inside_root_allowed(self, project_root: Path) -> None:
        (project_root / "a.txt").write_text("a")
        (project_root / "sub").mkdir()
        out = await list_directory(str(project_root))
        assert "a.txt" in out
        assert "sub" in out

    async def test_dotdot_escape_rejected(self, project_root: Path) -> None:
        with pytest.raises(ValueError, match="path outside project root"):
            await list_directory(str(project_root / ".."))

    async def test_absolute_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        sibling = tmp_path_factory.mktemp("sib_ls")
        with pytest.raises(ValueError, match="path outside project root"):
            await list_directory(str(sibling))

    async def test_symlink_dir_to_outside_rejected(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside_dir = tmp_path_factory.mktemp("ls_outside")
        (outside_dir / "secret.txt").write_text("nope")
        link = project_root / "linked_dir"
        link.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(ValueError, match="path outside project root"):
            await list_directory(str(link))

    async def test_unset_root_backward_compat(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        (tmp_path / "x.txt").write_text("x")
        out = await list_directory(str(tmp_path))
        assert "x.txt" in out


# ---------------------------------------------------------------------------
# glob_files
# ---------------------------------------------------------------------------


class TestGlobFiles:
    async def test_inside_root_allowed(self, project_root: Path) -> None:
        (project_root / "a.py").write_text("a")
        (project_root / "sub").mkdir()
        (project_root / "sub" / "b.py").write_text("b")
        out = await glob_files("**/*.py", str(project_root))
        assert "a.py" in out
        assert "b.py" in out

    async def test_dotdot_base_rejected(self, project_root: Path) -> None:
        with pytest.raises(ValueError, match="path outside project root"):
            await glob_files("*", str(project_root / ".."))

    async def test_glob_strips_symlinks_pointing_outside(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        # Set up an outside file the attacker would love to see.
        outside = tmp_path_factory.mktemp("glob_outside")
        outside_py = outside / "secret.py"
        outside_py.write_text("# secret")

        # Plant a symlink inside the root, pointing to that outside file.
        (project_root / "real.py").write_text("# real")
        evil = project_root / "evil.py"
        evil.symlink_to(outside_py)

        out = await glob_files("**/*.py", str(project_root))
        # The legitimate file is returned…
        assert "real.py" in out
        # …but the symlink to the outside file must be filtered out.
        assert "evil.py" not in out

    async def test_glob_does_not_leak_via_dotdot_pattern(
        self, project_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        # If a glob pattern itself includes ``..``, none of the
        # matches outside the root may be returned.
        outside = tmp_path_factory.mktemp("dotdot_target")
        outside_file = outside / "leak.py"
        outside_file.write_text("leaked")

        # Pattern attempts to climb out using ../
        out = await glob_files(
            f"../{outside.name}/*.py", str(project_root)
        )
        # Either "(no matches)" or at least no entry resolving outside.
        assert "leak.py" not in out

    async def test_unset_root_backward_compat(
        self, unset_root: None, tmp_path: Path
    ) -> None:
        (tmp_path / "z.py").write_text("z")
        out = await glob_files("*.py", str(tmp_path))
        assert "z.py" in out


# ---------------------------------------------------------------------------
# Sanity check: no-leak when env var is unset
# ---------------------------------------------------------------------------


async def test_unset_env_var_completely_pre13_compat(
    unset_root: None, tmp_path: Path
) -> None:
    """End-to-end sanity: with no ``MEMFUN_PROJECT_ROOT``, we behave
    exactly like the pre-#13 tools — all of read/write/list/glob/edit
    work on absolute paths anywhere on disk (modulo blocked
    prefixes/names)."""
    assert os.environ.get("MEMFUN_PROJECT_ROOT") is None

    f = tmp_path / "anywhere.py"
    msg = await write_file(str(f), "x = 1\n")
    assert "Written" in msg

    out = await read_file(str(f))
    assert "x = 1" in out

    msg = await edit_file(str(f), "x = 1", "x = 2")
    assert "Edited" in msg

    out = await list_directory(str(tmp_path))
    assert "anywhere.py" in out

    out = await glob_files("*.py", str(tmp_path))
    assert "anywhere.py" in out
