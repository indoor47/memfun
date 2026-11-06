"""Rooted MCP REPL tools must not enumerate names outside the task root.

Production swarms do NOT set ``MEMFUN_PROJECT_ROOT``; the per-task root
comes only from the RLM's rooted wrapper (``_root_extra_tools``).  These
tests therefore run with the env var UNSET and go through the REAL
production gateway + ``create_tool_bridge()`` + rooted wrapper.

Two layers are exercised:

* the bridge's REPL wrappers (``MCPToolBridge.get_repl_tools``) reject
  an absolute or ``..``-bearing ``pattern`` (``glob_files``), ``glob``
  (``grep``) or ``file`` (``git_diff``) -- the rooted wrapper only
  rewrites ``path``, so this check lives in the bridge;
* ``fs_glob_files`` / ``fs_list_directory`` / ``repo_repo_map`` drop
  every entry resolving outside their base, which also covers in-root
  symlinks pointing out.

The root is a guard against accidents, not a sandbox (the REPL has full
builtins and ``run_cmd``); these tests pin the guard, nothing more.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent.rlm import _root_extra_tools  # pyright: ignore[reportPrivateUsage]
from memfun_agent.tool_bridge import MCPToolBridge, create_tool_bridge

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Names that exist only OUTSIDE the root; none may ever appear in output.
OUTSIDE_NAMES = ("outside_note.txt", "sibling_secret_dir", "id_rsa_FAKE", "far_away_marker")


@dataclass
class Layout:
    base: Path
    root: Path
    bridge: MCPToolBridge
    tools: dict[str, Callable[..., Any]]


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Layout:
    monkeypatch.delenv("MEMFUN_PROJECT_ROOT", raising=False)
    far = tmp_path.resolve()
    (far / "far_away_marker").write_text("far\n")
    base = far / "a" / "b" / "c"
    base.mkdir(parents=True)
    (base / "sibling_secret_dir").mkdir()
    (base / "sibling_secret_dir" / "id_rsa_FAKE").write_text("FAKEKEY-OUTSIDE\n")
    (base / "outside_note.txt").write_text("OUTSIDE-CONTENT\n")

    root = base / "wt"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "inside.py").write_text("x = 'INSIDE'\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("y = 1\n")
    (root / "link_out").symlink_to(base / "sibling_secret_dir", target_is_directory=True)
    (root / "note_link.txt").symlink_to(base / "outside_note.txt")

    bridge = asyncio.run(create_tool_bridge())
    assert bridge.tool_count > 0
    tools = _root_extra_tools(bridge.get_repl_tools(), str(root))
    return Layout(base=base, root=root, bridge=bridge, tools=tools)


def _assert_no_outside(out: str) -> None:
    for name in OUTSIDE_NAMES:
        assert name not in out, f"leaked {name!r}: {out[:300]!r}"


# ── glob_files through the rooted wrapper ──────────────────────


@pytest.mark.parametrize(
    "pattern",
    [
        "../*",
        "../*/*",
        "../../../../*",
        "pkg/../../*",
        "..\\*",
        "/etc/*",
    ],
)
def test_glob_rejects_escaping_patterns(layout: Layout, pattern: str) -> None:
    out = layout.tools["glob_files"](pattern=pattern)
    assert out.startswith("[tool error: fs_glob_files]"), out
    assert "must not contain '..'" in out
    _assert_no_outside(out)


def test_glob_rejects_absolute_pattern_into_outside_dir(layout: Layout) -> None:
    out = layout.tools["glob_files"](pattern=f"{layout.base}/*")
    assert out.startswith("[tool error: fs_glob_files]"), out
    _assert_no_outside(out)
    # The rejected value is not echoed back.
    assert str(layout.base) not in out


def test_glob_through_in_root_symlink_returns_nothing_outside(layout: Layout) -> None:
    out = layout.tools["glob_files"](pattern="link_out/*")
    assert out == "(no matches)"


def test_glob_recursive_in_root_happy_path(layout: Layout) -> None:
    out = layout.tools["glob_files"](pattern="**/*.py")
    lines = set(out.splitlines())
    assert {"inside.py", "pkg/mod.py"} <= lines
    _assert_no_outside(out)

    star = layout.tools["glob_files"](pattern="*")
    assert "inside.py" in star.splitlines()
    assert "link_out" not in star.splitlines()
    assert "note_link.txt" not in star.splitlines()
    _assert_no_outside(star)


# ── fs_glob_files on its own (filesystem layer, env var unset) ─


@pytest.mark.parametrize("pattern", ["../*", "../*/*", "../../../../*", "link_out/*", "*"])
def test_fs_glob_files_bounds_matches_to_base_without_env(layout: Layout, pattern: str) -> None:
    res = asyncio.run(
        layout.bridge.call_tool("fs_glob_files", pattern=pattern, path=str(layout.root))
    )
    assert res.success, res.error
    _assert_no_outside(res.content)


# ── other enumerating tools ────────────────────────────────────


def test_list_directory_does_not_descend_through_symlink(layout: Layout) -> None:
    out = layout.tools["list_directory"](path=".", recursive=True)
    assert "f inside.py" in out.splitlines()
    assert "f pkg/mod.py" in out.splitlines()
    _assert_no_outside(out)
    assert "link_out" not in out

    flat = layout.tools["list_directory"](path=".")
    assert any(line.startswith("f inside.py") for line in flat.splitlines())
    _assert_no_outside(flat)


def test_list_directory_of_symlink_itself_rejected(layout: Layout) -> None:
    out = layout.tools["list_directory"](path="link_out")
    assert out.startswith("[tool error: list_directory]")
    # Nothing inside the symlink target is listed.  (The rooted wrapper's
    # PathOutsideRootError message does name the resolved target path.)
    assert "id_rsa_FAKE" not in out


def test_repo_map_skips_symlinks_pointing_outside(layout: Layout) -> None:
    out = layout.tools["repo_map"]()
    assert "inside.py" in out
    assert "mod.py" in out
    assert "link_out" not in out
    assert "note_link.txt" not in out
    _assert_no_outside(out)


@pytest.mark.parametrize("glob", ["../**", "/etc/*", "pkg/../../*"])
def test_grep_glob_escape_rejected(layout: Layout, glob: str) -> None:
    out = layout.tools["grep"](pattern="OUTSIDE", glob=glob)
    assert out.startswith("[tool error: search_grep]"), out
    assert "'glob' must be relative" in out


def test_git_diff_file_escape_rejected(layout: Layout) -> None:
    out = layout.tools["git_diff"](file="../outside_note.txt")
    assert out.startswith("[tool error: git_git_diff]"), out
    assert "'file' must be relative" in out
    _assert_no_outside(out)


def test_in_root_pattern_args_pass_through(layout: Layout) -> None:
    # A relative, in-root grep glob / git_diff file is not rejected by the
    # wrapper (the tool itself may still fail, e.g. ripgrep missing).
    grep_out = layout.tools["grep"](pattern="INSIDE", glob="*.py")
    assert "must be relative" not in grep_out
    diff_out = layout.tools["git_diff"](file="inside.py")
    assert "must be relative" not in diff_out


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        ("glob_files", {"pattern": "../*"}),
        ("glob_files", {"pattern": "/etc/*"}),
        ("grep", {"pattern": "x", "glob": "!../**"}),
        ("git_diff", {"file": "../outside_note.txt"}),
    ],
)
def test_unrooted_bridge_wrappers_reject_escaping_args(
    layout: Layout, tool: str, kwargs: dict[str, Any]
) -> None:
    """The check is in the bridge, so it holds without the rooted wrapper."""
    out = layout.bridge.get_repl_tools()[tool](path=str(layout.root), **kwargs)
    assert out.startswith("[tool error: "), out
    assert "must be relative to the search path" in out
