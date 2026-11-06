"""Real MCP tool discovery through the production gateway and tool bridge.

These tests build the ACTUAL ``memfun_tools.gateway.create_gateway()``
gateway (no mocks) and drive ``memfun_agent.tool_bridge`` against it, so
an incompatible fastmcp upgrade fails here instead of silently leaving
every specialist without MCP tools.

Regression for: ``Failed to discover MCP tools: 'FastMCP' object has no
attribute 'get_tools'`` (fastmcp 3.x replaced ``get_tools()`` with
``list_tools()`` and ``call_tool()`` now returns a ``ToolResult``).

Hermetic: only local filesystem tools are *called*; web tools are only
listed by name.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.exceptions import FastMCPDeprecationWarning
from memfun_agent.tool_bridge import MCPToolBridge, create_tool_bridge
from memfun_tools.gateway import create_gateway

if TYPE_CHECKING:
    from pathlib import Path

# Tool names every mounted server must expose (namespace + "_" + tool).
EXPECTED_TOOLS: dict[str, set[str]] = {
    "fs": {
        "fs_read_file",
        "fs_write_file",
        "fs_edit_file",
        "fs_list_directory",
        "fs_glob_files",
    },
    "search": {"search_grep", "search_ast_grep"},
    "git": {
        "git_git_status",
        "git_git_diff",
        "git_git_log",
        "git_git_show",
        "git_git_blame",
    },
    "repo": {"repo_repo_map", "repo_file_summary"},
    "web_fetch": {"web_fetch_web_fetch"},
    "web_search": {"web_search_web_search"},
    "skills": {
        "skills_list_skills",
        "skills_get_skill_info",
        "skills_search_skills",
        "skills_invoke_skill",
    },
    "agents": {
        "agents_list_agents",
        "agents_get_agent_info",
        "agents_invoke_agent",
    },
}
ALL_EXPECTED: set[str] = set[str]().union(*EXPECTED_TOOLS.values())

EXPECTED_REPL_TOOLS = {
    "read_file",
    "write_file",
    "list_directory",
    "glob_files",
    "grep",
    "git_status",
    "git_diff",
    "git_log",
    "repo_map",
}


def _target(fn: Any) -> str:
    """MCP tool name a REPL wrapper calls (first docstring line)."""
    first = (fn.__doc__ or "").splitlines()[0]
    assert first.startswith("MCP tool: "), first
    return first.removeprefix("MCP tool: ")


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp project dir with one file; fs tools are confined to it."""
    root = tmp_path.resolve()
    (root / "hello.txt").write_text("hello from the bridge\nsecond line\n")
    monkeypatch.setenv("MEMFUN_PROJECT_ROOT", str(root))
    return root


# ── Gateway ────────────────────────────────────────────────────


async def test_gateway_lists_tools_for_every_mounted_server() -> None:
    tools = await create_gateway().list_tools()
    names = {t.name for t in tools}

    assert names, "production gateway exposes no tools"
    for namespace, expected in EXPECTED_TOOLS.items():
        missing = expected - names
        assert not missing, f"{namespace}: missing {sorted(missing)}"


def test_gateway_mounts_without_deprecated_prefix() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FastMCPDeprecationWarning)
        create_gateway()


# ── Bridge discovery ───────────────────────────────────────────


async def test_bridge_discovers_real_gateway_tools() -> None:
    # gateway=None -> the bridge builds the production gateway itself.
    bridge = await create_tool_bridge()

    assert bridge.tool_count > 0
    assert bridge.tool_count == len(bridge.available_tools)
    assert set(bridge.available_tools) >= ALL_EXPECTED


async def test_every_repl_wrapper_targets_a_discovered_tool() -> None:
    bridge = await create_tool_bridge()
    repl_tools = bridge.get_repl_tools()

    assert set(repl_tools) == EXPECTED_REPL_TOOLS
    for name, fn in repl_tools.items():
        assert callable(fn), name
        assert _target(fn) in bridge.available_tools, (
            f"REPL tool {name!r} calls {_target(fn)!r}, which the gateway does not expose"
        )


async def test_discovery_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenGateway:
        async def list_tools(self) -> list[Any]:
            raise AttributeError("no tools for you")

    with caplog.at_level(logging.WARNING, logger="memfun.agent.tool_bridge"):
        bridge = await create_tool_bridge(gateway=BrokenGateway())

    assert bridge.tool_count == 0
    msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no tools for you" in r.getMessage() for r in msgs)
    assert any(r.exc_info for r in msgs)


async def test_zero_tools_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    class EmptyGateway:
        name = "empty"

        async def list_tools(self) -> list[Any]:
            return []

    with caplog.at_level(logging.WARNING, logger="memfun.agent.tool_bridge"):
        bridge = await create_tool_bridge(gateway=EmptyGateway())

    assert bridge.tool_count == 0
    assert any("ZERO tools" in r.getMessage() for r in caplog.records)


# ── End-to-end REPL tool calls through the real gateway ────────


def test_repl_tools_read_and_list_real_files(project_root: Path) -> None:
    """No running loop: wrappers take the ``asyncio.run`` path."""
    tools = asyncio.run(create_tool_bridge()).get_repl_tools()
    target = str(project_root / "hello.txt")

    content = tools["read_file"](path=target)
    # Real tool text, one numbered line per file line -- not a pydantic
    # repr of the fastmcp ToolResult ("content=[TextContent(...)]").
    assert "TextContent(" not in content
    lines = content.splitlines()
    assert lines[0].endswith("hello from the bridge")
    assert lines[1].endswith("second line")

    listing = tools["list_directory"](path=str(project_root))
    assert "hello.txt" in listing

    matches = tools["glob_files"](pattern="*.txt", path=str(project_root))
    assert "hello.txt" in matches
    assert "[tool error" not in matches


async def test_repl_tool_call_from_worker_thread_without_loop(project_root: Path) -> None:
    """From a worker thread (no running loop there): the ``asyncio.run`` branch."""
    bridge = await create_tool_bridge()
    read_file = bridge.get_repl_tools()["read_file"]

    content = await asyncio.to_thread(read_file, path=str(project_root / "hello.txt"))
    assert content.splitlines()[0].endswith("hello from the bridge")


async def test_repl_tool_call_on_running_loop_thread(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Called directly on the loop thread (how the RLM REPL runs it): the
    wrapper must take its thread-pool branch and return without deadlock."""
    import concurrent.futures

    pools: list[int] = []
    real_pool = concurrent.futures.ThreadPoolExecutor

    class CountingPool(real_pool):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pools.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", CountingPool)

    bridge = await create_tool_bridge()
    read_file = bridge.get_repl_tools()["read_file"]
    asyncio.get_running_loop()  # we really are on a running loop

    content = read_file(path=str(project_root / "hello.txt"))
    assert content.splitlines()[0].endswith("hello from the bridge")
    assert pools == [1]


async def test_call_tool_reports_tool_errors(project_root: Path) -> None:
    bridge = await create_tool_bridge()

    ok = await bridge.call_tool("fs_read_file", path=str(project_root / "hello.txt"))
    assert ok.success is True
    assert ok.content.splitlines()[0].endswith("hello from the bridge")

    bad = await bridge.call_tool("fs_read_file", path=str(project_root / "missing.txt"))
    assert bad.success is False
    assert bad.error


async def test_bridge_without_list_tools_falls_back_to_get_tools() -> None:
    """A fastmcp 2.x-style gateway (``get_tools() -> dict``) still works."""

    class LegacyGateway:
        async def get_tools(self) -> dict[str, Any]:
            return {"fs_read_file": object(), "search_grep": object()}

    bridge = MCPToolBridge(gateway=LegacyGateway())
    await bridge.initialize()
    assert bridge.available_tools == ("fs_read_file", "search_grep")
