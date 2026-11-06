"""MCP Tool Bridge: exposes memfun-tools as Python functions for the RLM REPL.

Instead of importing memfun-tools directly, the bridge uses the MCP protocol
to invoke tools through the gateway. This keeps the agent decoupled from tool
implementations and enables tool discovery at runtime.

The bridge provides:
- ``MCPToolBridge`` -- wraps the MCP gateway for use inside the RLM REPL
- Pre-built Python function wrappers for common tools (read_file, grep, etc.)
- ``llm_query`` passthrough (provided by RLMModule, but re-exported for clarity)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("agent.tool_bridge")


# ── Tool Result ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result from invoking an MCP tool."""

    tool_name: str
    success: bool
    content: str
    error: str | None = None


# ── MCP Tool Bridge ────────────────────────────────────────────


class MCPToolBridge:
    """Bridge between the RLM REPL and MCP tool servers.

    Wraps the memfun-tools MCP gateway, providing synchronous Python
    function wrappers suitable for use inside the RLM's REPL namespace.
    Tools are invoked via the MCP protocol (``call_tool``), not via
    direct Python imports.

    Usage::

        bridge = MCPToolBridge()
        await bridge.initialize()

        # Get tool functions for REPL injection
        tools = bridge.get_repl_tools()
        # tools = {"read_file": <fn>, "grep": <fn>, ...}

        # Or invoke directly
        result = await bridge.call_tool("fs_read_file", path="README.md")
    """

    def __init__(
        self,
        gateway: Any | None = None,
    ) -> None:
        self._gateway = gateway
        self._initialized = False
        self._available_tools: list[str] = []

    @property
    def tool_count(self) -> int:
        """Number of MCP tools discovered on the gateway (0 if none/failed)."""
        return len(self._available_tools)

    @property
    def available_tools(self) -> tuple[str, ...]:
        """Names of the MCP tools discovered on the gateway."""
        return tuple(self._available_tools)

    async def initialize(self) -> None:
        """Initialize the bridge and discover available tools.

        If no gateway is provided, attempts a lazy import of the
        memfun-tools gateway. Gracefully degrades if tools are
        unavailable.
        """
        if self._initialized:
            return

        if self._gateway is None:
            try:
                from memfun_tools.gateway import (  # type: ignore[import-untyped]
                    create_gateway,
                )

                self._gateway = create_gateway()
            except ImportError:
                logger.warning(
                    "memfun-tools not available; MCP tools will be unavailable in the REPL"
                )
                self._initialized = True
                return

        # Discover available tools.  A failure here leaves every specialist
        # without MCP tools, so it is logged loudly (WARNING + traceback).
        try:
            self._available_tools = await _discover_tool_names(self._gateway)
        except Exception as exc:
            logger.warning(
                "Failed to discover MCP tools (%s: %s); MCP tools will be unavailable in the REPL",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self._available_tools = []
        else:
            if not self._available_tools:
                logger.warning(
                    "MCP tool discovery found ZERO tools on gateway %r; "
                    "MCP tools will be unavailable in the REPL",
                    getattr(self._gateway, "name", self._gateway),
                )
            else:
                logger.info(
                    "MCP Tool Bridge initialized with %d tools: %s",
                    len(self._available_tools),
                    ", ".join(self._available_tools[:10]),
                )
                missing = sorted(set(_REPL_TOOL_TARGETS) - set(self._available_tools))
                if missing:
                    logger.warning(
                        "MCP gateway is missing tools the REPL wrappers expect: %s",
                        ", ".join(missing),
                    )

        self._initialized = True

    async def call_tool(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Invoke an MCP tool by name.

        Args:
            tool_name: Namespaced gateway tool name (e.g., ``fs_read_file``).
            **kwargs: Tool arguments.

        Returns:
            ToolResult with the tool output or error.
        """
        if self._gateway is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error="MCP gateway not available",
            )

        try:
            result = await self._gateway.call_tool(tool_name, kwargs)
            content, is_error = _render_tool_result(result)
            if is_error:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    content=content,
                    error=content or "tool reported an error",
                )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                content=content,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error=f"{type(exc).__name__}: {exc}",
            )

    def get_repl_tools(self) -> dict[str, Callable[..., Any]]:
        """Build synchronous Python wrappers for common MCP tools.

        These wrappers are designed for injection into the RLM REPL
        namespace. They use synchronous APIs by running the async
        tool calls in an event loop.

        Returns:
            Dict mapping function names to callable wrappers.
        """
        tools: dict[str, Callable[..., Any]] = {}

        # File system tools
        tools["read_file"] = self._make_sync_tool(
            "fs_read_file",
            "Read a file. Args: path (str), offset (int)=0, limit (int)=2000",
        )
        tools["write_file"] = self._make_sync_tool(
            "fs_write_file",
            "Write content to a file. Args: path (str), content (str)",
        )
        tools["list_directory"] = self._make_sync_tool(
            "fs_list_directory",
            "List directory contents. Args: path (str), recursive (bool)=False",
        )
        tools["glob_files"] = self._make_sync_tool(
            "fs_glob_files",
            "Glob for files. Args: pattern (str), path (str)='.'",
            pattern_args=("pattern",),
        )

        # Search tools
        tools["grep"] = self._make_sync_tool(
            "search_grep",
            "Search file contents with ripgrep. Args: "
            "pattern (str), path (str)='.', glob (str|None), "
            "case_insensitive (bool)=False",
            pattern_args=("glob",),
        )

        # Git tools
        tools["git_status"] = self._make_sync_tool(
            "git_git_status",
            "Git status. Args: path (str)='.'",
        )
        tools["git_diff"] = self._make_sync_tool(
            "git_git_diff",
            "Git diff. Args: path (str)='.', staged (bool)=False, file (str|None)=None",
            pattern_args=("file",),
        )
        tools["git_log"] = self._make_sync_tool(
            "git_git_log",
            "Git log. Args: path (str)='.', count (int)=10",
        )

        # Repo map
        tools["repo_map"] = self._make_sync_tool(
            "repo_repo_map",
            "Generate repository map. Args: path (str)='.'",
        )

        return tools

    def _make_sync_tool(
        self,
        tool_name: str,
        docstring: str,
        pattern_args: tuple[str, ...] = (),
    ) -> Callable[..., str]:
        """Create a synchronous wrapper for an MCP tool.

        The wrapper runs the async ``call_tool`` in a new event loop
        or uses the existing one via a thread-safe mechanism.

        *pattern_args* name keyword arguments the tool resolves relative
        to its ``path`` (a glob pattern, a file inside the repo).  A value
        that is absolute or has a ``..`` component would step outside
        that base -- and outside the RLM's per-task root, which only
        rewrites ``path`` -- so it is rejected before the call.  Together
        with the tools' own bounds this guards against ACCIDENTAL escapes;
        it is not a sandbox (the REPL keeps full builtins and ``run_cmd``).
        """
        bridge = self

        def tool_wrapper(**kwargs: Any) -> str:
            """Invoke an MCP tool synchronously."""
            for arg in pattern_args:
                value = kwargs.get(arg)
                if value is not None and _escapes_base(str(value)):
                    # The value is not echoed back; the message says what to fix.
                    return (
                        f"[tool error: {tool_name}]: {arg!r} must be relative to the "
                        "search path and must not contain '..' components"
                    )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're in an async context (e.g., inside RLM loop).
                # Use a thread to avoid deadlock.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        bridge.call_tool(tool_name, **kwargs),
                    )
                    result = future.result(timeout=60)
            else:
                result = asyncio.run(bridge.call_tool(tool_name, **kwargs))

            if result.success:
                return result.content
            return f"[tool error: {tool_name}]: {result.error}"

        tool_wrapper.__name__ = tool_name.split("_", 1)[-1] if "_" in tool_name else tool_name
        tool_wrapper.__doc__ = f"MCP tool: {tool_name}\n\n{docstring}"
        return tool_wrapper


# ── fastmcp compatibility helpers ──────────────────────────────


async def _discover_tool_names(gateway: Any) -> list[str]:
    """Return the tool names exposed by *gateway*.

    fastmcp 3.x (the locked version) exposes ``list_tools()`` returning a
    sequence of ``Tool`` objects; fastmcp 2.x's ``get_tools()`` (returning
    ``dict[name, Tool]``) no longer exists.  The 2.x shape is still
    accepted so a caller-supplied legacy gateway keeps working.
    """
    if hasattr(gateway, "list_tools"):
        tools: Any = await gateway.list_tools()
    else:
        tools = await gateway.get_tools()
    if isinstance(tools, dict):
        return [str(name) for name in cast("dict[Any, Any]", tools)]
    names: list[str] = []
    for tool in tools:
        names.append(tool if isinstance(tool, str) else str(tool.name))
    # list_tools() returns every version of a tool; keep unique names in order.
    return list(dict.fromkeys(names))


def _render_tool_result(result: Any) -> tuple[str, bool]:
    """Flatten a ``call_tool`` result to ``(text, is_error)``.

    fastmcp 3.x returns a ``ToolResult`` whose ``content`` is a list of
    content blocks (fastmcp 2.x returned the list directly).  Without
    unwrapping, ``str(result)`` hands the REPL the pydantic repr
    (``content=[TextContent(...)] ...``) instead of the tool output.
    """
    is_error = bool(getattr(result, "is_error", False))
    blocks: Any = getattr(result, "content", result)
    if isinstance(blocks, list):
        parts: list[str] = []
        for item in cast("list[Any]", blocks):
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else str(item))
        return "\n".join(parts), is_error
    return str(cast("object", blocks)), is_error


def _escapes_base(value: str) -> bool:
    """True if *value* is absolute or has a ``..`` path component.

    ``/`` and ``\\`` both count as separators so a Windows-style value
    cannot hide a ``..``; a leading ``!`` (ripgrep's negated glob) is
    ignored for the absolute-path test.
    """
    import os

    bare = value[1:] if value.startswith("!") else value
    if not bare:
        return False
    if os.path.isabs(bare) or bare.startswith(("/", "\\")):
        return True
    return ".." in bare.replace("\\", "/").split("/")


# MCP tool names the ``get_repl_tools`` wrappers call.  Kept in one place
# so discovery can warn when the gateway stops exposing one of them.
_REPL_TOOL_TARGETS: tuple[str, ...] = (
    "fs_read_file",
    "fs_write_file",
    "fs_list_directory",
    "fs_glob_files",
    "search_grep",
    "git_git_status",
    "git_git_diff",
    "git_git_log",
    "repo_repo_map",
)


# ── Convenience factory ────────────────────────────────────────


async def create_tool_bridge(
    gateway: Any | None = None,
) -> MCPToolBridge:
    """Create and initialize an MCPToolBridge.

    Args:
        gateway: Optional FastMCP gateway instance. If None, the
            default memfun-tools gateway is used.

    Returns:
        Initialized MCPToolBridge ready for use.
    """
    bridge = MCPToolBridge(gateway=gateway)
    await bridge.initialize()
    return bridge
