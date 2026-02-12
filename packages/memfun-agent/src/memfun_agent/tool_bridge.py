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
from typing import TYPE_CHECKING, Any

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
        result = await bridge.call_tool("fs.read_file", path="/etc/hosts")
    """

    def __init__(
        self,
        gateway: Any | None = None,
    ) -> None:
        self._gateway = gateway
        self._initialized = False
        self._available_tools: list[str] = []

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
                    "memfun-tools not available; MCP tools "
                    "will be unavailable in the REPL"
                )
                self._initialized = True
                return

        # Discover available tools
        try:
            tools = await self._gateway.get_tools()
            self._available_tools = [
                t if isinstance(t, str) else t.name for t in tools
            ]
            logger.info(
                "MCP Tool Bridge initialized with %d tools: %s",
                len(self._available_tools),
                ", ".join(self._available_tools[:10]),
            )
        except Exception as exc:
            logger.warning(
                "Failed to discover MCP tools: %s", exc
            )
            self._available_tools = []

        self._initialized = True

    async def call_tool(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Invoke an MCP tool by name.

        Args:
            tool_name: Fully qualified tool name (e.g., ``fs.read_file``).
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
            result = await self._gateway.call_tool(
                tool_name, kwargs
            )
            # FastMCP call_tool returns a list of content blocks
            content_parts: list[str] = []
            if isinstance(result, list):
                for item in result:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))
                content = "\n".join(content_parts)
            else:
                content = str(result)

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
            "Read a file. Args: path (str), offset (int)=0, "
            "limit (int)=2000",
        )
        tools["write_file"] = self._make_sync_tool(
            "fs_write_file",
            "Write content to a file. Args: path (str), "
            "content (str)",
        )
        tools["list_directory"] = self._make_sync_tool(
            "fs_list_directory",
            "List directory contents. Args: path (str), "
            "recursive (bool)=False",
        )
        tools["glob_files"] = self._make_sync_tool(
            "fs_glob",
            "Glob for files. Args: pattern (str), path (str)='.'",
        )

        # Search tools
        tools["grep"] = self._make_sync_tool(
            "search_grep",
            "Search file contents with ripgrep. Args: "
            "pattern (str), path (str)='.', glob (str|None), "
            "case_insensitive (bool)=False",
        )

        # Git tools
        tools["git_status"] = self._make_sync_tool(
            "git_git_status",
            "Git status. Args: path (str)='.'",
        )
        tools["git_diff"] = self._make_sync_tool(
            "git_git_diff",
            "Git diff. Args: path (str)='.', staged (bool)=False, "
            "file (str|None)=None",
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
    ) -> Callable[..., str]:
        """Create a synchronous wrapper for an MCP tool.

        The wrapper runs the async ``call_tool`` in a new event loop
        or uses the existing one via a thread-safe mechanism.
        """
        bridge = self

        def tool_wrapper(**kwargs: Any) -> str:
            """Invoke an MCP tool synchronously."""
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
                result = asyncio.run(
                    bridge.call_tool(tool_name, **kwargs)
                )

            if result.success:
                return result.content
            return f"[tool error: {tool_name}]: {result.error}"

        tool_wrapper.__name__ = tool_name.split("_", 1)[-1] if "_" in tool_name else tool_name
        tool_wrapper.__doc__ = f"MCP tool: {tool_name}\n\n{docstring}"
        return tool_wrapper


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
