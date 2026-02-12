"""Tool name mapping: translates skill-level tool names to MCP gateway names."""
from __future__ import annotations

import logging

logger = logging.getLogger("memfun.skills.tool_mapping")

# ── Default tool map ──────────────────────────────────────────

# Maps human-friendly skill-level tool names (as declared in SKILL.md
# ``allowed-tools``) to their MCP gateway equivalents.  The keys are
# compared case-insensitively.
DEFAULT_TOOL_MAP: dict[str, str] = {
    # Filesystem tools
    "Read": "fs_read_file",
    "Write": "fs_write_file",
    "ListDirectory": "fs_list_directory",
    "Glob": "fs_glob",
    # Search tools
    "Grep": "search_grep",
    # Git tools
    "GitStatus": "git_git_status",
    "GitDiff": "git_git_diff",
    "GitLog": "git_git_log",
    # Shell / execution
    "Bash": "bash_exec",
    # Repository tools
    "RepoMap": "repo_repo_map",
}

# Reverse lookup: MCP name -> human-friendly description.
_TOOL_DESCRIPTIONS: dict[str, str] = {
    "fs_read_file": "Read a file (path, offset, limit)",
    "fs_write_file": "Write content to a file (path, content)",
    "fs_list_directory": "List directory contents (path, recursive)",
    "fs_glob": "Glob for files matching a pattern (pattern, path)",
    "search_grep": "Search file contents with ripgrep (pattern, path, glob)",
    "git_git_status": "Show git working-tree status (path)",
    "git_git_diff": "Show git diff (path, staged, file)",
    "git_git_log": "Show git log (path, count)",
    "bash_exec": "Execute a shell command (command, timeout)",
    "repo_repo_map": "Generate a repository map (path)",
}


# ── Mapper ────────────────────────────────────────────────────


class ToolNameMapper:
    """Maps skill-level tool names to MCP gateway tool names.

    Supports a configurable mapping table that is merged on top of
    the built-in ``DEFAULT_TOOL_MAP``.
    """

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        # Build the case-insensitive lookup table
        self._map: dict[str, str] = {
            k.lower(): v for k, v in DEFAULT_TOOL_MAP.items()
        }
        if overrides:
            for k, v in overrides.items():
                self._map[k.lower()] = v

    def map_tool(self, skill_tool: str) -> str:
        """Map a single skill-level tool name to its MCP equivalent.

        If no mapping is found, the original name is returned unchanged
        (it may already be an MCP name or a custom tool).
        """
        mapped = self._map.get(skill_tool.lower())
        if mapped is not None:
            return mapped
        logger.debug(
            "No mapping for tool '%s'; using name as-is", skill_tool
        )
        return skill_tool

    def map_tools(self, skill_tools: list[str]) -> list[str]:
        """Map a list of skill-level tool names to MCP equivalents.

        Preserves order and removes duplicates while keeping the
        first occurrence of each mapped name.
        """
        seen: set[str] = set()
        result: list[str] = []
        for tool in skill_tools:
            mapped = self.map_tool(tool)
            if mapped not in seen:
                seen.add(mapped)
                result.append(mapped)
        return result

    @staticmethod
    def get_tool_description(tool_name: str) -> str:
        """Return a human-readable description for a tool.

        Looks up by MCP gateway name first; falls back to a generic
        description if the tool is unknown.

        Args:
            tool_name: Either a skill-level or MCP gateway tool name.

        Returns:
            A short description string.
        """
        # Direct lookup by MCP name
        desc = _TOOL_DESCRIPTIONS.get(tool_name)
        if desc is not None:
            return desc

        # Try mapping the skill-level name first, then look up
        mapped = DEFAULT_TOOL_MAP.get(tool_name)
        if mapped is not None:
            desc = _TOOL_DESCRIPTIONS.get(mapped)
            if desc is not None:
                return desc

        return f"Tool: {tool_name}"
