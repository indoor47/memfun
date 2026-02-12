"""Agent definition discovery and loading from the filesystem."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from memfun_agent.definitions.parser import parse_agent_md

if TYPE_CHECKING:
    from memfun_agent.definitions.types import AgentDefinition

logger = logging.getLogger("memfun.agent.definitions.loader")

# Default paths searched for agent definition directories
DEFAULT_DISCOVERY_PATHS: list[Path] = [
    Path("./agents"),
    Path.home() / ".memfun" / "agents",
]

_AGENT_FILENAME = "AGENT.md"


class AgentLoader:
    """Discovers and loads agent definitions from the filesystem.

    Agent definitions are directories containing an ``AGENT.md`` file.
    The loader scans configurable search paths (including nested
    sub-directories) and parses each agent definition.
    """

    def __init__(self, extra_paths: list[Path] | None = None) -> None:
        self._extra_paths: list[Path] = extra_paths or []

    @property
    def search_paths(self) -> list[Path]:
        """All paths that will be scanned for agent definitions."""
        return DEFAULT_DISCOVERY_PATHS + self._extra_paths

    def discover(self, paths: list[Path] | None = None) -> list[AgentDefinition]:
        """Scan directories for AGENT.md files and return parsed definitions.

        Args:
            paths: Directories to scan.  When *None*, uses the default
                discovery paths plus any extra paths configured on this
                loader instance.

        Returns:
            A list of successfully parsed AgentDefinition objects.
            Agents that fail to parse are logged and skipped.
        """
        scan_paths = paths if paths is not None else self.search_paths
        agents: list[AgentDefinition] = []
        seen_names: set[str] = set()

        for base in scan_paths:
            resolved = base.expanduser().resolve()
            if not resolved.is_dir():
                logger.debug("Skipping non-existent path: %s", resolved)
                continue

            for agent_file in sorted(resolved.rglob(_AGENT_FILENAME)):
                try:
                    agent_def = self.load_agent(agent_file.parent)
                except Exception:
                    logger.warning(
                        "Failed to load agent definition from %s",
                        agent_file,
                        exc_info=True,
                    )
                    continue

                if agent_def.name in seen_names:
                    logger.warning(
                        "Duplicate agent name '%s' at %s (skipping)",
                        agent_def.name,
                        agent_file,
                    )
                    continue

                seen_names.add(agent_def.name)
                agents.append(agent_def)

        logger.info("Discovered %d agent definition(s)", len(agents))
        return agents

    def load_agent(self, path: Path) -> AgentDefinition:
        """Load a single agent definition from its directory.

        Args:
            path: The agent directory (must contain an ``AGENT.md`` file).

        Returns:
            The parsed AgentDefinition.

        Raises:
            FileNotFoundError: If AGENT.md does not exist in *path*.
            AgentValidationError: If the agent file is malformed.
        """
        agent_file = path / _AGENT_FILENAME
        return parse_agent_md(agent_file)
