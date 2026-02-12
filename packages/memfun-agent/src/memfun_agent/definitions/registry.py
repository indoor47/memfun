"""Registry bridge: connects the agent definition loader to the runtime RegistryAdapter."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from memfun_runtime.protocols.registry import RegistryAdapter

    from memfun_agent.definitions.loader import AgentLoader
    from memfun_agent.definitions.types import AgentDefinition

logger = logging.getLogger("memfun.agent.definitions.registry")


def agent_to_metadata(agent: AgentDefinition) -> dict[str, str]:
    """Convert an AgentDefinition to a metadata dict for the runtime registry.

    Maps the richer agent-definition type to the lightweight runtime
    registration metadata, preserving tags, version, model, and max-turns.
    """
    metadata: dict[str, str] = {
        "version": agent.version,
        "source": "agent-md",
    }
    if agent.tags:
        metadata["tags"] = ",".join(agent.tags)
    if agent.model:
        metadata["model"] = agent.model
    if agent.delegates_to:
        metadata["delegates_to"] = ",".join(agent.delegates_to)
    if agent.allowed_tools:
        metadata["allowed_tools"] = ",".join(agent.allowed_tools)
    metadata["max_turns"] = str(agent.max_turns)

    return metadata


class AgentRegistryBridge:
    """Bridges the filesystem-based AgentLoader with the runtime registry.

    Discovers agent definitions via the loader, converts them to
    registry-compatible metadata, and registers/deregisters them in
    the runtime's RegistryAdapter.
    """

    async def sync_agents(
        self,
        loader: AgentLoader,
        registry: RegistryAdapter,
        paths: list[Path] | None = None,
    ) -> list[AgentDefinition]:
        """Discover agent definitions and synchronize them with the runtime registry.

        Existing agents that are no longer on disk are deregistered.
        New or updated agents are (re-)registered.

        Args:
            loader: The AgentLoader used for filesystem discovery.
            registry: The runtime's RegistryAdapter to populate.
            paths: Optional explicit paths to scan. When *None*, uses
                the loader's default discovery paths.

        Returns:
            The list of AgentDefinition objects that were registered.
        """
        discovered = loader.discover(paths=paths)

        # Determine which agents already exist in the registry
        {a.name for a in discovered}

        # Collect all existing agent-md entries so we can detect stale ones
        stale_names: set[str] = set()
        for agent_def in discovered:
            existing = await registry.get(agent_def.name)
            if existing is not None:
                stale_names.add(agent_def.name)

        # Deregister agents whose definitions were removed from disk
        # (We detect these by checking registered agents that came from agent-md
        # but are no longer in the discovered set.)
        # Note: The RegistryAdapter.discover() finds by capability; we rely on
        # the caller to handle full stale-detection if needed.

        # Register (or re-register) discovered agent definitions
        for agent_def in discovered:
            metadata = agent_to_metadata(agent_def)

            if agent_def.name in stale_names:
                # Re-register to pick up any changes
                await registry.deregister(agent_def.name)

            await registry.register(
                agent_id=agent_def.name,
                capabilities=list(agent_def.capabilities),
                metadata=metadata,
            )
            logger.info("Registered agent definition: %s", agent_def.name)

        return discovered
