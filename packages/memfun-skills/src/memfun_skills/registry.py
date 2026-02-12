"""Registry bridge: connects the skill loader to the runtime SkillRegistryAdapter."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memfun_runtime.protocols.skill_registry import SkillInfo

if TYPE_CHECKING:
    from pathlib import Path

    from memfun_runtime.protocols.skill_registry import SkillRegistryAdapter

    from memfun_skills.loader import SkillLoader
    from memfun_skills.types import SkillDefinition

logger = logging.getLogger("memfun.skills.registry")


def skill_to_info(skill: SkillDefinition) -> SkillInfo:
    """Convert a SkillDefinition to the runtime's SkillInfo.

    Maps the richer skill-package type to the lightweight runtime
    registration type, preserving tags and version in the metadata dict.
    """
    metadata: dict[str, str] = {
        "version": skill.version,
    }
    if skill.tags:
        metadata["tags"] = ",".join(skill.tags)

    return SkillInfo(
        name=skill.name,
        description=skill.description,
        source_path=str(skill.source_path),
        user_invocable=skill.user_invocable,
        model_invocable=skill.model_invocable,
        allowed_tools=list(skill.allowed_tools),
        metadata=metadata,
    )


class SkillRegistryBridge:
    """Bridges the filesystem-based SkillLoader with the runtime registry.

    Discovers skills via the loader, converts them to SkillInfo, and
    registers/deregisters them in the runtime's SkillRegistryAdapter.
    """

    async def sync_skills(
        self,
        loader: SkillLoader,
        registry: SkillRegistryAdapter,
        paths: list[Path] | None = None,
    ) -> list[SkillDefinition]:
        """Discover skills and synchronize them with the runtime registry.

        Existing skills that are no longer on disk are deregistered.
        New or updated skills are (re-)registered.

        Args:
            loader: The SkillLoader used for filesystem discovery.
            registry: The runtime's SkillRegistryAdapter to populate.
            paths: Optional explicit paths to scan. When *None*, uses
                the loader's default discovery paths.

        Returns:
            The list of SkillDefinition objects that were registered.
        """
        discovered = loader.discover(paths=paths)

        # Determine which skills already exist in the registry
        existing = await registry.list_skills()
        existing_names = {s.name for s in existing}
        discovered_names = {s.name for s in discovered}

        # Deregister skills that are no longer on disk
        stale = existing_names - discovered_names
        for name in stale:
            logger.info("Deregistering stale skill: %s", name)
            await registry.deregister_skill(name)

        # Register (or re-register) discovered skills
        for skill in discovered:
            info = skill_to_info(skill)
            if skill.name in existing_names:
                # Re-register to pick up any changes
                await registry.deregister_skill(skill.name)
            await registry.register_skill(info)
            logger.info("Registered skill: %s", skill.name)

        return discovered
