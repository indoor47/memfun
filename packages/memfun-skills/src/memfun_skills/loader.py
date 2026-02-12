"""Skill discovery and loading from the filesystem."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from memfun_skills.parser import parse_skill_md

if TYPE_CHECKING:
    from memfun_skills.types import SkillDefinition

logger = logging.getLogger("memfun.skills.loader")

# Default paths searched for skill directories
DEFAULT_DISCOVERY_PATHS: list[Path] = [
    Path("./skills"),
    Path.home() / ".memfun" / "skills",
]

_SKILL_FILENAME = "SKILL.md"


class SkillLoader:
    """Discovers and loads skills from the filesystem.

    Skills are directories containing a ``SKILL.md`` file.  The loader
    scans configurable search paths (including nested sub-directories)
    and parses each skill definition.
    """

    def __init__(self, extra_paths: list[Path] | None = None) -> None:
        self._extra_paths: list[Path] = extra_paths or []

    @property
    def search_paths(self) -> list[Path]:
        """All paths that will be scanned for skills."""
        return DEFAULT_DISCOVERY_PATHS + self._extra_paths

    def discover(self, paths: list[Path] | None = None) -> list[SkillDefinition]:
        """Scan directories for SKILL.md files and return parsed definitions.

        Args:
            paths: Directories to scan.  When *None*, uses the default
                discovery paths plus any extra paths configured on this
                loader instance.

        Returns:
            A list of successfully parsed SkillDefinition objects.
            Skills that fail to parse are logged and skipped.
        """
        scan_paths = paths if paths is not None else self.search_paths
        skills: list[SkillDefinition] = []
        seen_names: set[str] = set()

        for base in scan_paths:
            resolved = base.expanduser().resolve()
            if not resolved.is_dir():
                logger.debug("Skipping non-existent path: %s", resolved)
                continue

            for skill_file in sorted(resolved.rglob(_SKILL_FILENAME)):
                try:
                    skill = self.load_skill(skill_file.parent)
                except Exception:
                    logger.warning(
                        "Failed to load skill from %s",
                        skill_file,
                        exc_info=True,
                    )
                    continue

                if skill.name in seen_names:
                    logger.warning(
                        "Duplicate skill name '%s' at %s (skipping)",
                        skill.name,
                        skill_file,
                    )
                    continue

                seen_names.add(skill.name)
                skills.append(skill)

        logger.info("Discovered %d skill(s)", len(skills))
        return skills

    def load_skill(self, path: Path) -> SkillDefinition:
        """Load a single skill from its directory.

        Args:
            path: The skill directory (must contain a ``SKILL.md`` file).

        Returns:
            The parsed SkillDefinition.

        Raises:
            FileNotFoundError: If SKILL.md does not exist in *path*.
            SkillValidationError: If the skill file is malformed.
        """
        skill_file = path / _SKILL_FILENAME
        return parse_skill_md(skill_file)
