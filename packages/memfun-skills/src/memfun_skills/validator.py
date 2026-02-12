"""Skill validation: checks skill definitions against the agentskills.io spec."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from memfun_core.errors import SkillValidationError

if TYPE_CHECKING:
    from memfun_skills.types import SkillDefinition

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_NAME_MAX_LENGTH = 64
_DESCRIPTION_MAX_LENGTH = 1024
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)


class SkillValidator:
    """Validates a SkillDefinition against the agentskills.io specification."""

    def validate(self, skill: SkillDefinition) -> list[str]:
        """Return a list of validation error messages.

        An empty list means the skill is valid.
        """
        errors: list[str] = []

        # Name checks
        if not skill.name:
            errors.append("Skill name is required.")
        else:
            if len(skill.name) > _NAME_MAX_LENGTH:
                errors.append(
                    f"Skill name exceeds {_NAME_MAX_LENGTH} characters: "
                    f"'{skill.name}' ({len(skill.name)} chars)."
                )
            if not _NAME_PATTERN.match(skill.name):
                errors.append(
                    f"Skill name must be lowercase alphanumeric with hyphens: "
                    f"'{skill.name}'."
                )

        # Description checks
        if not skill.description:
            errors.append("Skill description is required.")
        elif len(skill.description) > _DESCRIPTION_MAX_LENGTH:
            errors.append(
                f"Skill description exceeds {_DESCRIPTION_MAX_LENGTH} characters "
                f"({len(skill.description)} chars)."
            )

        # Version check (semver)
        if skill.version and not _SEMVER_PATTERN.match(skill.version):
            errors.append(
                f"Skill version is not valid SemVer: '{skill.version}'."
            )

        # If scripts directory exists, allowed_tools should not be empty
        if skill.scripts_dir is not None and not skill.allowed_tools:
            errors.append(
                "Skill has a scripts/ directory but no allowed-tools declared."
            )

        return errors

    def validate_strict(self, skill: SkillDefinition) -> None:
        """Validate and raise SkillValidationError if invalid.

        Raises:
            SkillValidationError: With all validation errors joined.
        """
        errors = self.validate(skill)
        if errors:
            combined = "; ".join(errors)
            msg = f"Skill '{skill.name or '<unnamed>'}' validation failed: {combined}"
            raise SkillValidationError(msg)
