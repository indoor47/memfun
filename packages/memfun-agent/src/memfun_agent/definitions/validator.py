"""Agent definition validation: checks agent definitions against the spec."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from memfun_core.errors import AgentValidationError

if TYPE_CHECKING:
    from memfun_agent.definitions.types import AgentDefinition

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_NAME_MAX_LENGTH = 64
_DESCRIPTION_MAX_LENGTH = 1024
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)
_CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


class AgentValidator:
    """Validates an AgentDefinition against the AGENT.md specification."""

    def validate(self, agent: AgentDefinition) -> list[str]:
        """Return a list of validation error messages.

        An empty list means the agent definition is valid.
        """
        errors: list[str] = []

        # Name checks
        if not agent.name:
            errors.append("Agent name is required.")
        else:
            if len(agent.name) > _NAME_MAX_LENGTH:
                errors.append(
                    f"Agent name exceeds {_NAME_MAX_LENGTH} characters: "
                    f"'{agent.name}' ({len(agent.name)} chars)."
                )
            if not _NAME_PATTERN.match(agent.name):
                errors.append(
                    f"Agent name must be lowercase alphanumeric with hyphens: "
                    f"'{agent.name}'."
                )

        # Description checks
        if not agent.description:
            errors.append("Agent description is required.")
        elif len(agent.description) > _DESCRIPTION_MAX_LENGTH:
            errors.append(
                f"Agent description exceeds {_DESCRIPTION_MAX_LENGTH} characters "
                f"({len(agent.description)} chars)."
            )

        # Version check (semver)
        if agent.version and not _SEMVER_PATTERN.match(agent.version):
            errors.append(
                f"Agent version is not valid SemVer: '{agent.version}'."
            )

        # Capabilities check
        for cap in agent.capabilities:
            if not _CAPABILITY_PATTERN.match(cap):
                errors.append(
                    f"Capability must be lowercase alphanumeric with hyphens: "
                    f"'{cap}'."
                )

        # max_turns check
        if agent.max_turns < 1:
            errors.append(
                f"Agent max_turns must be at least 1, got {agent.max_turns}."
            )

        return errors

    def validate_strict(self, agent: AgentDefinition) -> None:
        """Validate and raise AgentValidationError if invalid.

        Raises:
            AgentValidationError: With all validation errors joined.
        """
        errors = self.validate(agent)
        if errors:
            combined = "; ".join(errors)
            msg = f"Agent '{agent.name or '<unnamed>'}' validation failed: {combined}"
            raise AgentValidationError(msg)
