from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfun_runtime.protocols.skill_registry import SkillInfo


class InProcessSkillRegistry:
    """T0 skill registry: in-memory skill discovery."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillInfo] = {}

    async def register_skill(self, skill: SkillInfo) -> None:
        self._skills[skill.name] = skill

    async def deregister_skill(self, name: str) -> None:
        self._skills.pop(name, None)

    async def get_skill(self, name: str) -> SkillInfo | None:
        return self._skills.get(name)

    async def list_skills(self) -> list[SkillInfo]:
        return list(self._skills.values())

    async def search_skills(self, query: str) -> list[SkillInfo]:
        query_lower = query.lower()
        return [
            skill for skill in self._skills.values()
            if query_lower in skill.name.lower() or query_lower in skill.description.lower()
        ]
