from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """Metadata about a registered skill."""
    name: str
    description: str
    source_path: str
    user_invocable: bool = True
    model_invocable: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SkillRegistryAdapter(Protocol):
    """Skill discovery and registration."""

    async def register_skill(self, skill: SkillInfo) -> None: ...
    async def deregister_skill(self, name: str) -> None: ...
    async def get_skill(self, name: str) -> SkillInfo | None: ...
    async def list_skills(self) -> list[SkillInfo]: ...
    async def search_skills(self, query: str) -> list[SkillInfo]: ...
