from __future__ import annotations

import pytest
from memfun_runtime.protocols.skill_registry import SkillInfo


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def skill_registry(request):
    backends = {
        "memory": "memory_skill_registry",
        "sqlite": "sqlite_skill_registry",
        "redis": "redis_skill_registry",
        "nats": "nats_skill_registry",
    }
    return request.getfixturevalue(backends[request.param])


class TestSkillRegistryConformance:
    """Conformance tests for SkillRegistryAdapter implementations."""

    async def test_register_and_get(self, skill_registry):
        skill = SkillInfo(name="test-skill", description="A test skill", source_path="/tmp/test")
        await skill_registry.register_skill(skill)
        result = await skill_registry.get_skill("test-skill")
        assert result is not None
        assert result.name == "test-skill"
        assert result.description == "A test skill"

    async def test_get_nonexistent(self, skill_registry):
        assert await skill_registry.get_skill("nonexistent") is None

    async def test_deregister(self, skill_registry):
        skill = SkillInfo(name="rm-skill", description="To be removed", source_path="/tmp")
        await skill_registry.register_skill(skill)
        await skill_registry.deregister_skill("rm-skill")
        assert await skill_registry.get_skill("rm-skill") is None

    async def test_list_skills(self, skill_registry):
        await skill_registry.register_skill(
            SkillInfo(name="skill-a", description="Skill A", source_path="/a")
        )
        await skill_registry.register_skill(
            SkillInfo(name="skill-b", description="Skill B", source_path="/b")
        )
        skills = await skill_registry.list_skills()
        names = {s.name for s in skills}
        assert "skill-a" in names
        assert "skill-b" in names

    async def test_search_skills(self, skill_registry):
        await skill_registry.register_skill(
            SkillInfo(name="code-review", description="Review code quality", source_path="/cr")
        )
        await skill_registry.register_skill(
            SkillInfo(name="deploy", description="Deploy to production", source_path="/dp")
        )
        results = await skill_registry.search_skills("code")
        assert len(results) == 1
        assert results[0].name == "code-review"
