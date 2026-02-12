"""MCP tool server exposing memfun skills as tools."""
from __future__ import annotations

from fastmcp import FastMCP

skills_server = FastMCP("memfun-skills")


@skills_server.tool()
async def invoke_skill(
    skill_name: str, query: str, context: str = ""
) -> str:
    """Invoke a skill by name (dry-run prompt assembly).

    Discovers the skill definition, activates it (resolves references
    and maps tools), builds the execution prompt with the given query,
    and returns the assembled prompt.  No LLM is called.

    Args:
        skill_name: The exact name of the skill to invoke.
        query: The user query / task to pass to the skill.
        context: Optional additional context to include.
    """
    from pathlib import Path

    from memfun_skills.activator import SkillActivator
    from memfun_skills.executor import SkillExecutor
    from memfun_skills.loader import SkillLoader
    from memfun_skills.router import SkillRouter
    from memfun_skills.types import SkillExecutionContext

    loader = SkillLoader()
    skills = loader.discover()

    if not skills:
        return "No skills discovered. Ensure skill directories exist."

    router = SkillRouter()
    skill = router.route(skill_name, skills)

    if skill is None:
        available = ", ".join(s.name for s in skills)
        return (
            f"Skill '{skill_name}' not found. "
            f"Available skills: {available}"
        )

    exec_context = SkillExecutionContext(
        skill=skill,
        arguments={"context": context} if context else {},
        working_dir=Path.cwd(),
    )

    activator = SkillActivator()
    activated = await activator.activate(skill, exec_context)

    executor = SkillExecutor(llm_callback=None)
    task_payload: dict[str, str] = {"query": query}
    if context:
        task_payload["context"] = context

    result = await executor.execute(activated, task_payload)

    if not result.success:
        return f"Skill execution failed: {result.error}"

    return result.output


@skills_server.tool()
async def list_skills() -> str:
    """List all available skills with name, description, version, and tags.

    Discovers skills from the default search paths and returns a
    formatted listing.
    """
    from memfun_skills.loader import SkillLoader

    loader = SkillLoader()
    skills = loader.discover()

    if not skills:
        return "No skills discovered."

    lines: list[str] = [f"Found {len(skills)} skill(s):\n"]
    for s in skills:
        tags = ", ".join(s.tags) if s.tags else "(none)"
        lines.append(f"- **{s.name}** v{s.version}")
        lines.append(f"  {s.description}")
        lines.append(f"  Tags: {tags}")
        lines.append("")

    return "\n".join(lines)


@skills_server.tool()
async def search_skills(query: str) -> str:
    """Search for skills matching a query string.

    Uses the SkillRouter to score all discovered skills against the
    query.  Returns scored results sorted by relevance.

    Args:
        query: Search query (skill name, tag, or keyword).
    """
    from memfun_skills.loader import SkillLoader
    from memfun_skills.router import SkillRouter

    loader = SkillLoader()
    skills = loader.discover()

    if not skills:
        return "No skills discovered."

    router = SkillRouter()
    scored = router.route_all(query, skills)

    if not scored:
        return f"No skills matched query: '{query}'"

    lines: list[str] = [
        f"Found {len(scored)} match(es) for '{query}':\n"
    ]
    for s in scored:
        reasons = ", ".join(s.match_reasons)
        lines.append(
            f"- **{s.skill.name}** (score: {s.score}) -- {reasons}"
        )
        lines.append(f"  {s.skill.description}")
        lines.append("")

    return "\n".join(lines)


@skills_server.tool()
async def get_skill_info(skill_name: str) -> str:
    """Get detailed information about a specific skill.

    Returns the skill's name, description, version, tags,
    allowed_tools, and an instructions preview (first 500 chars).

    Args:
        skill_name: The exact name of the skill.
    """
    from memfun_skills.loader import SkillLoader
    from memfun_skills.router import SkillRouter

    loader = SkillLoader()
    skills = loader.discover()

    if not skills:
        return "No skills discovered."

    router = SkillRouter()
    skill = router.route(skill_name, skills)

    if skill is None:
        available = ", ".join(s.name for s in skills)
        return (
            f"Skill '{skill_name}' not found. "
            f"Available skills: {available}"
        )

    tags = ", ".join(skill.tags) if skill.tags else "(none)"
    tools = ", ".join(skill.allowed_tools) if skill.allowed_tools else "(none)"
    preview = skill.instructions[:500] if skill.instructions else "(no instructions)"
    if len(skill.instructions) > 500:
        preview += "..."

    lines: list[str] = [
        f"# {skill.name}\n",
        f"**Version:** {skill.version}",
        f"**Description:** {skill.description}",
        f"**Tags:** {tags}",
        f"**Allowed tools:** {tools}",
        f"**Source:** {skill.source_path}",
        "\n## Instructions preview\n",
        preview,
    ]

    return "\n".join(lines)
