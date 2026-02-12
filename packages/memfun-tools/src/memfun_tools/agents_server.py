"""MCP tool server exposing memfun agent definitions as tools."""
from __future__ import annotations

from fastmcp import FastMCP

agents_server = FastMCP("memfun-agents")


@agents_server.tool()
async def invoke_agent(
    agent_name: str, task: str, context: str = ""
) -> str:
    """Invoke an agent definition by name (dry-run prompt assembly).

    Loads the agent definition, builds a prompt from its instructions
    and the task payload, and returns the assembled prompt.  No LLM
    is actually called.

    Args:
        agent_name: The exact name of the agent to invoke.
        task: The task description / query to pass to the agent.
        context: Optional additional context to include.
    """
    from memfun_agent.definitions.loader import AgentLoader

    loader = AgentLoader()
    agents = loader.discover()

    if not agents:
        return "No agent definitions discovered."

    agent = next((a for a in agents if a.name == agent_name), None)

    if agent is None:
        available = ", ".join(a.name for a in agents)
        return (
            f"Agent '{agent_name}' not found. "
            f"Available agents: {available}"
        )

    # Build the dry-run prompt from the agent definition
    sections: list[str] = []

    # Header
    sections.append(f"# Agent: {agent.name} (v{agent.version})")
    sections.append(f"\n{agent.description}\n")

    # Capabilities
    if agent.capabilities:
        sections.append("## Capabilities\n")
        for cap in agent.capabilities:
            sections.append(f"- {cap}")
        sections.append("")

    # Allowed tools
    if agent.allowed_tools:
        sections.append("## Allowed Tools\n")
        for tool in agent.allowed_tools:
            sections.append(f"- {tool}")
        sections.append("")

    # Delegates to
    if agent.delegates_to:
        sections.append("## Delegates To\n")
        for delegate in agent.delegates_to:
            sections.append(f"- {delegate}")
        sections.append("")

    # Instructions
    if agent.instructions:
        sections.append("## Instructions\n")
        sections.append(agent.instructions)
        sections.append("")

    # Task payload
    sections.append("## Task\n")
    sections.append(task)

    if context:
        sections.append("\n### Additional Context\n")
        sections.append(context)

    return "\n".join(sections)


@agents_server.tool()
async def list_agents() -> str:
    """List all available agent definitions with name, description, version, and capabilities.

    Discovers agent definitions from the default search paths and
    returns a formatted listing.
    """
    from memfun_agent.definitions.loader import AgentLoader

    loader = AgentLoader()
    agents = loader.discover()

    if not agents:
        return "No agent definitions discovered."

    lines: list[str] = [f"Found {len(agents)} agent(s):\n"]
    for a in agents:
        caps = ", ".join(a.capabilities) if a.capabilities else "(none)"
        lines.append(f"- **{a.name}** v{a.version}")
        lines.append(f"  {a.description}")
        lines.append(f"  Capabilities: {caps}")
        lines.append("")

    return "\n".join(lines)


@agents_server.tool()
async def get_agent_info(agent_name: str) -> str:
    """Get detailed information about a specific agent definition.

    Returns the agent's name, description, version, capabilities,
    allowed_tools, delegates_to, and an instructions preview.

    Args:
        agent_name: The exact name of the agent.
    """
    from memfun_agent.definitions.loader import AgentLoader

    loader = AgentLoader()
    agents = loader.discover()

    if not agents:
        return "No agent definitions discovered."

    agent = next((a for a in agents if a.name == agent_name), None)

    if agent is None:
        available = ", ".join(a.name for a in agents)
        return (
            f"Agent '{agent_name}' not found. "
            f"Available agents: {available}"
        )

    caps = ", ".join(agent.capabilities) if agent.capabilities else "(none)"
    tools = ", ".join(agent.allowed_tools) if agent.allowed_tools else "(none)"
    delegates = ", ".join(agent.delegates_to) if agent.delegates_to else "(none)"
    preview = agent.instructions[:500] if agent.instructions else "(no instructions)"
    if len(agent.instructions) > 500:
        preview += "..."

    lines: list[str] = [
        f"# {agent.name}\n",
        f"**Version:** {agent.version}",
        f"**Description:** {agent.description}",
        f"**Capabilities:** {caps}",
        f"**Allowed tools:** {tools}",
        f"**Delegates to:** {delegates}",
        f"**Model:** {agent.model or '(default)'}",
        f"**Max turns:** {agent.max_turns}",
        f"**Source:** {agent.source_path}",
        "\n## Instructions preview\n",
        preview,
    ]

    return "\n".join(lines)
