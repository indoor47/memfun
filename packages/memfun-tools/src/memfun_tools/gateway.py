from __future__ import annotations

from fastmcp import FastMCP

from memfun_tools.agents_server import agents_server
from memfun_tools.code.filesystem import fs_server
from memfun_tools.code.git import git_server
from memfun_tools.code.repo_map import repo_map_server
from memfun_tools.code.search import search_server
from memfun_tools.skills_server import skills_server
from memfun_tools.web.fetch import fetch_server
from memfun_tools.web.search import search_server as web_search_server


def create_gateway() -> FastMCP:
    """Create the main Memfun MCP gateway composing all tool servers.

    Tool namespaces:
    - fs.*       — File system operations (read, write, list, glob)
    - search.*   — Code search (ripgrep, ast-grep)
    - git.*      — Git operations (status, diff, log, show, blame)
    - repo.*     — Repository map and file summary
    - web.*      — Web fetch and search
    - skills.*   — Skill discovery, search, and invocation
    - agents.*   — Agent definition discovery and invocation
    """
    gateway = FastMCP("memfun-gateway")

    # Mount code tools
    gateway.mount(fs_server, prefix="fs")
    gateway.mount(search_server, prefix="search")
    gateway.mount(git_server, prefix="git")
    gateway.mount(repo_map_server, prefix="repo")

    # Mount web tools
    gateway.mount(fetch_server, prefix="web_fetch")
    gateway.mount(web_search_server, prefix="web_search")

    # Mount skills and agents
    gateway.mount(skills_server, prefix="skills")
    gateway.mount(agents_server, prefix="agents")

    return gateway


# Singleton gateway instance
gateway = create_gateway()
