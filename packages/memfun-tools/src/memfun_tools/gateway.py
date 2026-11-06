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

    Each server is mounted under a namespace; fastmcp joins namespace and
    tool name with ``_`` (e.g. ``fs`` + ``read_file`` -> ``fs_read_file``):
    - fs_*         — File system operations (read, write, edit, list, glob)
    - search_*     — Code search (ripgrep, ast-grep)
    - git_*        — Git operations (status, diff, log, show, blame)
    - repo_*       — Repository map and file summary
    - web_fetch_*  — Web fetch
    - web_search_* — Web search
    - skills_*     — Skill discovery, search, and invocation
    - agents_*     — Agent definition discovery and invocation

    ``mount(..., namespace=...)`` is the fastmcp 3.x spelling of the
    deprecated ``prefix=`` keyword and yields identical tool names.
    """
    gateway = FastMCP("memfun-gateway")

    # Mount code tools
    gateway.mount(fs_server, namespace="fs")
    gateway.mount(search_server, namespace="search")
    gateway.mount(git_server, namespace="git")
    gateway.mount(repo_map_server, namespace="repo")

    # Mount web tools
    gateway.mount(fetch_server, namespace="web_fetch")
    gateway.mount(web_search_server, namespace="web_search")

    # Mount skills and agents
    gateway.mount(skills_server, namespace="skills")
    gateway.mount(agents_server, namespace="agents")

    return gateway


# Singleton gateway instance
gateway = create_gateway()
