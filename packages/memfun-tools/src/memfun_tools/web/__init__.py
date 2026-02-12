"""Web tools: fetch and search MCP servers with SSRF prevention."""
from __future__ import annotations

from memfun_tools.web.fetch import fetch_server
from memfun_tools.web.search import search_server

__all__ = ["fetch_server", "search_server"]
