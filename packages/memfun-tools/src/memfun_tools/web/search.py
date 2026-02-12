from __future__ import annotations

import hashlib

from fastmcp import FastMCP

from memfun_tools.web.cache import TTLCache
from memfun_tools.web.search_backends import SearchResult, get_best_backend
from memfun_tools.web.security import RateLimiter

search_server = FastMCP("memfun-web-search")

_cache = TTLCache(default_ttl=3600)  # 1 hour
_rate_limiter = RateLimiter(rate_per_minute=30, burst=5)


@search_server.tool()
async def web_search(
    query: str,
    max_results: int = 10,
    backend: str = "duckduckgo",
) -> str:
    """Search the web and return results.

    Uses pluggable search backends. DuckDuckGo is the default (no API key).
    Brave Search and Tavily are available with API keys.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        backend: Search backend (duckduckgo, brave, tavily).
    """
    if not _rate_limiter.acquire():
        raise RuntimeError("Rate limited. Please wait before making more search requests.")

    # Check cache
    cache_key = hashlib.sha256(f"{backend}:{query}:{max_results}".encode()).hexdigest()
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    search_backend = await get_best_backend(preferred=backend)
    results = await search_backend.search(query, max_results=max_results)

    formatted = _format_results(results, search_backend.name)
    _cache.set(cache_key, formatted)
    return formatted


def _format_results(results: list[SearchResult], backend_name: str) -> str:
    if not results:
        return "(no results found)"

    lines = [f"Search results (via {backend_name}):", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.title}**")
        lines.append(f"   {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        lines.append("")

    return "\n".join(lines)
