from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for web search backends."""

    @property
    def name(self) -> str: ...

    @property
    def requires_api_key(self) -> bool: ...

    async def is_available(self) -> bool: ...

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...


class DuckDuckGoBackend:
    """DuckDuckGo search (no API key required)."""

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def requires_api_key(self) -> bool:
        return False

    async def is_available(self) -> bool:
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            return True
        except ImportError:
            return False

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        import asyncio

        from duckduckgo_search import DDGS

        def _search() -> list[SearchResult]:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                    for r in results
                ]

        return await asyncio.get_event_loop().run_in_executor(None, _search)


class BraveBackend:
    """Brave Search API backend."""

    def __init__(self, api_key_env: str = "BRAVE_API_KEY") -> None:
        self._api_key_env = api_key_env

    @property
    def name(self) -> str:
        return "brave"

    @property
    def requires_api_key(self) -> bool:
        return True

    async def is_available(self) -> bool:
        import os
        return bool(os.environ.get(self._api_key_env))

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        import os

        import httpx

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {self._api_key_env} environment variable")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            ))
        return results


class TavilyBackend:
    """Tavily Search API backend (designed for AI agents)."""

    def __init__(self, api_key_env: str = "TAVILY_API_KEY") -> None:
        self._api_key_env = api_key_env

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def requires_api_key(self) -> bool:
        return True

    async def is_available(self) -> bool:
        import os
        return bool(os.environ.get(self._api_key_env))

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        import os

        import httpx

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {self._api_key_env} environment variable")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            response.raise_for_status()
            data = response.json()

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in data.get("results", [])[:max_results]
        ]


# Backend registry
_BACKENDS: dict[str, type] = {
    "duckduckgo": DuckDuckGoBackend,
    "brave": BraveBackend,
    "tavily": TavilyBackend,
}


async def get_best_backend(preferred: str = "duckduckgo") -> SearchBackend:
    """Get the best available search backend.

    Tries the preferred backend first, then falls back through the list.
    """
    # Try preferred first
    if preferred in _BACKENDS:
        backend = _BACKENDS[preferred]()
        if await backend.is_available():
            return backend

    # Fallback order: Brave > Tavily > DuckDuckGo
    for name in ["brave", "tavily", "duckduckgo"]:
        if name == preferred:
            continue
        backend = _BACKENDS[name]()
        if await backend.is_available():
            return backend

    raise RuntimeError(
        "No search backend available. "
        "Install duckduckgo-search: pip install duckduckgo-search"
    )
