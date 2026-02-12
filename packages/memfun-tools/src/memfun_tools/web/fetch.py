from __future__ import annotations

import hashlib

import httpx
from fastmcp import FastMCP

from memfun_tools.web.cache import TTLCache
from memfun_tools.web.html_to_md import html_to_markdown
from memfun_tools.web.security import (
    RateLimiter,
    URLValidationError,
    validate_url,
)

fetch_server = FastMCP("memfun-web-fetch")

_cache = TTLCache(default_ttl=900)  # 15 minutes
_rate_limiter = RateLimiter(rate_per_minute=30, burst=5)

_MAX_CONTENT_LENGTH = 500_000  # 500 KB text cap
_MAX_TIMEOUT_SECONDS = 60
_MAX_REDIRECTS = 5


@fetch_server.tool()
async def web_fetch(
    url: str,
    max_length: int = 50_000,
    timeout: int = 30,
) -> str:
    """Fetch a URL and return its content as markdown.

    Converts HTML to clean markdown. Caches responses for 15 min.
    Includes SSRF prevention (blocks private/reserved IPs).

    Args:
        url: The URL to fetch (must be http or https).
        max_length: Maximum content length to return (capped).
        timeout: Request timeout in seconds (max 60).
    """
    # Clamp user-controllable parameters
    max_length = min(max(1, max_length), _MAX_CONTENT_LENGTH)
    timeout = min(max(1, timeout), _MAX_TIMEOUT_SECONDS)

    # Rate limiting
    if not _rate_limiter.acquire():
        raise RuntimeError(
            "Rate limited. Please wait before making"
            " more web requests."
        )

    # URL validation (SSRF prevention)
    try:
        url = validate_url(url)
    except URLValidationError as e:
        raise ValueError(str(e)) from e

    # Check cache
    cache_key = hashlib.sha256(url.encode()).hexdigest()
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch with manual redirect handling so every hop is
    # validated against the SSRF blocklist.
    response = await _safe_fetch(url, timeout)

    content_type = response.headers.get("content-type", "")

    if "text/html" in content_type:
        result = html_to_markdown(
            response.text, max_length=max_length
        )
    elif any(
        ct in content_type
        for ct in ("text/plain", "application/json", "text/")
    ):
        result = response.text[:max_length]
    else:
        result = (
            f"(Binary content: {content_type},"
            f" {len(response.content)} bytes)"
        )

    _cache.set(cache_key, result)
    return result


async def _safe_fetch(
    url: str,
    timeout: int,
) -> httpx.Response:
    """Fetch with redirect-safe SSRF validation.

    Instead of letting httpx follow redirects automatically
    (which bypasses SSRF checks), we follow redirects manually
    and validate each hop against the URL blocklist.
    """
    current_url = url
    for _ in range(_MAX_REDIRECTS):
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            headers={
                "User-Agent": "Memfun/0.1 (autonomous agent)",
            },
        ) as client:
            response = await client.get(current_url)

        if response.is_redirect:
            location = response.headers.get("location", "")
            if not location:
                raise ValueError(
                    "Redirect with no Location header"
                )
            # Validate the redirect target
            try:
                current_url = validate_url(location)
            except URLValidationError as e:
                raise ValueError(
                    f"Redirect blocked (SSRF): {e}"
                ) from e
            continue

        response.raise_for_status()
        return response

    raise ValueError(
        f"Too many redirects (>{_MAX_REDIRECTS})"
    )
