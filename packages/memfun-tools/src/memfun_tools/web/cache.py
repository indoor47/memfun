from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """Simple in-memory TTL cache."""

    def __init__(self, default_ttl: int = 900) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get a value if it exists and hasn't expired."""
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() >= expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with TTL."""
        ttl = ttl or self._default_ttl
        self._store[key] = (value, time.time() + ttl)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def cleanup(self) -> int:
        """Remove expired entries. Returns number removed."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]
        return len(expired)
