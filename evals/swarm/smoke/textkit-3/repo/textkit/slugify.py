"""URL slugs."""

from __future__ import annotations


def slugify(text: str, sep: str = "-") -> str:
    """Lowercase ASCII slug; runs of non-[a-z0-9] become one ``sep``; no edge seps."""
    raise NotImplementedError
