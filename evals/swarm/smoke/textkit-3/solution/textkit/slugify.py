"""URL slugs."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str, sep: str = "-") -> str:
    """Lowercase ASCII slug; runs of non-[a-z0-9] become one ``sep``; no edge seps."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    parts = re.split(r"[^a-z0-9]+", ascii_text.lower())
    return sep.join(p for p in parts if p)
