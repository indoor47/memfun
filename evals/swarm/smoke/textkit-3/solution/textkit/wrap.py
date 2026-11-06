"""Greedy word wrapping."""

from __future__ import annotations


def wrap(text: str, width: int) -> list[str]:
    """Greedy wrap; over-long words are split into ``width``-sized chunks."""
    if width < 1:
        raise ValueError("width must be >= 1")
    words: list[str] = []
    for word in text.split():
        words.extend(word[i : i + width] for i in range(0, len(word), width))
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
