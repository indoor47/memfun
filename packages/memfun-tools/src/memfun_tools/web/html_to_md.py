from __future__ import annotations


def html_to_markdown(html: str, max_length: int = 50000) -> str:
    """Convert HTML to clean markdown.

    Uses markdownify if available, falls back to basic tag stripping.
    """
    try:
        from markdownify import markdownify
        md = markdownify(html, heading_style="ATX", strip=["img", "script", "style"])
    except ImportError:
        md = _basic_strip(html)

    # Clean up excessive whitespace
    lines = md.splitlines()
    cleaned = []
    prev_empty = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if not prev_empty:
                cleaned.append("")
                prev_empty = True
        else:
            cleaned.append(stripped)
            prev_empty = False

    result = "\n".join(cleaned).strip()
    if len(result) > max_length:
        result = result[:max_length] + "\n\n... (truncated)"
    return result


def _basic_strip(html: str) -> str:
    """Basic HTML tag stripping fallback."""
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", "", text)
    return text
