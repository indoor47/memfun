from textkit.roman import to_roman
from textkit.slugify import slugify
from textkit.wrap import wrap


def test_chapter_slugs_wrap():
    titles = [f"Chapter {to_roman(n)}: The End?" for n in (1, 4, 9)]
    slugs = [slugify(t) for t in titles]
    assert slugs == ["chapter-i-the-end", "chapter-iv-the-end", "chapter-ix-the-end"]
    assert wrap(" ".join(slugs), 18) == [
        "chapter-i-the-end",
        "chapter-iv-the-end",
        "chapter-ix-the-end",
    ]
