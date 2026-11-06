from textkit.roman import from_roman, to_roman
from textkit.slugify import slugify
from textkit.wrap import wrap


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_wrap_basic():
    assert wrap("a bb ccc", 4) == ["a bb", "ccc"]


def test_roman_basic():
    assert to_roman(4) == "IV"
    assert from_roman("IV") == 4
