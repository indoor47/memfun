import pytest
from textkit.slugify import slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Many   spaces  ", "many-spaces"),
        ("Crème Brûlée", "creme-brulee"),
        ("--already--slugged--", "already-slugged"),
        ("C++ & Python 3.12", "c-python-3-12"),
        ("日本語", ""),
        ("", ""),
        ("ABC_def", "abc-def"),
    ],
)
def test_cases(text, expected):
    assert slugify(text) == expected


def test_custom_sep():
    assert slugify("Hello big World", sep="_") == "hello_big_world"
