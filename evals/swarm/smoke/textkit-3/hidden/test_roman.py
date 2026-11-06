import pytest
from textkit.roman import from_roman, to_roman


@pytest.mark.parametrize(
    ("n", "s"),
    [
        (1, "I"),
        (9, "IX"),
        (14, "XIV"),
        (40, "XL"),
        (90, "XC"),
        (400, "CD"),
        (1994, "MCMXCIV"),
        (2026, "MMXXVI"),
        (3999, "MMMCMXCIX"),
    ],
)
def test_roundtrip(n, s):
    assert to_roman(n) == s
    assert from_roman(s) == n
    assert from_roman(s.lower()) == n


@pytest.mark.parametrize("n", [0, -1, 4000])
def test_to_roman_range(n):
    with pytest.raises(ValueError):
        to_roman(n)


@pytest.mark.parametrize("s", ["", "IIII", "IC", "MMMM", "VX", "ABC", "IIV"])
def test_from_roman_rejects(s):
    with pytest.raises(ValueError):
        from_roman(s)


def test_all_values_roundtrip():
    assert all(from_roman(to_roman(n)) == n for n in range(1, 4000))
