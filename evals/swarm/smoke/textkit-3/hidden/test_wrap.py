import pytest
from textkit.wrap import wrap


def test_exact_fit():
    assert wrap("aaa bbb", 7) == ["aaa bbb"]


def test_greedy():
    assert wrap("the quick brown fox jumps", 10) == ["the quick", "brown fox", "jumps"]


def test_long_word_split():
    assert wrap("abcdefghij k", 4) == ["abcd", "efgh", "ij k"]


def test_whitespace_collapsed():
    assert wrap("  a\n\tb   c ", 3) == ["a b", "c"]


def test_empty():
    assert wrap("   ", 5) == []


def test_bad_width():
    with pytest.raises(ValueError):
        wrap("a", 0)
