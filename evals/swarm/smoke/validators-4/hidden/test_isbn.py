import pytest
from checks.isbn import isbn10_to_13, isbn10_valid, isbn13_valid


@pytest.mark.parametrize("s", ["0-306-40615-2", "0306406152", "080442957X", "0 8044 2957 x"])
def test_isbn10_valid(s):
    assert isbn10_valid(s)


@pytest.mark.parametrize("s", ["0306406153", "030640615", "X306406152", "", "03064O6152"])
def test_isbn10_invalid(s):
    assert isbn10_valid(s) is False


@pytest.mark.parametrize("s", ["978-0-306-40615-7", "9780306406157", "978 1 86197 876 9"])
def test_isbn13_valid(s):
    assert isbn13_valid(s)


@pytest.mark.parametrize("s", ["9780306406158", "978030640615", "97803064061577", "abc"])
def test_isbn13_invalid(s):
    assert isbn13_valid(s) is False


def test_convert():
    assert isbn10_to_13("0-306-40615-2") == "9780306406157"
    assert isbn10_to_13("080442957X") == "9780804429573"
    with pytest.raises(ValueError):
        isbn10_to_13("0306406153")
