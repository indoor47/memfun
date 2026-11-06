from checks import validate


def test_luhn():
    assert validate("luhn", "4539 1488 0343 6467")


def test_semver():
    assert validate("semver", "1.2.3")
