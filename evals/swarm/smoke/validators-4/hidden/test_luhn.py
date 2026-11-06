import pytest
from checks.luhn import luhn_check_digit, luhn_valid


@pytest.mark.parametrize("s", ["79927398713", "4539 1488 0343 6467", "0000 0000", "18"])
def test_valid(s):
    assert luhn_valid(s)


@pytest.mark.parametrize(
    "s", ["79927398710", "4539-1488-0343-6467", "", "0", "12a4", "\u0661\u0662"]
)
def test_invalid(s):
    assert luhn_valid(s) is False


def test_check_digit():
    assert luhn_check_digit("7992739871") == "3"
    assert luhn_check_digit("453914880343646") == "7"
    for p in ("1", "12", "98765", "40128888888818"):
        assert luhn_valid(p + luhn_check_digit(p))
