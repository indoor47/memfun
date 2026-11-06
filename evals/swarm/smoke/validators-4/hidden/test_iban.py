import pytest
from checks.iban import iban_valid


@pytest.mark.parametrize(
    "s",
    [
        "DE89 3704 0044 0532 0130 00",
        "GB82 WEST 1234 5698 7654 32",
        "gb82west12345698765432",
        "NL91ABNA0417164300",
        "FR1420041010050500013M02606",
        "ES9121000418450200051332",
        "CH9300762011623852957",
    ],
)
def test_valid(s):
    assert iban_valid(s)


@pytest.mark.parametrize(
    "s",
    [
        "DE89 3704 0044 0532 0130 01",
        "DE89370400440532013",
        "GB82WEST1234569876543",
        "XX00",
        "1234567890123456",
        "DE89-3704-0044-0532-0130-00",
        "",
        "NL91ABNA04171643000",
    ],
)
def test_invalid(s):
    assert iban_valid(s) is False
