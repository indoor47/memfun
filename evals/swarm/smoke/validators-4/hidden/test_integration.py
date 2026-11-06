from checks import validate


def test_dispatch_all_kinds():
    assert validate("luhn", "79927398713")
    assert validate("isbn", "978-0-306-40615-7") and validate("isbn", "0306406152")
    assert validate("iban", "DE89 3704 0044 0532 0130 00")
    assert validate("semver", "2.0.0-rc.1+x")
    assert not any(validate(k, "garbage") for k in ("luhn", "isbn", "iban", "semver"))
