import pytest
from formats.duration import format_duration, parse_duration


@pytest.mark.parametrize(
    ("s", "n"),
    [("45s", 45), ("2d", 172800), ("1d2h3m4s", 93784), ("90m", 5400), ("0s", 0), ("1h0m", 3600)],
)
def test_parse(s, n):
    assert parse_duration(s) == n


@pytest.mark.parametrize("s", ["", "1m1h", "1h 30m", "1.5h", "-1s", "h", "1x", "1h1h"])
def test_parse_rejects(s):
    with pytest.raises(ValueError):
        parse_duration(s)


@pytest.mark.parametrize(
    ("n", "s"), [(0, "0s"), (59, "59s"), (3600, "1h"), (3661, "1h1m1s"), (90061, "1d1h1m1s")]
)
def test_format(n, s):
    assert format_duration(n) == s


def test_roundtrip_and_negative():
    assert all(parse_duration(format_duration(n)) == n for n in range(0, 200000, 997))
    with pytest.raises(ValueError):
        format_duration(-1)
