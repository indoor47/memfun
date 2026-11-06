from formats.duration import parse_duration
from formats.inilite import parse_ini


def test_duration():
    assert parse_duration("1h30m") == 5400


def test_ini():
    assert parse_ini("[a]\nx = 1\n") == {"a": {"x": "1"}}
