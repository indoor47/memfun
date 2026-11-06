from formats.duration import format_duration, parse_duration
from formats.inilite import dump_ini, parse_ini


def test_timeouts_config():
    cfg = parse_ini("[timeouts]\nconnect = 30s\nidle = 1h30m\n")
    seconds = {k: parse_duration(v) for k, v in cfg["timeouts"].items()}
    assert seconds == {"connect": 30, "idle": 5400}
    again = {"timeouts": {k: format_duration(v * 2) for k, v in seconds.items()}}
    assert parse_ini(dump_ini(again)) == {"timeouts": {"connect": "1m", "idle": "3h"}}
