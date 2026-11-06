import io

import pytest
from minicli.main import main


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["1", "mi", "km"], "1.60934 km"),
        (["12", "in", "ft"], "1 ft"),
        (["5", "km", "m"], "5000 m"),
        (["100", "cm", "in"], "39.3701 in"),
        (["0", "mm", "mi"], "0 mi"),
    ],
)
def test_conversions(argv, expected):
    out = io.StringIO()
    assert main(["convert", *argv], out) == 0
    assert out.getvalue().strip() == expected


def test_unknown_unit():
    out = io.StringIO()
    assert main(["convert", "1", "m", "parsec"], out) == 2
    assert out.getvalue().strip() == "error: unknown unit parsec"
