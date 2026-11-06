import pytest
from calc import CalcError, evaluate


@pytest.mark.parametrize(
    ("src", "value"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("7 / 2", 3.5),
        ("7 % 3", 1),
        ("-7 % 3", 2),
        ("2 ** 3 ** 2", 512),
        ("-2 ** 2", -4),
        ("2 ** -1", 0.5),
        ("--3", 3),
        ("+4 - -1", 5),
        (".5 + 1.25", 1.75),
        ("min(3, 1, 2) + max(4) + abs(-5)", 10),
        ("x = 4; y = x * 2; y + 1", 9),
        ("a = 2;;", 2),
        ("10 - 4 - 3", 3),
        ("2 * 3 % 4", 2),
    ],
)
def test_values(src, value):
    result = evaluate(src)
    assert result == value
    assert type(result) is type(value)


def test_env_used_not_mutated():
    env = {"r": 2}
    assert evaluate("r = r * 3; r ** 2", env) == 36
    assert env == {"r": 2}
    assert evaluate("pi * r", {"pi": 3.5, "r": 2}) == 7.0


@pytest.mark.parametrize(
    ("src", "pos"),
    [
        ("1 +", 3),
        ("(1 + 2", 6),
        ("1 $ 2", 2),
        ("foo", 0),
        ("1 / 0", 2),
        ("2 % 0", 2),
        ("nope(1)", 0),
        ("abs(1, 2)", 0),
        ("max()", 0),
        ("1 2", 2),
        ("", 0),
        ("x = ", 4),
    ],
)
def test_errors(src, pos):
    with pytest.raises(CalcError) as exc:
        evaluate(src)
    assert exc.value.pos == pos
