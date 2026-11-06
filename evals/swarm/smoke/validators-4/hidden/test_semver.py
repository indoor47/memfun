import pytest
from checks.semver import compare_semver, parse_semver, semver_valid


def test_parse():
    assert parse_semver("1.2.3") == (1, 2, 3, None)
    assert parse_semver("10.20.30-rc.1+build.5") == (10, 20, 30, "rc.1")
    assert parse_semver("1.0.0-0A.is.legal") == (1, 0, 0, "0A.is.legal")


@pytest.mark.parametrize(
    "s", ["1.2", "01.2.3", "1.02.3", "1.2.3-", "1.2.3-01", "1.2.3+", "v1.2.3", "1.2.3-a..b", ""]
)
def test_invalid(s):
    assert semver_valid(s) is False
    with pytest.raises(ValueError):
        parse_semver(s)


def test_precedence_chain():
    chain = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
        "1.0.1",
        "1.1.0",
        "2.0.0",
    ]
    for lo, hi in zip(chain, chain[1:]):
        assert compare_semver(lo, hi) == -1, (lo, hi)
        assert compare_semver(hi, lo) == 1, (hi, lo)


def test_build_ignored():
    assert compare_semver("1.0.0+a", "1.0.0+b") == 0
    assert compare_semver("1.10.0", "1.9.0") == 1
