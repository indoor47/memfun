"""Progress-aware fix-loop helpers (memfun_agent.fixloop): hermetic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_agent import fixloop as fl

FAIL = """\
FFF.....
=================================== FAILURES ===================================
____________________ ClsTest.test_a ____________________
tests/test_x.py:5: in test_a
    assert f(1) == 2
E   assert 1 == 2
____________________ test_b ____________________
tests/test_x.py:9: in test_b
    assert g() is None
E   AssertionError
=========================== short test summary info ============================
FAILED tests/test_x.py::ClsTest::test_a - assert 1 == 2
FAILED tests/test_x.py::test_b - AssertionError
3 failed, 5 passed in 0.12s
"""
BROKEN = """\
==================================== ERRORS ====================================
ImportError while importing test module 'tests/test_x.py'.
E   NameError: name 'nope' is not defined
=========================== short test summary info ============================
ERROR tests/test_x.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.05s
"""

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_summarize_counts_and_ids() -> None:
    s = fl.summarize(FAIL)
    assert (s.passed, s.failed, s.errors) == (5, 3, 0) and not s.broken
    assert s.failed_ids == ("tests/test_x.py::ClsTest::test_a", "tests/test_x.py::test_b")


def test_import_error_is_broken_and_scores_worst() -> None:
    assert fl.summarize(BROKEN).broken
    assert fl.score(BROKEN, False) > fl.score(FAIL, False) > fl.score("", True)


def test_score_orders_fewer_failures_first() -> None:
    one = FAIL.replace("3 failed, 5 passed", "1 failed, 7 passed")
    assert fl.score(one, False) < fl.score(FAIL, False)
    ansi = "\x1b[31mFAILED\x1b[0m tests/a.py::t\n\x1b[31m1 failed\x1b[0m, 2 passed in 0.1s"
    assert fl.summarize(ansi).failed == 1


def test_lint_output_scores_by_error_lines() -> None:
    lint = "a.py:1:1: F401 unused\na.py:2:1: E501 long\nFound 2 errors."
    assert fl.score(lint, False) == (0, 2, 0)
    assert fl.score("something failed", False) == (0, 1, 0)


def test_clip_keeps_head_and_tail() -> None:
    out = fl.clip("A" * 100 + "M" * 5000 + "Z" * 100, 400)
    assert out.startswith("A") and out.endswith("Z") and "clipped" in out and len(out) < 520


def test_failing_tests_source(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "class ClsTest:\n    def test_a(self):\n        assert f(1) == 2\n\n\n"
        "def test_b():\n    assert g() is None\n\ndef test_c():\n    pass\n"
    )
    src = fl.failing_tests_source(
        tmp_path, ("tests/test_x.py::ClsTest::test_a", "tests/test_x.py::test_b[param]")
    )
    assert "assert f(1) == 2" in src and "assert g() is None" in src and "test_c" not in src
    assert fl.failing_tests_source(tmp_path, ("tests/missing.py::t",)) == ""


def test_compact_errors_lists_failures_first() -> None:
    out = fl.compact_errors(FAIL)
    assert out.startswith("5 passed, 3 failed")
    assert "FAILED tests/test_x.py::test_b" in out and "ClsTest.test_a" in out


def test_best_so_far_restores_the_best_state(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text("v0")
    best = fl.BestSoFar([f])
    assert best.observe(1, (0, 5, -3))  # first score is the best so far
    f.write_text("v1")
    assert best.observe(2, (0, 2, -6))  # better: snapshot v1
    f.write_text("v2-broken")
    assert not best.observe(3, (1, 9, 0))  # worse: no snapshot
    assert best.restore() == 1 and f.read_text() == "v1" and best.best_round == 2
    assert best.restore() == 0  # already the best state


def test_parse_probes_accepts_json_fences_and_a_bare_snippet() -> None:
    assert fl.parse_probes('["print(1)", "print(2)"]') == ["print(1)", "print(2)"]
    assert fl.parse_probes('```json\n["print(1)"]\n```') == ["print(1)"]
    assert fl.parse_probes("print(wc('x'))") == ["print(wc('x'))"]
    assert fl.parse_probes("no code here") == []
    assert len(fl.parse_probes(str([f"print({i})" for i in range(9)]).replace("'", '"'))) == 2


def test_probes_that_change_things_or_reach_out_are_refused() -> None:
    assert fl.probe_refusal("print(1)") == ""
    for bad in (
        "import os; os.remove('x')",
        "import subprocess; subprocess.run(['ls'])",
        "import urllib.request; urllib.request.urlopen('http://x')",
    ):
        assert fl.probe_refusal(bad), bad


def test_run_probes_shows_real_output_and_refusals(tmp_path: Path) -> None:
    import asyncio

    (tmp_path / "m.py").write_text("def f(x):\n    return x * 2\n")
    out = asyncio.run(
        fl.run_probes(
            tmp_path,
            ["from m import f\nprint(f(21))", "import os; os.remove('m.py')"],
        )
    )
    assert "=> 42" in out and "refused" in out
    assert (tmp_path / "m.py").exists()


def _probe(tmp_path: Path, snippet: str) -> str:
    import asyncio

    return asyncio.run(fl.run_probes(tmp_path, [snippet]))


def test_a_probe_imports_the_project_copy_never_an_installed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library under repair may also be installed (memfun's own venv has wcwidth and
    cachetools as dependencies): the probe must see only the project's copy, even with the
    installed one on PYTHONPATH."""
    installed = tmp_path / "site" / "site-packages"
    installed.mkdir(parents=True)
    (installed / "thelib.py").write_text("WHERE = 'installed'\n")
    (installed / "othlib.py").write_text("WHERE = 'installed'\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "thelib.py").write_text("WHERE = 'project'\n")
    monkeypatch.setenv("PYTHONPATH", str(installed))
    assert "=> project" in _probe(proj, "import thelib; print(thelib.WHERE)")
    out = _probe(proj, "import othlib; print(othlib.WHERE)")  # only installed: must not import
    assert "ModuleNotFoundError" in out and "installed" not in out.split("=>")[1].split("Error")[0]
    # reading the installed copy's file directly is refused too
    out = _probe(proj, f"print(open({str(installed / 'thelib.py')!r}).read())")
    assert "PermissionError" in out and "WHERE" not in out.split("=>", 1)[1]


def test_probes_are_read_only_and_local(tmp_path: Path) -> None:
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("hush")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.txt").write_text("mine")
    assert "=> mine" in _probe(proj, "print(open('a.txt').read())")  # a/w/x-free reads still work
    for bad in (
        "open('new.txt', 'w').write('x')",
        "open('a.txt', 'a').write('x')",
        "import pathlib; pathlib.Path('n.txt').write_text('x')",
        "import os; os.mkdir('d')",
        "import os; os.rename('a.txt', 'b.txt')",
        "import socket; socket.socket().connect(('127.0.0.1', 9))",
        "import os; os.system('touch z')",
        "import subprocess; subprocess.Popen(['touch', 'z'])",
        "import os; os.fork()",
        "import ctypes; ctypes.CDLL(None)",
        "import pathlib; w = getattr(pathlib.Path('n.txt'), 'write' + '_text'); w('x')",
        "import os; getattr(os, 'sys' + 'tem')('touch z')",
        f"print(open({str(secret)!r}).read())",
        f"import os; print(os.listdir({str(secret.parent)!r}))",
        f"import glob; print(glob.glob({str(secret.parent) + '/*'!r}))",
    ):
        out = _probe(proj, bad)
        assert "Error" in out or "refused" in out, (bad, out)
    assert sorted(p.name for p in proj.iterdir()) == ["a.txt"]
    assert "hush" not in _probe(proj, f"print(open({str(secret)!r}).read())")


def test_a_probe_with_a_nul_byte_is_reported_not_raised(tmp_path: Path) -> None:
    assert "could not run" in _probe(tmp_path, "print(1)\x00")


def test_state_key_and_short_diff(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    before = {f: "a = 1\nb = 2\n"}
    after = {f: "a = 1\nb = 3\n"}
    assert fl.state_key(before) == fl.state_key({f: "a = 1\nb = 2\n"})
    assert fl.state_key(before) != fl.state_key(after)
    d = fl.short_diff(before, after)
    assert "-b = 2" in d and "+b = 3" in d and "a = 1" not in d
    assert fl.short_diff(before, before) == ""
    assert "clipped" in fl.short_diff({f: ""}, {f: "x\n" * 2000}, limit=50)


# ── the LOOK step: untruncated failures, probe fingerprint ───────────────────────────────


def test_pytest_detail_command_replaces_selectors_and_quietness() -> None:
    cmd = (
        "python -m pytest -q --tb=short -p no:cacheprovider tests/test_a.py::test_x "
        "tests/test_b.py::TestC::test_y"
    )
    out = fl.pytest_detail_command([cmd], ["tests/test_a.py::test_x"])
    assert out is not None
    assert out[:3] == ["python", "-m", "pytest"] and "-vv" in out and "-q" not in out
    assert "--tb=short" in out and out[out.index("-p") : out.index("-p") + 2] == [
        "-p",
        "no:cacheprovider",
    ]
    assert out[-1] == "tests/test_a.py::test_x" and "tests/test_b.py::TestC::test_y" not in out
    assert fl.pytest_detail_command(["ruff check ."], ["x"]) is None
    assert fl.pytest_detail_command(["pytest tests"], ["t.py::a"])[-2:] == [
        "--tb=short",
        "t.py::a",
    ]  # type: ignore[index]


LONG = """\
=================================== FAILURES ===================================
____________________ test_emoji_sequences ____________________
tests/test_e.py:9: in test_emoji_sequences
    assert bad == []
E   assert [{'seq': 'a', 'expected': 2, 'got': 1}, {'seq': 'b', 'expected': 2, 'got': 1}] == []
E     Left contains 2 more items, first extra item: {'seq': 'a', 'expected': 2, 'got': 1}
E     Full diff:
E     + [
E     +     {'seq': 'a', 'expected': 2, 'got': 1},
E     +     {'seq': 'b', 'expected': 2, 'got': 1},
E     + ]
=========================== short test summary info ============================
FAILED tests/test_e.py::test_emoji_sequences - assert [{'seq': 'a'...
1 failed in 0.1s
"""


def test_extract_failure_detail_keeps_the_head_and_the_mismatching_items() -> None:
    out = fl.extract_failure_detail(LONG)
    assert "--- test_emoji_sequences ---" in out and "'seq': 'b', 'expected': 2, 'got': 1" in out
    assert "tests/test_e.py:9" in out and "short test summary" not in out
    big = LONG.replace("E     + ]", "".join(f"E     + {i}\n" for i in range(5000)))
    clipped = fl.extract_failure_detail(big, per_test=1_000, total=1_500)
    assert len(clipped) < 1_700 and "clipped" in clipped


def test_run_detail_runs_the_failing_test_with_full_assertion_output(tmp_path: Path) -> None:
    import asyncio
    import sys

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_a():\n    got = [{'i': i, 'v': i * 2} for i in range(30)]\n"
        "    want = [{'i': i, 'v': i * 2} for i in range(29)] + [{'i': 29, 'v': 0}]\n"
        "    assert got == want\n"
    )
    cmd = f"{sys.executable} -m pytest -q --tb=short -p no:cacheprovider tests/test_x.py"
    detail = asyncio.run(fl.run_detail(tmp_path, [cmd], ["tests/test_x.py::test_a"]))
    assert (
        "test_a" in detail and "'v': 58" in detail
    ) or "'v': 0" in detail  # the differing item shows
    assert (
        asyncio.run(fl.run_detail(tmp_path, ["ruff check ."], ["tests/test_x.py::test_a"])) == ""
    )


def test_probe_fingerprint_ignores_refusals_and_empty_output_and_matches_repeats() -> None:
    a = "--- experiment 1 ---\nprint(1)\n=> mismatches: 3 of 9\n7 expected 2 got 1"
    b = "--- experiment 1 ---\nDIFFERENT SCRIPT\n=> mismatches: 3 of 9\n7 expected 2 got 1"
    assert fl.probe_fingerprint(a) and fl.probe_fingerprint(a) != ""
    assert fl.probe_fingerprint("--- experiment 1 ---\nx\n=> (no output)") == ""
    assert (
        fl.probe_fingerprint("--- experiment 1 ---\nos.remove\n=> refused (uses 'os.remove')")
        == ""
    )
    assert fl.probe_fingerprint(a.replace("=> ", "=> ")) == fl.probe_fingerprint(a)
    assert fl.probe_fingerprint(b) != fl.probe_fingerprint(
        "--- experiment 1 ---\nz\n=> 0 mismatches"
    )


def test_log_probe_appends_a_record_for_audits(tmp_path: Path) -> None:
    import json

    fl.log_probe(tmp_path, ["print(1)"], "--- experiment 1 ---\nprint(1)\n=> 1")
    fl.log_probe(tmp_path, ["print(2)"], "out2")
    lines = (tmp_path / fl.PROBE_LOG).read_text().splitlines()
    rec = json.loads(lines[0])
    assert len(lines) == 2 and rec["scripts"] == ["print(1)"] and "=> 1" in rec["output"]
    fl.log_probe(tmp_path / "missing" / "deeper", ["x"], "y")  # best effort: never raises
