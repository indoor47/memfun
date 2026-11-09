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
