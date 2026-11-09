"""The progress-aware verify-and-fix loop of the context-first solver: hermetic, fake LLM steps."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent.context_first import (
    ContextFirstConfig,
    ContextFirstSolver,
    OperationExecutor,
    SolveResult,
    VerifyResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _out(failed: int, passed: int) -> str:
    return f"FAILED tests/t.py::test_x - boom\n{failed} failed, {passed} passed in 0.1s"


BROKEN = "ImportError while importing\nERROR tests/t.py\n1 error in 0.1s"


class Rig:
    """A project with one owned file; the 'verifier' judges the file's current text."""

    def __init__(self, tmp: Path, outputs: dict[str, str], fixes: list[str], **cfg: Any) -> None:
        self.file = tmp / "m.py"
        self.file.write_text("start")
        self.outputs = outputs
        self.fixes = list(fixes)
        self.verify_calls = 0
        self.fix_queries: list[str] = []
        self.solver = ContextFirstSolver(tmp, ContextFirstConfig(**cfg))

        async def averify(_cmds: list[str]) -> VerifyResult:
            self.verify_calls += 1
            text = self.file.read_text()
            out = self.outputs[text]
            return VerifyResult(passed=out == "PASS", errors=out, commands_run=[])

        async def afix(query: str, full_context: str, verification_errors: str) -> list[dict]:
            self.fix_queries.append(query)
            if not self.fixes:
                return []
            self.file.write_text(self.fixes.pop(0))
            return [{"op": "run_cmd", "cmd": "true"}]

        async def asolve(query: str, full_context: str) -> SolveResult:
            self.fix_queries.append("RESTART:" + query)
            if not self.fixes:
                return SolveResult("", "", [], False)
            self.file.write_text(self.fixes.pop(0))
            return SolveResult("", "", [{"op": "run_cmd", "cmd": "true"}], False)

        self.solver.verifier.averify = averify  # type: ignore[method-assign]
        self.solver.fix_solver.afix = afix  # type: ignore[method-assign]
        self.solver.solver.asolve = asolve  # type: ignore[method-assign]
        self.executor = OperationExecutor(tmp)
        self.executor.files_created.append(str(self.file))

    def run(self) -> None:
        asyncio.run(self.solver._verify_and_fix("task", "ctx", self.executor, ["pytest"]))


def test_rollback_to_the_best_state_when_a_late_edit_breaks_it(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        {"start": _out(5, 2), "good": _out(1, 6), "broken": BROKEN},
        ["good", "broken"],
        max_fix_attempts=3,
        fix_restarts=0,
    )
    rig.run()
    assert rig.file.read_text() == "good"  # not the broken last edit


def test_continues_past_the_base_budget_only_while_improving(tmp_path: Path) -> None:
    outs = {"start": _out(6, 1), "a": _out(5, 2), "b": _out(4, 3), "c": _out(4, 3)}
    rig = Rig(tmp_path, outs, ["a", "b", "c"], max_fix_attempts=2, fix_restarts=0, fix_patience=5)
    rig.run()
    # base 2: round 3 only because round 2 improved; round 4 ("c") does not improve -> stop.
    assert rig.verify_calls == 4 and rig.file.read_text() == "c"  # a tie keeps the latest


def test_stops_at_the_base_budget_without_progress(tmp_path: Path) -> None:
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": _out(3, 1)}
    rig = Rig(tmp_path, outs, ["a", "b"], max_fix_attempts=2, fix_restarts=0, fix_patience=5)
    rig.run()
    assert rig.verify_calls == 2  # attempt 2 >= base and no improvement


def test_a_passing_round_returns_at_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"start": _out(2, 1), "fixed": "PASS"}, ["fixed"], max_fix_attempts=4)
    rig.run()
    assert rig.verify_calls == 2 and rig.file.read_text() == "fixed"


def test_stalled_patches_trigger_a_clean_slate_rewrite(tmp_path: Path) -> None:
    outs = {"start": _out(4, 1), "p1": _out(4, 1), "p2": _out(4, 1), "rewrite": "PASS"}
    rig = Rig(
        tmp_path,
        outs,
        ["p1", "p2", "rewrite"],
        max_fix_attempts=3,
        fix_patience=2,
        fix_restarts=1,
        fix_hard_cap=8,
    )
    rig.run()
    assert any(q.startswith("RESTART:") for q in rig.fix_queries)
    assert rig.file.read_text() == "rewrite"


def test_history_and_failing_tests_reach_the_fix_step(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("def test_x():\n    assert m() == 1\n")
    outs = {"start": _out(2, 1), "a": _out(2, 1), "b": "PASS"}
    rig = Rig(tmp_path, outs, ["a", "b"], max_fix_attempts=4, fix_restarts=0)
    rig.run()
    assert len(rig.fix_queries) == 2
    assert "PREVIOUS FIX ATTEMPTS" not in rig.fix_queries[0]
    assert "PREVIOUS FIX ATTEMPTS" in rig.fix_queries[1] and "round 1" in rig.fix_queries[1]


@pytest.mark.parametrize("fixes", [[]])
def test_no_proposal_does_not_loop_forever(tmp_path: Path, fixes: list[str]) -> None:
    rig = Rig(tmp_path, {"start": _out(2, 1)}, fixes, max_fix_attempts=2, fix_restarts=0)
    rig.run()
    assert rig.verify_calls <= 10 and rig.file.read_text() == "start"
