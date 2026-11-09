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
            target = self.fixes.pop(0)
            current = self.file.read_text()
            if target == current:
                return [{"op": "run_cmd", "cmd": "true"}]
            # a real edit, applied by the loop's executor (so the loop can diff it)
            return [{"op": "edit_file", "path": str(self.file), "old": current, "new": target}]

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


def test_a_stalled_round_runs_experiments_and_feeds_their_output_to_the_fix(
    tmp_path: Path,
) -> None:
    (tmp_path / "m.py").write_text("start")  # replaced by Rig; the probe runs in tmp_path
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": "PASS"}
    rig = Rig(tmp_path, outs, ["a", "b"], max_fix_attempts=5, fix_restarts=0, fix_probe_rounds=1)
    proposals: list[str] = []

    async def apropose(query: str, full_context: str, verification_errors: str, history: str):
        proposals.append(history)
        return ["print('probe-says', 6 * 7)"]

    rig.solver.probe_solver.apropose = apropose  # type: ignore[method-assign]
    seen: list[str] = []
    orig = rig.solver.fix_solver.afix

    async def spy(query: str, full_context: str, verification_errors: str):
        seen.append(full_context)
        return await orig(query, full_context, verification_errors)

    rig.solver.fix_solver.afix = spy  # type: ignore[method-assign]
    rig.run()
    # one stalled round; the fake proposes the same script again, whose output is not new: stop
    assert len(proposals) == 2
    assert "probe-says 42" in seen[-1] and "EXPERIMENT RESULTS" in seen[-1]
    assert "EXPERIMENT RESULTS" not in seen[0]


def test_probes_can_be_switched_off(tmp_path: Path) -> None:
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": "PASS"}
    rig = Rig(tmp_path, outs, ["a", "b"], max_fix_attempts=5, fix_restarts=0, fix_probe_rounds=0)

    async def boom(*_a: Any, **_k: Any):
        raise AssertionError("probe solver must not run")

    rig.solver.probe_solver.apropose = boom  # type: ignore[method-assign]
    rig.run()
    assert rig.file.read_text() == "b"


def test_history_shows_what_each_round_changed_and_flags_a_revisited_state(
    tmp_path: Path,
) -> None:
    # round 1 "start" -> fix to "a" -> fix back to "start" (an oscillation) -> fix to "b" (passes)
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": "PASS"}
    rig = Rig(tmp_path, outs, ["a", "start", "b"], max_fix_attempts=6, fix_restarts=0)
    rig.run()
    assert len(rig.fix_queries) == 3
    second, third = rig.fix_queries[1], rig.fix_queries[2]
    assert "the change was" in second and "-start" in second and "+a" in second
    assert "IDENTICAL to round 1" in third  # the file went back to the state of round 1
    assert rig.file.read_text() == "b"


class _Pred:
    """A fake dspy predictor: replies come from a script; every query is recorded."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.queries: list[str] = []

    def __call__(self, query: str, full_context: str, **_k: Any) -> Any:
        self.queries.append(query)
        ops = self.replies.pop(0)
        return type("R", (), {"reasoning": "r", "answer": "a", "operations": ops})()


EDIT = '[{"op":"edit_file","path":"m.py","old":"a","new":"b"}]'
CMD = '[{"op":"run_cmd","cmd":"pytest"}]'


def test_a_commands_only_reply_is_retried_with_an_edit_nudge() -> None:
    from memfun_agent.context_first import NO_EDIT_NUDGE, SingleShotSolver

    solver = SingleShotSolver()
    solver.predictor = _Pred([CMD, EDIT])  # type: ignore[assignment]
    res = asyncio.run(solver.asolve("do it", "ctx"))
    assert [o["op"] for o in res.operations] == ["edit_file"]
    assert solver.predictor.queries == ["do it", "do it" + NO_EDIT_NUDGE]  # type: ignore[attr-defined]


def test_a_reply_cut_off_before_its_operations_is_retried_with_a_concision_nudge() -> None:
    from memfun_agent.context_first import TRUNCATION_NUDGE, SingleShotSolver

    solver = SingleShotSolver()
    cut = '[{"op":"edit_file","path":"m.py","old":"a","new":"def f():\\n    w = wcwidth(c'
    solver.predictor = _Pred([cut, EDIT])  # type: ignore[assignment]
    res = asyncio.run(solver.asolve("do it", "ctx"))
    assert [o["op"] for o in res.operations] == ["edit_file"] and not res.truncated
    assert solver.predictor.queries[1] == "do it" + TRUNCATION_NUDGE  # type: ignore[attr-defined]


def test_a_good_first_reply_is_not_retried_and_a_second_failure_is_returned() -> None:
    from memfun_agent.context_first import SingleShotSolver

    solver = SingleShotSolver()
    solver.predictor = _Pred([EDIT])  # type: ignore[assignment]
    asyncio.run(solver.asolve("q", "ctx"))
    assert len(solver.predictor.queries) == 1  # type: ignore[attr-defined]
    solver.predictor = _Pred([CMD, CMD])  # type: ignore[assignment]
    res = asyncio.run(solver.asolve("q", "ctx"))
    assert [o["op"] for o in res.operations] == ["run_cmd"]  # gave it one retry, then returned
    assert len(solver.predictor.queries) == 2  # type: ignore[attr-defined]


# ── the LOOK step in the loop ────────────────────────────────────────────────────────────


def _trust(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMFUN_TRUST_PROJECT_VERIFY", "1")


def test_a_12_round_budget_in_verify_json_is_read_and_out_of_range_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import json

    from memfun_agent.context_first import read_project_verify

    _trust(monkeypatch)
    (tmp_path / ".memfun").mkdir()
    vf = tmp_path / ".memfun" / "verify.json"
    vf.write_text(json.dumps({"commands": ["pytest"], "max_fix_attempts": 12}))
    assert read_project_verify(tmp_path) == (
        ["pytest"],
        12,
    )  # it was silently dropped before (>10)
    vf.write_text(json.dumps({"commands": ["pytest"], "max_fix_attempts": 500}))
    with caplog.at_level("WARNING"):
        assert read_project_verify(tmp_path) == (["pytest"], None)
    assert "ignoring max_fix_attempts" in caplog.text


def test_verify_json_look_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from memfun_agent.context_first import read_project_verify_options

    (tmp_path / ".memfun").mkdir()
    vf = tmp_path / ".memfun" / "verify.json"
    vf.write_text(json.dumps({"fix_detail": True, "probe_rounds": 3, "commands": []}))
    assert read_project_verify_options(tmp_path) == {}  # untrusted project: nothing
    _trust(monkeypatch)
    assert read_project_verify_options(tmp_path) == {"fix_detail": True, "probe_rounds": 3}
    vf.write_text(json.dumps({"fix_detail": "yes", "probe_rounds": -1}))
    assert read_project_verify_options(tmp_path) == {}


def test_full_failure_detail_reaches_the_fix_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from memfun_agent import fixloop

    _trust(monkeypatch)
    (tmp_path / ".memfun").mkdir()
    (tmp_path / ".memfun" / "verify.json").write_text(json.dumps({"fix_detail": True}))
    asked: list[list[str]] = []

    async def fake_detail(root: Path, cmds: list[str], ids: list[str]) -> str:
        asked.append(list(ids))
        return "--- test_x ---\nE   mismatch: 7 expected 2 got 1"

    monkeypatch.setattr(fixloop, "run_detail", fake_detail)
    rig = Rig(
        tmp_path, {"start": _out(2, 1), "b": "PASS"}, ["b"], max_fix_attempts=4, fix_restarts=0
    )
    seen: list[str] = []
    orig = rig.solver.fix_solver.afix

    async def spy(query: str, full_context: str, verification_errors: str):
        seen.append(full_context + "||" + verification_errors)
        return await orig(query, full_context, verification_errors)

    rig.solver.fix_solver.afix = spy  # type: ignore[method-assign]
    rig.run()
    assert asked == [["tests/t.py::test_x"]]
    assert "FULL FAILURE DETAIL" in seen[0] and "mismatch: 7 expected 2 got 1" in seen[0]


def test_several_probes_per_stall_while_each_prints_something_new(tmp_path: Path) -> None:
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": "PASS"}
    rig = Rig(
        tmp_path, outs, ["a", "b"], max_fix_attempts=6, fix_restarts=0,
        fix_probe_rounds=1, fix_probe_iterations=3,
    )  # fmt: skip
    calls: list[str] = []

    async def apropose(query: str, full_context: str, verification_errors: str, history: str):
        n = len(calls)
        calls.append(full_context)
        return [f"print('finding', {n})"] if n < 2 else ["print('finding', 1)"]  # 3rd repeats 2nd

    rig.solver.probe_solver.apropose = apropose  # type: ignore[method-assign]
    seen: list[str] = []
    orig = rig.solver.fix_solver.afix

    async def spy(query: str, full_context: str, verification_errors: str):
        seen.append(full_context)
        return await orig(query, full_context, verification_errors)

    rig.solver.fix_solver.afix = spy  # type: ignore[method-assign]
    rig.run()
    assert len(calls) == 3  # two new findings, then a repeat stops the stall's probing
    assert "EXPERIMENTS SO FAR" not in calls[0] and "finding 0" in calls[1]  # earlier ones shown
    assert "finding 0" in seen[-1] and "finding 1" in seen[-1]  # all reach the fix call


def test_a_stall_at_the_base_budget_probes_instead_of_stopping(tmp_path: Path) -> None:
    outs = {"start": _out(3, 1), "a": _out(3, 1), "b": _out(3, 1)}
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = Rig(
        tmp_path / "off", outs, ["a", "b"], max_fix_attempts=2, fix_restarts=0, fix_patience=9
    )
    off.run()
    assert off.verify_calls == 2  # base 2 reached without progress: stop (as before)
    on = Rig(
        tmp_path / "on", outs, ["a", "b"], max_fix_attempts=2, fix_restarts=0,
        fix_patience=9, fix_probe_rounds=1,
    )  # fmt: skip
    asked: list[int] = []

    async def apropose(query: str, full_context: str, verification_errors: str, history: str):
        asked.append(1)
        return ["print('x')"]

    on.solver.probe_solver.apropose = apropose  # type: ignore[method-assign]
    on.run()
    assert asked and on.verify_calls == 3  # one probed round beyond the base, then it stops
