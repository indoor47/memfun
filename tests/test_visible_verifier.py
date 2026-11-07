"""The coder runs the task's VISIBLE tests and iterates; no-edit / unparseable tries are retried.

Evidence (B300 wcwidth function tasks, cycle 1): every agent edited once and never ran a test
(1 edit, 0 commands), because the single-task solver only verifies with a linter it finds
(wcwidth has no ruff config), and 3 of 36 episodes ended with an unparseable solver reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from memfun_agent import context_first as cf
from memfun_agent.workflow import SINGLE_TASK_NO_EDIT_RETRIES, WorkflowEngine, WorkflowState
from memfun_core.types import TaskResult

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import commit0_adapter as c0  # noqa: E402
from evals.swarm import run_smoke as rs  # noqa: E402

# ── project verify file ──────────────────────────────────────────────────────


def _write_verify(root: Path, doc: Any) -> None:
    (root / ".memfun").mkdir(exist_ok=True)
    (root / ".memfun" / "verify.json").write_text(
        doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8"
    )


def test_read_project_verify(tmp_path: Path) -> None:
    assert cf.read_project_verify(tmp_path) == ([], None)  # absent
    _write_verify(tmp_path, {"commands": ["pytest -q a", "", 3], "max_fix_attempts": 4})
    assert cf.read_project_verify(tmp_path) == (["pytest -q a"], 4)
    _write_verify(tmp_path, {"commands": ["x"], "max_fix_attempts": 99})
    assert cf.read_project_verify(tmp_path) == (["x"], None)  # out of range: config default
    _write_verify(tmp_path, "not json")
    assert cf.read_project_verify(tmp_path) == ([], None)
    _write_verify(tmp_path, [1, 2])
    assert cf.read_project_verify(tmp_path) == ([], None)


def test_solver_uses_the_project_verify_file_before_auto_detection(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")  # would auto-detect ruff
    solver = cf.ContextFirstSolver.__new__(cf.ContextFirstSolver)
    solver.project_root = tmp_path
    solver.config = cf.ContextFirstConfig()
    assert solver._get_verify_commands() == ["ruff check ."]  # pyright: ignore[reportPrivateUsage]
    assert solver._max_fix_attempts() == 2  # pyright: ignore[reportPrivateUsage]
    _write_verify(tmp_path, {"commands": ["python -m pytest -q t"], "max_fix_attempts": 4})
    assert solver._get_verify_commands() == ["python -m pytest -q t"]  # pyright: ignore[reportPrivateUsage]
    assert solver._max_fix_attempts() == 4  # pyright: ignore[reportPrivateUsage]
    solver.config = cf.ContextFirstConfig(verify_commands=("explicit",))
    assert solver._get_verify_commands() == ["explicit"]  # pyright: ignore[reportPrivateUsage]


def test_failing_visible_tests_are_fed_back_until_they_pass(tmp_path: Path) -> None:
    """verify -> fix -> verify with a real pytest command; the fix is applied by the executor."""
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import f\n\ndef test_f():\n    assert f() == 2\n"
    )
    _write_verify(
        tmp_path, {"commands": [f"{sys.executable} -m pytest -q -p no:cacheprovider test_mod.py"]}
    )
    seen: list[str] = []

    class FakeFix:
        async def afix(self, query: str, full_context: str, verification_errors: str) -> list[Any]:
            seen.append(verification_errors)
            (tmp_path / "mod.py").write_text("def f():\n    return 2\n")  # the "fix"
            return [{"op": "noop"}]

    solver = cf.ContextFirstSolver.__new__(cf.ContextFirstSolver)
    solver.project_root = tmp_path
    solver.config = cf.ContextFirstConfig()
    solver.verifier = cf.Verifier(tmp_path)
    solver.fix_solver = FakeFix()  # type: ignore[assignment]
    solver._status = lambda *_a, **_k: None  # type: ignore[method-assign]
    solver._read_affected_files = AsyncMock(return_value="ctx")  # type: ignore[method-assign]
    executor = MagicMock(files_created=["mod.py"])
    executor.aexecute = AsyncMock(return_value=None)
    executor.execute_ops = AsyncMock(return_value=None)

    async def run() -> None:
        await solver._verify_and_fix(  # pyright: ignore[reportPrivateUsage]
            query="make f return 2",
            full_context="ctx",
            executor=executor,
            verify_cmds=solver._get_verify_commands(),  # pyright: ignore[reportPrivateUsage]
        )

    with contextlib.suppress(Exception):  # the fake executor lacks some executor hooks
        asyncio.run(run())
    assert seen and "assert f() == 2" in seen[0]  # the failing test output reached the fixer


# ── unparseable first reply is retried once ──────────────────────────────────


def _solver_with(replies: list[str]) -> tuple[cf.SingleShotSolver, list[int]]:
    calls: list[int] = []
    solver = cf.SingleShotSolver.__new__(cf.SingleShotSolver)

    def predictor(**_kw: Any) -> Any:
        calls.append(1)
        return types.SimpleNamespace(
            reasoning="r", answer="a", operations=replies[min(len(calls), len(replies)) - 1]
        )

    solver.predictor = predictor  # type: ignore[assignment]
    return solver, calls


GOOD = json.dumps([{"op": "write", "path": "a.py", "content": "x = 1\n"}])
BAD = "I will now edit the file with the following changes: " * 8  # > 100 chars, 0 parsed


def test_unparseable_reply_is_retried_once_and_the_retry_is_used() -> None:
    solver, calls = _solver_with([BAD, GOOD])
    out = asyncio.run(solver.asolve("q", "ctx"))
    assert len(calls) == 2 and out.operations and not out.truncated


def test_two_unparseable_replies_still_report_truncated_after_one_retry() -> None:
    solver, calls = _solver_with([BAD, BAD, GOOD])
    out = asyncio.run(solver.asolve("q", "ctx"))
    assert len(calls) == 2 and out.truncated and not out.operations  # bounded: ONE retry


def test_a_good_first_reply_is_not_retried() -> None:
    solver, calls = _solver_with([GOOD])
    assert asyncio.run(solver.asolve("q", "ctx")).operations
    assert len(calls) == 1


# ── single-task no-edit retry ────────────────────────────────────────────────


def _engine(results: list[TaskResult]) -> WorkflowEngine:
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng._manager = MagicMock()
    eng._manager.is_running.return_value = True
    eng._orchestrator = MagicMock()
    eng._orchestrator.dispatch = AsyncMock(side_effect=results)
    eng._project_root = None
    eng._emit_status = lambda _s: None  # type: ignore[method-assign]
    eng._run_auto_commit = lambda **_k: None  # type: ignore[method-assign]
    return eng


def _noedit() -> TaskResult:
    return TaskResult("t", "rlm-coder", True, {"answer": "", "ops": [], "iterations": 0})


def _edit() -> TaskResult:
    ops = [{"type": "edit", "target": "a.py", "detail": "1"}]
    return TaskResult("t", "rlm-coder", True, {"answer": "ok", "ops": ops, "iterations": 1})


async def _single(results: list[TaskResult]) -> Any:
    eng = _engine(results)
    out = await eng._execute_single_task(  # pyright: ignore[reportPrivateUsage]
        WorkflowState(workflow_id="wf"), "implement f", "", None, time.perf_counter()
    )
    return out, eng._orchestrator.dispatch.await_count  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_no_edit_first_result_is_retried_once_and_the_retry_can_succeed() -> None:
    out, calls = await _single([_noedit(), _edit()])
    assert calls == 2 and out.success and out.ops and not out.no_edits


@pytest.mark.asyncio
async def test_two_no_edit_results_still_fail_closed_after_one_retry() -> None:
    out, calls = await _single([_noedit(), _noedit(), _edit()])
    assert calls == 1 + SINGLE_TASK_NO_EDIT_RETRIES == 2
    assert not out.success and out.no_edits


@pytest.mark.asyncio
async def test_an_edit_on_the_first_try_is_not_retried() -> None:
    out, calls = await _single([_edit(), _edit()])
    assert calls == 1 and out.success


@pytest.mark.asyncio
async def test_escalate_marker_is_retried_too() -> None:
    esc = TaskResult("t", "rlm-coder", True, {"_escalate": True, "ops": [], "iterations": 0})
    out, calls = await _single([esc, _edit()])
    assert calls == 2 and out.success


# ── adapter + runner plumbing ────────────────────────────────────────────────


def test_visible_test_command_maps_hidden_ids_to_the_visible_tree(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("")
    (tmp_path / "tests" / "sub").mkdir()
    (tmp_path / "tests" / "sub" / "test_deep.py").write_text("")
    cmd = c0.visible_test_command(
        tmp_path, ["test_core.py::test_a", "test_core.py::test_b[x y]", "sub/test_deep.py::test_c"]
    )
    assert cmd is not None and cmd.startswith("python -m pytest -q")
    assert "tests/test_core.py::test_a" in cmd and "'tests/test_core.py::test_b[x y]'" in cmd
    assert "tests/sub/test_deep.py::test_c" in cmd
    assert c0.visible_test_command(tmp_path, ["test_missing.py::t"]) is None
    many = [f"test_core.py::t{i}" for i in range(c0.MAX_VERIFY_IDS + 5)]
    big = c0.visible_test_command(tmp_path, many)
    assert big is not None and "tests/test_core.py" in big and "::t0" not in big  # file level


def test_runner_writes_the_episode_verify_file(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "b" / "c0s-x-f"
    (bundle_dir / "repo").mkdir(parents=True)
    (bundle_dir / "repo" / "m.py").write_text("x = 1\n")
    (bundle_dir / "task.toml").write_text(
        'id = "c0s-x-f"\n[verify]\ncommands = ["python -m pytest -q t"]\nmax_fix_attempts = 4\n'
    )
    bundle = rs.Bundle(id="c0s-x-f", path=bundle_dir)
    rs.create_episode_repo(bundle, tmp_path / "repo", tmp_path / "home", "http://h/v1", "m", 1024)
    doc = json.loads((tmp_path / "repo" / ".memfun" / "verify.json").read_text())
    assert doc == {"commands": ["python -m pytest -q t"], "max_fix_attempts": 4}
    # a bundle without [verify] gets no file (smoke bundles are unchanged)
    (bundle_dir / "task.toml").write_text('id = "c0s-x-f"\n')
    rs.create_episode_repo(
        bundle, tmp_path / "repo2", tmp_path / "home2", "http://h/v1", "m", 1024
    )
    assert not (tmp_path / "repo2" / ".memfun" / "verify.json").exists()
