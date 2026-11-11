"""Answers the solver must repair or refuse, and verification commands that cannot run.

Seen on a real repository: a fix step answered with tool calls for ``/workspace`` (a place that
does not exist), and the verification asked for a linter that is not installed."""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent import context_first as cf

if TYPE_CHECKING:
    from pathlib import Path

TOOL_CALL = (
    "Let me look first.\n<tool_call>\n<function=Bash>\n<parameter=command>\n"
    "cd /workspace && ls\n</parameter>\n</function>\n</tool_call>"
)
EDIT = '[{"op": "edit_file", "path": "pkg/mod.py", "old": "x = 1", "new": "x = 2"}]'


def project(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", "utf-8")
    return tmp_path


async def run_ops(root: Path, ops: list[dict[str, Any]], **kw: Any) -> cf.OperationExecutor:
    ex = cf.OperationExecutor(root, **kw)
    await ex.execute(ops)
    return ex


@pytest.mark.asyncio
async def test_an_imagined_absolute_path_is_mapped_onto_the_project(tmp_path: Path) -> None:
    root = project(tmp_path)
    ghost = "/no-such-root-xyz/checkout"
    ex = await run_ops(root, [
        {"op": "edit_file", "path": f"{ghost}/pkg/mod.py", "old": "x = 1", "new": "x = 2"},
        {"op": "write_file", "path": f"{ghost}/pkg/new.py", "content": "y = 1\n"},
    ])  # fmt: skip
    assert (root / "pkg/mod.py").read_text("utf-8") == "x = 2\n"
    assert (root / "pkg/new.py").read_text("utf-8") == "y = 1\n"
    assert ex.failed == 0 and ex.files_created == [
        str(root / "pkg/mod.py"),
        str(root / "pkg/new.py"),
    ]


@pytest.mark.asyncio
async def test_a_path_nobody_has_is_refused_and_nothing_is_created(tmp_path: Path) -> None:
    root = project(tmp_path)
    ghost = tmp_path.parent / "no-such-root-xyz"
    ex = await run_ops(root, [
        {"op": "write_file", "path": f"{ghost}/elsewhere/new.py", "content": "y = 1\n"},
        {"op": "edit_file", "path": f"{ghost}/elsewhere/mod.py", "old": "x = 1", "new": "x = 2"},
        {"op": "run_cmd", "cmd": f"cd {ghost} && touch made-by-cmd"},
    ])  # fmt: skip
    assert ex.failed == 2 and ex.files_created == [] and not ghost.exists()
    assert ex.ops == [("cmd", f"cd {ghost} && touch made-by-cmd", "skipped: no such directory")]
    assert (root / "pkg/mod.py").read_text("utf-8") == "x = 1\n"


@pytest.mark.asyncio
async def test_a_real_absolute_path_is_taken_as_given(tmp_path: Path) -> None:
    (tmp_path / "proj").mkdir()
    root = project(tmp_path / "proj")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("old\n", "utf-8")
    ex = await run_ops(root, [
        {"op": "edit_file", "path": str(outside / "notes.txt"), "old": "old", "new": "new"},
        {"op": "write_file", "path": str(outside / "fresh.txt"), "content": "hi\n"},
        {"op": "run_cmd", "cmd": f"cd {outside} && true"},
    ])  # fmt: skip
    assert ex.failed == 0 and (outside / "notes.txt").read_text("utf-8") == "new\n"
    assert (outside / "fresh.txt").exists() and ex.ops[-1][2] == 0


@pytest.mark.asyncio
async def test_an_edit_that_changes_nothing_is_a_failed_edit(tmp_path: Path) -> None:
    """Seen on a real repository: the solver "replaced" a block by the same block, the file
    counted as edited, the checks passed and the run ended a success with an empty patch."""
    root = project(tmp_path)
    same = {"op": "edit_file", "path": str(root / "pkg/mod.py"), "old": "x = 1", "new": "x = 1"}
    ex = await run_ops(root, [same])
    assert ex.attempted == 1 and ex.failed == 1 and ex.files_created == [] and ex.ops == []
    assert ex.edit_diagnostics[0].strategy_tried.startswith("no-op")
    assert (root / "pkg/mod.py").read_text("utf-8") == "x = 1\n"
    real = {"op": "edit_file", "path": str(root / "pkg/mod.py"), "old": "x = 1", "new": "x = 2"}
    ex = await run_ops(root, [same, real])
    assert ex.failed == 1 and ex.files_created == [str(root / "pkg/mod.py")]


def test_repair_prefers_the_longest_tail_that_exists(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "mod.py").write_text("", "utf-8")
    got = cf.repair_foreign_path("/ghost-xyz/a/pkg/mod.py", root, for_write=False)
    assert got == root / "pkg" / "mod.py"  # not the top-level mod.py
    assert cf.repair_foreign_path("/ghost-xyz/zzz/none.py", root, for_write=False) is None
    # a new file is only placed in a project directory that exists, never at the top by accident
    assert cf.repair_foreign_path("/ghost-xyz/pkg/n.py", root, for_write=True) == root / "pkg/n.py"
    assert cf.repair_foreign_path("/ghost-xyz/n.py", root, for_write=True) is None


def test_tool_call_text_is_recognised() -> None:
    assert cf.looks_like_tool_call("", TOOL_CALL)
    assert cf.looks_like_tool_call('<function_calls>\n<invoke name="x">')
    assert not cf.looks_like_tool_call(EDIT, "I call the function f(x) and it returns a tool.")


class FakePredictor:
    def __init__(self, replies: list[Any]) -> None:
        self.replies, self.asked = list(replies), []

    def __call__(self, **kw: Any) -> Any:
        self.asked.append(kw["query"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(reasoning="", answer="", operations=reply)


@pytest.mark.asyncio
async def test_a_fix_answer_written_as_tool_calls_is_asked_again_once() -> None:
    fixer = cf.FixSolver()
    fixer.predictor = FakePredictor([TOOL_CALL, EDIT])  # type: ignore[assignment]
    ops = await fixer.afix("task", "ctx", "errors")
    assert [o["path"] for o in ops] == ["pkg/mod.py"]
    asked = fixer.predictor.asked  # type: ignore[attr-defined]
    assert asked[0] == "task" and asked[1] == "task" + cf.TOOL_CALL_NUDGE
    fixer.predictor = FakePredictor([TOOL_CALL, TOOL_CALL])  # type: ignore[assignment]
    assert await fixer.afix("task", "ctx", "errors") == []  # twice: rejected, no third call
    fixer.predictor = FakePredictor(["[]"])  # type: ignore[assignment]
    assert await fixer.afix("task", "ctx", "errors") == []  # "nothing to fix" is an answer
    assert len(fixer.predictor.asked) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_an_unparseable_fix_answer_is_asked_again_once() -> None:
    class AdapterParseError(Exception):
        pass

    fixer = cf.FixSolver()
    fixer.predictor = FakePredictor([AdapterParseError("no fields"), EDIT])  # type: ignore[assignment]
    assert len(await fixer.afix("task", "ctx", "errors")) == 1
    fixer.predictor = FakePredictor([AdapterParseError("a"), AdapterParseError("b")])  # type: ignore[assignment]
    with pytest.raises(AdapterParseError):
        await fixer.afix("task", "ctx", "errors")
    fixer.predictor = FakePredictor([RuntimeError("network")])  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await fixer.afix("task", "ctx", "errors")


@pytest.mark.asyncio
async def test_the_solver_asks_again_when_its_answer_is_tool_calls() -> None:
    solver = cf.SingleShotSolver()
    solver.predictor = FakePredictor([TOOL_CALL, EDIT])  # type: ignore[assignment]
    result = await solver.asolve("task", "ctx")
    assert len(result.operations) == 1 and not result.truncated
    assert solver.predictor.asked[1].endswith(cf.TOOL_CALL_NUDGE)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_command_that_is_not_installed_is_skipped_not_failed(tmp_path: Path) -> None:
    v = cf.Verifier(tmp_path)
    res = await v.averify(["no-such-linter-xyz check .", f"{sys.executable} -c 'print(1)'"])
    assert res.passed and res.errors == "" and res.skipped == ["no-such-linter-xyz check ."]
    res = await v.averify([f"{sys.executable} -c 'import sys; print(\"boom\"); sys.exit(3)'"])
    assert not res.passed and "(exit 3)" in res.errors and "boom" in res.errors
    # exit 127 from a program that DID run is a failure, not "not installed"
    res = await v.averify([f"{sys.executable} -c 'import sys; sys.exit(127)'"])
    assert not res.passed and res.skipped == []


@pytest.mark.asyncio
async def test_a_command_over_its_time_limit_is_killed_with_its_children(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    child = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    )
    cmd = f"{sys.executable} -c {child!r} & wait"  # the shell's child, not the shell itself
    res = await cf.Verifier(tmp_path, timeout=3.0).averify([cmd])
    assert not res.passed and "timed out after 3s" in res.errors
    pid = int(pid_file.read_text("utf-8"))
    gone = False
    for _ in range(200):  # reaping takes a moment; no assertion on how long
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        await asyncio.sleep(0.05)
    assert gone, "the timed-out command's child process is still alive"
