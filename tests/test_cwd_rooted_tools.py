"""Per-task root for the RLM REPL helpers.

These tests drive the REAL helper functions built by ``rlm.py`` (and,
end to end, a real ``CoderAgent._handle_impl`` with only the LLM
stubbed) to prove that a task root set via ``RLMConfig.root`` /
``payload["cwd"]`` actually confines file writes and commands to that
root, while the process cwd stays untouched.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import dspy
import pytest
from memfun_agent import specialists as specialists_mod
from memfun_agent.rlm import (
    PathOutsideRootError,
    RLMConfig,
    RLMModule,
    build_context_metadata,
)
from memfun_agent.specialists import CoderAgent
from memfun_core.types import TaskMessage

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Return (process_cwd A, root B, outside dir C); chdir into A."""
    a = (tmp_path / "A").resolve()
    b = (tmp_path / "B").resolve()
    c = (tmp_path / "C").resolve()
    for d in (a, b, c):
        d.mkdir()
    monkeypatch.chdir(a)
    return a, b, c


def _repl_ns(root: Path | None, **kwargs: Any) -> dict[str, Any]:
    """Build a real REPL namespace via ``RLMModule._create_repl``."""
    rlm = RLMModule(
        config=RLMConfig(root=str(root) if root is not None else None),
        **kwargs,
    )
    return rlm._create_repl("").namespace  # pyright: ignore[reportPrivateUsage]


# ── (a) rooted helpers land in B, not the process cwd ─────────


def test_write_read_edit_list_run_are_rooted(dirs: tuple[Path, Path, Path]) -> None:
    a, b, _ = dirs
    ns = _repl_ns(b)

    written = ns["write_file"]("pkg/x.txt", "hello\n")
    assert written == str(b / "pkg" / "x.txt")
    assert (b / "pkg" / "x.txt").read_text() == "hello\n"
    assert not (a / "pkg").exists()

    assert ns["read_file"]("pkg/x.txt") == "hello\n"
    assert ns["edit_file"]("pkg/x.txt", "hello", "bye") is True
    assert (b / "pkg" / "x.txt").read_text() == "bye\n"

    assert ns["list_files"]() == [os.path.join(".", "pkg", "x.txt")]
    assert ns["list_files"]("pkg") == [os.path.join("pkg", "x.txt")]

    assert ns["run_cmd"]("pwd").strip() == str(b)
    ns["run_cmd"]("echo made > from_cmd.txt")
    assert (b / "from_cmd.txt").exists()
    assert not (a / "from_cmd.txt").exists()

    # Tracked ops/files report the rooted absolute paths.
    assert str(b / "pkg" / "x.txt") in ns["state"]["_files"]
    assert ns["PROJECT_ROOT"] == str(b)
    assert list(a.iterdir()) == []


def test_absolute_path_inside_root_is_allowed(dirs: tuple[Path, Path, Path]) -> None:
    _, b, _ = dirs
    ns = _repl_ns(b)
    target = b / "abs.txt"
    assert ns["write_file"](str(target), "ok") == str(target)
    assert ns["read_file"](str(target)) == "ok"


def test_list_files_with_times_in_root(dirs: tuple[Path, Path, Path]) -> None:
    _, b, _ = dirs
    (b / "t.py").write_text("x")
    ns = _repl_ns(b)
    (entry,) = ns["list_files"](".", with_times=True)
    assert entry.startswith(os.path.join(".", "t.py") + " (modified: ")


# ── (b) concurrent namespaces never cross-write ───────────────


async def test_concurrent_roots_do_not_cross_write(dirs: tuple[Path, Path, Path]) -> None:
    a, b, _ = dirs
    b2 = (b.parent / "B2").resolve()
    b2.mkdir()
    ns1, ns2 = _repl_ns(b), _repl_ns(b2)

    def worker(ns: dict[str, Any], tag: str) -> list[str]:
        pwds: list[str] = []
        for i in range(25):
            ns["write_file"](f"out/{i}.txt", tag)
            ns["edit_file"](f"out/{i}.txt", tag, tag + "!")
            pwds.append(ns["run_cmd"]("pwd").strip())
        return pwds

    p1, p2 = await asyncio.gather(
        asyncio.to_thread(worker, ns1, "one"),
        asyncio.to_thread(worker, ns2, "two"),
    )
    assert set(p1) == {str(b)}
    assert set(p2) == {str(b2)}
    for i in range(25):
        assert (b / "out" / f"{i}.txt").read_text() == "one!"
        assert (b2 / "out" / f"{i}.txt").read_text() == "two!"
    assert list(a.iterdir()) == []


class _ScriptedExplorer:
    """Stands in for the DSPy explorer predictor (the LLM)."""

    def __init__(self, code: str) -> None:
        self._code = code
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(reasoning="scripted", next_code=self._code)


async def test_concurrent_rlm_loops_are_isolated(dirs: tuple[Path, Path, Path]) -> None:
    """Two real RLM loops in one event loop, different roots."""
    a, b, _ = dirs
    b2 = (b.parent / "B2").resolve()
    b2.mkdir()
    code = (
        "write_file('result.txt', run_cmd('pwd').strip())\n"
        "state['FINAL'] = read_file('result.txt')"
    )

    def make(root: Path) -> RLMModule:
        rlm = RLMModule(config=RLMConfig(root=str(root), max_iterations=2, enable_planning=False))
        rlm.explorer = _ScriptedExplorer(code)  # type: ignore[assignment]
        return rlm

    r1, r2 = await asyncio.gather(
        make(b).aforward(query="q", context=""),
        make(b2).aforward(query="q", context=""),
    )
    assert r1.answer == str(b)
    assert r2.answer == str(b2)
    assert r1.files_created == [str(b / "result.txt")]
    assert r2.files_created == [str(b2 / "result.txt")]
    assert list(a.iterdir()) == []


# ── (c) escapes are rejected ───────────────────────────────────


@pytest.mark.parametrize("helper", ["write_file", "read_file", "edit_file", "list_files"])
def test_dotdot_escape_rejected(dirs: tuple[Path, Path, Path], helper: str) -> None:
    _, b, c = dirs
    (c / "secret.txt").write_text("s")
    ns = _repl_ns(b)
    args: dict[str, tuple[Any, ...]] = {
        "write_file": ("../C/secret.txt", "pwned"),
        "read_file": ("../C/secret.txt",),
        "edit_file": ("../C/secret.txt", "s", "pwned"),
        "list_files": ("../C",),
    }
    with pytest.raises(PathOutsideRootError):
        ns[helper](*args[helper])
    assert (c / "secret.txt").read_text() == "s"


def test_absolute_path_outside_root_rejected(dirs: tuple[Path, Path, Path]) -> None:
    a, b, c = dirs
    ns = _repl_ns(b)
    with pytest.raises(PathOutsideRootError):
        ns["write_file"](str(c / "x.txt"), "pwned")
    # The process cwd is "outside" too -- the main tree is not writable.
    with pytest.raises(PathOutsideRootError):
        ns["write_file"](str(a / "x.txt"), "pwned")
    assert not (c / "x.txt").exists()
    assert not (a / "x.txt").exists()


def test_symlink_inside_root_pointing_outside_rejected(
    dirs: tuple[Path, Path, Path],
) -> None:
    _, b, c = dirs
    (c / "target.txt").write_text("orig")
    (b / "link_dir").symlink_to(c, target_is_directory=True)
    (b / "link_file.txt").symlink_to(c / "target.txt")
    ns = _repl_ns(b)

    with pytest.raises(PathOutsideRootError):
        ns["write_file"]("link_dir/new.txt", "pwned")
    with pytest.raises(PathOutsideRootError):
        ns["write_file"]("link_file.txt", "pwned")
    with pytest.raises(PathOutsideRootError):
        ns["read_file"]("link_file.txt")
    with pytest.raises(PathOutsideRootError):
        ns["edit_file"]("link_file.txt", "orig", "pwned")
    with pytest.raises(PathOutsideRootError):
        ns["list_files"]("link_dir")
    assert not (c / "new.txt").exists()
    assert (c / "target.txt").read_text() == "orig"


def test_symlinked_root_itself_is_accepted(dirs: tuple[Path, Path, Path]) -> None:
    """A root given via a symlink is canonicalised, not rejected."""
    _, b, _ = dirs
    alias = b.parent / "B_alias"
    alias.symlink_to(b, target_is_directory=True)
    ns = _repl_ns(alias)
    assert ns["write_file"]("y.txt", "ok") == str(b / "y.txt")
    assert ns["run_cmd"]("pwd").strip() == str(b)


# ── (d) no root -> legacy process-cwd behaviour ───────────────


def test_no_root_keeps_process_cwd_behaviour(dirs: tuple[Path, Path, Path]) -> None:
    a, b, c = dirs
    ns = _repl_ns(None)
    assert "PROJECT_ROOT" not in ns

    assert ns["write_file"]("x.txt", "a") == str(a / "x.txt")
    assert (a / "x.txt").read_text() == "a"
    assert not (b / "x.txt").exists()
    assert ns["run_cmd"]("pwd").strip() == str(a)
    assert ns["list_files"]() == [os.path.join(".", "x.txt")]

    # Legacy: no containment -- absolute and ../ paths still work.
    assert ns["write_file"](str(c / "free.txt"), "c") == str(c / "free.txt")
    assert ns["read_file"]("../C/free.txt") == "c"

    # The legacy helpers follow a *later* chdir (resolved at call time).
    os.chdir(b)
    assert ns["write_file"]("late.txt", "b") == str(b / "late.txt")


def test_context_metadata_mentions_root_only_when_set() -> None:
    assert "WORKING DIRECTORY" not in build_context_metadata("ctx")
    assert "WORKING DIRECTORY: /r" in build_context_metadata("ctx", root="/r")


# ── MCP bridge tools are rooted too ───────────────────────────


def test_bridge_path_tools_are_rooted(dirs: tuple[Path, Path, Path]) -> None:
    _, b, c = dirs
    calls: list[dict[str, Any]] = []

    def fake_grep(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    def custom(x: int) -> int:
        return x

    tools = {"grep": fake_grep, "git_status": fake_grep, "custom": custom}
    ns = _repl_ns(b, extra_tools=tools)

    assert ns["grep"](pattern="foo") == "ok"
    assert calls[-1] == {"pattern": "foo", "path": str(b)}
    ns["git_status"](path="sub")
    assert calls[-1] == {"path": str(b / "sub")}
    out = ns["grep"](pattern="foo", path=str(c))
    assert out.startswith("[tool error: grep]")
    assert len(calls) == 2
    assert ns["custom"] is custom

    # No root -> bridge tools passed through untouched.
    ns_legacy = _repl_ns(None, extra_tools=tools)
    assert ns_legacy["grep"] is fake_grep


# ── (e) real specialist _handle_impl honours payload["cwd"] ───


async def test_coder_agent_writes_into_payload_cwd(
    dirs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b, _ = dirs
    explorers: list[_ScriptedExplorer] = []
    real_rlm = specialists_mod.RLMModule

    class _LLMStubbedRLM(real_rlm):
        """Real RLMModule; only the explorer predictor (the LLM) is fake."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            explorer = _ScriptedExplorer(
                "write_file('feature.py', 'x = 1\\n')\n"
                "run_cmd('echo cmd > cmd.txt')\n"
                "state['FINAL'] = 'done'"
            )
            explorers.append(explorer)
            self.explorer = explorer  # type: ignore[assignment]

    monkeypatch.setattr(specialists_mod, "RLMModule", _LLMStubbedRLM)

    post_cwds: list[str | None] = []

    async def fake_post_process(
        self: Any, query: str, rlm_result: Any, sub_task_id: str, cwd: str | None = None
    ) -> Any:
        # Post-processing is LLM-driven (lint-fix / consistency review).
        post_cwds.append(cwd)
        return rlm_result

    monkeypatch.setattr(CoderAgent, "_post_process", fake_post_process)

    agent = CoderAgent(MagicMock())
    task = TaskMessage(
        task_id="t1",
        agent_id="coder-agent",
        payload={"query": "add feature", "context": "", "cwd": str(b)},
    )
    result = await agent._handle_impl(task)  # pyright: ignore[reportPrivateUsage]

    assert result.success, result.error
    assert (b / "feature.py").read_text() == "x = 1\n"
    assert (b / "cmd.txt").exists()
    assert list(a.iterdir()) == []
    assert result.result["files_created"] == [str(b / "feature.py")]
    assert post_cwds == [str(b)]
    assert f"WORKING DIRECTORY: {b}" in explorers[0].calls[0]["context_metadata"]


async def test_coder_agent_without_cwd_uses_process_cwd(
    dirs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b, _ = dirs
    real_rlm = specialists_mod.RLMModule

    class _LLMStubbedRLM(real_rlm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            assert self.config.root is None
            self.explorer = _ScriptedExplorer(  # type: ignore[assignment]
                "write_file('legacy.py', 'y = 2\\n')\nstate['FINAL'] = 'done'"
            )

    monkeypatch.setattr(specialists_mod, "RLMModule", _LLMStubbedRLM)

    async def passthrough(self: Any, query: str, rlm_result: Any, *a: Any, **k: Any) -> Any:
        return rlm_result

    monkeypatch.setattr(CoderAgent, "_post_process", passthrough)

    agent = CoderAgent(MagicMock())
    task = TaskMessage(task_id="t2", agent_id="coder-agent", payload={"query": "q"})
    result = await agent._handle_impl(task)  # pyright: ignore[reportPrivateUsage]
    assert result.success, result.error
    assert (a / "legacy.py").exists()
    assert not (b / "legacy.py").exists()


# ── WorkflowEngine shared context honours project_root ────────


async def test_gather_shared_context_uses_engine_project_root(
    dirs: tuple[Path, Path, Path],
) -> None:
    from memfun_agent.workflow import WorkflowEngine

    a, b, _ = dirs
    (a / "process_cwd_only.py").write_text("IN_PROCESS_CWD = 1\n")
    (b / "project_only.py").write_text("IN_PROJECT_ROOT = 1\n")

    engine = WorkflowEngine(MagicMock(), MagicMock(), MagicMock(), project_root=b)
    ctx = await engine._gather_shared_context("raw", MagicMock())  # pyright: ignore[reportPrivateUsage]
    assert "IN_PROJECT_ROOT" in ctx
    assert "IN_PROCESS_CWD" not in ctx

    legacy = WorkflowEngine(MagicMock(), MagicMock(), MagicMock())
    ctx_legacy = await legacy._gather_shared_context("raw", MagicMock())  # pyright: ignore[reportPrivateUsage]
    assert "IN_PROCESS_CWD" in ctx_legacy
    assert "IN_PROJECT_ROOT" not in ctx_legacy


# ── _post_process fix/polish edits land in the task cwd ───────


async def test_post_process_fix_and_polish_edit_task_cwd(
    dirs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real ``_post_process`` (real Verifier, OperationExecutor, file
    reads); only the LLM calls (FixSolver.afix, ConsistencyReviewer.areview)
    are stubbed.  The LLM emits paths *relative* to the ``=== FILE: rel ===``
    headers, exactly as read_affected_files presents them.
    """
    import shutil

    from memfun_agent import context_first
    from memfun_agent.rlm import RLMResult

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; needed for real verification")

    a, b, _ = dirs
    # Same relative file exists in the main tree (A) and the worktree (B),
    # mirroring a git worktree of the same repo.
    for root in (a, b):
        (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (root / "feature.py").write_text("import os\nVALUE = 1\n")
    a_before = {p.name: p.read_text() for p in a.iterdir()}

    afix_calls: list[str] = []

    async def fake_afix(self: Any, query: str, full_context: str, verification_errors: str) -> Any:
        afix_calls.append(verification_errors)
        assert "=== FILE: feature.py ===" in full_context
        if len(afix_calls) == 1:  # lint fix (Phase 1)
            return [
                {"op": "edit_file", "path": "feature.py", "old": "import os\n", "new": ""},
                {"op": "write_file", "path": "fix_added.py", "content": "ADDED = 1\n"},
            ]
        # polish (Phase 2)
        return [{"op": "edit_file", "path": "feature.py", "old": "VALUE = 1", "new": "VALUE = 2"}]

    async def fake_areview(self: Any, **kwargs: Any) -> Any:
        return context_first.ConsistencyResult(
            has_issues=True, issues=["VALUE should be 2"], reasoning="r"
        )

    monkeypatch.setattr(context_first.FixSolver, "afix", fake_afix)
    monkeypatch.setattr(context_first.ConsistencyReviewer, "areview", fake_areview)

    agent = CoderAgent(MagicMock())
    rlm_result = RLMResult(answer="done", files_created=[str(b / "feature.py")])
    out = await agent._post_process(  # pyright: ignore[reportPrivateUsage]
        query="q", rlm_result=rlm_result, sub_task_id="s1", cwd=str(b)
    )

    assert len(afix_calls) == 2  # verification really failed -> fix ran; polish ran
    assert "F401" in afix_calls[0]
    assert (b / "feature.py").read_text() == "VALUE = 2\n"
    assert (b / "fix_added.py").read_text() == "ADDED = 1\n"
    # Main tree untouched, no new file there.
    assert {p.name: p.read_text() for p in a.iterdir()} == a_before
    assert str(b / "fix_added.py") in out.files_created
