"""Unit tests for the Context-First Solver.

Tests the mechanical components (gatherer, executor, manifest, parsing)
without requiring an LLM.  DSPy-dependent components (planner, solver)
are tested with mocks.
"""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from memfun_agent.context_first import (
    ContextFirstConfig,
    ContextFirstResult,
    ContextFirstSolver,
    ContextGatherer,
    OperationExecutor,
    Verifier,
    VerifyResult,
    _detect_verify_commands,
    _normalize_list,
    _parse_operations,
    build_file_manifest,
    manifest_to_string,
)

# ── _normalize_list tests ─────────────────────────────────────


def test_normalize_list_from_list():
    assert _normalize_list(["a", "b"]) == ["a", "b"]


def test_normalize_list_from_json_string():
    assert _normalize_list('["a", "b"]') == ["a", "b"]


def test_normalize_list_from_newline_string():
    assert _normalize_list("a\nb\nc") == ["a", "b", "c"]


def test_normalize_list_from_single_string():
    assert _normalize_list("hello") == ["hello"]


def test_normalize_list_from_empty():
    assert _normalize_list("") == []
    assert _normalize_list(None) == []
    assert _normalize_list(42) == []


# ── _parse_operations tests ───────────────────────────────────


def test_parse_operations_json_array():
    ops = _parse_operations(
        '[{"op":"write_file","path":"a.py","content":"x"}]'
    )
    assert len(ops) == 1
    assert ops[0]["op"] == "write_file"
    assert ops[0]["path"] == "a.py"


def test_parse_operations_markdown_block():
    raw = textwrap.dedent("""\
        Here are the changes:
        ```json
        [{"op": "edit_file", "path": "b.py", "old": "x", "new": "y"}]
        ```
    """)
    ops = _parse_operations(raw)
    assert len(ops) == 1
    assert ops[0]["op"] == "edit_file"


def test_parse_operations_line_by_line():
    raw = (
        '{"op":"write_file","path":"a.py","content":"x"}\n'
        '{"op":"run_cmd","cmd":"ls"}\n'
    )
    ops = _parse_operations(raw)
    assert len(ops) == 2


def test_parse_operations_empty():
    assert _parse_operations("[]") == []
    assert _parse_operations("") == []
    assert _parse_operations("no ops here") == []


def test_parse_operations_filters_invalid():
    raw = '[{"op":"write_file","path":"a.py","content":"x"}, {"not_an_op": true}]'
    ops = _parse_operations(raw)
    assert len(ops) == 1


# ── build_file_manifest tests ─────────────────────────────────


def test_build_manifest_basic(tmp_path: Path):
    """Builds manifest from a simple project directory."""
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "README.md").write_text("# Hello")
    (tmp_path / "data.bin").write_bytes(b"\x00" * 100)

    manifest = build_file_manifest(tmp_path)

    paths = [p for p, _ in manifest]
    assert "app.py" in paths
    assert "README.md" in paths
    # Binary file excluded (no matching extension)
    assert "data.bin" not in paths


def test_build_manifest_skips_pycache(tmp_path: Path):
    """Skips __pycache__ directories."""
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_bytes(
        b"\x00"
    )
    (tmp_path / "main.py").write_text("pass")

    manifest = build_file_manifest(tmp_path)
    paths = [p for p, _ in manifest]
    assert len(paths) == 1
    assert "main.py" in paths


def test_build_manifest_skips_node_modules(tmp_path: Path):
    """Skips node_modules directories."""
    nm = tmp_path / "node_modules" / "lodash"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")
    (tmp_path / "index.js").write_text("console.log('hi')")

    manifest = build_file_manifest(tmp_path)
    paths = [p for p, _ in manifest]
    assert len(paths) == 1
    assert "index.js" in paths


def test_build_manifest_nested(tmp_path: Path):
    """Handles nested directory structures."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("app code")
    (tmp_path / "src" / "utils.py").write_text("utils")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("tests")

    manifest = build_file_manifest(tmp_path)
    paths = [p for p, _ in manifest]
    assert len(paths) == 3
    assert "src/app.py" in paths or "src\\app.py" in paths


def test_manifest_to_string():
    manifest = [("app.py", 100), ("utils.py", 200)]
    result = manifest_to_string(manifest)
    assert "app.py (100 bytes)" in result
    assert "utils.py (200 bytes)" in result


# ── ContextGatherer tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_gatherer_reads_files(tmp_path: Path):
    """Reads files and assembles context with headers."""
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def helper(): pass")

    gatherer = ContextGatherer()
    ctx = await gatherer.agather(
        files=["app.py", "utils.py"],
        project_root=tmp_path,
    )

    assert "=== FILE: app.py ===" in ctx
    assert "print('hello')" in ctx
    assert "=== FILE: utils.py ===" in ctx
    assert "def helper(): pass" in ctx


@pytest.mark.asyncio
async def test_gatherer_respects_budget(tmp_path: Path):
    """Stops reading files when max_bytes is reached."""
    (tmp_path / "big.py").write_text("x" * 1000)
    (tmp_path / "small.py").write_text("y" * 100)

    gatherer = ContextGatherer(max_bytes=500)
    ctx = await gatherer.agather(
        files=["big.py", "small.py"],
        project_root=tmp_path,
    )

    # big.py should be truncated; small.py may or may not be included
    assert "=== FILE: big.py ===" in ctx
    assert "truncated" in ctx


@pytest.mark.asyncio
async def test_gatherer_skips_missing_files(tmp_path: Path):
    """Skips files that don't exist."""
    (tmp_path / "exists.py").write_text("yes")

    gatherer = ContextGatherer()
    ctx = await gatherer.agather(
        files=["exists.py", "nope.py"],
        project_root=tmp_path,
    )

    assert "=== FILE: exists.py ===" in ctx
    assert "nope.py" not in ctx


def test_gatherer_read_all_files(tmp_path: Path):
    """read_all_files reads everything from manifest."""
    (tmp_path / "a.py").write_text("aaa")
    (tmp_path / "b.py").write_text("bbb")

    gatherer = ContextGatherer()
    manifest = [("a.py", 3), ("b.py", 3)]
    ctx = gatherer.read_all_files(manifest, tmp_path)

    assert "=== FILE: a.py ===" in ctx
    assert "aaa" in ctx
    assert "=== FILE: b.py ===" in ctx
    assert "bbb" in ctx


# ── OperationExecutor tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_executor_write_file(tmp_path: Path):
    """Executes write_file operations."""
    executor = OperationExecutor(tmp_path)
    await executor.execute([
        {
            "op": "write_file",
            "path": str(tmp_path / "new.py"),
            "content": "print('new')",
        }
    ])

    assert (tmp_path / "new.py").read_text() == "print('new')"
    assert len(executor.ops) == 1
    assert executor.ops[0][0] == "write"
    assert len(executor.files_created) == 1


@pytest.mark.asyncio
async def test_executor_edit_file(tmp_path: Path):
    """Executes edit_file operations."""
    target = tmp_path / "edit_me.py"
    target.write_text("old_value = 1")

    executor = OperationExecutor(tmp_path)
    await executor.execute([
        {
            "op": "edit_file",
            "path": str(target),
            "old": "old_value = 1",
            "new": "new_value = 2",
        }
    ])

    assert target.read_text() == "new_value = 2"
    assert len(executor.ops) == 1
    assert executor.ops[0][0] == "edit"


@pytest.mark.asyncio
async def test_executor_edit_file_missing_text(tmp_path: Path):
    """edit_file gracefully handles missing old text."""
    target = tmp_path / "no_match.py"
    target.write_text("something else")

    executor = OperationExecutor(tmp_path)
    await executor.execute([
        {
            "op": "edit_file",
            "path": str(target),
            "old": "not_found",
            "new": "replacement",
        }
    ])

    # File unchanged, no ops recorded, failure tracked.
    assert target.read_text() == "something else"
    assert len(executor.ops) == 0
    assert executor.attempted == 1
    assert executor.failed == 1


@pytest.mark.asyncio
async def test_executor_run_cmd(tmp_path: Path):
    """Executes run_cmd operations."""
    executor = OperationExecutor(tmp_path)
    await executor.execute([
        {"op": "run_cmd", "cmd": "echo hello"}
    ])

    assert len(executor.ops) == 1
    assert executor.ops[0][0] == "cmd"
    assert executor.ops[0][1] == "echo hello"
    assert executor.ops[0][2] == 0  # returncode


@pytest.mark.asyncio
async def test_executor_unknown_op(tmp_path: Path):
    """Unknown operations are skipped."""
    executor = OperationExecutor(tmp_path)
    await executor.execute([
        {"op": "unknown_op", "data": "test"}
    ])
    assert len(executor.ops) == 0


# ── ContextFirstResult tests ─────────────────────────────────


def test_result_to_dict():
    """to_result_dict produces the expected dict shape."""
    result = ContextFirstResult(
        answer="Fixed the bug",
        reasoning="The issue was X",
        ops=[("write", "/path/to/file.py", 100)],
        files_created=["/path/to/file.py"],
        success=True,
        method="context_first_fast",
        total_tokens=5000,
    )

    d = result.to_result_dict()
    assert d["answer"] == "Fixed the bug"
    assert d["method"] == "context_first_fast"
    assert d["iterations"] == 1
    assert d["total_tokens"] == 5000
    assert d["files_created"] == ["/path/to/file.py"]
    assert d["ops"] == [
        {"type": "write", "target": "/path/to/file.py", "detail": 100}
    ]


def test_result_to_dict_empty_ops():
    """to_result_dict handles empty ops."""
    result = ContextFirstResult(
        answer="No changes needed",
        reasoning="Everything is fine",
        ops=[],
        files_created=[],
        success=True,
        method="context_first_planned",
    )

    d = result.to_result_dict()
    assert d["ops"] == []
    assert d["files_created"] == []


# ── ContextFirstConfig tests ─────────────────────────────────


def test_config_defaults():
    """Default config values are reasonable."""
    config = ContextFirstConfig()
    assert config.max_context_bytes == 200_000
    assert config.max_gather_bytes == 400_000
    assert config.max_files == 50
    assert config.enable_planner is True
    assert config.verify_commands == ()
    assert config.max_fix_attempts == 2


def test_config_custom():
    """Custom config values are accepted."""
    config = ContextFirstConfig(
        max_context_bytes=500_000,
        max_gather_bytes=800_000,
        max_files=100,
        enable_planner=False,
        verify_commands=("ruff check .", "pytest -x"),
        max_fix_attempts=3,
    )
    assert config.max_context_bytes == 500_000
    assert config.enable_planner is False
    assert config.verify_commands == ("ruff check .", "pytest -x")
    assert config.max_fix_attempts == 3


# ── ContextFirstSolver integration tests ──────────────────────


async def _mock_asolve(query, full_context):
    """Helper: return a mock SolveResult."""
    from memfun_agent.context_first import SolveResult

    return SolveResult(
        reasoning="Simple fix",
        answer="Fixed it",
        operations=[],
    )


@pytest.mark.asyncio
async def test_solver_fast_path(tmp_path: Path):
    """Small project triggers fast path (no planner call)."""
    (tmp_path / "app.py").write_text("print('hello world')")
    (tmp_path / "utils.py").write_text("def add(a, b): return a + b")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    # Mock the SingleShotSolver.asolve to bypass DSPy LM requirement.
    with patch.object(
        solver.solver, "asolve", side_effect=_mock_asolve
    ):
        result = await solver.asolve(
            query="Fix the bug",
            context="=== CURRENT PROJECT STATE ===",
        )

    assert result.success is True
    assert result.method == "context_first_fast"
    assert result.answer == "Fixed it"


@pytest.mark.asyncio
async def test_solver_empty_project(tmp_path: Path):
    """Empty project returns unsuccessful result."""
    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(),
    )

    result = await solver.asolve(query="Fix something")

    assert result.success is False
    assert "No source files" in result.reasoning


@pytest.mark.asyncio
async def test_solver_exception_returns_failure(tmp_path: Path):
    """Exception during solving returns failure result."""
    (tmp_path / "app.py").write_text("code")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    async def _raise(*_a, **_kw):
        raise RuntimeError("LLM down")

    with patch.object(
        solver.solver, "asolve", side_effect=_raise
    ):
        result = await solver.asolve(query="Fix bug")

    assert result.success is False
    assert "LLM down" in result.reasoning


@pytest.mark.asyncio
async def test_solver_with_operations(tmp_path: Path):
    """Solver executes operations from LLM output."""
    (tmp_path / "app.py").write_text("old_code()")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    target_path = str(tmp_path / "app.py")

    async def _mock_solve_with_ops(query, full_context):
        from memfun_agent.context_first import SolveResult

        return SolveResult(
            reasoning="Replaced old_code with new_code",
            answer="Fixed the function call",
            operations=[{
                "op": "edit_file",
                "path": target_path,
                "old": "old_code()",
                "new": "new_code()",
            }],
        )

    with patch.object(
        solver.solver, "asolve", side_effect=_mock_solve_with_ops
    ):
        result = await solver.asolve(query="Fix the function")

    assert result.success is True
    assert len(result.ops) == 1
    assert result.ops[0][0] == "edit"
    # Verify the file was actually edited.
    assert (tmp_path / "app.py").read_text() == "new_code()"


@pytest.mark.asyncio
async def test_solver_status_callback(tmp_path: Path):
    """on_status callback is called during solving."""
    (tmp_path / "app.py").write_text("code")

    statuses: list[str] = []

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
        on_status=statuses.append,
    )

    with patch.object(
        solver.solver, "asolve", side_effect=_mock_asolve
    ):
        await solver.asolve(query="Do something")

    assert len(statuses) >= 2  # "Scanning...", "Reading...", "Solving..."
    assert any("Scanning" in s for s in statuses)
    assert any("Solving" in s for s in statuses)


# ── Verifier tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verifier_passing_command(tmp_path: Path):
    """Verifier returns passed=True when command succeeds."""
    verifier = Verifier(tmp_path)
    result = await verifier.averify(["echo ok"])

    assert result.passed is True
    assert result.errors == ""
    assert result.commands_run == ["echo ok"]


@pytest.mark.asyncio
async def test_verifier_failing_command(tmp_path: Path):
    """Verifier returns passed=False with error output."""
    verifier = Verifier(tmp_path)
    result = await verifier.averify(["false"])

    assert result.passed is False
    assert "exit 1" in result.errors
    assert result.commands_run == ["false"]


@pytest.mark.asyncio
async def test_verifier_multiple_commands(tmp_path: Path):
    """Verifier runs all commands and collects errors."""
    verifier = Verifier(tmp_path)
    result = await verifier.averify(["echo ok", "false", "echo done"])

    assert result.passed is False
    assert len(result.commands_run) == 3


@pytest.mark.asyncio
async def test_verifier_no_commands(tmp_path: Path):
    """Verifier with no commands passes."""
    verifier = Verifier(tmp_path)
    result = await verifier.averify([])

    assert result.passed is True


# ── _detect_verify_commands tests ─────────────────────────────


def test_detect_verify_python(tmp_path: Path):
    """Detects ruff for Python projects."""
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]")
    cmds = _detect_verify_commands(tmp_path)
    assert "ruff check ." in cmds


def test_detect_verify_empty(tmp_path: Path):
    """Returns empty for unknown projects."""
    cmds = _detect_verify_commands(tmp_path)
    assert cmds == []


def test_detect_verify_go(tmp_path: Path):
    """Detects go vet for Go projects."""
    (tmp_path / "go.mod").write_text("module example.com/foo")
    cmds = _detect_verify_commands(tmp_path)
    assert "go vet ./..." in cmds


def test_detect_verify_rust(tmp_path: Path):
    """Detects cargo check for Rust projects."""
    (tmp_path / "Cargo.toml").write_text("[package]")
    cmds = _detect_verify_commands(tmp_path)
    assert "cargo check" in cmds


# ── Verification loop integration tests ───────────────────────


@pytest.mark.asyncio
async def test_solver_verify_passes(tmp_path: Path):
    """When verification passes, no fix attempt is made."""
    (tmp_path / "app.py").write_text("print('hello')")

    statuses: list[str] = []

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            verify_commands=("echo ok",),
        ),
        on_status=statuses.append,
    )

    async def _mock_solve_with_write(query, full_context):
        from memfun_agent.context_first import SolveResult

        return SolveResult(
            reasoning="Added greeting",
            answer="Created file",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "print('done')",
            }],
        )

    with patch.object(
        solver.solver, "asolve", side_effect=_mock_solve_with_write
    ):
        result = await solver.asolve(query="Create a file")

    assert result.success is True
    assert any("Verification passed" in s for s in statuses)


@pytest.mark.asyncio
async def test_solver_verify_fails_and_fixes(tmp_path: Path):
    """When verification fails, fix solver is called and fixes applied."""
    (tmp_path / "app.py").write_text("x = 1")

    fix_call_count = 0

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            # First verify fails (exit 1), second passes (exit 0)
            verify_commands=("test -f " + str(tmp_path / "fixed.txt"),),
            max_fix_attempts=2,
        ),
    )

    async def _mock_solve(query, full_context):
        from memfun_agent.context_first import SolveResult

        return SolveResult(
            reasoning="Wrote file",
            answer="Created output",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "print('out')",
            }],
        )

    async def _mock_fix(query, full_context, verification_errors):
        nonlocal fix_call_count
        fix_call_count += 1
        # The fix creates the missing file so next verify passes
        return [{
            "op": "write_file",
            "path": str(tmp_path / "fixed.txt"),
            "content": "fixed",
        }]

    with (
        patch.object(
            solver.solver, "asolve", side_effect=_mock_solve
        ),
        patch.object(
            solver.fix_solver, "afix", side_effect=_mock_fix
        ),
    ):
        result = await solver.asolve(query="Create output")

    assert result.success is True
    assert fix_call_count == 1
    assert (tmp_path / "fixed.txt").read_text() == "fixed"


@pytest.mark.asyncio
async def test_solver_verify_max_attempts_exceeded(tmp_path: Path):
    """When fix attempts are exhausted, solver still returns success."""
    (tmp_path / "app.py").write_text("code")

    fix_call_count = 0

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            verify_commands=("false",),  # always fails
            max_fix_attempts=2,
        ),
    )

    async def _mock_solve(query, full_context):
        from memfun_agent.context_first import SolveResult

        return SolveResult(
            reasoning="Done",
            answer="Done",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "out",
            }],
        )

    async def _mock_fix(query, full_context, verification_errors):
        nonlocal fix_call_count
        fix_call_count += 1
        return []  # No fix ops

    with (
        patch.object(
            solver.solver, "asolve", side_effect=_mock_solve
        ),
        patch.object(
            solver.fix_solver, "afix", side_effect=_mock_fix
        ),
    ):
        result = await solver.asolve(query="Create output")

    # Result is still success (verification is best-effort)
    assert result.success is True
    # Fix was attempted once (first failure), then max reached on second
    assert fix_call_count == 1


@pytest.mark.asyncio
async def test_solver_no_verify_when_no_files_created(tmp_path: Path):
    """Verification is skipped when no files were created."""
    (tmp_path / "app.py").write_text("code")

    statuses: list[str] = []

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            verify_commands=("false",),  # would fail
        ),
        on_status=statuses.append,
    )

    # Return no operations (answer-only query).
    with patch.object(
        solver.solver, "asolve", side_effect=_mock_asolve
    ):
        result = await solver.asolve(query="What does this code do?")

    assert result.success is True
    # No verification should have run.
    assert not any("Verify" in s for s in statuses)


# ── VerifyResult tests ────────────────────────────────────────


def test_verify_result_passed():
    vr = VerifyResult(passed=True, errors="", commands_run=["echo ok"])
    assert vr.passed is True


def test_verify_result_failed():
    vr = VerifyResult(
        passed=False,
        errors="error: unused import",
        commands_run=["ruff check ."],
    )
    assert vr.passed is False
    assert "unused import" in vr.errors
