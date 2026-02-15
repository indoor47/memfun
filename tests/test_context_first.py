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
    ConsistencyResult,
    ContextFirstConfig,
    ContextFirstResult,
    ContextFirstSolver,
    ContextGatherer,
    EditDiagnostic,
    OperationExecutor,
    PlanResult,
    SolveResult,
    Verifier,
    VerifyResult,
    _detect_verify_commands,
    _fuzzy_find_and_replace,
    _normalize_list,
    _normalize_whitespace,
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


# ── ConsistencyResult tests ──────────────────────────────────


def test_consistency_result_no_issues():
    """ConsistencyResult with no issues."""
    cr = ConsistencyResult(
        has_issues=False,
        issues=[],
        reasoning="Everything looks correct",
    )
    assert cr.has_issues is False
    assert cr.issues == []
    assert "correct" in cr.reasoning


def test_consistency_result_with_issues():
    """ConsistencyResult with issues."""
    cr = ConsistencyResult(
        has_issues=True,
        issues=[
            "onClick handler references #delete-btn but HTML has no such id",
            "CSS class .modal-overlay defined but never used in HTML",
        ],
        reasoning="Cross-file reference mismatches found",
    )
    assert cr.has_issues is True
    assert len(cr.issues) == 2
    assert "delete-btn" in cr.issues[0]


# ── Config consistency review flag tests ─────────────────────


def test_config_enable_consistency_review_default():
    """Consistency review is enabled by default."""
    config = ContextFirstConfig()
    assert config.enable_consistency_review is True


def test_config_disable_consistency_review():
    """Consistency review can be disabled."""
    config = ContextFirstConfig(enable_consistency_review=False)
    assert config.enable_consistency_review is False


# ── _build_intended_changes tests ────────────────────────────


def test_build_intended_changes_write():
    """Build intended changes summary for write operations."""
    sr = SolveResult(
        reasoning="Created new file",
        answer="Added greeting module",
        operations=[{
            "op": "write_file",
            "path": "hello.py",
            "content": "print('hello world')",
        }],
    )
    text = ContextFirstSolver._build_intended_changes(sr)
    assert "Added greeting module" in text
    assert "write_file hello.py" in text
    assert "20 chars" in text


def test_build_intended_changes_edit():
    """Build intended changes summary for edit operations."""
    sr = SolveResult(
        reasoning="Fixed bug",
        answer="Replaced old function call",
        operations=[{
            "op": "edit_file",
            "path": "app.py",
            "old": "old_func()",
            "new": "new_func()",
        }],
    )
    text = ContextFirstSolver._build_intended_changes(sr)
    assert "edit_file app.py" in text
    assert "old_func()" in text


def test_build_intended_changes_no_ops():
    """Build intended changes with no operations."""
    sr = SolveResult(
        reasoning="Just an explanation",
        answer="The code does X",
        operations=[],
    )
    text = ContextFirstSolver._build_intended_changes(sr)
    assert "The code does X" in text


def test_build_intended_changes_empty():
    """Build intended changes with empty answer and no ops."""
    sr = SolveResult(reasoning="", answer="", operations=[])
    text = ContextFirstSolver._build_intended_changes(sr)
    assert text == "(no changes intended)"


# ── Consistency review integration tests ─────────────────────


@pytest.mark.asyncio
async def test_solver_consistency_review_skipped_no_ops(tmp_path: Path):
    """Consistency review is skipped when no files were created."""
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
        result = await solver.asolve(query="Explain this code")

    assert result.success is True
    assert not any("consistency" in s.lower() for s in statuses)


@pytest.mark.asyncio
async def test_solver_consistency_review_passes(tmp_path: Path):
    """When consistency review finds no issues, no polish is done."""
    (tmp_path / "app.py").write_text("code")

    statuses: list[str] = []
    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
        on_status=statuses.append,
    )

    async def _mock_solve_write(query, full_context):
        return SolveResult(
            reasoning="Wrote file",
            answer="Created output",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "print('done')",
            }],
        )

    async def _mock_review(user_request, intended_changes, actual_file_contents):
        return ConsistencyResult(
            has_issues=False, issues=[], reasoning="All good"
        )

    with (
        patch.object(solver.solver, "asolve", side_effect=_mock_solve_write),
        patch.object(
            solver.consistency_reviewer, "areview", side_effect=_mock_review
        ),
    ):
        result = await solver.asolve(query="Create output")

    assert result.success is True
    assert any("Consistency check passed" in s for s in statuses)


@pytest.mark.asyncio
async def test_solver_consistency_review_triggers_polish(tmp_path: Path):
    """When consistency review finds issues, polish operations are applied."""
    (tmp_path / "app.py").write_text("code")

    polish_called = False

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    async def _mock_solve_write(query, full_context):
        return SolveResult(
            reasoning="Wrote file",
            answer="Created output",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "incomplete",
            }],
        )

    async def _mock_review(user_request, intended_changes, actual_file_contents):
        return ConsistencyResult(
            has_issues=True,
            issues=["out.py has placeholder content instead of real implementation"],
            reasoning="The file content is incomplete",
        )

    async def _mock_fix(query, full_context, verification_errors):
        nonlocal polish_called
        polish_called = True
        # Polish runs in edit_only mode, so must use edit_file (not write_file).
        return [{
            "op": "edit_file",
            "path": str(tmp_path / "out.py"),
            "old": "incomplete",
            "new": "complete_implementation()",
        }]

    with (
        patch.object(solver.solver, "asolve", side_effect=_mock_solve_write),
        patch.object(
            solver.consistency_reviewer, "areview", side_effect=_mock_review
        ),
        patch.object(solver.fix_solver, "afix", side_effect=_mock_fix),
    ):
        result = await solver.asolve(query="Create output")

    assert result.success is True
    assert polish_called is True
    assert (tmp_path / "out.py").read_text() == "complete_implementation()"


@pytest.mark.asyncio
async def test_solver_consistency_review_exception_handled(tmp_path: Path):
    """Consistency review exceptions are caught gracefully."""
    (tmp_path / "app.py").write_text("code")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    async def _mock_solve_write(query, full_context):
        return SolveResult(
            reasoning="Wrote file",
            answer="Created output",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "good code",
            }],
        )

    async def _mock_review_explode(
        user_request, intended_changes, actual_file_contents
    ):
        raise RuntimeError("LLM timeout")

    with (
        patch.object(solver.solver, "asolve", side_effect=_mock_solve_write),
        patch.object(
            solver.consistency_reviewer,
            "areview",
            side_effect=_mock_review_explode,
        ),
    ):
        result = await solver.asolve(query="Create output")

    # Pipeline continues despite review failure.
    assert result.success is True
    assert (tmp_path / "out.py").read_text() == "good code"


@pytest.mark.asyncio
async def test_solver_consistency_review_disabled(tmp_path: Path):
    """Consistency review is skipped when disabled in config."""
    (tmp_path / "app.py").write_text("code")

    statuses: list[str] = []
    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            enable_consistency_review=False,
        ),
        on_status=statuses.append,
    )

    async def _mock_solve_write(query, full_context):
        return SolveResult(
            reasoning="Wrote file",
            answer="Created output",
            operations=[{
                "op": "write_file",
                "path": str(tmp_path / "out.py"),
                "content": "print('done')",
            }],
        )

    with patch.object(
        solver.solver, "asolve", side_effect=_mock_solve_write
    ):
        result = await solver.asolve(query="Create output")

    assert result.success is True
    # No consistency review status messages (ignore file paths that
    # may contain the test name).
    assert not any(
        s.startswith("Reviewing consistency") for s in statuses
    )


# ── Fuzzy edit matching tests ─────────────────────────────────


def test_fuzzy_exact_match():
    """Exact match still works and is preferred."""
    content = "def foo():\n    return 1\n"
    old = "return 1"
    new = "return 2"
    result, ratio, _snippet, strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is not None
    assert "return 2" in result
    assert "return 1" not in result
    assert ratio == 1.0
    assert strategy == "exact"


def test_fuzzy_trailing_whitespace():
    """Whitespace normalization matches trailing spaces."""
    content = "def foo():  \n    return 1  \n"
    old = "def foo():\n    return 1\n"
    new = "def bar():\n    return 2\n"
    result, _ratio, _snippet, strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is not None
    assert "def bar():" in result
    assert "return 2" in result
    assert strategy == "whitespace"


def test_fuzzy_line_matching():
    """Line-based fuzzy matching with minor differences."""
    content = "class Foo:\n    def bar(self):\n        return 42\n"
    # LLM produced slightly different text (extra space before colon)
    old = "class Foo :\n    def bar(self) :\n        return 42\n"
    new = "class Baz:\n    def bar(self):\n        return 99\n"
    result, ratio, _snippet, strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is not None
    assert "class Baz:" in result
    assert "return 99" in result
    assert strategy == "fuzzy"
    assert ratio >= 0.80


def test_fuzzy_no_match_below_threshold():
    """Returns None when the best match is below the threshold."""
    content = "completely different content here\n"
    old = "nothing like this at all and also much longer text to ensure low ratio\n"
    new = "replacement\n"
    result, _ratio, _snippet, strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is None
    assert strategy == "none"


def test_fuzzy_indentation_preservation():
    """Preserves the file's actual indentation in exact match."""
    content = "    if True:\n        x = 1\n"
    old = "    if True:\n        x = 1\n"
    new = "    if True:\n        x = 2\n"
    result, _ratio, _snippet, strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is not None
    assert result == "    if True:\n        x = 2\n"
    assert strategy == "exact"


def test_fuzzy_window_off_by_one():
    """LLM includes one extra line — fuzzy window catches it."""
    content = "line1\nline2\nline3\nline4\n"
    # LLM produced 3 lines but actual target is only 2 lines
    old = "line2\nline3\nline4"
    new = "lineA\nlineB"
    result, _ratio, _snippet, _strategy = _fuzzy_find_and_replace(
        content, old, new
    )
    assert result is not None
    assert "lineA" in result
    assert "lineB" in result


def test_normalize_whitespace_strips_trailing():
    """_normalize_whitespace strips trailing whitespace per line."""
    text = "hello   \nworld\t\n  foo  \n"
    result = _normalize_whitespace(text)
    assert result == "hello\nworld\n  foo\n"


# ── Executor fuzzy edit tests ────────────────────────────────


@pytest.mark.asyncio
async def test_executor_edit_fuzzy_whitespace(tmp_path: Path):
    """Executor applies fuzzy whitespace-normalized match."""
    target = tmp_path / "ws.py"
    target.write_text("def foo():  \n    return 1  \n")

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "edit_file",
        "path": str(target),
        "old": "def foo():\n    return 1\n",
        "new": "def bar():\n    return 2\n",
    }])

    content = target.read_text()
    assert "def bar():" in content
    assert "return 2" in content
    assert executor.failed == 0
    assert len(executor.ops) == 1


@pytest.mark.asyncio
async def test_executor_edit_collects_diagnostics(tmp_path: Path):
    """Failed edits create EditDiagnostic entries."""
    target = tmp_path / "diag.py"
    target.write_text("totally different content\n")

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "edit_file",
        "path": str(target),
        "old": "nothing like the file content at all and much longer\n",
        "new": "replacement\n",
    }])

    assert executor.failed == 1
    assert len(executor.edit_diagnostics) == 1
    diag = executor.edit_diagnostics[0]
    assert diag.path == str(target)
    assert "nothing like" in diag.old_text
    assert diag.strategy_tried == "none"
    assert "totally different" in diag.file_excerpt


# ── EditDiagnostic dataclass tests ───────────────────────────


def test_edit_diagnostic_fields():
    """EditDiagnostic holds all expected fields."""
    diag = EditDiagnostic(
        path="/tmp/test.py",
        old_text="old code",
        new_text="new code",
        best_match_ratio=0.75,
        best_match_snippet="old cde",
        strategy_tried="none",
        file_excerpt="full file content here",
    )
    assert diag.path == "/tmp/test.py"
    assert diag.old_text == "old code"
    assert diag.new_text == "new code"
    assert diag.best_match_ratio == 0.75
    assert diag.best_match_snippet == "old cde"
    assert diag.strategy_tried == "none"
    assert diag.file_excerpt == "full file content here"


# ── Config enable_edit_retry tests ───────────────────────────


def test_config_enable_edit_retry_default():
    """enable_edit_retry defaults to True."""
    config = ContextFirstConfig()
    assert config.enable_edit_retry is True


def test_config_disable_edit_retry():
    """enable_edit_retry can be disabled."""
    config = ContextFirstConfig(enable_edit_retry=False)
    assert config.enable_edit_retry is False


# ── Retry pipeline tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_solver_retries_failed_edits(tmp_path: Path):
    """Retry pipeline feeds diagnostics to FixSolver and recovers."""
    target = tmp_path / "app.py"
    target.write_text("actual_code_here()")

    fix_called = False

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),
    )

    target_path = str(target)

    async def _mock_solve(query, full_context):
        return SolveResult(
            reasoning="Edit the code",
            answer="Fixed it",
            operations=[{
                "op": "edit_file",
                "path": target_path,
                # Deliberately wrong old text to trigger diagnostic.
                "old": "wrong_old_text_not_in_file_at_all_completely_different",
                "new": "new_code()",
            }],
        )

    async def _mock_fix(query, full_context, verification_errors):
        nonlocal fix_called
        fix_called = True
        assert "FAILED EDIT" in verification_errors
        assert "wrong_old_text" in verification_errors
        # Return corrected operation with actual content.
        return [{
            "op": "edit_file",
            "path": target_path,
            "old": "actual_code_here()",
            "new": "new_code()",
        }]

    with (
        patch.object(solver.solver, "asolve", side_effect=_mock_solve),
        patch.object(solver.fix_solver, "afix", side_effect=_mock_fix),
    ):
        result = await solver.asolve(query="Fix the code")

    assert fix_called is True
    assert result.success is True
    assert target.read_text() == "new_code()"


@pytest.mark.asyncio
async def test_solver_retry_disabled(tmp_path: Path):
    """When enable_edit_retry=False, no retry is attempted."""
    target = tmp_path / "app.py"
    target.write_text("actual_code_here()")

    fix_called = False

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(
            max_context_bytes=10_000,
            enable_edit_retry=False,
        ),
    )

    target_path = str(target)

    async def _mock_solve(query, full_context):
        return SolveResult(
            reasoning="Edit the code",
            answer="Fixed it",
            operations=[{
                "op": "edit_file",
                "path": target_path,
                "old": "wrong_old_text_not_found_anywhere_at_all",
                "new": "new_code()",
            }],
        )

    async def _mock_fix(query, full_context, verification_errors):
        nonlocal fix_called
        fix_called = True
        return []

    with (
        patch.object(solver.solver, "asolve", side_effect=_mock_solve),
        patch.object(solver.fix_solver, "afix", side_effect=_mock_fix),
    ):
        result = await solver.asolve(query="Fix the code")

    # Retry was NOT called because it's disabled.
    assert fix_called is False
    # All edits failed -> success is False.
    assert result.success is False


# ── Write guard tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_guard_blocks_destructive_overwrite(
    tmp_path: Path,
) -> None:
    """write_file on an existing file with >30% line loss is blocked."""
    target = tmp_path / "big.js"
    original = "\n".join(f"line {i}" for i in range(100))
    target.write_text(original, "utf-8")

    # New content is only 50 lines (50% lost).
    new_content = "\n".join(f"new line {i}" for i in range(50))

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "write_file",
        "path": str(target),
        "content": new_content,
    }])

    assert executor.failed == 1
    # File should NOT have been overwritten.
    assert target.read_text("utf-8") == original
    assert len(executor.edit_diagnostics) == 1
    diag = executor.edit_diagnostics[0]
    assert diag.strategy_tried == "write_file_guard"


@pytest.mark.asyncio
async def test_write_guard_allows_small_change(
    tmp_path: Path,
) -> None:
    """write_file on existing file with <30% line loss is allowed."""
    target = tmp_path / "small_change.js"
    original = "\n".join(f"line {i}" for i in range(100))
    target.write_text(original, "utf-8")

    # New content is 80 lines (only 20% lost — under threshold).
    new_content = "\n".join(f"updated line {i}" for i in range(80))

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "write_file",
        "path": str(target),
        "content": new_content,
    }])

    assert executor.failed == 0
    assert target.read_text("utf-8") == new_content


@pytest.mark.asyncio
async def test_write_guard_allows_new_file(
    tmp_path: Path,
) -> None:
    """write_file on a non-existent file always succeeds."""
    target = tmp_path / "brand_new.js"
    content = "\n".join(f"line {i}" for i in range(50))

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "write_file",
        "path": str(target),
        "content": content,
    }])

    assert executor.failed == 0
    assert target.read_text("utf-8") == content
    assert len(executor.edit_diagnostics) == 0


@pytest.mark.asyncio
async def test_write_guard_creates_diagnostic(
    tmp_path: Path,
) -> None:
    """Blocked write produces EditDiagnostic with correct fields."""
    target = tmp_path / "app.js"
    original = "\n".join(f"function f{i}() {{}}" for i in range(200))
    target.write_text(original, "utf-8")

    # Rewrite attempt with only 50 lines (75% lost).
    new_content = "\n".join(f"new f{i}()" for i in range(50))

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "write_file",
        "path": str(target),
        "content": new_content,
    }])

    assert len(executor.edit_diagnostics) == 1
    diag = executor.edit_diagnostics[0]
    assert diag.path == str(target)
    assert "200 lines" in diag.old_text
    assert "50 lines" in diag.new_text
    assert diag.strategy_tried == "write_file_guard"
    assert diag.best_match_ratio == pytest.approx(0.25)
    assert len(diag.file_excerpt) <= 2000


@pytest.mark.asyncio
async def test_write_guard_small_file_exempt(
    tmp_path: Path,
) -> None:
    """Files with <=10 lines are exempt from the guard."""
    target = tmp_path / "tiny.js"
    # Original is 10 lines.
    original = "\n".join(f"line {i}" for i in range(10))
    target.write_text(original, "utf-8")

    # Rewrite with only 3 lines (70% lost — but file is tiny).
    new_content = "a\nb\nc"

    executor = OperationExecutor(tmp_path)
    await executor.execute([{
        "op": "write_file",
        "path": str(target),
        "content": new_content,
    }])

    # Should be allowed because old_lines <= 10.
    assert executor.failed == 0
    assert target.read_text("utf-8") == new_content


# ── Web search integration tests ──────────────────────────────


def test_plan_result_web_searches():
    """PlanResult stores web_searches field."""
    pr = PlanResult(
        files_to_read=["app.py"],
        search_patterns=[],
        reasoning="Need web info",
        web_searches=["python httpx timeout", "FastAPI docs"],
    )
    assert pr.web_searches == ["python httpx timeout", "FastAPI docs"]


def test_plan_result_web_searches_default():
    """PlanResult web_searches defaults to empty list."""
    pr = PlanResult(
        files_to_read=["app.py"],
        search_patterns=[],
        reasoning="No web needed",
    )
    assert pr.web_searches == []


@pytest.mark.asyncio
async def test_context_planner_extracts_web_searches(tmp_path: Path):
    """Planner extracts web_searches from DSPy output."""
    from memfun_agent.context_first import ContextPlanner

    planner = ContextPlanner()

    class MockPrediction:
        files_to_read = ["app.py"]  # noqa: RUF012
        search_patterns = []  # noqa: RUF012
        web_searches = '["httpx timeout docs", "python async http"]'
        reasoning = "Need external docs"

    # Bypass asyncio.to_thread so mock __call__ works in-thread.
    async def _fake_to_thread(fn, *args, **kwargs):
        return MockPrediction()

    with patch("memfun_agent.context_first.asyncio.to_thread", side_effect=_fake_to_thread):
        result = await planner.aplan(
            query="How to set httpx timeout?",
            file_manifest="app.py (100 bytes)",
            project_summary="Python project",
        )

    assert result.web_searches == ["httpx timeout docs", "python async http"]


@pytest.mark.asyncio
async def test_context_gatherer_web_search(tmp_path: Path):
    """ContextGatherer runs web searches and adds to context."""
    (tmp_path / "app.py").write_text("code")

    gatherer = ContextGatherer()

    with patch.object(
        gatherer, "_web_search", return_value="**Result Title**\nhttps://example.com\nSome snippet"
    ) as mock_search:
        ctx = await gatherer.agather(
            files=["app.py"],
            project_root=tmp_path,
            web_searches=["test query"],
        )

    mock_search.assert_called_once_with("test query")
    assert "=== WEB SEARCH RESULTS ===" in ctx
    assert "### Search: test query" in ctx
    assert "Result Title" in ctx


@pytest.mark.asyncio
async def test_context_gatherer_web_search_cap(tmp_path: Path):
    """Max 5 web searches enforced."""
    (tmp_path / "app.py").write_text("code")

    call_count = 0

    async def _mock_search(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result for {query}"

    gatherer = ContextGatherer()

    with patch.object(gatherer, "_web_search", side_effect=_mock_search):
        queries = [f"query {i}" for i in range(10)]
        await gatherer.agather(
            files=["app.py"],
            project_root=tmp_path,
            web_searches=queries,
        )

    assert call_count == 5  # Capped at 5


@pytest.mark.asyncio
async def test_context_gatherer_no_web_searches(tmp_path: Path):
    """Empty web_searches list produces no web calls."""
    (tmp_path / "app.py").write_text("code")

    gatherer = ContextGatherer()

    with patch.object(gatherer, "_web_search") as mock_search:
        ctx = await gatherer.agather(
            files=["app.py"],
            project_root=tmp_path,
            web_searches=[],
        )

    mock_search.assert_not_called()
    assert "WEB SEARCH" not in ctx


@pytest.mark.asyncio
async def test_web_search_import_error():
    """Graceful fallback when ddgs not installed."""
    gatherer = ContextGatherer()

    with patch.dict("sys.modules", {"ddgs": None}):
        # _web_search catches ImportError internally
        result = await gatherer._web_search("test query")

    assert "unavailable" in result or "error" in result


@pytest.mark.asyncio
async def test_web_context_in_solver(tmp_path: Path):
    """Web search results appear in solver context via planned path."""
    (tmp_path / "app.py").write_text("print('hello')")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=1),  # Force planner path
    )

    plan = PlanResult(
        files_to_read=["app.py"],
        search_patterns=[],
        reasoning="Need web info",
        web_searches=["test web query"],
    )

    captured_context = {}

    async def _mock_solve(query, full_context):
        captured_context["full"] = full_context
        return SolveResult(
            reasoning="Done",
            answer="Answer with web info",
            operations=[],
        )

    async def _mock_plan(query, file_manifest, project_summary):
        return plan

    with (
        patch.object(solver.planner, "aplan", side_effect=_mock_plan),
        patch.object(solver.solver, "asolve", side_effect=_mock_solve),
        patch.object(
            solver.gatherer, "_web_search",
            return_value="**Web Result**\nhttps://example.com\nAnswer here",
        ),
    ):
        result = await solver.asolve(query="What is httpx?")

    assert result.success is True
    assert "WEB SEARCH RESULTS" in captured_context["full"]
    assert "Web Result" in captured_context["full"]


@pytest.mark.asyncio
async def test_web_category_forces_planner(tmp_path: Path):
    """category='web' forces planner path even for small projects."""
    (tmp_path / "app.py").write_text("tiny")

    solver = ContextFirstSolver(
        project_root=tmp_path,
        config=ContextFirstConfig(max_context_bytes=10_000),  # Small enough for fast path
    )

    planner_called = False

    async def _mock_plan(query, file_manifest, project_summary):
        nonlocal planner_called
        planner_called = True
        return PlanResult(
            files_to_read=["app.py"],
            search_patterns=[],
            reasoning="Web query needs planner",
            web_searches=["latest python docs"],
        )

    with (
        patch.object(solver.planner, "aplan", side_effect=_mock_plan),
        patch.object(solver.solver, "asolve", side_effect=_mock_asolve),
        patch.object(
            solver.gatherer, "_web_search",
            return_value="web result",
        ),
    ):
        result = await solver.asolve(
            query="What is the latest Python version?",
            category="web",
        )

    # Planner was called despite project being small enough for fast path.
    assert planner_called is True
    assert result.success is True
    assert result.method == "context_first_planned"
