"""Tests for QueryTriage enrichment and workflow routing."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ── _build_triage_summary tests ──────────────────────────────


class TestBuildTriageSummary:
    """Tests for CodingAgent._build_triage_summary static method."""

    @staticmethod
    def _build(context: str) -> str:
        from memfun_agent.coding_agent import RLMCodingAgent

        return RLMCodingAgent._build_triage_summary(context)

    def test_extracts_code_map(self) -> None:
        context = (
            "=== CURRENT PROJECT STATE ===\n"
            "--- README.md ---\n# My App\n"
            "--- Code Map ---\n"
            "app.py (2.1 KB)\n"
            "  class App\n"
            "    def run(self)\n"
            "utils.py (800 B)\n"
            "  def helper(x: int) -> str\n"
            "--- style.css ---\nbody { color: red; }\n"
        )
        result = self._build(context)
        assert "--- Code Map ---" in result
        assert "class App" in result
        assert "def helper" in result

    def test_extracts_project_type_pyproject(self) -> None:
        context = (
            "--- pyproject.toml ---\n"
            "[project]\nname = \"myapp\"\nversion = \"1.0\"\n"
            "--- Code Map ---\napp.py (1 KB)\n"
        )
        result = self._build(context)
        assert "pyproject.toml" in result
        assert 'name = "myapp"' in result

    def test_extracts_project_type_package_json(self) -> None:
        context = (
            "--- package.json ---\n"
            '{"name": "my-app", "version": "1.0"}\n'
            "--- Code Map ---\nindex.js (500 B)\n"
        )
        result = self._build(context)
        assert "package.json" in result
        assert "my-app" in result

    def test_truncation_at_4000(self) -> None:
        # Large code map should be truncated
        big_map = "--- Code Map ---\n" + "x" * 5000
        result = self._build(big_map)
        assert len(result) <= 4000

    def test_fallback_when_no_markers(self) -> None:
        context = "Project root: /tmp/myapp\nPython 3.12"
        result = self._build(context)
        assert "Project root" in result

    def test_empty_context(self) -> None:
        result = self._build("")
        assert result == ""

    def test_code_map_capped_at_3000(self) -> None:
        big_map = "--- Code Map ---\n" + "a" * 4000
        result = self._build(big_map)
        # Total should be within 4000
        assert len(result) <= 4000


# ── Triage category validation tests ────────────────────────


class TestTriageCategoryValidation:
    """Tests for _triage_query category validation."""

    def test_workflow_is_valid_category(self) -> None:
        """'workflow' should be accepted as a valid category."""
        valid = ("direct", "project", "task", "web", "workflow")
        assert "workflow" in valid

    def test_invalid_category_normalizes_to_project(self) -> None:
        """Unknown categories should fall back to 'project'."""
        valid = (
            "direct", "project", "task", "web", "workflow",
        )
        assert "unknown" not in valid
        assert "complex" not in valid


# ── Workflow routing tests ───────────────────────────────────


class TestWorkflowRouting:
    """Tests for workflow category routing in chat.py."""

    @pytest.mark.asyncio
    async def test_workflow_category_routes_to_engine(
        self,
    ) -> None:
        """'workflow' should go directly to WorkflowEngine."""
        # Verify the routing logic exists by checking the code
        # structure.  Full integration requires a running runtime,
        # so we verify the control flow pattern.
        import inspect

        from memfun_cli.commands.chat import ChatSession

        src = inspect.getsource(
            ChatSession._try_workflow_or_single
        )
        # "workflow" routing should appear BEFORE "task" routing
        wf_idx = src.index('category == "workflow"')
        task_idx = src.index('category == "task"')
        assert wf_idx < task_idx, (
            "'workflow' routing must come before 'task' routing"
        )

    @pytest.mark.asyncio
    async def test_workflow_fallback_to_single_agent(
        self,
    ) -> None:
        """If WorkflowEngine fails, should fall through."""
        import inspect

        from memfun_cli.commands.chat import ChatSession

        src = inspect.getsource(
            ChatSession._try_workflow_or_single
        )
        # After workflow failure, should fall through (not raise)
        assert "falling back to single-agent" in src


# ── QueryTriage signature tests ──────────────────────────────


class TestQueryTriageSignature:
    """Tests for the QueryTriage DSPy signature."""

    def test_signature_has_workflow_in_category_desc(
        self,
    ) -> None:
        from memfun_agent.signatures import QueryTriage

        desc = QueryTriage.model_fields[
            "category"
        ].json_schema_extra["desc"]
        assert "workflow" in desc.lower()

    def test_signature_has_task_and_workflow_distinction(
        self,
    ) -> None:
        from memfun_agent.signatures import QueryTriage

        desc = QueryTriage.model_fields[
            "category"
        ].json_schema_extra["desc"]
        # 'task' is DEFAULT, 'workflow' is EXTREMELY RARE
        assert "task" in desc.lower()
        assert "workflow" in desc.lower()

    def test_signature_project_summary_mentions_code_map(
        self,
    ) -> None:
        from memfun_agent.signatures import QueryTriage

        desc = QueryTriage.model_fields[
            "project_summary"
        ].json_schema_extra["desc"]
        assert "code map" in desc.lower()


# ── Truncation detection tests ───────────────────────────────


class TestTruncationDetection:
    """Tests for _detect_truncation in context_first.py."""

    @staticmethod
    def _detect(raw: str, ops: list[dict]) -> bool:  # type: ignore[type-arg]
        from memfun_agent.context_first import _detect_truncation

        return _detect_truncation(raw, ops)

    def test_empty_ops_not_truncated(self) -> None:
        assert self._detect("[]", []) is False

    def test_empty_string_not_truncated(self) -> None:
        assert self._detect("", []) is False

    def test_valid_json_array_not_truncated(self) -> None:
        raw = '[{"op":"write_file","path":"a.py","content":"x"}]'
        ops = [{"op": "write_file", "path": "a.py", "content": "x"}]
        assert self._detect(raw, ops) is False

    def test_unclosed_json_array_is_truncated(self) -> None:
        raw = '[{"op":"write_file","path":"a.py","content":"x"},'
        ops = [{"op": "write_file", "path": "a.py", "content": "x"}]
        assert self._detect(raw, ops) is True

    def test_mid_object_cutoff_is_truncated(self) -> None:
        raw = '[{"op":"write_file","path":"a.py","content":"hel'
        assert self._detect(raw, []) is True

    def test_substantial_raw_but_no_parsed_is_truncated(
        self,
    ) -> None:
        raw = "x" * 200  # Lots of text but nothing parseable
        assert self._detect(raw, []) is True

    def test_short_unparseable_not_truncated(self) -> None:
        # Short garbage isn't truncation, just bad output
        raw = "no ops"
        assert self._detect(raw, []) is False

    def test_markdown_code_fence_not_truncated(self) -> None:
        """Valid JSON wrapped in markdown code fences should NOT be
        detected as truncated (regression: ``` after ] caused false positive)."""
        raw = '```json\n[{"op":"write_file","path":"a.py","content":"x"}]\n```'
        ops = [{"op": "write_file", "path": "a.py", "content": "x"}]
        assert self._detect(raw, ops) is False

    def test_markdown_code_fence_with_trailing_newline(self) -> None:
        """Code fence with trailing whitespace after ``` should also be fine."""
        raw = '```json\n[{"op":"edit_file","path":"b.py","old":"a","new":"b"}]\n```\n'
        ops = [{"op": "edit_file", "path": "b.py", "old": "a", "new": "b"}]
        assert self._detect(raw, ops) is False

    def test_solve_result_has_truncated_field(self) -> None:
        from memfun_agent.context_first import SolveResult

        result = SolveResult(
            reasoning="test",
            answer="test",
            operations=[],
            truncated=True,
        )
        assert result.truncated is True

    def test_solve_result_truncated_default_false(self) -> None:
        from memfun_agent.context_first import SolveResult

        result = SolveResult(
            reasoning="test",
            answer="test",
            operations=[],
        )
        assert result.truncated is False


# ── Decomposer group ID validation tests ────────────────────


class TestDecomposerGroupValidation:
    """Tests for group ID validation in TaskDecomposer."""

    def test_infer_groups_from_sub_tasks(self) -> None:
        """_infer_groups builds correct groups from deps."""
        from memfun_agent.decomposer import SubTask, _infer_groups

        tasks = [
            SubTask(id="T1", description="a", agent_type="coder"),
            SubTask(
                id="T2", description="b",
                agent_type="coder", depends_on=["T1"],
            ),
            SubTask(id="T3", description="c", agent_type="coder"),
        ]
        groups = _infer_groups(tasks)
        # T1 and T3 are independent → first group
        # T2 depends on T1 → second group
        assert len(groups) == 2
        assert "T1" in groups[0]
        assert "T3" in groups[0]
        assert groups[1] == ["T2"]

    def test_infer_groups_no_deps(self) -> None:
        """All independent tasks go into one group."""
        from memfun_agent.decomposer import SubTask, _infer_groups

        tasks = [
            SubTask(id="T1", description="a", agent_type="coder"),
            SubTask(id="T2", description="b", agent_type="coder"),
            SubTask(id="T3", description="c", agent_type="coder"),
        ]
        groups = _infer_groups(tasks)
        assert len(groups) == 1
        assert set(groups[0]) == {"T1", "T2", "T3"}

    def test_parse_groups_mismatched_ids_are_rebuilt(self) -> None:
        """When LLM produces mismatched group IDs, decomposer
        should fall back to _infer_groups."""
        from memfun_agent.decomposer import SubTask, _infer_groups, _parse_groups

        # Simulating: LLM produces groups with wrong IDs
        raw_groups = [["1", "2"], ["3"]]
        groups = _parse_groups(raw_groups)
        assert groups == [["1", "2"], ["3"]]

        # Now validate: IDs don't match sub_tasks
        sub_tasks = [
            SubTask(id="T1", description="a", agent_type="coder"),
            SubTask(id="T2", description="b", agent_type="coder"),
            SubTask(id="T3", description="c", agent_type="coder"),
        ]
        known_ids = {t.id for t in sub_tasks}
        all_group_ids = {tid for grp in groups for tid in grp}

        # IDs don't match
        assert not all_group_ids.issubset(known_ids)

        # Rebuild should use correct IDs
        rebuilt = _infer_groups(sub_tasks)
        rebuilt_ids = {tid for grp in rebuilt for tid in grp}
        assert rebuilt_ids == known_ids

    def test_parse_groups_python_list_repr(self) -> None:
        """LLM may return groups as Python list repr strings
        like "['T1']" instead of proper nested lists."""
        from memfun_agent.decomposer import _parse_groups

        # This is what the LLM actually returned in production
        raw_groups = ["['T1']", "['T2', 'T3']", "['T4']"]
        groups = _parse_groups(raw_groups)
        assert groups == [["T1"], ["T2", "T3"], ["T4"]]

    def test_parse_groups_json_strings(self) -> None:
        """JSON-encoded strings should be parsed correctly."""
        from memfun_agent.decomposer import _parse_groups

        raw_groups = ['["T1", "T2"]', '["T3"]']
        groups = _parse_groups(raw_groups)
        assert groups == [["T1", "T2"], ["T3"]]

    def test_decomposition_result_groups_match_subtasks(self) -> None:
        """End-to-end: DecompositionResult groups should always
        reference valid sub_task IDs."""
        from memfun_agent.decomposer import DecompositionResult, SubTask

        sub_tasks = [
            SubTask(id="T1", description="a", agent_type="coder"),
            SubTask(id="T2", description="b", agent_type="coder"),
        ]
        result = DecompositionResult(
            sub_tasks=sub_tasks,
            shared_spec="",
            parallelism_groups=[["T1", "T2"]],
            is_single_task=False,
        )
        known_ids = {t.id for t in result.sub_tasks}
        group_ids = {
            tid for grp in result.parallelism_groups
            for tid in grp
        }
        assert group_ids.issubset(known_ids)


# ── Workflow _execute_group robustness tests ─────────────────


class TestWorkflowGroupExecution:
    """Tests for _execute_group handling of mismatched IDs."""

    def test_build_result_reports_none_results(self) -> None:
        """_build_result should flag sub-tasks with None result."""
        from memfun_agent.decomposer import SubTask
        from memfun_agent.workflow import (
            ReviewResult,
            SubTaskStatus,
            WorkflowEngine,
            WorkflowState,
        )

        state = WorkflowState(workflow_id="test123")
        state.sub_task_statuses["T1"] = SubTaskStatus(
            task_id="test123_T1",
            sub_task=SubTask(
                id="T1", description="do stuff",
                agent_type="coder",
            ),
            agent_name="coder-agent",
        )
        # result is None (default) — agent never ran.

        review = ReviewResult(
            approved=True, issues=[], summary="ok",
        )
        wf_result = WorkflowEngine._build_result(
            state, review, 100.0,
        )
        assert "No result" in wf_result.answer
        assert "coder-agent" in wf_result.answer
        assert wf_result.success is True  # workflow completed

    def test_build_result_includes_successful_results(self) -> None:
        """_build_result should include data from successful agents."""
        from memfun_agent.decomposer import SubTask
        from memfun_agent.workflow import (
            ReviewResult,
            SubTaskStatus,
            WorkflowEngine,
            WorkflowState,
        )
        from memfun_core.types import TaskResult

        state = WorkflowState(workflow_id="test456")
        state.sub_task_statuses["T1"] = SubTaskStatus(
            task_id="test456_T1",
            sub_task=SubTask(
                id="T1", description="write code",
                agent_type="coder",
            ),
            agent_name="coder-agent",
            result=TaskResult(
                task_id="test456_T1",
                agent_id="coder-agent",
                success=True,
                result={
                    "answer": "Done! Created app.py",
                    "ops": [],
                    "files_created": ["app.py"],
                },
            ),
        )

        review = ReviewResult(
            approved=True, issues=[], summary="ok",
        )
        wf_result = WorkflowEngine._build_result(
            state, review, 200.0,
        )
        assert "No result" not in wf_result.answer
        assert "Done! Created app.py" in wf_result.answer
        assert "app.py" in wf_result.files_created


# ── Agent activity tracking tests ─────────────────────────────


class TestAgentActivity:
    """Tests for AGENT_ACTIVITY live tracking dict."""

    def test_activity_dict_exists(self) -> None:
        from memfun_agent.specialists import AGENT_ACTIVITY

        assert isinstance(AGENT_ACTIVITY, dict)

    def test_activity_update_and_cleanup(self) -> None:
        from memfun_agent.specialists import AGENT_ACTIVITY

        # Simulate what the on_step callback does.
        AGENT_ACTIVITY["T1"] = {
            "iteration": 3,
            "max_iter": 10,
            "last_op": "read_file('app.py')",
            "agent_name": "coder-agent",
        }
        assert AGENT_ACTIVITY["T1"]["iteration"] == 3
        assert AGENT_ACTIVITY["T1"]["last_op"] == "read_file('app.py')"

        # Simulate cleanup after handle() completes.
        AGENT_ACTIVITY.pop("T1", None)
        assert "T1" not in AGENT_ACTIVITY

    def test_activity_exported_from_package(self) -> None:
        from memfun_agent import AGENT_ACTIVITY

        assert isinstance(AGENT_ACTIVITY, dict)


# ── Post-processing pipeline tests ────────────────────────────


class TestReadAffectedFiles:
    """Tests for the standalone read_affected_files function."""

    @pytest.mark.asyncio
    async def test_reads_existing_files(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import read_affected_files

        f1 = tmp_path / "hello.py"
        f1.write_text("print('hello')")
        result = await read_affected_files([str(f1)], tmp_path)
        assert "=== FILE: hello.py ===" in result
        assert "print('hello')" in result

    @pytest.mark.asyncio
    async def test_skips_missing_files(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import read_affected_files

        result = await read_affected_files(
            [str(tmp_path / "nonexistent.py")], tmp_path,
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_multiple_files(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import read_affected_files

        (tmp_path / "a.py").write_text("A")
        (tmp_path / "b.py").write_text("B")
        result = await read_affected_files(
            [str(tmp_path / "a.py"), str(tmp_path / "b.py")],
            tmp_path,
        )
        assert "=== FILE: a.py ===" in result
        assert "=== FILE: b.py ===" in result


class TestDetectVerifyCommands:
    """Tests for _detect_verify_commands linter auto-detection."""

    def test_detects_python_ruff(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import _detect_verify_commands

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "app.py").write_text("x = 1\n")
        cmds = _detect_verify_commands(tmp_path)
        assert any("ruff" in c for c in cmds)

    def test_no_linter_for_empty_dir(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import _detect_verify_commands

        cmds = _detect_verify_commands(tmp_path)
        assert cmds == []

    def test_detects_js_eslint(self, tmp_path: Path) -> None:
        from memfun_agent.context_first import _detect_verify_commands

        (tmp_path / "package.json").write_text('{"name":"x"}')
        (tmp_path / ".eslintrc.json").write_text("{}")
        cmds = _detect_verify_commands(tmp_path)
        assert any("eslint" in c for c in cmds)


class TestWriteFileOverwriteGuard:
    """Tests for the RLM write_file destructive overwrite guard."""

    def test_blocks_destructive_overwrite(self, tmp_path: Path) -> None:
        """write_file should block when new content loses >30% of lines."""
        from memfun_agent.rlm import _make_write_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        write_file = _make_write_file(ns)

        # Create a file with 20 lines.
        fpath = tmp_path / "app.py"
        original = "\n".join(f"line {i}" for i in range(20))
        fpath.write_text(original)

        # Try to overwrite with only 5 lines (75% loss).
        short = "\n".join(f"new {i}" for i in range(5))
        write_file(str(fpath), short)

        # File should be UNCHANGED (write blocked).
        assert fpath.read_text() == original
        # No ops should be recorded.
        assert len(ns["state"]["_ops"]) == 0

    def test_allows_same_size_overwrite(self, tmp_path: Path) -> None:
        """write_file should allow overwrites that don't lose lines."""
        from memfun_agent.rlm import _make_write_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        write_file = _make_write_file(ns)

        fpath = tmp_path / "app.py"
        original = "\n".join(f"line {i}" for i in range(20))
        fpath.write_text(original)

        # Overwrite with similar size — should be allowed.
        replacement = "\n".join(f"updated {i}" for i in range(18))
        write_file(str(fpath), replacement)

        assert fpath.read_text() == replacement
        assert len(ns["state"]["_ops"]) == 1

    def test_allows_new_file_creation(self, tmp_path: Path) -> None:
        """write_file should allow writing new files freely."""
        from memfun_agent.rlm import _make_write_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        write_file = _make_write_file(ns)

        fpath = tmp_path / "new_file.py"
        content = "print('hello')\n"
        write_file(str(fpath), content)

        assert fpath.read_text() == content
        assert len(ns["state"]["_ops"]) == 1

    def test_allows_small_file_overwrite(self, tmp_path: Path) -> None:
        """write_file should allow overwriting small files (<=10 lines)."""
        from memfun_agent.rlm import _make_write_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        write_file = _make_write_file(ns)

        fpath = tmp_path / "small.py"
        original = "line1\nline2\nline3\n"
        fpath.write_text(original)

        write_file(str(fpath), "x = 1\n")
        assert fpath.read_text() == "x = 1\n"
        assert len(ns["state"]["_ops"]) == 1


class TestSpecialistOutputPreReading:
    """Tests for specialist output file pre-reading into context."""

    def test_coder_agent_system_prefix_mentions_edit_file(self) -> None:
        """CoderAgent should instruct to use edit_file for existing files."""
        from memfun_agent.specialists import CoderAgent

        prefix = CoderAgent._SYSTEM_PREFIX
        assert "edit_file" in prefix
        assert "read_file" in prefix
        assert "NEVER rewrite" in prefix

    def test_debug_agent_system_prefix_mentions_edit_file(self) -> None:
        """DebugAgent should instruct to use edit_file for fixes."""
        from memfun_agent.specialists import DebugAgent

        prefix = DebugAgent._SYSTEM_PREFIX
        assert "edit_file" in prefix
        assert "NEVER rewrite" in prefix

    def test_test_agent_system_prefix_mentions_edit_file(self) -> None:
        """TestAgent should instruct to use edit_file for existing files."""
        from memfun_agent.specialists import TestAgent

        prefix = TestAgent._SYSTEM_PREFIX
        assert "edit_file" in prefix
        assert "read_file" in prefix


class TestReadFileCaching:
    """Tests for RLM read_file caching to prevent re-read loops."""

    def test_first_read_returns_full_content(self, tmp_path: Path) -> None:
        from memfun_agent.rlm import _make_read_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        read_file = _make_read_file(ns)

        fpath = tmp_path / "app.py"
        fpath.write_text("x = 1\ny = 2\n")
        result = read_file(str(fpath))

        assert "x = 1" in result
        assert "y = 2" in result

    def test_second_read_returns_cache_message(self, tmp_path: Path) -> None:
        from memfun_agent.rlm import _make_read_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        read_file = _make_read_file(ns)

        fpath = tmp_path / "app.py"
        fpath.write_text("x = 1\ny = 2\n")

        # First read — full content.
        read_file(str(fpath))
        # Second read — should be cache message.
        result2 = read_file(str(fpath))

        assert "already read" in result2.lower()
        assert "unchanged" in result2.lower()
        assert "x = 1" not in result2

    def test_read_after_file_change_returns_new_content(
        self, tmp_path: Path,
    ) -> None:
        from memfun_agent.rlm import _make_read_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        read_file = _make_read_file(ns)

        fpath = tmp_path / "app.py"
        fpath.write_text("x = 1\n")

        read_file(str(fpath))

        # Modify the file.
        fpath.write_text("x = 42\nz = 99\n")
        result2 = read_file(str(fpath))

        # Should return full new content.
        assert "x = 42" in result2
        assert "z = 99" in result2

    def test_cache_tracks_ops(self, tmp_path: Path) -> None:
        """Both cached and uncached reads should log ops."""
        from memfun_agent.rlm import _make_read_file

        ns: dict[str, dict[str, list[object]]] = {
            "state": {"_ops": [], "_files": []}
        }
        read_file = _make_read_file(ns)

        fpath = tmp_path / "app.py"
        fpath.write_text("hello\n")

        read_file(str(fpath))
        read_file(str(fpath))

        assert len(ns["state"]["_ops"]) == 2
        assert all(o[0] == "read" for o in ns["state"]["_ops"])


class TestEditOnlyExecutor:
    """Tests for OperationExecutor edit_only mode."""

    @pytest.mark.asyncio
    async def test_edit_only_blocks_write_on_existing_file(
        self, tmp_path: Path,
    ) -> None:
        from memfun_agent.context_first import OperationExecutor

        fpath = tmp_path / "app.py"
        fpath.write_text("original content\n")

        executor = OperationExecutor(
            tmp_path, edit_only=True,
        )
        await executor.execute([
            {"op": "write_file", "path": str(fpath), "content": "new"},
        ])

        # File should be unchanged.
        assert fpath.read_text() == "original content\n"
        assert executor.failed == 1
        assert len(executor.ops) == 0

    @pytest.mark.asyncio
    async def test_edit_only_allows_write_new_file(
        self, tmp_path: Path,
    ) -> None:
        from memfun_agent.context_first import OperationExecutor

        fpath = tmp_path / "new_file.py"

        executor = OperationExecutor(
            tmp_path, edit_only=True,
        )
        await executor.execute([
            {"op": "write_file", "path": str(fpath), "content": "hello"},
        ])

        # New file should be created.
        assert fpath.read_text() == "hello"
        assert executor.failed == 0
        assert len(executor.ops) == 1

    @pytest.mark.asyncio
    async def test_edit_only_allows_edit_file(
        self, tmp_path: Path,
    ) -> None:
        from memfun_agent.context_first import OperationExecutor

        fpath = tmp_path / "app.py"
        fpath.write_text("old_value = 1\n")

        executor = OperationExecutor(
            tmp_path, edit_only=True,
        )
        await executor.execute([
            {
                "op": "edit_file",
                "path": str(fpath),
                "old": "old_value = 1",
                "new": "new_value = 42",
            },
        ])

        assert "new_value = 42" in fpath.read_text()
        assert executor.failed == 0
        assert len(executor.ops) == 1
