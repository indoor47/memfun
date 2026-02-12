"""End-to-end integration tests for the single-agent flow (Phase 2).

Tests cover: agent lifecycle, sandbox execution, skill discovery/validation/sync,
runtime builder configuration, and trace collection round-trips.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pytest
from memfun_core.config import BackendConfig, MemfunConfig
from memfun_core.types import ExecutionResult, TaskMessage, TaskResult
from memfun_runtime.agent import get_agent_registry
from memfun_runtime.backends.memory import (
    InProcessSkillRegistry,
    InProcessStateStore,
)
from memfun_runtime.backends.sandbox.local import LocalSandbox
from memfun_runtime.builder import RuntimeBuilder
from memfun_runtime.context import RuntimeContext

# ── Fixtures ──────────────────────────────────────────────────

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@pytest.fixture
def memory_config() -> MemfunConfig:
    return MemfunConfig(backend=BackendConfig(tier="memory"))


@pytest.fixture
async def memory_runtime(memory_config: MemfunConfig) -> RuntimeContext:
    return await RuntimeBuilder(memory_config).build()


# ── 1. Agent Lifecycle Tests ──────────────────────────────────


def _make_mock_prediction(**fields: str) -> MagicMock:
    """Build a mock DSPy prediction object with attribute access."""
    pred = MagicMock()
    for k, v in fields.items():
        setattr(pred, k, v)
    return pred


class TestAgentLifecycle:
    """Integration tests for creating, starting, handling, and stopping an agent."""

    async def test_agent_start_handle_stop(
        self, memory_runtime: RuntimeContext
    ) -> None:
        """Full lifecycle: on_start -> handle -> on_stop with mocked LLM."""
        from memfun_agent.coding_agent import RLMCodingAgent

        agent = RLMCodingAgent(memory_runtime)

        # Mock away DSPy LLM calls and tool bridge initialization
        mock_pred = _make_mock_prediction(
            analysis="Looks good",
            issues=["none"],
            suggestions=["none"],
        )

        with (
            patch.object(
                dspy.ChainOfThought,
                "__call__",
                return_value=mock_pred,
            ),
            patch(
                "memfun_agent.coding_agent.create_tool_bridge",
                side_effect=RuntimeError("no MCP in tests"),
            ),
        ):
            await agent.on_start()

            task = TaskMessage(
                task_id="task-001",
                agent_id=agent.agent_id,
                payload={
                    "type": "analyze",
                    "query": "Check quality",
                    "code": "x = 1",
                },
            )

            result = await agent.handle(task)

            assert isinstance(result, TaskResult)
            assert result.success is True
            assert result.task_id == "task-001"
            assert result.agent_id == agent.agent_id
            assert result.duration_ms > 0

            await agent.on_stop()

    async def test_agent_unknown_task_type(
        self, memory_runtime: RuntimeContext
    ) -> None:
        """Sending an unknown task type returns success=False."""
        from memfun_agent.coding_agent import RLMCodingAgent

        agent = RLMCodingAgent(memory_runtime)

        with patch(
            "memfun_agent.coding_agent.create_tool_bridge",
            side_effect=RuntimeError("no MCP in tests"),
        ):
            await agent.on_start()

        task = TaskMessage(
            task_id="task-bad",
            agent_id=agent.agent_id,
            payload={"type": "deploy", "query": "do it"},
        )
        result = await agent.handle(task)

        assert result.success is False
        assert "Unknown task type" in (result.error or "")

        await agent.on_stop()

    async def test_agent_registers_in_registry(self) -> None:
        """Importing RLMCodingAgent registers it in the global agent registry."""
        # The @agent decorator registers the class at import time
        from memfun_agent.coding_agent import RLMCodingAgent

        registry = get_agent_registry()
        assert "rlm-coder" in registry
        assert registry["rlm-coder"] is RLMCodingAgent

    async def test_agent_metadata(
        self, memory_runtime: RuntimeContext
    ) -> None:
        """Agent exposes correct id and version from decorator metadata."""
        from memfun_agent.coding_agent import RLMCodingAgent

        agent = RLMCodingAgent(memory_runtime)
        assert agent.agent_id == "rlm-coder"
        assert agent.version == "1.0"

    async def test_agent_handle_explain_task(
        self, memory_runtime: RuntimeContext
    ) -> None:
        """Handle an 'explain' task through the direct predictor path."""
        from memfun_agent.coding_agent import RLMCodingAgent

        agent = RLMCodingAgent(memory_runtime)

        mock_pred = _make_mock_prediction(
            explanation="This adds two numbers",
            key_concepts=["addition", "variables"],
        )

        with (
            patch.object(
                dspy.ChainOfThought,
                "__call__",
                return_value=mock_pred,
            ),
            patch(
                "memfun_agent.coding_agent.create_tool_bridge",
                side_effect=RuntimeError("no MCP in tests"),
            ),
        ):
            await agent.on_start()

            task = TaskMessage(
                task_id="task-explain",
                agent_id=agent.agent_id,
                payload={
                    "type": "explain",
                    "query": "What does this do?",
                    "code": "def add(a, b): return a + b",
                    "audience": "beginner",
                },
            )
            result = await agent.handle(task)

        assert result.success is True
        assert "explain" in result.result.get("task_type", "")
        await agent.on_stop()


# ── 2. Sandbox Integration Tests ──────────────────────────────


class TestSandboxIntegration:
    """Integration tests for LocalSandbox code execution."""

    async def test_local_sandbox_execute(self) -> None:
        """Execute simple Python code and verify stdout + exit_code."""
        sandbox = LocalSandbox()
        result = await sandbox.execute('print("hello world")', context={})

        assert isinstance(result, ExecutionResult)
        assert result.stdout.strip() == "hello world"
        assert result.exit_code == 0
        assert result.duration_ms > 0

    async def test_sandbox_with_context(self) -> None:
        """Execute code that uses injected context variables."""
        sandbox = LocalSandbox()
        result = await sandbox.execute(
            "print(f'{greeting}, {name}!')",
            context={"greeting": "Hello", "name": "Memfun"},
        )

        assert result.exit_code == 0
        assert "Hello, Memfun!" in result.stdout

    async def test_sandbox_timeout(self) -> None:
        """Execute an infinite loop with a short timeout; verify it stops."""
        sandbox = LocalSandbox()
        code = "import time\nwhile True:\n    time.sleep(0.1)"
        result = await sandbox.execute(code, context={}, timeout=1)

        assert result.exit_code != 0
        assert "timed out" in result.stderr.lower()

    async def test_sandbox_captures_stderr(self) -> None:
        """Verify stderr output is captured separately."""
        sandbox = LocalSandbox()
        result = await sandbox.execute(
            "import sys; sys.stderr.write('error msg\\n')",
            context={},
        )
        assert "error msg" in result.stderr

    async def test_sandbox_syntax_error(self) -> None:
        """Syntax errors produce non-zero exit code and stderr."""
        sandbox = LocalSandbox()
        result = await sandbox.execute("def (broken", context={})

        assert result.exit_code != 0
        assert result.stderr != ""

    async def test_sandbox_multi_statement(self) -> None:
        """Execute multiple statements and verify combined output."""
        sandbox = LocalSandbox()
        code = "x = 10\ny = 20\nprint(x + y)"
        result = await sandbox.execute(code, context={})

        assert result.exit_code == 0
        assert result.stdout.strip() == "30"


# ── 3. Skills Integration Tests ───────────────────────────────


class TestSkillsIntegration:
    """Integration tests for skill discovery, parsing, validation, and sync."""

    def test_discover_builtin_skills(self) -> None:
        """SkillLoader discovers all 8 built-in skills from skills/ directory."""
        from memfun_skills.loader import SkillLoader

        loader = SkillLoader()
        skills = loader.discover([SKILLS_DIR])

        assert len(skills) == 8
        names = {s.name for s in skills}
        assert names == {
            "analyze-code", "ask", "explain-code", "fix-bugs",
            "generate-tests", "refactor", "review-code", "security-audit",
        }

    def test_parse_and_validate_all_skills(self) -> None:
        """Parse each built-in skill and validate with SkillValidator."""
        from memfun_skills.loader import SkillLoader
        from memfun_skills.validator import SkillValidator

        loader = SkillLoader()
        skills = loader.discover([SKILLS_DIR])
        validator = SkillValidator()

        assert len(skills) == 8

        for skill in skills:
            errors = validator.validate(skill)
            assert errors == [], (
                f"Skill '{skill.name}' has validation errors: {errors}"
            )

    def test_all_builtin_skills_have_instructions(self) -> None:
        """Every built-in skill should have non-empty instructions."""
        from memfun_skills.loader import SkillLoader

        loader = SkillLoader()
        skills = loader.discover([SKILLS_DIR])

        for skill in skills:
            assert skill.instructions.strip(), (
                f"Skill '{skill.name}' has empty instructions"
            )
            assert skill.description.strip(), (
                f"Skill '{skill.name}' has empty description"
            )

    def test_all_builtin_skills_have_version(self) -> None:
        """Every built-in skill should have a valid version string."""
        import re

        from memfun_skills.loader import SkillLoader

        semver_re = re.compile(r"^\d+\.\d+\.\d+")

        loader = SkillLoader()
        skills = loader.discover([SKILLS_DIR])

        for skill in skills:
            assert semver_re.match(skill.version), (
                f"Skill '{skill.name}' has invalid version: {skill.version}"
            )

    async def test_skill_registry_sync(self) -> None:
        """Sync discovered skills to InProcessSkillRegistry via the bridge."""
        from memfun_skills.loader import SkillLoader
        from memfun_skills.registry import SkillRegistryBridge

        loader = SkillLoader(extra_paths=[SKILLS_DIR])
        registry = InProcessSkillRegistry()
        bridge = SkillRegistryBridge()

        synced = await bridge.sync_skills(loader, registry, paths=[SKILLS_DIR])

        assert len(synced) == 8

        registered = await registry.list_skills()
        names = {s.name for s in registered}
        assert names == {
            "analyze-code", "ask", "explain-code", "fix-bugs",
            "generate-tests", "refactor", "review-code", "security-audit",
        }

    async def test_skill_registry_get_after_sync(self) -> None:
        """After syncing, each skill can be retrieved individually."""
        from memfun_skills.loader import SkillLoader
        from memfun_skills.registry import SkillRegistryBridge

        loader = SkillLoader(extra_paths=[SKILLS_DIR])
        registry = InProcessSkillRegistry()
        bridge = SkillRegistryBridge()

        await bridge.sync_skills(loader, registry)

        for name in ["analyze-code", "explain-code", "fix-bugs", "review-code"]:
            info = await registry.get_skill(name)
            assert info is not None, f"Skill '{name}' not found in registry"
            assert info.name == name
            assert info.description != ""
            assert info.metadata.get("version") is not None

    async def test_skill_registry_search_after_sync(self) -> None:
        """After syncing, skills can be found by search query."""
        from memfun_skills.loader import SkillLoader
        from memfun_skills.registry import SkillRegistryBridge

        loader = SkillLoader(extra_paths=[SKILLS_DIR])
        registry = InProcessSkillRegistry()
        bridge = SkillRegistryBridge()

        await bridge.sync_skills(loader, registry)

        results = await registry.search_skills("analyze")
        assert len(results) >= 1
        assert any(s.name == "analyze-code" for s in results)


# ── 4. Config / Runtime Builder Tests ────────────────────────


class TestConfigIntegration:
    """Integration tests for RuntimeBuilder with different backend tiers."""

    async def test_runtime_builder_memory(self) -> None:
        """Build RuntimeContext with memory backend; verify all 8 adapters."""
        config = MemfunConfig(backend=BackendConfig(tier="memory"))
        ctx = await RuntimeBuilder(config).build()

        assert isinstance(ctx, RuntimeContext)
        assert ctx.config.backend.tier == "memory"

        # Verify all 8 adapters are present and not None
        assert ctx.event_bus is not None
        assert ctx.state_store is not None
        assert ctx.sandbox is not None
        assert ctx.lifecycle is not None
        assert ctx.registry is not None
        assert ctx.session is not None
        assert ctx.health is not None
        assert ctx.skill_registry is not None

    async def test_runtime_builder_sqlite(self, tmp_path: Path) -> None:
        """Build RuntimeContext with sqlite backend; verify all 8 adapters."""
        db_path = str(tmp_path / "test_integration.db")
        config = MemfunConfig(
            backend=BackendConfig(tier="sqlite", sqlite_path=db_path)
        )
        ctx = await RuntimeBuilder(config).build()

        assert isinstance(ctx, RuntimeContext)
        assert ctx.config.backend.tier == "sqlite"

        assert ctx.event_bus is not None
        assert ctx.state_store is not None
        assert ctx.sandbox is not None
        assert ctx.lifecycle is not None
        assert ctx.registry is not None
        assert ctx.session is not None
        assert ctx.health is not None
        assert ctx.skill_registry is not None

    async def test_runtime_builder_produces_local_sandbox(self) -> None:
        """Memory backend builder should produce a LocalSandbox."""
        config = MemfunConfig(backend=BackendConfig(tier="memory"))
        ctx = await RuntimeBuilder(config).build()

        assert isinstance(ctx.sandbox, LocalSandbox)

    async def test_runtime_context_config_passthrough(self) -> None:
        """RuntimeContext carries the original MemfunConfig."""
        config = MemfunConfig(
            project_name="test-project",
            backend=BackendConfig(tier="memory"),
        )
        ctx = await RuntimeBuilder(config).build()

        assert ctx.config.project_name == "test-project"
        assert ctx.config.llm.provider == "anthropic"

    async def test_runtime_builder_unknown_tier_raises(self) -> None:
        """Unknown backend tier should raise ValueError."""
        config = MemfunConfig(backend=BackendConfig(tier="unknown"))
        with pytest.raises(ValueError, match="Unknown backend tier"):
            await RuntimeBuilder(config).build()


# ── 5. Trace Collection Tests ─────────────────────────────────


class TestTraceIntegration:
    """Integration tests for TraceCollector with a real StateStore backend."""

    async def test_trace_collection_with_state_store(self) -> None:
        """Save a trace via TraceCollector backed by InProcessStateStore,
        then load it back and verify round-trip fidelity."""
        from memfun_agent.traces import (
            ExecutionTrace,
            TokenUsage,
            TraceCollector,
            TraceStep,
        )

        state_store = InProcessStateStore()
        collector = TraceCollector(state_store=state_store)

        assert collector.has_backend is True

        trace = ExecutionTrace(
            task_type="analyze",
            query="Find bugs",
            context_length=5000,
            trajectory=[
                TraceStep(
                    iteration=1,
                    reasoning="Looking at imports",
                    code='print(len(context.split("\\n")))',
                    output="42",
                    duration_ms=12.5,
                ),
                TraceStep(
                    iteration=2,
                    reasoning="Found the issue",
                    code="state['FINAL'] = 'Bug in line 42'",
                    output="",
                    duration_ms=5.0,
                ),
            ],
            final_answer="Bug in line 42",
            success=True,
            duration_ms=100.0,
            agent_id="rlm-coder",
            token_usage=TokenUsage(
                prompt_tokens=500,
                completion_tokens=200,
                total_tokens=700,
                sub_lm_calls=1,
                sub_lm_tokens=50,
            ),
            metadata={"method": "rlm", "iterations": 2},
        )

        trace_id = await collector.save(trace)
        assert trace_id == trace.trace_id

        # Load back
        loaded = await collector.load(trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id
        assert loaded.task_type == "analyze"
        assert loaded.query == "Find bugs"
        assert loaded.context_length == 5000
        assert loaded.final_answer == "Bug in line 42"
        assert loaded.success is True
        assert loaded.agent_id == "rlm-coder"
        assert loaded.duration_ms == 100.0

        # Verify trajectory round-trip
        assert len(loaded.trajectory) == 2
        assert loaded.trajectory[0].iteration == 1
        assert loaded.trajectory[0].reasoning == "Looking at imports"
        assert loaded.trajectory[0].code == 'print(len(context.split("\\n")))'
        assert loaded.trajectory[0].output == "42"
        assert loaded.trajectory[1].iteration == 2
        assert loaded.trajectory[1].code == "state['FINAL'] = 'Bug in line 42'"

        # Verify token usage round-trip
        assert loaded.token_usage.prompt_tokens == 500
        assert loaded.token_usage.completion_tokens == 200
        assert loaded.token_usage.total_tokens == 700
        assert loaded.token_usage.sub_lm_calls == 1
        assert loaded.token_usage.sub_lm_tokens == 50

        # Verify metadata round-trip
        assert loaded.metadata["method"] == "rlm"
        assert loaded.metadata["iterations"] == 2

    async def test_trace_list_and_delete(self) -> None:
        """Save multiple traces, list them, then delete one."""
        from memfun_agent.traces import ExecutionTrace, TraceCollector

        state_store = InProcessStateStore()
        collector = TraceCollector(state_store=state_store)

        trace1 = ExecutionTrace(
            task_type="analyze",
            query="query1",
            final_answer="answer1",
        )
        trace2 = ExecutionTrace(
            task_type="review",
            query="query2",
            final_answer="answer2",
        )

        id1 = await collector.save(trace1)
        id2 = await collector.save(trace2)

        ids = await collector.list_trace_ids()
        assert id1 in ids
        assert id2 in ids

        deleted = await collector.delete(id1)
        assert deleted is True

        # Verify deletion
        assert await collector.load(id1) is None
        assert await collector.load(id2) is not None

        # Delete non-existent returns False
        deleted_again = await collector.delete(id1)
        assert deleted_again is False

    async def test_trace_collector_without_backend(self) -> None:
        """TraceCollector without a state store falls back to in-memory."""
        from memfun_agent.traces import ExecutionTrace, TraceCollector

        collector = TraceCollector(state_store=None)

        assert collector.has_backend is False

        trace = ExecutionTrace(
            task_type="explain",
            query="what is this?",
            final_answer="it is a test",
        )

        trace_id = await collector.save(trace)
        loaded = await collector.load(trace_id)
        assert loaded is not None
        assert loaded.final_answer == "it is a test"

    async def test_trace_load_nonexistent(self) -> None:
        """Loading a nonexistent trace returns None."""
        from memfun_agent.traces import TraceCollector

        state_store = InProcessStateStore()
        collector = TraceCollector(state_store=state_store)

        result = await collector.load("nonexistent-trace-id")
        assert result is None

    async def test_trace_json_serialization(self) -> None:
        """ExecutionTrace round-trips through JSON serialization."""
        from memfun_agent.traces import (
            ExecutionTrace,
            TokenUsage,
            TraceStep,
        )

        trace = ExecutionTrace(
            task_type="fix",
            query="fix the bug",
            context_length=1234,
            trajectory=[
                TraceStep(
                    iteration=1,
                    reasoning="analyzing",
                    code="print(context[:100])",
                    output="first 100 chars",
                    output_truncated=False,
                    duration_ms=10.0,
                ),
            ],
            final_answer="fixed it",
            success=True,
            duration_ms=50.0,
            agent_id="test-agent",
            token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
            metadata={"method": "direct"},
        )

        json_str = trace.to_json()
        parsed = json.loads(json_str)
        assert parsed["task_type"] == "fix"
        assert parsed["query"] == "fix the bug"
        assert len(parsed["trajectory"]) == 1

        restored = ExecutionTrace.from_json(json_str)
        assert restored.trace_id == trace.trace_id
        assert restored.task_type == trace.task_type
        assert restored.final_answer == trace.final_answer
        assert restored.trajectory[0].reasoning == "analyzing"
        assert restored.token_usage.prompt_tokens == 100


# ── 6. RLM Module Tests (with mocked LLM) ────────────────────


class TestRLMModuleIntegration:
    """Integration tests for the RLM exploration loop with mocked DSPy."""

    async def test_rlm_sets_final_in_one_iteration(self) -> None:
        """RLM loop stops when code sets state['FINAL']."""
        from memfun_agent.rlm import RLMConfig, RLMModule

        config = RLMConfig(max_iterations=5, verbose=False)
        rlm = RLMModule(config=config, sandbox=None)

        # Mock the DSPy Predict to generate code that sets FINAL immediately
        mock_pred = _make_mock_prediction(
            reasoning="I know the answer",
            next_code="state['FINAL'] = 'the answer is 42'",
        )

        with patch.object(
            dspy.Predict,
            "__call__",
            return_value=mock_pred,
        ):
            result = await rlm.aforward(
                query="What is the answer?",
                context="some context data",
            )

        assert result.success is True
        assert result.answer == "the answer is 42"
        assert result.iterations == 1
        assert len(result.trajectory) == 1
        assert result.trajectory[0].iteration == 1

    async def test_rlm_respects_max_iterations(self) -> None:
        """RLM loop stops after max_iterations even without FINAL."""
        from memfun_agent.rlm import RLMConfig, RLMModule

        config = RLMConfig(max_iterations=3, verbose=False)
        rlm = RLMModule(config=config, sandbox=None)

        # Mock code that explores but never sets FINAL
        mock_pred = _make_mock_prediction(
            reasoning="Exploring further",
            next_code="print('searching...')",
        )

        with patch.object(
            dspy.Predict,
            "__call__",
            return_value=mock_pred,
        ):
            result = await rlm.aforward(
                query="Find something",
                context="lots of data here",
            )

        assert result.iterations == 3
        assert len(result.trajectory) == 3

    async def test_rlm_empty_code_stops(self) -> None:
        """RLM loop stops when LLM produces empty code."""
        from memfun_agent.rlm import RLMConfig, RLMModule

        config = RLMConfig(max_iterations=10, verbose=False)
        rlm = RLMModule(config=config, sandbox=None)

        mock_pred = _make_mock_prediction(
            reasoning="I have nothing more to do",
            next_code="",
        )

        with patch.object(
            dspy.Predict,
            "__call__",
            return_value=mock_pred,
        ):
            result = await rlm.aforward(
                query="Quick question",
                context="data",
            )

        # Should stop without executing any code
        assert result.iterations == 1
        assert len(result.trajectory) == 0


# ── 7. Tool Bridge Tests ──────────────────────────────────────


class TestToolBridgeIntegration:
    """Integration tests for MCPToolBridge (no real MCP gateway)."""

    async def test_bridge_without_gateway(self) -> None:
        """MCPToolBridge works gracefully without a gateway."""
        import sys

        from memfun_agent.tool_bridge import MCPToolBridge

        # Block the lazy import so the gateway stays None
        with patch.dict(sys.modules, {"memfun_tools.gateway": None}):
            bridge = MCPToolBridge(gateway=None)
            await bridge.initialize()

        # Calling a tool without gateway returns an error result
        result = await bridge.call_tool("fs_read_file", path="/etc/hosts")
        assert result.success is False
        assert "not available" in (result.error or "")

    async def test_bridge_get_repl_tools(self) -> None:
        """get_repl_tools returns a dict of callable wrappers."""
        from memfun_agent.tool_bridge import MCPToolBridge

        bridge = MCPToolBridge(gateway=None)
        tools = bridge.get_repl_tools()

        assert isinstance(tools, dict)
        assert "read_file" in tools
        assert "grep" in tools
        assert "git_status" in tools
        assert "repo_map" in tools

        # Each tool should be callable
        for name, func in tools.items():
            assert callable(func), f"Tool '{name}' is not callable"

    async def test_create_tool_bridge_factory(self) -> None:
        """create_tool_bridge factory initializes and returns a bridge."""
        from memfun_agent.tool_bridge import MCPToolBridge, create_tool_bridge

        # Pass None gateway so it skips real MCP discovery
        bridge = await create_tool_bridge(gateway=None)
        assert isinstance(bridge, MCPToolBridge)


# ── 8. Local REPL Tests ───────────────────────────────────────


class TestLocalREPLIntegration:
    """Integration tests for the LocalREPL used by the RLM module."""

    def test_repl_execute_and_namespace_persistence(self) -> None:
        """Variables set in one execute() call persist to the next."""
        from memfun_agent.rlm import LocalREPL

        repl = LocalREPL()
        r1 = repl.execute("x = 42")
        assert r1.success is True

        r2 = repl.execute("print(x * 2)")
        assert r2.success is True
        assert r2.stdout.strip() == "84"

    def test_repl_captures_error(self) -> None:
        """Errors in the REPL are captured in stderr."""
        from memfun_agent.rlm import LocalREPL

        repl = LocalREPL()
        result = repl.execute("raise ValueError('test error')")
        assert result.success is False
        assert "ValueError" in result.stderr
        assert "test error" in result.stderr

    def test_repl_context_metadata_helper(self) -> None:
        """build_context_metadata produces usable metadata."""
        from memfun_agent.rlm import build_context_metadata

        meta = build_context_metadata("hello world" * 100)
        assert "type: str" in meta
        assert "length:" in meta
        assert "preview" in meta
