"""Tests for per-role models (``[llm.roles.<role>]``).

Covers the config resolver, the shared LiteLLM mapping, the per-role LM
factory/cache, task-local ``dspy.context`` binding under
``asyncio.gather``, the role each agent boundary binds (worker / auditor /
decomposer / reviewer / orchestrator), and the ``model`` + ``role`` fields
on every ``llm.call`` payload.

All hermetic: fake LMs (a ``DummyLM`` subclass with a configurable model
id) and an in-memory fake event bus.  No network, no wall-clock asserts.
"""

# DSPy is untyped; strict-mode "unknown type" noise from its API is expected.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from memfun_agent import role_models
from memfun_agent.llm_telemetry import (
    DEFAULT_ROLE,
    UNKNOWN_MODEL,
    agent_context,
    current_role,
    make_dspy_callback,
    publish_llm_call_event,
    set_active_runtime,
)
from memfun_agent.role_models import (
    lm_for_role,
    lm_settings,
    role_for_agent,
    role_scope,
    set_role_config,
)
from memfun_core.config import LLM_ROLES, LLMConfig, MemfunConfig
from memfun_core.errors import ConfigError
from memfun_core.types import TaskMessage

# ── Fixtures / fakes ─────────────────────────────────────────────────

SWARM_TOML = """
[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.2
max_tokens = 4096

[llm.roles.orchestrator]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[llm.roles.decomposer]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[llm.roles.worker]
provider = "openai"
model = "qwen3.8-27b"
base_url = "http://127.0.0.1:18000/v1"
api_key_env = "MEMFUN_WORKER_API_KEY"

[llm.roles.auditor]
provider = "openai"
model = "qwen3.8-27b-auditor"
base_url = "http://127.0.0.1:18000/v1"
api_key_env = "MEMFUN_WORKER_API_KEY"
"""


def _cfg(toml: str, tmp_path: Path) -> MemfunConfig:
    path = tmp_path / "memfun.toml"
    path.write_text(toml)
    return MemfunConfig.from_toml(path)


class _NamedLM(DummyLM):
    """DummyLM with a configurable model id; counts calls via its history.

    No ``forward`` override: DSPy 3.4 routes DummyLM through its
    DummyEngine (the legacy ``forward`` hook is deprecated, removed in
    3.5) and records a normalized response whose ``model`` is always
    ``"dummy"`` -- the LM's own ``model`` is what telemetry must report.
    """

    def __init__(self, model: str) -> None:
        super().__init__({"": {"answer": f"from {model}"}})
        self.model = model
        self.cache = False

    @property
    def calls(self) -> int:
        return len(self.history)


class _FakeEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str:
        self.published.append((topic, message))
        return "ok"


@dataclass
class _FakeRuntime:
    event_bus: _FakeEventBus


@pytest.fixture(autouse=True)
def isolate_role_state():
    role_models.clear_lm_cache()
    set_role_config(None)
    yield
    role_models.clear_lm_cache()
    set_role_config(None)
    set_active_runtime(None)


@pytest.fixture
def fake_lms(monkeypatch: pytest.MonkeyPatch) -> dict[str, _NamedLM]:
    """Make build_lm return one _NamedLM per resolved model id."""
    built: dict[str, _NamedLM] = {}

    def _fake_build(llm: LLMConfig) -> _NamedLM:
        lm = _NamedLM(llm.model)
        built[llm.model] = lm
        return lm

    monkeypatch.setattr(role_models, "build_lm", _fake_build)
    return built


def _events(bus: _FakeEventBus) -> list[dict[str, Any]]:
    return [json.loads(raw) for _, raw in bus.published]


# ── Config resolver ──────────────────────────────────────────────────


def test_no_roles_section_means_every_role_is_llm(tmp_path: Path) -> None:
    cfg = _cfg('[llm]\nprovider = "openai"\nmodel = "gpt-x"\n', tmp_path)
    assert cfg.llm.roles == {}
    for role in LLM_ROLES:
        assert cfg.llm_for_role(role) == cfg.llm
        assert not cfg.has_role_override(role)


def test_default_config_has_no_roles() -> None:
    cfg = MemfunConfig()
    assert cfg.llm.roles == {}
    assert cfg.llm_for_role("worker") == LLMConfig()


def test_resolver_falls_back_field_by_field(tmp_path: Path) -> None:
    cfg = _cfg(
        '[llm]\nprovider = "anthropic"\nmodel = "base"\ntemperature = 0.3\n'
        "max_tokens = 1000\n"
        '[llm.roles.worker]\nmodel = "w"\n'
        "[llm.roles.auditor]\ntemperature = 0.9\nmax_tokens = 50\n",
        tmp_path,
    )
    worker = cfg.llm_for_role("worker")
    assert worker.model == "w"
    assert worker.provider == "anthropic"  # inherited
    assert worker.temperature == 0.3  # inherited
    assert worker.max_tokens == 1000  # inherited
    assert worker.api_key_env == "ANTHROPIC_API_KEY"  # inherited default

    auditor = cfg.llm_for_role("auditor")
    assert auditor.model == "base"
    assert auditor.temperature == 0.9
    assert auditor.max_tokens == 50

    # Unconfigured role == [llm]; resolved configs carry no roles.
    assert cfg.llm_for_role("reviewer").model == "base"
    assert worker.roles == {}
    hash(worker)  # resolved config is hashable


def test_swarm_example_resolves(tmp_path: Path) -> None:
    cfg = _cfg(SWARM_TOML, tmp_path)
    orch = cfg.llm_for_role("orchestrator")
    assert (orch.provider, orch.model, orch.base_url) == ("anthropic", "claude-opus-5-5", None)
    assert orch.temperature == 0.2  # inherited from [llm]
    worker = cfg.llm_for_role("worker")
    assert worker.provider == "openai"
    assert worker.base_url == "http://127.0.0.1:18000/v1"
    assert worker.api_key_env == "MEMFUN_WORKER_API_KEY"
    assert cfg.llm_for_role("reviewer") == cfg.llm_for_role("reviewer")
    assert cfg.llm_for_role("reviewer").model == "claude-sonnet-4-6"


def test_empty_base_url_clears_inherited(tmp_path: Path) -> None:
    cfg = _cfg(
        '[llm]\nprovider = "openai"\nbase_url = "http://127.0.0.1:18000/v1"\n'
        '[llm.roles.orchestrator]\nprovider = "anthropic"\nbase_url = ""\n',
        tmp_path,
    )
    assert cfg.llm_for_role("orchestrator").base_url is None
    assert cfg.llm_for_role("worker").base_url == "http://127.0.0.1:18000/v1"


def test_unknown_role_in_toml_is_dropped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        cfg = _cfg('[llm.roles.wrker]\nmodel = "x"\n', tmp_path)
    assert cfg.llm.roles == {}
    assert any("wrker" in r.getMessage() for r in caplog.records)


def test_unknown_key_in_role_is_dropped(tmp_path: Path) -> None:
    cfg = _cfg('[llm.roles.worker]\nmodel = "x"\ncolour = "blue"\n', tmp_path)
    assert cfg.llm_for_role("worker").model == "x"


def test_unknown_role_lookup_raises() -> None:
    with pytest.raises(ConfigError):
        MemfunConfig().llm_for_role("janitor")
    with pytest.raises(ConfigError), role_scope("janitor"):
        pass


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("model = 5", "must be a string"),
        ('model = ""', "must not be empty"),
        ('temperature = "hot"', "must be a number"),
        ("max_tokens = 0", "positive integer"),
        ("max_tokens = 1.5", "positive integer"),
        ('api_key_env = "sk-ant-not-a-name"', "NAME of an environment variable"),
    ],
)
def test_invalid_role_values_rejected(tmp_path: Path, body: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment) as excinfo:
        _cfg(f"[llm.roles.worker]\n{body}\n", tmp_path)
    # Never echo a value that might be a secret.
    assert "sk-ant" not in str(excinfo.value)


def test_global_and_project_roles_merge_per_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".memfun").mkdir(parents=True)
    (home / ".memfun" / "config.toml").write_text(
        '[llm.roles.worker]\nbase_url = "http://127.0.0.1:18000/v1"\nmodel = "g"\n'
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "memfun.toml").write_text('[llm.roles.worker]\nmodel = "p"\n')
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    worker = MemfunConfig.load(proj).llm_for_role("worker")
    assert worker.model == "p"
    assert worker.base_url == "http://127.0.0.1:18000/v1"


def test_repo_memfun_toml_still_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = MemfunConfig.from_toml(root / "memfun.toml")
    assert cfg.llm.provider == "anthropic"


# ── Provider mapping ─────────────────────────────────────────────────


def test_lm_settings_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    name, kw = lm_settings(LLMConfig(provider="anthropic", model="claude-opus-5-5"))
    assert name == "anthropic/claude-opus-5-5"
    assert kw["api_key"] == "test-anthropic"
    assert "api_base" not in kw
    assert kw["timeout"] == 3600


def test_lm_settings_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMFUN_WORKER_API_KEY", raising=False)
    llm = LLMConfig(
        provider="openai",
        model="qwen3.8-27b",
        base_url="http://127.0.0.1:18000/v1",
        api_key_env="MEMFUN_WORKER_API_KEY",
    )
    name, kw = lm_settings(llm)
    assert name == "openai/qwen3.8-27b"
    assert kw["api_base"] == "http://127.0.0.1:18000/v1"
    # Keyless self-hosted endpoint gets a placeholder, not a failure.
    assert kw["api_key"] == role_models.LOCAL_PLACEHOLDER_KEY


def test_lm_settings_other_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    assert lm_settings(LLMConfig(provider="ollama", model="m"))[0] == "ollama_chat/m"
    assert lm_settings(LLMConfig(provider="openai-compat", model="m"))[0] == "openai/m"
    # Hosted openai without a key and without base_url: no placeholder.
    _, kw = lm_settings(LLMConfig(provider="openai", model="m", api_key_env="NOPE_KEY"))
    assert "api_key" not in kw


def test_build_lm_constructs_dspy_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    lm = role_models.build_lm(LLMConfig(provider="anthropic", model="claude-opus-5-5"))
    assert isinstance(lm, dspy.LM)
    assert lm.model == "anthropic/claude-opus-5-5"


# ── Factory cache ────────────────────────────────────────────────────


def test_lm_for_role_caches_per_resolved_config(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    cfg = _cfg(SWARM_TOML, tmp_path)
    # orchestrator and decomposer resolve identically -> one LM.
    assert lm_for_role(cfg, "orchestrator") is lm_for_role(cfg, "decomposer")
    assert lm_for_role(cfg, "worker") is not lm_for_role(cfg, "auditor")
    assert lm_for_role(cfg, "worker") is lm_for_role(cfg, "worker")
    assert sorted(fake_lms) == ["claude-opus-5-5", "qwen3.8-27b", "qwen3.8-27b-auditor"]


# ── role_scope binding ───────────────────────────────────────────────


def test_role_scope_without_roles_keeps_global_lm(tmp_path: Path) -> None:
    global_lm = _NamedLM("global")
    set_role_config(_cfg('[llm]\nmodel = "x"\n', tmp_path))
    with dspy.context(lm=global_lm), role_scope("worker"):
        assert dspy.settings.lm is global_lm
        assert current_role() == "worker"
    assert current_role() == DEFAULT_ROLE


def test_role_scope_without_registered_config_is_telemetry_only() -> None:
    global_lm = _NamedLM("global")
    with dspy.context(lm=global_lm), role_scope("auditor"):
        assert dspy.settings.lm is global_lm
        assert current_role() == "auditor"


def test_role_scope_overrides_and_restores(tmp_path: Path, fake_lms: dict[str, _NamedLM]) -> None:
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    global_lm = _NamedLM("global")
    with dspy.context(lm=global_lm):
        with role_scope("worker"):
            assert dspy.settings.lm is fake_lms["qwen3.8-27b"]
            with role_scope("decomposer"):
                assert dspy.settings.lm is fake_lms["claude-opus-5-5"]
                assert current_role() == "decomposer"
            assert dspy.settings.lm is fake_lms["qwen3.8-27b"]
            assert current_role() == "worker"
        # Role without a table binds the [llm] LM explicitly (not the
        # enclosing context's LM) once any role table exists.
        with role_scope("reviewer"):
            assert dspy.settings.lm is fake_lms["claude-sonnet-4-6"]
        assert dspy.settings.lm is global_lm


def test_role_scope_build_failure_binds_llm_not_enclosing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """C1: a role whose LM can't be built runs on [llm], never on the
    enclosing (orchestrator) LM, and the WARNING names the role."""
    built: dict[str, _NamedLM] = {}

    def _build(llm: LLMConfig) -> _NamedLM:
        if llm.model == "qwen3.8-27b":
            raise RuntimeError("cannot build")
        built.setdefault(llm.model, _NamedLM(llm.model))
        return built[llm.model]

    monkeypatch.setattr(role_models, "build_lm", _build)
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    with caplog.at_level(logging.WARNING), role_scope("orchestrator"):
        assert dspy.settings.lm is built["claude-opus-5-5"]
        with role_scope("worker"):
            assert dspy.settings.lm is built["claude-sonnet-4-6"]  # [llm]
            assert current_role() == "worker"
    assert any("'worker'" in r.getMessage() for r in caplog.records)


def test_role_scope_fails_closed_when_llm_lm_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(llm: LLMConfig) -> Any:
        raise RuntimeError("cannot build")

    monkeypatch.setattr(role_models, "build_lm", _boom)
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    global_lm = _NamedLM("global")
    with dspy.context(lm=global_lm), pytest.raises(RuntimeError), role_scope("worker"):
        pass


async def test_concurrent_roles_are_isolated_under_gather(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    """Interleaved tasks with different roles each keep their own LM."""
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    roles = ["worker", "auditor", "decomposer", "worker", "auditor"]
    expected_model = {
        "worker": "qwen3.8-27b",
        "auditor": "qwen3.8-27b-auditor",
        "decomposer": "claude-opus-5-5",
    }
    all_entered = asyncio.Barrier(len(roles))
    seen: list[tuple[str, str, str, str]] = []

    async def _run(role: str) -> None:
        with role_scope(role):
            # Everyone is inside its own scope before anyone calls an LM.
            await all_entered.wait()
            predictor = dspy.Predict("question -> answer")
            # Calls run in worker threads (like RLM/decomposer do).
            out = await asyncio.to_thread(predictor, question=role)
            await asyncio.sleep(0)
            lm = dspy.settings.lm
            seen.append((role, lm.model, str(out.answer), current_role()))

    await asyncio.gather(*(_run(r) for r in roles))

    assert len(seen) == len(roles)
    for role, model, answer, bound_role in seen:
        assert model == expected_model[role]
        assert answer == f"from {expected_model[role]}"
        assert bound_role == role
    assert fake_lms["qwen3.8-27b"].calls == 2
    assert fake_lms["qwen3.8-27b-auditor"].calls == 2
    assert fake_lms["claude-opus-5-5"].calls == 1


# ── Agent boundaries ─────────────────────────────────────────────────


def test_role_for_agent_mapping() -> None:
    assert role_for_agent("review-agent") == "auditor"
    assert role_for_agent("pr-reviewer") == "reviewer"
    assert role_for_agent("coder-agent") == "worker"
    assert role_for_agent("some-defined-agent", default=None) is None


def _task(tid: str) -> TaskMessage:
    return TaskMessage(task_id=tid, agent_id="x", payload={"query": tid, "workflow_id": "wf"})


async def test_specialists_bind_worker_and_auditor_roles(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    from memfun_agent.specialists import CoderAgent, ReviewAgent, TestAgent
    from memfun_core.types import TaskResult

    set_role_config(_cfg(SWARM_TOML, tmp_path))
    agents = [CoderAgent(MagicMock()), ReviewAgent(MagicMock()), TestAgent(MagicMock())]
    assert [a.llm_role for a in agents] == ["worker", "auditor", "worker"]

    barrier = asyncio.Barrier(len(agents))
    observed: dict[str, tuple[str, str, str | None]] = {}

    def _make_impl(agent: Any) -> Any:
        async def _impl(task: TaskMessage) -> TaskResult:
            await barrier.wait()
            pred = await asyncio.to_thread(dspy.Predict("question -> answer"), question="q")
            observed[agent.agent_id] = (dspy.settings.lm.model, str(pred.answer), current_role())
            return TaskResult(task_id=task.task_id, agent_id=agent.agent_id, success=True)

        return _impl

    for a in agents:
        a._handle_impl = _make_impl(a)  # type: ignore[method-assign]

    results = await asyncio.gather(*(a.handle(_task(a.agent_id)) for a in agents))
    assert all(r.success for r in results)
    assert observed["coder-agent"][:2] == ("qwen3.8-27b", "from qwen3.8-27b")
    assert observed["test-agent"][0] == "qwen3.8-27b"
    assert observed["review-agent"][:2] == (
        "qwen3.8-27b-auditor",
        "from qwen3.8-27b-auditor",
    )
    assert observed["coder-agent"][2] == "worker"
    assert observed["review-agent"][2] == "auditor"
    assert fake_lms["qwen3.8-27b"].calls == 2
    assert fake_lms["qwen3.8-27b-auditor"].calls == 1


async def test_decomposer_uses_decomposer_role_including_recursion(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    from memfun_agent.decomposer import DecompositionConfig, TaskDecomposer

    set_role_config(_cfg(SWARM_TOML, tmp_path))
    dec = TaskDecomposer(DecompositionConfig(enable_hierarchical=False))
    seen: list[tuple[Any, str]] = []

    def _fake_predict(**kwargs: Any) -> Any:
        # Runs in the to_thread worker: the task-local override must be here.
        seen.append((dspy.settings.lm, current_role()))
        return dspy.Prediction(sub_tasks=[], shared_spec="", parallelism_groups=[])

    dec.decomposer = _fake_predict  # type: ignore[assignment]

    global_lm = _NamedLM("global")
    with dspy.context(lm=global_lm), role_scope("worker"):
        await dec.adecompose("build it", "ctx")
        await dec.adecompose("build it again", "ctx", _depth=1)  # recursion entry
        # Back in the caller's scope after the decomposer returns.
        assert dspy.settings.lm is fake_lms["qwen3.8-27b"]

    assert len(seen) == 2
    for lm, role in seen:
        assert lm is fake_lms["claude-opus-5-5"]
        assert role == "decomposer"


async def test_pr_reviewer_defined_agent_binds_reviewer_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_lms: dict[str, _NamedLM]
) -> None:
    from memfun_agent.definitions.defined_agent import DefinedAgent
    from memfun_agent.definitions.types import AgentDefinition

    toml = SWARM_TOML + '\n[llm.roles.reviewer]\nmodel = "claude-opus-5-5-review"\n'
    set_role_config(_cfg(toml, tmp_path))
    seen: list[tuple[str, str]] = []

    agent = DefinedAgent(MagicMock(), AgentDefinition(name="pr-reviewer", description="d"))
    other = DefinedAgent(MagicMock(), AgentDefinition(name="custom", description="d"))

    def _spy(self: DefinedAgent, task: TaskMessage) -> str:
        seen.append((dspy.settings.lm.model, current_role()))
        return "prompt"

    monkeypatch.setattr(DefinedAgent, "_build_prompt", _spy)
    global_lm = _NamedLM("global")
    with dspy.context(lm=global_lm):
        await agent.handle(_task("pr"))
        await other.handle(_task("other"))
    assert seen == [("claude-opus-5-5-review", "reviewer"), ("global", DEFAULT_ROLE)]


def test_orchestrator_role_resolves(tmp_path: Path, fake_lms: dict[str, _NamedLM]) -> None:
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    with role_scope("orchestrator"):
        assert dspy.settings.lm is fake_lms["claude-opus-5-5"]
        assert current_role() == "orchestrator"


# ── Telemetry: every llm.call has model + role ───────────────────────


async def test_llm_call_events_carry_model_and_role(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    global_lm = _NamedLM("global-model")

    def _call() -> None:
        dspy.Predict("question -> answer")(question="hi")

    with dspy.context(lm=global_lm, callbacks=[make_dspy_callback()]):
        _call()  # no role bound
        with agent_context(agent_name="coder-agent", task_id="T1"), role_scope("worker"):
            await asyncio.to_thread(_call)
        with role_scope("decomposer"):
            _call()
    for _ in range(10):  # let fire-and-forget publishes run (no wall clock)
        await asyncio.sleep(0)

    events = [e for e in _events(bus) if e["event_type"] == "llm.call"]
    assert len(events) == 3
    for ev in events:
        assert ev["model"]
        assert ev["role"]
    # Publish order depends on loop scheduling (thread calls publish
    # synchronously, loop calls via create_task), so key by role.
    by_role = {ev["role"]: ev for ev in events}
    assert by_role[DEFAULT_ROLE]["model"] == "global-model"
    assert by_role["worker"]["model"] == "qwen3.8-27b"
    assert by_role["worker"]["agent_name"] == "coder-agent"
    assert by_role["decomposer"]["model"] == "claude-opus-5-5"


def test_direct_publish_fills_missing_model_and_role() -> None:
    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]
    publish_llm_call_event({"event_type": "llm.call", "prompt_preview": "p"})
    with role_scope("auditor"):
        publish_llm_call_event({"event_type": "llm.call", "model": "m1"})
    evs = _events(bus)
    assert (evs[0]["model"], evs[0]["role"]) == (UNKNOWN_MODEL, DEFAULT_ROLE)
    assert (evs[1]["model"], evs[1]["role"]) == ("m1", "auditor")


def test_callback_reads_model_from_lm_when_history_empty() -> None:
    from memfun_agent.llm_telemetry import MemfunDSPyCallback

    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]

    class _BareLM:
        model = "openai/qwen3.8-27b"
        history: ClassVar[list[Any]] = []

    cb = MemfunDSPyCallback()
    with role_scope("worker"):
        cb.on_lm_start("c1", _BareLM(), {"prompt": "x"})
        cb.on_lm_end("c1", ["out"])
    cb.on_lm_start("c2", object(), {"prompt": "y"})  # LM with no model at all
    cb.on_lm_end("c2", ["out"])
    evs = _events(bus)
    assert (evs[0]["model"], evs[0]["role"]) == ("openai/qwen3.8-27b", "worker")
    assert (evs[1]["model"], evs[1]["role"]) == (UNKNOWN_MODEL, DEFAULT_ROLE)


def test_distributed_event_roundtrips_role() -> None:
    from memfun_runtime.distributed import DistributedEvent

    ev = DistributedEvent(event_type="llm.call", model="m", role="worker")
    back = DistributedEvent.from_bytes(ev.to_bytes())
    assert (back.model, back.role) == ("m", "worker")
    # Old payloads without a role still parse.
    legacy = json.dumps({"event_type": "llm.call", "model": "m"}).encode()
    assert DistributedEvent.from_bytes(legacy).role is None


# ── C1: partial config never leaks the enclosing role's model ────────

PARTIAL_TOML = """
[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"

[llm.roles.orchestrator]
model = "claude-opus-5-5"
"""


class _FakeManager:
    """AgentOrchestrator only needs is_running / get_agent."""

    def __init__(self, agents: dict[str, Any]) -> None:
        self._agents = agents

    def is_running(self, name: str) -> bool:
        return name in self._agents

    def get_agent(self, name: str) -> Any:
        return self._agents[name]

    async def start_agent(self, name: str) -> str:
        return name


def _orchestrator(agents: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    from memfun_runtime.backends.memory import InProcessEventBus
    from memfun_runtime.orchestrator import AgentOrchestrator

    return AgentOrchestrator(
        context=SimpleNamespace(event_bus=InProcessEventBus()),  # type: ignore[arg-type]
        manager=_FakeManager(agents),  # type: ignore[arg-type]
    )


def test_partial_config_nested_scopes_use_llm_lm(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    set_role_config(_cfg(PARTIAL_TOML, tmp_path))
    with role_scope("orchestrator"):
        assert dspy.settings.lm is fake_lms["claude-opus-5-5"]
        for role in ("worker", "decomposer", "auditor", "reviewer"):
            with role_scope(role):
                assert dspy.settings.lm is fake_lms["claude-sonnet-4-6"], role
                assert current_role() == role


async def test_partial_config_dispatch_inside_orchestrator_uses_llm_lm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_lms: dict[str, _NamedLM]
) -> None:
    """Agents dispatched in-process from the orchestrator scope (context is
    copied by asyncio.wait_for) run on [llm], not the orchestrator model."""
    from memfun_agent.decomposer import DecompositionConfig, TaskDecomposer
    from memfun_agent.definitions.defined_agent import DefinedAgent
    from memfun_agent.definitions.types import AgentDefinition
    from memfun_agent.specialists import CoderAgent, ReviewAgent
    from memfun_core.types import TaskResult

    set_role_config(_cfg(PARTIAL_TOML, tmp_path))
    seen: dict[str, tuple[str, str]] = {}

    def _record(name: str) -> None:
        pred = dspy.Predict("question -> answer")(question=name)
        seen[name] = (str(pred.answer), current_role())

    coder, review = CoderAgent(MagicMock()), ReviewAgent(MagicMock())
    for agent in (coder, review):

        def _mk(a: Any) -> Any:
            async def _impl(task: TaskMessage) -> TaskResult:
                await asyncio.to_thread(_record, a.agent_id)
                return TaskResult(task_id=task.task_id, agent_id=a.agent_id, success=True)

            return _impl

        agent._handle_impl = _mk(agent)  # type: ignore[method-assign]

    pr = DefinedAgent(MagicMock(), AgentDefinition(name="pr-reviewer", description="d"))

    def _spy_prompt(self: DefinedAgent, task: TaskMessage) -> str:
        _record("pr-reviewer")
        return "p"

    monkeypatch.setattr(DefinedAgent, "_build_prompt", _spy_prompt)
    orch = _orchestrator({"coder-agent": coder, "review-agent": review, "pr-reviewer": pr})

    dec = TaskDecomposer(DecompositionConfig(enable_hierarchical=False))

    def _fake_predict(**kwargs: Any) -> Any:
        _record("decomposer")
        return dspy.Prediction(sub_tasks=[], shared_spec="", parallelism_groups=[])

    dec.decomposer = _fake_predict  # type: ignore[assignment]

    with role_scope("orchestrator"):
        await dec.adecompose("x", "ctx")
        for name in ("coder-agent", "review-agent", "pr-reviewer"):
            await orch.dispatch(_task(name), name)
        _record("orchestrator")

    base = "from claude-sonnet-4-6"
    assert seen == {
        "decomposer": (base, "decomposer"),
        "coder-agent": (base, "worker"),
        "review-agent": (base, "auditor"),
        "pr-reviewer": (base, "reviewer"),
        "orchestrator": ("from claude-opus-5-5", "orchestrator"),
    }


# ── C2: single-task workflow path runs the coder on the worker role ──


class _RecordingCoder:
    """Stands in for RLMCodingAgent (which binds no role itself)."""

    agent_id = "rlm-coder"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    async def handle(self, task: TaskMessage) -> Any:
        from memfun_core.types import TaskResult

        def _call() -> str:
            return str(dspy.Predict("question -> answer")(question="code").answer)

        answer = await asyncio.to_thread(_call)
        self.seen.append((answer, current_role()))
        return TaskResult(
            task_id=task.task_id, agent_id="rlm-coder", success=True, result={"answer": answer}
        )


async def test_single_task_workflow_dispatch_uses_worker_role(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from memfun_agent.decomposer import DecompositionResult, SubTask
    from memfun_agent.workflow import WorkflowEngine

    set_role_config(_cfg(SWARM_TOML, tmp_path))
    coder = _RecordingCoder()
    orch = _orchestrator({"rlm-coder": coder})
    state_store = AsyncMock()
    state_store.get.return_value = None
    ctx = SimpleNamespace(state_store=state_store, event_bus=orch._context.event_bus)
    engine = WorkflowEngine(ctx, orch, _FakeManager({"rlm-coder": coder}))  # type: ignore[arg-type]
    single = DecompositionResult(
        sub_tasks=[SubTask(id="T1", description="Fix", agent_type="coder")],
        shared_spec="",
        parallelism_groups=[["T1"]],
        is_single_task=True,
    )

    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]
    with (
        dspy.context(callbacks=[make_dspy_callback()]),
        patch.object(engine._decomposer, "adecompose", return_value=single),  # pyright: ignore[reportPrivateUsage]
        role_scope("orchestrator"),
    ):
        result = await engine.execute_workflow("Fix the bug", "ctx", None)

    assert result.success is True
    assert coder.seen == [("from qwen3.8-27b", "worker")]
    events = [e for e in _events(bus) if e["event_type"] == "llm.call"]
    assert [(e["model"], e["role"]) for e in events] == [("qwen3.8-27b", "worker")]


async def test_direct_coder_path_outside_workflow_stays_default(
    tmp_path: Path, fake_lms: dict[str, _NamedLM]
) -> None:
    """Plain chat / direct RLM dispatch is unchanged: role=default on [llm]."""
    set_role_config(_cfg(SWARM_TOML, tmp_path))
    coder = _RecordingCoder()
    global_lm = _NamedLM("global-llm")
    with dspy.context(lm=global_lm):
        await _orchestrator({"rlm-coder": coder}).dispatch(_task("rlm-coder"), "rlm-coder")
    assert coder.seen == [("from global-llm", DEFAULT_ROLE)]


# ── C3: [llm]'s key never follows a role to another endpoint ─────────


@pytest.mark.parametrize(
    ("role_body", "inherits"),
    [
        ('provider = "openai"', False),  # provider change alone
        ('base_url = "http://127.0.0.1:18000/v1"', False),  # base_url change alone
        ('provider = "openai"\nbase_url = "http://127.0.0.1:18000/v1"', False),  # both
        ('model = "other"', True),  # neither
        ('provider = "anthropic"\nmodel = "m"', True),  # same provider restated
    ],
)
def test_api_key_env_inherited_only_for_same_endpoint(
    tmp_path: Path, role_body: str, inherits: bool
) -> None:
    cfg = _cfg(
        '[llm]\nprovider = "anthropic"\napi_key_env = "ANTHROPIC_API_KEY"\n'
        f"[llm.roles.worker]\n{role_body}\n",
        tmp_path,
    )
    expected = "ANTHROPIC_API_KEY" if inherits else ""
    assert cfg.llm_for_role("worker").api_key_env == expected


def test_explicit_role_api_key_env_is_kept_on_endpoint_change(tmp_path: Path) -> None:
    cfg = _cfg(
        '[llm]\nprovider = "anthropic"\n'
        '[llm.roles.worker]\nprovider = "openai"\napi_key_env = "MEMFUN_WORKER_API_KEY"\n',
        tmp_path,
    )
    assert cfg.llm_for_role("worker").api_key_env == "MEMFUN_WORKER_API_KEY"


@pytest.mark.parametrize(
    "base_url", ["https://api.hosted.example/v1", "http://127.0.0.1:18000/v1"]
)
def test_llm_key_value_never_reaches_other_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    secret = "test-anthropic-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    cfg = _cfg(
        '[llm]\nprovider = "anthropic"\napi_key_env = "ANTHROPIC_API_KEY"\n'
        f'[llm.roles.worker]\nprovider = "openai"\nmodel = "q"\nbase_url = "{base_url}"\n',
        tmp_path,
    )
    _, kw = lm_settings(cfg.llm_for_role("worker"))
    assert kw.get("api_key") != secret
    lm = role_models.build_lm(cfg.llm_for_role("worker"))
    assert secret not in repr(lm.kwargs)
    # The [llm] LM itself still gets its key.
    assert lm_settings(cfg.llm)[1]["api_key"] == secret


# ── A2: placeholder key only for local hosts + startup preflight ─────


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:18000/v1", True),
        ("http://LOCALHOST/v1", True),
        ("http://127.0.0.1:18000/v1", True),
        ("http://127.5.6.7/v1", True),
        ("http://[::1]:18000/v1", True),
        ("http://10.1.2.3/v1", True),
        ("http://172.16.0.1/v1", True),
        ("http://172.31.255.255/v1", True),
        ("http://192.168.1.10/v1", True),
        ("http://169.254.1.1/v1", True),
        ("http://gpu-box.local:8000/v1", True),
        ("http://172.32.0.1/v1", False),
        ("https://api.openai.com/v1", False),
        ("https://openrouter.ai/api/v1", False),
        ("http://127.0.0.1.evil.com/v1", False),
        ("http://localhost.evil.com/v1", False),
        ("http://127.0.0.1@evil.com/v1", False),
        ("http://evil.com/127.0.0.1", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_local_base_url(url: str, local: bool) -> None:
    assert role_models.is_local_base_url(url) is local


def test_hosted_base_url_without_key_gets_no_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    hosted = LLMConfig(
        provider="openai", model="m", base_url="https://api.openai.com/v1", api_key_env="NOPE_KEY"
    )
    assert "api_key" not in lm_settings(hosted)[1]
    tricky = LLMConfig(
        provider="openai", model="m", base_url="http://127.0.0.1@evil.com/v1", api_key_env="NOPE"
    )
    assert "api_key" not in lm_settings(tricky)[1]


def test_preflight_warns_per_broken_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_lms: dict[str, _NamedLM],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.delenv("MEMFUN_WORKER_API_KEY", raising=False)
    cfg = _cfg(
        '[llm]\nprovider = "anthropic"\n'
        '[llm.roles.orchestrator]\nmodel = "claude-opus-5-5"\n'
        '[llm.roles.worker]\nprovider = "openai"\nmodel = "q"\n'
        'base_url = "http://127.0.0.1:18000/v1"\n'
        '[llm.roles.auditor]\nprovider = "openai"\nmodel = "q"\n'
        'base_url = "https://api.hosted.example/v1"\n',
        tmp_path,
    )
    with caplog.at_level(logging.WARNING):
        set_role_config(cfg)
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1
    assert "'auditor'" in msgs[0]
    assert role_models.preflight_roles(cfg) == ["auditor"]


# ── A5: api_key_env must be a whole env var name ─────────────────────


@pytest.mark.parametrize("value", ["TRAILING\n", "A B", "1ABC", "X;rm"])
def test_api_key_env_rejects_non_names(tmp_path: Path, value: str) -> None:
    path = tmp_path / "memfun.toml"
    path.write_text(f"[llm.roles.worker]\napi_key_env = {json.dumps(value)}\n")
    with pytest.raises(ConfigError):
        MemfunConfig.from_toml(path)


def test_api_key_env_empty_means_no_key(tmp_path: Path) -> None:
    cfg = _cfg('[llm.roles.worker]\napi_key_env = ""\n', tmp_path)
    assert cfg.llm_for_role("worker").api_key_env == ""


# ── Custom (model-routed) providers, URL redaction, provider spelling ─


@pytest.mark.parametrize(
    ("llm_model", "role_model", "inherits"),
    [
        ("openrouter/x", "together_ai/y", False),  # route prefix changes
        ("openrouter/x", "y", False),  # prefix dropped -> LiteLLM infers route
        ("openrouter/x", "openrouter/z", True),  # same route
        ("openrouter/x", "openrouter/x", True),  # same model
        ("gpt-4o", "claude-opus-5-5", False),  # bare names: any change differs
    ],
)
def test_custom_provider_model_route_change_drops_key(
    tmp_path: Path, llm_model: str, role_model: str, inherits: bool
) -> None:
    cfg = _cfg(
        f'[llm]\nprovider = "custom"\nmodel = "{llm_model}"\napi_key_env = "MAIN_KEY"\n'
        f'[llm.roles.auditor]\nmodel = "{role_model}"\n',
        tmp_path,
    )
    assert cfg.llm_for_role("auditor").api_key_env == ("MAIN_KEY" if inherits else "")


def test_custom_route_change_never_sends_main_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAIN_KEY", "test-main-secret")
    cfg = _cfg(
        '[llm]\nprovider = "custom"\nmodel = "openrouter/x"\napi_key_env = "MAIN_KEY"\n'
        '[llm.roles.auditor]\nmodel = "together_ai/y"\n',
        tmp_path,
    )
    assert lm_settings(cfg.llm_for_role("auditor"))[1].get("api_key") != "test-main-secret"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-compat"])
def test_known_provider_model_only_change_still_inherits(tmp_path: Path, provider: str) -> None:
    cfg = _cfg(
        f'[llm]\nprovider = "{provider}"\nmodel = "a"\napi_key_env = "MAIN_KEY"\n'
        '[llm.roles.worker]\nmodel = "other/b"\n',
        tmp_path,
    )
    assert cfg.llm_for_role("worker").api_key_env == "MAIN_KEY"


def test_redact_url_strips_userinfo_query_fragment() -> None:
    url = "https://user:URLSECRET@hosted.example:8443/v1?api_key=QSECRET#frag"
    assert role_models.redact_url(url) == "https://hosted.example:8443/v1"
    assert role_models.redact_url("http://[::1]:18000/v1?k=s") == "http://[::1]:18000/v1"
    assert role_models.redact_url(None) == ""


def test_preflight_log_redacts_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    cfg = _cfg(
        '[llm.roles.worker]\nprovider = "openai"\nmodel = "q"\napi_key_env = "NOPE_KEY"\n'
        'base_url = "https://user:URLSECRET@hosted.example/v1?api_key=QSECRET"\n',
        tmp_path,
    )
    with caplog.at_level(logging.WARNING):
        assert role_models.preflight_roles(cfg) == ["worker"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "https://hosted.example/v1" in text
    assert "URLSECRET" not in text
    assert "QSECRET" not in text
    assert "user:" not in text


def test_provider_spelling_is_normalized(tmp_path: Path) -> None:
    cfg = _cfg(
        '[llm]\nprovider = " Anthropic "\n'
        '[llm.roles.worker]\nprovider = "OpenAI"\nmodel = "q"\n'
        'base_url = "http://127.0.0.1:18000/v1"\n',
        tmp_path,
    )
    assert cfg.llm.provider == "anthropic"
    worker = cfg.llm_for_role("worker")
    assert worker.provider == "openai"
    assert lm_settings(worker)[0] == "openai/q"  # mapped, not passed through raw


def test_unrecognized_provider_warns_with_role(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        cfg = _cfg('[llm.roles.auditor]\nprovider = "opnai"\nmodel = "q"\n', tmp_path)
    assert cfg.llm_for_role("auditor").provider == "opnai"
    msgs = [r.getMessage() for r in caplog.records if "unrecognized provider" in r.getMessage()]
    assert len(msgs) == 1
    assert "[llm.roles.auditor]" in msgs[0]
    assert "'opnai'" in msgs[0]


# ── Real dspy.LM through a stubbed transport (no network) ────────────


@pytest.fixture
def stub_transport(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace DSPy's HTTP layer with a canned OpenAI chat completion.

    DSPy 3.4 sends OpenAI-compatible calls through its vendored lm15
    ``StdlibTransport``; LiteLLM's ``completion`` is stubbed to fail too,
    so no code path can reach the network.  The server echoes a model
    name (``echoed``) that differs from the LM's configured id, like a
    real SGLang/vLLM server reporting its served name or weights path.
    """
    import litellm
    from dspy._vendor.lm15.transports import _sync  # pyright: ignore[reportPrivateUsage]
    from dspy._vendor.lm15.transports._types import TransportResponse

    sent: list[dict[str, Any]] = []

    def _release(consumed: bool) -> None:
        return None

    def _stream(self: Any, request: Any) -> TransportResponse:
        sent.append({"url": request.url, "body": json.loads(request.body or b"{}")})
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "/models/served-weights-path",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
        return TransportResponse(
            status=200,
            reason="OK",
            headers=[("content-type", "application/json"), ("content-length", str(len(body)))],
            http_version="HTTP/1.1",
            chunks=iter([body]),
            release=_release,
        )

    def _no_litellm(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unexpected LiteLLM network path")

    monkeypatch.setattr(_sync.StdlibTransport, "stream", _stream)
    monkeypatch.setattr(litellm, "completion", _no_litellm)
    # Production role LMs keep DSPy's default response cache; disable it
    # here so every call reaches the stub and nothing is written to the
    # user's on-disk DSPy cache.
    from dspy.clients.cache import Cache

    monkeypatch.setattr(
        dspy,
        "cache",
        Cache(enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir=None),
        raising=False,
    )
    return sent


def test_real_dspy_lm_event_model_is_the_lms_model(
    stub_transport: list[dict[str, Any]],
) -> None:
    """Regression guard for DSPy upgrades: the llm.call ``model`` is the
    calling dspy.LM's model string, not the server-echoed name."""
    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]
    lm = dspy.LM(
        "openai/qwen3.8-27b",
        api_base="http://127.0.0.1:18000/v1",
        api_key="sk-local",
        cache=False,
    )
    with dspy.context(lm=lm, callbacks=[make_dspy_callback()]):
        out = dspy.Predict("question -> answer")(question="hi")

    assert str(out.answer) == "ok"
    assert len(stub_transport) == 1  # the stub (not the network) served it
    assert stub_transport[0]["body"]["model"] == "qwen3.8-27b"
    events = [e for e in _events(bus) if e["event_type"] == "llm.call"]
    assert len(events) == 1
    assert events[0]["model"] == lm.model == "openai/qwen3.8-27b"
    assert events[0]["role"] == DEFAULT_ROLE


async def test_real_role_lm_event_carries_role_and_model(
    tmp_path: Path, stub_transport: list[dict[str, Any]]
) -> None:
    """End to end with the real factory: [llm.roles.worker] -> build_lm ->
    dspy.LM -> stub transport -> llm.call {model, role}."""
    bus = _FakeEventBus()
    set_active_runtime(_FakeRuntime(event_bus=bus))  # type: ignore[arg-type]
    set_role_config(_cfg(QWEN_EVERYWHERE_TOML, tmp_path))

    def _call() -> None:
        dspy.Predict("question -> answer")(question="hi")

    with dspy.context(callbacks=[make_dspy_callback()]), role_scope("worker"):
        await asyncio.to_thread(_call)

    events = [e for e in _events(bus) if e["event_type"] == "llm.call"]
    assert [(e["model"], e["role"]) for e in events] == [("openai/qwen3.8-27b", "worker")]
    assert stub_transport[0]["url"] == "http://127.0.0.1:18000/v1/chat/completions"


# ── Shipped example: Qwen everywhere, optional Opus comparison arm ───

QWEN_EVERYWHERE_TOML = """
[llm]
provider = "openai"
model = "qwen3.8-27b"
base_url = "http://127.0.0.1:18000/v1"
api_key_env = "MEMFUN_WORKER_API_KEY"

[llm.roles.worker]
max_tokens = 16384
"""

OPUS_ARM_TOML = (
    QWEN_EVERYWHERE_TOML
    + """
[llm.roles.orchestrator]
provider = "anthropic"
model = "claude-opus-5-5"
base_url = ""
api_key_env = "ANTHROPIC_API_KEY"

[llm.roles.decomposer]
provider = "anthropic"
model = "claude-opus-5-5"
base_url = ""
"""
)


def test_qwen_everywhere_all_roles_resolve_to_one_endpoint(tmp_path: Path) -> None:
    cfg = _cfg(QWEN_EVERYWHERE_TOML, tmp_path)
    for role in LLM_ROLES:
        llm = cfg.llm_for_role(role)
        assert (llm.provider, llm.model, llm.base_url, llm.api_key_env) == (
            "openai",
            "qwen3.8-27b",
            "http://127.0.0.1:18000/v1",
            "MEMFUN_WORKER_API_KEY",
        )
        assert lm_settings(llm)[0] == "openai/qwen3.8-27b"


def test_opus_comparison_arm_over_openai_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An anthropic role over an openai [llm] never inherits the openai
    key env, and clears the tunnel base_url."""
    monkeypatch.setenv("MEMFUN_WORKER_API_KEY", "test-worker-secret")
    cfg = _cfg(OPUS_ARM_TOML, tmp_path)
    orch = cfg.llm_for_role("orchestrator")
    assert (orch.provider, orch.model, orch.base_url) == ("anthropic", "claude-opus-5-5", None)
    assert orch.api_key_env == "ANTHROPIC_API_KEY"
    dec = cfg.llm_for_role("decomposer")
    assert dec.api_key_env == ""  # NOT MEMFUN_WORKER_API_KEY
    assert lm_settings(dec)[1].get("api_key") != "test-worker-secret"
    assert cfg.llm_for_role("worker").model == "qwen3.8-27b"


def test_anthropic_role_over_openai_llm_without_base_url_override(tmp_path: Path) -> None:
    cfg = _cfg(
        '[llm]\nprovider = "openai"\napi_key_env = "OPENAI_KEY"\n'
        '[llm.roles.orchestrator]\nprovider = "anthropic"\nmodel = "claude-opus-5-5"\n',
        tmp_path,
    )
    assert cfg.llm_for_role("orchestrator").api_key_env == ""


def test_no_code_default_references_opus() -> None:
    import memfun_core.config as config_mod

    assert "opus" not in LLMConfig().model.lower()
    assert "claude-opus-5-5" not in Path(config_mod.__file__).read_text()
    assert "claude-opus-5-5" not in Path(role_models.__file__).read_text()


def test_provider_switch_does_not_inherit_tunnel_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An anthropic role over an openai tunnel [llm] must not send its
    (explicit) Anthropic key to the tunnel base_url."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-secret")
    cfg = _cfg(
        QWEN_EVERYWHERE_TOML
        + '[llm.roles.orchestrator]\nprovider = "anthropic"\nmodel = "claude-opus-5-5"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\n',
        tmp_path,
    )
    orch = cfg.llm_for_role("orchestrator")
    assert orch.base_url is None
    name, kw = lm_settings(orch)
    assert name == "anthropic/claude-opus-5-5"
    assert "api_base" not in kw
    assert kw["api_key"] == "test-anthropic-secret"
    # Same provider keeps the inherited base_url.
    assert cfg.llm_for_role("worker").base_url == "http://127.0.0.1:18000/v1"


def test_docs_example_blocks_parse_and_match(tmp_path: Path) -> None:
    """The TOML blocks in docs/configuration.md's roles section parse and
    the default resolves every role to Qwen (no Anthropic)."""
    import re

    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "configuration.md").read_text()
    section = doc[doc.index("#### Default example") : doc.index("### [backend]")]
    blocks = re.findall(r"```toml\n(.*?)```", section, flags=re.S)
    assert len(blocks) == 2
    default = _cfg(blocks[0], tmp_path)
    for role in LLM_ROLES:
        assert default.llm_for_role(role).model == "qwen3.8-27b"
        assert default.llm_for_role(role).provider == "openai"
    arm = _cfg(blocks[0] + blocks[1], tmp_path)
    for role in ("orchestrator", "decomposer"):
        llm = arm.llm_for_role(role)
        assert (llm.provider, llm.model, llm.base_url) == ("anthropic", "claude-opus-5-5", None)
    for role in ("worker", "auditor", "reviewer"):
        assert arm.llm_for_role(role).model == "qwen3.8-27b"
