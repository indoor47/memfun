"""Quality limits: configurable iteration caps, stall stop, truncation as an error."""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from memfun_agent import llm_telemetry as tel
from memfun_agent import role_models
from memfun_agent.decomposer import (
    DecompositionTruncatedError,
    TaskDecomposer,
)
from memfun_agent.rlm import (  # pyright: ignore[reportPrivateUsage]
    RLMConfig,
    RLMModule,
    _made_progress,
)
from memfun_agent.specialists import CoderAgent, FileAgent
from memfun_core.config import LimitsConfig, MemfunConfig
from memfun_core.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

# ── [limits] config ──────────────────────────────────────────────────────────


def test_limits_defaults_and_overrides() -> None:
    assert MemfunConfig._from_raw({}).limits == LimitsConfig()  # pyright: ignore[reportPrivateUsage]
    cfg = MemfunConfig._from_raw(  # pyright: ignore[reportPrivateUsage]
        {"limits": {"stall_steps": 7, "iterations": {"coder-agent": 90, "rlm-coder": 200}}}
    )
    assert cfg.limits.stall_steps == 7
    assert cfg.limits.iterations == {"coder-agent": 90, "rlm-coder": 200}


@pytest.mark.parametrize(
    "raw",
    [
        {"stall_steps": -1},
        {"stall_steps": "12"},
        {"stall_steps": True},
        {"iterations": [1]},
        {"iterations": {"coder-agent": 0}},
        {"iterations": {"coder-agent": "60"}},
    ],
)
def test_limits_invalid_values_are_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        MemfunConfig._from_raw({"limits": raw})  # pyright: ignore[reportPrivateUsage]


def test_iteration_limit_reads_the_registered_config() -> None:
    try:
        role_models.set_role_config(None)
        assert role_models.iteration_limit("coder-agent", 60) == 60
        assert role_models.stall_steps() == 12
        role_models.set_role_config(
            MemfunConfig._from_raw(  # pyright: ignore[reportPrivateUsage]
                {"limits": {"stall_steps": 5, "iterations": {"coder-agent": 99}}}
            )
        )
        assert role_models.iteration_limit("coder-agent", 60) == 99
        assert role_models.iteration_limit("file-agent", 32) == 32
        assert role_models.stall_steps() == 5
    finally:
        role_models.set_role_config(None)


def test_default_caps_are_raised() -> None:
    assert CoderAgent._MAX_ITERATIONS == 60  # was 15
    assert FileAgent._MAX_ITERATIONS == 32  # was 8


# ── stall detection ──────────────────────────────────────────────────────────


def test_made_progress_rules() -> None:
    reads: set[str] = set()
    cmds: set[str] = set()
    assert _made_progress([("read", "a.py", 10)], reads, cmds)  # a new file
    assert not _made_progress([("read", "a.py", 10)], reads, cmds)  # same file again
    assert not _made_progress([], reads, cmds)  # nothing at all
    assert _made_progress([("cmd", "pytest -q", 1)], reads, cmds)
    assert not _made_progress([("cmd", "pytest -q", 1)], reads, cmds)  # same, no edit between
    assert _made_progress([("edit", "a.py", "x")], reads, cmds)
    assert _made_progress([("cmd", "pytest -q", 0)], reads, cmds)  # after an edit it counts again
    assert _made_progress([("grep", "foo", 3)], reads, cmds)
    assert not _made_progress([("grep", "foo", 3)], reads, cmds)


def _module(tmp_path: Path, code_for: Any, **cfg: Any) -> RLMModule:
    module = RLMModule(config=RLMConfig(enable_planning=False, root=str(tmp_path), **cfg))
    counter = itertools.count(1)

    def explorer(**_kw: object) -> Any:
        return types.SimpleNamespace(reasoning="r", next_code=code_for(next(counter)))

    module.explorer = explorer  # type: ignore[assignment]
    return module


def test_rereading_the_same_file_stops_the_agent_with_a_reason(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world")
    module = _module(
        tmp_path,
        lambda _n: "print(read_file('a.txt')[:5])",
        max_iterations=50,
        stall_steps=3,
    )
    result = asyncio.run(module.aforward(query="q", context="ctx"))
    assert result.stop_reason.startswith("stalled: 3 consecutive iterations")
    assert result.iterations == 4  # 1 new read, then 3 idle iterations
    assert result.answer.startswith("[STOPPED: stalled")


def test_progress_resets_the_stall_counter_and_the_cap_is_only_a_backstop(
    tmp_path: Path,
) -> None:
    module = _module(
        tmp_path,
        lambda n: f"write_file('f{n}.txt', 'x')",
        max_iterations=8,
        stall_steps=3,
    )
    result = asyncio.run(module.aforward(query="q", context="ctx"))
    assert result.iterations == 8
    assert result.stop_reason == "max_iterations"


def test_stall_stopping_can_be_disabled(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    module = _module(
        tmp_path,
        lambda _n: "print(read_file('a.txt'))",
        max_iterations=9,
        stall_steps=0,
    )
    result = asyncio.run(module.aforward(query="q", context="ctx"))
    assert result.iterations == 9 and result.stop_reason == "max_iterations"


# ── truncated responses are errors ───────────────────────────────────────────


def _end_call(monkeypatch: pytest.MonkeyPatch, finish_reason: str) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(tel, "publish_llm_call_event", seen.append)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish_reason)])
    instance = types.SimpleNamespace(
        model="openai/x",
        history=[
            {
                "usage": {"prompt_tokens": 10, "completion_tokens": 32768},
                "outputs": ["{partial"],
                "response": response,
            }
        ],
    )
    tel._call_start_state.set(  # pyright: ignore[reportPrivateUsage]
        {"start": time.perf_counter(), "instance": instance, "prompt_preview": "p"}
    )
    tel.MemfunDSPyCallback().on_lm_end("call-1", ["{partial"], None)
    assert len(seen) == 1
    return seen[0]


def test_truncated_response_is_a_failed_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    before = tel.truncation_counts().get("default", 0)
    payload = _end_call(monkeypatch, "length")
    assert payload["success"] is False
    assert payload["error"] == tel.TRUNCATED_ERROR
    assert tel.truncation_counts().get("default", 0) == before + 1


def test_normal_response_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    before = dict(tel.truncation_counts())
    payload = _end_call(monkeypatch, "stop")
    assert payload["success"] is True and payload["error"] is None
    assert tel.truncation_counts() == before


def test_truncated_plan_fails_visibly_instead_of_falling_back() -> None:
    def truncated(**_kw: object) -> Any:
        with tel.role_context("decomposer"):
            tel._record_truncation("decomposer")  # pyright: ignore[reportPrivateUsage]
        raise ValueError("LM response cannot be serialized to a JSON object")

    decomposer = TaskDecomposer()
    decomposer.decomposer = truncated  # type: ignore[assignment]
    with pytest.raises(DecompositionTruncatedError, match="max_tokens"):
        asyncio.run(decomposer.adecompose("build it", "ctx"))


def test_a_non_truncation_failure_still_falls_back_to_a_single_task() -> None:
    def broken(**_kw: object) -> Any:
        raise ValueError("garbage")

    decomposer = TaskDecomposer()
    decomposer.decomposer = broken  # type: ignore[assignment]
    result = asyncio.run(decomposer.adecompose("build it", "ctx"))
    assert result.is_single_task


# ── runner defaults ──────────────────────────────────────────────────────────


def _args(tmp_path: Path, *extra: str) -> Any:
    args = rs.parse_args(
        ["--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path / "out"), *extra]
    )
    rs._validate_args(args)  # pyright: ignore[reportPrivateUsage]
    return args


def test_runner_defaults_per_suite(tmp_path: Path) -> None:
    smoke = _args(tmp_path)
    assert (smoke.episode_timeout, smoke.task_timeout, smoke.grader_timeout) == (
        2400.0,
        900,
        300.0,
    )
    assert smoke.max_tokens == 32768

    commit0_dir = tmp_path / "c0"
    (commit0_dir / "bundles").mkdir(parents=True)
    bundle = commit0_dir / "bundles" / "c0-x"
    bundle.mkdir()
    (bundle / "task.toml").write_text("id = 'c0-x'\n")
    c0 = _args(tmp_path, "--suite", "commit0", "--commit0-dir", str(commit0_dir))
    assert (c0.episode_timeout, c0.task_timeout, c0.grader_timeout) == (5400.0, 5400, 900.0)
    explicit = _args(
        tmp_path,
        "--suite", "commit0", "--commit0-dir", str(commit0_dir),
        "--episode-timeout", "100", "--task-timeout", "50", "--max-tokens", "9000",
    )  # fmt: skip
    assert (explicit.episode_timeout, explicit.task_timeout, explicit.max_tokens) == (
        100,
        50,
        9000,
    )


def test_runner_counts_truncations_and_stall_stops() -> None:
    log = (
        "LLM RESPONSE TRUNCATED (role=decomposer, model=m, 32768 completion tokens): x\n"
        "RLM STOPPED at iteration 7 (stalled: 3 consecutive iterations with no new file read)\n"
        "RLM STOPPED at iteration 9 (stalled: 3 consecutive iterations with no new file read)\n"
    )
    sig = rs.log_signals(log)
    assert sig["truncated_responses"] == 1 and sig["stalled_stops"] == 2
