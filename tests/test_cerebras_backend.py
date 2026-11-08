"""Cerebras backend wiring: config, LM kwargs, runner plan/validation, outcome rules."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from memfun_agent import role_models
from memfun_core.config import ConfigError, LLMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import cerebras_proxy as cb  # noqa: E402
from evals.swarm import run_smoke as rs  # noqa: E402


def test_reasoning_effort_is_validated_and_sent_only_when_set() -> None:
    base = {"provider": "openai", "model": "m", "api_key_env": "", "base_url": "http://h/v1"}
    _, plain = role_models.lm_settings(LLMConfig(**base))
    assert "reasoning_effort" not in plain
    for level in ("none", "low", "medium", "high"):
        _, kw = role_models.lm_settings(LLMConfig(**base, reasoning_effort=level))
        assert kw["reasoning_effort"] == level
    with pytest.raises(ConfigError):
        LLMConfig(**base, reasoning_effort="max")
    a = LLMConfig(**base, reasoning_effort="none")
    b = LLMConfig(**base, reasoning_effort="high")
    assert role_models._cache_key(a) != role_models._cache_key(b)  # pyright: ignore[reportPrivateUsage]


def test_episode_toml_carries_reasoning_effort_and_no_penalty() -> None:
    sampling = rs.Sampling(temperature=0.6, loop_guard=True, reasoning_effort="none")
    doc = tomllib.loads(
        rs.memfun_toml("http://127.0.0.1:1/e/x/v1", "qwen-3.8-27b", 16384, sampling, False)
    )
    llm = doc["llm"]
    assert llm["reasoning_effort"] == "none" and llm["max_tokens"] == 16384
    assert "repetition_penalty" not in llm and llm["constrained_json"] is False
    assert "reasoning_effort" not in tomllib.loads(rs.memfun_toml("http://h/v1", "m", 1))["llm"]


def test_each_episode_gets_its_own_proxy_path_and_sglang_keeps_the_base_url() -> None:
    common = {
        "model": "m", "memfun_cmd": ["x"], "max_agents": 1, "task_timeout": 1,
        "episode_timeout": 1.0, "grader_timeout": 1.0, "max_tokens": 1,
    }  # fmt: skip
    plain = rs.RunContext(base_url="http://127.0.0.1:18000/v1", **common)
    assert plain.episode_base_url("e1") == "http://127.0.0.1:18000/v1"
    proxy = cb.CerebrasProxy("k")
    try:
        ctx = rs.RunContext(base_url="x", proxy=proxy, proxy_url="http://127.0.0.1:5", **common)
        assert ctx.episode_base_url("c0s-a-b-t1") == "http://127.0.0.1:5/e/c0s-a-b-t1/v1"
        assert cb.EPISODE_RE.match("c0s-wcwidth-wcswidth-t1")
    finally:
        proxy.stop()


def test_provider_stop_is_a_harness_error_only_when_nothing_was_produced() -> None:
    stop = "provider gave up after retries"
    out, reason = rs.classify(provider_stopped=stop, any_output=False)
    assert out == "harness_error" and "provider" in reason
    # output exists: graded on its merits, never excluded by a rate-limit blip
    kw = {"hidden_passed": 0, "all_pass": False, "any_output": True}
    assert rs.classify(provider_stopped=stop, **kw)[0] == "model_failure"
    assert rs.classify(**kw)[0] == "model_failure"
    assert rs.classify(provider_stopped="x", hidden_passed=3, all_pass=True)[0] == "pass"


def test_progress_line_shows_cost_for_cerebras_only() -> None:
    ep = {
        "episode": "c0s-x-f-t1", "outcome": "pass", "wall_s": 30.0,
        "grade": {"hidden_passed": 3, "hidden_failed": 0},
        "metrics": {"tokens": {"prompt": 1000, "completion": 900}},
        "reasoning_tokens": 750,
        "provider": {"cost_usd": 0.0123, "requests": 4, "cached_tokens": 512, "retries_429": 2},
    }  # fmt: skip
    line = rs.progress_line(ep, 3)
    assert "$0.0123" in line and "4 req" in line and "512 cached" in line and "2x429" in line
    del ep["provider"]
    assert "$" not in rs.progress_line(ep, 3)


def test_cerebras_validation_and_dry_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--backend", "cerebras", "--out-dir", str(tmp_path / "runs"), "--memfun-cmd", "false"]
    assert rs.main([*base, "--max-tokens", "65536", "--dry-plan"]) == 2  # paid cap is 40,960
    assert "40960" in capsys.readouterr().err
    assert rs.main([*base, "--max-cost-usd", "0", "--dry-plan"]) == 2
    capsys.readouterr()
    rc = rs.main([
        *base, "--bundles", "textkit-3", "--max-tokens", "16384", "--reasoning-effort", "none",
        "--dry-plan", "--cerebras-key-file", str(tmp_path / "does-not-exist"),
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "backend       : cerebras via a local proxy" in out
    assert "spend cap $8.00" in out and "reasoning_effort none" in out
    assert "qwen-3.8-27b" in out
    assert not (tmp_path / "runs").exists()  # a dry plan neither reads the key nor creates a run


def test_sglang_still_requires_a_base_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert rs.main(["--dry-plan"]) == 2
    assert "--base-url is required" in capsys.readouterr().err
