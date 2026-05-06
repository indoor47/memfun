"""Tests for the multi-provider init wizard (issue #15).

These tests exercise the wizard's provider/model picker by mocking
the InquirerPy `select`, `text`, and `secret` prompts. They verify
that:

* Each provider writes the correct config (provider, model,
  api_key_env, base_url where relevant).
* `LLMConfig` defaults to `claude-sonnet-4-6`.
* `_pick_model` and `_get_api_key_env` cover all 5 provider paths.
* OpenAI-compatible accepts an empty API key without erroring.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from memfun_cli.commands import init as init_mod
from memfun_core.config import LLMConfig, MemfunConfig

# ---------------------------------------------------------------------------
# LLMConfig default
# ---------------------------------------------------------------------------


class TestLLMConfigDefault:
    def test_default_model_is_sonnet_4_6(self) -> None:
        """The new default model is Sonnet 4.6 (was Opus 4.6)."""
        cfg = LLMConfig()
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"

    def test_memfun_config_inherits_default(self) -> None:
        cfg = MemfunConfig()
        assert cfg.llm.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# _get_api_key_env coverage
# ---------------------------------------------------------------------------


class TestGetApiKeyEnv:
    def test_anthropic(self) -> None:
        assert init_mod._get_api_key_env("anthropic") == "ANTHROPIC_API_KEY"

    def test_openai(self) -> None:
        assert init_mod._get_api_key_env("openai") == "OPENAI_API_KEY"

    def test_openai_compat_uses_placeholder(self) -> None:
        assert init_mod._get_api_key_env("openai-compat") == "MEMFUN_LOCAL_API_KEY"

    def test_ollama_returns_none(self) -> None:
        assert init_mod._get_api_key_env("ollama") is None

    def test_custom(self) -> None:
        assert init_mod._get_api_key_env("custom") == "MEMFUN_API_KEY"

    def test_unknown(self) -> None:
        assert init_mod._get_api_key_env("nonsense") is None


# ---------------------------------------------------------------------------
# _pick_model — per-provider model picker
# ---------------------------------------------------------------------------


class TestPickModel:
    def test_anthropic_select_default_returns_sonnet(self) -> None:
        with patch.object(init_mod, "_prompt_select", return_value="claude-sonnet-4-6"):
            assert init_mod._pick_model("anthropic") == "claude-sonnet-4-6"

    def test_anthropic_select_opus(self) -> None:
        with patch.object(init_mod, "_prompt_select", return_value="claude-opus-4-6"):
            assert init_mod._pick_model("anthropic") == "claude-opus-4-6"

    def test_anthropic_custom_model(self) -> None:
        with (
            patch.object(init_mod, "_prompt_select", return_value="__custom__"),
            patch.object(init_mod, "_prompt_text", return_value="claude-future-9-9"),
        ):
            assert init_mod._pick_model("anthropic") == "claude-future-9-9"

    def test_anthropic_custom_blank_falls_back_to_default(self) -> None:
        with (
            patch.object(init_mod, "_prompt_select", return_value="__custom__"),
            patch.object(init_mod, "_prompt_text", return_value="  "),
        ):
            # Whitespace falls back to the first curated model.
            assert init_mod._pick_model("anthropic") == "claude-sonnet-4-6"

    def test_openai_select_default(self) -> None:
        with patch.object(init_mod, "_prompt_select", return_value="gpt-5"):
            assert init_mod._pick_model("openai") == "gpt-5"

    def test_openai_custom_model(self) -> None:
        with (
            patch.object(init_mod, "_prompt_select", return_value="__custom__"),
            patch.object(init_mod, "_prompt_text", return_value="gpt-5-experimental"),
        ):
            assert init_mod._pick_model("openai") == "gpt-5-experimental"

    def test_ollama_default(self) -> None:
        with patch.object(init_mod, "_prompt_text", return_value=""):
            # Empty input falls back to the default.
            assert init_mod._pick_model("ollama") == "qwen2.5-coder:7b"

    def test_ollama_explicit(self) -> None:
        with patch.object(init_mod, "_prompt_text", return_value="llama3.1"):
            assert init_mod._pick_model("ollama") == "llama3.1"

    def test_openai_compat_required_model(self) -> None:
        with patch.object(init_mod, "_prompt_text", return_value="qwen-coder-7b"):
            assert init_mod._pick_model("openai-compat") == "qwen-coder-7b"

    def test_openai_compat_blank_then_value(self) -> None:
        """Blank input re-prompts; second response wins."""
        responses = iter(["", "  ", "qwen-coder-7b"])
        with patch.object(
            init_mod,
            "_prompt_text",
            side_effect=lambda *a, **kw: next(responses),
        ):
            assert init_mod._pick_model("openai-compat") == "qwen-coder-7b"

    def test_custom_default(self) -> None:
        with patch.object(init_mod, "_prompt_text", return_value=""):
            assert init_mod._pick_model("custom") == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# Wizard end-to-end (mocked prompts) — config write
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path):
    """Redirect Path.home() to a tmp dir for the duration of a test."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    with patch.object(Path, "home", return_value=fake_home):
        yield fake_home


def _read_global_config(home: Path) -> dict:
    """Helper: read the wizard-written config.toml back as a dict."""
    cfg_path = home / ".memfun" / "config.toml"
    assert cfg_path.exists(), f"config.toml not written at {cfg_path}"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Strip provider-relevant env vars so prompts always fire."""
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MEMFUN_API_KEY",
        "MEMFUN_LOCAL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestWizardEndToEnd:
    def test_anthropic_default_writes_sonnet_config(
        self, isolated_home: Path, clean_env: None
    ) -> None:
        """Anthropic + sonnet-4-6 (default selection)."""
        with (
            patch.object(
                init_mod,
                "_prompt_select",
                side_effect=["anthropic", "claude-sonnet-4-6"],
            ),
            patch.object(init_mod, "_prompt_secret", return_value="sk-ant-fake"),
        ):
            init_mod.run_global_setup()

        cfg = _read_global_config(isolated_home)
        assert cfg["llm"]["provider"] == "anthropic"
        assert cfg["llm"]["model"] == "claude-sonnet-4-6"
        assert cfg["llm"]["api_key_env"] == "ANTHROPIC_API_KEY"
        assert "base_url" not in cfg["llm"]

        # Credentials saved + injected into env.
        creds_path = isolated_home / ".memfun" / "credentials.json"
        assert creds_path.exists()
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-fake"

    def test_openai_with_gpt5(self, isolated_home: Path, clean_env: None) -> None:
        """OpenAI + gpt-5 model."""
        with (
            patch.object(
                init_mod,
                "_prompt_select",
                side_effect=["openai", "gpt-5"],
            ),
            patch.object(init_mod, "_prompt_secret", return_value="sk-openai-fake"),
        ):
            init_mod.run_global_setup()

        cfg = _read_global_config(isolated_home)
        assert cfg["llm"]["provider"] == "openai"
        assert cfg["llm"]["model"] == "gpt-5"
        assert cfg["llm"]["api_key_env"] == "OPENAI_API_KEY"
        assert "base_url" not in cfg["llm"]

    def test_openai_compat_with_blank_key(self, isolated_home: Path, clean_env: None) -> None:
        """OpenAI-compatible + base_url + model + blank API key.

        Verifies issue #15's primary unblocked path: local llama.cpp
        endpoints don't require an API key.
        """
        with (
            patch.object(init_mod, "_prompt_select", return_value="openai-compat"),
            patch.object(
                init_mod,
                "_prompt_text",
                side_effect=[
                    "http://localhost:8088/v1",  # base_url
                    "qwen-coder-7b",  # model (free-text)
                ],
            ),
            patch.object(init_mod, "_prompt_secret", return_value=""),
        ):
            init_mod.run_global_setup()

        cfg = _read_global_config(isolated_home)
        assert cfg["llm"]["provider"] == "openai-compat"
        assert cfg["llm"]["model"] == "qwen-coder-7b"
        assert cfg["llm"]["base_url"] == "http://localhost:8088/v1"
        assert cfg["llm"]["api_key_env"] == "MEMFUN_LOCAL_API_KEY"

        # No credentials file written when the key was blank.
        creds_path = isolated_home / ".memfun" / "credentials.json"
        assert not creds_path.exists()

    def test_openai_compat_with_key(self, isolated_home: Path, clean_env: None) -> None:
        """OpenAI-compatible with a provided API key."""
        with (
            patch.object(init_mod, "_prompt_select", return_value="openai-compat"),
            patch.object(
                init_mod,
                "_prompt_text",
                side_effect=[
                    "http://localhost:8088/v1",
                    "meta-llama/Llama-3.1-70B",
                ],
            ),
            patch.object(init_mod, "_prompt_secret", return_value="sk-or-fake"),
        ):
            init_mod.run_global_setup()

        cfg = _read_global_config(isolated_home)
        assert cfg["llm"]["provider"] == "openai-compat"
        assert cfg["llm"]["model"] == "meta-llama/Llama-3.1-70B"
        assert cfg["llm"]["base_url"] == "http://localhost:8088/v1"
        creds_path = isolated_home / ".memfun" / "credentials.json"
        assert creds_path.exists()
        assert os.environ.get("MEMFUN_LOCAL_API_KEY") == "sk-or-fake"

    def test_ollama_with_default_model(self, isolated_home: Path, clean_env: None) -> None:
        """Ollama with default qwen2.5-coder:7b model."""
        with (
            patch.object(init_mod, "_prompt_select", return_value="ollama"),
            patch.object(init_mod, "_prompt_text", return_value=""),
        ):
            init_mod.run_global_setup()

        cfg = _read_global_config(isolated_home)
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["llm"]["model"] == "qwen2.5-coder:7b"
        # Ollama has no api_key_env.
        assert "api_key_env" not in cfg["llm"]


# ---------------------------------------------------------------------------
# DSPy configuration: openai-compat round-trips correctly
# ---------------------------------------------------------------------------


class TestDspyOpenAICompat:
    def test_configure_dspy_openai_compat_injects_sk_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no MEMFUN_LOCAL_API_KEY is set, _configure_dspy
        should inject 'sk-local' for openai-compat so LiteLLM can
        instantiate the LM object.
        """
        from memfun_cli.commands import chat as chat_mod
        from memfun_core.config import (
            BackendConfig,
            SandboxBackendConfig,
            WebToolsConfig,
        )

        # Clean env.
        monkeypatch.delenv("MEMFUN_LOCAL_API_KEY", raising=False)

        cfg = MemfunConfig(
            project_name="test",
            llm=LLMConfig(
                provider="openai-compat",
                model="qwen-coder-7b",
                api_key_env="MEMFUN_LOCAL_API_KEY",
                base_url="http://localhost:8080/v1",
            ),
            backend=BackendConfig(),
            sandbox=SandboxBackendConfig(),
            web=WebToolsConfig(),
        )

        captured: dict = {}

        class FakeLM:
            def __init__(self, model: str, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        with patch("dspy.LM", FakeLM), patch("dspy.configure", lambda **kw: None):
            chat_mod._configure_dspy(cfg)

        # LiteLLM model name should be openai/qwen-coder-7b.
        assert captured.get("model") == "openai/qwen-coder-7b"
        kwargs = captured.get("kwargs", {})
        # sk-local injected when env was empty.
        assert kwargs.get("api_key") == "sk-local"
        # api_base passed through.
        assert kwargs.get("api_base") == "http://localhost:8080/v1"

    def test_configure_dspy_openai_compat_uses_real_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When MEMFUN_LOCAL_API_KEY is set, it overrides sk-local."""
        from memfun_cli.commands import chat as chat_mod
        from memfun_core.config import (
            BackendConfig,
            SandboxBackendConfig,
            WebToolsConfig,
        )

        monkeypatch.setenv("MEMFUN_LOCAL_API_KEY", "real-key-123")

        cfg = MemfunConfig(
            project_name="test",
            llm=LLMConfig(
                provider="openai-compat",
                model="qwen-coder-7b",
                api_key_env="MEMFUN_LOCAL_API_KEY",
                base_url="http://localhost:8080/v1",
            ),
            backend=BackendConfig(),
            sandbox=SandboxBackendConfig(),
            web=WebToolsConfig(),
        )

        captured: dict = {}

        class FakeLM:
            def __init__(self, model: str, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        with patch("dspy.LM", FakeLM), patch("dspy.configure", lambda **kw: None):
            chat_mod._configure_dspy(cfg)

        kwargs = captured.get("kwargs", {})
        assert kwargs.get("api_key") == "real-key-123"
