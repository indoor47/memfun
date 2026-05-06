"""Tests for the ``memfun doctor`` command.

These tests never reach the network — every ``litellm.completion`` call is
mocked.  They cover:

* happy path (valid key + reachable endpoint),
* 401 / authentication error,
* timeout error (both LiteLLM ``Timeout`` and stdlib ``TimeoutError``),
* missing config / unset provider,
* missing ``credentials.json`` but env var is set (env var wins),
* no API key anywhere,
* key masking — the actual key never appears in stdout.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from memfun_cli.commands.doctor import (
    _mask_key,
    _resolve_lite_llm_model,
    doctor_command,
)
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestMaskKey:
    def test_empty(self) -> None:
        assert _mask_key("") == "(empty)"

    def test_short_key_fully_masked(self) -> None:
        # No leading bytes leak — even short keys reveal nothing.
        assert _mask_key("abc") == "***"

    def test_long_key_shows_only_last_four(self) -> None:
        masked = _mask_key("sk-ant-api03-very-secret-AAAA")
        # Provider prefix MUST NOT appear.
        assert "sk-ant" not in masked
        assert masked.endswith("AAAA (29 chars)")


class TestResolveLiteLlmModel:
    def test_anthropic(self) -> None:
        assert _resolve_lite_llm_model("anthropic", "claude-opus-4-6") == (
            "anthropic/claude-opus-4-6",
            None,
        )

    def test_openai(self) -> None:
        assert _resolve_lite_llm_model("openai", "gpt-4o") == (
            "openai/gpt-4o",
            None,
        )

    def test_ollama_supplies_fallback_key(self) -> None:
        # Ollama doesn't need a real key but LiteLLM still requires non-empty.
        assert _resolve_lite_llm_model("ollama", "qwen2.5-coder") == (
            "ollama_chat/qwen2.5-coder",
            "ollama",
        )

    def test_custom_passes_through(self) -> None:
        # Custom / openai-compat — caller provides full LiteLLM string.
        assert _resolve_lite_llm_model("custom", "openai/qwen-coder-7b") == (
            "openai/qwen-coder-7b",
            None,
        )


# ---------------------------------------------------------------------------
# CLI integration via CliRunner
# ---------------------------------------------------------------------------


def _build_app() -> typer.Typer:
    """Build a tiny Typer app exposing only ``doctor`` for isolation."""
    app = typer.Typer()
    app.command("doctor")(doctor_command)
    return app


def _fake_response(completion_tokens: int = 1) -> MagicMock:
    """Mimic the LiteLLM ``ModelResponse`` shape used by doctor."""
    response = MagicMock()
    response.usage = MagicMock(completion_tokens=completion_tokens)
    return response


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to a clean temp directory for the test.

    Doctor reads ``~/.memfun/credentials.json`` and ``~/.memfun/config.toml``;
    redirecting HOME isolates each test from the developer's real install.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() consults HOME on POSIX; macOS / Linux developer machines.
    return tmp_path


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run doctor from a directory with no project ``.memfun/config.toml``.

    Without this, ``MemfunConfig.load()`` would discover the actual repo's
    config and the test would not be hermetic.
    """
    sub = tmp_path / "proj"
    sub.mkdir()
    monkeypatch.chdir(sub)
    return sub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestDoctorHappyPath:
    def test_succeeds_with_valid_env_key(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-AAAA")

        with patch("litellm.completion", return_value=_fake_response(1)) as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 0, result.output
        assert "all checks passed" in result.output
        # The actual key is never printed in full.
        assert "sk-ant-test-AAAA" not in result.output
        # Last four chars are shown.
        assert "AAAA" in result.output
        # 1 ping = 1 LLM call.
        assert mock_completion.call_count == 1
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["model"] == "anthropic/claude-opus-4-6"
        assert call_kwargs["api_key"] == "sk-ant-test-AAAA"


# ---------------------------------------------------------------------------
# Auth failure (the original 3-month-stale-key bug)
# ---------------------------------------------------------------------------


class TestDoctorAuthFailure:
    def test_reports_authentication_error(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stale-XXXX")

        from litellm.exceptions import AuthenticationError

        # AuthenticationError requires (message, llm_provider, model) on
        # the LiteLLM API surface.
        err = AuthenticationError(
            message="invalid x-api-key",
            llm_provider="anthropic",
            model="claude-opus-4-6",
        )
        with patch("litellm.completion", side_effect=err):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        assert "Authentication failed" in result.output
        assert "ANTHROPIC_API_KEY" in result.output
        assert "credentials.json" in result.output
        # Stale key value never leaks.
        assert "sk-ant-stale-XXXX" not in result.output


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestDoctorTimeout:
    def test_reports_litellm_timeout(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-AAAA")

        from litellm.exceptions import Timeout

        err = Timeout(
            message="endpoint timed out",
            llm_provider="anthropic",
            model="claude-opus-4-6",
        )
        with patch("litellm.completion", side_effect=err):
            result = CliRunner().invoke(_build_app(), ["--timeout", "1"])

        assert result.exit_code == 1, result.output
        assert "timed out" in result.output

    def test_reports_stdlib_timeout(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some asyncio paths surface stdlib TimeoutError; doctor catches both.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-AAAA")

        with patch("litellm.completion", side_effect=TimeoutError("asyncio timeout")):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        assert "timed out" in result.output


# ---------------------------------------------------------------------------
# Connection refused / DNS / unreachable endpoint
# ---------------------------------------------------------------------------


class TestDoctorConnectionError:
    def test_reports_endpoint_unreachable(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Use a custom config with a base_url so the error message can
        # surface the offending host.
        cfg_dir = tmp_path / ".memfun"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text(
            '[llm]\nprovider = "openai"\nmodel = "gpt-4o"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'base_url = "http://127.0.0.1:65535/v1"\n'
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-AAAA")

        from litellm.exceptions import APIConnectionError

        err = APIConnectionError(
            message="connection refused",
            llm_provider="openai",
            model="gpt-4o",
        )
        with patch("litellm.completion", side_effect=err):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        assert "Endpoint unreachable" in result.output
        assert "127.0.0.1:65535" in result.output


# ---------------------------------------------------------------------------
# Missing config / unset provider
# ---------------------------------------------------------------------------


class TestDoctorMissingConfig:
    def test_load_failure_reports_cleanly(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
    ) -> None:
        # No env key, no creds file — but the default provider is
        # "anthropic", so doctor will reach the ping step and fail there.
        # To exercise the "config did not load" branch directly, monkey-
        # patch MemfunConfig.load() to raise.
        with patch(
            "memfun_cli.commands.doctor.MemfunConfig.load",
            side_effect=RuntimeError("synthetic load failure"),
        ):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        assert "Failed to load config" in result.output
        assert "synthetic load failure" in result.output

    def test_blank_provider_reports_cleanly(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
    ) -> None:
        from memfun_core.config import LLMConfig, MemfunConfig

        # Construct a config with an empty provider — defensive path,
        # not currently reachable from TOML but worth covering.
        broken = MemfunConfig(llm=LLMConfig(provider=""))

        with patch(
            "memfun_cli.commands.doctor.MemfunConfig.load",
            return_value=broken,
        ):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        assert "No LLM provider set" in result.output


# ---------------------------------------------------------------------------
# Credential file presence interaction with env vars
# ---------------------------------------------------------------------------


class TestDoctorCredentialResolution:
    def test_env_var_wins_when_credentials_json_missing(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No credentials.json; env var alone must be enough.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-only-BBBB")

        with patch("litellm.completion", return_value=_fake_response(1)) as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 0, result.output
        assert "all checks passed" in result.output
        # File is reported as absent.
        assert "not present" in result.output
        # env var is reported as set.
        assert "set in env" in result.output
        assert mock_completion.call_args.kwargs["api_key"] == "sk-env-only-BBBB"

    def test_credentials_json_used_when_env_var_unset(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        creds_dir = isolated_home / ".memfun"
        creds_dir.mkdir()
        creds_path = creds_dir / "credentials.json"
        creds_path.write_text(json.dumps({"ANTHROPIC_API_KEY": "sk-from-file-CCCC"}))

        with patch("litellm.completion", return_value=_fake_response(1)) as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 0, result.output
        # Doctor warns the env var isn't set …
        assert "not in env" in result.output
        # … but the credential file's value made it into the LiteLLM call.
        assert mock_completion.call_args.kwargs["api_key"] == "sk-from-file-CCCC"
        # And the file's value was masked, not printed in full.
        assert "sk-from-file-CCCC" not in result.output

    def test_no_api_key_anywhere_fails_with_clean_message(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # No credentials file, no env var.
        with patch("litellm.completion") as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1, result.output
        # We never even attempted the network call …
        assert mock_completion.call_count == 0
        # … and the message tells the user exactly which env var to set.
        assert "ANTHROPIC_API_KEY" in result.output
        assert "memfun init" in result.output

    def test_corrupt_credentials_json_does_not_crash(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Garbage in credentials.json must produce a clean error, not a
        # JSONDecodeError traceback.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-DDDD")
        creds_dir = isolated_home / ".memfun"
        creds_dir.mkdir()
        (creds_dir / "credentials.json").write_text("{not valid json")

        with patch("litellm.completion", return_value=_fake_response(1)):
            result = CliRunner().invoke(_build_app(), [])

        # Env var still works — corrupt file is reported but non-fatal.
        assert result.exit_code == 0, result.output
        assert "unreadable" in result.output


# ---------------------------------------------------------------------------
# Verbose flag surfaces tracebacks
# ---------------------------------------------------------------------------


class TestDoctorVerbose:
    def test_verbose_prints_exception_details(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-AAAA")

        from litellm.exceptions import AuthenticationError

        err = AuthenticationError(
            message="distinctive-error-token-987",
            llm_provider="anthropic",
            model="claude-opus-4-6",
        )
        with patch("litellm.completion", side_effect=err):
            result = CliRunner().invoke(_build_app(), ["--verbose"])

        assert result.exit_code == 1
        assert "distinctive-error-token-987" in result.output

    def test_non_verbose_hides_exception_details(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-AAAA")

        from litellm.exceptions import AuthenticationError

        err = AuthenticationError(
            message="distinctive-error-token-987",
            llm_provider="anthropic",
            model="claude-opus-4-6",
        )
        with patch("litellm.completion", side_effect=err):
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 1
        # Provider message stays out of stdout in non-verbose mode.
        assert "distinctive-error-token-987" not in result.output


# ---------------------------------------------------------------------------
# Cost discipline — exactly one call, no retries
# ---------------------------------------------------------------------------


class TestDoctorCostDiscipline:
    def test_one_ping_one_call_even_on_failure(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Doctor must NOT retry. Each invocation = exactly one ping."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-AAAA")

        from litellm.exceptions import AuthenticationError

        err = AuthenticationError(
            message="bad key",
            llm_provider="anthropic",
            model="claude-opus-4-6",
        )
        with patch("litellm.completion", side_effect=err) as mock_completion:
            CliRunner().invoke(_build_app(), [])

        assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# kwargs are wired correctly across providers
# ---------------------------------------------------------------------------


class TestDoctorPerProvider:
    def _write_global_config(self, home: Path, body: str) -> None:
        cfg_dir = home / ".memfun"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.toml").write_text(body)

    def test_openai_with_base_url(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_global_config(
            isolated_home,
            '[llm]\nprovider = "openai"\nmodel = "qwen-coder-7b"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'base_url = "http://127.0.0.1:8089/v1"\n',
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-EEEE")

        with patch("litellm.completion", return_value=_fake_response(2)) as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 0, result.output
        kwargs: dict[str, Any] = mock_completion.call_args.kwargs
        assert kwargs["model"] == "openai/qwen-coder-7b"
        assert kwargs["api_base"] == "http://127.0.0.1:8089/v1"
        assert kwargs["api_key"] == "sk-test-EEEE"

    def test_ollama_uses_dummy_key(
        self,
        isolated_home: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_global_config(
            isolated_home,
            '[llm]\nprovider = "ollama"\nmodel = "qwen2.5-coder"\n'
            'api_key_env = "OLLAMA_API_KEY"\n',
        )
        # Deliberately do NOT set OLLAMA_API_KEY — ollama doesn't need one.
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        with patch("litellm.completion", return_value=_fake_response(1)) as mock_completion:
            result = CliRunner().invoke(_build_app(), [])

        assert result.exit_code == 0, result.output
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "ollama_chat/qwen2.5-coder"
        # Doctor injected the placeholder key LiteLLM requires.
        assert kwargs["api_key"] == "ollama"
