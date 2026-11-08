"""Context budget: the gathered context fits the provider's window; overflow is its own class.

Hermetic: no network, no model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from memfun_agent.context_first import (
    CONTEXT_LENGTH_METHOD,
    ContextFirstConfig,
    ContextGatherer,
    build_file_manifest,
    context_budget_bytes,
    is_context_length_error,
    trim_large_file,
)
from memfun_agent.workflow import context_length_failed, no_edit_reason
from memfun_core.config import LLMConfig, MemfunConfig, default_context_window
from memfun_core.errors import ConfigError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "swarm"))
import run_smoke as rs


def test_default_windows() -> None:
    assert default_context_window("anthropic") == 200_000
    assert default_context_window("openai", "https://api.cerebras.ai/v1") == 131_072
    assert default_context_window("openai", "http://localhost:8000/v1") == 131_072
    assert LLMConfig(context_window=262_144).context_window_effective == 262_144


def test_context_window_from_toml_and_validation(tmp_path: Path) -> None:
    p = tmp_path / "memfun.toml"
    p.write_text('[llm]\nprovider = "openai"\ncontext_window = 262144\n')
    assert MemfunConfig.from_toml(p).llm.context_window == 262_144
    with pytest.raises(ConfigError):
        LLMConfig(context_window=-1)


def test_budget_fits_cerebras_window() -> None:
    # 414 KB (172k tokens) overflowed; the budget must sit far below 131072 tokens.
    assert context_budget_bytes(131_072) == 163_840
    cfg = ContextFirstConfig.for_llm(
        LLMConfig(provider="openai", base_url="https://api.cerebras.ai/v1")
    )
    assert cfg.max_prompt_bytes == 163_840
    assert cfg.max_gather_bytes <= 163_840
    assert cfg.max_context_bytes <= 163_840
    big = ContextFirstConfig.for_llm(LLMConfig(context_window=262_144))
    assert big.max_prompt_bytes == 327_680
    assert big.max_gather_bytes == 327_680  # still capped by the window, not the 400k default


def test_unicode_table_is_outlined() -> None:
    table = (
        "# Auto-generated unicode table\nTABLE = [\n" + "    (0x1100, 0x115F),\n" * 8000 + "]\n"
    )
    out = trim_large_file(table, "wcwidth/table_wide.py", 50_000)
    assert len(out) < 3_000
    assert "generated data" in out or "too large" in out
    # A small ordinary file is untouched, and 0 disables trimming.
    small = "def f():\n    return 1\n"
    assert trim_large_file(small, "a.py", 50_000) == small
    assert trim_large_file(table, "wcwidth/table_wide.py", 0) == table


def test_large_python_gets_symbol_outline() -> None:
    src = "class A:\n    def m(self): ...\n\ndef g(): ...\n" + "x = 1\n" * 20_000
    out = trim_large_file(src, "big.py", 50_000)
    assert "class A" in out and "def m" in out and "def g" in out


def test_gatherer_trims_per_file(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("def keep(): ...\n")
    (tmp_path / "tables.py").write_text("T = [\n" + "    1,\n" * 60_000 + "]\n")
    g = ContextGatherer(max_bytes=400_000, max_file_bytes=50_000)
    text = asyncio.run(g.agather(["small.py", "tables.py"], tmp_path))
    assert "def keep" in text and len(text) < 10_000
    manifest = build_file_manifest(tmp_path)
    fast = g.read_all_files(manifest, tmp_path)
    assert "def keep" in fast and len(fast) < 10_000


def test_read_all_drops_whole_files_over_budget(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"m{i}.py").write_text(f"# {i}\n" + "x = 1\n" * 3000)
    g = ContextGatherer(max_bytes=40_000, max_file_bytes=50_000)
    out = g.read_all_files(build_file_manifest(tmp_path), tmp_path)
    assert len(out) <= 40_000
    assert out.count("=== FILE:") < 5
    assert "truncated" not in out


@pytest.mark.parametrize(
    "msg",
    [
        "Error code: 400 - context_length_exceeded: Please reduce the length of the messages",
        "This model's maximum context length is 131072 tokens",
        "prompt is too long: 172000 tokens > 131072 maximum",
    ],
)
def test_context_length_error_detected(msg: str) -> None:
    assert is_context_length_error(RuntimeError(msg))
    assert not is_context_length_error(RuntimeError("connection reset"))


def test_context_length_is_not_no_edits() -> None:
    data = {
        "failure_class": CONTEXT_LENGTH_METHOD,
        "ops": [],
        "files_created": [],
        "_escalate": True,
    }
    assert context_length_failed(data)
    reason = no_edit_reason("rlm-coder", data)
    assert reason is not None and reason.startswith("context_length_exceeded")
    assert not reason.startswith("no file changes")


def test_classify_context_length_is_its_own_reason() -> None:
    assert rs.classify(no_edits=True, context_length_exceeded=True) == (
        "model_failure",
        "context_length_exceeded",
    )
    assert rs.classify(no_edits=True)[1] == "no file changes produced"
