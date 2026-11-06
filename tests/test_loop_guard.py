"""Repetition-loop guard, sampling penalties and the runner's sampling flags (hermetic).

The LLM is a tiny in-process OpenAI-compatible server (``_Fake``) that can stream either a
short normal answer or an endless repetition loop; nothing touches the network or a GPU.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from memfun_agent import loop_guard as lg
from memfun_agent import role_models
from memfun_agent.loop_guard import LoopDetector
from memfun_core.config import LLMConfig, MemfunConfig
from memfun_core.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

# ── detector ─────────────────────────────────────────────────────────────────


def _feed_all(det: LoopDetector, text: str, step: int = 37) -> bool:
    return any(det.feed(text[i : i + step]) for i in range(0, len(text), step))


def test_detects_the_observed_loop() -> None:
    text = "Implement the core Selector class: " + " re, reall, re_first," * 400
    det = LoopDetector()
    assert _feed_all(det, text)
    assert "re" in det.block


def test_detects_a_long_repeated_block() -> None:
    block = "def helper(x):\n    return x + 1  # same line again and again\n"
    assert _feed_all(LoopDetector(), "prefix\n" + block * 150)


def test_ordinary_code_and_data_are_not_loops() -> None:
    code = "\n".join(f"def f{i}(a, b):\n    return a * {i} + b  # item {i}" for i in range(400))
    table = ",\n".join(f"(0x{i:04x}, 0x{i + 7:04x})" for i in range(0, 4000, 7))
    assert not _feed_all(LoopDetector(), code)
    assert not _feed_all(LoopDetector(), table)


def test_legitimate_repetition_is_not_a_loop() -> None:
    """Audit finding: each of these aborted at ~2060 chars with the first thresholds."""
    long_line = "x" * 249 + "\n"
    assert not _feed_all(LoopDetector(), long_line * 12)  # 12 identical 250-char lines
    assert not _feed_all(LoopDetector(), "| 0 | 0 |\n" * 300)  # table rows
    assert not _feed_all(LoopDetector(), "    pass\n" * 400)
    assert not _feed_all(LoopDetector(), "0, " * 1000)  # padding
    assert not _feed_all(LoopDetector(), "0, 0, 0,\n" * 600)  # no letter in the block
    assert _feed_all(LoopDetector(), "ab " * 4000)  # ... but a real 12 000-char loop is cut


def test_single_character_runs_and_short_repeats_are_not_loops() -> None:
    assert not _feed_all(LoopDetector(), "x" * 20000)  # padding / separators
    assert not _feed_all(LoopDetector(), "-" * 80 + "\n" + "=" * 80 + "\n" * 3)
    assert not _feed_all(LoopDetector(), "abc, def, " * 5)  # a few copies only
    assert not _feed_all(LoopDetector(), "abc, def, " * 150)  # < MIN_REPEAT_CHARS of repeat


def test_detection_is_incremental_and_bounded() -> None:
    det = LoopDetector()
    start = time.perf_counter()
    assert not _feed_all(det, "".join(f"line {i} of normal output\n" for i in range(5000)), 211)
    assert time.perf_counter() - start < 5
    assert len(det._tail) <= lg.MIN_REPEAT_CHARS + lg.MAX_BLOCK * 2  # pyright: ignore[reportPrivateUsage]


def test_mid_stream_errors_are_retried(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(lg, "RETRY_BACKOFF_S", 0.0)
    lg.install_loop_guard()
    lg.register_guarded("openai/m", fake.base)
    calls = {"n": 0}
    real = litellm.completion

    class Boom:
        def __iter__(self) -> Any:
            raise ConnectionError("dropped mid-stream")

    orig_inner = getattr(real, "__wrapped__", real)

    def flaky(**kw: Any) -> Any:
        calls["n"] += 1
        return Boom() if calls["n"] == 1 else orig_inner(**kw)

    out = lg._guarded_completion(  # pyright: ignore[reportPrivateUsage]
        litellm,
        flaky,
        model="openai/m",
        api_base=fake.base,
        api_key="x",
        messages=[{"role": "user", "content": f"retry {time.time_ns()}"}],
        num_retries=2,
    )
    assert calls["n"] == 2
    assert out.choices[0].message.content == "hello world"
    with pytest.raises(ConnectionError):
        lg._guarded_completion(  # pyright: ignore[reportPrivateUsage]
            litellm,
            lambda **_k: Boom(),
            model="openai/m",
            api_base=fake.base,
            messages=[{"role": "user", "content": "x"}],
            num_retries=0,
        )


# ── fake OpenAI-compatible server ────────────────────────────────────────────


class _Fake:
    def __init__(self) -> None:
        self.mode = "normal"
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                fake.requests.append(body)
                base = {"id": "x", "created": 1, "model": body["model"]}
                if not body.get("stream"):
                    data = json.dumps(
                        {
                            **base,
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "hello world"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()

                def ev(d: dict[str, Any]) -> None:
                    self.wfile.write(
                        b"data: "
                        + json.dumps({**base, "object": "chat.completion.chunk", **d}).encode()
                        + b"\n\n"
                    )
                    self.wfile.flush()

                try:
                    ev(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                    if fake.mode == "normal":
                        for w in ("hello ", "world"):
                            ev(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": w},
                                            "finish_reason": None,
                                        }
                                    ]
                                }
                            )
                        ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                        ev(
                            {
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 3,
                                    "completion_tokens": 2,
                                    "total_tokens": 5,
                                },
                            }
                        )
                    else:
                        parts = (" re,", " reall,", " re_first,")  # varied chunks
                        same = " re, reall, re_first,"  # 'loop-identical': one chunk repeated
                        for i in range(200000):  # ~4 MB if never cut
                            ev(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": parts[i % 3]
                                                if fake.mode == "loop"
                                                else same
                                            },
                                            "finish_reason": None,
                                        }
                                    ]
                                }
                            )
                            if i % 100 == 0:
                                time.sleep(0.001)
                        ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    fake.client_hung_up = True

        self.client_hung_up = False
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def fake() -> Any:
    f = _Fake()
    yield f
    f.close()


def _cfg(base: str, **kw: Any) -> LLMConfig:
    kw.setdefault("loop_guard", True)
    return LLMConfig(
        provider="openai",
        model="m",
        api_key_env="",
        base_url=base,
        temperature=0.7,
        max_tokens=32768,
        **kw,
    )


def _finish_reason(lm: Any) -> str:
    return lm.history[-1]["response"].choices[0].finish_reason


# ── guard end to end through a real dspy.LM ──────────────────────────────────


def test_loop_is_aborted_early_and_reported_as_a_cut_off_response(
    fake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(lg.ENV_DISABLE, raising=False)
    before = lg.loop_abort_counts().get("default", 0)
    fake.mode = "loop"
    lm = role_models.build_lm(_cfg(fake.base))
    start = time.perf_counter()
    out = lm("hi")
    assert time.perf_counter() - start < 20
    assert fake.requests[-1]["stream"] is True
    assert _finish_reason(lm) == "length"  # visible to every truncation path
    assert 2000 <= len(str(out)) < 200_000  # cut long before the 4 MB the server would send
    assert lg.loop_abort_counts().get("default", 0) == before + 1
    assert lm.history[-1]["usage"]["completion_tokens"] > 0


def test_identical_prompts_are_not_served_from_a_cache(fake: Any) -> None:
    lm = role_models.build_lm(_cfg(fake.base))
    lm("same prompt")
    lm("same prompt")
    assert len(fake.requests) == 2  # a cached loop response would otherwise be replayed


def test_litellm_identical_chunk_guard_counts_as_a_loop_abort(fake: Any) -> None:
    """Single-token repetition: litellm raises 'repeating the same chunk'; we must not retry."""
    fake.mode = "loop-identical"
    before = lg.loop_abort_counts().get("default", 0)
    lm = role_models.build_lm(_cfg(fake.base))
    out = lm(f"identical {time.time_ns()}")
    assert _finish_reason(lm) == "length" and len(str(out)) > 0
    assert lg.loop_abort_counts().get("default", 0) == before + 1
    assert len(fake.requests) == 1  # not retried


def test_normal_answers_pass_through_unchanged(fake: Any) -> None:
    lm = role_models.build_lm(_cfg(fake.base))
    out = lm("hi")
    assert "".join(map(str, out)) == "hello world"
    assert _finish_reason(lm) == "stop"
    before = dict(lg.loop_abort_counts())
    lm("again")
    assert lg.loop_abort_counts() == before


def test_guard_can_be_disabled(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lg.ENV_DISABLE, "0")
    lm = role_models.build_lm(_cfg(fake.base))
    lm(f"guard disabled {time.time_ns()}")  # unique: dspy's default cache is back when off
    assert not fake.requests[-1].get("stream")  # plain request, no streaming


def test_guard_only_touches_registered_endpoints() -> None:
    lg.register_guarded("openai/x", "http://h")
    guard = lg._guardable  # pyright: ignore[reportPrivateUsage]
    assert guard({"model": "openai/x", "api_base": "http://h"}) is True
    assert guard({"model": "openai/x", "api_base": "http://other"}) is False  # not registered
    assert guard({"model": "openai/y", "api_base": "http://h"}) is False
    assert guard({"model": "anthropic/x", "api_base": "http://h"}) is False
    assert guard({"model": "openai/x"}) is False
    assert guard({"model": "openai/x", "api_base": "http://h", "stream": True}) is False


def test_an_unguarded_lm_on_another_endpoint_is_not_streamed(fake: Any) -> None:
    """The litellm wrapper is process-wide; an LM built without loop_guard is untouched."""
    role_models.build_lm(_cfg("http://127.0.0.1:1/v1"))  # installs the wrapper, registers it
    lm = role_models.build_lm(_cfg(fake.base, loop_guard=False))
    lm(f"unguarded {time.time_ns()}")
    assert not fake.requests[-1].get("stream")


def test_guard_is_opt_in_per_llm_config(fake: Any) -> None:
    """Without [llm] loop_guard the LM is built exactly as before (native engine, no stream)."""
    lm = role_models.build_lm(_cfg(fake.base, loop_guard=False))
    lm(f"opt-in off {time.time_ns()}")
    assert not fake.requests[-1].get("stream")
    assert LLMConfig().loop_guard is False
    with pytest.raises(ConfigError):
        MemfunConfig._from_raw({"llm": {"loop_guard": "yes"}})  # pyright: ignore[reportPrivateUsage]
    cfg = MemfunConfig._from_raw({"llm": {"loop_guard": True}})  # pyright: ignore[reportPrivateUsage]
    assert cfg.llm_for_role("worker").loop_guard is True


# ── sampling penalties reach the server ──────────────────────────────────────


def test_penalties_are_sent_to_the_openai_compatible_server(fake: Any) -> None:
    lm = role_models.build_lm(
        _cfg(fake.base, frequency_penalty=0.3, presence_penalty=0.2, repetition_penalty=1.05)
    )
    lm("hi")
    body = fake.requests[-1]
    assert body["temperature"] == 0.7
    assert body["frequency_penalty"] == 0.3
    assert body["presence_penalty"] == 0.2
    assert body["repetition_penalty"] == 1.05
    assert body["max_tokens"] == 32768


def test_zero_penalties_send_nothing_extra(fake: Any) -> None:
    lm = role_models.build_lm(_cfg(fake.base))
    lm("hi")
    body = fake.requests[-1]
    assert not {"frequency_penalty", "presence_penalty", "repetition_penalty"} & set(body)


def test_lm_cache_key_separates_penalties() -> None:
    a = role_models._cache_key(_cfg("http://h/v1"))  # pyright: ignore[reportPrivateUsage]
    b = role_models._cache_key(_cfg("http://h/v1", repetition_penalty=1.1))  # pyright: ignore[reportPrivateUsage]
    assert a != b


@pytest.mark.parametrize(
    "raw",
    [
        {"frequency_penalty": 3},
        {"presence_penalty": -2.5},
        {"frequency_penalty": "0.1"},
        {"repetition_penalty": 0},
        {"repetition_penalty": 2.5},
        {"repetition_penalty": True},
    ],
)
def test_invalid_penalties_are_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        MemfunConfig._from_raw({"llm": raw})  # pyright: ignore[reportPrivateUsage]


def test_penalties_parse_from_toml_sections() -> None:
    cfg = MemfunConfig._from_raw(  # pyright: ignore[reportPrivateUsage]
        {"llm": {"temperature": 0.7, "repetition_penalty": 1.05, "frequency_penalty": 0.1}}
    )
    assert (cfg.llm.temperature, cfg.llm.repetition_penalty, cfg.llm.frequency_penalty) == (
        0.7,
        1.05,
        0.1,
    )
    assert cfg.llm_for_role("worker").repetition_penalty == 1.05  # roles inherit


# ── runner flags ─────────────────────────────────────────────────────────────


def _c0_dir(tmp_path: Path) -> Path:
    d = tmp_path / "c0"
    b = d / "bundles" / "c0-lib"
    b.mkdir(parents=True)
    (b / "task.toml").write_text("id = 'c0-lib'\n")
    return d


def _args(tmp_path: Path, *extra: str) -> Any:
    args = rs.parse_args(
        ["--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path / "out"), *extra]
    )
    rs._validate_args(args)  # pyright: ignore[reportPrivateUsage]
    return args


def test_runner_default_sampling_is_not_greedy(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert args.temperature == 0.7
    assert args.repetition_penalty == 1.05
    assert args.frequency_penalty == 0.0 and args.presence_penalty == 0.0


def test_memfun_toml_carries_the_sampling_settings() -> None:
    s = rs.Sampling(
        temperature=0.7, frequency_penalty=0.2, presence_penalty=0.1, repetition_penalty=1.05
    )
    text = rs.memfun_toml("http://h/v1", "m", 32768, s)
    for line in (
        "temperature = 0.7",
        "frequency_penalty = 0.2",
        "presence_penalty = 0.1",
        "repetition_penalty = 1.05",
        "max_tokens = 32768",
    ):
        assert line in text
    old = rs.memfun_toml("http://h/v1", "m", 8192)  # library callers keep the old output
    assert "temperature = 0.0" in old and "penalty" not in old and "loop_guard" not in old
    assert "loop_guard = true" in rs.memfun_toml(
        "http://h/v1", "m", 1, rs.Sampling(loop_guard=True)
    )


def test_runner_rejects_bad_sampling_flags(tmp_path: Path) -> None:
    for flag, val in (
        ("--temperature", "3"),
        ("--repetition-penalty", "0"),
        ("--frequency-penalty", "5"),
        ("--presence-penalty", "-9"),
    ):
        with pytest.raises(rs.UsageError):
            _args(tmp_path, flag, val)


def test_single_library_with_n_trials(tmp_path: Path) -> None:
    c0 = _c0_dir(tmp_path)
    args = _args(
        tmp_path,
        "--suite", "commit0", "--commit0-dir", str(c0),
        "--bundles", "c0-lib", "--trials", "3", "--parallel-tasks", "3",
    )  # fmt: skip
    plans = rs.plan_episodes(args, tmp_path / "run")  # pyright: ignore[reportPrivateUsage]
    assert [p.name for p in plans] == ["c0-lib-t1", "c0-lib-t2", "c0-lib-t3"]


def test_runner_counts_loop_aborts() -> None:
    log = (
        "LLM LOOP ABORTED (role=decomposer, model=m): output repeated ' re,' for >= 2000 chars\n"
        * 2
    )
    assert rs.log_signals(log)["loop_aborts"] == 2
