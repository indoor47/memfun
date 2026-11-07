"""Thinking on: SGLang ``--reasoning-parser qwen3`` puts the model's reasoning in
``reasoning_content`` and keeps ``content`` clean.  memfun/DSPy/litellm must keep the answer
clean, count the reasoning tokens, and keep the loop guard and truncation handling sane with
long reasoning.  The server is an in-process fake: no network, no GPU.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import dspy
import pytest
from memfun_agent import llm_telemetry as tel
from memfun_agent import loop_guard as lg
from memfun_agent import role_models
from memfun_core.config import LLMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

ANSWER = "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"
THINK = "Let me think about the width rules first. " * 30


class _Think:
    """OpenAI-compatible server whose reply has ``reasoning_content`` next to ``content``."""

    def __init__(self) -> None:
        self.reasoning = THINK
        self.content = ANSWER
        self.reasoning_tokens = 321
        self.finish = "stop"
        self.loop = False  # reasoning that never ends (streamed)
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                fake.requests.append(body)
                base = {"id": "x", "created": 1, "model": body["model"]}
                usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": fake.reasoning_tokens + 20,
                    "total_tokens": fake.reasoning_tokens + 30,
                    "completion_tokens_details": {"reasoning_tokens": fake.reasoning_tokens},
                }
                if not body.get("stream"):
                    msg = {
                        "role": "assistant",
                        "content": fake.content,
                        "reasoning_content": fake.reasoning,
                    }
                    data = json.dumps(
                        {
                            **base,
                            "object": "chat.completion",
                            "choices": [
                                {"index": 0, "message": msg, "finish_reason": fake.finish}
                            ],
                            "usage": usage,
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
                    chunk = {**base, "object": "chat.completion.chunk", **d}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()

                try:
                    first = {"role": "assistant", "content": ""}
                    ev({"choices": [{"index": 0, "delta": first, "finish_reason": None}]})
                    if fake.loop:
                        words = (" wait,", " no,", " hmm,")
                        for i in range(200000):
                            delta = {"reasoning_content": words[i % 3]}
                            ev({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                            if i % 100 == 0:
                                time.sleep(0.001)
                    else:
                        for part in (fake.reasoning, fake.content):
                            key = "reasoning_content" if part is fake.reasoning else "content"
                            for i in range(0, len(part), 16):
                                delta = {key: part[i : i + 16]}
                                ev(
                                    {
                                        "choices": [
                                            {"index": 0, "delta": delta, "finish_reason": None}
                                        ]
                                    }
                                )
                        ev({"choices": [{"index": 0, "delta": {}, "finish_reason": fake.finish}]})
                        ev({"choices": [], "usage": usage})
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def fake() -> Any:
    f = _Think()
    yield f
    f.close()


class Sig(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _lm(base: str, guard: bool) -> Any:
    cfg = LLMConfig(
        provider="openai",
        model="m",
        api_key_env="",
        base_url=base,
        temperature=0.6,
        max_tokens=32768,
        loop_guard=guard,
    )
    return role_models.build_lm(cfg)


def _ask(lm: Any) -> tuple[Any, Any]:
    with dspy.context(lm=lm, callbacks=[tel.make_dspy_callback()]):
        pred = dspy.Predict(Sig)(question=f"q {time.time_ns()}")
    return pred, lm.history[-1]


# ── content stays clean, reasoning stays apart, reasoning tokens are reported ─


@pytest.mark.parametrize("guard", [False, True], ids=["native-engine", "streamed-guard"])
def test_answer_is_clean_and_reasoning_is_kept_separately(fake: Any, guard: bool) -> None:
    pred, entry = _ask(_lm(fake.base, guard))
    assert pred.answer == "42"  # parsed from content only
    out = entry["outputs"][0]
    assert isinstance(out, dict)
    assert "Let me think" not in out["text"]  # no reasoning in the parsed text
    assert "Let me think" in str(out.get("reasoning_content", ""))  # kept apart, not lost
    details = entry["usage"].get("completion_tokens_details")
    reasoning = (
        details["reasoning_tokens"] if isinstance(details, dict) else details.reasoning_tokens
    )
    assert reasoning == 321 and entry["usage"]["completion_tokens"] == 341


@pytest.mark.parametrize("guard", [False, True], ids=["native-engine", "streamed-guard"])
def test_reasoning_tokens_are_counted_per_role_and_reset(fake: Any, guard: bool) -> None:
    tel.reset_truncations()
    _ask(_lm(fake.base, guard))
    _ask(_lm(fake.base, guard))
    assert tel.total_reasoning_tokens() == 642  # 2 calls x 321
    assert sum(tel.reasoning_token_counts().values()) == 642
    tel.reset_truncations()
    assert tel.total_reasoning_tokens() == 0


def test_a_server_without_reasoning_counts_zero(fake: Any) -> None:
    fake.reasoning, fake.reasoning_tokens = "", 0
    tel.reset_truncations()
    pred, _ = _ask(_lm(fake.base, False))
    assert pred.answer == "42" and tel.total_reasoning_tokens() == 0


def test_swarm_result_and_runner_carry_the_reasoning_tokens() -> None:
    from memfun_cli.commands import swarm as swarm_mod

    tel.reset_truncations()
    tel._record_reasoning_tokens("worker", 700)  # pyright: ignore[reportPrivateUsage]
    tel._record_reasoning_tokens("decomposer", 50)  # pyright: ignore[reportPrivateUsage]
    assert swarm_mod._reasoning_tokens() == 750  # pyright: ignore[reportPrivateUsage]
    tel.reset_truncations()
    assert swarm_mod._reasoning_tokens() == 0  # pyright: ignore[reportPrivateUsage]
    ep = {
        "episode": "c0s-x-f-t1",
        "outcome": "pass",
        "wall_s": 60.0,
        "grade": {"hidden_passed": 3, "hidden_failed": 0},
        "metrics": {"tokens": {"prompt": 1000, "completion": 900}},
        "reasoning_tokens": 750,
    }
    assert "(reasoning 750)" in rs.progress_line(ep, 3)


# ── long reasoning vs the loop guard ─────────────────────────────────────────


def test_long_legitimate_reasoning_is_not_aborted(fake: Any) -> None:
    fake.reasoning = "".join(
        f"Step {i}: check the case for code point {i * 7}.\n" for i in range(2500)
    )
    assert len(fake.reasoning) > 100_000
    before = dict(lg.loop_abort_counts())
    pred, entry = _ask(_lm(fake.base, True))
    assert pred.answer == "42"
    assert lg.loop_abort_counts() == before
    assert entry["response"].choices[0].finish_reason == "stop"


def test_a_reasoning_loop_is_aborted_and_reported_as_cut_off(fake: Any) -> None:
    fake.loop = True
    before = lg.loop_abort_counts().get("default", 0)
    lm = _lm(fake.base, True)
    start = time.perf_counter()
    with dspy.context(lm=lm), pytest.raises(Exception):  # noqa: B017 - no answer to parse
        dspy.Predict(Sig)(question=f"loop {time.time_ns()}")
    assert time.perf_counter() - start < 30  # cut long before the ~4 MB the server would send
    # DSPy's adapter makes a second (JSON) attempt after the first parse failure: >= 1 abort
    assert lg.loop_abort_counts().get("default", 0) >= before + 1
    assert lm.history[-1]["response"].choices[0].finish_reason == "length"


# ── reasoning that eats the whole token budget ───────────────────────────────


def test_reasoning_that_uses_the_whole_budget_is_a_visible_truncation(fake: Any) -> None:
    """finish_reason=length with the reasoning present and an empty content: the call must be
    reported as truncated (the answer was never produced), never as a quiet empty success."""
    fake.content, fake.finish = "", "length"
    tel.reset_truncations()
    before = tel.total_truncations()
    with pytest.raises(Exception):  # noqa: B017 - nothing to parse
        _ask(_lm(fake.base, False))
    assert tel.total_truncations() >= before + 1  # >= 1: the adapter retries once in JSON mode
    assert tel.reasoning_token_counts()  # the reasoning tokens of the failed call still count
