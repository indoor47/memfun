"""Truncated LLM responses fail visibly in ``memfun swarm`` (hermetic: no network, no GPU)."""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from memfun_agent import llm_telemetry as tel
from memfun_agent.context_first import ContextFirstSolver
from memfun_agent.decomposer import DecompositionTruncatedError, TaskDecomposer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import run_smoke as rs  # noqa: E402

_DSPY_WARNING = "LM response was truncated due to exceeding max_tokens=4096; increase max_tokens"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Any:  # pyright: ignore[reportUnusedFunction]
    tel.reset_truncations()
    tel.install_truncation_hook()
    monkeypatch.setattr(tel, "publish_llm_call_event", lambda _p: None)
    yield
    tel.reset_truncations()


def _dspy_warn() -> None:
    logging.getLogger("dspy.clients.lm").warning(_DSPY_WARNING)


# ── log hook ─────────────────────────────────────────────────────────────────


def test_hook_counts_once_per_truncation_and_per_role() -> None:
    _dspy_warn()
    with tel.role_context("decomposer"):
        _dspy_warn()
        _dspy_warn()
    assert tel.truncation_counts() == {"default": 1, "decomposer": 2}
    assert tel.total_truncations() == 3


def test_unrelated_warnings_are_ignored() -> None:
    logging.getLogger("dspy.clients.lm").warning("rate limited, retrying")
    logging.getLogger("dspy.clients.lm").info(_DSPY_WARNING)
    assert tel.total_truncations() == 0


def test_hook_install_is_idempotent() -> None:
    tel.install_truncation_hook()
    tel.install_truncation_hook()
    handlers = [
        h
        for h in logging.getLogger("dspy.clients.lm").handlers
        if isinstance(h, tel._TruncationLogHandler)  # pyright: ignore[reportPrivateUsage]
    ]
    assert len(handlers) == 1
    _dspy_warn()
    assert tel.total_truncations() == 1


def _lm_history(reason: str) -> Any:
    lm = MagicMock()
    lm.history = [{"response": {"choices": [{"finish_reason": reason}]}, "model": "m"}]
    return lm


def test_callback_and_hook_do_not_double_count() -> None:
    cb = tel.MemfunDSPyCallback()
    events: list[dict[str, Any]] = []
    with patch.object(tel, "publish_llm_call_event", events.append):
        cb.on_lm_start("c1", _lm_history("length"), {"prompt": "x"})
        _dspy_warn()  # DSPy warns before on_lm_end
        cb.on_lm_end("c1", ["{partial"], None)
        assert tel.total_truncations() == 1
        assert events[0]["success"] is False and events[0]["error"] == tel.TRUNCATED_ERROR
        # Callback-only truncation (hook silent) is still counted exactly once.
        cb.on_lm_start("c2", _lm_history("length"), {"prompt": "x"})
        cb.on_lm_end("c2", ["{partial"], None)
        assert tel.total_truncations() == 2
        # A hook count never suppresses counting of a later callback-only call.
        cb.on_lm_start("c3", _lm_history("length"), {})
        _dspy_warn()
        cb.on_lm_end("c3", ["x"], None)
        cb.on_lm_start("c4", _lm_history("length"), {})
        cb.on_lm_end("c4", ["x"], None)
    assert tel.total_truncations() == 4


def test_install_dspy_callback_also_installs_the_hook() -> None:
    logger = logging.getLogger("dspy.clients.lm")
    for h in list(logger.handlers):
        if isinstance(h, tel._TruncationLogHandler):  # pyright: ignore[reportPrivateUsage]
            logger.removeHandler(h)
    tel.install_dspy_callback()
    assert any(
        isinstance(h, tel._TruncationLogHandler)  # pyright: ignore[reportPrivateUsage]
        for h in logger.handlers
    )


def test_hook_logs_the_marker_the_runner_counts(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        _dspy_warn()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert len(rs.TRUNCATED_LOG_RE.findall(text)) == 1


# ── decomposer / context-first ───────────────────────────────────────────────


def test_decomposer_raises_when_only_the_log_hook_saw_the_truncation() -> None:
    def predictor(**_kw: Any) -> Any:
        _dspy_warn()
        raise ValueError("Adapter JSONAdapter failed to parse")

    decomposer = TaskDecomposer()
    decomposer.decomposer = predictor  # type: ignore[assignment]

    async def run() -> Any:
        with tel.role_context("decomposer"):
            return await decomposer.adecompose("build it", "ctx")

    with pytest.raises(DecompositionTruncatedError, match="max_tokens"):
        asyncio.run(run())


def _solver(internal: Any) -> ContextFirstSolver:
    solver = ContextFirstSolver.__new__(ContextFirstSolver)
    solver._solve_internal = internal  # type: ignore[method-assign]
    return solver


def test_context_first_truncation_raises() -> None:
    async def internal(*_a: Any, **_k: Any) -> Any:
        _dspy_warn()
        raise ValueError("parse failure")

    with pytest.raises(tel.ContextFirstTruncatedError, match=r"truncated.*max_tokens"):
        asyncio.run(_solver(internal).asolve("q"))


def test_context_first_non_truncation_failure_still_falls_back() -> None:
    async def internal(*_a: Any, **_k: Any) -> Any:
        raise ValueError("garbage")

    result = asyncio.run(_solver(internal).asolve("q"))
    assert result.success is False and result.method == "context_first_error"


def _agent(asolve: Any) -> Any:
    from memfun_agent.coding_agent import RLMCodingAgent

    agent = RLMCodingAgent.__new__(RLMCodingAgent)
    agent._rlm = MagicMock()  # type: ignore[attr-defined]
    agent._context_solver = MagicMock(asolve=asolve)  # type: ignore[attr-defined]
    agent._build_rlm_query = lambda *_a, **_k: "q"  # type: ignore[method-assign]
    return agent


def test_coding_agent_propagates_truncation_instead_of_escalating() -> None:
    async def asolve(**_kw: Any) -> Any:
        raise tel.ContextFirstTruncatedError(f"truncated; {tel.truncation_advice('worker')}")

    agent = _agent(asolve)
    with pytest.raises(tel.LLMTruncatedError):
        asyncio.run(agent._handle_rlm("ask", "q", "", {}))


def test_coding_agent_non_truncation_failure_still_escalates() -> None:
    async def asolve(**_kw: Any) -> Any:
        raise ValueError("boom")

    out = asyncio.run(_agent(asolve)._handle_rlm("ask", "q", "", {}))
    assert out["_escalate"] is True


def test_workflow_single_task_result_is_failed_with_truncated_error() -> None:
    from memfun_agent.workflow import WorkflowEngine, WorkflowState
    from memfun_core.types import TaskResult

    err = f"ContextFirstTruncatedError: truncated; {tel.truncation_advice('worker')}"
    task_result = TaskResult("t", "rlm-coder", False, None, error=err)
    engine = WorkflowEngine.__new__(WorkflowEngine)
    engine._manager = MagicMock()  # type: ignore[attr-defined]
    engine._manager.is_running.return_value = True
    engine._emit_status = lambda *_a, **_k: None  # type: ignore[method-assign]

    async def dispatch(*_a: Any, **_k: Any) -> TaskResult:
        return task_result

    engine._orchestrator = MagicMock(dispatch=dispatch)  # type: ignore[attr-defined]
    state = WorkflowState.__new__(WorkflowState)
    state.workflow_id = "wf"
    state.decomposition = None
    result = asyncio.run(engine._execute_single_task(state, "do it", "", None, 0.0))
    assert result.success is False
    assert "truncated" in (result.error or "") and "max_tokens" in (result.error or "")


# ── swarm-result.json / CLI ──────────────────────────────────────────────────


def test_result_payload_carries_truncation_even_when_the_workflow_reported_success() -> None:
    from memfun_agent.workflow import WorkflowResult
    from memfun_cli.commands import swarm as swarm_mod

    ok = WorkflowResult(workflow_id="wf", success=True, answer="done")
    clean = swarm_mod._swarm_result_payload(ok, 1.0, {})  # pyright: ignore[reportPrivateUsage]
    assert clean["success"] is True and clean["truncated_responses"] == 0
    with tel.role_context("worker"):
        _dspy_warn()
        _dspy_warn()
    doc = swarm_mod._swarm_result_payload(ok, 1.0, {})  # pyright: ignore[reportPrivateUsage]
    assert doc["success"] is False and doc["truncated_responses"] == 2
    assert "truncated" in doc["error"] and "max_tokens" in doc["error"]
    assert swarm_mod._truncated_responses() == 2  # pyright: ignore[reportPrivateUsage]
    out = swarm_mod._swarm_result_payload(None, 1.0, {}, error="DecompositionTruncatedError: x")  # pyright: ignore[reportPrivateUsage]
    assert out["success"] is False and out["truncated_responses"] == 2


def test_summary_panel_prints_the_red_truncation_line() -> None:
    from memfun_agent.workflow import WorkflowResult
    from memfun_cli.commands import swarm as swarm_mod
    from rich.console import Console

    with tel.role_context("worker"):
        _dspy_warn()
    console = Console(record=True, width=120)
    ok = WorkflowResult(workflow_id="wf", success=True, answer="")
    with patch.object(swarm_mod, "console", console):
        swarm_mod._display_final_summary(ok, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert "LLM RESPONSE TRUNCATED" in console.export_text()


def test_result_json_file_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from memfun_agent.workflow import WorkflowResult
    from memfun_cli.commands import swarm as swarm_mod

    out = tmp_path / "r.json"
    monkeypatch.setenv(swarm_mod.RESULT_JSON_ENV, str(out))
    _dspy_warn()
    ok = WorkflowResult(workflow_id="wf", success=True, answer="")
    swarm_mod._write_swarm_result_json(  # pyright: ignore[reportPrivateUsage]
        swarm_mod._swarm_result_payload(ok, 1.0, {})  # pyright: ignore[reportPrivateUsage]
    )
    doc = json.loads(out.read_text())
    assert doc["success"] is False and doc["truncated_responses"] == 1 and doc["error"]


# ── runner ───────────────────────────────────────────────────────────────────


def test_classify_truncated_failure_is_a_model_failure() -> None:
    assert rs.classify(truncated_failure=True) == ("model_failure", "truncated response")
    assert rs.classify() == ("model_failure", "no hidden test passes")
    assert rs.classify(truncated_failure=True, all_pass=True)[0] == "pass"


# ── real dspy.LM against an in-process fake OpenAI server ───────────────────


class _Handler(http.server.BaseHTTPRequestHandler):
    finish_reason = "length"

    def log_message(self, *_a: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"a":'},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_openai() -> Any:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def _real_call(api_base: str, engine: str, *, hook: bool) -> list[dict[str, Any]]:
    import dspy

    logger = logging.getLogger("dspy.clients.lm")
    hooks = [h for h in logger.handlers if isinstance(h, tel._TruncationLogHandler)]  # pyright: ignore[reportPrivateUsage]
    if not hook:
        for h in hooks:
            logger.removeHandler(h)
    events: list[dict[str, Any]] = []
    try:
        lm = dspy.LM(
            "openai/m", api_base=api_base, api_key="k", engine=engine, max_tokens=5, cache=False
        )
        with (
            patch.object(tel, "publish_llm_call_event", events.append),
            dspy.context(callbacks=[tel.make_dspy_callback()]),
            tel.role_context("worker"),
        ):
            lm("hi")
    finally:
        if not hook:
            for h in hooks:
                logger.addHandler(h)
    return events


@pytest.mark.parametrize("engine", ["auto", "litellm"])
@pytest.mark.parametrize("hook", [True, False])
def test_real_lm_truncation_counted_exactly_once(
    fake_openai: str, engine: str, hook: bool
) -> None:
    events = _real_call(fake_openai, engine, hook=hook)
    assert tel.truncation_counts() == {"worker": 1}
    assert len(events) == 1 and events[0]["success"] is False
    assert events[0]["error"] == tel.TRUNCATED_ERROR


@pytest.mark.parametrize("engine", ["auto", "litellm"])
def test_real_lm_normal_response_is_not_flagged(fake_openai: str, engine: str) -> None:
    _Handler.finish_reason = "stop"
    try:
        events = _real_call(fake_openai, engine, hook=True)
    finally:
        _Handler.finish_reason = "length"
    assert tel.total_truncations() == 0
    assert events and events[0]["success"] is True


def test_finish_reason_handles_both_response_shapes() -> None:
    class Flat:
        finish_reason = "max_tokens"

    lm = MagicMock()
    lm.history = [{"response": Flat()}]
    assert tel._finish_reason(lm, None) == "length"  # pyright: ignore[reportPrivateUsage]
    lm.history = [{"response": {"choices": [{"finish_reason": "length"}]}}]
    assert tel._finish_reason(lm, None) == "length"  # pyright: ignore[reportPrivateUsage]
