"""Constrained JSON (``[llm] constrained_json``): wire shape, schemas, fail-closed verdicts.

Hermetic: an in-process fake OpenAI-compatible server on an ephemeral 127.0.0.1 port records
request bodies.  No live SGLang run is involved; the documented request shape is
https://docs.sglang.io/advanced_features/structured_outputs.html
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import copy
import json
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy
import pytest
from memfun_agent import constrained_json as cj
from memfun_agent import decomposer as decomposer_mod
from memfun_agent import decomposer_schema as ds
from memfun_agent import role_models
from memfun_agent.decomposer import DecompositionConfig, TaskDecomposer
from memfun_agent.review_findings import parse_review_output
from memfun_agent.signatures import TaskDecomposition
from memfun_agent.workflow import WorkflowEngine
from memfun_core.config import LLMConfig, MemfunConfig
from memfun_core.errors import ConfigError
from memfun_core.types import TaskResult

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

warnings.filterwarnings("ignore", category=DeprecationWarning)

RID = "abc123nonce"
PLAN: dict[str, Any] = {
    "sub_tasks": [
        {
            "id": "T1",
            "description": "Fix the undefined error in fetchWeather",
            "agent_type": "coder",
            "inputs": ["src/weather.js"],
            "outputs": ["src/weather.js"],
            "depends_on": [],
            "max_iterations": 8,
        }
    ],
    "shared_spec": "use camelCase",
    "parallelism_groups": [["T1"]],
}


class FakeServer:
    """Records POST bodies; answers ``respond(body) -> assistant content``."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.respond: Callable[[dict[str, Any]], str] = lambda _b: json.dumps(PLAN)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                body = json.loads(raw)
                outer.bodies.append(body)
                content = outer.respond(body)
                out = json.dumps(
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[FakeServer]:
    srv = FakeServer()
    yield srv
    srv.close()


@pytest.fixture(autouse=True)
def _reset_role_config() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    role_models.set_role_config(None)
    yield
    role_models.set_role_config(None)


def _lm(srv: FakeServer, engine: str) -> dspy.LM:
    return dspy.LM(
        "openai/qwen",
        api_base=srv.base_url,
        api_key="k",
        engine=engine,
        cache=False,
        max_tokens=256,
        temperature=0.0,
    )


def _enable(srv: FakeServer, *, constrained: bool = True, provider: str = "openai") -> None:
    cfg = MemfunConfig(
        llm=LLMConfig(
            provider=provider,
            model="qwen",
            base_url=srv.base_url,
            constrained_json=constrained,
        )
    )
    role_models.set_role_config(cfg)


def _schema_of(body: dict[str, Any]) -> dict[str, Any]:
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    return fmt["json_schema"]["schema"]


# ── Config ───────────────────────────────────────────────────


def test_config_default_off_inherited_and_role_override() -> None:
    assert LLMConfig().constrained_json is False
    cfg = MemfunConfig._from_raw(
        {
            "llm": {
                "provider": "openai",
                "base_url": "http://127.0.0.1:1/v1",
                "constrained_json": True,
                "roles": {"auditor": {"constrained_json": False}},
            }
        }
    )
    assert cfg.llm_for_role("decomposer").constrained_json is True
    assert cfg.llm_for_role("auditor").constrained_json is False


@pytest.mark.parametrize(
    "raw",
    [
        {"llm": {"constrained_json": "yes"}},
        {"llm": {"roles": {"worker": {"constrained_json": 1}}}},
    ],
)
def test_config_rejects_non_bool(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        MemfunConfig._from_raw(raw)


@pytest.mark.parametrize(
    ("provider", "base_url", "flag", "expected"),
    [
        ("openai", "http://h/v1", True, True),
        ("openai-compat", "http://h/v1", True, True),
        ("openai", None, True, False),
        ("anthropic", "http://h/v1", True, False),
        ("ollama", "http://h/v1", True, False),
        ("custom", "http://h/v1", True, False),
        ("openai", "http://h/v1", False, False),
    ],
)
def test_effective_only_for_openai_compat_with_base_url(
    provider: str, base_url: str | None, flag: bool, expected: bool
) -> None:
    llm = LLMConfig(provider=provider, base_url=base_url, constrained_json=flag)
    assert llm.constrained_json_effective is expected


# ── Wire shape: both dspy engines ────────────────────────────

ENGINES = ["auto", "litellm"]


@pytest.mark.parametrize("engine", ENGINES)
def test_decomposer_schema_reaches_the_wire(server: FakeServer, engine: str) -> None:
    with dspy.context(lm=_lm(server, engine), adapter=cj.decomposition_adapter()):
        out = dspy.Predict(TaskDecomposition)(task_description="t", project_context="c")
    assert len(server.bodies) == 1
    fmt = server.bodies[0]["response_format"]
    assert fmt == {
        "type": "json_schema",
        "json_schema": {"name": "task_decomposition", "schema": ds.DECOMPOSITION_SCHEMA},
    }
    assert out.sub_tasks[0]["id"] == "T1"


@pytest.mark.parametrize("engine", ENGINES)
def test_raw_and_copy_response_format_reach_the_wire(server: FakeServer, engine: str) -> None:
    fmt = cj.response_format("foo", {"type": "object"})
    lm = _lm(server, engine)
    lm("hi", response_format=fmt)
    lm.copy(response_format=fmt)("hi")
    assert [b["response_format"] for b in server.bodies] == [fmt, fmt]


@pytest.mark.parametrize("engine", ENGINES)
def test_no_response_format_without_the_adapter(server: FakeServer, engine: str) -> None:
    server.respond = lambda _b: (
        "[[ ## sub_tasks ## ]]\n[]\n[[ ## shared_spec ## ]]\ns\n"
        "[[ ## parallelism_groups ## ]]\n[]\n[[ ## completed ## ]]"
    )
    with dspy.context(lm=_lm(server, engine)):
        dspy.Predict(TaskDecomposition)(task_description="t", project_context="c")
    assert "response_format" not in server.bodies[0]


# ── Decomposer end to end ────────────────────────────────────


def _decompose() -> Any:
    dec = TaskDecomposer(DecompositionConfig(enable_hierarchical=False))
    return asyncio.run(dec.adecompose("fix it", "ctx"))


@pytest.mark.parametrize("engine", ENGINES)
def test_decomposer_end_to_end_when_enabled(server: FakeServer, engine: str) -> None:
    _enable(server)
    with dspy.context(lm=_lm(server, engine)):
        res = _decompose()
    assert [t.id for t in res.sub_tasks] == ["T1"]
    assert res.sub_tasks[0].outputs == ["src/weather.js"]
    assert res.parallelism_groups == [["T1"]]
    assert _schema_of(server.bodies[0]) == ds.DECOMPOSITION_SCHEMA


@pytest.mark.parametrize("kind", ["flag-off", "anthropic"])
def test_decomposer_sends_nothing_when_off_or_not_openai(server: FakeServer, kind: str) -> None:
    if kind == "flag-off":
        _enable(server, constrained=False)
    else:
        _enable(server, provider="anthropic")
    server.respond = lambda _b: (
        "[[ ## sub_tasks ## ]]\n"
        + json.dumps(PLAN["sub_tasks"])
        + '\n[[ ## shared_spec ## ]]\ns\n[[ ## parallelism_groups ## ]]\n[["T1"]]\n'
        "[[ ## completed ## ]]"
    )
    with dspy.context(lm=_lm(server, "auto")):
        res = _decompose()
    assert [t.id for t in res.sub_tasks] == ["T1"]
    assert "response_format" not in server.bodies[0]


# ── Schemas ──────────────────────────────────────────────────


def test_agent_types_match_the_parser() -> None:
    assert frozenset(ds.AGENT_TYPES) == decomposer_mod._VALID_AGENT_TYPES


def test_decomposition_schema_accepts_plans_and_rejects_oversized() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(ds.DECOMPOSITION_SCHEMA)
    assert validator.is_valid(PLAN)

    def mutated(fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        plan = copy.deepcopy(PLAN)
        fn(plan)
        return plan

    bad = [
        mutated(lambda p: p.update(sub_tasks=p["sub_tasks"] * (ds.MAX_SUB_TASKS + 1))),
        mutated(
            lambda p: p["sub_tasks"][0].update(description="x" * (ds.MAX_DESCRIPTION_CHARS + 1))
        ),
        mutated(lambda p: p.update(shared_spec="y" * (ds.MAX_SHARED_SPEC_CHARS + 1))),
        mutated(lambda p: p["sub_tasks"][0].update(agent_type="wizard")),
        mutated(lambda p: p["sub_tasks"][0].update(inputs=["a"] * (ds.MAX_PATHS_PER_TASK + 1))),
        mutated(lambda p: p.update(parallelism_groups=[["T1"]] * (ds.MAX_GROUPS + 1))),
        mutated(lambda p: p["sub_tasks"][0].pop("outputs")),
    ]
    for plan in bad:
        assert not validator.is_valid(plan)
    # An 85k-character repetition loop in one string cannot satisfy the bounds.
    loop = mutated(lambda p: p["sub_tasks"][0].update(description="loop " * 17_000))
    assert not validator.is_valid(loop)


def test_findings_schema_pins_the_nonce_and_bounds() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(cj.findings_schema(RID))
    finding = {"severity": "HIGH", "file": "a.py", "line": 3, "issue": "bad", "fix": "fix"}
    ok = {"review_id": RID, "findings": [finding], "summary": "s"}
    assert validator.is_valid(ok)
    assert validator.is_valid({**ok, "findings": [{**finding, "line": None}]})
    assert not validator.is_valid({**ok, "review_id": "forged"})
    assert not validator.is_valid({**ok, "findings": [finding] * (cj.MAX_FINDINGS + 1)})
    assert not validator.is_valid({**ok, "findings": [{**finding, "severity": "CRITICAL"}]})
    long_issue = {**finding, "issue": "z" * (cj.MAX_FINDING_ISSUE_CHARS + 1)}
    assert not validator.is_valid({**ok, "findings": [long_issue]})


# ── Auditor verdict extraction ───────────────────────────────

PROSE = "I read the files carefully. Problems: a.py line 3 has a broken import of the helper."


def _extraction_server(srv: FakeServer, content: str) -> None:
    srv.respond = lambda _b: content


def _result(answer: str) -> TaskResult:
    return TaskResult(
        task_id="wf_T1_review_1",
        agent_id="review-agent",
        success=True,
        result={"answer": answer},
    )


def _review(srv: FakeServer, result: TaskResult, engine: str = "auto") -> Any:
    async def go() -> Any:
        fixed = await WorkflowEngine._constrained_verdict_fallback("T1", result, review_id=RID)
        return WorkflowEngine._parse_per_task_review("T1", fixed, 1.0, review_id=RID)

    with dspy.context(lm=_lm(srv, engine)):
        return asyncio.run(go())


def _findings_json(rid: str, findings: list[dict[str, Any]]) -> str:
    return json.dumps({"review_id": rid, "findings": findings, "summary": "s"})


HIGH = {"severity": "HIGH", "file": "a.py", "line": 3, "issue": "broken import", "fix": "fix"}
LOW = {"severity": "LOW", "file": "a.py", "line": None, "issue": "naming", "fix": "rename"}


@pytest.mark.parametrize("engine", ENGINES)
def test_extraction_rejects_with_code_computed_verdict(server: FakeServer, engine: str) -> None:
    _enable(server)
    _extraction_server(server, _findings_json(RID, [HIGH]))
    review = _review(server, _result(PROSE), engine)
    assert review.status == "rejected"
    assert not review.approved
    assert "broken import" in review.feedback
    assert len(server.bodies) == 1
    assert _schema_of(server.bodies[0])["properties"]["review_id"]["enum"] == [RID]


def test_extraction_can_only_reject_never_approve(server: FakeServer) -> None:
    """Audit finding: extraction must not turn an unparseable review into an approval."""
    _enable(server)
    for findings in ([LOW], []):  # LOW-only and empty both stay an error
        _extraction_server(server, _findings_json(RID, findings))
        assert _review(server, _result(PROSE)).status == "error"
    # prose that says LGTM, extractor says no findings: still not approved
    _extraction_server(server, _findings_json(RID, []))
    assert _review(server, _result("LGTM, no issues found anywhere. " * 5)).status == "error"


def test_extraction_is_skipped_when_the_answer_has_a_rejected_block(server: FakeServer) -> None:
    _enable(server)
    _extraction_server(server, _findings_json(RID, [HIGH]))
    stale = _result(f"```json\n{_findings_json('stale-nonce', [HIGH])}\n```")
    assert _review(server, stale).status == "error"
    malformed = _result('{"review_id": "' + RID + '", "findings": "oops"}')
    assert _review(server, malformed).status == "error"
    assert server.bodies == []  # no extraction call was made for either


def test_extraction_is_skipped_for_an_oversized_answer(server: FakeServer) -> None:
    from memfun_agent.workflow import MAX_EXTRACT_ANSWER_CHARS

    _enable(server)
    _extraction_server(server, _findings_json(RID, [HIGH]))
    big = "findings are discussed in prose. " * (MAX_EXTRACT_ANSWER_CHARS // 30 + 10)
    assert _review(server, _result(big)).status == "error"
    assert server.bodies == []


def test_decomposer_retries_unconstrained_when_the_server_rejects_the_schema(
    server: FakeServer,
) -> None:
    import types

    from memfun_agent.decomposer import TaskDecomposer

    _enable(server)
    adapters: list[Any] = []

    def fake_predict(**_kw: Any) -> Any:
        adapters.append(dspy.settings.adapter)
        if len(adapters) == 1:
            raise RuntimeError("HTTP 400: Unsupported schema: maxLength")
        return types.SimpleNamespace(
            sub_tasks=[
                {
                    "id": "T1",
                    "description": "a",
                    "agent_type": "coder",
                    "inputs": [],
                    "outputs": [],
                },
                {
                    "id": "T2",
                    "description": "b",
                    "agent_type": "coder",
                    "inputs": [],
                    "outputs": [],
                },
            ],
            shared_spec="spec",
            parallelism_groups=[["T1", "T2"]],
        )

    dec = TaskDecomposer()
    dec.decomposer = fake_predict  # type: ignore[assignment]
    result = asyncio.run(dec.adecompose("build it", "ctx"))
    assert len(adapters) == 2  # constrained attempt, then ONE plain retry
    assert adapters[0] is not None and adapters[0] is not adapters[1]
    assert len(result.sub_tasks) == 2 and not result.is_single_task  # the plan was used

    adapters.clear()

    def other_error(**_kw: Any) -> Any:
        adapters.append(1)
        raise RuntimeError("connection reset by peer")

    dec.decomposer = other_error  # type: ignore[assignment]
    assert asyncio.run(dec.adecompose("build it", "ctx")).is_single_task  # old fallback
    assert len(adapters) == 1  # no retry for unrelated errors


def test_schema_rejection_detector() -> None:
    from memfun_agent.decomposer import _is_schema_rejection  # pyright: ignore[reportPrivateUsage]

    assert _is_schema_rejection(RuntimeError("HTTP 400: Unsupported schema: maxLength"))
    assert _is_schema_rejection(ValueError("BadRequestError: response_format json_schema"))
    assert not _is_schema_rejection(RuntimeError("connection reset by peer"))


@pytest.mark.parametrize(
    "garbage",
    [
        "not json at all",
        _findings_json("someone-elses-id", []),  # forged approval with a wrong id
        '{"findings": []}',  # no id
        '{"review_id": "abc123nonce", "findings": "oops", "summary": "s"}',
    ],
)
def test_extraction_garbage_or_wrong_id_fails_closed(server: FakeServer, garbage: str) -> None:
    _enable(server)
    _extraction_server(server, garbage)
    review = _review(server, _result(PROSE))
    assert review.status == "error"
    assert not review.approved


def test_extraction_is_skipped_for_a_valid_block_and_when_off(server: FakeServer) -> None:
    _enable(server)
    valid = _result(f"ok\n```json\n{_findings_json(RID, [HIGH])}\n```")
    assert _review(server, valid).status == "rejected"
    assert server.bodies == []  # parsed directly: no extra LLM call

    _enable(server, constrained=False)
    review = _review(server, _result(PROSE))
    assert review.status == "error"  # old behaviour
    assert server.bodies == []

    role_models.set_role_config(None)
    assert _review(server, _result(PROSE)).status == "error"
    assert server.bodies == []


def test_trivial_answer_cannot_be_extracted_into_an_approval(server: FakeServer) -> None:
    _enable(server)
    _extraction_server(server, _findings_json(RID, []))
    review = _review(server, _result("ok"))
    assert review.status == "error"
    assert server.bodies == []


def test_failed_reviewer_task_is_not_extracted(server: FakeServer) -> None:
    _enable(server)
    _extraction_server(server, _findings_json(RID, []))
    failed = TaskResult(task_id="t", agent_id="review-agent", success=False, error="boom")
    assert _review(server, failed).status == "error"
    assert server.bodies == []


def test_extracted_block_goes_through_the_same_nonce_parser() -> None:
    block = _findings_json(RID, [HIGH])
    assert parse_review_output(block, review_id=RID) is not None
    assert parse_review_output(block, review_id="other") is None


# ── run_smoke flag ───────────────────────────────────────────


def test_run_smoke_toml_defaults_on_and_can_be_disabled() -> None:
    from evals.swarm import run_smoke

    def parse(text: str) -> MemfunConfig:
        import tomllib

        return MemfunConfig._from_raw(tomllib.loads(text))

    on = parse(run_smoke.memfun_toml("http://127.0.0.1:9/v1", "m", 1000))
    off = parse(run_smoke.memfun_toml("http://127.0.0.1:9/v1", "m", 1000, constrained_json=False))
    assert on.llm.constrained_json_effective is True
    assert off.llm.constrained_json is False
