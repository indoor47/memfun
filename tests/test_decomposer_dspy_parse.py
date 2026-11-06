"""TaskDecomposition parses object sub-tasks under dspy 3.4 (real adapters).

Regression for the swarm silently degrading to a single agent:
``TaskDecomposition.sub_tasks`` was annotated ``list[str]`` while its own
description (and ``_parse_sub_tasks``) want objects.  dspy 3.4's adapters
validate parsed values against the annotation, so a correct model answer
failed with ``Failed to parse field sub_tasks: 5 validation errors for
list[str] ... Input should be a valid string [type=string_type ...
input_type=dict]`` and ``TaskDecomposer`` fell back to one coder task.

``SMOKE_LOG_JSON`` is the verbatim LM response from that failing live
``memfun swarm`` run.  The tests drive dspy's real ``JSONAdapter`` /
``ChatAdapter`` parse paths with ``dspy.utils.DummyLM`` -- no network.
"""

# SMOKE_LOG_JSON is kept byte-for-byte as the model emitted it (long lines).
# ruff: noqa: E501

from __future__ import annotations

import json
from typing import Any

import dspy
import pytest
from dspy.utils import DummyLM
from memfun_agent.decomposer import SubTask, TaskDecomposer
from memfun_agent.signatures import TaskDecomposition

SMOKE_LOG_JSON = """\
{
  "sub_tasks": [
    {
      "id": "T1",
      "description": "Implement slugify(text, sep='-') in textkit/slugify.py: lowercase, NFKD-normalise and drop non-ASCII, collapse every run of non-[a-z0-9] chars into a single sep, strip leading/trailing sep, return '' for empty result. Keep the existing signature and docstring.",
      "agent_type": "coder",
      "inputs": ["textkit/slugify.py"],
      "outputs": ["textkit/slugify.py"],
      "depends_on": [],
      "max_iterations": 8
    },
    {
      "id": "T2",
      "description": "Implement wrap(text, width) in textkit/wrap.py: greedy word wrap on whitespace-separated words joined by single spaces; a word longer than width is split into chunks of exactly width (last chunk shorter); width < 1 raises ValueError; text with no words returns []. Keep the existing signature and docstring.",
      "agent_type": "coder",
      "inputs": ["textkit/wrap.py"],
      "outputs": ["textkit/wrap.py"],
      "depends_on": [],
      "max_iterations": 8
    },
    {
      "id": "T3",
      "description": "Implement to_roman(n) and from_roman(s) in textkit/roman.py: to_roman handles 1..3999 in canonical subtractive form (IV, IX, XL, XC, CD, CM) and raises ValueError outside range; from_roman is case-insensitive and rejects non-canonical input (e.g. 'IIII', 'IC', '', 'MMMM') with ValueError. Keep existing signatures and docstrings.",
      "agent_type": "coder",
      "inputs": ["textkit/roman.py"],
      "outputs": ["textkit/roman.py"],
      "depends_on": [],
      "max_iterations": 10
    },
    {
      "id": "T4",
      "description": "Run the existing test suite under tests/ against the three implemented modules and verify all pass; report any failures and fix the corresponding module if needed.",
      "agent_type": "test",
      "inputs": ["tests/", "textkit/slugify.py", "textkit/wrap.py", "textkit/roman.py"],
      "outputs": ["textkit/slugify.py", "textkit/wrap.py", "textkit/roman.py"],
      "depends_on": ["T1", "T2", "T3"],
      "max_iterations": 10
    },
    {
      "id": "T5",
      "description": "Review the three implementations for correctness, edge cases, and adherence to the contract (signatures unchanged, no test files modified, canonical forms, error handling). Provide a final QA pass.",
      "agent_type": "review",
      "inputs": ["textkit/slugify.py", "textkit/wrap.py", "textkit/roman.py", "tests/"],
      "outputs": ["textkit/slugify.py", "textkit/wrap.py", "textkit/roman.py"],
      "depends_on": ["T4"],
      "max_iterations": 6
    }
  ],
  "shared_spec": "Package: textkit. Each utility lives in its own module (slugify.py, wrap.py, roman.py). Do NOT change any public function signatures or docstrings in the stub files; only fill in the bodies. Do NOT modify anything under tests/. Conventions: pure functions, no side effects, no external dependencies beyond the Python standard library (use unicodedata for NFKD). Error handling: raise ValueError with a clear message for invalid inputs (width < 1 in wrap; n outside 1..3999 in to_roman; non-canonical/empty/invalid strings in from_roman). Naming: keep module-level function names exactly as stubbed (slugify, wrap, to_roman, from_roman). Use single quotes for strings, 4-space indentation, and keep code readable and minimal. from_roman must be case-insensitive and must reject non-canonical forms (e.g. 'IIII', 'IC', '', 'MMMM') with ValueError. to_roman must produce canonical subtractive form only.",
  "parallelism_groups": [
    ["T1", "T2", "T3"],
    ["T4"],
    ["T5"]
  ]
}
"""
SMOKE_LOG_RESPONSE = f"```json\n{SMOKE_LOG_JSON}\n```"
ANSWER: dict[str, Any] = json.loads(SMOKE_LOG_JSON)
EXPECTED_SUB_TASKS = [SubTask(**raw) for raw in ANSWER["sub_tasks"]]
EXPECTED_GROUPS = [["T1", "T2", "T3"], ["T4"], ["T5"]]

# None -> dspy's default ChatAdapter (with its JSONAdapter fallback).
ADAPTERS = [
    pytest.param(dspy.JSONAdapter, id="json_adapter"),
    pytest.param(None, id="default_chat_adapter"),
]


def _lm(adapter_cls: type[dspy.Adapter] | None, n: int = 1) -> DummyLM:
    adapter = adapter_cls() if adapter_cls is not None else dspy.ChatAdapter()
    return DummyLM([ANSWER] * n, adapter=adapter)


def test_json_adapter_parses_verbatim_smoke_response() -> None:
    parsed = dspy.JSONAdapter().parse(TaskDecomposition, SMOKE_LOG_RESPONSE)

    assert parsed["sub_tasks"] == ANSWER["sub_tasks"]
    assert all(isinstance(t, dict) for t in parsed["sub_tasks"])
    assert parsed["shared_spec"] == ANSWER["shared_spec"]
    assert parsed["parallelism_groups"] == EXPECTED_GROUPS


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_predict_task_decomposition_returns_object_sub_tasks(
    adapter_cls: type[dspy.Adapter] | None,
) -> None:
    lm = _lm(adapter_cls)
    adapter = adapter_cls() if adapter_cls is not None else None
    with dspy.context(lm=lm, adapter=adapter):  # pyright: ignore[reportUnknownMemberType]
        pred: Any = dspy.Predict(TaskDecomposition)(
            task_description="Implement three text utilities",
            project_context="textkit/ package with stubs",
        )

    assert pred.sub_tasks == ANSWER["sub_tasks"]
    assert pred.shared_spec == ANSWER["shared_spec"]
    assert pred.parallelism_groups == EXPECTED_GROUPS
    # Parsed on the first attempt: no ChatAdapter -> JSONAdapter retry call.
    history: list[Any] = lm.history  # pyright: ignore[reportUnknownMemberType]
    assert len(history) == 1


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
async def test_adecompose_builds_multi_task_dag_not_single_fallback(
    adapter_cls: type[dspy.Adapter] | None,
) -> None:
    # Spare answers in case hierarchical recursion asks again; it must not
    # change the result for these already-fine-grained sub-tasks.
    lm = _lm(adapter_cls, n=8)
    adapter = adapter_cls() if adapter_cls is not None else None
    with dspy.context(lm=lm, adapter=adapter):  # pyright: ignore[reportUnknownMemberType]
        result = await TaskDecomposer().adecompose(
            "Implement three text utilities", "textkit/ package with stubs"
        )

    assert result.is_single_task is False
    assert result.sub_tasks == EXPECTED_SUB_TASKS
    assert result.parallelism_groups == EXPECTED_GROUPS
    assert result.shared_spec == ANSWER["shared_spec"]


# ── Reply shapes: objects, plain strings, and a mix ────────────

_OBJ_B: dict[str, Any] = {
    "id": "T2",
    "description": "Implement wrap(text, width) in textkit/wrap.py",
    "agent_type": "coder",
    "inputs": ["textkit/wrap.py"],
    "outputs": ["textkit/wrap.py"],
    "depends_on": [],
    "max_iterations": 8,
}
_SHAPES: dict[str, list[Any]] = {
    "dict_only": [
        {
            **_OBJ_B,
            "id": "T1",
            "description": "Implement slugify in textkit/slugify.py",
            "inputs": ["textkit/slugify.py"],
            "outputs": ["textkit/slugify.py"],
        },
        _OBJ_B,
    ],
    "str_only": ["Implement slugify in textkit/slugify.py", "Implement wrap in textkit/wrap.py"],
    "mixed": ["Implement slugify in textkit/slugify.py", _OBJ_B],
}
# (id, agent_type, description) per shape after TaskDecomposer parsing.
_EXPECTED_IDS: dict[str, list[tuple[str, str, str]]] = {
    "dict_only": [
        ("T1", "coder", "Implement slugify in textkit/slugify.py"),
        ("T2", "coder", _OBJ_B["description"]),
    ],
    "str_only": [
        ("T1", "coder", "Implement slugify in textkit/slugify.py"),
        ("T2", "coder", "Implement wrap in textkit/wrap.py"),
    ],
    "mixed": [
        ("T1", "coder", "Implement slugify in textkit/slugify.py"),
        ("T2", "coder", _OBJ_B["description"]),
    ],
}
SHAPES = [pytest.param(name, id=name) for name in _SHAPES]


def _shape_answer(shape: str) -> dict[str, Any]:
    return {
        "sub_tasks": _SHAPES[shape],
        "shared_spec": "Keep signatures; stdlib only.",
        "parallelism_groups": [["T1", "T2"]],
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("adapter_cls", ADAPTERS)
def test_predict_accepts_dict_str_and_mixed_sub_tasks(
    adapter_cls: type[dspy.Adapter] | None, shape: str
) -> None:
    adapter = adapter_cls() if adapter_cls is not None else None
    lm = DummyLM([_shape_answer(shape)], adapter=adapter or dspy.ChatAdapter())
    with dspy.context(lm=lm, adapter=adapter):  # pyright: ignore[reportUnknownMemberType]
        pred: Any = dspy.Predict(TaskDecomposition)(task_description="t", project_context="c")

    assert pred.sub_tasks == _SHAPES[shape]
    history: list[Any] = lm.history  # pyright: ignore[reportUnknownMemberType]
    assert len(history) == 1


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("adapter_cls", ADAPTERS)
async def test_adecompose_multi_task_for_dict_str_and_mixed_replies(
    adapter_cls: type[dspy.Adapter] | None, shape: str
) -> None:
    adapter = adapter_cls() if adapter_cls is not None else None
    lm = DummyLM([_shape_answer(shape)] * 8, adapter=adapter or dspy.ChatAdapter())
    with dspy.context(lm=lm, adapter=adapter):  # pyright: ignore[reportUnknownMemberType]
        result = await TaskDecomposer().adecompose("Implement two utilities", "textkit/")

    assert result.is_single_task is False
    got = [(t.id, t.agent_type, t.description) for t in result.sub_tasks]
    assert got == _EXPECTED_IDS[shape]
    assert result.parallelism_groups == [["T1", "T2"]]
