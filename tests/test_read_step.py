"""Runs that ended without a patch, and what stops each from happening again.

Seen on real repositories: the code to change sat in a part of a large file that was not
shown, so the solver asked to read it, was not answered, and the run ended "successfully"
with nothing edited; the planner answered with a bare list of files and the parse error
ended the solve; a task text with ``[/x]`` in it crashed the command before its first step.
Hermetic: a tiny project, the model replaced by fixed answers.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent import context_first as cf
from memfun_agent import read_requests as rr
from memfun_agent.workflow import NO_EDITS_PREFIX, no_edit_reason

if TYPE_CHECKING:
    from pathlib import Path

BIG = (
    "def unrelated():\n    return 0\n\n\n"
    + "X = 1\n" * 900
    + "\n\ndef apply_tax(v):\n    return v * 1.2\n"
)
ISSUE = "Prices come out wrong: the tax must be 25 percent, not 20."
EDIT = {"op": "edit_file", "path": "shop/big_rules.py", "old": "v * 1.2", "new": "v * 1.25"}
READ = {"op": "run_cmd", "cmd": "sed -n '905,908p' shop/big_rules.py"}


@pytest.fixture(autouse=True)
def in_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent runs from the project root: relative paths in its operations mean that."""
    monkeypatch.chdir(tmp_path)


def project(root: Path) -> Path:
    files = {
        "shop/__init__.py": "",
        "shop/big_rules.py": BIG,
        "shop/cart.py": "def total(items):\n    return sum(items)\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, "utf-8")
    return root


class Replies:
    """Stands in for the solver's DSPy predictor: fixed replies, and what each call was shown."""

    def __init__(self, *operations: list[dict[str, Any]]) -> None:
        self.left = list(operations)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        ops = self.left.pop(0) if len(self.left) > 1 else self.left[0]
        return SimpleNamespace(reasoning="r", answer="a", operations=json.dumps(ops))


def solver_for(root: Path, replies: Replies, **cfg: Any) -> cf.ContextFirstSolver:
    settings: dict[str, Any] = {
        "enable_consistency_review": False,
        "max_file_bytes": 2_000,
        "verify_commands": ("true",),
        **cfg,
    }
    config = cf.ContextFirstConfig(**settings)
    solver = cf.ContextFirstSolver(project_root=root, config=config)
    solver.solver.predictor = replies  # type: ignore[assignment]
    return solver


# ── the read step ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_solver_that_asks_to_read_is_answered_and_then_edits(tmp_path: Path) -> None:
    root = project(tmp_path)
    replies = Replies([READ], [EDIT])
    solver = solver_for(root, replies, read_rounds=2)
    result = await solver.asolve(query=ISSUE)

    first, second = replies.calls
    # the code to change is NOT in what the solver is first shown: only head and outline
    assert "return v * 1.2" not in first["full_context"]
    assert "only the head and an outline are shown" in first["full_context"]
    assert "=== READING MORE BEFORE YOU EDIT ===" in first["full_context"]
    assert "at most 2 time(s)" in first["full_context"]
    # the second call: the same task, the same context, and what was asked for at its end
    assert second["query"] == first["query"] == ISSUE
    assert second["full_context"].startswith(first["full_context"])
    added = second["full_context"][len(first["full_context"]) :]
    assert added.startswith("\n\n=== WHAT YOU ASKED TO READ (round 1) ===\n")
    assert "--- lines 905-908 of shop/big_rules.py (908 lines in the file) ---" in added
    assert "def apply_tax(v):\n    return v * 1.2" in added
    assert added.rstrip().endswith("ask for more reads instead (1 round(s) left).")
    # and the edit landed
    assert (root / "shop/big_rules.py").read_text("utf-8").endswith("return v * 1.25\n")
    assert result.success and result.method != cf.NO_EDITS_METHOD
    log = [json.loads(x) for x in (root / rr.READ_LOG).read_text("utf-8").splitlines()]
    assert [(r["round"], r["kind"], r["served"]) for r in log] == [(1, "lines", True)]


@pytest.mark.asyncio
async def test_the_read_step_is_bounded_and_a_run_without_an_edit_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solver that never stops asking: two rounds are answered, the edit is demanded once,
    nothing it wrote is ever run, and the run is reported as ``no_edits``."""
    root = project(tmp_path)
    ran: list[Any] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a) or pytest.fail("ran"))
    greedy = [READ, {"op": "run_cmd", "cmd": "touch made-by-cmd"}]
    replies = Replies(greedy)
    solver = solver_for(root, replies, read_rounds=2, verify_commands=())
    result = await solver.asolve(query=ISSUE)

    assert len(replies.calls) == 4  # the first answer, two rounds of reads, one demand
    contexts = [c["full_context"] for c in replies.calls]
    assert "=== WHAT YOU ASKED TO READ (round 2) ===" in contexts[2]
    assert "This was the last round of reads" in contexts[2]
    assert "REFUSED:" in contexts[1] and "touch" in contexts[1]  # the non-read was refused
    assert contexts[3] == contexts[2]  # no third round
    assert replies.calls[3]["query"] == ISSUE + cf.NO_EDIT_NUDGE
    assert not ran and not (root / "made-by-cmd").exists()
    assert result.success is False and result.method == cf.NO_EDITS_METHOD
    assert result.ops == [] and result.files_created == []
    assert (root / "shop/big_rules.py").read_text("utf-8") == BIG


@pytest.mark.asyncio
async def test_without_the_read_step_the_solver_call_is_what_it_was(tmp_path: Path) -> None:
    """Off by default: no offer in the prompt, the same single retry, and a command-only
    reply is still an ordinary user's command (run, and not a failure)."""
    root = project(tmp_path)
    assert cf.ContextFirstConfig().read_rounds == 0 and cf.ContextFirstConfig().offline is False
    replies = Replies([{"op": "run_cmd", "cmd": "true"}])
    solver = solver_for(root, replies, verify_commands=())
    result = await solver.asolve(query=ISSUE)
    assert len(replies.calls) == 2
    assert "READING MORE" not in replies.calls[0]["full_context"]
    assert replies.calls[1]["full_context"] == replies.calls[0]["full_context"]
    assert replies.calls[1]["query"] == ISSUE + cf.NO_EDIT_NUDGE
    assert result.success and result.ops == [("cmd", "true", 0)]
    assert not (root / ".memfun").exists()


@pytest.mark.asyncio
async def test_with_its_own_checks_on_a_run_that_edits_nothing_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark mode without the read step: a command is not a change."""
    from memfun_agent import repo_tests

    monkeypatch.setattr(repo_tests, "candidate_pythons", lambda _root: [])
    root = project(tmp_path)
    replies = Replies([{"op": "run_cmd", "cmd": "true"}])
    solver = solver_for(root, replies, verify_tests=True, verify_commands=())
    result = await solver.asolve(query=ISSUE)
    assert result.success is False and result.method == cf.NO_EDITS_METHOD
    checks = (root / ".memfun/checks.jsonl").read_text("utf-8")
    assert '"kind": "no_edits"' in checks


def test_the_read_switches_are_read_from_the_project_verify_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".memfun").mkdir()
    doc = {"read_rounds": 3, "offline": True}
    (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(doc), "utf-8")
    assert cf.read_project_verify_options(tmp_path) == {}  # not trusted: nothing is read
    monkeypatch.setenv(cf.TRUST_PROJECT_VERIFY_ENV, "1")
    assert cf.read_project_verify_options(tmp_path) == doc
    for bad in ({"read_rounds": 99}, {"read_rounds": True}, {"read_rounds": "2"}, {"offline": 1}):
        (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(bad), "utf-8")
        assert cf.read_project_verify_options(tmp_path) == {}


# ── "no file changed" must not be hidden by memfun's own files ───────────────────────────────


def test_memfuns_own_state_directory_is_not_a_change_of_the_project(tmp_path: Path) -> None:
    """How the earlier rule let these runs through: a run whose only operation is a command
    is judged by ``git status``, and memfun's own ``.memfun`` directory made the tree look
    changed, so a run that had only asked to read counted as a success."""
    git = ("git", "-c", "user.name=t", "-c", "user.email=t@example.com")
    subprocess.run((*git, "init", "-q"), cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n", "utf-8")
    subprocess.run((*git, "add", "a.py"), cwd=tmp_path, check=True)
    subprocess.run((*git, "commit", "-qm", "init"), cwd=tmp_path, check=True)
    (tmp_path / ".memfun").mkdir()
    (tmp_path / ".memfun" / "verify.json").write_text("{}", "utf-8")
    only_a_command = {"ops": [{"type": "cmd", "target": "sed -n '1,5p' a.py", "detail": 0}]}
    reason = no_edit_reason("rlm-coder", only_a_command, tmp_path)
    assert reason is not None and reason.startswith(NO_EDITS_PREFIX)
    (tmp_path / "a.py").write_text("x = 2\n", "utf-8")  # e.g. a formatter really changed it
    assert no_edit_reason("rlm-coder", only_a_command, tmp_path) is None


# ── a planner answer that cannot be parsed ───────────────────────────────────────────────────


class UnparsedError(Exception):
    """What DSPy raises for an answer it cannot map to the fields (it carries the text)."""

    def __init__(self, lm_response: str) -> None:
        super().__init__("Adapter JSONAdapter failed to parse the LM response.")
        self.lm_response = lm_response


def large(root: Path) -> Path:
    project(root)
    for i in range(40):
        (root / f"shop/extra_{i:02d}.py").write_text(f"def helper_{i}():\n    return {i}\n" * 20)
    return root


def test_the_files_an_unparsed_planner_answer_names_are_salvaged() -> None:
    known = ["shop/cart.py", "shop/big_rules.py", "README.md"]
    bare = '[\n "shop/big_rules.py",\n "./shop/cart.py",\n "not/a/file.py"\n]'
    assert cf.salvage_planned_files(bare, known) == ["shop/big_rules.py", "shop/cart.py"]
    as_object = '```json\n{"files_to_read": ["shop/cart.py"], "reasoning": "x"}\n```'
    assert cf.salvage_planned_files(as_object, known) == ["shop/cart.py"]
    prose = "I would read `shop/cart.py` first, then shop/big_rules.py. </function>"
    assert cf.salvage_planned_files(prose, known) == ["shop/cart.py", "shop/big_rules.py"]
    assert cf.salvage_planned_files("no idea", known) == []
    assert cf.salvage_planned_files('["a.py", "b.py"]', None) == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_a_planner_answer_that_is_a_bare_list_does_not_end_the_solve(tmp_path: Path) -> None:
    root = large(tmp_path)
    replies = Replies([EDIT])
    solver = solver_for(root, replies, max_context_bytes=1_000, max_file_bytes=0)

    def bare_list(**_kwargs: Any) -> Any:
        raise UnparsedError('["shop/big_rules.py", "shop/cart.py"]')

    solver.planner.predictor = bare_list  # type: ignore[assignment]
    result = await solver.asolve(query=ISSUE)
    assert result.success and result.method == "context_first_planned"
    ctx = replies.calls[0]["full_context"]
    assert "=== FILE: shop/big_rules.py ===" in ctx and "=== FILE: shop/cart.py ===" in ctx
    assert "extra_07.py" not in ctx
    assert (root / "shop/big_rules.py").read_text("utf-8").endswith("return v * 1.25\n")


@pytest.mark.asyncio
async def test_a_planner_answer_that_names_no_file_falls_back_to_a_search(tmp_path: Path) -> None:
    root = large(tmp_path)
    replies = Replies([EDIT])
    solver = solver_for(root, replies, max_context_bytes=1_000, max_file_bytes=0)

    def nonsense(**_kwargs: Any) -> Any:
        raise UnparsedError("</function>\n<tool_call>\n<function=completed>\n[]")

    solver.planner.predictor = nonsense  # type: ignore[assignment]
    result = await solver.asolve(query="`apply_tax` must use 25 percent, not 20.")
    assert result.success
    assert "=== FILE: shop/big_rules.py ===" in replies.calls[0]["full_context"]


@pytest.mark.asyncio
async def test_a_planner_failure_that_is_not_an_answer_still_fails_the_solve(
    tmp_path: Path,
) -> None:
    root = large(tmp_path)
    solver = solver_for(root, Replies([EDIT]), max_context_bytes=1_000)

    def no_model(**_kwargs: Any) -> Any:
        raise ConnectionError("no route to the model")

    solver.planner.predictor = no_model  # type: ignore[assignment]
    result = await solver.asolve(query=ISSUE)
    assert result.success is False and result.method == "context_first_error"


@pytest.mark.asyncio
async def test_the_planner_asks_for_a_fresh_sample_only_on_the_workflows_retry(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []

    def predictor(**kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs)
        return SimpleNamespace(files_to_read=["a.py"], search_patterns=[], reasoning="")

    planner = cf.ContextPlanner()
    planner.predictor = predictor  # type: ignore[assignment]
    await planner.aplan(query="q", file_manifest="m", project_summary="s")
    cf.set_attempt_salt(1)
    try:
        await planner.aplan(query="q", file_manifest="m", project_summary="s")
    finally:
        cf.set_attempt_salt(0)
    assert "config" not in seen[0]  # the first attempt is the call it always was
    assert seen[1]["config"] == {"rollout_id": 10}  # not the cached answer of the failed attempt


# ── no web step ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_offline_the_planner_is_not_offered_the_web_and_nothing_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = large(tmp_path)
    replies = Replies([EDIT])
    solver = solver_for(root, replies, max_context_bytes=1_000, max_file_bytes=0, offline=True)
    asked: list[Any] = []

    def plan(**kwargs: Any) -> SimpleNamespace:
        asked.append(kwargs)
        return SimpleNamespace(
            files_to_read=["shop/big_rules.py"],
            search_patterns=[],
            web_searches='["how to compute tax"]',  # a model that answers it unasked
            reasoning="",
        )

    async def no_web(_query: str) -> str:
        pytest.fail("offline: the web is never searched")

    monkeypatch.setattr(solver.gatherer, "_web_search", no_web)
    monkeypatch.setattr(solver.gatherer, "_web_fetch", no_web)
    original = cf.dspy.Predict
    built: list[Any] = []

    def predict(signature: Any) -> Any:
        built.append(signature)
        return plan

    monkeypatch.setattr(cf.dspy, "Predict", predict)
    try:
        result = await solver.asolve(query=ISSUE, category="web")
    finally:
        monkeypatch.setattr(cf.dspy, "Predict", original)
    assert result.success and len(asked) == 1 and len(built) == 1
    assert list(built[0].output_fields) == ["files_to_read", "search_patterns", "reasoning"]
    assert "web" not in json.dumps(list(built[0].fields)).lower()
    assert "WEB SEARCH RESULTS" not in replies.calls[0]["full_context"]


def test_online_the_planner_keeps_its_web_step() -> None:
    planner = cf.ContextPlanner()
    assert "web_searches" in planner.predictor.signature.output_fields


# ── a solver reply that cannot be split into its fields ──────────────────────────────────────

TOOL_CALLS = (
    "I need to see the code first.\n<tool_call>\n"
    '{"op": "read_file", "path": "shop/big_rules.py", "start_line": 905, "end_line": 908}\n'
    '<tool_call>\n{"op": "list_dir", "path": "shop"}'
)


def test_operations_are_found_in_a_reply_that_has_no_fields() -> None:
    assert [op["op"] for op in cf.salvage_operations(TOOL_CALLS)] == ["read_file", "list_dir"]
    prose = 'A note [1], then:\n[{"op": "edit_file", "path": "a.py", "old": "x", "new": "y"}] ok'
    assert cf.salvage_operations(prose) == [
        {"op": "edit_file", "path": "a.py", "old": "x", "new": "y"}
    ]
    spread = 'Change this:\n{\n  "op": "edit_file", "path": "a.py",\n  "old": "x", "new": "y"\n}'
    assert cf.salvage_operations(spread)[0]["path"] == "a.py"
    assert cf.salvage_operations("I am not sure what to do.") == []
    xml = (
        "Let me look.\n<tool_call>\n<function=search>\n<parameter=pattern>\nparse_enum\n"
        "</parameter>\n<parameter=path>\nsrc\n</parameter>\n</function>\n</tool_call>\n"
        "<function=completed>\n</function>"
    )
    assert cf.salvage_operations(xml) == [{"op": "search", "pattern": "parse_enum", "path": "src"}]
    named = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    assert cf.salvage_operations(named) == [{"op": "read_file", "path": "a.py"}]
    assert cf.salvage_operations('<tool_call>{"name": "bash", "arguments": {"c": "ls"}}') == []


@pytest.mark.asyncio
async def test_the_whole_reply_is_searched_not_the_piece_the_adapter_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seen on the box: the reply held its fields as one JSON object after some code, the
    adapter kept only the first braces it found (a code snippet) and reported that; the
    operations were in the full reply, which the model's call history still has."""
    root = project(tmp_path)
    full = (
        "The resolver does:\n{ k: v(aliases=aliases) for k, v in classes.items() }\n"
        + json.dumps({"reasoning": "r", "answer": "a", "operations": [EDIT]})
    )
    history: list[dict[str, Any]] = [{"outputs": ["an earlier, unrelated call"]}]
    monkeypatch.setattr(cf, "_lm_history", lambda: history)

    def predictor(**_kwargs: Any) -> SimpleNamespace:
        history.append({"outputs": [{"text": full}]})
        raise UnparsedError("{ k: v(aliases=aliases) for k, v in classes.items() }")

    solver = solver_for(root, Replies([EDIT]))
    solver.solver.predictor = predictor  # type: ignore[assignment]
    result = await solver.asolve(query=ISSUE)
    assert result.success
    assert (root / "shop/big_rules.py").read_text("utf-8").endswith("return v * 1.25\n")


@pytest.mark.asyncio
async def test_a_solver_reply_without_its_fields_does_not_end_the_attempt(tmp_path: Path) -> None:
    """Seen on the box: the reply after a read offer came as tool-call markup, the parse error
    ended the whole attempt and the task was started again from scratch."""
    root = project(tmp_path)
    calls: list[dict[str, Any]] = []

    def predictor(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnparsedError(TOOL_CALLS)
        return SimpleNamespace(reasoning="r", answer="a", operations=json.dumps([EDIT]))

    solver = solver_for(root, Replies([EDIT]), read_rounds=2)
    solver.solver.predictor = predictor  # type: ignore[assignment]
    result = await solver.asolve(query=ISSUE)
    assert result.success and len(calls) == 2
    assert "def apply_tax(v):\n    return v * 1.2" in calls[1]["full_context"]
    assert (root / "shop/big_rules.py").read_text("utf-8").endswith("return v * 1.25\n")


@pytest.mark.asyncio
async def test_a_solver_reply_with_nothing_to_salvage_fails_as_before(tmp_path: Path) -> None:
    root = project(tmp_path)

    asked: list[str] = []

    def predictor(**kwargs: Any) -> SimpleNamespace:
        asked.append(kwargs["query"])
        raise UnparsedError("I cannot decide. " * 10)

    solver = solver_for(root, Replies([EDIT]))
    solver.solver.predictor = predictor  # type: ignore[assignment]
    result = await solver.asolve(query=ISSUE)
    assert result.success is False and result.method == "context_first_error"
    # one retry in place, told what was wrong, before the failure it always was
    assert asked == [ISSUE, ISSUE + cf.TRUNCATION_NUDGE]
