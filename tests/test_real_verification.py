"""The context-first solver verifying with the project's OWN tests, a reproduction script and
repository-wide search.  Hermetic: a tiny project, the model replaced by fixed answers, the
tests' interpreter standing in for the project's."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent import context_first as cf
from memfun_agent import repo_tests

if TYPE_CHECKING:
    from pathlib import Path

CART = "from shop.tax import RATE\n\n\ndef total(items):\n    return sum(items)\n"
TESTS = (
    "from shop.cart import total\n\n\ndef test_total():\n    assert total([1, 2]) == 3\n\n\n"
    "def test_old_failure():\n    assert total([]) == 99\n"
)
ISSUE = "`total` must add tax: `total([100])` should be 120 with RATE 0.2."


def project(root: Path) -> Path:
    files = {
        "shop/__init__.py": "",
        "shop/cart.py": CART,
        "shop/tax.py": "RATE = 0.2\n",
        "test/test_cart.py": TESTS,
        "test/test_far.py": "def test_unrelated():\n    assert False\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, "utf-8")
    return root


def edit(root: Path, old: str, new: str, rel: str = "shop/cart.py") -> dict[str, Any]:
    return {"op": "edit_file", "path": str(root / rel), "old": old, "new": new}


class Model:
    """The solver's model calls, answered from fixed lists; records what each was shown."""

    def __init__(self, solver: cf.ContextFirstSolver, solve: list[Any], fixes: list[Any]) -> None:
        self.solve, self.fixes = list(solve), list(fixes)
        self.solve_contexts: list[str] = []
        self.fix_errors: list[str] = []
        self.fix_queries: list[str] = []
        self.repro_feedback: list[str] = []
        self.scripts: list[str] = []
        #: Answers of the failing-test classification (none left: every test is a regression).
        self.triage: list[dict[int, str]] = []
        self.triage_asked: list[tuple[str, str]] = []  # (the change, the failing tests)
        solver.solver.asolve = self.asolve  # type: ignore[method-assign]
        solver.fix_solver.afix = self.afix  # type: ignore[method-assign]
        solver.repro_solver.awrite = self.awrite  # type: ignore[method-assign]
        solver.test_triage.aclassify = self.aclassify  # type: ignore[method-assign]

    async def asolve(self, query: str, full_context: str) -> cf.SolveResult:
        self.solve_contexts.append(full_context)
        return cf.SolveResult(reasoning="", answer="done", operations=self.solve.pop(0))

    async def afix(self, query: str, full_context: str, verification_errors: str) -> list[Any]:
        self.fix_errors.append(verification_errors)
        self.fix_queries.append(query)
        return self.fixes.pop(0) if self.fixes else []

    async def aclassify(self, query: str, change: str, failing_tests: str) -> dict[int, str]:
        self.triage_asked.append((change, failing_tests))
        return self.triage.pop(0) if self.triage else {}

    async def awrite(self, query: str, full_context: str, feedback: str) -> str:
        self.repro_feedback.append(feedback)
        return self.scripts.pop(0) if self.scripts else ""


@pytest.fixture
def root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(repo_tests, "candidate_pythons", lambda _root: [sys.executable])
    return project(tmp_path)


def make_solver(root: Path, statuses: list[str], **cfg: Any) -> cf.ContextFirstSolver:
    config = cf.ContextFirstConfig(enable_consistency_review=False, **cfg)
    return cf.ContextFirstSolver(project_root=root, config=config, on_status=statuses.append)


@pytest.mark.asyncio
async def test_a_change_that_breaks_an_import_is_caught_and_fixed(root: Path) -> None:
    """Seen on a real repository: the patch imported a name that does not exist and no test
    was ever run.  Now the import check and the related tests fail, and the fix sees why."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    broken = edit(root, "from shop.tax import RATE\n", "from shop.tax import RATE, NO_SUCH_NAME\n")
    taxed = edit(root, "return sum(items)", "return sum(list(items))")
    repair = edit(root, "from shop.tax import RATE, NO_SUCH_NAME\n", "from shop.tax import RATE\n")
    model = Model(solver, solve=[[broken, taxed]], fixes=[[repair]])
    result = await solver.asolve(query=ISSUE)
    assert result.success and len(model.fix_errors) == 1
    errors = model.fix_errors[0]
    assert "import check of the changed modules: shop.cart" in errors
    assert "cannot import name 'NO_SUCH_NAME'" in errors
    assert "NO_SUCH_NAME" not in (root / "shop/cart.py").read_text("utf-8")
    assert "Python " in model.solve_contexts[0]  # the project's interpreter version
    assert not list(root.rglob("__pycache__"))  # nothing left behind by the checks


@pytest.mark.asyncio
async def test_only_failures_the_change_caused_reach_the_fix(root: Path) -> None:
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    wrong = edit(root, "return sum(items)", "return sum(items) + 1")  # breaks test_total
    right = edit(root, "return sum(items) + 1", "return sum(list(items))")
    model = Model(solver, solve=[[wrong]], fixes=[[right]])
    await solver.asolve(query=ISSUE)
    assert len(model.fix_errors) == 1
    errors = model.fix_errors[0]
    assert "test/test_cart.py::test_total" in errors
    assert "test_old_failure" not in errors  # failed before the change: says nothing about it
    assert "test_far" not in errors  # unrelated to the changed file: never run
    assert any("Verification passed" in s for s in statuses)


@pytest.mark.asyncio
async def test_a_good_change_passes_without_any_fix_call(root: Path) -> None:
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    model = Model(
        solver, solve=[[edit(root, "return sum(items)", "return sum(list(items))")]], fixes=[]
    )
    await solver.asolve(query=ISSUE)
    assert model.fix_errors == [] and any("Verification passed" in s for s in statuses)
    assert any("Verifying with 1 related test files" in s for s in statuses)
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in log] == ["tests_selected", "verify"]
    assert log[0]["files"] == ["test/test_cart.py"] and log[0]["runner"] == "pytest"
    assert log[0]["left_out_failing_before"] == ["test/test_cart.py::test_old_failure"]
    assert log[1]["passed"] is True and len(log[1]["commands"]) == 3  # syntax, import, tests


@pytest.mark.asyncio
async def test_without_opt_in_no_project_test_is_run(root: Path, monkeypatch) -> None:
    monkeypatch.setattr(repo_tests, "discover", lambda *_a, **_k: pytest.fail("not opted in"))
    solver = make_solver(root, [])
    model = Model(
        solver, solve=[[edit(root, "return sum(items)", "return sum(items) + 1")]], fixes=[]
    )
    assert (await solver.asolve(query=ISSUE)).success and model.fix_errors == []


@pytest.mark.asyncio
async def test_tests_that_mention_the_tasks_names_are_used_when_none_match_the_file(
    root: Path,
) -> None:
    """A project that tests through its public interface: no test is named after or imports
    the changed module, but one mentions what the task is about."""
    (root / "shop/engine.py").write_text("def checkout_total(items):\n    return sum(items)\n")
    (root / "test/test_api.py").write_text(
        "import shop\n\n\ndef test_checkout_total_via_api():\n"
        "    from shop import engine as e\n    assert e.checkout_total([1, 2]) == 3\n"
    )
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    wrong = edit(root, "return sum(items)", "return sum(items) - 1", rel="shop/engine.py")
    right = edit(root, "return sum(items) - 1", "return sum(list(items))", rel="shop/engine.py")
    model = Model(solver, solve=[[wrong]], fixes=[[right]])
    await solver.asolve(query="`checkout_total` returns the wrong sum for a basket.")
    assert len(model.fix_errors) == 1
    assert "test/test_api.py::test_checkout_total_via_api" in model.fix_errors[0]
    assert any("Verification passed" in s for s in statuses)


REPRO = (
    "import sys\nfrom shop.cart import total\ngot = total([100])\n"
    "if abs(got - 120) > 1e-9:\n    print('REPRO: FAIL - total([100]) is', got, 'expected 120')\n"
    "    sys.exit(1)\nprint('REPRO: PASS')\n"
)


@pytest.mark.asyncio
async def test_the_reproduction_is_shown_to_the_solver_and_checked_after_the_change(
    root: Path,
) -> None:
    statuses: list[str] = []
    solver = make_solver(root, statuses, reproduce=True, fix_probe_guard="off")
    wrong = edit(root, "return sum(items)", "return sum(items) * RATE")
    right = edit(root, "return sum(items) * RATE", "return sum(items) * (1 + RATE)")
    script_edit = edit(root, "expected 120", "expected 20", rel=cf.REPRO_FILE)
    model = Model(solver, solve=[[wrong, script_edit]], fixes=[[right]])
    model.scripts = [REPRO]
    await solver.asolve(query=ISSUE)
    shown = model.solve_contexts[0]
    assert (
        "=== REPRODUCTION" in shown and "REPRO: FAIL - total([100]) is 100 expected 120" in shown
    )
    assert model.repro_feedback == [""]
    assert (
        len(model.fix_errors) == 1 and "REPRO: FAIL - total([100]) is 20.0" in model.fix_errors[0]
    )
    assert cf.REPRO_LABEL in model.fix_errors[0]  # a label, not the long command line
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")
    assert any("Verification passed" in s for s in statuses)
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in log] == ["reproduction", "verify", "verify"]
    assert log[0]["verdict"] == "reproduced" and log[0]["exit"] == 1
    assert log[0]["script"] == REPRO  # the script is kept in the log, not editable
    assert not (root / cf.REPRO_FILE).exists()  # its file is there only while it runs
    # it still reported what it reported on the unchanged code: that, and only that, counts
    assert log[1]["passed"] is False and log[1]["reproduction"] == "same"
    assert "REPRO: FAIL" in log[1]["reproduction_output"]
    assert log[2]["passed"] is True and log[2]["reproduction"] == "passes"


@pytest.mark.asyncio
async def test_when_the_reproduction_never_passes_the_first_solution_is_kept(root: Path) -> None:
    """A model-written check that no edit satisfies must not leave the code at the last
    desperate attempt: on a tie the earliest best state wins."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, reproduce=True, fix_probe_guard="off", fix_restarts=0)
    first = edit(root, "return sum(items)", "return sum(items) * 2")
    second = edit(root, "return sum(items) * 2", "return sum(items) * 3")
    third = edit(root, "return sum(items) * 3", "return sum(items) * 4")
    model = Model(solver, solve=[[first]], fixes=[[second], [third]])
    model.scripts = [REPRO]
    await solver.asolve(query=ISSUE)
    assert len(model.fix_errors) == 1  # one fix round for the reproduction alone, never more
    assert "return sum(items) * 2" in (root / "shop/cart.py").read_text("utf-8")
    assert any("Only the reproduction still fails" in s for s in statuses)
    assert any("Restored the best state (round 1)" in s for s in statuses)


@pytest.mark.asyncio
async def test_a_script_that_does_not_show_the_problem_is_retried_once_then_dropped(
    root: Path,
) -> None:
    statuses: list[str] = []
    solver = make_solver(root, statuses, reproduce=True, fix_probe_guard="off")
    model = Model(
        solver, solve=[[edit(root, "return sum(items)", "return sum(items) * 1.2")]], fixes=[]
    )
    model.scripts = [
        "print('REPRO: PASS')\n",
        "import nothing_like_this\nprint('REPRO: FAIL - x')\n",
    ]
    await solver.asolve(query=ISSUE)
    assert len(model.repro_feedback) == 2
    assert "printed REPRO: PASS on the UNCHANGED code" in model.repro_feedback[1]
    assert "=== REPRODUCTION" not in model.solve_contexts[0]
    assert not (root / cf.REPRO_FILE).exists() and model.fix_errors == []
    assert any("No reproduction of the problem" in s for s in statuses)


# ── The reproduction is evidence, not a gate ───────────────────────────────────────────────

#: Reports the task's problem first; once that is gone it goes on to a theory of its own.
REPRO_WITH_OWN_THEORY = (
    "import sys\nfrom shop.cart import total\ngot = total([100])\n"
    "if abs(got - 120) > 1e-9:\n    print('REPRO: FAIL - total([100]) is', got, 'expected 120')\n"
    "    sys.exit(1)\n"
    "if total([]) != 5:\n    print('REPRO: FAIL - an empty basket must cost the fee of 5')\n"
    "    sys.exit(1)\nprint('REPRO: PASS')\n"
)


@pytest.mark.asyncio
async def test_a_reproduction_that_fails_differently_after_the_change_is_not_a_failure(
    root: Path,
) -> None:
    """Seen on real repositories: the change removes what the script reported, and the script
    then fails on something the task never said.  That is no failed round, no fix call and no
    reason to prefer another state."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, reproduce=True, fix_probe_guard="off")
    taxed = edit(root, "return sum(items)", "return sum(items) * (1 + RATE)")
    model = Model(solver, solve=[[taxed]], fixes=[[edit(root, "(1 + RATE)", "(9 + RATE)")]])
    model.scripts = [REPRO_WITH_OWN_THEORY]
    await solver.asolve(query=ISSUE)
    assert model.fix_errors == [] and any("Verification passed" in s for s in statuses)
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in log] == ["reproduction", "verify"]
    assert log[1]["passed"] is True and log[1]["reproduction"] == "other"
    assert "an empty basket" in log[1]["reproduction_output"]


@pytest.mark.asyncio
async def test_a_reproduction_that_fails_for_its_own_reasons_is_rewritten_once_then_dropped(
    root: Path,
) -> None:
    """A script that reports an error the task does not speak of (a wrong attribute, objects
    that do not fit, a missing module) has shown a mistake of its own, not the problem."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, reproduce=True, fix_probe_guard="off")
    model = Model(
        solver, solve=[[edit(root, "return sum(items)", "return sum(items) * 1.2")]], fixes=[]
    )
    own_error = (
        "import sys\ntry:\n    import shop.cart as c\n    c.Basket()\n"
        "except Exception as e:\n"
        "    print('REPRO: FAIL - Unexpected error:', type(e).__name__, e)\n    sys.exit(1)\n"
    )
    missing = "import sys\nprint('REPRO: FAIL - No module named shop_plugins')\nsys.exit(1)\n"
    model.scripts = [own_error, missing, REPRO]  # a third script is never asked for
    await solver.asolve(query=ISSUE)
    assert len(model.repro_feedback) == 2 and model.scripts == [REPRO]
    assert "an error the task does not speak of" in model.repro_feedback[1]
    assert "has no attribute 'Basket'" in model.repro_feedback[1]
    assert "=== REPRODUCTION" not in model.solve_contexts[0] and model.fix_errors == []
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["verdict"] for r in log if r["event"] == "reproduction"] == ["own_error"] * 2
    assert not (root / ".memfun/repro").exists()


def test_a_report_of_an_error_counts_only_when_the_task_speaks_of_that_error() -> None:
    plain = "Lineage of `INSERT OVERWRITE` statements is wrong for subqueries."
    # a wrong result is taken as it is
    assert cf.repro_shows_the_task(plain, "REPRO: FAIL - expected sources={'a'}, got {'b'}\n")
    # an error of the script's own making (all three seen on real repositories)
    for report in (
        "REPRO: FAIL - exception while analyzing: 'list' object is not callable",
        "REPRO: FAIL - Unexpected error: PipelineConnectError: Cannot connect 'a.y' with 'b.y'",
        "REPRO: FAIL - Exception occurred: module 'pkg.api' has no attribute 'BaseThing'",
        "REPRO: FAIL - could not run: No module named 'slycot'",
    ):
        assert not cf.repro_shows_the_task(plain, "noise\n" + report + "\n"), report
    # the same kind of report when the task is about that error
    crash = "Parsing fails with `TypeError: 'list' object is not callable` since 1.4."
    assert cf.repro_shows_the_task(crash, "REPRO: FAIL - raised: 'list' object is not callable")
    named = "Creating the class raises ValueError although the transformer reorders fields."
    assert cf.repro_shows_the_task(named, "REPRO: FAIL - class creation raised ValueError: x")


def test_the_signature_of_a_report_ignores_values_but_not_what_is_reported() -> None:
    a = "REPRO: FAIL - pricetag() uses dot as thousands separator. Examples: ['75.309,58 p.']"
    b = "REPRO: FAIL - pricetag() uses dot as thousands separator. Examples: ['1.234,00 p.', 'x']"
    c = "REPRO: FAIL - price_formats contain '#' which maps to dot. Formats: ['#,##']"
    assert cf.repro_signature("log line\n" + a + "\nmore") == cf.repro_signature(b) != ""
    assert cf.repro_signature(c) != cf.repro_signature(a)
    assert cf.repro_signature("REPRO: FAIL - got 3 expected 4") == cf.repro_signature(
        "REPRO: FAIL - got 20.5 expected 4"
    )
    assert cf.repro_signature("Traceback (most recent call last):\nValueError") == ""


@pytest.mark.asyncio
async def test_the_reproduction_file_is_not_there_while_the_checks_run(root: Path) -> None:
    """memfun's scratch is never linted or tested: the script's file exists only while the
    script itself runs (a project linter once flagged it in every round)."""
    statuses: list[str] = []
    no_scratch = "test -z \"$(find .memfun -name '*.py' 2>/dev/null)\""
    solver = make_solver(
        root, statuses, reproduce=True, fix_probe_guard="off", verify_commands=(no_scratch,)
    )
    taxed = edit(root, "return sum(items)", "return sum(items) * (1 + RATE)")
    model = Model(solver, solve=[[taxed]], fixes=[])
    model.scripts = [REPRO]
    await solver.asolve(query=ISSUE)
    assert "=== REPRODUCTION" in model.solve_contexts[0]
    assert model.fix_errors == [] and any("Verification passed" in s for s in statuses)
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert log[-1]["passed"] is True and log[-1]["reproduction"] == "passes"


# ── The best state ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_state_that_does_not_import_is_never_kept_over_one_that_does(root: Path) -> None:
    """Seen on a real repository: round 1 could not be imported, round 2 could, later rounds
    failed only on the reproduction, and the end of the loop put round 1 back."""
    statuses: list[str] = []
    solver = make_solver(
        root, statuses, verify_tests=True, reproduce=True, fix_probe_guard="off", fix_restarts=0
    )
    broken = edit(root, "from shop.tax import RATE\n", "from shop.tax import RATE, NO_SUCH_NAME\n")
    same_result = edit(root, "return sum(items)", "return sum(list(items))")
    repair = edit(root, "from shop.tax import RATE, NO_SUCH_NAME\n", "from shop.tax import RATE\n")
    model = Model(solver, solve=[[broken, same_result]], fixes=[[repair], []])
    model.scripts = [REPRO]
    await solver.asolve(query=ISSUE)
    text = (root / "shop/cart.py").read_text("utf-8")
    assert "NO_SUCH_NAME" not in text and "sum(list(items))" in text
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    rounds = [r for r in log if r["event"] == "verify"]
    assert "import check of the changed modules" in rounds[0]["failed"][0]
    assert "reproduction" not in rounds[0]  # not run on code that does not import
    assert model.triage_asked == []  # nor is any test classified on such code
    assert rounds[1]["failed"] == [] and rounds[1]["reproduction"] == "same"
    assert not any("Restored the best state (round 1)" in s for s in statuses)


def test_a_broken_round_scores_worse_than_any_round_that_runs(root: Path) -> None:
    solver = make_solver(root, [])
    solver._log_checks = True
    solver._check_kinds = {"imp": "import", "syn": "syntax", "t": "tests"}
    tests_fail = (
        "$ t (exit 1)\nFAILED test/test_cart.py::test_total - assert 1\n9 failed, 2 passed in 0.1s"
    )

    def score(
        cmds: list[str], blocks: list[str], repro_same: bool = False
    ) -> tuple[int, int, int]:
        vr = cf.VerifyResult(False, "\n\n".join(blocks), cmds, blocks=blocks, failed_cmds=cmds)
        return solver._round_score(vr, repro_same)

    cannot_import = score(["imp"], ["$ import check (exit 1)\nImportError: x"])
    bad_syntax = score(["syn"], ["$ syntax check (exit 1)\nSyntaxError: x"])
    not_run = score(["t"], ["$ t (timed out after 300s)"])
    many_failing = score(["t"], [tests_fail])
    lint_and_repro = score(["ruff check ."], ["$ ruff check . (exit 1)\nx.py:1:1: E1"], True)
    only_repro = score([], [], True)
    assert min(cannot_import, bad_syntax, not_run) > many_failing > lint_and_repro > only_repro
    assert many_failing == (0, 36, -2) and only_repro == (0, 1, 0)


# ── A failing existing test is not automatically a regression ──────────────────────────────


@pytest.mark.asyncio
async def test_an_existing_test_that_asserts_the_old_behaviour_is_left_out(root: Path) -> None:
    """The task changes what ``total`` returns; ``test_total`` asserts the old result.  The
    change is not taken back to satisfy it: the test is classified, left out, and the round
    passes."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    taxed = edit(root, "return sum(items)", "return sum(items) * (1 + RATE)")
    revert = edit(root, "return sum(items) * (1 + RATE)", "return sum(items)")
    model = Model(solver, solve=[[taxed]], fixes=[[revert]])
    model.triage = [{1: "old_behaviour"}]
    await solver.asolve(query=ISSUE)
    assert model.fix_errors == [] and any("Verification passed" in s for s in statuses)
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")
    change, asked = model.triage_asked[0]
    assert "+    return sum(items) * (1 + RATE)" in change
    assert "TEST 1: test/test_cart.py::test_total" in asked
    assert "assert total([1, 2]) == 3" in asked  # its source
    assert "assert 3.5999999999999996 == 3" in asked  # what it said
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in log] == ["tests_selected", "failing_tests_classified", "verify"]
    assert log[1]["old_behaviour"] == ["test/test_cart.py::test_total"]
    assert log[2]["passed"] is True and log[2]["left_out_old_behaviour"] == log[1]["old_behaviour"]


@pytest.mark.asyncio
async def test_a_fix_that_takes_the_change_back_does_not_end_on_the_starting_tree(
    root: Path,
) -> None:
    """Seen on a real repository: the fix step edited the code until an old test passed
    again, which was the original code, and the answer was an empty patch."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    taxed = edit(root, "return sum(items)", "return sum(items) * (1 + RATE)")
    revert = edit(root, "return sum(items) * (1 + RATE)", "return sum(items)")
    model = Model(solver, solve=[[taxed]], fixes=[[revert]])
    model.triage = [{1: "regression"}]
    await solver.asolve(query=ISSUE)
    assert len(model.fix_errors) == 1 and cf.FIX_NOTE_OLD_TESTS in model.fix_queries[0]
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")  # the candidate is kept
    assert any("The change was undone: keeping the best candidate" in s for s in statuses)
    assert not any("Verification passed" in s for s in statuses)


@pytest.mark.asyncio
async def test_a_test_is_asked_about_again_only_when_it_fails_differently(root: Path) -> None:
    """A wrong first change fails the old test in a way the task does not call for (a
    regression: fix it); the corrected change fails it only because it asserts the old result."""
    statuses: list[str] = []
    solver = make_solver(root, statuses, verify_tests=True)
    wrong = edit(root, "return sum(items)", "return sum(items) * RATE")
    right = edit(root, "return sum(items) * RATE", "return sum(items) * (1 + RATE)")
    model = Model(solver, solve=[[wrong]], fixes=[[right]])
    model.triage = [{1: "regression"}, {1: "old_behaviour"}]
    await solver.asolve(query=ISSUE)
    assert len(model.triage_asked) == 2 and len(model.fix_errors) == 1
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")
    assert any("Verification passed" in s for s in statuses)


def test_the_answer_of_the_test_classification_is_read_defensively() -> None:
    class Predictor:
        def __init__(self, verdicts: str) -> None:
            self.verdicts = verdicts

        def __call__(self, **_kw: Any) -> Any:
            return type("R", (), {"verdicts": self.verdicts})()

    import asyncio

    def read(raw: str) -> dict[int, str]:
        triage = cf.TestTriage()
        triage.predictor = Predictor(raw)  # type: ignore[assignment]
        return asyncio.run(triage.aclassify(query="q", change="c", failing_tests="t"))

    good = (
        '```json\n[{"test": 1, "verdict": "old_behaviour", "why": "x"}, '
        '{"test": "T2", "verdict": "regression"}]\n```'
    )
    assert read(good) == {1: "old_behaviour", 2: "regression"}
    assert read(
        '[{"test": 1, "verdict": "probably old"}, {"test": true, "verdict": "old_behaviour"}]'
    ) == {1: "regression"}
    assert read("no json at all") == {} and read("[1, 2") == {}


# ── Checks that already fail say nothing about the change ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_detected_linter_that_already_fails_on_the_unchanged_code_is_left_out(
    root: Path, monkeypatch
) -> None:
    statuses: list[str] = []
    already = "echo 'old.py:1:1: E501 line too long' && false"
    new_only = "! grep -q BADWORD shop/cart.py"
    monkeypatch.setattr(cf, "_detect_verify_commands", lambda _root: [already, new_only])
    solver = make_solver(root, statuses, verify_tests=True)
    bad = edit(root, "return sum(items)", "return sum(list(items))  # BADWORD")
    good = edit(root, "  # BADWORD", "")
    model = Model(solver, solve=[[bad]], fixes=[[good]])
    await solver.asolve(query=ISSUE)
    assert len(model.fix_errors) == 1
    assert "BADWORD" in model.fix_errors[0] and "E501" not in model.fix_errors[0]
    assert any("Verification passed" in s for s in statuses)
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert {"event": "checks_failing_before", "left_out": [already]}.items() <= log[1].items()


def test_classify_repro_follows_the_protocol() -> None:
    assert cf.classify_repro(1, "x\nREPRO: FAIL - wrong\n") == "reproduced"
    assert cf.classify_repro(0, "REPRO: PASS\n") == "passes"
    assert cf.classify_repro(1, "Traceback...\nValueError: x\n") == "broken"
    assert cf.classify_repro(0, "REPRO: FAIL - but exit 0\n") == "broken"
    assert cf.classify_repro(None, "(timed out after 90s)") == "broken"


def test_traceback_context_adds_the_code_around_a_failing_project_line(root: Path) -> None:
    out = (
        f'Traceback:\n  File "{root}/shop/cart.py", line 5, in total\n'
        '  File "/usr/lib/x.py", line 1\n'
    )
    got = cf.traceback_context(out, root.resolve(), have="")
    assert got.startswith(
        "=== FILE: shop/cart.py (lines 1-5, from the reproduction traceback) ==="
    )
    assert "return sum(items)" in got and "x.py" not in got
    assert cf.traceback_context(out, root.resolve(), have="=== FILE: shop/cart.py ===") == ""


@pytest.mark.asyncio
async def test_a_large_project_is_planned_from_a_search_of_every_file(root: Path) -> None:
    for i in range(40):  # many files that have nothing to do with the task
        (root / f"shop/extra_{i:02d}.py").write_text(f"def helper_{i}():\n    return {i}\n" * 20)
    big = (
        "def unrelated():\n    return 0\n\n\n"
        + "X = 1\n" * 900
        + "\n\ndef apply_tax(v):\n    return v * 1.2\n"
    )
    (root / "shop/big_rules.py").write_text(big, "utf-8")
    statuses: list[str] = []
    solver = make_solver(
        root, statuses, max_context_bytes=1_000, max_file_bytes=2_000, repo_search=True
    )
    seen: dict[str, str] = {}

    async def aplan(query: str, file_manifest: str, project_summary: str) -> cf.PlanResult:
        seen["manifest"] = file_manifest
        return cf.PlanResult(
            files_to_read=["shop/tax.py"], search_patterns=["apply_tax"], reasoning=""
        )

    solver.planner.aplan = aplan  # type: ignore[method-assign]
    model = Model(solver, solve=[[]], fixes=[])
    await solver.asolve(query=ISSUE)
    manifest, ctx = seen["manifest"], model.solve_contexts[0]
    assert manifest.startswith("FILES THAT DEFINE OR MENTION NAMES FROM THE TASK (all 46 source")
    assert manifest.index("shop/cart.py") < manifest.index("TESTS THAT MENTION THE SAME NAMES")
    assert "test/test_cart.py" in manifest and "DIRECTORIES OF THE PROJECT" in manifest
    # the planner named one file; the search-ranked file, the file its pattern hit and the
    # related test are read as well
    for rel in ("shop/tax.py", "shop/cart.py", "shop/big_rules.py", "test/test_cart.py"):
        assert f"=== FILE: {rel} ===" in ctx
    assert "=== SEARCH RESULTS (whole repository, path:line: text) ===" in ctx
    assert "shop/big_rules.py:907: def apply_tax(v):" in ctx
    # the big file is cut, but the definition the planner searched for is shown whole
    assert "# --- apply_tax (lines 907-908) ---\ndef apply_tax(v):\n    return v * 1.2" in ctx
    assert "extra_07.py ===" not in ctx


def test_the_new_switches_are_read_from_the_project_verify_file(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".memfun").mkdir()
    doc = {
        "discover_tests": True,
        "reproduce": True,
        "repo_search": True,
        "test_timeout": 600,
        "probe_guard": "off",
    }
    (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(doc), "utf-8")
    assert cf.read_project_verify_options(tmp_path) == {}  # not trusted: nothing is read
    monkeypatch.setenv(cf.TRUST_PROJECT_VERIFY_ENV, "1")
    assert cf.read_project_verify_options(tmp_path) == doc
    bad = {"discover_tests": "yes", "reproduce": 1, "repo_search": "on", "test_timeout": 5}
    (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(bad), "utf-8")
    assert cf.read_project_verify_options(tmp_path) == {}


# ── Everything above is opt-in: with every switch off the solver works as it did before ──────


def large_project(root: Path) -> None:
    for i in range(40):
        (root / f"shop/extra_{i:02d}.py").write_text(f"def helper_{i}():\n    return {i}\n" * 20)


def test_every_new_switch_is_off_by_default() -> None:
    cfg = cf.ContextFirstConfig()
    assert (cfg.repo_search, cfg.verify_tests, cfg.reproduce) == (False, False, False)
    assert (cfg.fix_probe_rounds, cfg.fix_detail) == (0, False)


@pytest.mark.asyncio
async def test_with_every_switch_off_the_solver_plans_reads_and_fixes_as_before(
    root: Path, monkeypatch
) -> None:
    """The default configuration on a project too large to read whole: the planner chooses
    from the plain code map, exactly the files and patterns it names are gathered, the solver
    is shown exactly the gathered text, the fix step gets the joined errors of the configured
    commands, and none of the opt-in machinery runs or leaves a file behind."""
    from memfun_agent import fixloop, repo_search
    from memfun_agent.code_map import build_code_map, code_map_to_string

    large_project(root)
    for name in ("issue_terms", "rank_files", "grep"):
        monkeypatch.setattr(repo_search, name, lambda *_a, **_k: pytest.fail("search is opt-in"))
    for name in ("discover", "candidate_pythons", "python_version"):
        monkeypatch.setattr(repo_tests, name, lambda *_a, **_k: pytest.fail("checks are opt-in"))
    fail_a = f"{sys.executable} -c \"import sys; print('first check'); sys.exit(1)\""
    fail_b = f"{sys.executable} -c \"import sys; print('second check'); sys.exit(2)\""
    solver = make_solver(
        root, [], max_context_bytes=1_000, verify_commands=(fail_a, fail_b), max_fix_attempts=1
    )
    expected_manifest = code_map_to_string(
        build_code_map(root, manifest=cf.build_file_manifest(root)), max_tokens=2000
    )
    seen: dict[str, Any] = {}
    plan = cf.PlanResult(
        files_to_read=["shop/tax.py", "shop/cart.py"], search_patterns=["RATE"], reasoning=""
    )

    async def aplan(query: str, file_manifest: str, project_summary: str) -> cf.PlanResult:
        seen["manifest"] = file_manifest
        return plan

    gather = solver.gatherer.agather

    async def agather(**kwargs: Any) -> str:
        seen["gather"] = kwargs
        seen["gathered"] = await gather(**kwargs)
        return seen["gathered"]

    solver.planner.aplan = aplan  # type: ignore[method-assign]
    solver.gatherer.agather = agather  # type: ignore[method-assign]
    change = edit(root, "return sum(items)", "return sum(list(items))")
    model = Model(solver, solve=[[change]], fixes=[])
    solver.repro_solver.awrite = None  # type: ignore[method-assign]  # must never be called
    await solver.asolve(query=ISSUE)

    assert seen["manifest"] == expected_manifest
    assert seen["gather"] == {
        "files": plan.files_to_read,
        "project_root": solver.project_root,
        "search_patterns": plan.search_patterns,
        "web_searches": None,
    }
    assert solver.gatherer.focus_terms == []
    assert model.solve_contexts == [seen["gathered"]]  # nothing added: no search, no environment
    # two failing commands: the fix step is shown what it was always shown
    vr = await cf.Verifier(root).averify([fail_a, fail_b])
    assert model.fix_errors == [fixloop.compact_errors(vr.errors)]
    assert "first check" in model.fix_errors[0] and "second check" in model.fix_errors[0]
    assert not (root / ".memfun").exists()  # no check log, no reproduction, no probe log


@pytest.mark.asyncio
async def test_repo_search_is_switched_on_by_the_project_verify_file(
    root: Path, monkeypatch
) -> None:
    """The benchmark runner writes its agent options to ``.memfun/verify.json``."""
    large_project(root)
    (root / ".memfun").mkdir()
    (root / cf.PROJECT_VERIFY_FILE).write_text(json.dumps({"repo_search": True}), "utf-8")
    monkeypatch.setenv(cf.TRUST_PROJECT_VERIFY_ENV, "1")
    solver = make_solver(root, [], max_context_bytes=1_000)
    seen: dict[str, str] = {}

    async def aplan(query: str, file_manifest: str, project_summary: str) -> cf.PlanResult:
        seen["manifest"] = file_manifest
        return cf.PlanResult(files_to_read=["shop/tax.py"], search_patterns=[], reasoning="")

    solver.planner.aplan = aplan  # type: ignore[method-assign]
    model = Model(solver, solve=[[]], fixes=[])
    await solver.asolve(query=ISSUE)
    assert seen["manifest"].startswith("FILES THAT DEFINE OR MENTION NAMES FROM THE TASK")
    assert "=== FILE: shop/cart.py ===" in model.solve_contexts[0]  # found by the search
