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
        self.repro_feedback: list[str] = []
        self.scripts: list[str] = []
        solver.solver.asolve = self.asolve  # type: ignore[method-assign]
        solver.fix_solver.afix = self.afix  # type: ignore[method-assign]
        solver.repro_solver.awrite = self.awrite  # type: ignore[method-assign]

    async def asolve(self, query: str, full_context: str) -> cf.SolveResult:
        self.solve_contexts.append(full_context)
        return cf.SolveResult(reasoning="", answer="done", operations=self.solve.pop(0))

    async def afix(self, query: str, full_context: str, verification_errors: str) -> list[Any]:
        self.fix_errors.append(verification_errors)
        return self.fixes.pop(0) if self.fixes else []

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
    right = edit(root, "return sum(items) + 1", "return sum(items)")
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
    right = edit(root, "return sum(items) - 1", "return sum(items)", rel="shop/engine.py")
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
    assert (root / cf.REPRO_FILE).read_text("utf-8") == REPRO + "\n"  # the script is not editable
    assert "(1 + RATE)" in (root / "shop/cart.py").read_text("utf-8")
    assert any("Verification passed" in s for s in statuses)
    log = [json.loads(x) for x in (root / ".memfun/checks.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in log] == ["reproduction", "verify", "verify"]
    assert log[0]["verdict"] == "reproduced" and log[0]["exit"] == 1
    assert log[1]["passed"] is False and log[1]["commands"] == [cf.REPRO_LABEL]
    assert "REPRO: FAIL" in log[1]["failed"][0] and log[2]["passed"] is True


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
    assert len(model.fix_errors) >= 1
    assert "return sum(items) * 2" in (root / "shop/cart.py").read_text("utf-8")
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
    solver = make_solver(root, statuses, max_context_bytes=1_000, max_file_bytes=2_000)
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
    doc = {"discover_tests": True, "reproduce": True, "test_timeout": 600, "probe_guard": "off"}
    (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(doc), "utf-8")
    assert cf.read_project_verify_options(tmp_path) == {}  # not trusted: nothing is read
    monkeypatch.setenv(cf.TRUST_PROJECT_VERIFY_ENV, "1")
    assert cf.read_project_verify_options(tmp_path) == doc
    bad = {"discover_tests": "yes", "reproduce": 1, "test_timeout": 5}
    (tmp_path / cf.PROJECT_VERIFY_FILE).write_text(json.dumps(bad), "utf-8")
    assert cf.read_project_verify_options(tmp_path) == {}
