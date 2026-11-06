"""Commit0 adapter: bundle building, suite-unit grading and the --suite flag (hermetic).

The library comes from local git repositories created in ``tmp_path``; nothing here
touches the network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Same as the other run_smoke tests: ``evals`` is not an installed package, so put the
# repo root on sys.path (CI runs from a checkout where it is not importable).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import commit0_adapter as c0  # noqa: E402
from evals.swarm import run_smoke as rs  # noqa: E402
from evals.swarm import validate_bundles as vb  # noqa: E402

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}

STUB = (
    '"""mini lib."""\n\n'
    "def add(a, b):\n    '''Add.'''\n    pass\n\n"
    "def sub(a, b):\n    pass\n\n"
    "def untouched():\n    return 1\n"
)
REFERENCE = (
    '"""mini lib."""\n\n'
    "def add(a, b):\n    '''Add.'''\n    return a + b\n\n"
    "def sub(a, b):\n    return a - b\n\n"
    "def untouched():\n    return 1\n"
)
TESTS = (
    "import pytest\n"
    "from minilib import add, sub\n\n"
    "def test_add():\n    assert add(1, 2) == 3\n\n"
    "def test_sub():\n    assert sub(3, 1) == 2\n\n"
    "def test_broken_in_reference():\n    assert False\n\n"
    "@pytest.mark.skip(reason='platform')\n"
    "def test_skipped():\n    assert True\n"
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=_ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def fake_origin(tmp_path: Path) -> tuple[Path, c0.Row]:
    """``<base>/minilib.git`` with a stub commit and a reference commit."""
    work = tmp_path / "work"
    (work / "minilib").mkdir(parents=True)
    (work / "tests").mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "minilib" / "__init__.py").write_text(STUB)
    (work / "tests" / "test_mini.py").write_text(TESTS)
    (work / "docs-link").symlink_to("/etc/hostname")  # must not reach the bundle
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "stub")
    base = _git(work, "rev-parse", "HEAD")
    (work / "minilib" / "__init__.py").write_text(REFERENCE)
    _git(work, "commit", "-q", "-am", "reference")
    ref = _git(work, "rev-parse", "HEAD")
    git_base = tmp_path / "git"
    git_base.mkdir()
    _git(tmp_path, "clone", "-q", "--bare", str(work), str(git_base / "minilib.git"))
    row = c0.Row(
        name="minilib",
        original_repo="example/minilib",
        base_commit=base,
        reference_commit=ref,
        src_dir="minilib/",
        test_dir="tests/",
        test_cmd="pytest",
        specification="",
    )
    return git_base, row


def _build(tmp_path: Path, fake_origin: tuple[Path, c0.Row]) -> tuple[Path, dict[str, object]]:
    git_base, row = fake_origin
    cache = tmp_path / "cache"
    summary = c0.build_bundle(
        row, cache, git_base=f"{git_base}/", timeout=120, log=lambda _m: None
    )
    return cache / "bundles" / "c0-minilib", summary


def test_build_bundle_layout_and_deselect(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    bundle, summary = _build(tmp_path, fake_origin)
    task = tomllib.loads((bundle / "task.toml").read_text())
    assert task["id"] == "c0-minilib" and task["kind"] == "control"
    assert task["units"][0]["hidden_test"] == vb.SUITE_UNIT
    assert task["units"][0]["owns"] == ["minilib/__init__.py"]
    assert task["source"]["stubs"] == 2
    # the failing and the skipped reference tests are deselected, the rest are graded
    deselected = (bundle / "hidden" / vb.DESELECT_FILE).read_text().split()
    assert sorted(deselected) == [
        "test_mini.py::test_broken_in_reference",
        "test_mini.py::test_skipped",
    ]
    assert summary["tests_kept"] == 2 and summary["ok"] is True
    # stub tree in repo/, reference only in solution/, no symlink anywhere
    assert "return a + b" not in (bundle / "repo" / "minilib" / "__init__.py").read_text()
    assert "return a + b" in (bundle / "solution" / "minilib" / "__init__.py").read_text()
    assert not any(p.is_symlink() for p in bundle.rglob("*"))
    assert rs.load_bundle(bundle).problems == ()


def test_suite_unit_grading_counts_every_test(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    bundle_dir, _ = _build(tmp_path, fake_origin)
    bundle = rs.load_bundle(bundle_dir)
    assert bundle.units[0].module == vb.ALL_MODULES
    solved = rs.grade_candidate(bundle, bundle_dir / "solution", tmp_path / "g1", 120)
    assert solved["all_pass"] and solved["hidden_passed"] == 2 and solved["hidden_failed"] == 0
    null = rs.grade_candidate(bundle, None, tmp_path / "g2", 120)
    assert not null["all_pass"] and null["hidden_passed"] == 0 and null["hidden_failed"] == 2
    assert not null["crashes"]


def test_import_error_is_a_model_failure_not_a_crash(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    bundle_dir, _ = _build(tmp_path, fake_origin)
    bundle = rs.load_bundle(bundle_dir)
    cand = tmp_path / "cand" / "minilib"
    cand.mkdir(parents=True)
    (cand / "__init__.py").write_text("def broken(:\n")
    grade = rs.grade_candidate(bundle, tmp_path / "cand", tmp_path / "g", 120)
    assert not grade["all_pass"] and grade["crashes"] == {}


def test_unit_fails_all_modules() -> None:
    ok = vb.RunResult(0, 3, 0, {}, "")
    assert vb.unit_fails(ok, vb.ALL_MODULES) is False
    assert vb.unit_fails(vb.RunResult(1, 3, 1, {}, ""), vb.ALL_MODULES) is True
    assert vb.unit_fails(vb.RunResult(5, 0, 0, {}, ""), vb.ALL_MODULES) is True


def test_export_tree_drops_symlinks_and_strips_subdir(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    git_base, row = fake_origin
    clone = git_base / "minilib.git"
    out = tmp_path / "out"
    c0.export_tree(clone, row.reference_commit, out)
    assert (out / "minilib" / "__init__.py").is_file() and not (out / "docs-link").exists()
    sub = tmp_path / "sub"
    c0.export_tree(clone, row.reference_commit, sub, subdir="tests/")
    assert (sub / "test_mini.py").is_file()


def test_unsupported_layouts_are_refused(tmp_path: Path) -> None:
    base = dict(
        name="x",
        original_repo="o/x",
        base_commit="a",
        reference_commit="b",
        test_cmd="",
        specification="",
    )
    for src, tests in (("pkg/", "pkg/tests/"), ("pkg/", "."), ("", "tests/")):
        row = c0.Row(src_dir=src, test_dir=tests, **base)
        with pytest.raises(c0.AdapterError):
            c0.build_bundle(row, tmp_path, git_base="file:///nonexistent/", log=lambda _m: None)


def test_suite_flag_selects_built_bundles(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    _build(tmp_path, fake_origin)
    args = rs.parse_args(
        [
            "--suite", "commit0", "--commit0-dir", str(tmp_path / "cache"),
            "--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    rs._validate_args(args)  # pyright: ignore[reportPrivateUsage]
    assert args.bundles == ["c0-minilib"]
    smoke = rs.parse_args(["--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path)])
    rs._validate_args(smoke)  # pyright: ignore[reportPrivateUsage]
    assert smoke.bundles == list(rs.DEFAULT_BUNDLES)


def test_reviewer_rejections_are_counted() -> None:
    line = (
        "Per-task review for T1 errored (Reviewer output had no valid findings JSON block "
        "carrying review_id 'x' (verdict cannot be computed). Answer head: 'ok') - retrying\n"
    )
    assert rs.log_signals(line * 3 + "other\n")["reviewer_json_rejections"] == 3
    assert rs.log_signals("nothing\n")["reviewer_json_rejections"] == 0


def test_dataset_fields_cannot_inject_git_options_or_paths(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    git_base, row = fake_origin
    clone = git_base / "minilib.git"
    pwned = tmp_path / "pwned"
    with pytest.raises(c0.AdapterError):  # an option, not a commit
        c0.export_tree(clone, f"--output={pwned}", tmp_path / "o")
    assert not pwned.exists()
    for bad in ("--output=x", "/abs", "../up", "a/../../b", "a\\b"):
        with pytest.raises(c0.AdapterError):
            c0.export_tree(clone, row.reference_commit, tmp_path / "o2", subdir=bad)
    for kwargs in (
        {"src_dir": "../x/"},
        {"test_dir": "/etc"},
        {"base_commit": "--output=y"},
        {"reference_commit": "HEAD"},
    ):
        bad_row = c0.Row(**{**row.__dict__, **kwargs})
        with pytest.raises(c0.AdapterError):
            c0.build_bundle(bad_row, tmp_path / "c", git_base=f"{git_base}/", log=lambda _m: None)


# ── function-level tasks (split) ─────────────────────────────────────────────

_CLASS_SRC = '''\
import functools


class Box:
    """A box."""

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def twice(x):
        """Double."""
        pass

    def keep(self):
        return 1


def helper(a):
    return a
'''


def test_function_spans_and_single_function_stubbing() -> None:
    ref = _CLASS_SRC.replace("        pass\n", "        return x * 2\n")
    spans = c0.function_spans(ref)
    assert set(spans) == {"Box.twice", "Box.keep", "helper"}
    assert spans["Box.twice"].start < spans["Box.twice"].end  # decorators are included
    assert c0.function_spans(_CLASS_SRC)["Box.twice"].is_stub
    assert not c0.function_spans(_CLASS_SRC)["Box.keep"].is_stub
    out = c0.stub_one(ref, _CLASS_SRC, "Box.twice")
    assert out == _CLASS_SRC  # only the target went back to its stub
    two = ref.replace("        return 1\n", "        return 2\n")
    assert "return 2" in c0.stub_one(two, _CLASS_SRC, "Box.twice")  # the others stay as in ref


def test_split_builds_one_independent_bundle_per_stubbed_function(
    tmp_path: Path, fake_origin: tuple[Path, c0.Row]
) -> None:
    pytest.importorskip("coverage")
    _build(tmp_path, fake_origin)
    cache = tmp_path / "cache"
    rows = c0.split_lib("minilib", cache, timeout=180, log=lambda _m: None)
    assert sorted(r["function"] for r in rows) == ["add", "sub"]
    for r in rows:
        bundle_dir = cache / "bundles" / r["id"]
        assert r["id"].startswith("c0s-minilib-")
        src = (bundle_dir / "repo" / "minilib" / "__init__.py").read_text()
        other = "return a - b" if r["function"] == "add" else "return a + b"
        mine = "return a + b" if r["function"] == "add" else "return a - b"
        assert other in src and mine not in src  # only this function is stubbed
        # graded by the tests that execute it, and only those
        deselected = (bundle_dir / "hidden" / vb.DESELECT_FILE).read_text()
        own_test = f"test_{r['function']}"
        assert f"test_mini.py::{own_test}\n" not in deselected + "\n"
        bundle = rs.load_bundle(bundle_dir)
        assert bundle.problems == ()
        assert bundle.units[0].owns == ("minilib/__init__.py",)
        solved = rs.grade_candidate(
            bundle, bundle_dir / "solution", tmp_path / f"g-{r['id']}", 120
        )
        assert solved["all_pass"] and solved["hidden_passed"] == r["tests"] == 1
        null = rs.grade_candidate(bundle, None, tmp_path / f"n-{r['id']}", 120)
        assert not null["all_pass"] and null["hidden_passed"] == 0
        assert rs.tests_kept(bundle_dir) == 1


def test_split_requires_a_built_bundle(tmp_path: Path) -> None:
    with pytest.raises(c0.AdapterError, match="build"):
        c0.split_lib("minilib", tmp_path, log=lambda _m: None)


# ── no planner (--no-decompose) and the incremental progress file ────────────


def test_swarm_cmd_and_suite_defaults_for_decomposition(tmp_path: Path) -> None:
    assert "--no-decompose" not in rs.swarm_cmd(["memfun"], "spec", 1, 60)
    assert "--no-decompose" in rs.swarm_cmd(["memfun"], "spec", 1, 60, decompose=False)
    base = ["--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path / "out")]
    c0dir = tmp_path / "c0" / "bundles" / "c0-x"
    c0dir.mkdir(parents=True)
    (c0dir / "task.toml").write_text("id = 'c0-x'\n")
    smoke = rs.parse_args(base)
    rs._validate_args(smoke)  # pyright: ignore[reportPrivateUsage]
    assert smoke.decompose is True
    commit0 = rs.parse_args([*base, "--suite", "commit0", "--commit0-dir", str(tmp_path / "c0")])
    rs._validate_args(commit0)  # pyright: ignore[reportPrivateUsage]
    assert commit0.decompose is False
    forced = rs.parse_args(
        [*base, "--suite", "commit0", "--commit0-dir", str(tmp_path / "c0"), "--decompose"]
    )
    rs._validate_args(forced)  # pyright: ignore[reportPrivateUsage]
    assert forced.decompose is True


def test_workflow_without_decompose_never_calls_the_planner() -> None:
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from memfun_agent.decomposer import TaskDecomposer
    from memfun_agent.workflow import WorkflowEngine, WorkflowState

    engine = WorkflowEngine.__new__(WorkflowEngine)
    engine._decompose = False  # pyright: ignore[reportPrivateUsage]
    engine._decomposer = MagicMock(  # pyright: ignore[reportPrivateUsage]
        adecompose=AsyncMock(side_effect=AssertionError("planner called")),
        single_task=TaskDecomposer.single_task,
    )
    engine._emit_dashboard = AsyncMock()  # type: ignore[method-assign]
    engine._emit_status = MagicMock()  # type: ignore[method-assign]
    sentinel = object()
    engine._execute_single_task = AsyncMock(return_value=sentinel)  # type: ignore[method-assign]
    state = WorkflowState(workflow_id="w", project="p")
    out = asyncio.run(engine._run(state, "implement f", "ctx", None))  # pyright: ignore[reportPrivateUsage]
    assert out is sentinel
    assert state.decomposition is not None and state.decomposition.is_single_task
    engine._decomposer.adecompose.assert_not_called()  # pyright: ignore[reportPrivateUsage]


def test_swarm_cli_decompose_flag() -> None:
    from unittest.mock import AsyncMock, patch

    import typer
    from memfun_cli.commands.swarm import swarm_command
    from typer.testing import CliRunner

    app = typer.Typer()
    app.command("swarm")(swarm_command)
    for extra, expected in (([], True), (["--no-decompose"], False)):
        with patch("memfun_cli.commands.swarm._run_swarm", new=AsyncMock(return_value=0)) as run:
            result = CliRunner().invoke(app, ["do it", *extra])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["decompose"] is expected


def test_progress_file_gets_one_line_per_episode(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles" / "c0-x"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "task.toml").write_text('id = "c0-x"\n[source]\ntests_kept = 40\n')
    assert rs.tests_kept(bundle_dir) == 40 and rs.tests_kept(tmp_path) is None
    plan = rs.EpisodePlan(rs.load_bundle(bundle_dir), 1, tmp_path / "run")
    ep = {
        "episode": "c0-x-t1",
        "outcome": "partial",
        "reason": "some hidden tests pass",
        "wall_s": 150.0,
        "grade": {"hidden_passed": 31, "hidden_failed": 9},
        "metrics": {"tokens": {"prompt": 1000, "completion": 200}},
        "stalled_stops": 1,
        "truncated_responses": 0,
        "loop_aborts": 2,
        "no_edits": False,
        "reviewer_json_rejections": 3,
    }
    out = tmp_path / "logs" / "progress.md"
    rs.append_progress(out, ep, plan)
    rs.append_progress(out, {**ep, "episode": "c0-x-t2", "outcome": "model_failure"}, plan)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert "c0-x-t1: tests 31/40" in lines[0] and "wall 2.5 min" in lines[0]
    assert "1000 in / 200 out" in lines[0] and "loop-aborts 2" in lines[0]
    assert "stalled 1" in lines[0] and "partial" in lines[0]
    assert lines[1].endswith("model_failure (some hidden tests pass)")


def test_bundles_accept_fnmatch_patterns(tmp_path: Path) -> None:
    base = tmp_path / "c0" / "bundles"
    for b in ("c0-lib", "c0s-lib-a", "c0s-lib-b", "c0s-other-a"):
        (base / b).mkdir(parents=True)
        (base / b / "task.toml").write_text(f"id = '{b}'\n")
    common = [
        "--base-url", "http://127.0.0.1:8441/v1", "--out-dir", str(tmp_path / "out"),
        "--suite", "commit0", "--commit0-dir", str(tmp_path / "c0"),
    ]  # fmt: skip
    args = rs.parse_args([*common, "--bundles", "c0s-lib-*", "c0-lib"])
    rs._validate_args(args)  # pyright: ignore[reportPrivateUsage]
    assert args.bundles == ["c0s-lib-a", "c0s-lib-b", "c0-lib"]
    bad = rs.parse_args([*common, "--bundles", "nothing-*"])
    with pytest.raises(rs.UsageError, match="matches no bundle"):
        rs._validate_args(bad)  # pyright: ignore[reportPrivateUsage]


# ── audit follow-ups for the function tasks ──────────────────────────────────

_COND_SRC = """\
try:
    import fast

    def speed(x):
        return fast.go(x)
except ImportError:
    def speed(x):
        return x


if True:

    def only_here(x):
        pass


class P:
    @property
    def value(self):
        return 1

    @value.setter
    def value(self, v):
        pass

    def plain(self):
        pass
"""


def test_function_spans_see_conditional_definitions_and_flag_ambiguous_ones() -> None:
    dups: list[str] = []
    spans = c0.function_spans(_COND_SRC, dups)
    assert "only_here" in spans and spans["only_here"].is_stub  # inside `if`
    assert "P.plain" in spans
    assert sorted(dups) == ["P.value", "speed"]  # getter/setter and try/except alternatives
    assert "speed" not in spans and "P.value" not in spans


def test_context_test_ids_keep_pipes_inside_parametrized_ids() -> None:
    assert c0._context_test_id("test_a.py::test_x[a|b]|run") == "test_a.py::test_x[a|b]"  # pyright: ignore[reportPrivateUsage]
    assert c0._context_test_id("test_a.py::test_x|setup") == "test_a.py::test_x"  # pyright: ignore[reportPrivateUsage]
    assert c0._context_test_id("test_a.py::test_x[a|b]") == "test_a.py::test_x[a|b]"  # pyright: ignore[reportPrivateUsage]
    assert c0._context_test_id("") == ""  # pyright: ignore[reportPrivateUsage]


def test_function_task_ids_are_unique_and_bounded() -> None:
    used: set[str] = set()
    first = c0.function_task_id("lib", "a.py", "A.run", used)
    used.add(first)
    second = c0.function_task_id("lib", "b.py", "A_run", used)  # same sanitised name
    assert first != second and len(second) <= 64
    long_a = c0.function_task_id("lib", "a.py", "X" * 80 + ".one", set())
    long_b = c0.function_task_id("lib", "a.py", "X" * 80 + ".two", set())
    assert long_a != long_b and len(long_a) <= 64 and len(long_b) <= 64
    assert rs.BUNDLE_ID_RE.match(long_a) and rs.BUNDLE_ID_RE.match(second)


def test_huge_deselect_lists_do_not_overflow_the_command_line(tmp_path: Path) -> None:
    """Audit finding: one `--deselect` argv entry per id crashed with ARG_MAX at ~6000 ids."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "def test_a():\n    pass\n\n"
        "def test_a_long():\n    pass\n\n"
        "def test_b():\n    assert False\n"
    )
    ids = [f"test_x.py::test_{'y' * 100}_{i}" for i in range(9000)]  # ~1 MB of ids
    (tests / vb.DESELECT_FILE).write_text(
        "\n".join(["test_x.py::test_a", *ids, "test_x.py::test_b"])
    )
    tree = tmp_path / "tree"
    tree.mkdir()
    res = vb.run_grader(tree, tests, tmp_path / "scratch", 120)
    # exact matching: `test_a` is dropped, `test_a_long` (same prefix) still runs
    assert res.returncode == 0 and (res.passed, res.failed) == (1, 0)
    assert res.by_file["test_x"]["passed"] == 1
