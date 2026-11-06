"""Commit0 adapter: bundle building, suite-unit grading and the --suite flag (hermetic).

The library comes from local git repositories created in ``tmp_path``; nothing here
touches the network.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from typing import TYPE_CHECKING

import pytest

from evals.swarm import commit0_adapter as c0
from evals.swarm import run_smoke as rs
from evals.swarm import validate_bundles as vb

if TYPE_CHECKING:
    from pathlib import Path

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
