"""Finding and running a repository's own tests (``memfun_agent.repo_tests``).  Hermetic: tiny
projects in ``tmp_path``, the tests' own interpreter as "the project's python"."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
from memfun_agent import repo_tests as rt

if TYPE_CHECKING:
    from pathlib import Path


def make(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, "utf-8")
    return root


PROJECT = {
    "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["test"]\n',
    "src/shop/__init__.py": "",
    "src/shop/cart.py": "def total(items):\n    return sum(items)\n\n\ndef count(items):\n"
    "    return len(items)\n",
    "src/shop/tax.py": "RATE = 0.2\n",
    "test/test_cart.py": "from shop.cart import total\n\n\ndef test_total():\n"
    "    assert total([1, 2]) == 3\n",
    "test/unit/test_misc.py": "import shop.cart\n\n\ndef test_count():\n"
    "    assert shop.cart.count([1]) == 1\n",
    "test/test_tax.py": "from shop import tax\n\n\ndef test_rate():\n    assert tax.RATE\n",
    "test/test_other.py": "def test_nothing():\n    assert True\n",
    "test/conftest.py": "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n",
}


def test_pytest_config_is_read_in_pytests_order(tmp_path: Path) -> None:
    make(tmp_path, {"pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = "a b"\n'})
    assert rt.read_pytest_config(tmp_path) == (["a", "b"], [], "pyproject.toml")
    make(tmp_path, {"setup.cfg": "[tool:pytest]\ntestpaths = c\n"})
    assert rt.read_pytest_config(tmp_path)[2] == "pyproject.toml"  # pyproject before setup.cfg
    make(tmp_path, {"pytest.ini": "[pytest]\ntestpaths = t\npython_files = check_*.py\n"})
    assert rt.read_pytest_config(tmp_path) == (["t"], ["check_*.py"], "pytest.ini")
    assert rt.read_pytest_config(tmp_path / "missing") == ([], [], "")


def test_discover_finds_the_real_test_directory_and_the_projects_patterns(tmp_path: Path) -> None:
    make(tmp_path, PROJECT)
    make(tmp_path, {"node_modules/x/test_skip.py": "", "docs/test_doc.py": ""})
    setup = rt.discover(tmp_path, python=sys.executable)
    assert setup.usable and setup.runner == "pytest" and setup.config_file == "pyproject.toml"
    assert setup.testpaths == ("test",)
    # the configured directory first, nothing from skipped directories
    assert setup.test_files[:2] == ("test/test_cart.py", "test/test_other.py")
    assert "docs/test_doc.py" in setup.test_files
    assert not any("node_modules" in f for f in setup.test_files)
    custom = make(tmp_path / "c", {"pytest.ini": "[pytest]\npython_files = check_*.py\n",
                                   "checks/check_a.py": "", "checks/test_b.py": ""})  # fmt: skip
    assert rt.discover(custom, python=sys.executable).test_files == ("checks/check_a.py",)


def test_the_interpreter_is_never_memfuns_own(tmp_path: Path, monkeypatch) -> None:
    mine = os.path.realpath(sys.executable)
    assert all(os.path.realpath(c) != mine for c in rt.candidate_pythons(tmp_path))
    monkeypatch.setattr(rt, "candidate_pythons", lambda _root: [])
    setup = rt.discover(make(tmp_path, PROJECT))
    assert setup.python is None and not setup.usable and rt.test_command(setup, ["a.py"]) == []


def test_changed_symbols_names_the_edited_functions_and_classes() -> None:
    before = (
        "class Basket:\n    def add(self):\n        return 1\n\n\ndef price():\n    return 2\n"
    )
    after = before.replace("return 1", "return 10")
    assert rt.changed_symbols(before, after) == ["Basket", "add"]
    after = before.replace("return 2", "return 3\n\n\ndef new_one():\n    return 4")
    assert rt.changed_symbols(before, after) == ["new_one", "price"]
    assert rt.changed_symbols(before, before) == []
    assert rt.changed_symbols("def broken(:\n", "x = 1\n") == []
    short = "def f():\n    return 1\n"  # too short a name to search for
    assert rt.changed_symbols(short, short.replace("1", "2")) == []


def test_module_names_cover_src_layouts() -> None:
    assert rt.module_names("src/shop/cart.py") == ["src.shop.cart", "shop.cart", "cart"]
    assert rt.module_names("shop/__init__.py") == ["shop"]
    assert rt.module_names("my-app/x.py") == ["x"]  # ``my-app`` cannot be imported


def test_select_by_name_by_import_and_by_symbol(tmp_path: Path) -> None:
    make(tmp_path, PROJECT)
    setup = rt.discover(tmp_path, python=sys.executable)
    got = rt.select_tests(tmp_path, ["src/shop/cart.py"], setup, symbols=["count"])
    assert [s.path for s in got] == ["test/test_cart.py", "test/unit/test_misc.py"]
    assert "named after src/shop/cart.py" in got[0].reasons
    assert "imports src/shop/cart.py" in got[1].reasons and "mentions count" in got[1].reasons
    # ``from shop import tax`` counts as importing shop/tax.py
    assert [s.path for s in rt.select_tests(tmp_path, ["src/shop/tax.py"], setup)] == [
        "test/test_tax.py"
    ]
    # a changed test file is selected itself; a non-Python file selects nothing
    assert rt.select_tests(tmp_path, ["test/test_other.py"], setup)[0].path == "test/test_other.py"
    assert rt.select_tests(tmp_path, ["README.md"], setup) == []
    assert len(rt.select_tests(tmp_path, ["src/shop/cart.py"], setup, max_files=1)) == 1


def test_nothing_is_selected_without_a_signal(tmp_path: Path) -> None:
    make(tmp_path, {"lib/alone.py": "X = 1\n", "tests/test_far.py": "def test_x():\n    pass\n"})
    setup = rt.discover(tmp_path, python=sys.executable)
    assert rt.select_tests(tmp_path, ["lib/alone.py"], setup) == []


def test_failures_that_were_already_there_are_left_out(tmp_path: Path) -> None:
    make(tmp_path, PROJECT)
    make(tmp_path, {
        "test/test_cart.py": "from shop.cart import total\n\n\ndef test_total():\n"
        "    assert total([1, 2]) == 3\n\n\ndef test_old_failure():\n    assert total([]) == 1\n",
        "test/test_broken.py": "import a_module_that_is_not_installed\n",
    })  # fmt: skip
    setup = rt.discover(tmp_path, python=sys.executable)
    cart = tmp_path / "src/shop/cart.py"
    original = cart.read_text("utf-8")
    cart.write_text(original.replace("sum(items)", "sum(items) + 1"), "utf-8")  # the "change"
    files = ["test/test_cart.py", "test/test_broken.py"]
    failing, broken, note = rt.baseline_failures(tmp_path, setup, files, {cart: original}, 120)
    assert failing == ["test/test_cart.py::test_old_failure"] and broken == ["test/test_broken.py"]
    assert note == "" and "+ 1" in cart.read_text("utf-8")  # the change is back in place
    # with the old failure deselected, the only failure left is the one the change caused
    code, out = rt.run(rt.test_command(setup, ["test/test_cart.py"], failing), tmp_path, 120)
    assert code == 1 and "test_total" in out and "test_old_failure" not in out
    cart.write_text(original, "utf-8")
    assert rt.run(rt.test_command(setup, ["test/test_cart.py"], failing), tmp_path, 120)[0] == 0
    assert not list(tmp_path.rglob("__pycache__")) and not (tmp_path / ".pytest_cache").exists()


def test_original_files_are_put_back_even_when_the_body_fails(tmp_path: Path) -> None:
    a, b = tmp_path / "a.py", tmp_path / "new.py"
    a.write_text("changed", "utf-8")
    b.write_text("created by the change", "utf-8")
    with pytest.raises(RuntimeError), rt.original_files({a: "original", b: None}):
        assert a.read_text("utf-8") == "original" and not b.exists()
        raise RuntimeError("boom")
    assert a.read_text("utf-8") == "changed" and b.read_text("utf-8") == "created by the change"


def test_a_run_that_does_not_finish_is_killed_and_unusable(tmp_path: Path) -> None:
    make(tmp_path, {"test_slow.py": "import time\n\n\ndef test_s():\n    time.sleep(600)\n"})
    setup = rt.discover(tmp_path, python=sys.executable)
    code, out = rt.run(rt.test_command(setup, ["test_slow.py"]), tmp_path, 1.0)
    assert code is None and "timed out" in out
    failing, broken, note = rt.baseline_failures(tmp_path, setup, ["test_slow.py"], {}, 1.0)
    assert failing == [] and broken == ["test_slow.py"] and "did not finish" in note
    assert rt.run(["/no/such/interpreter"], tmp_path, 5)[0] is None


def test_the_compile_check_reports_a_syntax_error_and_writes_nothing(tmp_path: Path) -> None:
    make(tmp_path, {"ok.py": "x = 1\n", "bad.py": "def f(:\n    pass\n"})
    code, out = rt.run(rt.compile_command(sys.executable, ["ok.py", "bad.py"]), tmp_path, 60)
    assert code == 1 and "bad.py:1:" in out and "SyntaxError" in out and "ok.py" not in out
    assert rt.run(rt.compile_command(sys.executable, ["ok.py"]), tmp_path, 60)[0] == 0
    assert not list(tmp_path.rglob("*.pyc"))
