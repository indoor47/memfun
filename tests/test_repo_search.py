"""Repository-wide search for file localization (``memfun_agent.repo_search``).  Hermetic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_agent import repo_search as rs

if TYPE_CHECKING:
    from pathlib import Path

ISSUE = """Crash when saving an empty basket

Calling `Basket.save_all()` with no items raises `EmptyBasketError` instead of doing nothing.
The message is "cannot save a basket without items". It happens in shop/storage/writer.py:

  File "/app/shop/storage/writer.py", line 12, in write_rows
    raise EmptyBasketError(...)

I would expect the function to return and the user to see nothing. See https://example.com/x.
"""


def make(root: Path, files: dict[str, str]) -> list[str]:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, "utf-8")
    return sorted(files)


def test_issue_terms_are_code_names_and_quoted_messages_not_english() -> None:
    terms = rs.issue_terms(ISSUE)
    for name in ("Basket", "save_all", "EmptyBasketError", "writer", "write_rows"):
        assert name in terms
    assert "cannot save a basket without items" in terms
    for word in ("Crash", "empty", "items", "function", "expect", "example", "https", "nothing"):
        assert word not in terms
    assert terms.index("write_rows") < terms.index("cannot save a basket without items")
    assert rs.issue_terms("please make it faster, thanks") == []
    assert len(rs.issue_terms(" ".join(f"`name_{i}`" for i in range(100)))) == rs.MAX_TERMS


def test_rank_puts_the_defining_file_first_and_keeps_tests_apart(tmp_path: Path) -> None:
    files = make(tmp_path, {
        "shop/basket.py": "class Basket:\n    def save_all(self):\n        write_rows(self)\n",
        "shop/storage/writer.py": "class EmptyBasketError(Exception):\n    pass\n\n\n"
        "def write_rows(basket):\n"
        "    raise EmptyBasketError('cannot save a basket without items')\n",
        "shop/cli.py": "from shop.basket import Basket\n\nBasket().save_all()\n",
        "shop/unrelated.py": "def add(a, b):\n    return a + b\n",
        "tests/test_basket.py": "from shop.basket import Basket\n\n\ndef test_save():\n"
        "    Basket().save_all()\n",
        "big.py": "x = 1\n" * 300_000,
    })  # fmt: skip
    hits = rs.rank_files(tmp_path, files, rs.issue_terms(ISSUE))
    paths = [h.path for h in hits]
    assert paths[0] == "shop/storage/writer.py"  # defines two terms, holds the message
    assert paths[1] == "shop/basket.py" and "shop/unrelated.py" not in paths
    assert "big.py" not in paths  # over the size limit: not searched
    top = hits[0]
    assert top.defines == ["EmptyBasketError", "write_rows"] and not top.is_test
    assert (5, "def write_rows(basket):") in top.lines
    assert next(h for h in hits if h.path == "tests/test_basket.py").is_test
    text = rs.format_ranked([h for h in hits if not h.is_test])
    assert text.startswith("shop/storage/writer.py  [defines: EmptyBasketError, write_rows]")
    assert "  5: def write_rows(basket):" in text
    assert rs.rank_files(tmp_path, files, []) == []
    assert len(rs.format_ranked(hits, max_chars=120).splitlines()) < len(text.splitlines())


def test_grep_searches_every_file_and_survives_a_bad_pattern(tmp_path: Path) -> None:
    files = make(tmp_path, {
        "a/one.py": "def alpha():\n    return 'x(y'\n",
        "b/two.py": "def beta():\n    return alpha()\n",
        "tests/test_one.py": "def test_alpha():\n    assert alpha()\n",
    })  # fmt: skip
    hits, counts = rs.grep(tmp_path, files, [r"alpha\(", "x(y"])  # the second is not a regex
    assert hits[0] == ("a/one.py", 1, "def alpha():") and ("a/one.py", 2, "return 'x(y'") in hits
    assert [h[0] for h in hits][-1] == "tests/test_one.py"  # sources before tests
    assert counts == {"a/one.py": 2, "b/two.py": 1, "tests/test_one.py": 2}
    assert rs.format_grep(hits[:1]) == "a/one.py:1: def alpha():"
    assert rs.grep(tmp_path, files, []) == ([], {})
    assert len(rs.grep(tmp_path, files, ["alpha"], max_hits=2)[0]) == 2


def test_directory_overview_and_test_paths() -> None:
    out = rs.directory_overview(["a/b/x.py", "a/b/y.py", "a/c/z.py", "top.py"])
    assert out.splitlines() == ["a/b/ (2 files)", "./ (1 files)", "a/c/ (1 files)"]
    assert "more directories" in rs.directory_overview(
        [f"d{i}/x.py" for i in range(50)], max_chars=60
    )
    assert rs.is_test_path("pkg/tests/helpers.py") and rs.is_test_path("pkg/thing_test.py")
    assert rs.is_test_path("conftest.py") and not rs.is_test_path("pkg/contest.py")


def test_focused_excerpt_shows_the_matching_definitions_of_a_large_file() -> None:
    src = (
        "import os\n\n\nclass Store:\n    def open(self):\n        return 1\n\n"
        "    @staticmethod\n    def save_all(rows):\n        return write_rows(rows)\n\n\n"
        "def write_rows(rows):\n    return len(rows)\n\n\ndef other():\n    return 0\n"
    )
    out = rs.focused_excerpt(src, ["save_all", "write_rows"])
    assert (
        "# --- Store.save_all (lines 8-10) ---\n    @staticmethod\n    def save_all(rows):" in out
    )
    assert "# --- write_rows (lines 13-14) ---" in out
    assert "def other" not in out and "def open" not in out
    assert out.index("Store.save_all") < out.index("--- write_rows")  # in file order
    assert rs.focused_excerpt(src, ["write_rows"], budget=40) == ""  # nothing fits
    assert rs.focused_excerpt("def broken(:\n", ["broken"]) == ""
    assert rs.focused_excerpt(src, ["no_such_name"]) == ""
    # a class is shown whole only when the class itself is the match and no method is
    assert "# --- Store (lines 4-10) ---\nclass Store:" in rs.focused_excerpt(src, ["Store"])
    assert "# --- Store (" not in out
