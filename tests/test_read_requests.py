"""The reads a solver may ask for before it edits: parsed, confined to the project, answered
by memfun's own code.  Nothing here may reach a shell, leave the project or change a file."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from memfun_agent import read_requests as rr

if TYPE_CHECKING:
    from pathlib import Path

MOD = "".join(f"line {i}\n" for i in range(1, 41))
BIG = (
    "def unrelated():\n    return 0\n\n\n"
    + "X = 1\n" * 900
    + "\n\ndef apply_tax(v):\n    return v * 1.2\n"
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small project next to things that must stay out of reach."""
    proj = tmp_path / "proj"
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": MOD,
        "pkg/big.py": BIG,
        "pkg/llm/augmented_gemini.py": "class GeminiLLM:\n    def call(self, n): return n\n",
        "pkg/llm/base.py": "def call(n):\n    return foo(n) or bar\n",
        "tests/test_mod.py": "def test_x():\n    assert True\n",
        "README.md": "# demo\n",
        ".env": "TOKEN=in-project-secret\n",
        ".git/HEAD": "ref: refs/heads/main\n",
        ".git/packed-refs": "future-ref\n",
        ".memfun/verify.json": '{"reproduce": true}\n',
    }
    for rel, text in files.items():
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text(text, "utf-8")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("OUTSIDE-SECRET\n", "utf-8")
    (proj / "pkg" / "link_out.txt").symlink_to(tmp_path / "outside" / "secret.txt")
    (proj / "linkdir").symlink_to(tmp_path / "outside")

    def no_process(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a read request must never start a process")

    for name in ("run", "Popen", "call", "check_output", "check_call"):
        monkeypatch.setattr(subprocess, name, no_process)
    monkeypatch.setattr(os, "system", no_process)
    monkeypatch.setattr(os, "popen", no_process)
    return proj


def answer(root: Path, cmd: str) -> list[rr.Served]:
    return rr.serve(root, rr.parse_command(cmd))


def one(root: Path, cmd: str) -> rr.Served:
    got = answer(root, cmd)
    assert len(got) == 1, got
    return got[0]


# ── the shapes a model writes, answered without a shell ──────────────────────────────────────


def test_a_range_of_lines_is_shown_as_the_file_has_it(root: Path) -> None:
    got = one(root, "sed -n '10,12p' pkg/mod.py")
    assert got.ok and got.kind == "lines"
    assert got.heading == "lines 10-12 of pkg/mod.py (40 lines in the file)"
    assert got.output == "line 10\nline 11\nline 12"  # exact text: an edit can copy it
    assert one(root, "sed -n 5p pkg/mod.py").output == "line 5"
    assert one(root, "sed -n '38,$p' pkg/mod.py").output == "line 38\nline 39\nline 40"
    assert one(root, "head -n 2 pkg/mod.py").output == "line 1\nline 2"
    assert one(root, "head -3 pkg/mod.py").output == "line 1\nline 2\nline 3"
    assert one(root, "tail -n 2 pkg/mod.py").output == "line 39\nline 40"
    assert one(root, "tail -n +39 pkg/mod.py").output == "line 39\nline 40"
    whole = one(root, "cat pkg/llm/base.py")
    assert whole.heading == "pkg/llm/base.py (whole file, 2 lines)"
    assert whole.output == "def call(n):\n    return foo(n) or bar"


def test_the_json_read_operations(root: Path) -> None:
    ops: list[dict[str, Any]] = [
        {"op": "read_file", "path": "pkg/mod.py", "start_line": 3, "end_line": 4},
        {"op": "read_file", "path": "pkg/mod.py", "lines": "7-8"},
        {"op": "read_file", "path": "pkg/mod.py", "offset": 20, "limit": 2},
        {"op": "search", "pattern": "def apply_tax|class Gemini", "path": "pkg"},
        {"op": "find_files", "name": "*gemini*", "path": "."},
        {"op": "list_dir", "path": "pkg/llm"},
        {"op": "edit_file", "path": "pkg/mod.py", "old": "a", "new": "b"},  # not a read
    ]
    got = rr.serve(root, rr.requests_of(ops))
    assert [g.kind for g in got] == ["lines", "lines", "lines", "search", "find", "list"]
    assert got[0].output == "line 3\nline 4" and got[1].output == "line 7\nline 8"
    assert got[2].output == "line 20\nline 21"
    assert got[3].output.splitlines() == [
        "pkg/big.py:907: def apply_tax(v):",
        "pkg/llm/augmented_gemini.py:1: class GeminiLLM:",
    ]
    assert got[4].output == "pkg/llm/augmented_gemini.py"
    assert got[5].output.splitlines() == ["augmented_gemini.py (49 bytes)", "base.py (38 bytes)"]


def test_grep_in_its_usual_spellings(root: Path) -> None:
    hit = "pkg/big.py:907: def apply_tax(v):"
    assert one(root, 'grep -n "def apply_tax" pkg/big.py').output == hit
    assert one(root, 'grep -rn "def apply_tax" . --include="*.py"').output == hit
    assert one(root, "rg -n 'def apply_tax' pkg").output == hit
    assert one(root, 'grep -rn "DEF APPLY_TAX" -i pkg 2>/dev/null').output == hit
    assert one(root, 'grep -rl "apply_tax" pkg').output == "pkg/big.py"
    assert one(root, 'grep -rn "nothing_like_this" pkg').output == "(no match)"
    ctx = one(root, 'grep -n -A1 "def apply_tax" pkg/big.py').output
    assert ctx == "pkg/big.py:907: def apply_tax(v):\npkg/big.py:908-     return v * 1.2"
    # plain grep: "(" is a literal and "\|" is the alternation, the other way round from Python
    assert one(root, 'grep -n "foo(n)" pkg/llm/base.py').output.endswith("return foo(n) or bar")
    both = one(root, r'grep -rn "class Gemini\|def apply_tax" pkg').output.splitlines()
    assert len(both) == 2
    assert one(root, 'grep -rnE "class (Gemini|Nothing)LLM" pkg').output.startswith("pkg/llm/aug")
    assert one(root, "cd pkg && grep -rn 'def apply_tax' .").output == hit


def test_find_and_pipelines(root: Path) -> None:
    cmd = 'find . -name "*gemini*" -o -name "*Gemini*" | grep -v __pycache__ | grep -v .git'
    assert one(root, cmd).output == "pkg/llm/augmented_gemini.py"
    assert one(root, "find pkg -name '*.py' -type f | wc -l").output == "5"
    assert one(root, "find . -iname 'README*' -maxdepth 1").output == "README.md"
    assert one(root, "find pkg -type d").output == "pkg/llm/"
    assert one(root, "find pkg -name '*.py' | sort | head -2").output.splitlines() == [
        "pkg/__init__.py",
        "pkg/big.py",
    ]
    listed = one(root, "find pkg -name '*.py' | xargs grep -n 'class Gemini'")
    assert listed.output == "pkg/llm/augmented_gemini.py:1: class GeminiLLM:"
    assert one(root, "wc -l pkg/mod.py").output == "40 pkg/mod.py"
    assert one(root, "ls -la pkg/llm").output.startswith("augmented_gemini.py")
    two = answer(root, "ls pkg/llm && sed -n '1,1p' pkg/mod.py; echo done")
    assert [g.kind for g in two] == ["list", "lines"] and two[1].output == "line 1"
    with_fallback = one(root, "cat pkg/nope.py 2>/dev/null || echo 'not found'")
    assert not with_fallback.ok and "does not exist" in with_fallback.output


# ── refusals ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd",
    [
        "python -c \"open('made-by-cmd','w').write('x')\"",
        "python3 pkg/mod.py",
        "pytest tests -q",
        "pip install requests",
        "touch made-by-cmd",
        "rm -rf pkg",
        "cat pkg/mod.py > made-by-cmd",
        "grep -rn x pkg >> made-by-cmd",
        "sed -i 's/line 1/changed/' pkg/mod.py",
        "sed 's/line 1/changed/' pkg/mod.py",
        "find . -name '*.py' -delete",
        "find . -name '*.py' -exec rm {} \\;",
        "cat $(touch made-by-cmd)",
        "cat `touch made-by-cmd`",
        "git log --all --oneline",
        "git show HEAD~1:pkg/mod.py",
        "curl https://example.com",
        "FOO=1 cat pkg/mod.py",
        "cat pkg/mod.py | tee made-by-cmd",
        "cat pkg/mod.py | xargs rm",
        "cat pkg/mod.py &",
        "bash -c 'cat pkg/mod.py'",
        "cat 'pkg/mod.py",
    ],
)
def test_anything_that_is_not_a_read_is_refused_and_nothing_happens(root: Path, cmd: str) -> None:
    before = sorted(str(p) for p in root.rglob("*"))
    got = answer(root, cmd)
    assert got and not any(g.ok for g in got), got
    assert all(g.kind == "refused" for g in got)
    assert sorted(str(p) for p in root.rglob("*")) == before  # nothing created or removed
    assert (root / "pkg/mod.py").read_text("utf-8") == MOD


def test_a_command_after_a_read_does_not_run_either(root: Path) -> None:
    got = answer(root, "cat pkg/llm/base.py; touch made-by-cmd && rm -rf pkg")
    assert [g.ok for g in got] == [True, False, False]
    assert not (root / "made-by-cmd").exists() and (root / "pkg/mod.py").is_file()


@pytest.mark.parametrize(
    ("path", "why"),
    [
        ("../outside/secret.txt", "outside the project"),
        ("pkg/../../outside/secret.txt", "outside the project"),
        ("/etc/passwd", "outside the project"),
        ("~/.ssh/id_rsa", "not a path in the project"),
        ("pkg/link_out.txt", "outside the project"),  # a symbolic link that leaves the project
        ("linkdir/secret.txt", "outside the project"),  # through a linked directory
        (".git/HEAD", "version-control data"),
        ("./.git/packed-refs", "version-control data"),
        (".memfun/verify.json", "memfun's own state"),
        (".env", "secrets"),
    ],
)
def test_paths_that_leave_the_project_or_are_off_limits_are_refused(
    root: Path, path: str, why: str
) -> None:
    for item in (
        *rr.parse_command(f"cat {path}"),
        *rr.parse_command(f"sed -n '1,5p' {path}"),
        *rr.parse_command(f"grep -n . {path}"),
        *rr.parse_operation({"op": "read_file", "path": path}),
    ):
        got = rr.serve_one(root, item)
        assert not got.ok and why in got.output, (path, got)
        assert "OUTSIDE-SECRET" not in got.output and "in-project-secret" not in got.output


def test_a_walk_never_reads_outside_the_project_or_its_off_limits_parts(root: Path) -> None:
    everything = one(root, "grep -rn . .").output
    for leaked in ("OUTSIDE-SECRET", "in-project-secret", "future-ref", "reproduce"):
        assert leaked not in everything
    names = one(root, "find . -type f").output.splitlines()
    assert "pkg/mod.py" in names
    assert not [n for n in names if n.startswith((".git", ".memfun", "linkdir")) or ".env" in n]
    assert "link_out.txt" not in "\n".join(names)
    top = one(root, "ls -la").output
    assert ".git" not in top and ".memfun" not in top and ".env" not in top
    assert not one(root, "ls linkdir").ok and not one(root, "find /etc -name passwd").ok
    assert not one(root, "cd .. && cat outside/secret.txt").ok
    assert not one(root, "find . -name '*.txt' | xargs grep -n SECRET").output.count("OUTSIDE")


# ── bounds ───────────────────────────────────────────────────────────────────────────────────


def test_a_long_read_is_cut_at_a_line_and_says_how_to_go_on(root: Path) -> None:
    got = rr.serve_one(root, rr.parse_command("cat pkg/big.py")[0], limit=200)
    assert got.ok and len(got.output) < 400
    body, note = got.output.rsplit("\n", 1)
    assert body.splitlines()[-1] == "X = 1"  # whole lines only
    kept = len(body.splitlines())
    assert note == (
        f"... (cut after line {kept} of 908 to keep the reply small: ask for lines "
        f"{kept + 1}-908 if you need them)"
    )
    assert got.heading == f"lines 1-{kept} of pkg/big.py (908 lines in the file)"


def test_one_round_has_a_budget_and_a_request_limit(root: Path) -> None:
    many = rr.parse_command("cat pkg/big.py") * 1 + [
        item for i in range(1, 30) for item in rr.parse_command(f"sed -n '{i},{i}p' pkg/mod.py")
    ]
    got = rr.serve(root, many, per_request=3_000, budget=3_500)
    assert got[0].ok and len(got[0].output) <= 3_200
    served = [g for g in got if g.ok]
    refused = [g for g in got if not g.ok]
    assert sum(len(g.output) for g in served) <= 3_500
    assert refused and all("budget is used up" in g.output for g in refused)
    unlimited = rr.serve(root, many, budget=10**9)
    assert sum(g.ok for g in unlimited) == rr.MAX_REQUESTS
    assert len(unlimited) == rr.MAX_REQUESTS + 1  # one line for all the rest
    assert unlimited[-1].heading.startswith(f"not served: {len(many) - rr.MAX_REQUESTS} more")
    twice = rr.serve(root, rr.parse_command("cat pkg/mod.py") * 2)
    assert len(twice) == 1  # the same request asked twice is answered once


def test_a_search_stops_at_its_line_limit(root: Path) -> None:
    got = one(root, "grep -n 'X = 1' pkg/big.py")
    assert len(got.output.splitlines()) == rr.MAX_HITS
    assert "stopped at 200 lines" in got.heading


def test_what_was_asked_and_served_is_logged(root: Path) -> None:
    served = answer(root, "sed -n '1,2p' pkg/mod.py; python -c 'print(1)'")
    rr.log_reads(root, 2, served)
    rows = [json.loads(x) for x in (root / rr.READ_LOG).read_text("utf-8").splitlines()]
    assert [(r["round"], r["kind"], r["served"]) for r in rows] == [
        (2, "lines", True),
        (2, "refused", False),
    ]
    assert rows[0]["answer"] == "lines 1-2 of pkg/mod.py (40 lines in the file)"
    assert rows[0]["bytes"] == len("line 1\nline 2") and "nothing is executed" in rows[1]["answer"]
    text = rr.format_served(served)
    assert text.startswith("--- lines 1-2 of pkg/mod.py (40 lines in the file) ---\nline 1\n")
    assert "REFUSED: nothing is executed before the edit" in text


# ── what an independent audit broke, and must stay fixed ─────────────────────────────────────


def test_a_pattern_that_backtracks_without_end_cannot_hold_the_process(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``re`` has no time limit: under it each of these patterns held the whole process on a
    short line.  (No clock is asserted; a regression shows as a test that does not come back.)"""
    (root / "pkg/code.py").write_text("result = compute_total(items, tax_rate) + shipping\n")
    (root / "pkg/aaa.txt").write_text("a" * 60 + "!\n" + "x" * 60 + "\n")
    for pattern in ("(\\w+\\W*)+\\x01", "(a+)+$", "(x+x+)+y", "^(([a-z])+.)+[A-Z]([a-z])+$"):
        got = rr.serve(root, rr.requests_of([{"op": "search", "pattern": pattern, "path": "pkg"}]))
        assert got[0].ok and got[0].output == "(no match)", pattern
    # what cannot be compiled at all is searched as plain text instead of raising
    for pattern in ("(" * 3000 + ")" * 3000, "a{99999999999}", "x" * 5000, "(unclosed"):
        assert rr.serve(root, rr.requests_of([{"op": "search", "pattern": pattern}]))[0].ok

    class OutOfTime:
        def search(self, text: str, timeout: float) -> None:
            assert 0 < timeout <= rr.MATCH_SECONDS  # every match is given a limit
            raise TimeoutError

    real = rr._compile

    def slow(pattern: str, ignore_case: bool) -> Any:
        matcher = real(pattern, ignore_case)
        matcher._rx = OutOfTime()
        return matcher

    # the backstop behind the engine: a match that does run out of time ends the search
    monkeypatch.setattr(rr, "_compile", slow)
    assert "takes too long" in one(root, "grep -n a pkg/aaa.txt").heading
    assert "takes too long" in one(root, "grep -rl a pkg").heading
    piped = one(root, "cat pkg/aaa.txt | grep a")
    assert not piped.ok and "takes too long" in piped.output


def test_a_named_pipe_in_the_project_does_not_block_a_search(root: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes here")
    os.mkfifo(root / "pkg" / "fifo")
    (root / "pkg" / "fifo_link").symlink_to(root / "pkg" / "fifo")
    assert "pkg/mod.py:1: line 1" in one(root, "grep -rn 'line 1$' pkg").output
    assert not one(root, "cat pkg/fifo").ok and not one(root, "wc -l pkg/fifo").ok


@pytest.mark.parametrize(
    "op",
    [
        {"op": "read_file", "path": "pkg/mod.py", "start_line": 1e999},
        {"op": "read_file", "path": "pkg/mod.py", "end_line": float("inf"), "offset": 1e999},
        {"op": "read_file", "path": "pkg/mod.py", "lines": "1-" + "9" * 5000},
        {"op": "read_file", "path": "pkg/mod.py", "lines": [1e999, 2]},
        {"op": "search", "pattern": "line", "path": "pkg/mod.py", "context": 1e999},
        {"op": "read_file", "path": "pkg/\ud800.py"},
        {"op": "search", "pattern": "\udcff", "path": "pkg"},
        {"op": "find_files", "name": "\ud800*", "path": "."},
        {"op": "run_cmd", "cmd": "head -n \u00b2 pkg/mod.py"},
        {"op": "run_cmd", "cmd": "head -n " + "9" * 5000 + " pkg/mod.py"},
        {"op": "run_cmd", "cmd": "sed -n '" + "9" * 5000 + "p' pkg/mod.py"},
        {"op": "run_cmd", "cmd": "find . -maxdepth \u00b2 -name '*.py'"},
        {"op": "run_cmd", "cmd": "grep -A \u00b2 line pkg/mod.py"},
        {"op": "run_cmd", "cmd": "cat pkg/mod.py | head -n \u00b2"},
        {"op": "run_cmd", "cmd": "cat pkg/long.txt | xargs grep tax"},
        {"op": "run_cmd", "cmd": 12345},
        {"op": "search", "pattern": ["a", "b"], "path": {"x": 1}},
        {"op": "read_file", "path": None, "start_line": "abc"},
    ],
)
def test_no_request_however_malformed_raises(root: Path, op: dict[str, Any]) -> None:
    """Each of these ended the solve with an uncaught exception; a request is answered or
    refused, and the answer and the log survive text that cannot be encoded."""
    (root / "pkg/long.txt").write_text("word " * 80 + "\n")  # one line: a 400-byte "file name"
    served = rr.serve(root, rr.requests_of([op]))
    rr.format_served(served).encode("utf-8")
    rr.log_reads(root, 1, served)
    assert (root / "pkg/mod.py").read_text("utf-8") == MOD


def test_off_limits_names_in_another_case_are_not_served(root: Path) -> None:
    """On a disk that ignores case ``.Git/config`` is ``.git/config``."""
    for path in (".Git/HEAD", ".GIT/packed-refs", ".MEMFUN/verify.json", ".Env", ".ENV"):
        for cmd in (f"cat {path}", f"grep -rn . {path}", f"cd {path.split('/')[0]} && ls"):
            for got in answer(root, cmd):
                text = got.heading + got.output
                assert "future-ref" not in text and "in-project-secret" not in text, (cmd, got)
                assert "refs/heads" not in text and "reproduce" not in text, (cmd, got)


def test_a_link_inside_the_project_to_an_off_limits_file_is_not_read(root: Path) -> None:
    (root / "pkg/env_alias.txt").symlink_to("../.env")
    (root / "pkg/git_alias.txt").symlink_to("../.git/packed-refs")
    (root / "pkg/state_alias.txt").symlink_to("../.memfun/verify.json")
    everything = one(root, "grep -rn . pkg").output + one(root, "ls -la pkg").output
    everything += one(root, "find pkg -type f | xargs grep -n .").output
    for leaked in ("in-project-secret", "future-ref", "reproduce", "_alias.txt ("):
        assert leaked not in everything
    assert not one(root, "cat pkg/env_alias.txt").ok
    for name in (".envrc", ".env-local", ".git-credentials", ".pgpass", "kubeconfig", "k.p12"):
        (root / name).write_text("NEW-SECRET\n")
    assert "NEW-SECRET" not in one(root, "grep -rn SECRET .").output


def test_the_answers_of_a_round_cannot_outgrow_its_budget(root: Path) -> None:
    flood = {"op": "run_cmd", "cmd": ";".join(f"a{i}" for i in range(5000))}
    served = rr.serve(root, rr.requests_of([flood]))
    assert len(served) <= rr.MAX_REQUESTS + 1
    assert len(rr.format_served(served)) < 20_000
    (root / "pkg/SOURCES.txt").write_text("pkg/big.py\n" * 50)
    wide = one(root, "cat pkg/SOURCES.txt | xargs grep -n -C 20 'X = 1'")
    assert len(wide.output) <= rr.PER_REQUEST_BYTES + 200
    long_name = rr.serve(root, rr.requests_of([{"op": "search", "pattern": "x" * 900}]))
    assert len(long_name[0].heading) < 400


def test_the_read_log_is_not_written_through_a_link(root: Path, tmp_path: Path) -> None:
    victim = tmp_path / "outside" / "victim.txt"
    victim.write_text("ORIGINAL\n")
    (root / rr.READ_LOG).symlink_to(victim)
    rr.log_reads(root, 1, answer(root, "cat pkg/nope.py"))
    assert victim.read_text() == "ORIGINAL\n"
