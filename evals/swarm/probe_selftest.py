"""Self-test of the look step's probe interpreter (run under the episode sandbox by
run_harness_benchmark.sh, with memfun's own interpreter).

For each library name: a project-local stand-in module is created in a scratch project; a probe
that imports it must see the PROJECT copy, never an installed one, and a probe that reads the
installed copy's file by path must be refused.  Exit 0 only if every line holds.

Usage: python probe_selftest.py LIB [LIB ...]
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

from memfun_agent import fixloop


def main(libs: list[str]) -> int:
    fail = 0
    with tempfile.TemporaryDirectory(prefix="probe-selftest-") as tmp:
        proj = Path(tmp)
        for lib in libs:
            installed = None
            try:
                spec = importlib.util.find_spec(lib)
                installed = spec.origin if spec else None
            except (ImportError, ValueError):
                pass
            (proj / f"{lib}.py").write_text("WHERE = 'project'\n")
            run = lambda s: fixloop_run(proj, s)  # noqa: E731
            out = run(f"import {lib}; print({lib}.WHERE)")
            ok = "=> project" in out
            print(f"  {'ok' if ok else 'FAIL'}: probe imports the project copy of {lib}")
            fail += not ok
            if installed and Path(installed).resolve() != (proj / f"{lib}.py").resolve():
                out = run(f"print(open({installed!r}).read()[:20])")
                ok = "PermissionError" in out
                print(f"  {'ok' if ok else 'FAIL'}: probe cannot read the installed {lib} by path")
                fail += not ok
            (proj / f"{lib}.py").unlink()
        for label, snippet in (
            ("writing a file", "open('x.txt', 'w').write('1')"),
            ("a child process", "import os; os.system('true')"),
            ("the network", "import socket; socket.create_connection(('127.0.0.1', 9), 1)"),
        ):
            out = fixloop_run(proj, snippet)
            ok = "Error" in out or "refused" in out
            print(f"  {'ok' if ok else 'FAIL'}: probe denied {label}")
            fail += not ok
    return 1 if fail else 0


def fixloop_run(proj: Path, snippet: str) -> str:
    return asyncio.run(fixloop.run_probes(proj, [snippet], strict=True))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
