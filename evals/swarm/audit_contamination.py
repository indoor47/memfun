#!/usr/bin/env python3
"""Contamination audit of a harness-benchmark run: did an agent SEE reference code?

For every episode of the given run directories it answers two questions:

1. **Probe**: did a command / tool call reach outside the episode repo (``find /``, ``pip show``,
   ``inspect.getsource``, ``git show`` of another repo, the commit0 clones/bundles cache, the
   home directory, site-packages ...)?
2. **Leak**: did any observation (command output, file read) contain the reference
   implementation of the target function?  The reference is the bundle's ``solution/`` file;
   a leak is >= 30 % (and >= 2) of the function body's distinctive lines (lines that are not
   already in the starting stub) appearing verbatim in ONE observation.

Each leak is attributed to the command that produced it and classified by source
(``solution/hidden bundle cache``, ``clones cache``, ``site-packages / pip``, ``other``).

Also reported, as information not proof: whether the final submitted function equals the
reference (normalised line overlap).

Arms: ``mini-swe-agent`` (trajectory file), ``memfun`` (result ops + log), ``aider`` (log;
no shell tool, only the visible test command), ``single-shot`` (no tools).

Usage::

    python evals/swarm/audit_contamination.py --run RUN_DIR [--run ...] \
        [--bundles ~/.cache/memfun-commit0/bundles] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

#: Command patterns that reach outside the episode repo.
PROBES: dict[str, re.Pattern[str]] = {
    "find-root": re.compile(r"\bfind\s+(/(?=\s|$)|/\w|~|\$HOME)"),
    "pip": re.compile(r"\bpip3?\s+(show|download|install|list|cache)\b|-m\s+pip\b"),
    "getsource": re.compile(r"inspect\.getsource|__file__|importlib|pkgutil"),
    "git-other": re.compile(
        r"\bgit\s+(-C\s+\S+\s+)?(show|log|cat-file|rev-list|reflog|stash|worktree)\b"
    ),
    "site-packages": re.compile(r"site-packages|dist-packages"),
    "cache-dirs": re.compile(r"~/\.cache|\.cache/|memfun-commit0|/clones/|/bundles/|/solution/"),
    "home-listing": re.compile(r"(~|\$HOME|/Users/\w+)(/Ventures|/Developer|/\.|\s|/?$)"),
    "outside-abs-path": re.compile(r"(?<![\w.])/(Users|private|tmp|var|opt|usr/local)/[\w.\-/]+"),
}
LEAK_FRACTION = 0.3
MIN_LINE = 10  # shortest line (normalised) that counts as distinctive


def norm(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def function_source(text: str, qualname: str) -> str:
    """Source of ``qualname`` (``f`` or ``Class.method``) in *text*, or ""."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    parts = qualname.split(".")
    nodes: list[ast.stmt] = list(tree.body)
    found: ast.AST | None = None
    for i, name in enumerate(parts):
        found = None
        for n in nodes:
            if (
                isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and n.name == name
            ):
                found = n
                break
        if found is None:
            return ""
        if i < len(parts) - 1:
            nodes = list(getattr(found, "body", []))
    return ast.get_source_segment(text, found) or "" if found is not None else ""


class Reference:
    """Distinctive reference lines of one bundle's target function."""

    def __init__(self, bundle: Path) -> None:
        task = tomllib.loads((bundle / "task.toml").read_text("utf-8"))
        src = task.get("source", {})
        rel, qual = src.get("file", ""), src.get("function", "")
        ref_text = (bundle / "solution" / rel).read_text("utf-8", "replace") if rel else ""
        stub_text = (bundle / "repo" / rel).read_text("utf-8", "replace") if rel else ""
        body = function_source(ref_text, qual)
        stub_lines = {norm(x) for x in stub_text.splitlines()}
        self.rel, self.qual = rel, qual
        self.lines = [
            n
            for n in (norm(x) for x in body.splitlines())
            if len(n) >= MIN_LINE and n not in stub_lines and not n.startswith(("#", '"""'))
        ]
        self.all_body = {norm(x) for x in body.splitlines()}
        # Distinctive lines of the WHOLE original file (not in the stub): seeing many of them
        # means the agent read the original library source, even if the target function is
        # too short for the per-function test.
        self.file_lines = sorted(
            {
                n
                for n in (norm(x) for x in ref_text.splitlines())
                if len(n) >= 20 and n not in stub_lines and not n.startswith(("#", '"""'))
            }
        )

    def overlap(self, text: str) -> tuple[int, int]:
        """(distinctive lines found in *text*, distinctive lines total)."""
        hay = {norm(x) for x in text.splitlines()}
        return sum(1 for ln in self.lines if ln in hay), len(self.lines)

    def file_exposed_in(self, text: str, min_lines: int = 5, seen: set[str] | None = None) -> int:
        """Distinctive original-file lines NEW in *text* (not in *seen*) when >= *min_lines*."""
        hay = {norm(x) for x in text.splitlines()} - (seen or set())
        n = sum(1 for ln in self.file_lines if ln in hay)
        return n if n >= min_lines else 0

    def leaked_in(self, text: str, seen: set[str] | None = None) -> bool:
        """True when >= 30 % (and >= 2) of the distinctive lines are NEW in *text*.

        *seen* holds lines the agent already wrote or was already shown: re-displaying its
        own edit (``cat``, ``git diff``) is not a leak.
        """
        found, total = self.overlap(
            "\n".join(x for x in text.splitlines() if norm(x) not in (seen or set()))
        )
        return (
            total > 0
            and found >= max(1, int(LEAK_FRACTION * total + 0.999))
            and (found >= 2 or total == 1)
        )


#: Path classes that must never be reachable by an agent: the reference implementation,
#: the hidden grader tests, the original library clones, the bundle cache itself.
PATH_CLASSES: dict[str, re.Pattern[str]] = {
    "reference-solution": re.compile(r"/solution/"),
    "hidden-tests": re.compile(r"/hidden/"),
    "bundle-cache": re.compile(r"/bundles/c0"),
    "clones-cache": re.compile(r"memfun-commit0|/clones/"),
}
WRITES_RE = re.compile(r"sed\s+-i|>\s*/|tee\s|\bcp\s|\bmv\s|git\s+(apply|checkout|stash)")


def classify_source(command: str, output: str) -> str:
    blob = command + "\n" + output
    if re.search(r"/solution/|/bundles/|/hidden/", blob):
        return "bundle cache (reference solution / hidden tests)"
    if re.search(r"memfun-commit0|/clones/", blob):
        return "clones cache (original library repo)"
    if re.search(r"site-packages|dist-packages|pip", blob):
        return "site-packages / pip"
    return "other"


def probes_in(command: str, repo: str) -> list[str]:
    hits = [name for name, rx in PROBES.items() if rx.search(command)]
    if "outside-abs-path" in hits:
        # absolute paths that are the episode repo itself are fine
        stripped = command.replace(repo, "")
        if not PROBES["outside-abs-path"].search(stripped):
            hits.remove("outside-abs-path")
    return hits


# ── Per-arm extraction: [(command, output)] ─────────────────────────────────────

CMD_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.S)


def mini_steps(ep: Path) -> list[tuple[str, str]]:
    f = ep / "home" / "mini-trajectory.json"
    try:
        msgs = json.loads(f.read_text("utf-8")).get("messages", [])
    except (OSError, ValueError):
        return []
    steps: list[tuple[str, str]] = []
    pending: list[str] = []
    for m in msgs:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        if m.get("role") == "assistant":
            pending = CMD_RE.findall(text)
        elif m.get("role") == "user" and pending:
            steps.append(("\n".join(pending), text))
            pending = []
    return steps


def probe_steps(ep: Path) -> list[tuple[str, str]]:
    """The look step's probe scripts (``.memfun/probes.jsonl`` in the episode repo): each is a
    command that ran inside the episode, with its output."""
    steps: list[tuple[str, str]] = []
    for log in ep.glob("*/.memfun/probes.jsonl"):
        try:
            for line in log.read_text("utf-8").splitlines():
                rec = json.loads(line)
                out = str(rec.get("output", ""))
                for script in rec.get("scripts", []):
                    steps.append((f"python -c {script}", out))
        except (OSError, ValueError):
            continue
    return steps


def memfun_steps(ep: Path) -> list[tuple[str, str]]:
    f = ep / "swarm-result.json"
    try:
        doc = json.loads(f.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    steps: list[tuple[str, str]] = []
    for st in (doc.get("sub_tasks") or {}).values():
        for op in st.get("ops") or []:
            steps.append((f"{op.get('type')} {op.get('target')} {op.get('detail')}", ""))
    return steps + probe_steps(ep)


def aider_steps(ep: Path) -> list[tuple[str, str]]:
    f = ep / "home" / "arm.log"
    return [("aider session", f.read_text("utf-8", "replace"))] if f.is_file() else []


STEPS = {"mini-swe-agent": mini_steps, "memfun": memfun_steps, "aider": aider_steps}


def audit_episode(ep: Path, arm: str, ref: Reference | None) -> dict[str, Any]:
    rec = json.loads((ep / "episode.json").read_text("utf-8"))
    repo = rec.get("repo", "")
    steps = STEPS.get(arm, lambda _e: [])(ep)
    probes: dict[str, int] = {}
    probe_cmds: list[str] = []
    leaks: list[dict[str, str]] = []
    touches: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    outside_reads = 0
    seen: set[str] = set()  # lines the agent wrote (commands) or was shown (earlier outputs)
    for cmd, out in steps:
        hit = probes_in(cmd, repo) if arm != "aider" else []
        for h in hit:
            probes[h] = probes.get(h, 0) + 1
        if hit:
            probe_cmds.append(cmd.strip().splitlines()[0][:160] if cmd.strip() else "")
        if arm != "aider":
            for cls, rx in PATH_CLASSES.items():
                if rx.search(cmd.replace(repo, "")):
                    touches.append(
                        {
                            "class": cls,
                            "command": cmd.strip()[:220],
                            "output_bytes": len(out),
                            "writes": bool(WRITES_RE.search(cmd)),
                        }
                    )
        if (
            arm != "aider"
            and hit
            and len(out) > 200
            and re.search(
                r"memfun-commit0|/clones/|/solution/|/bundles/|site-packages|pip|getsource", cmd
            )
        ):
            outside_reads += 1
        if ref is not None and out and arm == "mini-swe-agent":
            n_exposed = ref.file_exposed_in(
                out, seen=seen | {norm(x) for x in cmd.replace("\\n", "\n").splitlines()}
            )
            if n_exposed:
                exposures.append(
                    {
                        "command": cmd.strip()[:200],
                        "original_lines_seen": n_exposed,
                        "source": classify_source(cmd, out),
                    }
                )
        new_seen = seen | {norm(x) for x in cmd.replace("\\n", "\n").splitlines()}
        if ref is not None and out and arm == "mini-swe-agent" and ref.leaked_in(out, new_seen):
            leaks.append({"command": cmd.strip()[:200], "source": classify_source(cmd, out)})
        seen = new_seen | {norm(x) for x in out.splitlines()}
    if arm == "aider":
        log = steps[0][1] if steps else ""
        outside = sorted(set(re.findall(r"(?:/Users|/private/tmp)[\w./\-]+", log)) - {repo})
        outside = [p for p in outside if repo not in p and "/home/" not in p]
        if outside:
            probes["aider-log-outside-path"] = len(outside)
    final = ""
    for f in (ep / "collected").rglob("*"):
        if f.is_file() and ref is not None and f.name == Path(ref.rel).name:
            final = f.read_text("utf-8", "replace")
    same = None
    if ref is not None and final and ref.lines:
        fn = function_source(final, ref.qual)
        found, total = ref.overlap(fn)
        same = round(found / total, 2) if total else None
    return {
        "episode": rec["episode"],
        "outcome": rec["outcome"],
        "commands": len(steps),
        "probes": probes,
        "probe_commands": probe_cmds[:6],
        "leaks": leaks,
        "outside_reads_with_output": outside_reads,
        "reference_path_touches": touches,
        "original_file_exposures": exposures,
        "final_matches_reference": same,
    }


def audit_run(run: Path, bundles: Path) -> dict[str, Any]:
    meta = json.loads((run / "results.json").read_text("utf-8")).get("meta", {})
    arm = meta.get("arm") or ""
    if not arm:  # run_smoke records the arm in every episode.json
        for f in sorted(run.glob("*/*/episode.json")):
            try:
                arm = str(json.loads(f.read_text("utf-8")).get("arm") or "")
            except (OSError, ValueError):
                arm = ""
            if arm:
                break
    arm = arm or run.name.removeprefix("bench-")
    for known in ("mini-swe-agent", "single-shot", "claude-code", "memfun", "aider"):
        if arm == known or arm.startswith(known + "-"):  # e.g. a tagged run: mini-swe-agent-hardD
            arm = known
            break
    refs: dict[str, Reference | None] = {}
    eps: list[dict[str, Any]] = []
    for f in sorted(run.glob("*/*/episode.json")):
        ep = f.parent
        bid = ep.parent.name  # run/<bundle>/<trial>/episode.json
        if bid not in refs:
            try:
                refs[bid] = Reference(bundles / bid)
            except (OSError, ValueError, KeyError) as exc:
                print(f"WARNING: no reference for bundle {bid}: {exc}", file=sys.stderr)
                refs[bid] = None
        eps.append(audit_episode(ep, arm, refs[bid]))
    probed = [e for e in eps if e["probes"]]
    leaked = [e for e in eps if e["leaks"] or e["original_file_exposures"]]
    return {
        "arm": arm,
        "episodes": len(eps),
        "episodes_with_probe": len(probed),
        "episodes_with_leak": len(leaked),
        "leak_sources": sorted(
            {lk["source"] for e in leaked for lk in e["leaks"] + e["original_file_exposures"]}
        ),
        "episodes_touching": {
            cls: sum(1 for e in eps if any(t["class"] == cls for t in e["reference_path_touches"]))
            for cls in PATH_CLASSES
        },
        "episodes_function_leak": sum(1 for e in eps if e["leaks"]),
        "episodes_file_exposure": sum(1 for e in eps if e["original_file_exposures"]),
        "episodes_outside_read": sum(1 for e in eps if e["outside_reads_with_output"]),
        "leaked_and_passed": sum(1 for e in leaked if e["outcome"] == "pass"),
        "probed_and_passed": sum(1 for e in probed if e["outcome"] == "pass"),
        "details": eps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument(
        "--bundles", type=Path, default=Path.home() / ".cache" / "memfun-commit0" / "bundles"
    )
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    out = [audit_run(r.expanduser().resolve(), a.bundles.expanduser()) for r in a.run]
    for r in out:
        print(
            f"{r['arm']:<16} episodes {r['episodes']:>3}  probed {r['episodes_with_probe']:>3}  "
            f"outside-read {r['episodes_outside_read']:>3}  "
            f"fn-leak {r['episodes_function_leak']:>3}  "
            f"orig-file {r['episodes_file_exposure']:>3}  touched {r['episodes_touching']}  "
            f"(leak/exposure+passed {r['leaked_and_passed']})  "
            f"sources: {', '.join(r['leak_sources']) or '-'}"
        )
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
