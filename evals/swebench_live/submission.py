#!/usr/bin/env python3
"""From finished runs to a leaderboard submission folder, with a self-audit.

    submission.py merge --run-dir A --run-dir B ... --out RUN     # shards of one run, one per box
    submission.py audit --run-dir RUN --dataset lite.jsonl        # every trajectory, every rule
    submission.py build --run-dir RUN --dataset lite.jsonl --out submissions/lite/memfun/MODEL

``merge`` joins the shards of one run (same frozen settings, disjoint instances, all
finalized) into one run directory.  ``audit`` checks every instance against the rules of the
benchmark and prints what it found; ``build`` refuses to write a folder while the audit has
violations.  The folder holds what github.com/SWE-bench-Live/submission asks for:
``preds.json``, ``results.json`` (as the official script wrote it), ``trajs/`` (per instance:
every model request and response as sent and received, the agent's log, its patch and the
log of its own checks) and ``README.md`` (the scaffold, the frozen settings, the number of
rollouts, the audit).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.swebench_live import grade as gr

#: Dataset fields the agent must never see (rule 1); checked line by line.
FORBIDDEN_FIELDS = ("hints_text", "all_hints_text", "test_patch", "patch")
#: A line this long that is in a forbidden field and not in the problem statement is
#: distinctive.  One such line in a prompt can still be the repository's own text (a patch
#: often repeats a line that exists elsewhere), so it is a note for review; half of a
#: field's distinctive lines (and at least two) in one prompt is a violation.
MIN_LINE = 40
KEY_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{20,}", r"csk-[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{30,}",
    r"hf_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"Bearer [A-Za-z0-9._-]{24,}",
)  # fmt: skip
#: Signs in an agent log that something tried to reach the network (a violation).
NETWORK_ATTEMPT = (
    r"Web search failed", r"web fetch error", r"Model metadata refresh failed",
    r"Search Web:", r"=== WEB SEARCH RESULTS ===",
)  # fmt: skip
#: Signs that a command the agent ran (a test, a build) wanted the network: listed for review.
NETWORK_NOISE = (
    r"Temporary failure in name resolution", r"Could not resolve host", r"Max retries exceeded",
    r"Network is unreachable", r"NameResolutionError",
)  # fmt: skip
#: Copied into trajs/<instance>/ when present.
TRAJ_FILES = (
    "llm.jsonl", "agent.log", "patch.diff", "instance.json", "swarm-result.json",
    "checks.jsonl", "reads.jsonl", "probes.jsonl", "issue_repro.py",
)  # fmt: skip


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def load_rows(dataset: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in dataset.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["instance_id"])] = row
    return rows


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(str(v) for v in value) if isinstance(value, list) else ""


def distinctive_lines(row: dict[str, Any], field: str) -> list[str]:
    """Lines of a forbidden field that are long and absent from the problem statement.

    Of a diff (``patch``, ``test_patch``) only the ADDED lines count, and not those the same
    diff also has as removed or unchanged lines: everything else in a diff is the repository
    as it already was, which the agent reads legitimately.
    """
    statement = _text(row.get("problem_statement"))
    text = _text(row.get(field))
    is_diff = field in ("patch", "test_patch")
    old: set[str] = set()
    if is_diff:
        for line in text.splitlines():
            if line.startswith(("+++", "---", "@@", "diff ", "index ")):
                continue
            if not line.startswith("+"):
                old.add(line[1:].strip())
    out: list[str] = []
    for line in text.splitlines():
        if is_diff:
            if not line.startswith("+") or line.startswith("+++"):
                continue
            line = line[1:]
        line = line.strip()
        if len(line) >= MIN_LINE and line not in statement and line not in old:
            out.append(line)
    return out


def _content(message: dict[str, Any]) -> str:
    c = message.get("content")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def audit_instance(idir: Path, row: dict[str, Any], prompt_template: str) -> dict[str, list[str]]:
    """Violations and things to review for one instance (both empty = clean)."""
    bad: list[str] = []
    review: list[str] = []
    iid = str(row["instance_id"])
    rec = read_json(idir / "instance.json") or {}
    if rec.get("status") != "done":
        bad.append(f"not final (status {rec.get('status')!r})")
    net = rec.get("network_selftest")
    if net is not None and (
        not net.get("proxy") or net.get("internet") or net.get("dns") or net.get("host_ports")
    ):
        bad.append(f"network self-test: {json.dumps(net)}")
    if "lockdown ok" not in str(rec.get("lockdown", "")) and rec.get("failure_class") is None:
        bad.append("no history lock-down recorded")
    calls: list[dict[str, Any]] = []
    try:
        for line in (idir / "llm.jsonl").read_text("utf-8").splitlines():
            if line.strip():
                calls.append(json.loads(line))
    except (OSError, ValueError):
        if (rec.get("usage") or {}).get("calls"):
            bad.append("the trajectory (llm.jsonl) is missing or unreadable")
    expected = prompt_template.format(problem_statement=_text(row.get("problem_statement")))
    answered = ""  # everything the model itself has written so far
    leaked: set[str] = set()
    first_user = ""
    for call in calls:
        messages = (call.get("request") or {}).get("messages") or []
        asked = "\n".join(_content(m) for m in messages if isinstance(m, dict))
        seq = call.get("seq")
        if not first_user:
            first_user = asked
        if iid in asked:
            bad.append(f"call {seq}: the instance id is in a prompt")
        for field in FORBIDDEN_FIELDS:
            lines = list(dict.fromkeys(distinctive_lines(row, field)))
            # the agent's own earlier words coming back in a prompt are not a leak
            hits = [ln for ln in lines if ln in asked and ln not in answered]
            if not hits or field in leaked:
                continue
            if len(hits) >= 2 and len(hits) * 2 >= len(lines):
                leaked.add(field)
                bad.append(
                    f"call {seq}: {len(hits)} of the {len(lines)} distinctive lines of "
                    f"`{field}` are in a prompt, e.g. {hits[0][:80]}"
                )
            else:
                # one line, or a small part: the repository may hold the same text elsewhere
                for ln in hits:
                    note = f"a line that is also in `{field}` is in a prompt: {ln[:80]}"
                    if note not in review:
                        review.append(note)
        for name in _text(row.get("FAIL_TO_PASS")).splitlines():
            test = name.rsplit("::", 1)[-1].split("[", 1)[0]
            seen = len(test) >= 12 and test in asked and test not in answered
            if seen and test not in _text(row.get("problem_statement")):
                review.append(f"call {seq}: the name of a target test is in a prompt: {test}")
        with contextlib.suppress(KeyError, IndexError, TypeError):
            answered += "\n" + _content(call["response"]["choices"][0]["message"])
    if calls and expected not in first_user:
        bad.append("the first prompt is not the prompt template with the problem statement")
    log = ""
    with contextlib.suppress(OSError):
        log = (idir / "agent.log").read_text("utf-8", errors="replace")
    for pat in NETWORK_ATTEMPT:
        if re.search(pat, log):
            bad.append(f"the agent log shows an attempted network access: {pat}")
    for pat in NETWORK_NOISE:
        if re.search(pat, log):
            review.append(f"a command the agent ran wanted the network: {pat}")
    for name in TRAJ_FILES:
        try:
            text = (idir / name).read_text("utf-8", errors="replace")
        except OSError:
            continue
        for pat in KEY_PATTERNS:
            m = re.search(pat, text)
            if m:
                bad.append(f"{name}: text that looks like a credential ({pat[:12]}...)")
    return {"violations": bad, "review": review}


def audit(run_dir: Path, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The audit of a whole run: per instance, plus the rules that hold for the run."""
    meta = read_json(run_dir / "run.json") or {}
    settings = meta.get("settings") or {}
    ids = list(meta.get("instance_ids") or [])
    out: dict[str, Any] = {"instances": {}, "run": []}
    if meta.get("finalized") is not True:
        out["run"].append("run.json is not finalized")
    if len(ids) != len(set(ids)):
        out["run"].append("an instance is listed twice")
    for iid in ids:
        if iid not in rows:
            out["run"].append(f"{iid} is not in the dataset file")
            continue
        found = audit_instance(
            run_dir / "instances" / iid, rows[iid], str(settings.get("prompt_template", ""))
        )
        if found["violations"] or found["review"]:
            out["instances"][iid] = found
    attempts: list[dict[str, Any]] = []
    try:
        for line in (run_dir / "attempts.jsonl").read_text("utf-8").splitlines():
            attempts.append(json.loads(line))
    except (OSError, ValueError):
        pass
    for a in attempts:
        if a.get("status") not in ("infra_error", "agent_crash") or not a.get("error"):
            out["run"].append(
                f"an attempt of {a.get('instance_id')} was set aside without a documented "
                "infrastructure reason"
            )
    preds = read_json(run_dir / "preds.json") or {}
    if set(preds) != set(ids):
        out["run"].append("preds.json does not hold exactly the run's instances")
    out["n_instances"] = len(ids)
    out["n_restarts"] = sum(1 for a in attempts if not a.get("final"))
    per_instance = out["instances"].values()
    out["n_violations"] = len(out["run"]) + sum(len(v["violations"]) for v in per_instance)
    out["n_review"] = sum(len(v["review"]) for v in per_instance)
    return out


def merge(shards: list[Path], out: Path) -> dict[str, Any]:
    """Join the shards of one run.  Their frozen settings must be identical."""
    metas = [read_json(s / "run.json") for s in shards]
    if any(not isinstance(m, dict) or m.get("finalized") is not True for m in metas):
        raise SystemExit("every shard must be finalized")
    first = metas[0]["settings"]
    for s, m in zip(shards, metas, strict=True):
        if m["settings"] != first:
            diff = sorted(k for k in first if m["settings"].get(k) != first[k])
            raise SystemExit(f"{s} was run with different settings ({diff}): not one run")
    ids: list[str] = [i for m in metas for i in m["instance_ids"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("an instance is in more than one shard: it was rolled out twice")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "instances").mkdir(parents=True)
    preds: dict[str, Any] = {}
    with (out / "attempts.jsonl").open("w", encoding="utf-8") as log:
        for s in shards:
            for idir in sorted((s / "instances").glob("*")):
                shutil.copytree(idir, out / "instances" / idir.name)
            if (s / "attempts").is_dir():
                shutil.copytree(s / "attempts", out / "attempts", dirs_exist_ok=True)
            if (s / "attempts.jsonl").is_file():
                log.write((s / "attempts.jsonl").read_text("utf-8"))
            preds.update(read_json(s / "preds.json") or {})
    doc = {
        "settings": first,
        "instance_ids": ids,
        "created_at": min(m["created_at"] for m in metas),
        "finalized": True,
        "finalized_at": max(m.get("finalized_at", 0) for m in metas),
        "n_instances": len(ids),
        "merged_from": [str(s) for s in shards],
    }
    (out / "run.json").write_text(json.dumps(doc, indent=1), "utf-8")
    (out / "preds.json").write_text(json.dumps(preds, indent=1), "utf-8")
    return doc


def readme(run_dir: Path, report: dict[str, Any], name: str, model_label: str) -> str:
    meta = read_json(run_dir / "run.json") or {}
    s = meta.get("settings") or {}
    res = read_json(run_dir / "results.json") or {}
    tokens = cost = 0.0
    for iid in meta.get("instance_ids") or []:
        rec = read_json(run_dir / "instances" / iid / "instance.json") or {}
        u = rec.get("usage") or {}
        tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        cost += rec.get("cost_usd", 0.0)
    n = max(1, len(meta.get("instance_ids") or []))
    summary = {k: v for k, v in res.items() if not k.endswith("_ids")}
    options = json.dumps(s.get("agent_options") or {}, sort_keys=True)
    return f"""# {name} + {model_label}

## Scaffold
{name} (https://github.com/indoor47/memfun), commit `{s.get("agent_version", "")}` (bundle
digest `{s.get("bundle_digest", "")}`), run by `evals/swebench_live/runner.py` at commit
`{s.get("runner_commit", "")}`. The agent runs inside the instance's own Docker image, sees only
`problem_statement` in the prompt template below, has no internet (an internal Docker network
on which only the recording model proxy answers) and no git history beyond the base commit.

## Settings (frozen before the run, identical for every instance)
- model `{s.get("model")}`, temperature {s.get("temperature")}, max output tokens
  {s.get("llm_max_tokens")}, context window {s.get("context_window")}, reasoning effort
  `{s.get("reasoning_effort")}`
- solver `{s.get("solver")}`, agent options `{options}`
- per instance: at most {s.get("max_tokens_per_instance")} tokens and
  {s.get("instance_timeout")} s; {s.get("cpus")} CPUs, {s.get("memory")} memory
- prompt template: `{json.dumps(s.get("prompt_template", ""))}`

## Rollouts
One rollout per instance ({n} instances). {report.get("n_restarts", 0)} attempts were restarted
from scratch for documented infrastructure reasons (the model endpoint did not answer, or the
agent process ended before its first model call); each is listed in `attempts.jsonl` and kept
under `trajs/_attempts/`. The evaluation was run by the official script
(microsoft/SWE-bench-Live at `{gr.EVALUATOR_COMMIT}`) only after every rollout was final.

## Result
`{json.dumps(summary)}`; {tokens / 1e6:.1f}M tokens in total ({tokens / n / 1000:.0f}K per
instance), ${cost:.2f} at the endpoint's list price.

## Self-audit
Every trajectory was scanned for the dataset's forbidden fields (hints, test patch, reference
patch), the instance id, credentials and attempted network access:
{report.get("n_violations", 0)} violations, {report.get("n_review", 0)} notes for review
(`audit.json`).

## Files
`preds.json`, `results.json`, `run.json`, `attempts.jsonl`, `audit.json`, and per instance in
`trajs/<instance>/`: `llm.jsonl` (every request and response as sent and received, with
timestamps and usage), `agent.log`, `patch.diff`, `instance.json` and the agent's own check
logs.
"""


def build(run_dir: Path, rows: dict[str, dict[str, Any]], out: Path, name: str, label: str) -> int:
    report = audit(run_dir, rows)
    if report["n_violations"]:
        print(json.dumps(report, indent=1)[:4000])
        print(f"NOT BUILT: {report['n_violations']} violations")
        return 3
    if not (run_dir / "results.json").is_file():
        raise SystemExit("no results.json: grade the finalized run first (sealed.py unseal)")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "trajs").mkdir(parents=True)
    for fname in ("preds.json", "results.json", "run.json", "attempts.jsonl"):
        if (run_dir / fname).is_file():
            shutil.copyfile(run_dir / fname, out / fname)
    for idir in sorted((run_dir / "instances").glob("*")):
        dest = out / "trajs" / idir.name
        dest.mkdir()
        for fname in TRAJ_FILES:
            if (idir / fname).is_file():
                shutil.copyfile(idir / fname, dest / fname)
    if (run_dir / "attempts").is_dir():
        shutil.copytree(run_dir / "attempts", out / "trajs" / "_attempts")
    (out / "audit.json").write_text(json.dumps(report, indent=1), "utf-8")
    (out / "README.md").write_text(readme(run_dir, report, name, label), "utf-8")
    print(f"built {out}: {report['n_instances']} instances, {report['n_review']} notes for review")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--run-dir", type=Path, action="append", required=True)
    m.add_argument("--out", type=Path, required=True)
    for cmd in ("audit", "build"):
        p = sub.add_parser(cmd)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--dataset", type=Path, required=True, help="JSONL from dataset.py fetch")
        if cmd == "build":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--name", default="memfun")
            p.add_argument("--model-label", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        doc = merge([d.resolve() for d in a.run_dir], a.out.resolve())
        print(f"merged {len(doc['instance_ids'])} instances into {a.out}")
        return 0
    rows = load_rows(a.dataset)
    if a.cmd == "audit":
        report = audit(a.run_dir.resolve(), rows)
        print(json.dumps(report, indent=1))
        return 3 if report["n_violations"] else 0
    return build(a.run_dir.resolve(), rows, a.out.resolve(), a.name, a.model_label)


if __name__ == "__main__":
    sys.exit(main())
