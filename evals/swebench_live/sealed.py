#!/usr/bin/env python3
"""Grading that nobody can read before the run is over.

A 300-instance run on boxes with limited disk and a limit on image pulls has to grade an
instance and delete its image soon after its rollout.  The rule stays: no result of the
ground-truth evaluation may reach the agent, or whoever steers the run, before every rollout
is final.  The mechanism:

* an instance is graded only when its record is final (``status: done``); the runner never
  rolls a final instance out again, so its verdict cannot influence its own rollout;
* everything the evaluator writes (reports, test logs, its own output and exit code) goes
  under SEALED, a directory outside the run directory with mode 0700.  Nothing in the runner
  or the reports opens it.  The run directory only learns THAT an evaluation ran
  (``sealed.jsonl``: instance, step, time), never what it said;
* ``unseal`` refuses unless ``run.json`` is finalized.  It then runs the official evaluation
  once over all instances (existing per-instance reports are reused, missing ones are
  evaluated) and copies the official ``results.json`` next to ``preds.json``.

The gold check (the reference patch on the same image, no agent involved) can run in the same
pass and is sealed the same way: which tasks are broken is known only afterwards.

    sealed.py grade  --dir EVAL --run-dir RUN --sealed SEALED --split lite --ids ID [--gold]
    sealed.py unseal --dir EVAL --run-dir RUN --sealed SEALED --split lite
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.swebench_live import grade as gr

NOTE = (
    "Sealed evaluation output of a run in progress.  Do not open anything here before the run "
    "is finalized: `sealed.py unseal` is the only reader.\n"
)


def make(sealed: Path) -> Path:
    """Create the sealed directory (owner only) and return it."""
    sealed.mkdir(parents=True, exist_ok=True)
    sealed.chmod(0o700)
    for sub in ("preds", "logs", "eval", "gold"):
        (sealed / sub).mkdir(exist_ok=True)
    (sealed / "DO-NOT-READ.txt").write_text(NOTE, "utf-8")
    return sealed


def _is_final(run_dir: Path, iid: str) -> bool:
    try:
        doc = json.loads((run_dir / "instances" / iid / "instance.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("status") == "done"


def _evaluate_quietly(
    base: Path, patch: str, out: Path, split: str, iid: str, remove: bool, log: Path
) -> None:
    """One official evaluation whose output and exit code stay under the sealed directory."""
    clone = base / "SWE-bench-Live"
    cmd = [
        str(base / "venv" / "bin" / "python"), "-m", "evaluation.evaluation",
        "--dataset", gr.DATASET, "--split", split, "--platform", "linux",
        "--patch_dir", patch, "--output_dir", str(out),
        "--workers", "1", "--overwrite", "0", "--instance_ids", iid,
        *(["--remove-image"] if remove else []),
    ]  # fmt: skip
    with log.open("ab") as fh:
        code: int | str
        try:
            code = subprocess.run(
                cmd, cwd=clone, stdout=fh, stderr=subprocess.STDOUT, check=False, timeout=7200
            ).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = type(exc).__name__
        fh.write(f"\n[sealed] evaluator ended: {code}\n".encode())


def grade_final(
    base: Path,
    run_dir: Path,
    sealed: Path,
    iid: str,
    split: str,
    *,
    gold: bool = False,
    remove_image: bool = True,
) -> dict[str, object]:
    """Grade one FINAL instance into the sealed directory (and gold-check its image).

    Returns what may be known now: that the steps ran.  No verdict, no exit code.
    """
    if not _is_final(run_dir, iid):
        raise SystemExit(f"{iid} is not final: only a finished rollout may be graded")
    make(sealed)
    patch_file = run_dir / "instances" / iid / "patch.diff"
    patch = patch_file.read_text("utf-8") if patch_file.exists() else ""
    model = json.loads((run_dir / "run.json").read_text("utf-8"))["settings"]["model"]
    preds = sealed / "preds" / f"{iid}.json"
    preds.write_text(
        json.dumps({iid: {"model_name_or_path": model, "model_patch": patch}}), "utf-8"
    )
    record: dict[str, object] = {"instance_id": iid}
    _evaluate_quietly(
        base, str(preds), sealed / "eval", split, iid, remove_image and not gold,
        sealed / "logs" / f"{iid}.agent.log",
    )  # fmt: skip
    record["agent_patch_evaluated_at"] = time.time()
    if gold:
        _evaluate_quietly(
            base, "gold", sealed / "gold", split, iid, remove_image,
            sealed / "logs" / f"{iid}.gold.log",
        )  # fmt: skip
        record["gold_checked_at"] = time.time()
    with (run_dir / "sealed.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def unseal(base: Path, run_dir: Path, sealed: Path, split: str, workers: int) -> dict[str, object]:
    """After the run is finalized: the official results over every instance."""
    try:
        meta = json.loads((run_dir / "run.json").read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"no readable run.json in {run_dir}") from exc
    if meta.get("finalized") is not True:
        raise SystemExit(
            "this run is not finalized: the sealed results stay sealed until every rollout "
            "is final (runner.py finalize first)"
        )
    ids = list(meta["instance_ids"])
    res = gr.evaluate(
        base, str(run_dir / "preds.json"), sealed / "eval", split, workers, ids, True
    )
    shutil.copyfile(res, run_dir / "results.json")
    out: dict[str, object] = {
        k: v for k, v in json.loads(res.read_text("utf-8")).items() if not k.endswith("_ids")
    }
    if any((sealed / "gold").iterdir()):
        gold = gr.evaluate(base, "gold", sealed / "gold", split, workers, ids, True)
        shutil.copyfile(gold, run_dir / "gold-results.json")
        doc = json.loads(gold.read_text("utf-8"))
        out["gold"] = {k: v for k, v in doc.items() if not k.endswith("_ids")}
    shutil.copytree(sealed / "eval", run_dir / "eval", dirs_exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grade", "unseal"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, required=True, help="where the evaluator lives")
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--sealed", type=Path, required=True)
        p.add_argument("--split", default="lite", choices=("lite", "verified", "full", "test"))
        if name == "grade":
            p.add_argument("--ids", nargs="+", required=True)
            p.add_argument("--gold", action="store_true")
            p.add_argument("--keep-images", action="store_true")
        else:
            p.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    base, run_dir, sealed = a.dir.resolve(), a.run_dir.resolve(), a.sealed.resolve()
    if a.cmd == "grade":
        for iid in a.ids:
            grade_final(
                base, run_dir, sealed, iid, a.split, gold=a.gold, remove_image=not a.keep_images
            )
            print(f"{iid}: evaluated (sealed)", flush=True)
        return 0
    print(json.dumps(unseal(base, run_dir, sealed, a.split, a.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
