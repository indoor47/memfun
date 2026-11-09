#!/usr/bin/env python3
"""Official grading of a FINALIZED SWE-bench-Live run (and a gold-patch sanity check).

Rule: the agent must not use results of the ground-truth evaluation to refine its solution.
The mechanism: grading is a separate step that refuses a run whose ``run.json`` is not
finalized, and ``runner.py`` refuses to roll out anything in a finalized run.  So no rollout can
happen after any grading of that run.

    grade.py setup --dir /root/memfun-live/eval          # pinned clone + venv of the evaluator
    grade.py gold  --dir ... --split full --ids a b c    # do the images + evaluator agree?
    grade.py grade --dir ... --run-dir runs/dev1 --split full --workers 1

The evaluator is microsoft/SWE-bench-Live at a pinned commit (its ``launch`` submodule holds
the container runtime); ``results.json`` is copied next to ``preds.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EVALUATOR_REPO = "https://github.com/microsoft/SWE-bench-Live"
EVALUATOR_COMMIT = "afac77f2ce908ed37bc60ad0d42e7815a7589525"
DATASET = "SWE-bench-Live/SWE-bench-Live"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def setup(base: Path, uv: str) -> None:
    clone = base / "SWE-bench-Live"
    if not (clone / ".git").exists():
        base.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--recursive", EVALUATOR_REPO, str(clone)])
    run(["git", "-C", str(clone), "fetch", "-q", "origin"])
    run(["git", "-C", str(clone), "checkout", "-q", "--detach", EVALUATOR_COMMIT])
    run(["git", "-C", str(clone), "submodule", "update", "--init", "--recursive"])
    venv = base / "venv"
    if not (venv / "bin" / "python").exists():
        run([uv, "venv", "--python", "3.12", str(venv)])
    run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), "-e", ".", "./launch/"],
        cwd=clone)  # fmt: skip


def evaluate(
    base: Path, patch: str, out: Path, split: str, workers: int, ids: list[str], remove: bool
) -> Path:
    clone = base / "SWE-bench-Live"
    head = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != EVALUATOR_COMMIT:
        raise SystemExit(f"the evaluator clone is at {head}, not the pinned {EVALUATOR_COMMIT}")
    cmd = [
        str(base / "venv" / "bin" / "python"), "-m", "evaluation.evaluation",
        "--dataset", DATASET, "--split", split, "--platform", "linux",
        "--patch_dir", patch, "--output_dir", str(out),
        "--workers", str(workers), "--overwrite", "0",
        *(["--instance_ids", *ids] if ids else []),
        *(["--remove-image"] if remove else []),
    ]  # fmt: skip
    run(cmd, cwd=clone)
    return out / "results.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "gold", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, required=True, help="where the evaluator lives")
        if name == "setup":
            p.add_argument("--uv", default="uv")
            continue
        p.add_argument("--split", default="lite", choices=("lite", "verified", "full", "test"))
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--keep-images", action="store_true")
        if name == "gold":
            p.add_argument("--ids", nargs="+", required=True)
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    base = a.dir.resolve()
    if a.cmd == "setup":
        setup(base, a.uv)
        return 0
    if a.cmd == "gold":
        res = evaluate(base, "gold", a.out.resolve(), a.split, a.workers, a.ids, not a.keep_images)
        print(res.read_text("utf-8")[:2000])
        return 0
    run_dir = a.run_dir.resolve()
    try:
        meta = json.loads((run_dir / "run.json").read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"no readable run.json in {run_dir}") from exc
    if meta.get("finalized") is not True:
        raise SystemExit(
            "this run is not finalized: grading before every rollout is final would let "
            "evaluation results leak into later rollouts (runner.py finalize first)"
        )
    res = evaluate(
        base,
        str(run_dir / "preds.json"),
        run_dir / "eval",
        a.split,
        a.workers,
        list(meta["instance_ids"]),
        not a.keep_images,
    )
    shutil.copyfile(res, run_dir / "results.json")
    doc = json.loads(res.read_text("utf-8"))
    print(json.dumps({k: v for k, v in doc.items() if not k.endswith("_ids")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
