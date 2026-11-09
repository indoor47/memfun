#!/usr/bin/env python3
"""SWE-bench-Live dataset rows: fetch, select, and the ONE thing an agent may see of them.

The leaderboard rule: during a rollout the agent may access only the row's
``problem_statement`` and the container started from its ``docker_image``.  This module is the
only place a dataset row is read:

* :func:`episode_input` -- the agent's whole input, built from ``problem_statement`` alone;
* :class:`Task` -- what the RUNNER (not the agent) needs to drive the container: the image and
  the ids used for file names on the host.  Hints, test names, patches never leave the row.

Fetching uses the Hugging Face datasets-server JSON API (standard library only)::

    python evals/swebench_live/dataset.py fetch --split lite --out data/lite.jsonl
    python evals/swebench_live/dataset.py dev-sample --full data/full.jsonl \
        --exclude data/lite.jsonl data/verified.jsonl --n 30 --seed 18 --out data/dev30.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "SWE-bench-Live/SWE-bench-Live"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SPLITS = ("lite", "verified", "full", "test")
PAGE = 100
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
#: Fields of a row that must never reach an agent (everything but ``problem_statement``).
FORBIDDEN_FIELDS = (
    "instance_id",
    "pull_number",
    "issue_numbers",
    "base_commit",
    "patch",
    "test_patch",
    "hints_text",
    "all_hints_text",
    "commit_urls",
    "commit_url",
    "created_at",
    "test_cmds",
    "log_parser",
    "difficulty",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "rebuild_cmds",
    "docker_image",
)


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeInput:
    """Everything of a dataset row an agent is given."""

    problem_statement: str


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """What the runner needs of a row.  Never passed to an agent."""

    instance_id: str
    docker_image: str
    base_commit: str


def episode_input(row: dict[str, Any]) -> EpisodeInput:
    statement = row.get("problem_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("row has no problem_statement")
    return EpisodeInput(problem_statement=statement)


def task_of(row: dict[str, Any]) -> Task:
    iid, image = row.get("instance_id"), row.get("docker_image")
    if not isinstance(iid, str) or not INSTANCE_ID_RE.match(iid):
        raise ValueError(f"bad instance_id: {iid!r}")
    if not isinstance(image, str) or not image.strip():
        # The official evaluator's fallback (evaluation/evaluation.py, get_default_image_name).
        image = "starryzhang/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower()
    return Task(instance_id=iid, docker_image=image, base_commit=str(row.get("base_commit") or ""))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            doc = json.loads(line)
            if not isinstance(doc, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            rows.append(doc)
    return rows


def select(
    rows: list[dict[str, Any]],
    ids: list[str] | None = None,
    sample: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rows by explicit id list (in that order; unknown ids are an error) or a seeded sample."""
    if ids is not None:
        by_id = {r.get("instance_id"): r for r in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"instance ids not in the dataset file: {missing[:5]}")
        return [by_id[i] for i in dict.fromkeys(ids)]
    if sample:
        ordered = sorted(rows, key=lambda r: str(r.get("instance_id")))
        return random.Random(seed).sample(ordered, min(sample, len(ordered)))
    return rows


def dev_sample(
    full: list[dict[str, Any]], excluded: list[list[dict[str, Any]]], n: int, seed: int
) -> list[str]:
    """A seeded sample of ``full`` minus the frozen splits (never develop on those)."""
    frozen = {r.get("instance_id") for rows in excluded for r in rows}
    pool = sorted(str(r["instance_id"]) for r in full if r.get("instance_id") not in frozen)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def fetch(split: str, out: Path, sleep: float = 0.4) -> int:
    """Download one split to JSONL through the datasets-server rows API."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    total, offset = None, 0
    with tmp.open("w", encoding="utf-8") as fh:
        while total is None or offset < total:
            q = urllib.parse.urlencode(
                {
                    "dataset": DATASET,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": PAGE,
                }
            )
            doc = _get_json(f"{ROWS_API}?{q}")
            total = int(doc["num_rows_total"])
            rows = doc.get("rows") or []
            if not rows:
                break
            for item in rows:
                if item.get("truncated_cells"):
                    raise RuntimeError(
                        f"row {item.get('row_idx')} has truncated cells "
                        f"{item['truncated_cells']}: use the parquet files instead"
                    )
                fh.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
            offset += len(rows)
            time.sleep(sleep)
    if total is None or offset != total:
        raise RuntimeError(f"fetched {offset} of {total} rows of {split}")
    tmp.replace(out)
    return offset


def _get_json(url: str, attempts: int = 6) -> dict[str, Any]:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "swebench-live-runner"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                doc = json.load(resp)
            if isinstance(doc, dict) and "rows" in doc:
                return doc
            raise RuntimeError(f"unexpected answer: {str(doc)[:200]}")
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2.0 * (i + 1))
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--split", required=True, choices=SPLITS)
    f.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("dev-sample")
    d.add_argument("--full", type=Path, required=True)
    d.add_argument("--exclude", type=Path, nargs="+", required=True)
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(f"{fetch(a.split, a.out)} rows -> {a.out}")
    else:
        ids = dev_sample(load_rows(a.full), [load_rows(p) for p in a.exclude], a.n, a.seed)
        a.out.write_text("\n".join(ids) + "\n", "utf-8")
        print(f"{len(ids)} dev instance ids -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
