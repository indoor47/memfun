#!/usr/bin/env python3
"""Agents vs throughput: summarise a concurrency ramp and decide where the GPU saturates.

Input: the sampler CSV (``metrics_sampler.py``) and a steps JSON, a list of
``{"step", "trials", "agents", "start", "end", "results"}`` (epoch seconds; ``results`` is the
run's ``results.json``).  Output: a table per step and a verdict.  ``--decide`` exits 10 when the
LAST step is saturated (the ramp driver stops there), else 0.

Saturation (any of): the queue stays long (p90 queue above max(2, 20% of p90 running)); the
aggregate throughput grew under 10% while the agents grew 1.5x or more; the per-stream speed
fell under half of step 1's.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def load_samples(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("up") != "1":
                continue
            rows.append(
                {k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()}
            )
    return rows


def summarize_step(rows: list[dict[str, float]], step: dict[str, Any]) -> dict[str, Any]:
    win = [r for r in rows if step["start"] <= r["ts"] <= step["end"]]
    busy = [r for r in win if (r.get("running") or 0) >= 1]
    run = [r["running"] for r in busy]
    queue = [r["queue"] for r in busy if r["queue"] == r["queue"]]
    gen = [r["gen_throughput"] for r in busy if r["gen_throughput"] == r["gen_throughput"]]
    per = [
        r["gen_throughput"] / r["running"]
        for r in busy
        if r["running"] and r["gen_throughput"] == r["gen_throughput"]
    ]
    kv = [r["token_usage"] for r in win if r["token_usage"] == r["token_usage"]]
    hit = [r["cache_hit_rate"] for r in busy if r["cache_hit_rate"] == r["cache_hit_rate"]]
    wall = max(1.0, step["end"] - step["start"])
    gen_total = None
    if len(win) >= 2 and win[0]["gen_total"] == win[0]["gen_total"]:
        gen_total = win[-1]["gen_total"] - win[0]["gen_total"]
    out: dict[str, Any] = {
        "step": step["step"],
        "agents": step["agents"],
        "wall_s": round(wall),
        "running_p50": round(st.median(run), 1) if run else 0,
        "running_p90": _q(run, 0.9),
        "running_max": max(run) if run else 0,
        "queue_p50": round(st.median(queue), 1) if queue else 0,
        "queue_p90": _q(queue, 0.9),
        "queue_max": max(queue) if queue else 0,
        "gen_tps_p50": round(st.median(gen)) if gen else 0,
        "gen_tps_peak": round(max(gen)) if gen else 0,
        "gen_tps_mean": round(gen_total / wall) if gen_total else None,
        "per_stream_p50": round(st.median(per), 1) if per else 0,
        "kv_max": round(max(kv), 3) if kv else None,
        "cache_hit_p50": round(st.median(hit), 3) if hit else None,
    }
    res = step.get("results")
    if res and Path(res).is_file():
        ov = json.loads(Path(res).read_text()).get("overall", {})
        oc = ov.get("outcomes", {})
        n = sum(oc.values()) or 1
        out["pass"] = oc.get("pass", 0)
        out["episodes"] = n
        out["pass_rate"] = round(oc.get("pass", 0) / n, 3)
        out["harness_errors"] = oc.get("harness_error", 0)
    return out


def saturated(
    cur: dict[str, Any], prev: dict[str, Any] | None, first: dict[str, Any]
) -> list[str]:
    """The reasons *cur* looks saturated (empty = not)."""
    why: list[str] = []
    if cur["queue_p90"] > max(2.0, 0.2 * cur["running_p90"]):
        why.append(f"queue p90 {cur['queue_p90']:.0f} stays long")
    grew = prev and cur["agents"] >= 1.5 * prev["agents"] and prev["gen_tps_peak"]
    if grew and prev and cur["gen_tps_peak"] < 1.10 * prev["gen_tps_peak"]:
        why.append(
            f"aggregate throughput {cur['gen_tps_peak']} tok/s vs {prev['gen_tps_peak']} "
            f"with {cur['agents'] / prev['agents']:.1f}x the agents"
        )
    if first["per_stream_p50"] and cur["per_stream_p50"] < 0.5 * first["per_stream_p50"]:
        why.append(
            f"per-stream speed {cur['per_stream_p50']} tok/s under half of step 1's "
            f"{first['per_stream_p50']}"
        )
    return why


COLS = (
    ("step", "step"),
    ("agents", "agents"),
    ("wall_s", "wall s"),
    ("running_p50", "run p50"),
    ("running_max", "run max"),
    ("queue_p50", "queue p50"),
    ("queue_max", "queue max"),
    ("gen_tps_p50", "tok/s p50"),
    ("gen_tps_peak", "tok/s peak"),
    ("per_stream_p50", "tok/s/stream"),
    ("kv_max", "KV max"),
    ("cache_hit_p50", "cache hit"),
    ("pass", "pass"),
    ("episodes", "of"),
)


def table(summaries: list[dict[str, Any]]) -> str:
    lines = ["  ".join(f"{h:>11}" for _, h in COLS)]
    for s in summaries:
        lines.append("  ".join(f"{s.get(k, '-')!s:>11}" for k, _ in COLS))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument(
        "--decide", action="store_true", help="exit 10 when the last step is saturated"
    )
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = load_samples(a.samples)
    steps = json.loads(a.steps.read_text())
    summaries = [summarize_step(rows, s) for s in steps]
    print(table(summaries))
    verdicts: list[tuple[int, list[str]]] = []
    for i, s in enumerate(summaries):
        verdicts.append((s["step"], saturated(s, summaries[i - 1] if i else None, summaries[0])))
    for step, why in verdicts:
        print(f"step {step}: " + ("SATURATED: " + "; ".join(why) if why else "not saturated"))
    if a.json:
        a.json.write_text(json.dumps({"steps": summaries, "verdicts": verdicts}, indent=1))
    return 10 if a.decide and verdicts and verdicts[-1][1] else 0


if __name__ == "__main__":
    sys.exit(main())
