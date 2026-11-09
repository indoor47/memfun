"""metrics_sampler.parse_metrics and ramp_report: hermetic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.swarm import metrics_sampler as ms  # noqa: E402
from evals.swarm import ramp_report as rr  # noqa: E402

PROM = """\
# HELP sglang:num_running_reqs The number of running requests
sglang:num_running_reqs{model_name="q",tp_rank="0"} 37.0
sglang:num_queue_reqs{model_name="q",tp_rank="0"} 4.0
sglang:token_usage{model_name="q",tp_rank="0"} 0.31
sglang:gen_throughput{model_name="q",tp_rank="0"} 2400.5
sglang:cache_hit_rate{model_name="q",tp_rank="0"} 0.98
sglang:prompt_tokens_total{model_name="q"} 1000
sglang:generation_tokens_total{model_name="q"} 500
"""


def test_parse_metrics_reads_the_sampled_fields() -> None:
    m = ms.parse_metrics(PROM)
    assert m["running"] == 37.0 and m["queue"] == 4.0 and m["gen_throughput"] == 2400.5
    assert m["token_usage"] == 0.31 and m["cache_hit_rate"] == 0.98
    assert m["prompt_total"] == 1000 and m["gen_total"] == 500
    assert ms.parse_metrics("garbage\n") == {}


def _csv(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    lines = [
        "ts,running,queue,token_usage,gen_throughput,cache_hit_rate,prompt_total,gen_total,up"
    ]
    for i, (ts, run, queue, kv, gen) in enumerate(rows):
        lines.append(f"{ts},{run},{queue},{kv},{gen},0.9,{i * 100},{i * 50},1")
    path.write_text("\n".join(lines) + "\n")


def test_ramp_report_flags_saturation(tmp_path: Path) -> None:
    rows = []
    t = 0.0
    # step 1: 43 agents, 1000 tok/s, no queue; step 2: 86 agents, throughput barely up, long queue
    for _ in range(10):
        rows.append((t, 43, 0, 0.2, 1000.0))
        t += 10
    for _ in range(10):
        rows.append((t, 64, 30, 0.6, 1050.0))
        t += 10
    samples = tmp_path / "s.csv"
    _csv(samples, rows)
    steps = tmp_path / "steps.json"
    steps.write_text(
        json.dumps(
            [
                {"step": 1, "trials": 1, "agents": 43, "start": 0, "end": 99},
                {"step": 2, "trials": 2, "agents": 86, "start": 100, "end": 199},
            ]
        )
    )
    assert rr.main(["--samples", str(samples), "--steps", str(steps), "--decide"]) == 10
    out = tmp_path / "o.json"
    rr.main(["--samples", str(samples), "--steps", str(steps), "--json", str(out)])
    v = dict((s, w) for s, w in json.loads(out.read_text())["verdicts"])
    assert v[1] == [] and v[2]  # step 1 fine, step 2 saturated


def test_ramp_report_not_saturated_when_throughput_scales(tmp_path: Path) -> None:
    rows = [(float(i * 10), 43, 0, 0.2, 1000.0) for i in range(10)]
    rows += [(float(100 + i * 10), 86, 0, 0.4, 1900.0) for i in range(10)]
    samples = tmp_path / "s.csv"
    _csv(samples, rows)
    steps = tmp_path / "steps.json"
    steps.write_text(
        json.dumps(
            [
                {"step": 1, "agents": 43, "start": 0, "end": 99},
                {"step": 2, "agents": 86, "start": 100, "end": 199},
            ]
        )
    )
    assert rr.main(["--samples", str(samples), "--steps", str(steps), "--decide"]) == 0
