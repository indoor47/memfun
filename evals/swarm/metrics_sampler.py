#!/usr/bin/env python3
"""Sample a SGLang (or vLLM) Prometheus /metrics endpoint every N seconds into a CSV.

Used by the concurrency ramp (``run_concurrency_ramp.sh``) to see where a GPU saturates.  Stdlib
only; one row per sample; never logs anything but the numbers.  Stop with SIGTERM / Ctrl-C.

Columns: ts (epoch s), running, queue, token_usage (KV fraction 0..1), gen_throughput (aggregate
generated tok/s as SGLang reports it), cache_hit_rate, prompt_total / gen_total (cumulative token
counters), up (1 when the scrape worked).
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

FIELDS = (
    "ts", "running", "queue", "token_usage", "gen_throughput", "cache_hit_rate",
    "prompt_total", "gen_total", "up",
)  # fmt: skip
#: field -> metric name candidates (first found wins; labels are ignored, series summed)
NAMES = {
    "running": ("sglang:num_running_reqs", "vllm:num_requests_running"),
    "queue": ("sglang:num_queue_reqs", "vllm:num_requests_waiting"),
    "token_usage": ("sglang:token_usage", "vllm:gpu_cache_usage_perc"),
    "gen_throughput": ("sglang:gen_throughput",),
    "cache_hit_rate": ("sglang:cache_hit_rate", "vllm:gpu_prefix_cache_hit_rate"),
    "prompt_total": ("sglang:prompt_tokens_total", "vllm:prompt_tokens_total"),
    "gen_total": ("sglang:generation_tokens_total", "vllm:generation_tokens_total"),
}
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|Inf)\s*$")


def parse_metrics(text: str) -> dict[str, float]:
    """The sampled fields of one /metrics scrape (missing metrics are left out)."""
    sums: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        try:
            sums[m.group(1)] = sums.get(m.group(1), 0.0) + float(m.group(2))
        except ValueError:
            continue
    out: dict[str, float] = {}
    for field, names in NAMES.items():
        for n in names:
            if n in sums:
                out[field] = sums[n]
                break
    return out


def scrape(url: str, timeout: float = 5.0) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:18000/metrics")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args(argv)
    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        while not stop["now"]:
            t0 = time.time()
            row = scrape(a.url)
            w.writerow(
                [round(t0, 1)]
                + [("" if row is None or f not in row else row[f]) for f in FIELDS[1:-1]]
                + [0 if row is None else 1]
            )
            fh.flush()
            end = t0 + a.interval
            while time.time() < end and not stop["now"]:
                time.sleep(0.25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
