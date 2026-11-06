"""Play a synthetic swarm into a running memfun dashboard.

Exercises the Swarm / Timeline / Tasks views without any LLM: an
orchestrator request, sub-tasks in parallelism groups, implementer +
auditor pairs per task, a steady stream of ``llm.call`` events, one
rejected task, and workers going offline at the end.  Optionally serves
fake SGLang Prometheus metrics so the server strip can be checked too.

    memfun dashboard --port 8091            # or any running dashboard
    python evals/swarm/dashboard_sim.py --url http://127.0.0.1:8091 \\
        --tasks 10 --seconds 90 --metrics-port 30091
    # then start the dashboard with
    # MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:30091/metrics
"""

from __future__ import annotations

import argparse
import http.server
import json
import random
import threading
import time
import urllib.request

_LOAD = {"running": 0}


def _post(url: str, **event: object) -> None:
    event.setdefault("ts", time.time())
    req = urllib.request.Request(
        url + "/api/event",
        data=json.dumps(event).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=3).read()


def _serve_metrics(port: int) -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            running = _LOAD["running"]
            body = (
                "# TYPE sglang:num_running_reqs gauge\n"
                f'sglang:num_running_reqs{{model_name="qwen"}} {running}\n'
                f'sglang:num_queue_reqs{{model_name="qwen"}} {max(0, running - 12)}\n'
                f'sglang:token_usage{{model_name="qwen"}} {min(0.95, 0.05 + running * 0.03):.3f}\n'
                f'sglang:cache_hit_rate{{model_name="qwen"}} {random.uniform(0.7, 0.92):.3f}\n'
                f'sglang:gen_throughput{{model_name="qwen"}} '
                f"{running * random.uniform(45, 60):.1f}"
                "\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--metrics-port", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    if args.metrics_port:
        _serve_metrics(args.metrics_port)

    url = args.url.rstrip("/")
    wf = f"wf-sim-{int(time.time())}"
    _post(
        url,
        event_type="workflow.started",
        workflow_id=wf,
        detail="Build CRUD service: handlers, storage, tests, docs",
        project="sim",
    )
    roles = ["coder-agent", "test-agent", "file-agent"]
    tasks = []
    for i in range(args.tasks):
        tid = f"t{i:02d}-{wf[-4:]}"
        group = 1 if i < args.tasks * 0.7 else 2  # second wave depends on the first
        impl = (f"w-impl-{i:02d}", roles[i % len(roles)])
        aud = (f"w-aud-{i:02d}", "review-agent")
        tasks.append({"tid": tid, "group": group, "impl": impl, "aud": aud, "i": i})
        _post(
            url,
            event_type="task.published",
            task_id=tid,
            workflow_id=wf,
            agent_name=impl[1],
            group=group,
            detail=f"module_{i:02d}: handler + tests",
        )

    def run_task(t: dict) -> None:
        wid, name = t["impl"]
        _post(url, event_type="worker.online", worker_id=wid, agent_name=name)
        _post(
            url,
            event_type="task.picked_up",
            task_id=t["tid"],
            worker_id=wid,
            agent_name=name,
            workflow_id=wf,
        )
        _LOAD["running"] += 1
        end = time.time() + random.uniform(0.25, 0.45) * args.seconds
        while time.time() < end:
            _post(
                url,
                event_type="llm.call",
                worker_id=wid,
                agent_name=name,
                task_id=t["tid"],
                model="Qwen3.8-27B-FP8",
                role="worker",
                prompt_tokens=random.randint(3000, 12000),
                completion_tokens=random.randint(150, 900),
                duration_ms=random.randint(1500, 6000),
            )
            time.sleep(random.uniform(0.8, 2.5))
        # auditor checks the implementer's work on the same task
        aid, aname = t["aud"]
        _post(url, event_type="worker.online", worker_id=aid, agent_name=aname)
        _post(
            url,
            event_type="task.picked_up",
            task_id=t["tid"] + "-audit",
            worker_id=aid,
            agent_name=aname,
        )
        for _ in range(random.randint(2, 4)):
            _post(
                url,
                event_type="llm.call",
                worker_id=aid,
                agent_name=aname,
                task_id=t["tid"] + "-audit",
                model="claude-opus-5-5",
                role="auditor",
                prompt_tokens=random.randint(4000, 9000),
                completion_tokens=random.randint(100, 500),
                duration_ms=random.randint(2000, 7000),
            )
            time.sleep(random.uniform(1.0, 2.0))
        ok = t["i"] != 3  # one task gets rejected by its auditor
        _post(
            url,
            event_type="task.completed",
            task_id=t["tid"] + "-audit",
            worker_id=aid,
            agent_name=aname,
            success=ok,
            duration_ms=4000,
        )
        _post(
            url,
            event_type="task.completed",
            task_id=t["tid"],
            worker_id=wid,
            agent_name=name,
            workflow_id=wf,
            success=ok,
            duration_ms=int(args.seconds * 400),
        )
        _LOAD["running"] -= 1
        _post(url, event_type="worker.offline", worker_id=aid, agent_name=aname)

    for group in (1, 2):
        wave = [t for t in tasks if t["group"] == group]
        threads = [threading.Thread(target=run_task, args=(t,)) for t in wave]
        for th in threads:
            th.start()
            time.sleep(random.uniform(0.1, 0.6))
        for th in threads:
            th.join()
    for t in tasks:
        _post(url, event_type="worker.offline", worker_id=t["impl"][0], agent_name=t["impl"][1])
    print("simulation done")


if __name__ == "__main__":
    main()
