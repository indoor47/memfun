"""Hermetic tests for the RunPod launch kit (scripts/runpod).  No network, no real pods.

Every transport here is fake.  The dry-run tests use a spy that raises on any mutation, so
a regression that lets dry-run mutate fails loudly instead of renting a GPU.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import io
import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod import runpod_kit as kit  # noqa: E402
from scripts.runpod import verify_weights as vw  # noqa: E402

FAKE_KEY = "rpa_FAKEKEY_0123456789abcdefSECRET"
ENV = {"RUNPOD_API_KEY": FAKE_KEY}
NOW = dt.datetime(2026, 11, 6, 12, 0, tzinfo=dt.UTC)
B300 = kit.DEFAULT_GPU_TYPE
H200 = "NVIDIA H200"

GPU_TYPES: list[dict[str, Any]] = [
    {
        "id": B300,
        "displayName": "B300",
        "memoryInGb": 288,
        "secureCloud": True,
        "communityCloud": True,
        "securePrice": 8.99,
        "communityPrice": 6.94,
    },
    {
        "id": H200,
        "displayName": "H200 SXM",
        "memoryInGb": 141,
        "secureCloud": True,
        "communityCloud": True,
        "securePrice": 5.29,
        "communityPrice": 3.59,
    },
]


class SpyError(AssertionError):
    pass


class FakeRunPod:
    """Simulated RunPod API.  ``allow_mutations=False`` makes it a spy that raises."""

    def __init__(
        self,
        *,
        allow_mutations: bool = True,
        terminate_failures: int = 0,
        never_gone: bool = False,
        pod_errors: int = 0,
        echo_key_in_errors: bool = False,
    ) -> None:
        self.allow_mutations = allow_mutations
        self.terminate_failures = terminate_failures
        self.never_gone = never_gone
        self.pod_errors = pod_errors
        self.echo_key_in_errors = echo_key_in_errors
        self.calls: list[tuple[str, str]] = []
        self.mutations: list[str] = []
        self.deployed: list[dict[str, Any]] = []
        self.pods: dict[str, dict[str, Any]] = {}
        self.volumes: list[dict[str, Any]] = [
            {"id": "vol123", "name": "memfun-weights", "size": 80, "dataCenterId": "EU-RO-1"}
        ]
        self.terminate_calls = 0
        self.dc_stock = {"EU-RO-1": "AVAILABLE", "US-KS-2": "AVAILABLE", "EU-NL-1": "UNAVAILABLE"}
        # Failure-injection hooks.
        self.deploy_mode = "ok"  # | "timeout_after_create" | "fail_no_pod"
        self.cost_in_deploy = True
        self.on_pod_query: Callable[[], None] | None = None
        self.raise_any: Callable[[], BaseException] | None = None
        self.lookup_fails = False
        self.price_override: float | None = None

    def _mutate(self, what: str) -> None:
        if not self.allow_mutations:
            raise SpyError(f"spy transport saw a mutation: {what}")
        self.mutations.append(what)

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append(("graphql", query))
        if self.raise_any is not None:
            raise self.raise_any()
        if "mutation" in query.lower():
            if "podFindAndDeployOnDemand" in query:
                self._mutate("deploy")
                payload = dict((variables or {})["input"])
                self.deployed.append(payload)
                pod_id = f"pod{len(self.deployed)}"
                self.pods[pod_id] = {
                    "id": pod_id,
                    "name": payload["name"],
                    "desiredStatus": "RUNNING",
                    "costPerHr": 9.0,
                    "runtime": {
                        "uptimeInSeconds": 5,
                        "ports": [
                            {
                                "ip": "203.0.113.7",
                                "isIpPublic": True,
                                "privatePort": 22,
                                "publicPort": 40022,
                                "type": "tcp",
                            }
                        ],
                    },
                }
                if self.deploy_mode == "timeout_after_create":
                    raise kit.KitError("RunPod request failed: TimeoutError: timed out")
                if self.deploy_mode == "fail_no_pod":
                    self.pods.pop(pod_id)
                    raise kit.KitError("RunPod HTTP 500: internal error")
                ret = dict(self.pods[pod_id])
                if not self.cost_in_deploy:
                    ret["costPerHr"] = None
                return {"podFindAndDeployOnDemand": ret}
            if "podTerminate" in query:
                self._mutate("terminate")
                self.terminate_calls += 1
                if self.terminate_failures > 0:
                    self.terminate_failures -= 1
                    raise kit.KitError("RunPod HTTP 502: bad gateway")
                pod_id = re.search(r'podId: "([^"]+)"', query)
                assert pod_id
                if not self.never_gone:
                    self.pods.pop(pod_id.group(1), None)
                return {"podTerminate": None}
            raise SpyError(f"unexpected mutation {query}")
        if "gpuTypeDatacenters" in query:
            return {
                "gpuTypes": [
                    {
                        "lowestPrice": {
                            "gpuTypeDatacenters": [
                                {
                                    "dataCenterId": dc,
                                    "supportNetworkVolumes": True,
                                    "availability": avail,
                                }
                                for dc, avail in self.dc_stock.items()
                            ]
                        }
                    }
                ]
            }
        if "gpuTypes(input" in query:
            gid = re.search(r'id: "([^"]+)"', query)
            assert gid
            rows = [g for g in GPU_TYPES if g["id"] == gid.group(1)]
            price = rows[0]["securePrice"] if rows else None
            if self.price_override is not None:
                price = self.price_override
            return {
                "gpuTypes": [
                    {
                        "id": gid.group(1),
                        "lowestPrice": {
                            "stockStatus": "Low",
                            "uninterruptablePrice": price,
                            "availableGpuCounts": [1, 2],
                        },
                    }
                ]
            }
        if "gpuTypes" in query:
            return {"gpuTypes": GPU_TYPES}
        if "myself { pods" in query:
            if self.lookup_fails:
                raise kit.KitError("RunPod HTTP 503: unavailable")
            return {
                "myself": {
                    "pods": [
                        {"id": k, "name": v["name"], "desiredStatus": v.get("desiredStatus")}
                        for k, v in self.pods.items()
                    ]
                }
            }
        if "myself" in query:
            return {"myself": {"clientBalance": 50.0, "networkVolumes": self.volumes}}
        if "pod(input" in query:
            if self.on_pod_query is not None:
                hook, self.on_pod_query = self.on_pod_query, None
                hook()
            if self.pod_errors > 0:
                self.pod_errors -= 1
                msg = "RunPod HTTP 503: unavailable"
                if self.echo_key_in_errors:
                    msg += f" (auth {FAKE_KEY})"
                raise kit.KitError(msg)  # unredacted on purpose: output must still hide it
            pod_id = re.search(r'podId: "([^"]+)"', query)
            assert pod_id
            return {"pod": self.pods.get(pod_id.group(1))}
        raise SpyError(f"unexpected query {query}")

    def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        self.calls.append(("rest", f"{method} {path}"))
        if self.raise_any is not None:
            raise self.raise_any()
        if method.upper() != "GET":
            self._mutate(f"rest {method} {path}")
            if path == "/networkvolumes" and body:
                vol = {"id": "volNEW", "name": body["name"], "size": body["size"]}
                return vol
        raise SpyError(f"unexpected rest {method} {path}")


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(seconds, 0.001)


@pytest.fixture
def pubkey(tmp_path: Path) -> str:
    p = tmp_path / "id_test.pub"
    p.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEPUBLICKEY test@mac\n")
    return str(p)


def run(argv: list[str], fake: FakeRunPod, clock: FakeClock | None = None) -> int:
    c = clock or FakeClock()
    return kit.main(
        argv, factory=lambda _key: fake, env=dict(ENV), clock=c, sleep=c.sleep, now=NOW
    )


# ---------------------------------------------------------------------------
# Dry run is structurally read-only.
# ---------------------------------------------------------------------------


def test_dry_run_is_default_and_never_mutates(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--ssh-public-key", pubkey], spy) == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing was created" in out
    assert "EST. MAX COST: $17.98" in out  # 8.99 $/h x 1 GPU x 120 min
    assert spy.mutations == []
    assert all("mutation" not in q.lower() for kind, q in spy.calls if kind == "graphql")
    assert all(
        not q.startswith(("POST", "PUT", "DELETE")) for kind, q in spy.calls if kind == "rest"
    )


def test_dry_run_with_volume_creation_planned_sends_no_post(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    spy = FakeRunPod(allow_mutations=False)
    argv = ["launch", "--volume-name", "brand-new", "--data-center", "EU-RO-1"]
    assert run([*argv, "--ssh-public-key", pubkey], spy) == 0
    out = capsys.readouterr().out
    assert "CREATE 'brand-new' 80 GB in EU-RO-1" in out
    assert spy.mutations == []


def test_dry_run_with_yes_flag_alone_is_still_dry(pubkey: str) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--yes-spend-money", "--ssh-public-key", pubkey], spy) == 0
    assert spy.mutations == []


def test_readonly_transport_refuses_mutations_before_reaching_inner() -> None:
    inner = FakeRunPod(allow_mutations=True)
    ro = kit.ReadOnlyTransport(inner)
    for q in (
        'mutation { podTerminate(input: {podId: "x"}) }',
        '  MUTATION X { podStop(input: {podId: "x"}) { id } }',
        "query { a } mutation { b }",
        'subscription { pod(input: {podId: "x"}) { id } }',
    ):
        with pytest.raises(kit.MutationRefusedError):
            ro.graphql(q)
    with pytest.raises(kit.MutationRefusedError):
        ro.rest("POST", "/networkvolumes", {"name": "x", "size": 1, "dataCenterId": "y"})
    with pytest.raises(kit.MutationRefusedError):
        ro.rest("DELETE", "/pods/x")
    assert inner.calls == [] and inner.mutations == []
    assert ro.graphql("query { gpuTypes { id } }")["gpuTypes"]


def test_dry_run_plan_only_ever_gets_readonly_transport(pubkey: str) -> None:
    seen: list[str] = []

    def factory(_key: kit.Secret) -> kit.Transport:
        fake = FakeRunPod(allow_mutations=False)
        seen.append(type(fake).__name__)
        return fake

    rc = kit.main(["launch", "--ssh-public-key", pubkey], factory=factory, env=dict(ENV), now=NOW)
    assert rc == 0
    assert len(seen) == 1  # the mutating transport is never even constructed


# ---------------------------------------------------------------------------
# Real path guards (fake transport only).
# ---------------------------------------------------------------------------


def test_no_dry_run_without_confirmation_refuses(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod(allow_mutations=True)
    assert run(["launch", "--no-dry-run", "--ssh-public-key", pubkey], fake) == 2
    assert "needs --yes-spend-money" in capsys.readouterr().err
    assert fake.mutations == []


def test_refuses_when_estimate_exceeds_cost_cap(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod(allow_mutations=True)
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--max-minutes", "600"]
    assert run([*argv, "--max-cost-usd", "40", "--ssh-public-key", pubkey], fake) == 2
    assert "exceeds --max-cost-usd" in capsys.readouterr().err
    assert fake.mutations == []


def test_refuses_without_public_key(tmp_path: Path) -> None:
    fake = FakeRunPod(allow_mutations=True)
    argv = ["launch", "--no-dry-run", "--yes-spend-money"]
    assert run([*argv, "--ssh-public-key", str(tmp_path / "missing.pub")], fake) == 2
    assert fake.mutations == []


def test_full_launch_against_fake_deploys_then_watchdog_terminates(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod(allow_mutations=True)
    clock = FakeClock()
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--max-minutes", "60"]
    assert run([*argv, "--ssh-public-key", pubkey, "--poll-seconds", "300"], fake, clock) == 0
    assert fake.mutations == ["deploy", "terminate"]
    payload = fake.deployed[0]
    assert payload["gpuTypeId"] == B300
    assert payload["networkVolumeId"] == "vol123"
    assert payload["dataCenterIds"] == ["EU-RO-1"]
    assert payload["terminateAfter"] == "2026-11-06T13:00:00Z"
    assert payload["ports"] == "22/tcp"
    env = {e["key"]: e["value"] for e in payload["env"]}
    assert "RUNPOD_API_KEY" not in env
    assert FAKE_KEY not in json.dumps(payload)
    boot = base64.b64decode(env["MEMFUN_BOOT_B64"]).decode()
    assert boot == kit.BOOT_SCRIPT.read_text()
    out = capsys.readouterr().out
    assert "LAUNCHING: up to $8.99" in out
    assert "verified: pod pod1 is gone" in out
    assert fake.pods == {}


def test_launch_creates_volume_only_on_real_path(pubkey: str) -> None:
    fake = FakeRunPod(allow_mutations=True)
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--volume-name", "new"]
    clock = FakeClock()
    assert run([*argv, "--data-center", "US-KS-2", "--ssh-public-key", pubkey], fake, clock) == 0
    assert fake.mutations[0] == "rest POST /networkvolumes"
    assert fake.deployed[0]["networkVolumeId"] == "volNEW"
    assert fake.deployed[0]["dataCenterIds"] == ["US-KS-2"]


def test_refuses_when_gpu_unavailable_in_volume_data_center(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod(allow_mutations=True)
    fake.volumes[0]["dataCenterId"] = "EU-NL-1"
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey]
    assert run(argv, fake) == 2
    assert "not AVAILABLE in EU-NL-1" in capsys.readouterr().err
    assert fake.mutations == []


def test_gpus_where_lists_data_centers(capsys: pytest.CaptureFixture[str]) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["gpus", "--where", B300], spy) == 0
    out = capsys.readouterr().out
    assert "EU-RO-1        AVAILABLE" in out and "EU-NL-1        UNAVAILABLE" in out


def test_creating_volume_requires_data_center(capsys: pytest.CaptureFixture[str]) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--volume-name", "brand-new"], spy) == 1
    assert "needs --data-center" in capsys.readouterr().err


def test_assert_no_secret_blocks_leaking_payload() -> None:
    secret = kit.Secret(FAKE_KEY)
    with pytest.raises(kit.KitError) as exc:
        kit.assert_no_secret({"env": [{"key": "X", "value": FAKE_KEY}]}, secret, "payload")
    assert FAKE_KEY not in str(exc.value)


# ---------------------------------------------------------------------------
# Cost math and GPU type as a parameter.
# ---------------------------------------------------------------------------


def test_cost_estimate_math() -> None:
    assert kit.estimate_cost(8.99, 1, 120) == 17.98
    assert kit.estimate_cost(5.29, 2, 90) == 15.87
    assert kit.estimate_cost(8.99, 1, 1) == 0.15
    gpu = kit.GpuType.from_api(GPU_TYPES[0])
    hi = kit.Offer(stock_status="Low", lowest_price=9.5, available_counts=(1,))
    lo = kit.Offer(stock_status="Low", lowest_price=7.0, available_counts=(1,))
    none = kit.Offer(stock_status="Low", lowest_price=None, available_counts=())
    assert kit.conservative_price(gpu, hi, "secure") == 9.5
    assert kit.conservative_price(gpu, lo, "secure") == 8.99
    assert kit.conservative_price(gpu, none, "community") == 6.94
    free = kit.GpuType("x", "x", 1, None, None, True, True)
    with pytest.raises(kit.KitError):
        kit.conservative_price(free, none, "secure")


def test_gpu_type_is_a_parameter(capsys: pytest.CaptureFixture[str], pubkey: str) -> None:
    assert kit.build_parser().parse_args(["launch"]).gpu_type == B300
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--gpu-type", H200, "--ssh-public-key", pubkey], spy) == 0
    out = capsys.readouterr().out
    assert f'"gpuTypeId": "{H200}"' in out
    assert "EST. MAX COST: $10.58" in out  # 5.29 x 2 h


def test_gpu_type_never_fuzzy_matched(capsys: pytest.CaptureFixture[str]) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--gpu-type", "H200"], spy) == 1
    err = capsys.readouterr().err
    assert "unknown GPU type 'H200'" in err and H200 in err
    types = [kit.GpuType.from_api(g) for g in GPU_TYPES]
    assert kit.resolve_gpu_type(types, "h200 sxm").id == H200  # exact display name


def test_community_cloud_uses_community_price(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--cloud", "community", "--ssh-public-key", pubkey], spy) == 0
    out = capsys.readouterr().out
    assert '"cloudType": "COMMUNITY"' in out
    assert "network volumes may not attach" in out


# ---------------------------------------------------------------------------
# The API key never appears in any output.
# ---------------------------------------------------------------------------


def test_key_never_printed_on_any_path(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, pubkey: str
) -> None:
    clock = FakeClock()
    paths: list[tuple[list[str], FakeRunPod]] = [
        (["gpus"], FakeRunPod(allow_mutations=False)),
        (["launch", "--ssh-public-key", pubkey], FakeRunPod(allow_mutations=False)),
        (["launch", "--no-dry-run", "--ssh-public-key", pubkey], FakeRunPod()),
        (["launch", "--gpu-type", "nope"], FakeRunPod(allow_mutations=False)),
        (
            ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey],
            FakeRunPod(terminate_failures=2, pod_errors=1, echo_key_in_errors=True),
        ),
        (["status", "--pod-id", "nope"], FakeRunPod(allow_mutations=False)),
        (["watchdog", "--pod-id", "nope", "--max-minutes", "1"], FakeRunPod()),
        (["tunnel", "--pod-id", "nope", "--local-port", "8434"], FakeRunPod()),
        (["tunnel", "--pod-id", "nope", "--local-port", "18555"], FakeRunPod()),
    ]
    for argv, fake in paths:
        run(argv, fake, clock)
    captured = capsys.readouterr()
    for blob in (captured.out, captured.err):
        assert FAKE_KEY not in blob
        assert base64.b64encode(FAKE_KEY.encode()).decode() not in blob
    secret = kit.Secret(FAKE_KEY)
    assert FAKE_KEY not in repr(secret) + str(secret) + f"{secret}"


def test_key_loaded_from_file_and_redacted_in_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(FAKE_KEY + "\n")
    key_file.chmod(0o600)
    secret = kit.load_api_key(env={}, path=key_file)
    assert secret.reveal() == FAKE_KEY

    def boom(req: Any, timeout: float = 0) -> Any:
        assert req.get_header("Authorization") == f"Bearer {FAKE_KEY}"
        assert FAKE_KEY not in req.full_url
        body = io.BytesIO(f'{{"error": "bad key {FAKE_KEY}"}}'.encode())
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", Message(), body)

    monkeypatch.setattr(kit, "_auth_open", boom)
    transport = kit.HttpTransport(secret)
    with pytest.raises(kit.KitError) as exc:
        transport.graphql("query { gpuTypes { id } }")
    assert FAKE_KEY not in str(exc.value) and "***" in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__

    rc = kit.main(["gpus"], env={}, key_path=key_file)  # real HttpTransport, patched opener
    assert rc == 1
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err
    assert "HTTP 401" in captured.err


def test_missing_key_message_has_no_secret(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = kit.main(["gpus"], env={}, key_path=tmp_path / "absent")
    assert rc == 1
    assert "no RunPod API key" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Watchdog (fake clock, fake transport).
# ---------------------------------------------------------------------------


def _fake_with_pod() -> FakeRunPod:
    fake = FakeRunPod()
    fake.graphql(kit.DEPLOY_MUTATION, {"input": {"name": "w"}})
    fake.mutations.clear()
    return fake


def test_watchdog_terminates_at_deadline(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    clock = FakeClock()
    deadline = clock() + 3600
    res = kit.run_watchdog(
        fake, "pod1", deadline, clock=clock, sleep=clock.sleep, poll_seconds=600
    )
    assert res.reason == "deadline" and res.terminated and res.verified_gone
    assert clock() >= deadline
    assert fake.mutations == ["terminate"]
    assert abs(res.spent_usd - 9.0) < 0.01  # 9 $/h for 1 h, from costPerHr
    assert "spent $7.50" in capsys.readouterr().out  # at the 50-minute poll


def test_watchdog_retries_termination_and_verifies() -> None:
    fake = _fake_with_pod()
    fake.terminate_failures = 3
    clock = FakeClock()
    res = kit.run_watchdog(fake, "pod1", clock() + 60, clock=clock, sleep=clock.sleep, retries=6)
    assert res.verified_gone
    assert fake.terminate_calls == 4


def test_watchdog_reports_unverified_termination(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.never_gone = True
    clock = FakeClock()
    res = kit.run_watchdog(fake, "pod1", clock() + 60, clock=clock, sleep=clock.sleep, retries=3)
    assert res.terminated and not res.verified_gone
    assert fake.terminate_calls == 3
    assert "COULD NOT VERIFY POD pod1" in capsys.readouterr().err


def test_watchdog_terminates_on_ctrl_c() -> None:
    fake = _fake_with_pod()
    clock = FakeClock()
    calls = {"n": 0}

    def sleep(seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    res = kit.run_watchdog(fake, "pod1", clock() + 3600, clock=clock, sleep=sleep)
    assert res.reason == "interrupt" and res.verified_gone
    assert fake.mutations == ["terminate"]


def test_watchdog_terminates_after_repeated_poll_errors() -> None:
    fake = _fake_with_pod()
    fake.pod_errors = 5
    clock = FakeClock()
    res = kit.run_watchdog(
        fake, "pod1", clock() + 3600, clock=clock, sleep=clock.sleep, max_poll_errors=5
    )
    assert res.reason.startswith("error:")
    assert res.verified_gone and fake.mutations == ["terminate"]


def test_watchdog_tolerates_transient_poll_error() -> None:
    fake = _fake_with_pod()
    fake.pod_errors = 2
    clock = FakeClock()
    res = kit.run_watchdog(
        fake, "pod1", clock() + 600, clock=clock, sleep=clock.sleep, poll_seconds=60
    )
    assert res.reason == "deadline" and res.verified_gone


def test_watchdog_on_unexpected_exception_still_terminates() -> None:
    fake = _fake_with_pod()
    clock = FakeClock()

    def sleep(_seconds: float) -> None:
        raise RuntimeError("disk full")

    res = kit.run_watchdog(fake, "pod1", clock() + 3600, clock=clock, sleep=sleep)
    assert res.reason == "error: disk full"
    assert fake.mutations == ["terminate"]


# ---------------------------------------------------------------------------
# Tunnel.
# ---------------------------------------------------------------------------


def test_tunnel_hard_refuses_8434(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(kit.KitError):
        kit.check_local_port(8434)
    with pytest.raises(kit.KitError):
        kit.build_ssh_command("h", 22, 8434, 30000, None)
    fake = FakeRunPod(allow_mutations=False)
    assert run(["tunnel", "--pod-id", "pod1", "--local-port", "8434"], fake) == 1
    assert "refusing local port 8434" in capsys.readouterr().err
    assert fake.calls == []  # refused before any API call


def test_tunnel_refuses_port_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        with pytest.raises(kit.KitError, match="already in use"):
            kit.check_local_port(port)


def test_tunnel_prints_command_and_snippet_without_starting(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_call(*_a: Any, **_k: Any) -> int:
        raise AssertionError("ssh must not start without --run")

    monkeypatch.setattr(kit.subprocess, "call", no_call)
    fake = _fake_with_pod()
    fake.allow_mutations = False
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert run(["tunnel", "--pod-id", "pod1", "--local-port", str(free)], fake) == 0
    out = capsys.readouterr().out
    assert f"-L 127.0.0.1:{free}:127.0.0.1:30000" in out
    assert "-p 40022" in out and "root@203.0.113.7" in out
    assert "\n[llm]\n" in out
    assert f'base_url = "http://127.0.0.1:{free}/v1"' in out
    assert f'model = "{kit.DEFAULT_SERVED_NAME}"' in out


def test_ssh_endpoint_needs_public_tcp_22() -> None:
    pod = kit.PodInfo(
        "p",
        "n",
        "RUNNING",
        1.0,
        1,
        (
            {
                "ip": "10.0.0.1",
                "isIpPublic": False,
                "privatePort": 22,
                "publicPort": 1,
                "type": "tcp",
            },
        ),
    )
    assert pod.ssh_endpoint() is None


# ---------------------------------------------------------------------------
# In-pod boot script and weight verification.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_boot_script_self_terminates_with_pod_scoped_runpodctl(tmp_path: Path) -> None:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    record = tmp_path / "runpodctl.log"
    rp = fakebin / "runpodctl"
    rp.write_text(f'#!/bin/sh\necho "$@ key=${{RUNPOD_API_KEY:-}}" >> "{record}"\nexit 0\n')
    rp.chmod(0o755)
    env = {
        "PATH": f"{fakebin}:/usr/bin:/bin",
        "MEMFUN_MAX_MINUTES": "1",
        "MEMFUN_DEADLINE_EPOCH": "0",  # already past: arm fires immediately
        "MEMFUN_MODEL_DIR": str(tmp_path / "models" / "no-weights"),
        "MEMFUN_MODELS_ROOT": str(tmp_path / "models"),
        "MEMFUN_MODEL_REPO": "Qwen/Qwen3.8-27B-FP8",
        "MEMFUN_RUN_DIR": str(tmp_path / "run"),
        "RUNPOD_POD_ID": "podXYZ",
        "RUNPOD_API_KEY": "pod-scoped-injected",
        "PUBLIC_KEY": "",
        "MEMFUN_FAIL_GRACE_MIN": "0",
    }
    proc = subprocess.run(
        ["bash", str(kit.BOOT_SCRIPT)], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = record.read_text().splitlines()
    assert lines and all(line == "remove pod podXYZ key=pod-scoped-injected" for line in lines)
    assert "hard stop armed: 0s" in proc.stdout


def test_boot_script_never_references_account_key_file() -> None:
    text = kit.BOOT_SCRIPT.read_text()
    assert ".config/runpod" not in text
    assert "sglang.launch_server" in text and "--host 127.0.0.1" in text


def test_verify_weights_checks_sha256_and_blob_sha1(tmp_path: Path) -> None:
    local = tmp_path / "m"
    local.mkdir()
    big = local / "model.safetensors"
    big.write_bytes(b"\x00weights" * 1000)
    cfg = local / "config.json"
    cfg.write_text('{"a": 1}')
    blob = hashlib.sha1(f"blob {cfg.stat().st_size}\0".encode() + cfg.read_bytes()).hexdigest()
    meta: dict[str, Any] = {
        "sha": "abc",
        "siblings": [
            {
                "rfilename": "model.safetensors",
                "size": big.stat().st_size,
                "lfs": {
                    "sha256": hashlib.sha256(big.read_bytes()).hexdigest(),
                    "size": big.stat().st_size,
                },
            },
            {"rfilename": "config.json", "size": cfg.stat().st_size, "blobId": blob},
        ],
    }
    assert vw.verify(meta, local) == []
    meta_file = tmp_path / "meta.json"
    meta_file.write_text(json.dumps(meta))
    assert vw.main(["r/x", str(local), "--offline-meta", str(meta_file)]) == 0
    assert (local / ".memfun_verified").is_file()

    big.write_bytes(b"\x01weights" * 1000)  # same size, different bytes
    assert vw.verify(meta, local) == ["sha256 mismatch: model.safetensors"]
    assert vw.main(["r/x", str(local), "--offline-meta", str(meta_file)]) == 1
    assert not (local / ".memfun_verified").exists()
    cfg.unlink()
    assert "missing: config.json" in vw.verify(meta, local)


def test_private_key_refused_as_public_key(tmp_path: Path) -> None:
    p = tmp_path / "id"
    p.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nxxx\n")
    with pytest.raises(kit.KitError, match="PRIVATE"):
        kit.read_public_key(str(p))


# ---------------------------------------------------------------------------
# Dashboard wiring: metrics flag, run card, spend, and the watch wrapper.
# ---------------------------------------------------------------------------


def test_sglang_launch_command_enables_metrics() -> None:
    text = kit.BOOT_SCRIPT.read_text()
    launch = text[text.index("python3 -m sglang.launch_server") :].split("&\n", 1)[0]
    assert "--enable-metrics" in launch
    assert "--host 127.0.0.1" in launch


def test_dry_run_prints_run_card(capsys: pytest.CaptureFixture[str], pubkey: str) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--ssh-public-key", pubkey, "--max-minutes", "90"], spy) == 0
    out = capsys.readouterr().out
    local = (NOW + dt.timedelta(minutes=90)).astimezone()
    for expected in (
        "=== DRY RUN: run card (nothing created) ===",
        "pod id     : <none: dry run>",
        f"GPU        : 1 x {B300}",
        "rate       : $8.99/h (plan estimate from RunPod prices)",
        f"hard stop  : {local:%Y-%m-%d %H:%M:%S %Z} local (in 90 min, auto-terminate)",
        "live spend : uv run python scripts/runpod/launch_pod.py spend --pod-id <POD_ID>",
        "tunnel     : uv run python scripts/runpod/launch_pod.py tunnel --pod-id <POD_ID> --run",
        "watch      : uv run python scripts/runpod/launch_pod.py watch --pod-id <POD_ID>",
        "export MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:18000/metrics",
        "export MEMFUN_POD_RATE_USD=8.99",
        "dashboard: watch in the Swarm app",
    ):
        assert expected in out, expected
    assert FAKE_KEY not in out
    assert spy.mutations == []


def test_real_launch_card_uses_rate_reported_for_the_pod(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod(allow_mutations=True)
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey]
    assert run(argv, fake) == 0
    out = capsys.readouterr().out
    assert "=== run card ===" in out and "pod id     : pod1" in out
    assert "rate       : $9.00/h (reported by RunPod for this pod)" in out
    assert "export MEMFUN_POD_RATE_USD=9\n" in out
    assert "spend --pod-id pod1" in out and "watch --pod-id pod1" in out
    assert "dashboard: watch in the Swarm app" in out
    assert FAKE_KEY not in out


def test_spend_is_read_only_and_uses_api_rate(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.pods["pod1"]["runtime"]["uptimeInSeconds"] = 1800
    fake.allow_mutations = False
    assert run(["spend", "--pod-id", "pod1"], fake) == 0
    assert "$9.00/h x 30m00s uptime = ~$4.50" in capsys.readouterr().out
    assert fake.mutations == []


class FakeProc:
    def __init__(self, argv: list[str], pid: int, rc: int = 0) -> None:
        self.argv = argv
        self.pid = pid
        self.rc = rc
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self.returncode = self.rc
        return self.rc

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15


class FakeSystem:
    def __init__(self, caffeinate: str | None, ready_after: int = 1) -> None:
        self.caffeinate = caffeinate
        self.ready_after = ready_after
        self.procs: list[FakeProc] = []
        self.kwargs: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.sleeps: list[float] = []

    def popen(self, argv: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeProc(list(argv), pid=4242 + len(self.procs))
        self.procs.append(proc)
        self.kwargs.append(kwargs)
        return proc

    def which(self, name: str) -> str | None:
        return self.caffeinate if name == "caffeinate" else None

    def http_get(self, url: str, timeout: float) -> tuple[int, bytes]:
        self.gets.append(url)
        attempts = sum(1 for g in self.gets if g.endswith("/metrics"))
        if attempts < self.ready_after:  # fail the first ready_after - 1 rounds
            raise kit.KitError("connection refused")
        if url.endswith("/metrics"):
            return 200, b"sglang:num_running_reqs 0\n"
        return 200, json.dumps({"data": [{"id": kit.DEFAULT_SERVED_NAME}]}).encode()

    def deps(self) -> kit.WatchDeps:
        return kit.WatchDeps(
            popen=self.popen,
            which=self.which,
            http_get=self.http_get,
            sleep=self.sleeps.append,
            environ={"PATH": "/usr/bin", "RUNPOD_API_KEY": FAKE_KEY, "HOME": "/h"},
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _watch(argv: list[str], fake: FakeRunPod, system: FakeSystem) -> int:
    return kit.main(
        argv, factory=lambda _k: fake, env=dict(ENV), now=NOW, watch_deps=system.deps()
    )


def test_watch_tunnel_ready_memfun_caffeinate(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.allow_mutations = False
    system = FakeSystem(caffeinate="/usr/bin/caffeinate", ready_after=3)
    port = _free_port()
    rc = _watch(["watch", "--pod-id", "pod1", "--local-port", str(port)], fake, system)
    assert rc == 0
    tunnel, memfun, caff = system.procs
    assert tunnel.argv[:2] == ["ssh", "-N"]
    assert f"127.0.0.1:{port}:127.0.0.1:30000" in tunnel.argv
    assert system.kwargs[0]["start_new_session"] is True
    assert "RUNPOD_API_KEY" not in system.kwargs[0]["env"]
    assert memfun.argv == ["memfun", "chat"]
    env = system.kwargs[1]["env"]
    assert env["MEMFUN_INFERENCE_METRICS_URL"] == f"http://127.0.0.1:{port}/metrics"
    assert env["MEMFUN_POD_RATE_USD"] == "9"
    assert "RUNPOD_API_KEY" not in env
    assert caff.argv == ["/usr/bin/caffeinate", "-dimsu", "-w", str(memfun.pid)]
    assert system.kwargs[2] == {"start_new_session": True}
    assert len(system.sleeps) == 2  # two failed readiness rounds, then ready
    assert system.gets[-2:] == [
        f"http://127.0.0.1:{port}/metrics",
        f"http://127.0.0.1:{port}/v1/models",
    ]
    assert tunnel.terminated  # tunnel closed when memfun exits
    out = capsys.readouterr().out
    assert f"export MEMFUN_INFERENCE_METRICS_URL=http://127.0.0.1:{port}/metrics" in out
    assert "export MEMFUN_POD_RATE_USD=9" in out
    assert "dashboard: watch in the Swarm app" in out
    assert FAKE_KEY not in out
    assert fake.mutations == []


def test_watch_without_caffeinate_and_custom_command(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate=None)
    port = _free_port()
    argv = ["watch", "--pod-id", "pod1", "--local-port", str(port), "--", "memfun", "swarm", "x"]
    assert _watch(argv, fake, system) == 0
    assert [p.argv[0] for p in system.procs] == ["ssh", "memfun"]
    assert system.procs[1].argv == ["memfun", "swarm", "x"]
    assert "caffeinate not available" in capsys.readouterr().out


def test_watch_refuses_8434_before_any_call(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeRunPod(allow_mutations=False)
    system = FakeSystem(caffeinate="/usr/bin/caffeinate")
    assert _watch(["watch", "--pod-id", "pod1", "--local-port", "8434"], fake, system) == 1
    assert fake.calls == [] and system.procs == []
    assert "refusing local port 8434" in capsys.readouterr().err


def test_watch_gives_up_when_not_ready_and_closes_tunnel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate="/usr/bin/caffeinate", ready_after=100)
    argv = ["watch", "--pod-id", "pod1", "--local-port", str(_free_port())]
    assert _watch([*argv, "--ready-retries", "4"], fake, system) == 1
    assert len(system.procs) == 1 and system.procs[0].terminated  # only the tunnel
    assert len(system.sleeps) == 4
    assert "SGLang not ready" in capsys.readouterr().err


def test_watch_env_drops_key_and_never_hardcodes_rate() -> None:
    env = kit.build_watch_env(
        {"RUNPOD_API_KEY": FAKE_KEY, "MEMFUN_POD_RATE_USD": "1"}, 18001, None
    )
    assert "RUNPOD_API_KEY" not in env and "MEMFUN_POD_RATE_USD" not in env
    assert env["MEMFUN_INFERENCE_METRICS_URL"] == "http://127.0.0.1:18001/metrics"


# ===========================================================================
# Audit round 1: key handling (critical), watchdog coverage and cost guards (high),
# advisories.
# ===========================================================================

BAD_KEYS = [
    "rpa_GOODPART123\n# my runpod key",
    "rpa_GOODPART123\r\nsecondline",
    "rpa_GOOD PART123",
    "rpa_GOODPART123\x00tail_after_nul",
    "rpa_GOODPART123;rm",
]


def _fragments(value: str) -> list[str]:
    pieces = [p for p in re.split(r"[\s\x00;]+", value) if len(p) >= 6]
    return [*pieces, value.strip()[:10]]


def _assert_clean(text: str, secret: str) -> None:
    assert "Traceback" not in text
    for frag in _fragments(secret):
        assert frag not in text, frag


@pytest.mark.parametrize("bad", BAD_KEYS)
def test_malformed_key_in_env_rejected_without_echo(
    bad: str, capsys: pytest.CaptureFixture[str]
) -> None:
    def factory(_k: kit.Secret) -> kit.Transport:
        raise AssertionError("no transport may be built for a malformed key")

    rc = kit.main(["gpus"], factory=factory, env={"RUNPOD_API_KEY": bad})
    assert rc == 1
    captured = capsys.readouterr()
    assert "malformed" in captured.err and "not shown" in captured.err
    _assert_clean(captured.out + captured.err, bad)


@pytest.mark.parametrize("bad", BAD_KEYS)
def test_malformed_key_in_file_rejected_without_echo(
    bad: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(bad + "\n")
    key_file.chmod(0o600)
    with pytest.raises(kit.KitError) as exc:
        kit.load_api_key(env={}, path=key_file)
    _assert_clean(str(exc.value), bad)
    assert kit.main(["spend", "--pod-id", "p"], env={}, key_path=key_file) == 1
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err, bad)


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: ValueError(f"Invalid header value b'Bearer {FAKE_KEY}\\n'"),
        lambda: urllib.error.URLError(f"bad auth {FAKE_KEY}"),
        lambda: OSError(f"socket for Bearer {FAKE_KEY}"),
        lambda: UnicodeError(f"cannot encode {FAKE_KEY}"),
        lambda: RuntimeError(f"Authorization: Bearer {FAKE_KEY}"),
    ],
)
def test_http_transport_wraps_every_exception(
    make_exc: Callable[[], BaseException],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(req: Any, timeout: float = 0) -> Any:
        raise make_exc()

    monkeypatch.setattr(kit, "_auth_open", boom)
    for argv in (["gpus"], ["spend", "--pod-id", "x"], ["status", "--pod-id", "x"]):
        assert kit.main(argv, env=dict(ENV)) == 1
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err, FAKE_KEY)
    assert "error:" in captured.err


def test_http_error_body_redacted_before_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = kit.load_api_key(env=dict(ENV))
    for pad in (380, 395, 401, 490):

        def boom(req: Any, timeout: float = 0, pad: int = pad) -> Any:
            body = io.BytesIO(("x" * pad + FAKE_KEY + " tail").encode())
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", Message(), body)

        monkeypatch.setattr(kit, "_auth_open", boom)
        with pytest.raises(kit.KitError) as exc:
            kit.HttpTransport(secret).graphql("query { gpuTypes { id } }")
        _assert_clean(str(exc.value), FAKE_KEY)
        assert "rpa_FA" not in str(exc.value)


def test_redactor_hides_prefixes_and_inner_runs() -> None:
    kit.load_api_key(env=dict(ENV))
    for leaked in (FAKE_KEY[:6], FAKE_KEY[:11], FAKE_KEY[4:20], FAKE_KEY[-9:]):
        out = kit.REDACT(f"before {leaked} after")
        assert leaked not in out and "***" in out
    assert kit.REDACT("rpa_ short") == "rpa_ short"  # 4 chars: below the prefix threshold


def _all_subcommands(pubkey: str, port: int) -> list[list[str]]:
    return [
        ["gpus"],
        ["gpus", "--where", B300],
        ["launch", "--ssh-public-key", pubkey],
        ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey],
        ["status", "--pod-id", "pod1"],
        ["spend", "--pod-id", "pod1"],
        ["watchdog", "--pod-id", "pod1", "--max-minutes", "1"],
        ["tunnel", "--pod-id", "pod1", "--local-port", str(port)],
        ["watch", "--pod-id", "pod1", "--local-port", str(port)],
    ]


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: kit.KitError(f"RunPod HTTP 401: echo Authorization: Bearer {FAKE_KEY}"),
        lambda: RuntimeError(f"boom Authorization: Bearer {FAKE_KEY}"),
        lambda: ValueError(f"Invalid header value b'Bearer {FAKE_KEY}'"),
        lambda: KeyError(FAKE_KEY),
    ],
)
def test_no_subcommand_leaks_key_on_any_exception(
    make_exc: Callable[[], BaseException], capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    port = _free_port()
    for argv in _all_subcommands(pubkey, port):
        fake = FakeRunPod()
        fake.raise_any = make_exc
        system = FakeSystem(caffeinate=None)
        clock = FakeClock()
        rc = kit.main(
            argv,
            factory=lambda _k, f=fake: f,
            env=dict(ENV),
            clock=clock,
            sleep=clock.sleep,
            now=NOW,
            watch_deps=system.deps(),
        )
        assert rc != 0, argv
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err, FAKE_KEY)


def test_watch_with_api_errors_echoing_auth_header(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.pod_errors = 1
    fake.echo_key_in_errors = True
    system = FakeSystem(caffeinate=None)
    assert (
        _watch(["watch", "--pod-id", "pod1", "--local-port", str(_free_port())], fake, system) == 1
    )
    captured = capsys.readouterr()
    _assert_clean(captured.out + captured.err, FAKE_KEY)
    assert "RunPod HTTP 503" in captured.err


# ---------------------------------------------------------------- HIGH: watchdog coverage


def _real_launch(fake: FakeRunPod, pubkey: str, *extra: str) -> int:
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey, *extra]
    return run(argv, fake)


def test_get_pod_error_right_after_deploy_still_terminates(pubkey: str) -> None:
    fake = FakeRunPod()
    fake.cost_in_deploy = False

    def fail() -> None:
        raise kit.KitError("RunPod HTTP 502: bad gateway")

    fake.on_pod_query = fail
    assert _real_launch(fake, pubkey) == 1
    assert fake.mutations == ["deploy", "terminate"]
    assert fake.pods == {}


def test_ctrl_c_right_after_deploy_terminates(pubkey: str) -> None:
    fake = FakeRunPod()
    fake.cost_in_deploy = False

    def interrupt() -> None:
        raise KeyboardInterrupt

    fake.on_pod_query = interrupt
    assert _real_launch(fake, pubkey) == 130
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}


def test_sigterm_right_after_deploy_terminates(pubkey: str) -> None:
    fake = FakeRunPod()
    fake.cost_in_deploy = False
    old_term = signal.getsignal(signal.SIGTERM)
    old_hup = signal.getsignal(signal.SIGHUP)
    kit.install_exit_signals()
    try:
        fake.on_pod_query = lambda: signal.raise_signal(signal.SIGTERM)
        assert _real_launch(fake, pubkey) == 130
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGHUP, old_hup)
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}


def test_deploy_timeout_after_create_finds_pod_by_name_and_terminates(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod()
    fake.deploy_mode = "timeout_after_create"
    assert _real_launch(fake, pubkey) == 4
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}
    name = fake.deployed[0]["name"]
    assert re.fullmatch(r"memfun-swarm-[0-9a-f]{10}", name)
    assert "RunPod did create pod pod1" in capsys.readouterr().err


def test_deploy_failure_with_unknown_state_warns_loudly_incl_volume(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod()
    fake.deploy_mode = "fail_no_pod"
    fake.lookup_fails = True
    rc = _real_launch(fake, pubkey, "--volume-name", "fresh", "--data-center", "US-KS-2")
    assert rc == 4
    assert fake.mutations == ["rest POST /networkvolumes", "deploy"]  # nothing to terminate
    err = capsys.readouterr().err
    name = fake.deployed[0]["name"]
    assert "COULD NOT CONFIRM WHETHER POD" in err and name in err
    assert "runpodctl pod delete" in err
    assert "network volume volNEW" in err and "keeps billing storage" in err


def test_deploy_failure_without_pod_does_not_terminate(pubkey: str) -> None:
    fake = FakeRunPod()
    fake.deploy_mode = "fail_no_pod"
    assert _real_launch(fake, pubkey) == 4
    assert fake.mutations == ["deploy"]


# ---------------------------------------------------------------- HIGH: cost guards


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-cost-usd", "nan"],
        ["--max-cost-usd", "inf"],
        ["--max-cost-usd", "-1"],
        ["--max-cost-usd", "0"],
        ["--max-cost-usd", "abc"],
        ["--gpu-count", "0"],
        ["--gpu-count", "-3"],
        ["--gpu-count", "1.5"],
        ["--gpu-count", "abc"],
        ["--gpu-count", "999"],
        ["--max-minutes", "0"],
        ["--max-minutes", "-5"],
        ["--max-minutes", "nan"],
        ["--max-minutes", "inf"],
        ["--max-minutes", "abc"],
        ["--max-minutes", "100000"],
    ],
)
def test_bad_numeric_inputs_refused_before_any_transport(extra: list[str], pubkey: str) -> None:
    def factory(_k: kit.Secret) -> kit.Transport:
        raise AssertionError("transport built despite invalid input")

    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey, *extra]
    with pytest.raises(SystemExit) as exc:
        kit.main(argv, factory=factory, env=dict(ENV), now=NOW)
    assert exc.value.code == 2


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_api_price_refuses(
    price: float, capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPod()
    fake.price_override = price
    assert _real_launch(fake, pubkey) == 1
    assert fake.mutations == []
    assert "invalid secure price" in capsys.readouterr().err


def test_validate_launch_args_rejects_programmatic_bad_values(pubkey: str) -> None:
    good = kit.build_parser().parse_args(["launch", "--ssh-public-key", pubkey])
    kit.validate_launch_args(good)
    for field_name, value in (
        ("gpu_count", 0),
        ("gpu_count", True and 99),
        ("max_minutes", 0),
        ("max_cost_usd", float("nan")),
        ("max_cost_usd", 0.0),
    ):
        bad = argparse.Namespace(**vars(good))
        setattr(bad, field_name, value)
        with pytest.raises(kit.KitError):
            kit.validate_launch_args(bad)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--model-repo", "../../etc"),
        ("--model-repo", "/etc/passwd"),
        ("--model-repo", "a/b/c"),
        ("--model-repo", "Qwen/.."),
        ("--model-repo", "Qwen/x y"),
        ("--served-name", "../x"),
        ("--served-name", "name;rm"),
        ("--model-revision", "../main"),
    ],
)
def test_bad_model_names_refused(
    flag: str, value: str, capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", flag, value, "--ssh-public-key", pubkey], spy) == 1
    assert spy.calls == []
    assert "must" in capsys.readouterr().err


# ---------------------------------------------------------------- advisories


def test_ssh_child_env_has_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate=None)
    deps = system.deps()
    deps.environ = {
        "PATH": "/usr/bin",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "RUNPOD_API_KEY": FAKE_KEY,
        "ANTHROPIC_API_KEY": "sk-ant-xyz",
        "GITHUB_TOKEN": "ghp_x",
        "RUNPOD_OTHER": "x",
    }
    port = _free_port()
    rc = kit.main(
        ["watch", "--pod-id", "pod1", "--local-port", str(port)],
        factory=lambda _k: fake,
        env=dict(ENV),
        watch_deps=deps,
    )
    assert rc == 0
    ssh_env = system.kwargs[0]["env"]
    assert ssh_env == {"PATH": "/usr/bin", "SSH_AUTH_SOCK": "/tmp/agent.sock"}
    memfun_env = system.kwargs[1]["env"]
    assert "RUNPOD_API_KEY" not in memfun_env and memfun_env["ANTHROPIC_API_KEY"] == "sk-ant-xyz"

    seen: dict[str, Any] = {}

    def fake_call(cmd: list[str], env: dict[str, str] | None = None) -> int:
        seen["cmd"], seen["env"] = cmd, env
        return 0

    monkeypatch.setattr(kit.subprocess, "call", fake_call)
    monkeypatch.setenv("RUNPOD_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SOME_SECRET", "s")
    assert run(["tunnel", "--pod-id", "pod1", "--local-port", str(port), "--run"], fake) == 0
    assert "RUNPOD_API_KEY" not in seen["env"] and "SOME_SECRET" not in seen["env"]


def test_ssh_uses_per_pod_known_hosts_not_home(capsys: pytest.CaptureFixture[str]) -> None:
    cmd = kit.build_ssh_command("h", 22, 18000, 30000, None, known_hosts=Path("/tmp/kh"))
    assert "UserKnownHostsFile=/tmp/kh" in cmd and "GlobalKnownHostsFile=/dev/null" in cmd
    assert not any(".ssh/known_hosts" in c for c in cmd)
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate=None)
    assert (
        _watch(["watch", "--pod-id", "pod1", "--local-port", str(_free_port())], fake, system) == 0
    )
    kh = next(a for a in system.procs[0].argv if a.startswith("UserKnownHostsFile="))
    run_dir = Path(kh.split("=", 1)[1]).parent
    assert run_dir.name.startswith("memfun-runpod-")
    assert not run_dir.exists()  # removed when watch exits


def test_watch_strips_only_first_separator() -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate=None)
    argv = ["watch", "--pod-id", "pod1", "--local-port", str(_free_port())]
    assert _watch([*argv, "--", "memfun", "swarm", "--", "x"], fake, system) == 0
    assert system.procs[1].argv == ["memfun", "swarm", "--", "x"]


def test_watch_sigterm_stops_children_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate="/usr/bin/caffeinate")

    def popen(argv: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeSystem.popen(system, argv, **kwargs)
        if argv[0] == "memfun":

            def wait() -> int:
                raise kit.TerminateRequestedError("signal 15")

            proc.wait = wait  # type: ignore[method-assign]
        return proc

    deps = system.deps()
    deps.popen = popen
    rc = kit.main(
        ["watch", "--pod-id", "pod1", "--local-port", str(_free_port())],
        factory=lambda _k: fake,
        env=dict(ENV),
        watch_deps=deps,
    )
    assert rc == 143
    tunnel, memfun, caff = system.procs
    assert memfun.terminated and tunnel.terminated and caff.terminated
    assert "POD KEEPS RUNNING" in capsys.readouterr().err
    assert fake.mutations == []


def test_watch_plain_ctrl_c_keeps_waiting_for_memfun() -> None:
    fake = _fake_with_pod()
    system = FakeSystem(caffeinate=None)
    calls = {"n": 0}

    def popen(argv: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeSystem.popen(system, argv, **kwargs)
        if argv[0] == "memfun":

            def wait() -> int:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise KeyboardInterrupt
                return 0

            proc.wait = wait  # type: ignore[method-assign]
        return proc

    deps = system.deps()
    deps.popen = popen
    rc = kit.main(
        ["watch", "--pod-id", "pod1", "--local-port", str(_free_port())],
        factory=lambda _k: fake,
        env=dict(ENV),
        watch_deps=deps,
    )
    assert rc == 0 and calls["n"] == 2


FAKE_PY = r"""#!/bin/sh
# Fake python3: records SGLang argv; simulates huggingface_hub / pip per FAKE_* env.
D="@DIR@"
if [ "$1" = "-m" ] && [ "$2" = "sglang.launch_server" ]; then
  for a in "$@"; do printf "%s\n" "$a"; done >> "$D/python3.log"
  [ -n "${FAKE_SGLANG_OUT:-}" ] && echo "$FAKE_SGLANG_OUT"
  [ -n "${FAKE_SGLANG_SLEEP:-}" ] && sleep "$FAKE_SGLANG_SLEEP"
  exit 0
fi
case "$1" in
  --version)
    [ "${FAKE_PY_VERSION_HANG:-0}" = 1 ] && sleep 600
    echo "Python 3.12.99-fake"; exit 0 ;;
  -m)
    if [ "$2" = pip ]; then
      echo "$*" >> "$D/pip.log"
      [ -n "${FAKE_PIP_SLEEP:-}" ] && sleep "$FAKE_PIP_SLEEP"
      echo "ERROR: fake pip: no network"
      [ "${FAKE_PIP_OK:-0}" = 1 ] && { touch "$D/hub_installed"; exit 0; }
      exit 1
    fi ;;
  -c)
    case "$2" in
      "import huggingface_hub")
        [ "${FAKE_IMPORT_HANG:-0}" = 1 ] && sleep 600
        [ "${FAKE_HF_HUB:-0}" = 1 ] || [ -f "$D/hub_installed" ]; exit $? ;;
      "import "*) exit 1 ;;
      *snapshot_download*)
        shift 2; echo "$*" >> "$D/snapshot.log"
        if [ "${FAKE_SNAP_TRAP:-0}" = 1 ]; then  # ignores TERM, never grows
          trap "" TERM; while :; do sleep 1; done
        fi
        if [ "${FAKE_SNAP_PREALLOC:-0}" = 1 ]; then  # full size first, then fill in place
          dd if=/dev/zero of="$3/big.bin" bs=1024 count=256 2>/dev/null
          i=0
          while [ $i -lt 8 ]; do
            printf x | dd of="$3/big.bin" bs=1 seek=$i conv=notrunc 2>/dev/null
            sleep 0.5; i=$((i + 1))
          done
        fi
        [ -n "${FAKE_SNAP_SLEEP:-}" ] && sleep "$FAKE_SNAP_SLEEP"
        echo "fake snapshot output line"
        exit "${FAKE_SNAP_RC:-0}" ;;
      *) echo "  py fake-module 0.0"; exit 0 ;;
    esac ;;
esac
exit 1
"""


BOOT_ORPHANS: dict[str, list[str]] = {}


def _run_boot(
    tmp_path: Path,
    extra_env: dict[str, str],
    *,
    runpodctl: str | None = None,
    wait_bound: float = 120,
    path: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Run pod_boot.sh with fake runpodctl/python3/curl/pip; return (log, runpodctl, python3)."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir(exist_ok=True)
    rp_log, py_log = tmp_path / "runpodctl.log", tmp_path / "python3.log"
    (fakebin / "runpodctl").write_text(
        runpodctl or f'#!/bin/sh\necho "$@" >> "{rp_log}"\necho "pod removed"\nexit 0\n'
    )
    (fakebin / "python3").write_text(FAKE_PY.replace("@DIR@", str(tmp_path)))
    # Hermetic: never touch the network from the preflight reachability check.
    (fakebin / "curl").write_text("#!/bin/sh\necho 'HTTP/2 200 (fake curl)'\n")
    (fakebin / "pip").write_text(
        f'#!/bin/sh\necho "$@" >> "{tmp_path}/pip.log"\n'
        '[ -n "${FAKE_PIP_SLEEP:-}" ] && sleep "$FAKE_PIP_SLEEP"\nexit 1\n'
    )
    for f in fakebin.iterdir():
        f.chmod(0o755)
    models = tmp_path / "models"
    env = {
        "PATH": f"{fakebin}:{path or '/usr/bin:/bin'}",
        "MEMFUN_MAX_MINUTES": "60",
        "MEMFUN_DEADLINE_EPOCH": "9999999999",
        "MEMFUN_MODELS_ROOT": str(models),
        "MEMFUN_MODEL_DIR": str(models / "Qwen" / "M"),
        "MEMFUN_MODEL_REPO": "Qwen/M",
        "MEMFUN_SERVED_NAME": "m",
        "MEMFUN_CRASH_GRACE_MIN": "0",
        "MEMFUN_FAIL_GRACE_MIN": "0",
        "MEMFUN_PULL_RETRY_SLEEP": "0",
        "MEMFUN_RUN_DIR": str(tmp_path / "run"),
        "RUNPOD_POD_ID": "podB",
        "PUBLIC_KEY": "",
        **extra_env,
    }
    out_file = tmp_path / "boot.out"
    with out_file.open("w") as out:
        proc = subprocess.Popen(
            ["bash", str(kit.BOOT_SCRIPT)],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=tmp_path,
        )
        try:
            proc.wait(timeout=wait_bound)  # liveness bound: TimeoutExpired = PID 1 hung
            # Orphans the script left (other process groups survive the killpg below).
            probe = subprocess.run(
                ["pgrep", "-f", str(models)], capture_output=True, text=True, check=False
            )
            BOOT_ORPHANS[str(tmp_path)] = probe.stdout.split()
        finally:  # the armed deadline subshell is still sleeping: kill the whole group
            import os as _os

            with contextlib.suppress(ProcessLookupError):
                _os.killpg(proc.pid, signal.SIGKILL)
            subprocess.run(["pkill", "-KILL", "-f", str(models)], check=False)
    rp = rp_log.read_text().splitlines() if rp_log.exists() else []
    py = py_log.read_text().splitlines() if py_log.exists() else []
    return out_file.read_text(), rp, py


def _write_marker(tmp_path: Path, sha: str, repo: str = "Qwen/M") -> None:
    d = tmp_path / "models" / "Qwen" / "M"
    d.mkdir(parents=True, exist_ok=True)
    (d / ".memfun_verified").write_text(json.dumps({"repo": repo, "sha": sha}))


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_boot_extra_args_never_glob_and_metrics_enabled(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40)
    (tmp_path / "SHOULD_NOT_APPEAR.txt").write_text("x")  # a glob of * in cwd would find this
    _log, rp, py = _run_boot(tmp_path, {"MEMFUN_SGLANG_ARGS": "--tp 1 *"})
    assert "--enable-metrics" in py and "*" in py and "SHOULD_NOT_APPEAR.txt" not in py
    assert py[py.index("--tp") + 1] == "1"
    assert rp and rp[0] == "remove pod podB"  # SGLang exited -> grace 0 -> self-terminate


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_boot_refuses_marker_for_other_pinned_revision(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40)
    log, rp, py = _run_boot(tmp_path, {"MEMFUN_MODEL_REVISION": "b" * 40})
    assert py == []  # SGLang never started
    assert "marker sha != pinned" in log and rp[0] == "remove pod podB"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_boot_refuses_marker_for_other_repo_and_bad_names(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40, repo="Other/M")
    log, _rp, py = _run_boot(tmp_path, {})
    assert py == [] and "marker repo != Qwen/M" in log
    log, _rp, py = _run_boot(
        tmp_path,
        {"MEMFUN_MODEL_REPO": "../../etc", "MEMFUN_MODEL_DIR": "/workspace/models/../../etc"},
    )
    assert py == [] and "invalid MEMFUN_MODEL_REPO" in log


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_boot_in_pod_pull_honours_pinned_revision(tmp_path: Path) -> None:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    hf_log = tmp_path / "hf.log"
    (fakebin / "hf").write_text(f'#!/bin/sh\necho "$@" >> "{hf_log}"\nexit 0\n')
    (fakebin / "hf").chmod(0o755)
    sha = "c" * 40
    _log, _rp, py = _run_boot(tmp_path, {"MEMFUN_ALLOW_PULL": "1", "MEMFUN_MODEL_REVISION": sha})
    assert f"download Qwen/M --revision {sha} --local-dir" in hf_log.read_text()
    marker = json.loads((tmp_path / "models" / "Qwen" / "M" / ".memfun_verified").read_text())
    assert marker["sha"] == sha and marker["repo"] == "Qwen/M"
    assert "--enable-metrics" in py


def _assert_qwen_everywhere(out: str, port: int) -> None:
    assert "\n[llm]\n" in out
    assert f'base_url = "http://127.0.0.1:{port}/v1"' in out
    assert f'model = "{kit.DEFAULT_SERVED_NAME}"' in out
    assert 'provider = "openai"' in out
    assert 'api_key_env = "MEMFUN_SGLANG_DUMMY_KEY"' in out
    assert "[llm.roles.worker]" not in out
    assert "anthropic" not in out.lower()


def test_memfun_snippet_is_single_qwen_llm_block(
    capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    snippet = kit.memfun_snippet(18123, "served-x")
    assert snippet.splitlines().count("[llm]") == 1 and "anthropic" not in snippet.lower()
    assert 'base_url = "http://127.0.0.1:18123/v1"' in snippet and 'model = "served-x"' in snippet

    port = _free_port()
    fake = _fake_with_pod()
    assert run(["tunnel", "--pod-id", "pod1", "--local-port", str(port)], fake) == 0
    _assert_qwen_everywhere(capsys.readouterr().out, port)

    system = FakeSystem(caffeinate=None)
    assert _watch(["watch", "--pod-id", "pod1", "--local-port", str(port)], fake, system) == 0
    _assert_qwen_everywhere(capsys.readouterr().out, port)

    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--ssh-public-key", pubkey], spy) == 0
    _assert_qwen_everywhere(capsys.readouterr().out, kit.DEFAULT_LOCAL_PORT)


# ---------------------------------------------------------------------------
# Boot script: downloader without hf/huggingface-cli, diagnostics, grace, mirror.
# ---------------------------------------------------------------------------

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


@needs_bash
def test_boot_pull_uses_python_api_when_no_hf_cli(tmp_path: Path) -> None:
    sha = "d" * 40
    log, rp, py = _run_boot(
        tmp_path, {"MEMFUN_ALLOW_PULL": "1", "MEMFUN_MODEL_REVISION": sha, "FAKE_HF_HUB": "1"}
    )
    snap = _lines(tmp_path / "snapshot.log")
    assert snap == [f"Qwen/M {sha} {tmp_path / 'models' / 'Qwen' / 'M'}"], log
    assert "via python-api" in log and "not available: hf huggingface-cli" in log
    assert "download finished via python-api" in log
    marker = json.loads((tmp_path / "models" / "Qwen" / "M" / ".memfun_verified").read_text())
    assert marker["sha"] == sha and marker["repo"] == "Qwen/M"
    assert "--enable-metrics" in py  # SGLang started after the download
    assert not (tmp_path / "pip.log").exists()  # pip only as the last resort
    assert rp == ["remove pod podB"]


@needs_bash
def test_boot_pull_pip_installs_hub_as_last_resort(tmp_path: Path) -> None:
    log, _rp, py = _run_boot(tmp_path, {"MEMFUN_ALLOW_PULL": "1", "FAKE_PIP_OK": "1"})
    assert "pip install -q huggingface_hub" in (tmp_path / "pip.log").read_text()
    assert "via python-api-pip" in log and _lines(tmp_path / "snapshot.log")
    assert "--enable-metrics" in py


@needs_bash
def test_boot_pull_nothing_available_fails_clearly_then_removes_pod(tmp_path: Path) -> None:
    log, rp, py = _run_boot(tmp_path, {"MEMFUN_ALLOW_PULL": "1"})
    assert "pip install -q huggingface_hub" in (tmp_path / "pip.log").read_text()
    assert (
        "no way to download weights: missing: hf huggingface-cli python3:huggingface_hub "
        "pip-install-failed" in log
    )
    # The reason and the tail of pull.log are on stdout BEFORE the pod goes away.
    assert "last 30 lines of" in log and "ERROR: fake pip: no network" in log
    assert log.index("ERROR: fake pip") < log.index("FAILED: weight download failed")
    assert log.index("FAILED: weight download failed") < log.index("self-terminate:")
    assert "last 40 lines of" in log and "pull.log" in log
    assert py == [] and rp == ["remove pod podB"]


@needs_bash
def test_boot_pull_retries_are_bounded_and_resumable(tmp_path: Path) -> None:
    log, rp, py = _run_boot(
        tmp_path, {"MEMFUN_ALLOW_PULL": "1", "FAKE_HF_HUB": "1", "FAKE_SNAP_RC": "1"}
    )
    assert len(_lines(tmp_path / "snapshot.log")) == 3
    assert "download attempt 3/3 failed" in log and "attempt 4" not in log
    assert "all 3 download attempts via python-api failed" in log
    assert "fake snapshot output line" in log  # tail of pull.log shown
    assert py == [] and rp == ["remove pod podB"]
    assert not (tmp_path / "models" / "Qwen" / "M" / ".memfun_verified").exists()


@needs_bash
def test_boot_pull_prints_progress_lines(tmp_path: Path) -> None:
    log, _rp, _py = _run_boot(
        tmp_path,
        {
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_SLEEP": "1.6",
            "MEMFUN_PROGRESS_SEC": "1",
        },
    )
    assert "download progress:" in log and "elapsed (python-api)" in log


@needs_bash
def test_boot_preflight_prints_names_never_values(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40)
    secret = "rpa_PODSCOPED_value_7f3e9c1d"
    log, _rp, _py = _run_boot(
        tmp_path,
        {
            "RUNPOD_API_KEY": secret,
            "RUNPOD_SOMETHING_TOKEN": "tok_value_should_not_print",
            "PUBLIC_KEY": "ssh-ed25519 AAAAPUBLICVALUE test",
        },
    )
    pre = log[log.index("---- preflight ----") : log.index("---- end preflight ----")]
    assert log.index("hard stop armed") < log.index("---- preflight ----")
    assert "RUNPOD_API_KEY" in pre and "RUNPOD_SOMETHING_TOKEN" in pre and "MEMFUN_" in pre
    for value in (secret, "tok_value_should_not_print", "AAAAPUBLICVALUE"):
        assert value not in log
    needles = ["uname:", "python3: Python 3.12.99-fake", "tool hf: MISSING", "disk:"]
    needles += ["tool runpodctl:", "huggingface.co: HTTP/2 200", "ulimit -n:"]
    for needle in needles:
        assert needle in pre, needle


@needs_bash
def test_boot_remove_pod_first_then_pod_delete_fallback(tmp_path: Path) -> None:
    rp_log = tmp_path / "runpodctl.log"
    old_cli_fails = (
        f'#!/bin/sh\necho "$@" >> "{rp_log}"\n'
        'if [ "$1" = remove ]; then echo \'Error: unknown command "remove"\' >&2; exit 1; fi\n'
        "exit 0\n"
    )
    log, rp, _py = _run_boot(tmp_path, {}, runpodctl=old_cli_fails)  # no weights -> fail
    assert rp == ["remove pod podB", "pod delete podB"]
    assert "terminated via 'runpodctl pod delete'" in log
    assert "unknown command" not in log


@needs_bash
def test_boot_mirrors_sglang_log_to_stdout(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40)
    log, _rp, _py = _run_boot(tmp_path, {"FAKE_SGLANG_OUT": "sglang-mirror-probe-line"})
    assert "sglang-mirror-probe-line" in log
    # Shown live by the mirror, before the exit message and the post-mortem dump.
    assert log.index("sglang-mirror-probe-line") < log.index("SGLang exited rc=")


@needs_bash
def test_boot_fail_grace_waits_but_deadline_still_fires(tmp_path: Path) -> None:
    import os as _os
    import time as _time

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    rp_log = tmp_path / "runpodctl.log"
    (fakebin / "runpodctl").write_text(f'#!/bin/sh\necho "$@" >> "{rp_log}"\nexit 0\n')
    (fakebin / "curl").write_text("#!/bin/sh\necho fake\n")
    (fakebin / "python3").write_text("#!/bin/sh\nexit 1\n")
    for f in fakebin.iterdir():
        f.chmod(0o755)
    env = {
        "PATH": f"{fakebin}:/usr/bin:/bin",
        "MEMFUN_MAX_MINUTES": "60",
        "MEMFUN_DEADLINE_EPOCH": str(int(_time.time()) + 3),
        "MEMFUN_MODELS_ROOT": str(tmp_path / "models"),
        "MEMFUN_MODEL_DIR": str(tmp_path / "models" / "Qwen" / "M"),
        "MEMFUN_MODEL_REPO": "Qwen/M",
        "MEMFUN_FAIL_GRACE_MIN": "1",
        "MEMFUN_RUN_DIR": str(tmp_path / "run"),
        "RUNPOD_POD_ID": "podG",
        "PUBLIC_KEY": "",
    }
    out_file = tmp_path / "boot.out"
    with out_file.open("w") as out:
        proc = subprocess.Popen(
            ["bash", str(kit.BOOT_SCRIPT)], env=env, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        try:
            for _ in range(300):  # wait for the deadline subshell's removal, not a fixed time
                if rp_log.exists():
                    break
                _time.sleep(0.05)
        finally:
            with contextlib.suppress(ProcessLookupError):
                _os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    log = out_file.read_text()
    assert "FAILED: weights missing/unverified" in log
    assert "terminating in 1m (MEMFUN_FAIL_GRACE_MIN)" in log
    assert "self-terminate: deadline reached" in log  # the grace never outlives the deadline
    assert "self-terminate: weights missing" not in log  # still inside the grace
    assert _lines(rp_log) == ["remove pod podG"]


def test_fail_grace_flag_reaches_pod_env(capsys: pytest.CaptureFixture[str], pubkey: str) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--ssh-public-key", pubkey, "--fail-grace-minutes", "5"], spy) == 0
    assert '"key": "MEMFUN_FAIL_GRACE_MIN"' in capsys.readouterr().out
    args = kit.build_parser().parse_args(["launch", "--fail-grace-minutes", "4"])
    env = {e["key"]: e["value"] for e in kit.build_boot_env(args, 0, "")}
    assert env["MEMFUN_FAIL_GRACE_MIN"] == "4"
    assert kit.build_boot_env(kit.build_parser().parse_args(["launch"]), 0, "")[-1] == {
        "key": "MEMFUN_FAIL_GRACE_MIN",
        "value": "2",
    }
    with pytest.raises(SystemExit):
        kit.build_parser().parse_args(["launch", "--fail-grace-minutes", "-1"])


# ---------------------------------------------------------------------------
# Pod logs: SSE parsing, noise filter, retries, capture file, messages.
# ---------------------------------------------------------------------------


def _ev(line: str, source: str = "container", ts: str = "t") -> str:
    return "data: " + json.dumps({"ts": ts, "source": source, "line": line})


class FakeLogs:
    """Scripted log stream: each connection is a list of lines and/or an exception."""

    def __init__(self, *connections: list[Any] | BaseException) -> None:
        self.connections = list(connections)
        self.calls: list[tuple[str, dict[str, str], str | None]] = []

    def __call__(
        self, pod_id: str, params: dict[str, str], last_id: str | None, timeout: float
    ) -> Iterator[str]:
        self.calls.append((pod_id, dict(params), last_id))
        conn = self.connections.pop(0) if self.connections else kit.LogStreamError("x", status=404)
        if isinstance(conn, BaseException):
            raise conn
        for item in conn:
            if isinstance(item, BaseException):
                raise item
            yield item


@pytest.fixture(autouse=True)
def _fast_capture_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kit.LogCapture, "backoff", 0.01)


class FakeRunPodWithLogs(FakeRunPod):
    def __init__(self, logs: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self.stream_logs = logs


def _follow(logs: FakeLogs, **kw: Any) -> tuple[list[str], kit.LogStats, FakeClock]:
    out: list[str] = []
    clock = FakeClock()
    kw.setdefault("follow", False)
    stats = kit.follow_logs(logs, "pod1", out.append, clock=clock, sleep=clock.sleep, **kw)
    return out, stats, clock


def test_sse_parser_multiline_comments_ids_and_bad_json() -> None:
    p = kit.SseParser()
    got: list[kit.SseEvent] = []
    for line in [": keep-alive", "id: 7", 'data: {"line":', 'data: "a"}', "event: log",
                 "data: plain one", "data: plain two", "", "data: {bad json", ""]:  # fmt: skip
        got += p.feed(line)
    assert [e.data for e in got] == ['{"line":\n"a"}', "plain one\nplain two", "{bad json"]
    assert got[0].event_id == "7" and p.last_id == "7"
    assert kit.format_log_event(got[2]) == ("[raw] {bad json", False)
    text, noise = kit.format_log_event(kit.SseEvent(_ev("hi", "system", "T1")[6:], None))
    assert text == "T1 [syst] hi" and not noise


def test_log_noise_filter_hides_pull_progress_unless_raw() -> None:
    lines = [_ev("9f8e7d6c5b4a Extracting", "system"), _ev("abcdef012345 Pull complete ")]
    lines += [_ev("[memfun-boot] hard stop armed"), _ev("9f8e7d6c5b4a Extracting more")]
    out, stats, _ = _follow(FakeLogs(lines))
    assert out == [
        "t [cont] [memfun-boot] hard stop armed",
        "t [cont] 9f8e7d6c5b4a Extracting more",
    ]
    assert stats.hidden == 2 and stats.events == 4 and stats.outcome == "ended"
    out, stats, _ = _follow(FakeLogs(lines), raw=True)
    assert len(out) == 4 and stats.hidden == 0


def test_logs_404_means_pod_gone_and_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    logs = FakeLogs([_ev("last words")], kit.LogStreamError("HTTP 404", status=404))
    fake = FakeRunPodWithLogs(logs, allow_mutations=False)
    assert run(["logs", "--pod-id", "pod1", "--follow"], fake) == 0
    out = capsys.readouterr().out
    assert "last words" in out and "pod pod1 is gone or does not exist (HTTP 404" in out
    assert fake.mutations == [] and logs.calls[0][1] == {"tail": "2000"}
    assert run(["logs", "--pod-id", "pod1", "--source", "system", "--tail", "50"], fake) == 0
    assert logs.calls[-1][1] == {"tail": "50", "source": "system"}


def test_logs_429_honours_retry_after_with_fake_clock() -> None:
    logs = FakeLogs(kit.LogStreamError("HTTP 429", status=429, retry_after=7.0), [_ev("ok")])
    out, stats, clock = _follow(logs, follow=True)
    assert 7.0 in clock.sleeps and "t [cont] ok" in out
    assert any("rate-limited (429); retrying in 7s" in o for o in out)
    assert stats.outcome == "gone"  # the scripted stream then 404s


def test_logs_reconnect_sends_last_event_id_and_skips_duplicates() -> None:
    conn1: list[Any] = ["id: 1", _ev("one"), "", "id: 2", _ev("two"), ""]
    conn1.append(kit.LogStreamError("connection reset"))
    conn2 = ["id: 2", _ev("two"), "", "id: 3", _ev("three"), ""]
    logs = FakeLogs(conn1, conn2)
    out, stats, _ = _follow(logs, follow=True)
    assert [c[2] for c in logs.calls[:2]] == [None, "2"]
    assert out.count("t [cont] two") == 1 and "t [cont] three" in out
    assert stats.outcome == "gone"


def test_logs_errors_are_bounded() -> None:
    logs = FakeLogs(*[kit.LogStreamError("HTTP 502", status=502) for _ in range(10)])
    out, stats, clock = _follow(logs, follow=True, max_errors=3)
    assert stats.outcome == "error" and len(logs.calls) == 3 and len(clock.sleeps) == 2
    assert out[-1].startswith("# log stream error 3/3")


def test_logs_non_follow_has_hard_overall_timeout() -> None:
    class Endless:
        calls = 0

        def __call__(self, *_a: Any) -> Iterator[str]:
            self.calls += 1
            while True:
                clock.t += 1.0
                yield _ev("spam")

    clock = FakeClock()
    out: list[str] = []
    stats = kit.follow_logs(
        Endless(), "pod1", out.append, follow=False, clock=clock, sleep=clock.sleep,
        overall_timeout=10.0,
    )  # fmt: skip
    assert stats.outcome == "timeout" and 5 <= len(out) <= 11


def test_logs_redact_key_in_lines_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    logs = FakeLogs(
        [_ev(f"echo {FAKE_KEY} and prefix {FAKE_KEY[:12]}")],
        kit.LogStreamError(f"HTTP 500 body says Bearer {FAKE_KEY}", status=500),
        kit.LogStreamError("gone", status=404),
    )
    fake = FakeRunPodWithLogs(logs)
    assert run(["logs", "--pod-id", "pod1", "--follow"], fake, FakeClock()) == 0
    cap = capsys.readouterr()
    _assert_clean(cap.out + cap.err, FAKE_KEY)
    assert "echo *** and prefix ***" in cap.out


def test_logs_out_file_is_private_and_refuses_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "pod.log"
    fake = FakeRunPodWithLogs(FakeLogs([_ev("hello")]))
    assert run(["logs", "--pod-id", "pod1", "--out", str(out)], fake) == 0
    assert out.stat().st_mode & 0o777 == 0o600 and "t [cont] hello" in out.read_text()
    assert run(["logs", "--pod-id", "pod1", "--out", str(out)], fake) == 1
    assert "refusing to write" in capsys.readouterr().err
    assert run(["logs", "--pod-id", "../etc"], fake) == 1


def test_log_capture_writes_0600_file_and_refuses_planted_symlink(tmp_path: Path) -> None:
    target = tmp_path / "victim.txt"
    target.write_text("original\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "pod-pod1.log").symlink_to(target)
    logs = FakeLogs([_ev(f"boot line {FAKE_KEY}")], kit.LogStreamError("x", status=404))
    cap = kit.LogCapture.start(FakeRunPodWithLogs(logs), "pod1", logs_dir)
    assert cap is not None
    cap.finish(wait=10)
    assert cap.path == logs_dir / "pod-pod1.1.log"
    assert target.read_text() == "original\n"  # never written through the symlink
    assert cap.path.stat().st_mode & 0o777 == 0o600
    text = cap.path.read_text()
    assert "boot line ***" in text and "# capture ended: gone" in text and FAKE_KEY not in text


def test_launch_captures_logs_and_prints_path_in_run_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    logs = FakeLogs(["data: " + json.dumps({"ts": "T", "source": "container", "line": "hi"})])
    fake = FakeRunPodWithLogs(logs)
    assert _real_launch(fake, pubkey, "--log-dir", str(tmp_path / "L")) == 0
    out = capsys.readouterr().out
    path = tmp_path / "L" / "pod-pod1.log"
    assert f"log file   : {path}" in out and f"pod log saved: {path}" in out
    assert "logs --pod-id pod1 --follow" in out and "still initializing" in out
    assert "T [cont] hi" in path.read_text()
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}


def test_log_capture_failure_never_breaks_termination(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pubkey: str
) -> None:
    fake = FakeRunPodWithLogs(FakeLogs(RuntimeError("stream exploded")))
    assert _real_launch(fake, pubkey, "--log-dir", str(tmp_path / "L")) == 0
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    fake = FakeRunPodWithLogs(FakeLogs())
    assert _real_launch(fake, pubkey, "--log-dir", str(blocker)) == 0
    assert "log capture disabled" in capsys.readouterr().err
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}


def test_no_ssh_mapping_message_says_initializing(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.pods["pod1"]["runtime"] = {"uptimeInSeconds": None, "ports": []}
    assert run(["tunnel", "--pod-id", "pod1", "--local-port", str(_free_port())], fake) == 1
    err = capsys.readouterr().err
    assert "still initializing" in err and "status RUNNING" in err
    assert "10+ minutes" in err and "logs --pod-id pod1 --follow" in err
    system = FakeSystem(caffeinate=None)
    assert _watch(["watch", "--pod-id", "pod1", "--local-port", str(_free_port())], fake, system)
    assert "still initializing" in capsys.readouterr().err


def test_http_stream_logs_against_local_sse_server() -> None:
    import http.server
    import threading

    seen: list[dict[str, str]] = []
    plan = ["sse", "429", "404"]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

        def do_GET(self) -> None:
            seen.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
            mode = plan.pop(0)
            if mode == "sse":
                body = f"id: 9\n{_ev('srv line ' + FAKE_KEY)}\n\n: ping\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            elif mode == "429":
                body = f"slow down {FAKE_KEY}".encode()
                self.send_response(429)
                self.send_header("Retry-After", "3")
            else:
                body = b"not found"
                self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v2/pods/{{pod_id}}/logs"
        kit.REDACT.add(kit.Secret(FAKE_KEY))
        t = kit.HttpTransport(kit.Secret(FAKE_KEY), logs_url=url)
        assert kit.log_stream_for(kit.ReadOnlyTransport(t)) is not None
        out: list[str] = []
        clock = FakeClock()
        stats = kit.follow_logs(
            t.stream_logs, "podA", out.append, follow=True, clock=clock, sleep=clock.sleep,
            read_timeout=5.0,
        )  # fmt: skip
    finally:
        srv.shutdown()
    assert stats.outcome == "gone"
    assert out[0] == "t [cont] srv line ***"
    assert seen[0]["authorization"] == f"Bearer {FAKE_KEY}"
    assert seen[0]["accept"] == "text/event-stream" and seen[0]["path"].endswith("tail=2000")
    assert seen[1]["last-event-id"] == "9"  # reconnect resumes after the last event
    assert 3.0 in clock.sleeps
    assert not any(FAKE_KEY in o for o in out)


def test_launch_pod_wrapper_knows_every_kit_subcommand() -> None:
    text = (REPO_ROOT / "scripts" / "runpod" / "launch_pod.py").read_text()
    names = re.search(r"SUBCOMMANDS = \{([^}]*)\}", text)
    assert names
    wrapper = set(re.findall(r'"([a-z]+)"', names.group(1)))
    parser = kit.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # pyright: ignore[reportPrivateUsage]
    assert wrapper == set(sub.choices)


# ---------------------------------------------------------------------------
# Signal-proof termination (incident: a second SIGTERM skipped terminate_and_verify).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_termination_state() -> Iterator[None]:
    def reset() -> None:
        for name in ("VERIFIED_GONE", "LAST_STATUS", "_ABSENT_SINCE", "_BUDGET"):
            state = getattr(kit, name, None)
            if state is not None:
                state.clear()

    reset()
    yield
    reset()


class ChaosClock(FakeClock):
    """FakeClock whose clock()/sleep() raise on chosen call numbers (1-based)."""

    def __init__(self, clock_raise: set[int], sleep_raise: dict[int, type[BaseException]]):
        super().__init__()
        self.clock_raise, self.sleep_raise = clock_raise, sleep_raise
        self.n_clock = self.n_sleep = 0

    def __call__(self) -> float:
        self.n_clock += 1
        if self.n_clock in self.clock_raise:
            raise KeyboardInterrupt("second signal in clock()")
        return super().__call__()

    def sleep(self, seconds: float) -> None:
        self.n_sleep += 1
        exc = self.sleep_raise.get(self.n_sleep)
        if exc is not None:
            raise exc("signal during sleep")
        super().sleep(seconds)


SECOND_SIGNAL_POINTS = [
    # (name, emit substring that raises once, clock calls that raise, extra sleep raises,
    #  terminate failures)
    ("emit-stopping-pod", "stopping pod", set(), {}, 0),  # the incident window
    ("spent-computation", None, {4}, {}, 0),  # call 1 is the test computing the deadline
    ("emit-terminate-accepted", "terminate request 1/6 accepted", set(), {}, 0),
    ("emit-verified", "verified: pod", set(), {}, 0),
    ("verify-sleep", None, set(), {2: KeyboardInterrupt}, 1),
    ("between-retries-emit", "terminate attempt 1/6 failed", set(), {}, 1),
    ("still-listed-emit", "still listed", set(), {}, 1),
    ("terminate-call-itself", None, set(), {}, -1),
]


@pytest.mark.parametrize(
    ("name", "emit_match", "clock_raise", "sleep_raise", "term_fail"),
    SECOND_SIGNAL_POINTS,
    ids=[p[0] for p in SECOND_SIGNAL_POINTS],
)
def test_second_signal_anywhere_in_shutdown_still_terminates(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    emit_match: str | None,
    clock_raise: set[int],
    sleep_raise: dict[int, type[BaseException]],
    term_fail: int,
) -> None:
    fake = _fake_with_pod()
    fake.terminate_failures = max(term_fail, 0)
    if term_fail < 0:  # the first podTerminate call is interrupted by a signal
        orig = fake.graphql
        state = {"hit": False}

        def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
            if "podTerminate" in query and not state["hit"]:
                state["hit"] = True
                raise KeyboardInterrupt("second signal during podTerminate")
            return orig(query, variables)

        fake.graphql = graphql  # type: ignore[method-assign]
    # First signal: during the first poll's sleep (cost is known by then).
    clock = ChaosClock(clock_raise, {1: kit.TerminateRequestedError, **sleep_raise})
    real_emit = kit.emit
    fired = {"done": False}

    def chaos_emit(message: str = "", *, err: bool = False) -> None:
        if emit_match and emit_match in message and not fired["done"]:
            fired["done"] = True
            raise KeyboardInterrupt("second signal inside emit")
        real_emit(message, err=err)

    monkeypatch.setattr(kit, "emit", chaos_emit)
    try:
        result = kit.run_watchdog(
            fake, "pod1", clock() + 3600, clock=clock, sleep=clock.sleep, poll_seconds=30
        )
        assert result.verified_gone
    except KeyboardInterrupt:
        pass  # allowed to propagate AFTER termination; the pod must be gone either way
    assert emit_match is None or fired["done"], "injection point never reached"
    assert clock.n_clock >= max(clock_raise, default=0), "clock injection never reached"
    assert clock.n_sleep >= max(sleep_raise, default=1), "sleep injection never reached"
    assert fake.pods == {}, name  # terminated and verified (the fake pops it)
    assert "terminate" in fake.mutations


def test_launch_finally_terminates_when_watchdog_raises(
    monkeypatch: pytest.MonkeyPatch, pubkey: str
) -> None:
    def broken_watchdog(*_a: Any, **_k: Any) -> kit.WatchResult:
        raise KeyboardInterrupt("escaped run_watchdog")

    monkeypatch.setattr(kit, "run_watchdog", broken_watchdog)
    fake = FakeRunPod()
    assert _real_launch(fake, pubkey) == 130
    assert fake.mutations == ["deploy", "terminate"] and fake.pods == {}


def test_watchdog_subcommand_finally_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_watchdog(*_a: Any, **_k: Any) -> kit.WatchResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(kit, "run_watchdog", broken_watchdog)
    fake = _fake_with_pod()
    assert run(["watchdog", "--pod-id", "pod1", "--max-minutes", "5"], fake) == 1
    assert fake.pods == {} and fake.mutations == ["terminate"]


def test_listed_pod_with_any_live_status_is_not_gone(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    fake.never_gone = True
    fake.pods["pod1"]["desiredStatus"] = "EXITED"  # container stopped, pod still billing
    assert kit._pod_gone(fake, "pod1") is False  # pyright: ignore[reportPrivateUsage]
    called, gone = kit.terminate_and_verify(fake, "pod1", sleep=lambda _s: None, retries=4)
    assert called and not gone and fake.terminate_calls == 4  # keeps calling podTerminate
    err = capsys.readouterr().err
    assert "still listed with desiredStatus=EXITED" in err and "COULD NOT VERIFY" in err
    fake.pods["pod1"]["desiredStatus"] = "RUNNING"
    assert kit._pod_gone(fake, "pod1") is False  # pyright: ignore[reportPrivateUsage]
    fake.pods["pod1"]["desiredStatus"] = "TERMINATED"
    assert kit._pod_gone(fake, "pod1") is True  # pyright: ignore[reportPrivateUsage]
    assert kit._pod_gone(fake, "absent1") is True  # pyright: ignore[reportPrivateUsage]


def test_listing_failure_falls_back_and_unknown_is_not_gone() -> None:
    gone = kit._pod_gone  # pyright: ignore[reportPrivateUsage]
    fake = _fake_with_pod()
    fake.lookup_fails = True
    assert gone(fake, "pod1") is False  # fallback pod(input): still there
    fake.pods.clear()
    assert gone(fake, "pod1") is True  # fallback: pod is null
    fake.pod_errors = 1
    assert gone(fake, "pod1") is None  # both failed: unknown, never "gone"


def test_terminate_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    spy = _fake_with_pod()
    spy.allow_mutations = False
    assert run(["terminate", "--pod-id", "pod1"], spy) == 2
    assert "add --yes" in capsys.readouterr().err and spy.mutations == []
    assert run(["terminate", "--pod-id", "pod-1;rm", "--yes"], spy) == 1
    fake = _fake_with_pod()
    assert run(["terminate", "--pod-id", "pod1", "--yes"], fake) == 0
    assert fake.pods == {} and "final state: gone" in capsys.readouterr().out
    stuck = _fake_with_pod()
    stuck.never_gone = True
    assert run(["terminate", "--pod-id", "pod1", "--yes"], stuck) == 3
    out = capsys.readouterr()
    assert "NOT verified gone (last desiredStatus RUNNING)" in out.out
    echo = _fake_with_pod()
    echo.raise_any = lambda: kit.KitError(f"HTTP 500 Bearer {FAKE_KEY}")
    assert run(["terminate", "--pod-id", "pod1", "--yes"], echo) == 3
    cap = capsys.readouterr()
    _assert_clean(cap.out + cap.err, FAKE_KEY)


CHILD = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from scripts.runpod import runpod_kit as kit
marker = sys.argv[2]

class Fake:
    def __init__(self):
        self.pods = {"podX": "RUNNING"}
    def graphql(self, q, v=None):
        if "podTerminate" in q:
            open(marker, "a").write("terminate\n")
            time.sleep(0.3)  # a slow API call: the window for further signals
            self.pods.clear()
            return {"podTerminate": None}
        if "myself { pods" in q:
            pods = [{"id": k, "desiredStatus": s} for k, s in self.pods.items()]
            return {"myself": {"pods": pods}}
        if "pod(input" in q:
            open(marker + ".ready", "w").write("1")
            if not self.pods:
                return {"pod": None}
            return {"pod": {"id": "podX", "name": "x", "desiredStatus": "RUNNING",
                            "costPerHr": 9.0, "runtime": {"uptimeInSeconds": 1, "ports": []}}}
        raise AssertionError(q)
    def rest(self, *a):
        raise AssertionError(a)

kit.install_exit_signals()
fake = Fake()
argv = ["watchdog", "--pod-id", "podX", "--max-minutes", "30", "--poll-seconds", "0.1"]
env = {"RUNPOD_API_KEY": "rpa_FAKEKEY_0123456789abcdefSECRET"}
rc = kit.main(argv, factory=lambda _k: fake, env=env)
open(marker + ".rc", "w").write(str(rc) + " " + ("gone" if not fake.pods else "ALIVE"))
"""

WRAPPER = r"""
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", sys.argv[1], *sys.argv[2:]])
def fwd(signum, _f):
    child.send_signal(signum)  # like `uv run`: forward, on top of process-group delivery
    time.sleep(0.05)
    child.send_signal(signum)  # and once more, landing inside the shutdown path
signal.signal(signal.SIGTERM, fwd)
sys.exit(child.wait())
"""


def test_killpg_wrapper_plus_kit_double_sigterm_still_terminates(tmp_path: Path) -> None:
    import os as _os
    import time as _time

    marker = tmp_path / "events"
    proc = subprocess.Popen(
        [sys.executable, "-c", WRAPPER, CHILD, str(REPO_ROOT), str(marker)],
        start_new_session=True,
    )
    try:
        for _ in range(400):
            if (tmp_path / "events.ready").exists():
                break
            _time.sleep(0.05)
        assert (tmp_path / "events.ready").exists(), "kit never polled"
        _os.killpg(proc.pid, signal.SIGTERM)  # group delivery + the wrapper's forwards
        proc.wait(timeout=60)
    finally:
        with contextlib.suppress(ProcessLookupError):
            _os.killpg(proc.pid, signal.SIGKILL)
    assert (tmp_path / "events.rc").read_text().split()[1] == "gone"
    assert "terminate" in marker.read_text()


# Boot script: hangs can never keep PID 1 alive past the deadline.
# ---------------------------------------------------------------------------


def _rp_script(tmp_path: Path, body: str) -> str:
    return f'#!/bin/sh\necho "$@" >> "{tmp_path / "runpodctl.log"}"\n{body}\n'


def _soon(seconds: int) -> str:
    import time as _time

    return str(int(_time.time()) + seconds)


@needs_bash
def test_boot_hung_download_and_failed_removal_still_ends_pid1(tmp_path: Path) -> None:
    log, rp, py = _run_boot(
        tmp_path,
        {
            "MEMFUN_DEADLINE_EPOCH": _soon(3),
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_SLEEP": "600",  # the download hangs
            "MEMFUN_TERM_ATTEMPTS": "1",
            "MEMFUN_TERM_RETRY_SLEEP": "0",
            "MEMFUN_KILL_GRACE": "0",
        },
        runpodctl=_rp_script(tmp_path, "exit 1"),  # removal fails
        wait_bound=60,
    )
    assert "self-terminate: deadline reached" in log
    assert rp == ["remove pod podB", "pod delete podB"]
    # The main shell's TERM trap ran promptly (not deferred behind the download).
    assert "container shutdown: stopping all processes and exiting PID 1" in log
    assert py == []


@needs_bash
def test_boot_hung_runpodctl_is_bounded(tmp_path: Path) -> None:
    log, rp, _py = _run_boot(
        tmp_path,
        {"MEMFUN_RUNPODCTL_TIMEOUT": "1", "MEMFUN_TERM_ATTEMPTS": "1", "MEMFUN_KILL_GRACE": "0"},
        runpodctl=_rp_script(tmp_path, "sleep 600"),
        wait_bound=60,
    )  # no weights -> fail -> self_terminate in the main shell
    assert rp == ["remove pod podB", "pod delete podB"]
    assert "'runpodctl remove pod' timed out after 1s" in log
    assert "'runpodctl remove pod' failed (rc=124)" in log
    assert "pod-scoped termination failed" in log


@needs_bash
def test_boot_download_stall_detection_retries(tmp_path: Path) -> None:
    log, rp, py = _run_boot(
        tmp_path,
        {
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_SLEEP": "600",
            "MEMFUN_PROGRESS_SEC": "1",
            "MEMFUN_STALL_SEC": "2",
            "MEMFUN_PULL_RETRIES": "2",
        },
        wait_bound=60,
    )
    assert log.count("download stalled: no progress") == 2
    assert len(_lines(tmp_path / "snapshot.log")) == 2
    assert "all 2 download attempts via python-api failed" in log
    assert py == [] and rp == ["remove pod podB"]


@needs_bash
def test_boot_pip_hang_is_bounded(tmp_path: Path) -> None:
    log, rp, _py = _run_boot(
        tmp_path,
        {"MEMFUN_ALLOW_PULL": "1", "FAKE_PIP_SLEEP": "600", "MEMFUN_PIP_TIMEOUT": "1"},
        wait_bound=60,
    )
    assert "timed out after 1s: python3" in log and "timed out after 1s: pip" in log
    assert "no way to download weights" in log and rp == ["remove pod podB"]


# ---------------------------------------------------------------------------
# The Bearer key never follows a redirect.
# ---------------------------------------------------------------------------


def _start_server(handler: Any) -> tuple[Any, int]:
    import http.server
    import threading

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, int(srv.server_address[1])


def test_api_key_never_sent_to_redirect_target(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.server

    received: list[dict[str, str]] = []

    class Target(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

        def _any(self) -> None:
            received.append({k.lower(): v for k, v in self.headers.items()})
            body = b'{"data": {}}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._any()

        def do_POST(self) -> None:
            self._any()

    target, tport = _start_server(Target)

    class Redirector(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

        def _any(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{tport}/stolen?k={FAKE_KEY}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            self._any()

        def do_POST(self) -> None:
            self._any()

    redirector, rport = _start_server(Redirector)
    try:
        base = f"http://127.0.0.1:{rport}"
        kit.REDACT.add(kit.Secret(FAKE_KEY))
        monkeypatch.setattr(kit, "GRAPHQL_URL", base + "/graphql")
        monkeypatch.setattr(kit, "REST_URL", base + "/v1")
        t = kit.HttpTransport(kit.Secret(FAKE_KEY), logs_url=base + "/v2/pods/{pod_id}/logs")
        errors: list[str] = []
        for call in (
            lambda: t.graphql("query { myself { id } }"),
            lambda: t.rest("GET", "/pods"),
            lambda: list(t.stream_logs("pod1", {"tail": "1"}, None, 5.0)),
        ):
            with pytest.raises(kit.KitError) as info:
                call()
            errors.append(str(info.value))
    finally:
        redirector.shutdown()
        target.shutdown()
    assert received == []  # the redirect target never got a request, let alone the key
    for msg in errors:
        assert "refusing HTTP 302 redirect" in msg
        _assert_clean(msg, FAKE_KEY)


# ---------------------------------------------------------------------------
# Audit advisories: 404 confirmation, capture cap, bounded --timeout, no "-" names.
# ---------------------------------------------------------------------------


def test_logs_404_while_pod_exists_is_not_gone(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_with_pod()
    calls = {"n": 0}

    def stream(*_a: Any) -> Iterator[str]:
        calls["n"] += 1
        if calls["n"] == 3:
            yield _ev("container finally logging")
            fake.pods.clear()  # then the pod goes away
        raise kit.LogStreamError("HTTP 404", status=404)

    fake_l = FakeRunPodWithLogs(stream)
    fake_l.pods = fake.pods
    assert run(["logs", "--pod-id", "pod1"], fake_l) == 0  # non-follow: says so, exits
    assert "still exists: no logs yet" in capsys.readouterr().out
    calls["n"] = 1
    assert run(["logs", "--pod-id", "pod1", "--follow"], fake_l) == 0
    out = capsys.readouterr().out
    assert "waiting for its logs" in out and "container finally logging" in out
    assert "is gone or does not exist" in out


def test_logs_404_with_unknown_pod_state_is_bounded(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeRunPodWithLogs(FakeLogs(*[kit.LogStreamError("x", status=404) for _ in range(9)]))
    fake.lookup_fails = True
    fake.pod_errors = 99
    assert run(["logs", "--pod-id", "pod1", "--follow"], fake) == 1
    assert "pod state is unknown (5/5)" in capsys.readouterr().out


def test_log_capture_file_size_is_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kit.LogCapture, "max_bytes", 200)
    logs = FakeLogs([_ev("x" * 60) for _ in range(20)])
    cap = kit.LogCapture.start(FakeRunPodWithLogs(logs), "pod1", tmp_path)
    assert cap is not None
    cap.finish(wait=10)
    text = cap.path.read_text()
    assert "capture stopped: file reached its 200 byte cap" in text
    assert len(text.encode()) < 400


@pytest.mark.parametrize("value", ["inf", "nan", "0", "-5", "99999"])
def test_logs_timeout_must_be_finite_and_bounded(value: str) -> None:
    with pytest.raises(SystemExit):
        kit.build_parser().parse_args(["logs", "--pod-id", "p1", "--timeout", value])


@pytest.mark.parametrize(
    "extra",
    [["--model-repo", "-rf"], ["--model-repo", "org/-x"], ["--model-revision", "--help"]],
)
def test_names_starting_with_dash_refused(extra: list[str], pubkey: str) -> None:
    spy = FakeRunPod(allow_mutations=False)
    assert run(["launch", "--ssh-public-key", pubkey, *[f"{extra[0]}={extra[1]}"]], spy) == 1
    assert spy.calls == []


@needs_bash
def test_boot_refuses_names_starting_with_dash(tmp_path: Path) -> None:
    log, _rp, py = _run_boot(tmp_path, {"MEMFUN_MODEL_REVISION": "--help"})
    assert "invalid MEMFUN_MODEL_REVISION" in log and py == []
    log, _rp, py = _run_boot(tmp_path, {"MEMFUN_MODEL_REPO": "Qwen/-x"})
    assert "invalid MEMFUN_MODEL_REPO" in log and py == []


# ---------------------------------------------------------------------------
# Audit round 2, kit: deploy window, termination budget, strict gone semantics.
# ---------------------------------------------------------------------------


def _deploy_then_raise(fake: FakeRunPod, exc: BaseException) -> None:
    orig = fake.graphql

    def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        out = orig(query, variables)
        if "podFindAndDeployOnDemand" in query:
            raise exc  # RunPod created the pod, then the signal hit
        return out

    fake.graphql = graphql  # type: ignore[method-assign]


def test_sighup_in_deploy_with_dead_stderr_still_terminates(
    monkeypatch: pytest.MonkeyPatch, pubkey: str
) -> None:
    import errno

    fake = FakeRunPod()
    _deploy_then_raise(fake, kit.TerminateRequestedError("signal 1"))
    real_emit = kit.emit

    def emit(message: str = "", *, err: bool = False) -> None:
        if err:
            raise OSError(errno.EIO, "stderr closed")
        real_emit(message)

    monkeypatch.setattr(kit, "emit", emit)
    with contextlib.suppress(BaseException):
        _real_launch(fake, pubkey)
    assert "terminate" in fake.mutations and fake.pods == {}


def test_ctrl_c_in_deploy_then_ctrl_c_in_lookup_still_terminates(pubkey: str) -> None:
    fake = FakeRunPod()
    _deploy_then_raise(fake, KeyboardInterrupt())
    inner = fake.graphql
    hit = {"n": 0}

    def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if "myself { pods" in query and hit["n"] == 0:
            hit["n"] = 1
            raise KeyboardInterrupt()  # second Ctrl-C during the by-name lookup
        return inner(query, variables)

    fake.graphql = graphql  # type: ignore[method-assign]
    with contextlib.suppress(BaseException):
        _real_launch(fake, pubkey)
    assert "terminate" in fake.mutations and fake.pods == {}


class SlowFailingTransport:
    """Every call costs 30 s of FAKE time, then fails."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = 0

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls += 1
        self.clock.t += 30.0
        raise kit.KitError(f"RunPod request failed: TimeoutError (auth {FAKE_KEY})")

    def rest(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError("no REST")


def test_termination_has_one_shared_bounded_budget(capsys: pytest.CaptureFixture[str]) -> None:
    clock = FakeClock()
    t = SlowFailingTransport(clock)
    t0 = clock.t
    called, gone = kit.terminate_guarded(t, "pod1", sleep=clock.sleep, clock=clock)
    assert not called and not gone
    assert clock.t - t0 <= kit.TERM_BUDGET_SECONDS + 30 + 10  # one call + one sleep overrun
    n = t.calls
    called, gone = kit.terminate_guarded(t, "pod1", sleep=clock.sleep, clock=clock)  # net
    assert t.calls == n and not gone  # the shared budget is spent: no further calls
    cap = capsys.readouterr()
    assert "COULD NOT VERIFY POD pod1" in cap.err and "terminate --pod-id pod1 --yes" in cap.err
    _assert_clean(cap.out + cap.err, FAKE_KEY)


def test_launch_with_failing_api_after_deploy_ends_within_budget(pubkey: str) -> None:
    fake = FakeRunPod()
    clock = FakeClock()
    orig = fake.graphql

    def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if fake.deployed:  # after the pod exists, every call is slow and fails
            clock.t += 30.0
            raise kit.KitError("RunPod HTTP 503: unavailable")
        return orig(query, variables)

    fake.graphql = graphql  # type: ignore[method-assign]
    t0 = clock.t
    argv = ["launch", "--no-dry-run", "--yes-spend-money", "--ssh-public-key", pubkey]
    rc = run([*argv, "--max-minutes", "1", "--poll-seconds", "30"], fake, clock)
    assert rc == 3
    # watchdog (<= 5 failed polls) + ONE shared termination budget for both passes
    assert clock.t - t0 <= 60 + 5 * 60 + kit.TERM_BUDGET_SECONDS + 60


def test_signal_during_termination_is_reported_not_raised(
    capfd: pytest.CaptureFixture[str],
) -> None:
    fake = _fake_with_pod()
    orig = fake.graphql
    sent = {"n": 0}

    def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if "podTerminate" in query and sent["n"] == 0:
            sent["n"] = 1
            signal.raise_signal(signal.SIGTERM)
        return orig(query, variables)

    fake.graphql = graphql  # type: ignore[method-assign]
    old = signal.getsignal(signal.SIGTERM)
    kit.install_exit_signals()
    try:
        called, gone = kit.terminate_guarded(fake, "pod1", sleep=lambda _s: None)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert called and gone and fake.pods == {}
    assert "termination of pod pod1 in progress" in capfd.readouterr().err


def test_http_errors_in_fallback_are_unknown_not_gone() -> None:
    class Failing:
        def __init__(self, message: str) -> None:
            self.message = message

        def graphql(self, *_a: Any) -> dict[str, Any]:
            raise kit.KitError(self.message)

        def rest(self, *_a: Any) -> Any:
            raise AssertionError

    gone = kit._pod_gone  # pyright: ignore[reportPrivateUsage]
    assert (
        gone(Failing("RunPod HTTP 404 for https://api.runpod.io/graphql: Not Found"), "p1") is None
    )
    redirect = "RunPod request failed: KitError: refusing HTTP 302 redirect to /not-found"
    assert gone(Failing(redirect), "p1") is None
    assert gone(Failing("RunPod GraphQL error: pod not found"), "p1") is True


def test_single_absence_from_listing_needs_confirmation() -> None:
    gone = kit._pod_gone  # pyright: ignore[reportPrivateUsage]
    clock = FakeClock()
    fake = _fake_with_pod()
    fake.pods.clear()
    fake.pod_errors = 1  # pod(input) cannot confirm
    assert gone(fake, "pod1", clock=clock) is None
    clock.t += 5
    fake.pod_errors = 1
    assert gone(fake, "pod1", clock=clock) is None  # second absence too soon
    clock.t += 6
    fake.pod_errors = 1
    assert gone(fake, "pod1", clock=clock) is True  # absent twice >= 10 s apart
    assert gone(fake, "other", clock=clock) is True  # absent AND pod(input) null


def test_second_sigterm_after_the_first_is_a_no_op() -> None:
    old = signal.getsignal(signal.SIGTERM)
    kit.install_exit_signals()
    try:
        with pytest.raises(kit.TerminateRequestedError):
            signal.raise_signal(signal.SIGTERM)
        second_raised = False
        try:
            signal.raise_signal(signal.SIGTERM)  # latched: must not raise
        except kit.TerminateRequestedError:
            second_raised = True
        assert not second_raised, "a second SIGTERM must be a no-op (latch)"
    finally:
        signal.signal(signal.SIGTERM, old)


# ---------------------------------------------------------------------------
# Audit round 2, boot script: no foreground step may block PID 1's trap.
# ---------------------------------------------------------------------------

FAIL_FAST = {
    "MEMFUN_TERM_ATTEMPTS": "1",
    "MEMFUN_TERM_RETRY_SLEEP": "0",
    "MEMFUN_KILL_GRACE": "0",
}


@needs_bash
def test_boot_hung_python_version_cannot_block_pid1(tmp_path: Path) -> None:
    log, rp, _py = _run_boot(
        tmp_path,
        {"MEMFUN_DEADLINE_EPOCH": _soon(3), "FAKE_PY_VERSION_HANG": "1", **FAIL_FAST},
        runpodctl=_rp_script(tmp_path, "exit 1"),
        wait_bound=60,
    )
    assert rp == ["remove pod podB", "pod delete podB"]
    assert "container shutdown: stopping all processes and exiting PID 1" in log


@needs_bash
def test_boot_hung_import_probe_cannot_block_pid1(tmp_path: Path) -> None:
    log, _rp, _py = _run_boot(
        tmp_path,
        {
            "MEMFUN_DEADLINE_EPOCH": _soon(3),
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_IMPORT_HANG": "1",
            **FAIL_FAST,
        },
        runpodctl=_rp_script(tmp_path, "exit 1"),
        wait_bound=60,
    )
    assert "container shutdown: stopping all processes and exiting PID 1" in log


@needs_bash
def test_boot_runpodctl_success_but_container_alive_is_stopped(tmp_path: Path) -> None:
    _write_marker(tmp_path, "a" * 40)
    log, rp, py = _run_boot(
        tmp_path,
        {
            "MEMFUN_DEADLINE_EPOCH": _soon(3),
            "FAKE_SGLANG_SLEEP": "600",  # SGLang keeps serving
            "MEMFUN_REMOVE_WAIT_SEC": "1",
        },
        runpodctl=_rp_script(tmp_path, "echo ok; exit 0"),  # "succeeds", pod survives
        wait_bound=60,
    )
    assert "--enable-metrics" in py and rp == ["remove pod podB"]
    assert "runpodctl reported success but PID 1 still runs" in log
    assert "container shutdown: stopping all processes and exiting PID 1" in log


@needs_bash
def test_boot_stall_kills_whole_group_even_if_downloader_ignores_term(tmp_path: Path) -> None:
    log, _rp, _py = _run_boot(
        tmp_path,
        {
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_TRAP": "1",
            "MEMFUN_PROGRESS_SEC": "1",
            "MEMFUN_STALL_SEC": "2",
            "MEMFUN_PULL_RETRIES": "1",
            "MEMFUN_GROUP_KILL_WAIT": "1",
        },
        wait_bound=60,
    )
    assert "download stalled" in log
    assert BOOT_ORPHANS[str(tmp_path)] == []  # no stray writer survived


@needs_bash
def test_boot_stall_kill_without_pgrep_leaves_no_orphans(tmp_path: Path) -> None:
    restricted = tmp_path / "sysbin"
    restricted.mkdir()
    for d in ("/usr/bin", "/bin"):
        for f in Path(d).iterdir():
            if f.name not in ("pgrep", "pkill") and not (restricted / f.name).exists():
                with contextlib.suppress(OSError):
                    (restricted / f.name).symlink_to(f)
    log, _rp, _py = _run_boot(
        tmp_path,
        {
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_TRAP": "1",
            "MEMFUN_PROGRESS_SEC": "1",
            "MEMFUN_STALL_SEC": "2",
            "MEMFUN_PULL_RETRIES": "1",
            "MEMFUN_GROUP_KILL_WAIT": "1",
        },
        wait_bound=60,
        path=str(restricted),
    )
    assert "tool pgrep: MISSING" in log and "download stalled" in log
    assert BOOT_ORPHANS[str(tmp_path)] == []


@needs_bash
def test_boot_preallocated_file_is_not_a_stall(tmp_path: Path) -> None:
    log, _rp, py = _run_boot(
        tmp_path,
        {
            "MEMFUN_ALLOW_PULL": "1",
            "FAKE_HF_HUB": "1",
            "FAKE_SNAP_PREALLOC": "1",
            "MEMFUN_PROGRESS_SEC": "1",
            "MEMFUN_STALL_SEC": "2",
        },
        wait_bound=60,
    )
    assert "download stalled" not in log
    assert "download finished via python-api" in log and "--enable-metrics" in py


def test_conftest_guard_blocks_authenticated_requests_to_real_hosts() -> None:
    import urllib.request

    req = urllib.request.Request("https://api.runpod.io/graphql", method="GET")
    req.add_unredirected_header("Authorization", "Bearer x")
    with pytest.raises(RuntimeError, match="hermetic"):
        urllib.request.urlopen(req, timeout=1)
    with pytest.raises(RuntimeError, match="hermetic"):
        kit._auth_open(req, 1.0)  # pyright: ignore[reportPrivateUsage]


def test_sse_reader_marks_lines_over_1_mib_instead_of_splitting() -> None:
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

        def do_GET(self) -> None:
            big = "data: " + json.dumps(
                {"ts": "t", "source": "container", "line": "x" * (2 << 20)}
            )
            body = (big + "\n\n" + _ev("after") + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv, port = _start_server(H)
    try:
        t = kit.HttpTransport(
            kit.Secret(FAKE_KEY), logs_url=f"http://127.0.0.1:{port}/v2/pods/{{pod_id}}/logs"
        )
        lines = list(t.stream_logs("pod1", {}, None, 5.0))
    finally:
        srv.shutdown()
    assert all(len(line) < (1 << 20) + 16 for line in lines)
    out, _stats, _clock = _follow(FakeLogs(lines))
    assert any("over 1 MiB" in o for o in out) and "t [cont] after" in out
    assert not any(o.startswith("[raw]") for o in out)
