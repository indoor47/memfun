"""Hermetic tests for the RunPod launch kit (scripts/runpod).  No network, no real pods.

Every transport here is fake.  The dry-run tests use a spy that raises on any mutation, so
a regression that lets dry-run mutate fails loudly instead of renting a GPU.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

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

    def _mutate(self, what: str) -> None:
        if not self.allow_mutations:
            raise SpyError(f"spy transport saw a mutation: {what}")
        self.mutations.append(what)

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append(("graphql", query))
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
                return {"podFindAndDeployOnDemand": self.pods[pod_id]}
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
        if "myself" in query:
            return {"myself": {"clientBalance": 50.0, "networkVolumes": self.volumes}}
        if "pod(input" in query:
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

    monkeypatch.setattr(kit.urllib.request, "urlopen", boom)
    transport = kit.HttpTransport(secret)
    with pytest.raises(kit.KitError) as exc:
        transport.graphql("query { gpuTypes { id } }")
    assert FAKE_KEY not in str(exc.value) and "***" in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__

    rc = kit.main(["gpus"], env={}, key_path=key_file)  # real HttpTransport, patched urlopen
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
    assert "[llm.roles.worker]" in out
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
        "MEMFUN_MODEL_DIR": str(tmp_path / "no-weights"),
        "MEMFUN_RUN_DIR": str(tmp_path / "run"),
        "RUNPOD_POD_ID": "podXYZ",
        "RUNPOD_API_KEY": "pod-scoped-injected",
        "PUBLIC_KEY": "",
    }
    proc = subprocess.run(
        ["bash", str(kit.BOOT_SCRIPT)], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = record.read_text().splitlines()
    assert lines and all(line == "pod delete podXYZ key=pod-scoped-injected" for line in lines)
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
    assert system.kwargs[0] == {"start_new_session": True}
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
