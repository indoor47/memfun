from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@pytest.fixture(autouse=True)
def _no_authenticated_requests_to_real_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic guard for every test: any urllib request (urlopen or a custom opener such
    as the RunPod kit's) that carries an Authorization header may only reach localhost."""
    import urllib.parse
    import urllib.request

    real_open = urllib.request.OpenerDirector.open

    def guarded(self: urllib.request.OpenerDirector, fullurl, data=None, timeout=None):  # type: ignore[no-untyped-def]
        req = fullurl if isinstance(fullurl, urllib.request.Request) else None
        if req is not None and req.has_header("Authorization"):
            host = urllib.parse.urlsplit(req.full_url).hostname or ""
            if host not in _LOCAL_HOSTS:
                raise RuntimeError(f"hermetic test guard: authenticated request to {host}")
        if timeout is None:
            return real_open(self, fullurl, data)
        return real_open(self, fullurl, data, timeout)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", guarded)


class HermeticNetworkError(OSError):
    """A test tried to reach a non-loopback host."""


def _loopback(host: object) -> bool:
    import ipaddress

    if not isinstance(host, str):
        return False
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_non_loopback_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Socket-level hermetic guard for every test: DNS lookups and connections are only
    allowed to loopback (127.0.0.0/8, ::1, localhost) and unix sockets."""
    import socket

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *a, **k):  # type: ignore[no-untyped-def]
        if host is None or _loopback(host.decode() if isinstance(host, bytes) else host):
            return real_getaddrinfo(host, *a, **k)
        raise HermeticNetworkError(f"hermetic test guard: DNS lookup of {host!r}")

    def check(address: object) -> None:
        if isinstance(address, (str, bytes)):  # AF_UNIX
            return
        if isinstance(address, tuple) and address and _loopback(address[0]):
            return
        raise HermeticNetworkError(f"hermetic test guard: connect to {address!r}")

    def connect(self, address):  # type: ignore[no-untyped-def]
        check(address)
        return real_connect(self, address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def memory_event_bus():
    from memfun_runtime.backends.memory import InProcessEventBus

    return InProcessEventBus()


@pytest_asyncio.fixture
async def memory_state_store():
    from memfun_runtime.backends.memory import InProcessStateStore

    return InProcessStateStore()


@pytest_asyncio.fixture
async def memory_registry():
    from memfun_runtime.backends.memory import InProcessRegistry

    return InProcessRegistry()


@pytest_asyncio.fixture
async def memory_session_manager():
    from memfun_runtime.backends.memory import InProcessSessionManager

    return InProcessSessionManager()


@pytest_asyncio.fixture
async def memory_health_monitor():
    from memfun_runtime.backends.memory import InProcessHealthMonitor

    return InProcessHealthMonitor()


@pytest_asyncio.fixture
async def memory_lifecycle():
    from memfun_runtime.backends.memory import InProcessLifecycle

    return InProcessLifecycle()


@pytest_asyncio.fixture
async def memory_skill_registry():
    from memfun_runtime.backends.memory import InProcessSkillRegistry

    return InProcessSkillRegistry()


@pytest_asyncio.fixture
async def sqlite_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test.db")


@pytest_asyncio.fixture
async def sqlite_event_bus(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteEventBus

    return await SQLiteEventBus.create(sqlite_db_path, poll_interval=0.02)


@pytest_asyncio.fixture
async def sqlite_state_store(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteStateStore

    return await SQLiteStateStore.create(sqlite_db_path)


@pytest_asyncio.fixture
async def sqlite_registry(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteRegistry

    return await SQLiteRegistry.create(sqlite_db_path)


@pytest_asyncio.fixture
async def sqlite_session_manager(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteSessionManager

    return await SQLiteSessionManager.create(sqlite_db_path)


@pytest_asyncio.fixture
async def sqlite_health_monitor(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteHealthMonitor

    return SQLiteHealthMonitor(sqlite_db_path)


@pytest_asyncio.fixture
async def sqlite_lifecycle(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteLifecycle

    return SQLiteLifecycle(sqlite_db_path)


@pytest_asyncio.fixture
async def sqlite_skill_registry(sqlite_db_path):
    from memfun_runtime.backends.sqlite import SQLiteSkillRegistry

    return await SQLiteSkillRegistry.create(sqlite_db_path)


# ---------------------------------------------------------------------------
# T2 Redis fixtures
# ---------------------------------------------------------------------------


def _redis_prefix() -> str:
    return f"test_{uuid4().hex[:8]}:"


@pytest_asyncio.fixture
async def redis_event_bus():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisEventBus

    prefix = _redis_prefix()
    try:
        bus = await RedisEventBus.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield bus
    # cleanup: delete keys matching our test prefix
    try:
        async for key in bus._r.scan_iter(match=f"{prefix}*"):
            await bus._r.delete(key)
    except Exception:
        pass
    await bus._r.aclose()


@pytest_asyncio.fixture
async def redis_state_store():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisStateStore

    prefix = _redis_prefix()
    try:
        store = await RedisStateStore.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield store
    try:
        async for key in store._r.scan_iter(match=f"{prefix}*"):
            await store._r.delete(key)
    except Exception:
        pass
    await store._r.aclose()


@pytest_asyncio.fixture
async def redis_registry():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisRegistry

    prefix = _redis_prefix()
    try:
        reg = await RedisRegistry.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield reg
    try:
        async for key in reg._r.scan_iter(match=f"{prefix}*"):
            await reg._r.delete(key)
    except Exception:
        pass
    await reg._r.aclose()


@pytest_asyncio.fixture
async def redis_session_manager():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisSessionManager

    prefix = _redis_prefix()
    try:
        mgr = await RedisSessionManager.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield mgr
    try:
        async for key in mgr._r.scan_iter(match=f"{prefix}*"):
            await mgr._r.delete(key)
    except Exception:
        pass
    await mgr._r.aclose()


@pytest_asyncio.fixture
async def redis_health_monitor():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisHealthMonitor

    prefix = _redis_prefix()
    try:
        mon = await RedisHealthMonitor.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield mon
    try:
        async for key in mon._r.scan_iter(match=f"{prefix}*"):
            await mon._r.delete(key)
    except Exception:
        pass
    await mon._r.aclose()


@pytest_asyncio.fixture
async def redis_lifecycle():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisLifecycle

    prefix = _redis_prefix()
    try:
        lc = await RedisLifecycle.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield lc
    try:
        async for key in lc._r.scan_iter(match=f"{prefix}*"):
            await lc._r.delete(key)
    except Exception:
        pass
    await lc._r.aclose()


@pytest_asyncio.fixture
async def redis_skill_registry():
    pytest.importorskip("redis")
    from memfun_runtime.backends.redis import RedisSkillRegistry

    prefix = _redis_prefix()
    try:
        reg = await RedisSkillRegistry.create("redis://localhost:6379", prefix=prefix)
    except Exception:
        pytest.skip("Redis not available")
    yield reg
    try:
        async for key in reg._r.scan_iter(match=f"{prefix}*"):
            await reg._r.delete(key)
    except Exception:
        pass
    await reg._r.aclose()


# ---------------------------------------------------------------------------
# T3 NATS fixtures
# ---------------------------------------------------------------------------


def _nats_bucket_suffix() -> str:
    return uuid4().hex[:8]


@pytest_asyncio.fixture
async def nats_event_bus():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSEventBus

    stream_prefix = f"test_{_nats_bucket_suffix()}"
    try:
        bus = await NATSEventBus.create("nats://localhost:4222", stream_prefix=stream_prefix)
    except Exception:
        pytest.skip("NATS not available")
    yield bus
    # cleanup: drain and close
    with contextlib.suppress(Exception):
        await bus._nc.close()


@pytest_asyncio.fixture
async def nats_state_store():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSStateStore

    bucket = f"test_state_{_nats_bucket_suffix()}"
    try:
        store = await NATSStateStore.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield store
    with contextlib.suppress(Exception):
        await store._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await store._nc.close()


@pytest_asyncio.fixture
async def nats_registry():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSRegistry

    bucket = f"test_registry_{_nats_bucket_suffix()}"
    try:
        reg = await NATSRegistry.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield reg
    with contextlib.suppress(Exception):
        await reg._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await reg._nc.close()


@pytest_asyncio.fixture
async def nats_session_manager():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSSessionManager

    bucket = f"test_sessions_{_nats_bucket_suffix()}"
    try:
        mgr = await NATSSessionManager.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield mgr
    with contextlib.suppress(Exception):
        await mgr._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await mgr._nc.close()


@pytest_asyncio.fixture
async def nats_health_monitor():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSHealthMonitor

    bucket = f"test_health_{_nats_bucket_suffix()}"
    try:
        mon = await NATSHealthMonitor.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield mon
    with contextlib.suppress(Exception):
        await mon._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await mon._nc.close()


@pytest_asyncio.fixture
async def nats_lifecycle():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSLifecycle

    bucket = f"test_lifecycle_{_nats_bucket_suffix()}"
    try:
        lc = await NATSLifecycle.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield lc
    with contextlib.suppress(Exception):
        await lc._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await lc._nc.close()


@pytest_asyncio.fixture
async def nats_skill_registry():
    pytest.importorskip("nats")
    from memfun_runtime.backends.nats import NATSSkillRegistry

    bucket = f"test_skills_{_nats_bucket_suffix()}"
    try:
        reg = await NATSSkillRegistry.create("nats://localhost:4222", bucket=bucket)
    except Exception:
        pytest.skip("NATS not available")
    yield reg
    with contextlib.suppress(Exception):
        await reg._js.delete_key_value(bucket)
    with contextlib.suppress(Exception):
        await reg._nc.close()


# ── Reviewer nonce (fail-closed auditors) ─────────────────────
#
# Every reviewer dispatch carries a fresh ``review_id`` nonce that the
# reviewer must echo in its findings JSON.  Integration tests with
# canned reviewer answers opt into this fixture to pin the nonce to
# ``FIXED_REVIEW_ID`` so the canned answers can embed it.  Nonce
# mechanics themselves (fresh per dispatch/retry, stale nonce
# rejected) are tested in ``test_fail_closed_auditors.py`` with real
# nonces.

FIXED_REVIEW_ID = "test-review-id"


@pytest.fixture
def fixed_review_id(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("memfun_agent.workflow._new_review_id", lambda: FIXED_REVIEW_ID)
    return FIXED_REVIEW_ID
