from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio


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
