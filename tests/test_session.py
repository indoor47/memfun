from __future__ import annotations

import pytest
from memfun_core.errors import SessionNotFoundError


@pytest.fixture(params=["memory", "sqlite", "redis", "nats"])
def session_mgr(request):
    backends = {
        "memory": "memory_session_manager",
        "sqlite": "sqlite_session_manager",
        "redis": "redis_session_manager",
        "nats": "nats_session_manager",
    }
    return request.getfixturevalue(backends[request.param])


class TestSessionConformance:
    """Conformance tests for SessionManager implementations."""

    async def test_create_session(self, session_mgr):
        session = await session_mgr.create_session("user-1")
        assert session.session_id
        assert session.user_id == "user-1"

    async def test_get_session(self, session_mgr):
        session = await session_mgr.create_session("user-1")
        retrieved = await session_mgr.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    async def test_get_nonexistent_session(self, session_mgr):
        assert await session_mgr.get_session("nonexistent") is None

    async def test_update_session(self, session_mgr):
        session = await session_mgr.create_session("user-1")
        await session_mgr.update_session(session.session_id, {"key": "value"})
        updated = await session_mgr.get_session(session.session_id)
        assert updated is not None
        assert updated.data.get("key") == "value"

    async def test_update_nonexistent_raises(self, session_mgr):
        with pytest.raises(SessionNotFoundError):
            await session_mgr.update_session("nonexistent", {"key": "value"})

    async def test_end_session(self, session_mgr):
        session = await session_mgr.create_session("user-1")
        await session_mgr.end_session(session.session_id)
        assert await session_mgr.get_session(session.session_id) is None

    async def test_end_nonexistent_session(self, session_mgr):
        await session_mgr.end_session("nonexistent")  # Should not raise
