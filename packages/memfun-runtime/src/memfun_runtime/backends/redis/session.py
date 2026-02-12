from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.errors import SessionNotFoundError
from memfun_core.logging import get_logger
from memfun_core.types import Session, SessionConfig

from memfun_runtime.backends.redis._pool import create_pool

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("backend.redis.session")


class RedisSessionManager:
    """T2 session manager: Redis Hash for session data with TTL."""

    def __init__(self, client: Redis, *, prefix: str = "memfun:") -> None:
        self._r = client
        self._prefix = prefix

    @classmethod
    async def create(cls, redis_url: str, *, prefix: str = "memfun:") -> RedisSessionManager:
        client = await create_pool(redis_url)
        return cls(client, prefix=prefix)

    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}session:{session_id}"

    async def create_session(self, user_id: str, config: SessionConfig | None = None) -> Session:
        config = config or SessionConfig()
        session_id = uuid.uuid4().hex
        now = time.time()
        session = Session(session_id=session_id, user_id=user_id, created_at=now)

        payload = json.dumps({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "data": {},
            "history": [],
        })
        key = self._session_key(session_id)
        await self._r.set(key, payload.encode(), ex=config.ttl_seconds)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        raw = await self._r.get(self._session_key(session_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return Session(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=data["created_at"],
            data=data.get("data", {}),
            history=data.get("history", []),
        )

    async def update_session(self, session_id: str, data: dict) -> None:
        key = self._session_key(session_id)
        raw = await self._r.get(key)
        if raw is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")

        existing = json.loads(raw)
        existing.setdefault("data", {}).update(data)

        # Preserve remaining TTL
        ttl = await self._r.ttl(key)
        payload = json.dumps(existing).encode()
        if ttl and ttl > 0:
            await self._r.set(key, payload, ex=ttl)
        else:
            await self._r.set(key, payload)

    async def end_session(self, session_id: str) -> None:
        await self._r.delete(self._session_key(session_id))
