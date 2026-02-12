from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.errors import SessionNotFoundError
from memfun_core.logging import get_logger
from memfun_core.types import Session, SessionConfig

from memfun_runtime.backends.nats._connection import connect, ensure_kv_bucket

if TYPE_CHECKING:
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats.session")


class NATSSessionManager:
    """T3 session manager: NATS KV for sessions."""

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        kv: object,
    ) -> None:
        self._nc = nc
        self._js = js
        self._kv = kv

    @classmethod
    async def create(
        cls,
        nats_url: str,
        *,
        creds_file: str | None = None,
        bucket: str = "memfun_sessions",
        default_ttl: int = 3600,
    ) -> NATSSessionManager:
        nc, js = await connect(nats_url, creds_file=creds_file)
        # Use bucket-level TTL for session expiration.
        kv = await ensure_kv_bucket(js, bucket, ttl=default_ttl * 1_000_000_000)
        return cls(nc, js, kv)

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
            "ttl_seconds": config.ttl_seconds,
        })
        await self._kv.put(session_id, payload.encode())  # type: ignore[union-attr]
        return session

    async def get_session(self, session_id: str) -> Session | None:
        try:
            entry = await self._kv.get(session_id)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                return None
        except Exception:
            return None

        data = json.loads(entry.value)

        # Check application-level TTL.
        ttl_seconds = data.get("ttl_seconds", 3600)
        if (time.time() - data["created_at"]) > ttl_seconds:
            await self.end_session(session_id)
            return None

        return Session(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=data["created_at"],
            data=data.get("data", {}),
            history=data.get("history", []),
        )

    async def update_session(self, session_id: str, data: dict) -> None:
        try:
            entry = await self._kv.get(session_id)  # type: ignore[union-attr]
            if entry is None or entry.value is None:
                raise SessionNotFoundError(f"Session {session_id!r} not found")
        except SessionNotFoundError:
            raise
        except Exception as exc:
            raise SessionNotFoundError(f"Session {session_id!r} not found") from exc

        existing = json.loads(entry.value)
        existing.setdefault("data", {}).update(data)
        await self._kv.put(session_id, json.dumps(existing).encode())  # type: ignore[union-attr]

    async def end_session(self, session_id: str) -> None:
        with contextlib.suppress(Exception):
            await self._kv.delete(session_id)  # type: ignore[union-attr]
