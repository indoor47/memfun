from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.errors import SessionNotFoundError
from memfun_core.types import Session, SessionConfig

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    import aiosqlite

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    history TEXT NOT NULL DEFAULT '[]',
    ttl_seconds INTEGER NOT NULL DEFAULT 3600
);
"""


class SQLiteSessionManager:
    """T1 session manager: SQLite-backed sessions."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, db_path: str) -> SQLiteSessionManager:
        conn = await get_connection(db_path)
        await conn.executescript(_CREATE_SESSIONS)
        await conn.commit()
        return cls(conn)

    async def create_session(self, user_id: str, config: SessionConfig | None = None) -> Session:
        config = config or SessionConfig()
        session_id = uuid.uuid4().hex
        now = time.time()
        await self._conn.execute(
            "INSERT INTO sessions"
            " (session_id, user_id, created_at, ttl_seconds)"
            " VALUES (?, ?, ?, ?)",
            (session_id, user_id, now, config.ttl_seconds),
        )
        await self._conn.commit()
        return Session(session_id=session_id, user_id=user_id, created_at=now)

    async def get_session(self, session_id: str) -> Session | None:
        async with self._conn.execute(
            "SELECT session_id, user_id, created_at,"
            " data, history, ttl_seconds"
            " FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            if (time.time() - row[2]) > row[5]:
                await self.end_session(session_id)
                return None
            return Session(
                session_id=row[0], user_id=row[1], created_at=row[2],
                data=json.loads(row[3]), history=json.loads(row[4]),
            )

    async def update_session(self, session_id: str, data: dict) -> None:
        session = await self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        merged = {**session.data, **data}
        await self._conn.execute(
            "UPDATE sessions SET data = ? WHERE session_id = ?",
            (json.dumps(merged), session_id),
        )
        await self._conn.commit()

    async def end_session(self, session_id: str) -> None:
        await self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._conn.commit()
