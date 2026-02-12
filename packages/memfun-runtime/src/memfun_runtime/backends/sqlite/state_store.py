from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from memfun_core.types import StateChange, StateChangeKind

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiosqlite

_CREATE_STATE = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    expires_at REAL,
    updated_at REAL NOT NULL
);
"""


class SQLiteStateStore:
    """T1 state store: SQLite KV table with TTL."""

    def __init__(self, conn: aiosqlite.Connection, poll_interval: float = 0.5) -> None:
        self._conn = conn
        self._poll_interval = poll_interval
        self._revision = 0

    @classmethod
    async def create(cls, db_path: str) -> SQLiteStateStore:
        conn = await get_connection(db_path)
        await conn.executescript(_CREATE_STATE)
        await conn.commit()
        return cls(conn)

    async def _cleanup_expired(self) -> None:
        now = time.time()
        await self._conn.execute(
            "DELETE FROM state WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        await self._conn.commit()

    async def get(self, key: str) -> bytes | None:
        await self._cleanup_expired()
        async with self._conn.execute(
            "SELECT value FROM state WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (key, time.time()),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        now = time.time()
        expires_at = now + ttl if ttl else None
        self._revision += 1
        await self._conn.execute(
            """INSERT INTO state (key, value, revision, expires_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   revision = excluded.revision,
                   expires_at = excluded.expires_at,
                   updated_at = excluded.updated_at""",
            (key, value, self._revision, expires_at, now),
        )
        await self._conn.commit()

    async def delete(self, key: str) -> None:
        await self._conn.execute("DELETE FROM state WHERE key = ?", (key,))
        await self._conn.commit()

    async def exists(self, key: str) -> bool:
        async with self._conn.execute(
            "SELECT 1 FROM state WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (key, time.time()),
        ) as cursor:
            return (await cursor.fetchone()) is not None

    async def list_keys(self, prefix: str) -> AsyncIterator[str]:
        await self._cleanup_expired()
        # Escape LIKE wildcards in the prefix so that
        # characters like % and _ are matched literally.
        escaped = (
            prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        async with self._conn.execute(
            "SELECT key FROM state"
            " WHERE key LIKE ? ESCAPE '\\'"
            " AND (expires_at IS NULL OR expires_at > ?)",
            (escaped + "%", time.time()),
        ) as cursor:
            async for row in cursor:
                yield row[0]

    async def watch(self, key: str) -> AsyncIterator[StateChange]:
        last_revision = 0
        while True:
            async with self._conn.execute(
                "SELECT value, revision FROM state WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()

            if row and row[1] > last_revision:
                last_revision = row[1]
                yield StateChange(
                    key=key,
                    kind=StateChangeKind.PUT,
                    value=row[0],
                    revision=row[1],
                )
            elif not row and last_revision > 0:
                last_revision = 0
                yield StateChange(
                    key=key,
                    kind=StateChangeKind.DELETE,
                    value=None,
                    revision=0,
                )

            await asyncio.sleep(self._poll_interval)
