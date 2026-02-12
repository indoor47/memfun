from __future__ import annotations

import time
from typing import TYPE_CHECKING

from memfun_core.errors import AgentAlreadyRunningError
from memfun_core.types import AgentInfo, AgentStatus, AgentStatusKind

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    import aiosqlite

_CREATE_LIFECYCLE = """
CREATE TABLE IF NOT EXISTS agent_lifecycle (
    agent_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    uptime_start REAL,
    last_heartbeat REAL NOT NULL,
    error TEXT
);
"""


class SQLiteLifecycle:
    """T1 lifecycle: SQLite-backed agent status tracking."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await get_connection(self._db_path)
            await self._conn.executescript(_CREATE_LIFECYCLE)
            await self._conn.commit()
        return self._conn

    async def start(self, agent_id: str) -> None:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT status FROM agent_lifecycle WHERE agent_id = ?", (agent_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] == "running":
                raise AgentAlreadyRunningError(f"Agent {agent_id!r} is already running")

        now = time.time()
        await conn.execute(
            """INSERT INTO agent_lifecycle (agent_id, status, uptime_start, last_heartbeat)
               VALUES (?, 'running', ?, ?)
               ON CONFLICT(agent_id) DO UPDATE SET
                   status = 'running', uptime_start = excluded.uptime_start,
                   last_heartbeat = excluded.last_heartbeat, error = NULL""",
            (agent_id, now, now),
        )
        await conn.commit()

    async def stop(self, agent_id: str) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """UPDATE agent_lifecycle SET status = 'stopped', last_heartbeat = ?
               WHERE agent_id = ?""",
            (time.time(), agent_id),
        )
        await conn.commit()

    async def restart(self, agent_id: str) -> None:
        await self.stop(agent_id)
        await self.start(agent_id)

    async def status(self, agent_id: str) -> AgentStatus:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT status, uptime_start, last_heartbeat, error"
            " FROM agent_lifecycle WHERE agent_id = ?",
            (agent_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return AgentStatus(
                    agent_id=agent_id, status=AgentStatusKind.UNKNOWN,
                    uptime_seconds=0.0, last_heartbeat=0.0,
                )
            uptime = time.time() - row[1] if row[1] and row[0] == "running" else 0.0
            return AgentStatus(
                agent_id=agent_id,
                status=AgentStatusKind(row[0]),
                uptime_seconds=uptime,
                last_heartbeat=row[2],
                error=row[3],
            )

    async def list_agents(self) -> list[AgentInfo]:
        conn = await self._ensure_conn()
        async with conn.execute("SELECT agent_id, status FROM agent_lifecycle") as cursor:
            rows = await cursor.fetchall()
            return [
                AgentInfo(agent_id=row[0], name=row[0], version="unknown", capabilities=[])
                for row in rows
            ]
