from __future__ import annotations

import time
from typing import TYPE_CHECKING

from memfun_core.types import HealthState, HealthStatus

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import aiosqlite

_CREATE_HEALTH = """
CREATE TABLE IF NOT EXISTS health (
    agent_id TEXT PRIMARY KEY,
    last_heartbeat REAL NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}'
);
"""


class SQLiteHealthMonitor:
    """T1 health monitor: SQLite-backed heartbeat tracking."""

    def __init__(
        self,
        db_path: str,
        healthy_timeout: float = 30.0,
        degraded_timeout: float = 60.0,
    ) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._callbacks: list[Callable[[HealthStatus], Awaitable[None]]] = []
        self._healthy_timeout = healthy_timeout
        self._degraded_timeout = degraded_timeout

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await get_connection(self._db_path)
            await self._conn.executescript(_CREATE_HEALTH)
            await self._conn.commit()
        return self._conn

    async def heartbeat(self, agent_id: str, metrics: dict) -> None:
        import json
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO health (agent_id, last_heartbeat, metrics) VALUES (?, ?, ?)
               ON CONFLICT(agent_id) DO UPDATE SET
                   last_heartbeat = excluded.last_heartbeat, metrics = excluded.metrics""",
            (agent_id, time.time(), json.dumps({k: float(v) for k, v in metrics.items()})),
        )
        await conn.commit()

    async def check(self, agent_id: str) -> HealthStatus:
        import json
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT last_heartbeat, metrics FROM health WHERE agent_id = ?", (agent_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return HealthStatus(
                    agent_id=agent_id,
                    state=HealthState.UNKNOWN,
                    last_heartbeat=0.0,
                )
            elapsed = time.time() - row[0]
            if elapsed <= self._healthy_timeout:
                state = HealthState.HEALTHY
            elif elapsed <= self._degraded_timeout:
                state = HealthState.DEGRADED
            else:
                state = HealthState.UNHEALTHY
            return HealthStatus(
                agent_id=agent_id, state=state, last_heartbeat=row[0],
                metrics=json.loads(row[1]),
            )

    async def subscribe_alerts(self, callback: Callable[[HealthStatus], Awaitable[None]]) -> None:
        self._callbacks.append(callback)
