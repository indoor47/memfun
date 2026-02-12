from __future__ import annotations

import json
from typing import TYPE_CHECKING

from memfun_core.types import AgentInfo

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    import aiosqlite

_CREATE_REGISTRY = """
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    capabilities TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    endpoint TEXT
);
"""


class SQLiteRegistry:
    """T1 registry: SQLite-backed agent discovery."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, db_path: str) -> SQLiteRegistry:
        conn = await get_connection(db_path)
        await conn.executescript(_CREATE_REGISTRY)
        await conn.commit()
        return cls(conn)

    async def register(self, agent_id: str, capabilities: list[str], metadata: dict) -> None:
        await self._conn.execute(
            """INSERT INTO agent_registry (agent_id, name, version, capabilities, metadata)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(agent_id) DO UPDATE SET
                   name = excluded.name, version = excluded.version,
                   capabilities = excluded.capabilities, metadata = excluded.metadata""",
            (
                agent_id,
                metadata.get("name", agent_id),
                metadata.get("version", "0.0.0"),
                json.dumps(capabilities),
                json.dumps({k: str(v) for k, v in metadata.items()}),
            ),
        )
        await self._conn.commit()

    async def deregister(self, agent_id: str) -> None:
        await self._conn.execute("DELETE FROM agent_registry WHERE agent_id = ?", (agent_id,))
        await self._conn.commit()

    async def discover(self, capability: str) -> list[AgentInfo]:
        async with self._conn.execute("SELECT * FROM agent_registry") as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                caps = json.loads(row[3])
                if capability in caps:
                    results.append(AgentInfo(
                        agent_id=row[0], name=row[1], version=row[2],
                        capabilities=caps, metadata=json.loads(row[4]),
                        endpoint=row[5],
                    ))
            return results

    async def get(self, agent_id: str) -> AgentInfo | None:
        async with self._conn.execute(
            "SELECT * FROM agent_registry WHERE agent_id = ?", (agent_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return AgentInfo(
                agent_id=row[0], name=row[1], version=row[2],
                capabilities=json.loads(row[3]), metadata=json.loads(row[4]),
                endpoint=row[5],
            )
