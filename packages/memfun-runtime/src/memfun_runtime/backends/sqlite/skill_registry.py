from __future__ import annotations

import json
from typing import TYPE_CHECKING

from memfun_runtime.backends.sqlite._db import get_connection
from memfun_runtime.protocols.skill_registry import SkillInfo

if TYPE_CHECKING:
    import aiosqlite

_CREATE_SKILLS = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    source_path TEXT NOT NULL,
    user_invocable INTEGER NOT NULL DEFAULT 1,
    model_invocable INTEGER NOT NULL DEFAULT 1,
    allowed_tools TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


class SQLiteSkillRegistry:
    """T1 skill registry: SQLite-backed skill discovery."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def create(cls, db_path: str) -> SQLiteSkillRegistry:
        conn = await get_connection(db_path)
        await conn.executescript(_CREATE_SKILLS)
        await conn.commit()
        return cls(conn)

    async def register_skill(self, skill: SkillInfo) -> None:
        await self._conn.execute(
            """INSERT INTO skills (
                   name, description, source_path,
                   user_invocable, model_invocable,
                   allowed_tools, metadata
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   description = excluded.description,
                   source_path = excluded.source_path,
                   user_invocable = excluded.user_invocable,
                   model_invocable = excluded.model_invocable,
                   allowed_tools = excluded.allowed_tools,
                   metadata = excluded.metadata""",
            (skill.name, skill.description, skill.source_path,
             int(skill.user_invocable), int(skill.model_invocable),
             json.dumps(skill.allowed_tools), json.dumps(skill.metadata)),
        )
        await self._conn.commit()

    async def deregister_skill(self, name: str) -> None:
        await self._conn.execute("DELETE FROM skills WHERE name = ?", (name,))
        await self._conn.commit()

    async def get_skill(self, name: str) -> SkillInfo | None:
        async with self._conn.execute("SELECT * FROM skills WHERE name = ?", (name,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return SkillInfo(
                name=row[0], description=row[1], source_path=row[2],
                user_invocable=bool(row[3]), model_invocable=bool(row[4]),
                allowed_tools=json.loads(row[5]), metadata=json.loads(row[6]),
            )

    async def list_skills(self) -> list[SkillInfo]:
        async with self._conn.execute("SELECT * FROM skills") as cursor:
            return [
                SkillInfo(
                    name=row[0], description=row[1], source_path=row[2],
                    user_invocable=bool(row[3]), model_invocable=bool(row[4]),
                    allowed_tools=json.loads(row[5]), metadata=json.loads(row[6]),
                )
                async for row in cursor
            ]

    async def search_skills(self, query: str) -> list[SkillInfo]:
        # Escape LIKE wildcards so user input is treated as
        # literal text, not as pattern characters.
        escaped = (
            query
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        async with self._conn.execute(
            "SELECT * FROM skills"
            " WHERE name LIKE ? ESCAPE '\\'"
            " OR description LIKE ? ESCAPE '\\'",
            (f"%{escaped}%", f"%{escaped}%"),
        ) as cursor:
            return [
                SkillInfo(
                    name=row[0], description=row[1], source_path=row[2],
                    user_invocable=bool(row[3]), model_invocable=bool(row[4]),
                    allowed_tools=json.loads(row[5]), metadata=json.loads(row[6]),
                )
                async for row in cursor
            ]
