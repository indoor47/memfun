from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING

import aiosqlite
from memfun_core.types import Message, TopicConfig

from memfun_runtime.backends.sqlite._db import get_connection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    payload BLOB NOT NULL,
    key TEXT,
    timestamp REAL NOT NULL,
    headers TEXT NOT NULL DEFAULT '{}',
    consumed_by TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_topic_ts ON events(topic, timestamp);
"""

_CREATE_TOPICS = """
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL
);
"""


class SQLiteEventBus:
    """T1 event bus: SQLite WAL-mode with polling-based subscribe."""

    def __init__(self, conn: aiosqlite.Connection, poll_interval: float = 0.1) -> None:
        self._conn = conn
        self._poll_interval = poll_interval

    @classmethod
    async def create(cls, db_path: str, poll_interval: float = 0.1) -> SQLiteEventBus:
        conn = await get_connection(db_path)
        await conn.executescript(_CREATE_EVENTS + _CREATE_TOPICS)
        await conn.commit()
        return cls(conn, poll_interval)

    async def publish(self, topic: str, message: bytes, key: str | None = None) -> str:
        msg_id = uuid.uuid4().hex
        await self._conn.execute(
            "INSERT INTO events (id, topic, payload, key, timestamp) VALUES (?, ?, ?, ?, ?)",
            (msg_id, topic, message, key, time.time()),
        )
        await self._conn.commit()
        return msg_id

    async def subscribe(self, topic: str, group: str | None = None) -> AsyncIterator[Message]:
        consumer_id = group or uuid.uuid4().hex
        last_ts = time.time()

        while True:
            async with self._conn.execute(
                """SELECT id, topic, payload, key, timestamp, headers
                   FROM events
                   WHERE topic = ? AND timestamp > ?
                   AND json_extract(consumed_by, ?) IS NULL
                   ORDER BY timestamp ASC LIMIT 50""",
                (topic, last_ts - 0.001, f'$."{consumer_id}"'),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                await asyncio.sleep(self._poll_interval)
                continue

            for row in rows:
                msg_id = row[0]
                # Mark as consumed
                await self._conn.execute(
                    """UPDATE events SET consumed_by = json_set(consumed_by, ?, 1) WHERE id = ?""",
                    (f'$."{consumer_id}"', msg_id),
                )
                await self._conn.commit()

                last_ts = row[4]
                yield Message(
                    id=msg_id,
                    topic=row[1],
                    payload=row[2],
                    key=row[3],
                    timestamp=row[4],
                    headers=json.loads(row[5]) if row[5] else {},
                )

    async def create_topic(self, topic: str, config: TopicConfig | None = None) -> None:
        config = config or TopicConfig()
        config_json = json.dumps({
            "retention_seconds": config.retention_seconds,
            "max_consumers": config.max_consumers,
            "max_message_bytes": config.max_message_bytes,
        })
        try:
            await self._conn.execute(
                "INSERT INTO topics (name, config) VALUES (?, ?)",
                (topic, config_json),
            )
            await self._conn.commit()
        except aiosqlite.IntegrityError as err:
            async with self._conn.execute(
                "SELECT config FROM topics WHERE name = ?", (topic,)
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0] != config_json:
                    raise ValueError(
                        f"Topic {topic!r} already exists with"
                        " different config"
                    ) from err

    async def delete_topic(self, topic: str) -> None:
        await self._conn.execute("DELETE FROM topics WHERE name = ?", (topic,))
        await self._conn.execute("DELETE FROM events WHERE topic = ?", (topic,))
        await self._conn.commit()

    async def ack(self, message_id: str) -> None:
        pass  # Consumed tracking handles this

    async def nack(self, message_id: str) -> None:
        # Reset consumed_by for this message to allow redelivery
        await self._conn.execute(
            "UPDATE events SET consumed_by = '{}' WHERE id = ?",
            (message_id,),
        )
        await self._conn.commit()
