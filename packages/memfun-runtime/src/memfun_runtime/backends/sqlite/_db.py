from __future__ import annotations

from pathlib import Path

import aiosqlite


async def get_connection(db_path: str) -> aiosqlite.Connection:
    """Open a WAL-mode SQLite connection, creating the directory if needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = aiosqlite.Row
    return conn
