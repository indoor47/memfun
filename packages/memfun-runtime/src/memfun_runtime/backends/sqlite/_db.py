from __future__ import annotations

import contextlib
from pathlib import Path

import aiosqlite

#: Every connection this process opened.  aiosqlite runs each one on a NON-daemon worker thread,
#: so a connection nobody closes keeps the interpreter alive after the work is done (a finished
#: ``memfun swarm`` sat idle until its timeout).  :func:`close_all_connections` ends them.
_OPEN: list[aiosqlite.Connection] = []


async def get_connection(db_path: str) -> aiosqlite.Connection:
    """Open a WAL-mode SQLite connection, creating the directory if needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = aiosqlite.Row
    _OPEN.append(conn)
    return conn


async def close_all_connections(timeout: float = 5.0) -> int:
    """Close every connection :func:`get_connection` opened; returns how many were open.

    Best effort and idempotent: a connection that fails to close within *timeout* is skipped.
    Call it from the event loop that opened them, once the work is finished.
    """
    import asyncio

    conns, _OPEN[:] = list(_OPEN), []
    for conn in conns:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(conn.close(), timeout)
    return len(conns)
