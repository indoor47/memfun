"""SqliteStore: see task spec."""

from __future__ import annotations

import os
import sqlite3

from kv.base import check_key, check_value


class SqliteStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._conn = sqlite3.connect(os.fspath(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (check_key(key),)).fetchone()
        return None if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (check_key(key), check_value(value)),
        )
        self._conn.commit()

    def delete(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM kv WHERE k = ?", (check_key(key),))
        self._conn.commit()
        return cur.rowcount > 0

    def keys(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT k FROM kv ORDER BY k")]

    def close(self) -> None:
        self._conn.close()
