import pytest


def contract(store):
    assert store.get("missing") is None
    store.set("b", "2")
    store.set("a", "1")
    store.set("a", "one")
    assert store.get("a") == "one"
    assert store.keys() == ["a", "b"]
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert store.keys() == ["b"]
    store.set("unicode", "\u00e9\u00e8 \u2603")
    assert store.get("unicode") == "\u00e9\u00e8 \u2603"
    for bad in ("", None, 3):
        with pytest.raises(ValueError):
            store.set(bad, "x")
    with pytest.raises(ValueError):
        store.set("k", 5)


import sqlite3

from kv.sqlite_store import SqliteStore


def test_contract(tmp_path):
    s = SqliteStore(tmp_path / "s.db")
    contract(s)
    s.close()


def test_persists(tmp_path):
    p = tmp_path / "s.db"
    s = SqliteStore(p)
    s.set("k", "v")
    s.close()
    s2 = SqliteStore(p)
    assert s2.get("k") == "v"
    s2.close()


def test_injection_safe(tmp_path):
    s = SqliteStore(tmp_path / "s.db")
    evil = "x'); DROP TABLE kv; --"
    s.set(evil, evil)
    assert s.get(evil) == evil
    assert s.keys() == [evil]
    s.close()


def test_close(tmp_path):
    s = SqliteStore(tmp_path / "s.db")
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.get("k")
