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


from kv.base import KVStore
from kv.jsonfile import JsonFileStore
from kv.memory import MemoryStore
from kv.sqlite_store import SqliteStore
from kv.ttl import TTLStore


def test_every_store_meets_contract(tmp_path):
    sq = SqliteStore(tmp_path / "s.db")
    stores = [
        MemoryStore(),
        JsonFileStore(tmp_path / "s.json"),
        sq,
        TTLStore(MemoryStore(), 60),
        TTLStore(JsonFileStore(tmp_path / "t.json"), 60),
    ]
    for s in stores:
        assert isinstance(s, KVStore)
        contract(s)
    sq.close()
