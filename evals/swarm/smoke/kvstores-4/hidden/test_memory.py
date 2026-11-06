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
from kv.memory import MemoryStore


def test_contract():
    s = MemoryStore()
    assert isinstance(s, KVStore)
    contract(s)


def test_instances_independent():
    a, b = MemoryStore(), MemoryStore()
    a.set("x", "1")
    assert b.get("x") is None
