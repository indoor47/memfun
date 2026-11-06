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


from kv.ttl import TTLStore


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class DictStore:
    """Minimal independent inner store so this unit's test does not need MemoryStore."""

    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, value):
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError("bad")
        self.d[key] = value

    def delete(self, key):
        return self.d.pop(key, None) is not None

    def keys(self):
        return sorted(self.d)


def test_contract_without_expiry():
    contract(TTLStore(DictStore(), 60, clock=Clock()))


def test_expiry():
    c = Clock()
    inner = DictStore()
    s = TTLStore(inner, 10, clock=c)
    s.set("a", "1")
    c.t += 9.99
    assert s.get("a") == "1"
    c.t += 0.01
    assert s.get("a") is None
    assert "a" not in inner.d


def test_reset_on_set_and_keys_and_delete():
    c = Clock()
    inner = DictStore()
    s = TTLStore(inner, 10, clock=c)
    s.set("a", "1")
    s.set("b", "2")
    c.t += 6
    s.set("a", "again")
    c.t += 6
    assert s.keys() == ["a"]
    assert s.delete("b") is False
    assert inner.keys() == ["a"]
    c.t += 10
    assert s.delete("a") is False
    assert inner.keys() == []
