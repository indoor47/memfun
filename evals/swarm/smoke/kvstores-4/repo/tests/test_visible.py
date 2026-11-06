from kv.base import KVStore
from kv.memory import MemoryStore


def test_memory_roundtrip():
    s = MemoryStore()
    assert isinstance(s, KVStore)
    s.set("a", "1")
    assert s.get("a") == "1"
