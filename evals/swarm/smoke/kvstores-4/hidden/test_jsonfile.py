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


import json

from kv.jsonfile import JsonFileStore


def test_contract(tmp_path):
    contract(JsonFileStore(tmp_path / "s.json"))


def test_persists_and_is_json(tmp_path):
    p = tmp_path / "sub" / "s.json"
    JsonFileStore(p).set("k", "v")
    assert JsonFileStore(p).get("k") == "v"
    assert json.loads(p.read_text()) == {"k": "v"}


def test_no_temp_files_left(tmp_path):
    s = JsonFileStore(tmp_path / "s.json")
    for i in range(5):
        s.set(f"k{i}", str(i))
    assert sorted(x.name for x in tmp_path.iterdir()) == ["s.json"]
