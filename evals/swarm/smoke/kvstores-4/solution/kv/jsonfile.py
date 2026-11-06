"""JsonFileStore: see task spec."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from kv.base import check_key, check_value


class JsonFileStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".kv-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)
        os.replace(tmp, self._path)

    def get(self, key: str) -> str | None:
        return self._load().get(check_key(key))

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[check_key(key)] = check_value(value)
        self._save(data)

    def delete(self, key: str) -> bool:
        data = self._load()
        if data.pop(check_key(key), None) is None:
            return False
        self._save(data)
        return True

    def keys(self) -> list[str]:
        return sorted(self._load())
