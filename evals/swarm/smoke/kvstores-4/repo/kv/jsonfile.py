"""JsonFileStore: see task spec."""

from __future__ import annotations

import os


class JsonFileStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError
