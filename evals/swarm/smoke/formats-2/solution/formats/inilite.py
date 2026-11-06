"""A strict INI subset.  See the task spec."""

from __future__ import annotations


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    section: str | None = None
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]") and len(line) > 2:
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"line {n}: expected [section] or key = value")
        if section is None:
            raise ValueError(f"line {n}: key outside any section")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"line {n}: empty key")
        data[section][key] = value.strip()
    return data


def dump_ini(data: dict[str, dict[str, str]]) -> str:
    blocks = []
    for section, items in data.items():
        lines = [f"[{section}]"] + [f"{k} = {v}" for k, v in items.items()]
        blocks.append("\n".join(lines) + "\n")
    return "\n".join(blocks)
