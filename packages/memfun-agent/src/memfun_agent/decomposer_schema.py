"""Bounded JSON schema of the decomposer's plan (see ``constrained_json``).

Bounds make a runaway repetition loop inside one JSON string impossible
under grammar-constrained decoding: no string, list or the plan as a whole
can exceed these limits.  Fields mirror what ``decomposer._parse_sub_tasks``
and ``_parse_groups`` read; the fail-closed parse stays as the backstop.
"""

from __future__ import annotations

from typing import Any

MAX_SUB_TASKS = 30
MAX_ID_CHARS = 40
MAX_DESCRIPTION_CHARS = 2000
MAX_PATHS_PER_TASK = 40
MAX_PATH_CHARS = 300
MAX_DEPENDS_ON = 30
MAX_SHARED_SPEC_CHARS = 8000
MAX_GROUPS = 30
MAX_GROUP_SIZE = 30
# Same set as decomposer._VALID_AGENT_TYPES (kept in sync by a test).
AGENT_TYPES = (
    "file",
    "coder",
    "test",
    "review",
    "web_search",
    "web_fetch",
    "planner",
    "debug",
    "security",
)


def _str_list(max_items: int, max_chars: int) -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": max_items,
        "items": {"type": "string", "maxLength": max_chars},
    }


DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SUB_TASKS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": MAX_ID_CHARS},
                    "description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
                    "agent_type": {"type": "string", "enum": list(AGENT_TYPES)},
                    "inputs": _str_list(MAX_PATHS_PER_TASK, MAX_PATH_CHARS),
                    "outputs": _str_list(MAX_PATHS_PER_TASK, MAX_PATH_CHARS),
                    "depends_on": _str_list(MAX_DEPENDS_ON, MAX_ID_CHARS),
                    "max_iterations": {"type": "integer", "minimum": 3, "maximum": 25},
                },
                "required": [
                    "id",
                    "description",
                    "agent_type",
                    "inputs",
                    "outputs",
                    "depends_on",
                    "max_iterations",
                ],
                "additionalProperties": False,
            },
        },
        "shared_spec": {"type": "string", "maxLength": MAX_SHARED_SPEC_CHARS},
        "parallelism_groups": {
            "type": "array",
            "maxItems": MAX_GROUPS,
            "items": _str_list(MAX_GROUP_SIZE, MAX_ID_CHARS),
        },
    },
    "required": ["sub_tasks", "shared_spec", "parallelism_groups"],
    "additionalProperties": False,
}
