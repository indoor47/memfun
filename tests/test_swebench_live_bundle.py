"""The agent bundle must hold exactly the versions of its lock file."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evals.swebench_live import bundle_check as bc  # noqa: E402

LOCK = """
version = 1

[[package]]
name = "dspy"
version = "3.4.1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "litellm"
version = "1.103.10"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "typing-extensions"
version = "4.12.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "memfun-agent"
version = "0.2.0"
source = { editable = "packages/memfun-agent" }
"""


def test_the_locked_versions_leave_out_the_workspace_members() -> None:
    assert bc.locked_versions(LOCK) == {
        "dspy": "3.4.1",
        "litellm": "1.103.10",
        "typing-extensions": "4.12.0",
    }


def test_a_bundle_at_the_locked_versions_passes() -> None:
    installed = {
        "dspy": "3.4.1",
        "litellm": "1.103.10",
        "typing_extensions": "4.12.0",  # the same package, as the installer spells it
        "memfun-agent": "0.2.0",
        "pip-only-extra": "1.0",  # not locked: nothing to compare
    }
    assert bc.mismatches(installed, bc.locked_versions(LOCK)) == []


def test_a_rebuild_that_resolved_newer_versions_is_caught() -> None:
    """What happened on the second box: the same commit, a newer dspy and litellm."""
    installed = {"dspy": "3.5.0", "litellm": "1.108.4", "Typing.Extensions": "4.12.0"}
    assert bc.mismatches(installed, bc.locked_versions(LOCK)) == [
        "dspy: installed 3.5.0, locked 3.4.1",
        "litellm: installed 1.108.4, locked 1.103.10",
    ]


def test_dspy_and_litellm_must_be_there_and_locked() -> None:
    assert bc.mismatches({"dspy": "3.4.1"}, bc.locked_versions(LOCK)) == ["litellm: not installed"]
    assert bc.mismatches({"dspy": "3.4.1", "litellm": "1.0"}, {"dspy": "3.4.1"}) == [
        "litellm: not in the lock file"
    ]


def test_the_digest_names_the_installed_set() -> None:
    a = bc.freeze_digest({"dspy": "3.4.1", "Lite_LLM": "1.0"})
    assert a == bc.freeze_digest({"lite-llm": "1.0", "dspy": "3.4.1"})  # order and spelling
    assert a != bc.freeze_digest({"dspy": "3.4.2", "lite-llm": "1.0"}) and len(a) == 64


def test_the_repository_lock_file_names_both_packages() -> None:
    locked = bc.locked_versions((REPO_ROOT / "uv.lock").read_text("utf-8"))
    assert all(name in locked for name in bc.MUST_MATCH)
    assert not [name for name in locked if name.startswith("memfun")]
