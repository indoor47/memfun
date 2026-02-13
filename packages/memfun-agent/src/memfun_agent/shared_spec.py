"""SharedSpec: cross-agent shared specification for multi-agent workflows.

Provides a ``SharedSpec`` dataclass that all agents in a workflow can
read from and contribute to, and a ``SharedSpecStore`` that persists
it via a :class:`StateStoreAdapter`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from memfun_runtime.protocols.state_store import StateStoreAdapter

logger = get_logger("agent.shared_spec")


@dataclass(slots=True)
class SharedSpec:
    """A shared specification for a multi-agent workflow."""

    workflow_id: str
    spec_text: str = ""
    findings: list[str] = field(default_factory=list)
    interfaces: dict[str, str] = field(default_factory=dict)
    file_registry: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ── Serialisation ─────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "workflow_id": self.workflow_id,
            "spec_text": self.spec_text,
            "findings": self.findings,
            "interfaces": self.interfaces,
            "file_registry": self.file_registry,
            "created_at": self.created_at,
        })

    @classmethod
    def from_json(cls, raw: str | bytes) -> SharedSpec:
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        return cls(
            workflow_id=data["workflow_id"],
            spec_text=data.get("spec_text", ""),
            findings=data.get("findings", []),
            interfaces=data.get("interfaces", {}),
            file_registry=data.get("file_registry", {}),
            created_at=data.get("created_at", 0.0),
        )

    # ── Mutation helpers ──────────────────────────────────────

    def add_finding(self, agent_id: str, finding: str) -> None:
        """Record a discovery by an agent."""
        self.findings.append(f"[{agent_id}] {finding}")

    def register_file(self, path: str, task_id: str) -> None:
        """Register file ownership to detect conflicts."""
        self.file_registry[path] = task_id

    def get_conflict_files(self) -> list[str]:
        """Return files claimed by multiple tasks (duplicate owners)."""
        owners: dict[str, list[str]] = {}
        for path, task_id in self.file_registry.items():
            owners.setdefault(path, []).append(task_id)
        return [p for p, ids in owners.items() if len(set(ids)) > 1]

    # ── Context formatting ────────────────────────────────────

    def to_agent_context(self) -> str:
        """Format the spec as context for injection into agent queries."""
        parts: list[str] = ["=== SHARED SPECIFICATION ==="]
        if self.spec_text:
            parts.append(self.spec_text)

        if self.interfaces:
            parts.append("\n=== INTERFACE CONTRACTS ===")
            for name, contract in self.interfaces.items():
                parts.append(f"- {name}: {contract}")

        if self.findings:
            parts.append("\n=== DISCOVERED PATTERNS ===")
            for finding in self.findings[-15:]:
                parts.append(f"- {finding}")

        if self.file_registry:
            parts.append("\n=== FILE OWNERSHIP ===")
            for path, owner in sorted(self.file_registry.items()):
                parts.append(f"- {path} -> {owner}")

        return "\n".join(parts)


class SharedSpecStore:
    """Manages :class:`SharedSpec` persistence via a :class:`StateStoreAdapter`."""

    _PREFIX = "memfun:workflow:spec:"

    def __init__(self, state_store: StateStoreAdapter) -> None:
        self._store = state_store

    async def save(self, spec: SharedSpec) -> None:
        key = f"{self._PREFIX}{spec.workflow_id}"
        await self._store.set(key, spec.to_json().encode())

    async def load(self, workflow_id: str) -> SharedSpec | None:
        key = f"{self._PREFIX}{workflow_id}"
        raw = await self._store.get(key)
        if raw is None:
            return None
        return SharedSpec.from_json(raw)

    async def append_finding(
        self, workflow_id: str, agent_id: str, finding: str,
    ) -> None:
        """Append a finding to a workflow's shared spec."""
        spec = await self.load(workflow_id)
        if spec is not None:
            spec.add_finding(agent_id, finding)
            await self.save(spec)
