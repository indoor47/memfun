"""Execution trace collection for RLM agent runs.

Traces capture the full trajectory of an RLM execution: every code step,
its output, the final answer, timing, and token usage. These traces are
persisted via the StateStoreAdapter for later analysis, optimization
(MIPROv2/GEPA), and agent synthesis.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from memfun_runtime.protocols.state_store import StateStoreAdapter

logger = get_logger("agent.traces")

# Key prefix for trace storage in the state store
_TRACE_PREFIX = "memfun:traces:"


# ── Data Structures ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TraceStep:
    """A single step in an RLM execution trajectory."""

    iteration: int
    reasoning: str
    code: str
    output: str
    output_truncated: bool = False
    duration_ms: float = 0.0
    cumulative_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token consumption for an execution."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    sub_lm_calls: int = 0
    sub_lm_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Complete trace of an RLM agent execution.

    Captures everything needed to replay, analyze, and optimize
    agent behavior.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_type: str = ""
    query: str = ""
    context_length: int = 0
    trajectory: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the trace to a JSON string."""
        data = asdict(self)
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, raw: str | bytes) -> ExecutionTrace:
        """Deserialize a trace from a JSON string."""
        data = json.loads(raw)
        # Reconstruct nested dataclasses
        data["trajectory"] = [
            TraceStep(**step) for step in data.get("trajectory", [])
        ]
        data["token_usage"] = TokenUsage(
            **data.get("token_usage", {})
        )
        return cls(**data)


# ── Trace Collector ────────────────────────────────────────────


class TraceCollector:
    """Collects and persists execution traces via StateStoreAdapter.

    Usage::

        collector = TraceCollector(state_store)
        trace = ExecutionTrace(task_type="analyze", query="...", ...)
        await collector.save(trace)

        # Later: retrieve for analysis
        trace = await collector.load(trace_id)
        all_ids = [tid async for tid in collector.list_traces()]
    """

    def __init__(
        self,
        state_store: StateStoreAdapter | None = None,
    ) -> None:
        self._state_store = state_store
        self._in_memory: dict[str, ExecutionTrace] = {}

    @property
    def has_backend(self) -> bool:
        """Whether a persistent backend is available."""
        return self._state_store is not None

    async def save(self, trace: ExecutionTrace) -> str:
        """Persist a trace. Returns the trace_id.

        If no state store is available, traces are kept in memory.
        """
        key = f"{_TRACE_PREFIX}{trace.trace_id}"
        serialized = trace.to_json()

        if self._state_store is not None:
            await self._state_store.set(
                key, serialized.encode("utf-8")
            )
            logger.debug(
                "Saved trace %s (%d steps)",
                trace.trace_id,
                len(trace.trajectory),
            )
        else:
            self._in_memory[trace.trace_id] = trace
            logger.debug(
                "Saved trace %s in-memory (%d steps)",
                trace.trace_id,
                len(trace.trajectory),
            )

        return trace.trace_id

    async def load(self, trace_id: str) -> ExecutionTrace | None:
        """Load a trace by ID. Returns None if not found."""
        if self._state_store is not None:
            key = f"{_TRACE_PREFIX}{trace_id}"
            raw = await self._state_store.get(key)
            if raw is None:
                return None
            return ExecutionTrace.from_json(raw)

        return self._in_memory.get(trace_id)

    async def list_trace_ids(
        self, limit: int = 100
    ) -> list[str]:
        """List stored trace IDs (most recent first).

        Returns up to ``limit`` trace IDs.
        """
        if self._state_store is not None:
            ids: list[str] = []
            async for key in self._state_store.list_keys(
                _TRACE_PREFIX
            ):
                tid = key.removeprefix(_TRACE_PREFIX)
                ids.append(tid)
                if len(ids) >= limit:
                    break
            return ids

        return list(self._in_memory.keys())[:limit]

    async def delete(self, trace_id: str) -> bool:
        """Delete a trace by ID. Returns True if it existed."""
        if self._state_store is not None:
            key = f"{_TRACE_PREFIX}{trace_id}"
            exists = await self._state_store.exists(key)
            if exists:
                await self._state_store.delete(key)
            return exists

        return self._in_memory.pop(trace_id, None) is not None
