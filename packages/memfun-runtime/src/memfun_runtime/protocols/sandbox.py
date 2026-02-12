from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memfun_core.types import ExecutionResult, SandboxConfig, SandboxHandle


@runtime_checkable
class SandboxAdapter(Protocol):
    """Isolated code execution for RLM REPL loops."""

    async def execute(self, code: str, context: dict, timeout: int = 30) -> ExecutionResult: ...
    async def create_sandbox(self, config: SandboxConfig) -> SandboxHandle: ...
    async def destroy_sandbox(self, handle: SandboxHandle) -> None: ...
