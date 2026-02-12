from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfun_core.types import ExecutionResult, SandboxConfig, SandboxHandle


class StubSandbox:
    """Stub sandbox that raises NotImplementedError.

    Replaced by real sandbox backends in Phase 2.
    """

    async def execute(self, code: str, context: dict, timeout: int = 30) -> ExecutionResult:
        raise NotImplementedError("Sandbox not yet implemented. Coming in Phase 2.")

    async def create_sandbox(self, config: SandboxConfig) -> SandboxHandle:
        raise NotImplementedError("Sandbox not yet implemented. Coming in Phase 2.")

    async def destroy_sandbox(self, handle: SandboxHandle) -> None:
        pass
