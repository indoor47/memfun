from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import ExecutionResult, SandboxHandle

if TYPE_CHECKING:
    from memfun_core.types import SandboxConfig

logger = get_logger("sandbox.modal")

_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB
_MAX_TIMEOUT_SECONDS = 300      # 5 minutes hard cap


def _build_preamble(context: dict) -> str:
    """Build a Python preamble that injects *context* as local variables."""
    if not context:
        return ""
    ctx_json = json.dumps(context, default=str)
    lines = [
        "import json as __ctx_json",
        f"__ctx_data = __ctx_json.loads({ctx_json!r})",
        "globals().update(__ctx_data)",
        "del __ctx_json, __ctx_data",
    ]
    return "\n".join(lines) + "\n"


def _modal_available() -> bool:
    """Check whether the ``modal`` package is importable."""
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        return False


class ModalSandbox:
    """Execute Python code inside Modal cloud sandboxes.

    Implements the ``SandboxAdapter`` protocol from
    ``memfun_runtime.protocols.sandbox``.

    Requires the ``modal`` package (``pip install modal``) and valid
    Modal authentication (``modal token set`` or ``MODAL_TOKEN_ID`` /
    ``MODAL_TOKEN_SECRET`` environment variables).
    """

    def __init__(self, *, app_name: str = "memfun-sandbox") -> None:
        self._app_name = app_name
        # handle_id -> (config, modal Sandbox object)
        self._sandboxes: dict[str, tuple[SandboxConfig, object]] = {}

    # ── Protocol methods ────────────────────────────────────────────

    async def create_sandbox(self, config: SandboxConfig) -> SandboxHandle:
        if not _modal_available():
            raise RuntimeError(
                "The 'modal' package is not installed. Install it with "
                "'pip install modal' to use ModalSandbox, or switch to "
                "LocalSandbox."
            )

        handle = SandboxHandle(
            id=uuid.uuid4().hex,
            backend="modal",
        )

        sb = await self._create_modal_sandbox(config)
        self._sandboxes[handle.id] = (config, sb)

        logger.info("Created modal sandbox %s", handle.id[:8])
        return handle

    async def execute(
        self,
        code: str,
        context: dict,
        timeout: int = 30,
    ) -> ExecutionResult:
        timeout = max(1, min(timeout, _MAX_TIMEOUT_SECONDS))

        # If there is exactly one active sandbox, reuse it.
        config: SandboxConfig | None = None
        sb: object | None = None

        if len(self._sandboxes) == 1:
            only_id = next(iter(self._sandboxes))
            config, sb = self._sandboxes[only_id]
            timeout = min(timeout, config.timeout_seconds)

        if sb is None:
            # Ad-hoc execution: create a temporary sandbox, run, destroy.
            from memfun_core.types import SandboxConfig as SandboxCfg

            config = SandboxCfg(timeout_seconds=timeout)
            tmp_handle = await self.create_sandbox(config)
            _, sb = self._sandboxes[tmp_handle.id]
            try:
                return await self._run(code, context, sb, timeout)
            finally:
                await self.destroy_sandbox(tmp_handle)

        return await self._run(code, context, sb, timeout)

    async def destroy_sandbox(self, handle: SandboxHandle) -> None:
        entry = self._sandboxes.pop(handle.id, None)
        if entry is None:
            return
        _, sb = entry
        await self._terminate_modal_sandbox(sb)
        logger.info("Destroyed modal sandbox %s", handle.id[:8])

    # ── Modal helpers ───────────────────────────────────────────────

    async def _create_modal_sandbox(self, config: SandboxConfig) -> object:
        """Create a Modal Sandbox and return the sandbox object."""
        import asyncio

        import modal

        image = modal.Image.debian_slim(python_version="3.12")

        sb = await asyncio.to_thread(
            modal.Sandbox.create,
            "sleep", "infinity",
            image=image,
            timeout=config.timeout_seconds,
            **({"cpu": 1.0} if config.memory_limit_mb <= 512 else {"cpu": 2.0}),
        )

        return sb

    async def _terminate_modal_sandbox(self, sb: object) -> None:
        """Terminate a Modal sandbox (best-effort)."""
        import asyncio

        try:
            await asyncio.to_thread(sb.terminate)  # type: ignore[union-attr]
        except Exception:
            logger.debug("Failed to terminate modal sandbox", exc_info=True)

    async def _run(
        self,
        code: str,
        context: dict,
        sb: object,
        timeout: int,
    ) -> ExecutionResult:
        import asyncio

        # ── Assemble the script ─────────────────────────────────────
        parts: list[str] = []
        preamble = _build_preamble(context)
        if preamble:
            parts.append(preamble)
        parts.append(code)
        script = "\n".join(parts)

        # ── Execute via Modal sandbox exec ──────────────────────────
        t0 = time.monotonic()
        try:
            process = await asyncio.to_thread(
                sb.exec,  # type: ignore[union-attr]
                "python", "-c", script,
            )

            # Wait for the process to complete with a timeout.
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            async def _collect() -> int:
                # Modal's exec returns a ContainerProcess with
                # stdout/stderr iterables and a wait() method.
                for chunk in process.stdout:
                    stdout_chunks.append(chunk)
                for chunk in process.stderr:
                    stderr_chunks.append(chunk)
                return process.returncode

            exit_code = await asyncio.wait_for(
                _collect(), timeout=timeout,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=-1,
                duration_ms=duration_ms,
                truncated=False,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.error("Modal execution failed: %s", exc)
            return ExecutionResult(
                stdout="",
                stderr=f"Modal execution error: {exc}",
                exit_code=-1,
                duration_ms=duration_ms,
                truncated=False,
            )

        duration_ms = (time.monotonic() - t0) * 1000

        # ── Truncate output ─────────────────────────────────────────
        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)

        stdout_bytes = stdout_str.encode("utf-8", errors="replace")
        stderr_bytes = stderr_str.encode("utf-8", errors="replace")

        truncated = False
        if len(stdout_bytes) > _MAX_OUTPUT_BYTES:
            stdout_bytes = stdout_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True
        if len(stderr_bytes) > _MAX_OUTPUT_BYTES:
            stderr_bytes = stderr_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    # ── Cleanup on garbage-collection ───────────────────────────────

    def __del__(self) -> None:
        for _cfg, sb in self._sandboxes.values():
            with contextlib.suppress(Exception):
                sb.terminate()  # type: ignore[union-attr]
        self._sandboxes.clear()
