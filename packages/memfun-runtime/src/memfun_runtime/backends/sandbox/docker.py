from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import ExecutionResult, SandboxHandle

if TYPE_CHECKING:
    from memfun_core.types import SandboxConfig

logger = get_logger("sandbox.docker")

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


def _docker_available() -> bool:
    """Check whether the ``docker`` CLI is on PATH."""
    return shutil.which("docker") is not None


class DockerSandbox:
    """Execute Python code inside Docker containers.

    Implements the ``SandboxAdapter`` protocol from
    ``memfun_runtime.protocols.sandbox``.

    Uses the ``docker`` CLI via ``asyncio.create_subprocess_exec`` so that
    the only hard dependency is a working Docker installation -- the
    ``aiodocker`` package is **not** required.
    """

    def __init__(self, *, default_image: str = "python:3.12-slim") -> None:
        self._default_image = default_image
        # handle_id -> (config, container_id)
        self._sandboxes: dict[str, tuple[SandboxConfig, str]] = {}

    # ── Protocol methods ────────────────────────────────────────────

    async def create_sandbox(self, config: SandboxConfig) -> SandboxHandle:
        if not _docker_available():
            raise RuntimeError(
                "Docker CLI not found on PATH. Install Docker to use "
                "DockerSandbox, or switch to LocalSandbox."
            )

        handle = SandboxHandle(
            id=uuid.uuid4().hex,
            backend="docker",
        )

        container_id = await self._create_container(config, handle.id)
        self._sandboxes[handle.id] = (config, container_id)

        logger.info(
            "Created docker sandbox %s (container %s)",
            handle.id[:8],
            container_id[:12],
        )
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
        container_id: str | None = None

        if len(self._sandboxes) == 1:
            only_id = next(iter(self._sandboxes))
            config, container_id = self._sandboxes[only_id]
            timeout = min(timeout, config.timeout_seconds)

        if container_id is None:
            # Ad-hoc execution: create a temporary container, run, destroy.
            from memfun_core.types import SandboxConfig as SandboxCfg

            config = SandboxCfg(timeout_seconds=timeout)
            tmp_handle = await self.create_sandbox(config)
            _, container_id = self._sandboxes[tmp_handle.id]
            try:
                return await self._run(code, context, container_id, timeout)
            finally:
                await self.destroy_sandbox(tmp_handle)

        return await self._run(code, context, container_id, timeout)

    async def destroy_sandbox(self, handle: SandboxHandle) -> None:
        entry = self._sandboxes.pop(handle.id, None)
        if entry is None:
            return
        _, container_id = entry
        await self._remove_container(container_id)
        logger.info("Destroyed docker sandbox %s", handle.id[:8])

    # ── Docker helpers ──────────────────────────────────────────────

    async def _create_container(
        self, config: SandboxConfig, handle_id: str
    ) -> str:
        """Create and start a Docker container, returning its ID."""
        cmd: list[str] = [
            "docker", "create",
            "--label", f"memfun.sandbox.id={handle_id}",
            "--memory", f"{config.memory_limit_mb}m",
        ]

        # Network isolation: default to none.
        if not config.network_access:
            cmd.extend(["--network", "none"])

        # Environment variables.
        for key, value in config.env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Read-only bind-mounts.
        for path in config.read_paths:
            cmd.extend(["--volume", f"{path}:{path}:ro"])

        # Writable bind-mounts.
        for path in config.write_paths:
            cmd.extend(["--volume", f"{path}:{path}:rw"])

        # Image and entrypoint that keeps the container alive.
        cmd.extend([
            self._default_image,
            "sleep", "infinity",
        ])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Failed to create Docker container: {detail}"
            )

        container_id = stdout.decode().strip()

        # Start the container.
        start_proc = await asyncio.create_subprocess_exec(
            "docker", "start", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, start_err = await start_proc.communicate()
        if start_proc.returncode != 0:
            detail = start_err.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Failed to start Docker container {container_id[:12]}: "
                f"{detail}"
            )

        return container_id

    async def _remove_container(self, container_id: str) -> None:
        """Stop and remove a Docker container (best-effort)."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "--force", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _run(
        self,
        code: str,
        context: dict,
        container_id: str,
        timeout: int,
    ) -> ExecutionResult:
        # ── Assemble the script ─────────────────────────────────────
        parts: list[str] = []
        preamble = _build_preamble(context)
        if preamble:
            parts.append(preamble)
        parts.append(code)
        script = "\n".join(parts)

        # ── Execute via docker exec ─────────────────────────────────
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-i", container_id,
                "python", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            # Kill the exec process to avoid lingering work.
            try:
                proc.kill()  # type: ignore[possibly-unbound]
                await proc.wait()  # type: ignore[possibly-unbound]
            except (OSError, ProcessLookupError):
                pass
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=-1,
                duration_ms=duration_ms,
                truncated=False,
            )

        duration_ms = (time.monotonic() - t0) * 1000

        # ── Truncate output ─────────────────────────────────────────
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
            exit_code=proc.returncode or 0,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    # ── Cleanup on garbage-collection ───────────────────────────────

    def __del__(self) -> None:
        # Best-effort synchronous cleanup; callers should prefer
        # ``destroy_sandbox`` to ensure proper async teardown.
        for _cfg, container_id in self._sandboxes.values():
            try:
                import subprocess
                subprocess.run(
                    ["docker", "rm", "--force", container_id],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        self._sandboxes.clear()
