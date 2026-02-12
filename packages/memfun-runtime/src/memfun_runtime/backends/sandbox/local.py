from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
from typing import TYPE_CHECKING

from memfun_core.logging import get_logger
from memfun_core.types import ExecutionResult, SandboxHandle

if TYPE_CHECKING:
    from memfun_core.types import SandboxConfig

logger = get_logger("sandbox.local")

_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB
_MAX_TIMEOUT_SECONDS = 300     # 5 minutes hard cap


def _build_preamble(context: dict) -> str:
    """Build a Python preamble that injects *context* as local variables."""
    if not context:
        return ""
    # Serialise to JSON so the child process can safely deserialise.
    ctx_json = json.dumps(context, default=str)
    lines = [
        "import json as __ctx_json",
        f"__ctx_data = __ctx_json.loads({ctx_json!r})",
        "globals().update(__ctx_data)",
        "del __ctx_json, __ctx_data",
    ]
    return "\n".join(lines) + "\n"


def _build_resource_limit_code(config: SandboxConfig) -> str:
    """Return Python source that sets resource limits (Unix only)."""
    if platform.system() == "Windows":
        return ""
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    cpu_seconds = config.timeout_seconds
    return (
        "import resource as __rl\n"
        "try:\n"
        f"    __rl.setrlimit(__rl.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))\n"
        "except (ValueError, OSError):\n"
        "    pass\n"
        "try:\n"
        f"    __rl.setrlimit(__rl.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "except (ValueError, OSError):\n"
        "    pass\n"
        "del __rl\n"
    )


class LocalSandbox:
    """Execute Python code in isolated subprocesses with safety guardrails.

    Implements the ``SandboxAdapter`` protocol from
    ``memfun_runtime.protocols.sandbox``.
    """

    def __init__(self) -> None:
        # handle_id -> (config, work_dir)
        self._sandboxes: dict[str, tuple[SandboxConfig, str]] = {}

    # ── Protocol methods ────────────────────────────────────────────

    async def create_sandbox(self, config: SandboxConfig) -> SandboxHandle:
        work_dir = tempfile.mkdtemp(prefix="memfun_sandbox_")
        handle = SandboxHandle(
            id=uuid.uuid4().hex,
            backend="local",
        )
        self._sandboxes[handle.id] = (config, work_dir)
        logger.info(
            "Created local sandbox %s at %s", handle.id[:8], work_dir
        )
        return handle

    async def execute(
        self,
        code: str,
        context: dict,
        timeout: int = 30,
    ) -> ExecutionResult:
        # Enforce a hard cap on timeout to prevent resource
        # exhaustion via extremely large values.
        timeout = max(1, min(timeout, _MAX_TIMEOUT_SECONDS))

        # Use an ad-hoc temporary directory when no sandbox
        # has been created.
        work_dir = tempfile.mkdtemp(prefix="memfun_exec_")
        adhoc = True

        # If there is exactly one active sandbox, re-use its
        # directory.
        if len(self._sandboxes) == 1:
            only_id = next(iter(self._sandboxes))
            cfg, work_dir = self._sandboxes[only_id]
            adhoc = False
            timeout = min(timeout, cfg.timeout_seconds)

        try:
            return await self._run(code, context, work_dir, timeout)
        finally:
            if adhoc:
                shutil.rmtree(work_dir, ignore_errors=True)

    async def destroy_sandbox(self, handle: SandboxHandle) -> None:
        entry = self._sandboxes.pop(handle.id, None)
        if entry is None:
            return
        _, work_dir = entry
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Destroyed local sandbox %s", handle.id[:8])

    # ── Internal helpers ────────────────────────────────────────────

    async def _run(
        self,
        code: str,
        context: dict,
        work_dir: str,
        timeout: int,
    ) -> ExecutionResult:
        # Determine the SandboxConfig to use for resource limits.  When an
        # active sandbox exists we already resolved it in ``execute``; for
        # ad-hoc runs we fall back to safe defaults imported lazily.
        from memfun_core.types import SandboxConfig as SandboxCfg

        config: SandboxConfig | None = None
        if len(self._sandboxes) == 1:
            config = next(iter(self._sandboxes.values()))[0]
        if config is None:
            config = SandboxCfg(timeout_seconds=timeout)

        # ── Assemble the script ─────────────────────────────────────
        parts: list[str] = []
        rl_code = _build_resource_limit_code(config)
        if rl_code:
            parts.append(rl_code)
        preamble = _build_preamble(context)
        if preamble:
            parts.append(preamble)
        parts.append(code)

        script = "\n".join(parts)

        script_path = os.path.join(work_dir, "_memfun_exec.py")
        # Write script file in a thread to avoid blocking the event loop.
        await asyncio.to_thread(self._write_file, script_path, script)

        # ── Build subprocess environment ────────────────────────────
        env = os.environ.copy()
        if config and config.env_vars:
            env.update(config.env_vars)
        if config and not config.network_access:
            # Best-effort: setting an invalid proxy to hinder network calls.
            env["http_proxy"] = "http://0.0.0.0:0"
            env["https_proxy"] = "http://0.0.0.0:0"
            env["no_proxy"] = ""

        # ── Launch ──────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            # Kill the child and reap it to avoid zombies.
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

    @staticmethod
    def _write_file(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # ── Cleanup on garbage-collection ───────────────────────────────

    def __del__(self) -> None:
        for _cfg, work_dir in self._sandboxes.values():
            shutil.rmtree(work_dir, ignore_errors=True)
        self._sandboxes.clear()
