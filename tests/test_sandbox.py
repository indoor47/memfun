from __future__ import annotations

import pytest
from memfun_core.types import ExecutionResult, SandboxConfig, SandboxHandle
from memfun_runtime.backends.sandbox.local import LocalSandbox


@pytest.fixture
def sandbox():
    sb = LocalSandbox()
    yield sb
    # Cleanup any remaining sandboxes
    for handle_id in list(sb._sandboxes):
        handle = SandboxHandle(id=handle_id, backend="local")
        import asyncio
        asyncio.get_event_loop().run_until_complete(sb.destroy_sandbox(handle))


class TestLocalSandboxConformance:
    """Conformance tests for the LocalSandbox implementation."""

    async def test_execute_simple_code(self, sandbox: LocalSandbox):
        result = await sandbox.execute('print("hello")', context={})
        assert isinstance(result, ExecutionResult)
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0
        assert result.duration_ms > 0

    async def test_execute_with_context(self, sandbox: LocalSandbox):
        result = await sandbox.execute(
            'print(name, age)',
            context={"name": "Alice", "age": 30},
        )
        assert result.exit_code == 0
        assert "Alice" in result.stdout
        assert "30" in result.stdout

    async def test_execute_timeout(self, sandbox: LocalSandbox):
        code = "import time\nwhile True:\n    time.sleep(0.1)"
        result = await sandbox.execute(code, context={}, timeout=1)
        assert result.exit_code != 0
        assert "timed out" in result.stderr.lower()

    async def test_execute_syntax_error(self, sandbox: LocalSandbox):
        result = await sandbox.execute("def (broken", context={})
        assert result.exit_code != 0
        assert result.stderr != ""

    async def test_create_destroy_sandbox(self, sandbox: LocalSandbox):
        config = SandboxConfig(language="python", timeout_seconds=10)
        handle = await sandbox.create_sandbox(config)
        assert isinstance(handle, SandboxHandle)
        assert handle.backend == "local"
        assert handle.id in sandbox._sandboxes

        await sandbox.destroy_sandbox(handle)
        assert handle.id not in sandbox._sandboxes

    async def test_output_truncation(self, sandbox: LocalSandbox):
        # Generate output larger than the 100 KB limit.
        code = 'print("A" * 200_000)'
        result = await sandbox.execute(code, context={})
        assert result.exit_code == 0
        assert result.truncated is True
        # Stdout should be capped roughly at the limit.
        assert len(result.stdout) <= 100 * 1024 + 128  # small decode margin

    async def test_destroy_nonexistent_noop(self, sandbox: LocalSandbox):
        handle = SandboxHandle(id="nonexistent", backend="local")
        await sandbox.destroy_sandbox(handle)  # should not raise

    async def test_execute_stderr_capture(self, sandbox: LocalSandbox):
        code = "import sys; sys.stderr.write('oops\\n')"
        result = await sandbox.execute(code, context={})
        assert "oops" in result.stderr

    async def test_execute_in_created_sandbox(self, sandbox: LocalSandbox):
        config = SandboxConfig(timeout_seconds=10)
        handle = await sandbox.create_sandbox(config)
        result = await sandbox.execute('print("inside")', context={})
        assert result.stdout.strip() == "inside"
        assert result.exit_code == 0
        await sandbox.destroy_sandbox(handle)
