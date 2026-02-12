from __future__ import annotations

import pytest
from memfun_core.config import BackendConfig, MemfunConfig
from memfun_runtime.builder import RuntimeBuilder
from memfun_runtime.context import RuntimeContext


class TestRuntimeBuilder:
    async def test_build_memory_backend(self):
        config = MemfunConfig(backend=BackendConfig(tier="memory"))
        ctx = await RuntimeBuilder(config).build()
        assert isinstance(ctx, RuntimeContext)
        assert ctx.config.backend.tier == "memory"

    async def test_build_sqlite_backend(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        config = MemfunConfig(backend=BackendConfig(tier="sqlite", sqlite_path=db_path))
        ctx = await RuntimeBuilder(config).build()
        assert isinstance(ctx, RuntimeContext)

    async def test_build_unknown_tier_raises(self):
        config = MemfunConfig(backend=BackendConfig(tier="unknown"))
        with pytest.raises(ValueError, match="Unknown backend tier"):
            await RuntimeBuilder(config).build()

    async def test_build_redis_requires_connection(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        config = MemfunConfig(backend=BackendConfig(tier="redis"))
        with pytest.raises(RedisConnectionError):
            await RuntimeBuilder(config).build()
