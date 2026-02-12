from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

from memfun_runtime.context import RuntimeContext

if TYPE_CHECKING:
    from memfun_core.config import MemfunConfig

logger = get_logger("builder")


class RuntimeBuilder:
    """Build a RuntimeContext from configuration.

    Usage:
        config = MemfunConfig.from_toml("memfun.toml")
        ctx = await RuntimeBuilder(config).build()
    """

    def __init__(self, config: MemfunConfig) -> None:
        self._config = config

    async def build(self) -> RuntimeContext:
        tier = self._config.backend.tier
        logger.info("Building runtime with %s backend", tier)

        if tier == "memory":
            return await self._build_memory()
        elif tier == "sqlite":
            return await self._build_sqlite()
        elif tier == "redis":
            return await self._build_redis()
        elif tier == "nats":
            return await self._build_nats()
        else:
            raise ValueError(f"Unknown backend tier: {tier!r}")

    async def _build_memory(self) -> RuntimeContext:
        from memfun_runtime.backends.memory import (
            InProcessEventBus,
            InProcessHealthMonitor,
            InProcessLifecycle,
            InProcessRegistry,
            InProcessSessionManager,
            InProcessSkillRegistry,
            InProcessStateStore,
        )
        sandbox = self._build_sandbox()
        return RuntimeContext(
            event_bus=InProcessEventBus(),
            state_store=InProcessStateStore(),
            sandbox=sandbox,
            lifecycle=InProcessLifecycle(),
            registry=InProcessRegistry(),
            session=InProcessSessionManager(),
            health=InProcessHealthMonitor(),
            skill_registry=InProcessSkillRegistry(),
            config=self._config,
        )

    async def _build_sqlite(self) -> RuntimeContext:
        from memfun_runtime.backends.sqlite import (
            SQLiteEventBus,
            SQLiteHealthMonitor,
            SQLiteLifecycle,
            SQLiteRegistry,
            SQLiteSessionManager,
            SQLiteSkillRegistry,
            SQLiteStateStore,
        )
        db_path = self._config.backend.sqlite_path
        sandbox = self._build_sandbox()
        return RuntimeContext(
            event_bus=await SQLiteEventBus.create(db_path),
            state_store=await SQLiteStateStore.create(db_path),
            sandbox=sandbox,
            lifecycle=SQLiteLifecycle(db_path),
            registry=await SQLiteRegistry.create(db_path),
            session=await SQLiteSessionManager.create(db_path),
            health=SQLiteHealthMonitor(db_path),
            skill_registry=await SQLiteSkillRegistry.create(db_path),
            config=self._config,
        )

    async def _build_redis(self) -> RuntimeContext:
        from memfun_runtime.backends.redis import (
            RedisEventBus,
            RedisHealthMonitor,
            RedisLifecycle,
            RedisRegistry,
            RedisSessionManager,
            RedisSkillRegistry,
            RedisStateStore,
        )
        url = self._config.backend.redis_url
        prefix = self._config.backend.redis_prefix
        sandbox = self._build_sandbox()
        return RuntimeContext(
            event_bus=await RedisEventBus.create(url, prefix=prefix),
            state_store=await RedisStateStore.create(url, prefix=prefix),
            sandbox=sandbox,
            lifecycle=await RedisLifecycle.create(url, prefix=prefix),
            registry=await RedisRegistry.create(url, prefix=prefix),
            session=await RedisSessionManager.create(url, prefix=prefix),
            health=await RedisHealthMonitor.create(url, prefix=prefix),
            skill_registry=await RedisSkillRegistry.create(url, prefix=prefix),
            config=self._config,
        )

    async def _build_nats(self) -> RuntimeContext:
        from memfun_runtime.backends.nats import (
            NATSEventBus,
            NATSHealthMonitor,
            NATSLifecycle,
            NATSRegistry,
            NATSSessionManager,
            NATSSkillRegistry,
            NATSStateStore,
        )
        url = self._config.backend.nats_url
        creds = self._config.backend.nats_creds_file
        stream_prefix = self._config.backend.nats_stream_prefix
        sandbox = self._build_sandbox()
        return RuntimeContext(
            event_bus=await NATSEventBus.create(
                url, creds_file=creds, stream_prefix=stream_prefix,
            ),
            state_store=await NATSStateStore.create(url, creds_file=creds),
            sandbox=sandbox,
            lifecycle=await NATSLifecycle.create(url, creds_file=creds),
            registry=await NATSRegistry.create(url, creds_file=creds),
            session=await NATSSessionManager.create(url, creds_file=creds),
            health=await NATSHealthMonitor.create(url, creds_file=creds),
            skill_registry=await NATSSkillRegistry.create(url, creds_file=creds),
            config=self._config,
        )

    def _build_sandbox(self):
        sandbox_backend = self._config.sandbox.backend

        if sandbox_backend == "docker":
            return self._build_docker_sandbox()
        elif sandbox_backend == "modal":
            return self._build_modal_sandbox()
        else:
            return self._build_local_sandbox()

    def _build_local_sandbox(self):
        try:
            from memfun_runtime.backends.sandbox.local import LocalSandbox
            return LocalSandbox()
        except Exception:
            logger.warning(
                "Failed to create LocalSandbox, falling back to StubSandbox"
            )
            from memfun_runtime.backends.sandbox.stub import StubSandbox
            return StubSandbox()

    def _build_docker_sandbox(self):
        try:
            from memfun_runtime.backends.sandbox.docker import DockerSandbox
            image = self._config.sandbox.docker_image
            return DockerSandbox(default_image=image)
        except Exception:
            logger.warning(
                "Failed to create DockerSandbox, falling back to LocalSandbox"
            )
            return self._build_local_sandbox()

    def _build_modal_sandbox(self):
        try:
            from memfun_runtime.backends.sandbox.modal_sandbox import (
                ModalSandbox,
            )
            app_name = self._config.sandbox.modal_app_name
            return ModalSandbox(app_name=app_name)
        except Exception:
            logger.warning(
                "Failed to create ModalSandbox, falling back to LocalSandbox"
            )
            return self._build_local_sandbox()
