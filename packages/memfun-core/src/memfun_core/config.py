from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning empty dict if missing."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base (1 level deep for TOML sections)."""
    merged = dict(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-4-6"
    sub_model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 128_000


@dataclass(frozen=True, slots=True)
class BackendConfig:
    tier: str = "sqlite"
    sqlite_path: str = ".memfun/memfun.db"
    sqlite_wal: bool = True
    sqlite_poll_interval_ms: int = 100
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "memfun:"
    nats_url: str = "nats://localhost:4222"
    nats_creds_file: str | None = None
    nats_stream_prefix: str = "memfun"


@dataclass(frozen=True, slots=True)
class SandboxBackendConfig:
    backend: str = "local"
    docker_image: str = "python:3.12-slim"
    modal_app_name: str = "memfun-sandbox"
    timeout_seconds: int = 30
    memory_limit_mb: int = 512


@dataclass(frozen=True, slots=True)
class WebToolsConfig:
    fetch_timeout_seconds: int = 30
    fetch_max_size_bytes: int = 5_242_880  # 5 MiB
    fetch_cache_ttl_seconds: int = 900  # 15 min
    search_backend: str = "duckduckgo"  # duckduckgo | brave | tavily | searxng
    search_cache_ttl_seconds: int = 3600  # 1 hour
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 5
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    brave_api_key_env: str = "BRAVE_API_KEY"
    tavily_api_key_env: str = "TAVILY_API_KEY"
    searxng_url: str = "http://localhost:8888"


@dataclass(frozen=True, slots=True)
class MemfunConfig:
    """Top-level configuration, parsed from memfun.toml."""
    project_name: str = "memfun-project"
    llm: LLMConfig = field(default_factory=LLMConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    sandbox: SandboxBackendConfig = field(default_factory=SandboxBackendConfig)
    web: WebToolsConfig = field(default_factory=WebToolsConfig)

    @classmethod
    def from_toml(
        cls, path: Path | str = "memfun.toml"
    ) -> MemfunConfig:
        path = Path(path)
        raw = _load_toml(path)
        return cls._from_raw(raw)

    @classmethod
    def load(
        cls, project_dir: Path | str | None = None
    ) -> MemfunConfig:
        """Load config with global → project layering.

        Resolution order (later wins):
        1. Built-in defaults
        2. ~/.memfun/config.toml (global)
        3. .memfun/config.toml or memfun.toml (project)
        """
        global_path = Path.home() / ".memfun" / "config.toml"

        project_dir = (
            Path.cwd() if project_dir is None else Path(project_dir)
        )

        # Project config: .memfun/config.toml takes priority
        project_path = project_dir / ".memfun" / "config.toml"
        if not project_path.exists():
            project_path = project_dir / "memfun.toml"

        global_raw = _load_toml(global_path)
        project_raw = _load_toml(project_path)
        merged = _deep_merge(global_raw, project_raw)

        return cls._from_raw(merged)

    @classmethod
    def _from_raw(cls, raw: dict) -> MemfunConfig:
        """Build MemfunConfig from a raw TOML dict."""
        llm_raw = raw.get("llm", {})
        backend_raw = raw.get("backend", {})
        sandbox_raw = raw.get("sandbox", {})
        web_raw = raw.get("web", {})

        def _pick(section: dict, dc: type) -> dict:
            fields = dc.__dataclass_fields__
            return {
                k: v for k, v in section.items() if k in fields
            }

        return cls(
            project_name=raw.get("project", {}).get(
                "name", "memfun-project"
            ),
            llm=LLMConfig(**_pick(llm_raw, LLMConfig)),
            backend=BackendConfig(
                **_pick(backend_raw, BackendConfig)
            ),
            sandbox=SandboxBackendConfig(
                **_pick(sandbox_raw, SandboxBackendConfig)
            ),
            web=WebToolsConfig(
                **_pick(web_raw, WebToolsConfig)
            ),
        )
