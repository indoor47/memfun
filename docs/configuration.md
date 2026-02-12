# Configuration

Memfun reads its configuration from a `memfun.toml` file in the project root.
This file is created by `memfun init` or can be written manually.

## Full Reference

Below is a complete `memfun.toml` with all available options and their default
values:

```toml
[project]
name = "memfun-project"                # Project name

[llm]
provider = "anthropic"                 # LLM provider: anthropic, openai, etc.
model = "claude-sonnet-4-20250514"           # Model identifier
sub_model = ""                         # Optional sub-model for RLM recursion
api_key_env = "ANTHROPIC_API_KEY"      # Env var name containing the API key
base_url = ""                          # Optional custom API base URL
temperature = 0.0                      # Sampling temperature (0.0 = deterministic)
max_tokens = 8192                      # Max tokens per LLM response

[backend]
tier = "sqlite"                        # Backend tier: in-process, sqlite, redis, nats
sqlite_path = ".memfun/memfun.db"      # Path to SQLite database file
sqlite_wal = true                      # Enable WAL mode for SQLite
sqlite_poll_interval_ms = 100          # Polling interval for SQLite event bus
redis_url = "redis://localhost:6379"   # Redis connection URL (tier = redis)
redis_prefix = "memfun:"              # Key prefix for Redis
nats_url = "nats://localhost:4222"     # NATS connection URL (tier = nats)
nats_creds_file = ""                   # Path to NATS credentials file
nats_stream_prefix = "memfun"         # Stream prefix for NATS JetStream

[sandbox]
backend = "local"                      # Sandbox backend: local, docker, modal
docker_image = "python:3.12-slim"      # Docker image for sandbox containers
modal_app_name = "memfun-sandbox"      # Modal app name for cloud sandboxes
timeout_seconds = 30                   # Execution timeout per sandbox run
memory_limit_mb = 512                  # Memory limit per sandbox

[web]
fetch_timeout_seconds = 30             # Timeout for web fetch requests
fetch_max_size_bytes = 5242880         # Max response size (5 MiB default)
fetch_cache_ttl_seconds = 900          # Cache TTL for fetched pages (15 min)
search_backend = "duckduckgo"          # Search backend: duckduckgo, brave, tavily, searxng
search_cache_ttl_seconds = 3600        # Cache TTL for search results (1 hour)
rate_limit_per_minute = 30             # Rate limit for web requests
rate_limit_burst = 5                   # Burst allowance for rate limiter
allowed_domains = []                   # Allowlist (empty = all allowed)
blocked_domains = []                   # Blocklist for web tools
brave_api_key_env = "BRAVE_API_KEY"    # Env var for Brave Search API key
tavily_api_key_env = "TAVILY_API_KEY"  # Env var for Tavily API key
searxng_url = "http://localhost:8888"  # SearXNG instance URL
```

## Sections

### [project]

General project settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"memfun-project"` | Project name used in logs and UI |

### [llm]

Language model configuration. Memfun resolves API keys from environment
variables at runtime -- the configuration file never contains raw secrets.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | `"anthropic"` | LLM provider |
| `model` | string | `"claude-sonnet-4-20250514"` | Model identifier |
| `sub_model` | string | `""` | Sub-model for RLM recursive calls |
| `api_key_env` | string | `"ANTHROPIC_API_KEY"` | Environment variable containing the API key |
| `base_url` | string | `""` | Custom API base URL (for proxies or self-hosted) |
| `temperature` | float | `0.0` | Sampling temperature |
| `max_tokens` | int | `8192` | Maximum tokens per response |

### [backend]

Runtime backend configuration. The `tier` field selects which backend
implementation to use. Only the settings relevant to the selected tier need
to be provided.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tier` | string | `"sqlite"` | Backend tier: `in-process`, `sqlite`, `redis`, `nats` |
| `sqlite_path` | string | `".memfun/memfun.db"` | SQLite database file path |
| `sqlite_wal` | bool | `true` | Enable WAL mode for better concurrency |
| `sqlite_poll_interval_ms` | int | `100` | Event bus polling interval in ms |
| `redis_url` | string | `"redis://localhost:6379"` | Redis connection URL |
| `redis_prefix` | string | `"memfun:"` | Redis key prefix |
| `nats_url` | string | `"nats://localhost:4222"` | NATS server URL |
| `nats_creds_file` | string | `""` | NATS credentials file path |
| `nats_stream_prefix` | string | `"memfun"` | NATS JetStream stream prefix |

### [sandbox]

Code execution sandbox configuration. Controls how the agent runs code safely.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `"local"` | Sandbox type: `local`, `docker`, `modal` |
| `docker_image` | string | `"python:3.12-slim"` | Docker image for container sandbox |
| `modal_app_name` | string | `"memfun-sandbox"` | Modal app name for cloud sandbox |
| `timeout_seconds` | int | `30` | Execution timeout per sandbox run |
| `memory_limit_mb` | int | `512` | Memory limit in megabytes |

### [web]

Web tool configuration for URL fetching and web search.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fetch_timeout_seconds` | int | `30` | HTTP request timeout |
| `fetch_max_size_bytes` | int | `5242880` | Maximum response body size (5 MiB) |
| `fetch_cache_ttl_seconds` | int | `900` | Cache duration for fetched pages |
| `search_backend` | string | `"duckduckgo"` | Search provider |
| `search_cache_ttl_seconds` | int | `3600` | Cache duration for search results |
| `rate_limit_per_minute` | int | `30` | Requests per minute limit |
| `rate_limit_burst` | int | `5` | Burst request allowance |
| `allowed_domains` | list | `[]` | Domain allowlist (empty = allow all) |
| `blocked_domains` | list | `[]` | Domain blocklist |
| `brave_api_key_env` | string | `"BRAVE_API_KEY"` | Env var for Brave API key |
| `tavily_api_key_env` | string | `"TAVILY_API_KEY"` | Env var for Tavily API key |
| `searxng_url` | string | `"http://localhost:8888"` | SearXNG instance URL |

## Minimal Configuration

A minimal `memfun.toml` that uses all defaults:

```toml
[project]
name = "my-project"
```

This uses Anthropic's Claude with a SQLite backend, local sandbox, and
DuckDuckGo for web search. Make sure to set the `ANTHROPIC_API_KEY`
environment variable.

## Backend-Specific Examples

### SQLite (default)

```toml
[backend]
tier = "sqlite"
sqlite_path = ".memfun/memfun.db"
```

### Redis

```toml
[backend]
tier = "redis"
redis_url = "redis://localhost:6379"
redis_prefix = "myproject:"
```

### NATS JetStream

```toml
[backend]
tier = "nats"
nats_url = "nats://nats.example.com:4222"
nats_creds_file = "/path/to/nats.creds"
nats_stream_prefix = "myproject"
```

### Docker Sandbox

```toml
[sandbox]
backend = "docker"
docker_image = "python:3.12-slim"
timeout_seconds = 60
memory_limit_mb = 1024
```
