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
# [llm.roles.<role>] tables override [llm] per role -- see "per-role models" below

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

### [llm.roles.<role>] -- per-role models

A swarm can run different roles on different models. Each
`[llm.roles.<role>]` table overrides `[llm]` for one role; every field is
optional and falls back to `[llm]` **field by field**. Leave a role out and
it uses `[llm]` unchanged. With no `[llm.roles]` section at all, behaviour
is exactly as before: everything runs on `[llm]`.

| Role | What runs under it |
|------|--------------------|
| `orchestrator` | The workflow engine itself (`memfun swarm`, chat workflow runs): any LLM call not bound to a narrower role |
| `decomposer` | `TaskDecomposer` (task -> sub-task DAG), including hierarchical recursion |
| `worker` | Specialist agents (`coder-agent`, `test-agent`, `file-agent`, ...) and the single-task `rlm-coder` dispatch of a workflow |
| `auditor` | Per-task `review-agent` dispatches |
| `reviewer` | The `pr-reviewer` agent at the PR layer |

Allowed keys per role: `provider`, `model`, `base_url`, `api_key_env`,
`temperature`, `max_tokens`. Unknown role names and unknown keys are
ignored with a warning. Wrong value types raise a configuration error, and
so does an `api_key_env` that is not a valid environment variable *name*:
this field names the variable that holds the key and never holds the key
itself; `api_key_env = ""` means "no key". `base_url = ""` clears an
inherited `[llm].base_url`.

Two fields are exceptions to plain field-by-field fallback, so a role
never sends credentials to the wrong place:

- `base_url`: a role that switches `provider` does **not** inherit
  `[llm].base_url` (it uses the new provider's default endpoint unless it
  sets its own `base_url`). `base_url = ""` clears an inherited value
  explicitly.
- `api_key_env`: a role inherits `[llm].api_key_env` only when its
  `provider` **and** `base_url` both resolve to the same values as
  `[llm]`'s. A role that changes either gets no key unless it sets its own
  `api_key_env`, so the `[llm]` key is never sent to a different endpoint.
  For `custom` (and any other provider memfun does not map itself) the
  route is part of `model` (e.g. `openrouter/...`), so a role that changes
  the model's route prefix also gets no inherited key.

Provider names are matched case-insensitively; an unrecognized provider
logs a warning.

Global and project configs are merged before roles are resolved, so an
`api_key_env` from `~/.memfun/config.toml` combined with a project-level
`provider` or `base_url` change is sent to the project's endpoint -- the
same trust model as `[llm].base_url`. Review merged configs from untrusted
projects.

#### Default example: Qwen for every role

The shipped example runs **all five roles** on one self-hosted
OpenAI-compatible server (SGLang serving Qwen3.8-27B under the alias
`qwen3.8-27b`) reached through an SSH tunnel on `localhost:18000`. No
Anthropic key is needed. Set the endpoint once in `[llm]`; every role
inherits it, and a role table is only needed to tune that role:

```toml
[llm]
provider = "openai"                     # OpenAI-compatible endpoint
model = "qwen3.8-27b"                   # the alias the server serves
base_url = "http://127.0.0.1:18000/v1"  # local end of the SSH tunnel
api_key_env = "MEMFUN_WORKER_API_KEY"   # dummy; unset -> placeholder key

# Optional per-role tuning; omitted fields (endpoint, key) are inherited.
[llm.roles.worker]
max_tokens = 16384
```

| Role | Resolves to |
|------|-------------|
| `orchestrator` | `[llm]`: `openai/qwen3.8-27b` at `http://127.0.0.1:18000/v1` |
| `decomposer` | `[llm]`: same endpoint |
| `worker` | `[llm]` plus its own `max_tokens` |
| `auditor` | `[llm]`: same endpoint |
| `reviewer` | `[llm]`: same endpoint |

Open the tunnel before starting the swarm, e.g.
`ssh -N -L 18000:127.0.0.1:<server-port> <user>@<host>`.

#### Optional comparison arm: Claude Opus 5.5 as orchestrator/decomposer

Not the default. To compare against a frontier planner, add these tables
on top of the Qwen `[llm]` above; workers, auditors and the reviewer stay
on Qwen. Each anthropic role sets its own `api_key_env` (it would not
inherit `MEMFUN_WORKER_API_KEY` anyway, because the provider differs) and
does not inherit the tunnel `base_url`:

```toml
[llm.roles.orchestrator]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[llm.roles.decomposer]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"
```

Notes:

- `provider = "openai"` with an unset key variable sends the placeholder
  key `sk-local` only when `base_url` is a loopback / private / link-local
  host (`localhost`, `*.local`, 127.0.0.0/8, ::1, 10/8, 172.16/12,
  192.168/16), since self-hosted servers usually run without auth. A hosted
  `base_url` with no key gets no placeholder: memfun warns instead.
- At startup memfun logs one warning per configured role whose key is
  missing or whose model cannot be set up.
- Once any `[llm.roles.*]` table exists, every role binds its own resolved
  model explicitly; a role without a table runs on `[llm]`, never on the
  model of an enclosing role (for example a worker dispatched by the
  orchestrator).
- Chat requests answered outside a multi-agent workflow (direct answers,
  the single-agent and context-first paths) run as role `default` on
  `[llm]`.
- Each role's model is bound per call with a task-local `dspy.context`, so
  concurrent agents running different roles do not interfere.
- Every `llm.call` telemetry event carries `model` (the configured model id
  of the LM that made the call, e.g. `openai/qwen3.8-27b`, not the name the
  server echoes back) and `role` (`default` when no role is bound).

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

### [limits] -- step caps and stall stop

```toml
[limits]
stall_steps = 12            # stop an agent after 12 iterations in a row with nothing new

[limits.iterations]         # per-agent iteration caps (backstops), by agent name
coder-agent = 80
rlm-coder = 150             # the single-task path (one agent per task)
```

Step caps are a backstop, not a budget. An agent that is not making progress is stopped by
**stall detection**: `stall_steps` consecutive iterations that only re-read files it already
read, re-ran a command with no edit in between, or ran nothing. It then stops with a clear
reason (`[STOPPED: stalled ...]` at the start of its answer, `RLM STOPPED at iteration N (stalled ...)` in
the log, `stop_reason` in its result). `stall_steps = 0` turns stall stopping off.

Built-in iteration caps (4x the old ones): `coder-agent` 60, `test-agent` 40, `file-agent` /
`review-agent` and most other specialists 32, `rlm-coder` 100. The overall budget of a run is
a wall-clock and cost limit set outside memfun (for RunPod, the pod's hard stop and cost cap).

A response cut off by `max_tokens` is an error, never a silent success: the `llm.call` event
is marked failed with `response truncated (finish_reason=length)`, memfun logs
`LLM RESPONSE TRUNCATED` at error level, and a truncated decomposer plan fails the workflow
instead of falling back to a single task. Raise `[llm] max_tokens` (or
`[llm.roles.<role>].max_tokens`) if you see it; `evals/swarm/run_smoke.py` defaults to 32768
for every role.

### Sampling and repetition loops

At `temperature = 0.0` a 27B model can fall into an endless repetition loop that only ends at
`max_tokens`. Use a non-zero temperature plus a repetition penalty on OpenAI-compatible
servers (SGLang: `frequency_penalty` / `presence_penalty` in [-2, 2], `repetition_penalty` in
(0, 2], 1.0 = off):

```toml
[llm]
temperature = 0.7
repetition_penalty = 1.05   # sent as extra_body; 0/unset sends nothing
loop_guard = true           # abort a call whose output repeats (see below)
```

`loop_guard = true` (needs `base_url`, off by default) makes the call with streaming on
DSPy's LiteLLM engine (DSPy 3.x; older DSPy always uses litellm), with DSPy's response cache off for it, mid-stream connection errors retried, and cuts it as soon as the output
ends in a block repeated 8+ times over 6000+ characters (about 1500 tokens: shorter repeats such as table rows or padding are legitimate). The aborted response is returned as
cut off (`finish_reason=length`), so the truncation paths fail visibly, memfun logs
`LLM LOOP ABORTED`, and `evals/swarm/run_smoke.py` counts `loop_aborts` per episode.
`MEMFUN_LOOP_GUARD=0` switches the guard off for a process. `run_smoke.py` defaults to
temperature 0.7, repetition penalty 1.05 and the guard on (`--temperature`,
`--repetition-penalty`, `--frequency-penalty`, `--presence-penalty`, `--no-loop-guard`).

### Project verify file (`.memfun/verify.json`)

The single-task solver verifies its edit with a linter it auto-detects. A project can name its
own checks, for example the unit tests of the function being implemented:

```json
{"commands": ["python -m pytest -q tests/test_core.py::test_x"], "max_fix_attempts": 4}
```

The commands run through a shell, so a repository must not be able to trigger them by itself:
the file is read only when `MEMFUN_TRUST_PROJECT_VERIFY=1` is set in memfun's environment
(`evals/swarm/run_smoke.py` sets it for its episodes, whose file it wrote itself). Failing
output goes through the solver's verify-and-fix loop.

`.memfun/` is never gathered into the model's context (it can hold credentials), and neither
are files named like secrets (`.env*`, `credentials*`, `secrets*`, `*.pem`, `*.key`, ...).

#### Fix-loop and look-step options

All keys are optional. An invalid value is **not** applied silently: it logs a warning that
names the key and the allowed range, and the setting stays off or at its default.

| key | values | default | what it does |
|---|---|---|---|
| `max_fix_attempts` | integer 1..50 | 2 (4 for the benchmark bundles) | base number of verify-and-fix rounds; past it the loop continues only while rounds improve (up to `fix_hard_cap`) |
| `fix_detail` | `true` / `false` | `false` | after **every** failed round that has failing pytest tests, re-run them with `-vv` and show the fix step their untruncated assertion output (not only on a stall) |
| `probe_rounds` | integer 0..20 | 0 (off) | on a round that did not improve, up to this many times per solve, the model writes short scripts (up to `fix_probe_iterations`, default 3, per stall) that run in the project and print where the code and the tests' data disagree; the real output goes into the next fix call |
| `probe_strict` | `true` / `false` | `false` | run probes in `python -I -S -B` (standard library and the project only, no site-packages) instead of the project's interpreter; the benchmark runner sets it so that an installed copy of the library under repair cannot be used |

The loop itself is the same with or without probes: `probe_rounds` changes only whether the look
step runs, not when the loop stops.

**Probe safety.** A probe runs as a separate process with a 30 s timeout and a 3 KB output
cap, in the interpreter the project's own verify command uses (never memfun's own interpreter;
with none found, the probes report that they cannot run). Project packages import from the
project (`src/` and the root come first), and reading an installed copy of a package the project
defines is denied. An audit hook denies file writes, child processes, network access, `ctypes`,
and reads outside the project and the interpreter's own directories. This is **defence in
depth, not a sandbox**: native code or a determined script can get around an audit hook, so run
untrusted repositories in a container or VM. Every probe and its output is appended to
`.memfun/probes.jsonl`.

## Dashboard Environment Variables

The live dashboard (`memfun dashboard`, or embedded in `memfun chat`) reads these at request time:

| Variable | Description |
|----------|-------------|
| `MEMFUN_DASHBOARD_TOKEN` | Token non-local clients must present (falls back to `~/.config/memfun/dashboard_token`) |
| `MEMFUN_INFERENCE_METRICS_URL` | Prometheus endpoint of the SGLang/vLLM server for the swarm view's server strip (e.g. `http://127.0.0.1:30000/metrics`) |
| `MEMFUN_POD_RATE_USD` | Default pod cost in USD/hour (e.g. `8.99`) for the cost meter. Used only while the page's pod-rate box is empty; a value typed there always wins. Invalid or negative values are ignored. |

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
