# Memfun

**An autonomous coding agent that solves tasks in parallel, learns from every conversation, and scales from laptop to cluster.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![CI](https://github.com/indoor47/memfun/actions/workflows/ci.yml/badge.svg)](https://github.com/indoor47/memfun/actions/workflows/ci.yml)

---

## What Memfun Does

Give Memfun a task in natural language. It reads your codebase, plans an approach, writes code across multiple files, runs your linters, fixes errors, reviews its own output for consistency, and delivers the result -- all in one shot. For complex tasks, it splits work across parallel specialist agents (coder, tester, reviewer, debugger, security auditor) that work simultaneously and cross-check each other.

```
memfun > add JWT authentication to the Flask API

  Context-first: gathered 12 files (48 KB)

  Operations:
    write  src/auth/jwt.py           (JWT token creation, verification, refresh)
    write  src/auth/middleware.py     (require_auth decorator)
    edit   src/routes/api.py          (added @require_auth to protected routes)
    write  tests/test_auth.py         (12 test cases)
    run    ruff check src/            (passed)

  Verified: 0 lint errors

  32s  •  8.2k tokens
```

### How It Solves Tasks

Memfun uses a **three-tier escalation** strategy, starting cheap and fast:

**Tier 1: Context-First (2 LLM calls)** -- For most tasks. A planner selects which files to read (using a structural code map, not just filenames). A single-shot solver produces all file operations at once. An executor applies them, auto-detects your linter (ruff, eslint, go vet, cargo check), and fixes errors in up to 2 cycles. A consistency reviewer then checks the output against the original request and polishes if needed. Projects under 200 KB skip the planner entirely -- zero overhead.

**Tier 2: Multi-Agent Workflow (parallel specialists)** -- For complex tasks that need decomposition. A task decomposer breaks the work into a dependency DAG. Specialist agents run in parallel groups -- a coder implements features while a test agent writes tests and a security agent audits for vulnerabilities. Each specialist runs its own verify + review + polish pipeline. A cross-agent reviewer checks for conflicts between agents, and up to 2 revision rounds fix issues.

**Tier 3: RLM Exploration (iterative REPL)** -- Last resort for tasks requiring deep codebase exploration. The agent gets a sandboxed Python REPL where it can read files, run searches, execute commands, and make sub-LLM calls -- with variable state stored outside the token window so effective memory is unbounded.

```
Task arrives
  |
  v
Context-First Solver (2 LLM calls, ~30s)
  |
  +--> Success? Done.
  |
  +--> Failed/truncated?
         |
         v
       Multi-Agent Workflow (5-9 parallel agents, ~2min)
         |
         +--> Success? Done.
         |
         +--> No context solver?
                |
                v
              RLM Exploration (iterative, ~5min)
```

### What Makes It Different

| | Typical Agents | Memfun |
|---|---|---|
| **Solving** | Single LLM call or iterative loop | Context-first (2 calls) with automatic escalation to parallel multi-agent |
| **Quality** | Raw LLM output | Every output verified by linters + consistency reviewed + polished |
| **File edits** | Rewrites entire files | Targeted `edit_file` with fuzzy matching; destructive overwrites blocked |
| **Parallelism** | Sequential | Up to 9 specialists working simultaneously |
| **Memory** | Stateless | Persistent learning across sessions (TF-IDF + MEMORY.md) |
| **Scaling** | Single process | 4 backend tiers: in-process to distributed NATS cluster |

## Quick Start

```bash
# Install from source
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync
uv tool install packages/memfun-cli

# Initialize (LLM provider, API key, backend)
memfun init

# Start chatting
memfun
```

On first run, `memfun init` configures your LLM provider (Anthropic, OpenAI, Ollama, or custom endpoint), API key, backend tier (SQLite default), and sandbox.

## Interactive Chat

```
$ memfun

 ███╗   ███╗███████╗███╗   ███╗███████╗██╗   ██╗███╗   ██╗
 ████╗ ████║██╔════╝████╗ ████║██╔════╝██║   ██║████╗  ██║
 ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██║   ██║██╔██╗ ██║
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║   ██║██║╚██╗██║
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║     ╚██████╔╝██║ ╚████║
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═══╝

memfun > refactor the database module to use connection pooling

  Workflow: 3 agents in parallel
    ✓ coder-agent    8 iterations, 4 ops  (42s)
    ✓ test-agent     5 iterations, 2 ops  (28s)
    ✓ review-agent   3 iterations, 0 ops  (12s)

  Review: no issues found

  Operations:
    edit  src/db/connection.py    (replaced single connection with pool)
    edit  src/db/queries.py       (use pool.acquire() context manager)
    write tests/test_pool.py      (8 new test cases)
    run   ruff check src/         (passed)
    run   pytest tests/test_pool.py (passed)

  1m 48s  •  22.1k tokens

memfun > I always want pool_size=10 for dev

  Remembered: pool_size=10 for dev environments
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/remember <text>` | Store a preference or project fact |
| `/memory` | View stored memories |
| `/forget <target>` | Remove a memory entry |
| `/context` | Rescan project files |
| `/traces` | List recent execution traces |
| `/agents` | Show running specialist agents |
| `/workflow` | Show current workflow DAG and status |
| `/model` | Show or switch LLM model |
| `/clear` | Clear conversation history |

### CLI Commands

```bash
memfun                          # Interactive chat (default)
memfun init                     # Initialize project
memfun ask "how does auth work" # Ask a question
memfun analyze src/             # Analyze code structure
memfun fix "TypeError in login" # Fix a bug
memfun review src/auth.py       # Review code
memfun skill list               # List available skills
memfun agent list               # List agent definitions
```

## Architecture

### Solving Pipeline

```
                    ┌─────────────────────────────┐
                    │        Query Triage          │
                    │  (direct/project/task/web)   │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              v                v                v
        ┌──────────┐  ┌──────────────┐  ┌──────────────┐
        │  Direct   │  │ Context-First│  │  Multi-Agent │
        │  Answer   │  │   Solver     │  │  Workflow    │
        │ (1 call)  │  │  (2 calls)   │  │(5-9 agents) │
        └──────────┘  └──────┬───────┘  └──────┬───────┘
                              │                 │
                      ┌───────v─────────────────v───────┐
                      │        Quality Pipeline          │
                      │  Verify (linter) → Fix → Review  │
                      │      → Polish (edit-only)        │
                      └─────────────────────────────────┘
```

### Multi-Agent Workflow

```
Task Decomposer (DAG)
  │
  ├─ Group 1 (parallel):  FileAgent + PlannerAgent
  │
  ├─ Group 2 (parallel):  CoderAgent + TestAgent + SecurityAgent
  │     │
  │     └─ Each runs: RLM loop → Verify → Consistency Review → Polish
  │
  ├─ Cross-Agent Review:  ReviewAgent checks all outputs
  │
  └─ Revision (up to 2 rounds): re-run failing agents with feedback
```

### 9 Specialist Agents

| Agent | Role | Iterations |
|-------|------|-----------|
| **CoderAgent** | Writes production code, prefers `edit_file` over `write_file` | 15 |
| **TestAgent** | Writes and runs tests | 10 |
| **DebugAgent** | Diagnoses bugs, traces root causes | 12 |
| **ReviewAgent** | Code quality, consistency, cross-agent conflicts | 8 |
| **FileAgent** | Reads and analyzes files (never writes) | 8 |
| **SecurityAgent** | Vulnerability analysis (injection, SSRF, secrets) | 8 |
| **PlannerAgent** | Decomposes sub-problems, no code | 6 |
| **WebSearchAgent** | Web search via DuckDuckGo | 8 |
| **WebFetchAgent** | Fetches and extracts web page content | 8 |

### Code Map

Instead of feeding the planner a flat list of filenames, Memfun extracts a structural code map -- classes, functions, methods with their signatures -- using Python `ast` for `.py` files and regex for JS/TS, Go, Rust, and Java. The planner sees *what's in each file*, not just that it exists.

```
src/auth/jwt.py (2.1 KB)
  class JWTManager
    def __init__(self, secret: str, algorithm: str = "HS256")
    def create_token(self, payload: dict, expires_in: int = 3600) -> str
    def verify_token(self, token: str) -> dict | None
  def require_auth(f: Callable) -> Callable
src/db/models.py (1.4 KB)
  class User
    def check_password(self, password: str) -> bool
  class Session
```

### Quality Pipeline

Every code-producing path (context-first solver and each specialist agent) runs the same post-processing:

1. **Auto-detect linters**: Finds ruff, eslint, go vet, cargo check based on project files
2. **Verify**: Runs linters as subprocesses (0 LLM calls)
3. **Fix**: If lint errors found, feeds them to a fix solver (1 LLM call)
4. **Consistency review**: Semantic check comparing output against the original request (1 LLM call)
5. **Polish**: If issues found, targeted edits in `edit_only` mode -- never rewrites whole files (0-1 LLM calls)

### File Safety

- **Destructive write guard**: `write_file` on existing files is blocked if it would lose >30% of content
- **Edit-only mode**: Fix and polish steps can only use `edit_file`, never `write_file` on existing files
- **Fuzzy edit matching**: 3 strategies (exact, whitespace-normalized, difflib sliding window) so edits land even when LLM output has minor whitespace differences
- **Read caching**: MD5 hash per file prevents the agent from wasting iterations re-reading unchanged files
- **Stall detection**: Warns the agent when it's stuck reading without acting, or re-reading the same files

### Persistent Memory

After every conversation turn, the agent extracts reusable knowledge (preferences, patterns, project details) and stores it in two layers:

- **MEMORY.md**: Human-readable, editable file in `.memfun/MEMORY.md`. Always loaded as context.
- **SQLite MemoryStore**: TF-IDF indexed database for efficient retrieval at scale.

Before each turn, relevant memories are retrieved and injected as high-priority context. The agent genuinely learns -- it won't ask you what port to use twice.

## Backend Tiers

| Tier | Backend | Use Case |
|------|---------|----------|
| **T0** | In-Process (asyncio) | Unit tests, CI, instant startup |
| **T1** | SQLite (WAL mode) | Single developer, local projects *(default)* |
| **T2** | Redis | Team environments, shared state |
| **T3** | NATS JetStream | Production: distributed, fault-tolerant, multi-node |

All tiers implement the same 8 protocol interfaces (event bus, state store, task queue, agent registry, session manager, sandbox adapter, knowledge base, tool registry). Swap backends at config time -- no code changes.

## Packages

| Package | Description |
|---------|-------------|
| **memfun-core** | Types, config (`memfun.toml`), logging, errors |
| **memfun-runtime** | 8 protocol interfaces, 4 backend tiers, BaseAgent, `@agent` decorator |
| **memfun-agent** | Context-first solver, 9 specialist agents, workflow engine, RLM, code map, DSPy signatures |
| **memfun-skills** | Agent Skills runtime: discovery, loading, execution, synthesis |
| **memfun-tools** | MCP tool server (FastMCP 3.0): filesystem, search, git, web tools |
| **memfun-optimizer** | Trace analysis, agent synthesis, MIPROv2 optimization, persistent memory |
| **memfun-cli** | Interactive chat, setup wizard, slash commands, all CLI commands |

## Configuration

```toml
# memfun.toml
[project]
name = "my-project"

[llm]
provider = "anthropic"              # anthropic | openai | ollama | custom
model = "claude-sonnet-4-5-20250514"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.0
max_tokens = 128000

[backend]
tier = "sqlite"                     # in-process | sqlite | redis | nats
sqlite_path = ".memfun/memfun.db"

[sandbox]
backend = "local"                   # local | docker | modal
timeout_seconds = 30

[web]
search_backend = "duckduckgo"       # duckduckgo | brave | tavily | searxng
```

## Built-in Skills

8 portable skills following the [Agent Skills](https://agentskills.io) standard:

| Skill | Description |
|-------|-------------|
| `analyze-code` | Analyze code structure, quality, and issues |
| `review-code` | Structured code review with actionable feedback |
| `fix-bugs` | Diagnose and fix bugs from description |
| `explain-code` | Explain how code works |
| `generate-tests` | Generate test cases for code |
| `security-audit` | Security vulnerability analysis |
| `refactor` | Refactor code for improved quality |
| `ask` | General-purpose coding questions |

Skills are defined as `SKILL.md` files and are portable across Claude Code, Codex CLI, Cursor, Gemini CLI, and 20+ other AI tools.

## Development

```bash
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync

# Run tests (597 pass, 88 skip for Redis/NATS without servers)
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run pyright

# Run in development
uv run memfun
```

## Tech Stack

- **Python 3.12+**, asyncio, `typing.Protocol`, dataclasses
- **DSPy 2.6+** -- structured signatures, RLM module, MIPROv2 optimizer
- **FastMCP 3.0** -- MCP server with composition and proxy
- **aiosqlite** -- async SQLite (T1 backend)
- **redis** -- async Redis (T2 backend)
- **nats-py** -- NATS JetStream (T3 backend)
- **Typer + Rich** -- CLI and terminal UI
- **tree-sitter + ast-grep** -- AST parsing and code search
- **DuckDuckGo Search** -- zero-config web search
- **Ruff** -- linting and formatting
- **pytest + pytest-asyncio** -- testing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, conventions, and pull request process.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and security practices.

## License

Apache License 2.0. See [LICENSE](LICENSE).
