<p align="center">
  <h1 align="center">Memfun</h1>
  <p align="center">
    <strong>An open-source autonomous coding agent built on three ideas:<br/>a coding engine that verifies its own work, infinite memory via Recursive Language Models,<br/>and scalable multi-agent orchestration on an event-driven runtime.</strong>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" /></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python" /></a>
    <a href="https://github.com/indoor47/memfun/actions/workflows/ci.yml"><img src="https://github.com/indoor47/memfun/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <img src="https://img.shields.io/badge/tests-597_pass-brightgreen.svg" alt="Tests" />
    <img src="https://img.shields.io/badge/packages-7-orange.svg" alt="Packages" />
  </p>
</p>

---

```
$ memfun

memfun > add JWT auth to the Flask API with refresh tokens

  Context-first: gathered 12 files via code map (48 KB)
  Solving...

  Operations:
    write  src/auth/jwt.py           JWT creation, verification, refresh
    write  src/auth/middleware.py     require_auth decorator
    edit   src/routes/api.py          added @require_auth to protected endpoints
    write  tests/test_auth.py         12 test cases
    run    ruff check src/            passed
  Verified: 0 lint errors  •  Consistency: ok

  32s  •  8.2k tokens

memfun > now refactor the database layer to async with connection pooling

  Workflow: decomposed into 3 parallel agents
    ✓ coder-agent    8 iter, 4 ops    (42s)
    ✓ test-agent     5 iter, 2 ops    (28s)
    ✓ review-agent   3 iter, 0 ops    (12s)
  Cross-agent review: no conflicts

  Operations:
    edit   src/db/connection.py       replaced sync pool with asyncpg
    edit   src/db/queries.py          async context managers throughout
    edit   src/db/models.py           async classmethod factories
    write  tests/test_db_async.py     8 new async test cases
    run    ruff check src/            passed
    run    pytest tests/test_db*.py   6 passed

  1m 48s  •  22.1k tokens
```

## Three Pillars

Memfun is built on three foundational ideas that work together. Each one solves a
hard problem in autonomous coding agents. Together, they produce something greater
than the sum of their parts.

<table>
<tr>
<td width="33%" valign="top">

### 1. The Coding Agent

A multi-strategy solver that writes, verifies, and reviews its own code. Starts
fast (2 LLM calls), escalates to parallel specialists when needed, and always
runs your linters before delivering results.

</td>
<td width="33%" valign="top">

### 2. Recursive Language Models

DSPy RLM separates reasoning from memory. The agent navigates million-line
codebases via a sandboxed REPL -- variable state lives outside the token
window. Effective context is unbounded.

</td>
<td width="33%" valign="top">

### 3. Event-Driven Runtime

A pluggable backend architecture with 4 tiers (in-process to NATS JetStream)
and 8 protocol interfaces. Same code runs on a laptop or a distributed
cluster. Agents coordinate via event bus.

</td>
</tr>
</table>

---

## Pillar 1: The Coding Agent

Every coding agent today does roughly the same thing: cram files into a prompt,
call the LLM, dump the output. If it fails, maybe retry. Memfun takes a different
approach -- it has a **three-tier escalation strategy** that starts cheap and fast,
automatically scales up for complex tasks, and verifies everything it produces.

### Context-First Solver (Tier 1)

Most tasks don't need 20 iterative loops. They need the right context and one good
answer. The context-first solver does exactly this:

1. **Code Map** -- Extracts a structural index of your codebase (classes, functions,
   methods with full signatures) using Python `ast` and regex for JS/TS/Go/Rust/Java.
   The planner sees *what's in each file*, not just filenames.
2. **Planner** (1 LLM call) -- Selects files to read, patterns to search, and
   optional web queries. Projects under 200 KB skip this entirely.
3. **Gather** (0 LLM calls) -- Reads the selected files, runs searches, fetches
   web results. Pure I/O.
4. **Solve** (1 LLM call) -- Produces structured file operations (write, edit, run)
   in a single shot with all context available.
5. **Execute + Verify** -- Applies operations, auto-detects your linter
   (ruff/eslint/go vet/cargo check), runs it, fixes errors (up to 2 cycles).
6. **Consistency Review** -- Semantic check: does the output match what was asked?
   If not, targeted polish in edit-only mode.

Total: **2 LLM calls** for the happy path. ~30 seconds. No iteration loops.

### Multi-Agent Workflow (Tier 2)

When context-first fails or truncates, Memfun automatically escalates to a full
multi-agent workflow. No user intervention needed.

```
Task Decomposer (DAG with dependency analysis)
  │
  ├── Group 1 (parallel)     FileAgent   PlannerAgent
  │                          read/analyze  plan approach
  │
  ├── Group 2 (parallel)     CoderAgent   TestAgent   SecurityAgent
  │                          implement    write tests   audit vulns
  │                              │
  │                    Each runs: RLM → Verify → Review → Polish
  │
  ├── Cross-Agent Review     ReviewAgent checks all outputs for conflicts
  │
  └── Revision               Up to 2 rounds: re-run failing agents with feedback
```

**9 specialist agents**, each with a focused system prompt and iteration cap:

| Agent | Role | Max Iterations |
|-------|------|:-:|
| **CoderAgent** | Production code generation (prefers `edit_file` over `write_file`) | 15 |
| **TestAgent** | Test writing and execution | 10 |
| **DebugAgent** | Bug diagnosis and root cause analysis | 12 |
| **ReviewAgent** | Quality review and cross-agent conflict detection | 8 |
| **FileAgent** | File reading and analysis (never creates code) | 8 |
| **SecurityAgent** | Vulnerability detection (injection, SSRF, secrets, path traversal) | 8 |
| **PlannerAgent** | Sub-problem decomposition and approach planning | 6 |
| **WebSearchAgent** | Web search via DuckDuckGo | 8 |
| **WebFetchAgent** | URL content extraction and summarization | 8 |

Every code-producing agent runs the **same quality pipeline** after its RLM loop:
auto-detect linters, verify, fix, consistency review, polish. Issues are caught
per-agent *before* the cross-agent review, not after.

### RLM Exploration (Tier 3)

Last resort for tasks that require deep iterative exploration. The agent gets a
sandboxed Python REPL and can read files, run commands, make sub-LLM calls, and
accumulate state across iterations. See [Pillar 2](#pillar-2-recursive-language-models)
for how this works.

### Quality Pipeline

Every code path -- context-first and each specialist -- runs the same post-processing:

```
Code produced
  │
  ├─ Auto-detect linters (ruff, eslint, go vet, cargo check)
  ├─ Run verification (subprocess, 0 LLM calls)
  ├─ Fix lint errors if any (1 LLM call, edit-only)
  ├─ Consistency review: does output match the request? (1 LLM call)
  └─ Polish if issues found (1 LLM call, edit-only)
```

### File Safety

Agents are explicitly prevented from destroying your code:

- **Destructive write guard** -- `write_file` on existing files is blocked if it
  would lose >30% of content. Suggests `edit_file` instead.
- **Edit-only mode** -- Fix and polish steps can only modify existing files via
  targeted `edit_file`, never `write_file`.
- **Fuzzy edit matching** -- 3 strategies (exact, whitespace-normalized, difflib
  sliding window) so edits land even when LLM output has minor differences.
- **Read caching** -- MD5 hash per file; re-reads of unchanged files return a
  short summary instead of full content, preventing context burn.
- **Stall detection** -- Warns the agent when it's looping (no action ops at
  midpoint, same file read 3+ times, approaching iteration limit).

---

## Pillar 2: Recursive Language Models

Every coding agent today hits the same wall: **context windows are finite**. Feed a
100k-line codebase into any frontier model and the agent either truncates,
hallucinates, or gives up. Multi-file refactors across large monorepos remain out
of reach.

Memfun uses the **Recursive Language Model** pattern from DSPy. Instead of cramming
an entire codebase into a single prompt, the agent gets a sandboxed Python REPL where
variable state lives *outside* the token window.

```
Traditional Agent:              Memfun RLM Agent:
┌─────────────────┐            ┌──────────────────┐
│  Context Window  │            │  Context Window   │  <-- working register
│  (128k tokens)   │            │  (128k tokens)    │
│                  │            ├──────────────────┤
│  [entire codebase│            │  REPL Variables   │  <-- unbounded state
│   crammed in]    │            │  files, ASTs,     │
│                  │            │  search results,  │
│  [truncated...]  │            │  computations     │
│                  │            ├──────────────────┤
│                  │            │  Persistent DB    │  <-- infinite memory
│                  │            │  learnings,       │
│                  │            │  preferences,     │
│                  │            │  project facts    │
└─────────────────┘            └──────────────────┘
```

The agent can read files, parse ASTs, run searches, store intermediate results in
variables, and make recursive sub-LLM calls -- all without exceeding context limits.
The token window becomes a *working register*, not a hard ceiling. This is what enables
Memfun to handle repositories that are orders of magnitude larger than any model's
native context window.

### REPL Tools

The RLM sandbox exposes these tools to the agent:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read file with MD5 caching (re-reads return short summary) |
| `write_file(path, content)` | Write with destructive-overwrite guard |
| `edit_file(path, old, new)` | Targeted edit with fuzzy matching |
| `run_cmd(cmd)` | Execute shell commands |
| `list_files(pattern)` | Glob with optional timestamps |
| `llm_query(prompt, context)` | Sub-LLM call for complex reasoning |
| `llm_query_batched(queries)` | Parallel sub-LLM calls |
| `web_search(query)` | DuckDuckGo search |
| `web_fetch(url)` | Fetch URL, convert to markdown |
| `search_history(query)` | Search conversation history |

### Persistent Memory

The same unbounded-memory principle extends across sessions. After every conversation
turn, the agent extracts reusable knowledge and stores it in two layers:

- **MEMORY.md** -- Human-readable, editable file. Always loaded as context. Manage
  with `/remember` and `/forget`.
- **SQLite MemoryStore** -- TF-IDF indexed database for efficient retrieval. Scales
  to thousands of entries.

Before each turn, relevant memories are retrieved and injected as high-priority
context. The agent learns from how you work and applies it automatically.

---

## Pillar 3: Event-Driven Runtime

A single-process agent is fine for experiments. Production needs horizontal scaling,
fault tolerance, and shared state across nodes. Most frameworks punt on this entirely.

Memfun's runtime is built from the ground up on **8 protocol interfaces** that
abstract the infrastructure layer. Every agent, tool, and skill runs identically
across all backend tiers -- swap at config time, no code changes.

```
┌──────────────────────────────────────────────────────┐
│                    Your Agents                        │
│   CoderAgent  TestAgent  ReviewAgent  CustomAgent     │
├──────────────────────────────────────────────────────┤
│                 8 Protocol Interfaces                 │
│  EventBus │ StateStore │ TaskQueue │ AgentRegistry    │
│  SessionMgr │ SandboxAdapter │ KnowledgeBase │ Tools │
├──────────────────────────────────────────────────────┤
│                   Backend Tier                        │
│  T0: asyncio  │  T1: SQLite  │  T2: Redis  │  T3: NATS│
└──────────────────────────────────────────────────────┘
```

### Four Backend Tiers

| Tier | Backend | When to Use |
|:----:|---------|-------------|
| **T0** | In-Process (asyncio queues) | Unit tests, CI pipelines, quick experiments. Zero dependencies. |
| **T1** | SQLite (WAL mode) | Individual developers, local projects. Single file, zero infrastructure. **Default.** |
| **T2** | Redis (pub/sub + streams) | Team environments. Multiple developers sharing agent state. |
| **T3** | NATS JetStream | Production. Distributed, fault-tolerant, multi-node clustering. NATS is a single binary, Apache 2.0, no ZooKeeper or Kafka needed. |

### Agent Definitions

Agents are defined as **AGENT.md** files -- human-readable markdown documents that
specify capabilities, constraints, and delegation rules. The orchestrator discovers
and routes tasks to the right specialist dynamically.

```markdown
# coder-agent

## Capabilities
- Write production code across multiple files
- Prefer edit_file over write_file for existing code
- Run linters and fix errors

## Constraints
- Max 15 iterations per task
- Never modify test files (delegate to test-agent)

## Delegation
- test-agent: when tests need writing
- review-agent: before final delivery
```

Define custom agents in `.memfun/agents/` or `agents/`.

### MCP Tool Integration

All tools are exposed via **FastMCP 3.0** (Model Context Protocol), making them
portable across any MCP-compatible AI tool:

- **Code tools**: file read/write/edit, ripgrep search, ast-grep, git operations
- **Web tools**: DuckDuckGo search, URL fetch with markdown conversion, SSRF prevention
- **Agent tools**: agents and skills exposed as MCP tools for cross-system interoperability

### Live Dashboard

Every `memfun` session launches a real-time web dashboard at `http://localhost:8081`
that shows what the agent is doing as it works:

- **Active requests** with status, elapsed time, and token count
- **Sub-task breakdown** for multi-agent workflows (which specialist is running, how many iterations)
- **Event stream** with live WebSocket updates
- **Session history** of all requests and results

Multiple terminals in the same project share a single dashboard instance via lockfile
coordination and HTTP event forwarding -- open 3 terminals, see all their work in one place.

---

## Quick Start

```bash
# One-line install
curl -fsSL https://raw.githubusercontent.com/indoor47/memfun/main/install.sh | bash

# Initialize (LLM provider, API key, backend)
memfun init

# Start chatting
memfun
```

Or install manually:

```bash
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync --all-packages
```

`memfun init` walks you through:
1. **LLM provider** -- Anthropic (Claude), OpenAI, Ollama, or custom endpoint
2. **API key** -- securely stored in `~/.memfun/credentials.json`
3. **Backend** -- SQLite (default), in-memory, Redis, or NATS
4. **Sandbox** -- local (default), Docker, or Modal

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/remember <text>` | Store a preference or project fact |
| `/memory` | View stored memories |
| `/forget <target>` | Remove a memory entry |
| `/context` | Rescan project files |
| `/agents` | Show running specialist agents |
| `/workflow` | Show current workflow DAG and status |
| `/model` | Show or switch LLM model |
| `/traces` | List recent execution traces |
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

## Configuration

```toml
# memfun.toml
[project]
name = "my-project"

[llm]
provider = "anthropic"                # anthropic | openai | ollama | custom
model = "claude-opus-4-6"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.0
max_tokens = 128000

[backend]
tier = "sqlite"                       # in-process | sqlite | redis | nats
sqlite_path = ".memfun/memfun.db"

[sandbox]
backend = "local"                     # local | docker | modal
timeout_seconds = 30

[web]
search_backend = "duckduckgo"         # duckduckgo | brave | tavily | searxng
```

## Packages

```
memfun/
├── packages/
│   ├── memfun-core/        Types, config (memfun.toml), logging, errors
│   ├── memfun-runtime/     8 protocol interfaces, 4 backend tiers, BaseAgent
│   ├── memfun-agent/       Coding agent, context-first solver, 9 specialists,
│   │                       workflow engine, RLM, code map, DSPy signatures
│   ├── memfun-tools/       MCP server (FastMCP 3.0): code, git, web tools
│   ├── memfun-skills/      Agent Skills: discovery, loading, execution, synthesis
│   ├── memfun-optimizer/   Trace analysis, agent synthesis, MIPROv2, memory
│   └── memfun-cli/         Interactive chat, setup wizard, live dashboard, CLI
├── skills/                 8 built-in Agent Skills (SKILL.md format)
├── agents/                 Built-in agent definitions (AGENT.md format)
├── evals/                  SWE-bench + Terminal-Bench evaluation harnesses
└── tests/                  597 tests (88 skipped for Redis/NATS without servers)
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

Skills are `SKILL.md` files, portable across Claude Code, Codex CLI, Cursor,
Gemini CLI, and 20+ other AI tools.

## Development

```bash
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync --all-packages

make test                       # 597 pass, 88 skip
make lint                       # 0 errors
make typecheck                  # Pyright strict mode
uv run memfun                   # run in dev mode
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Intelligence** | DSPy 2.6+ (RLM, MIPROv2, structured signatures) |
| **Tools** | FastMCP 3.0 (Model Context Protocol) |
| **Backends** | asyncio (T0), aiosqlite (T1), redis (T2), nats-py (T3) |
| **Code Analysis** | tree-sitter, ast-grep, Python ast |
| **Web** | DuckDuckGo Search, httpx, markdownify |
| **CLI** | Typer + Rich |
| **Quality** | Ruff, Pyright, pytest + pytest-asyncio |
| **Language** | Python 3.12+, asyncio, typing.Protocol |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, conventions,
and pull request process.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and security practices.

## License

Apache License 2.0. See [LICENSE](LICENSE).
