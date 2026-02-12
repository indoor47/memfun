# Memfun

**An autonomous coding agent with infinite memory and scalable multi-agent orchestration.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

---

## Why Memfun?

Every coding agent today hits the same wall: **context windows are finite**. Feed a
100k-line codebase to GPT-4, Claude, or any frontier model and the agent either
truncates, hallucinates, or gives up. Multi-file refactors across large monorepos
remain out of reach. Agents forget what you told them two conversations ago. And
scaling from one agent to a coordinated team? Most frameworks don't even try.

Memfun solves this with two foundational ideas:

### 1. Infinite Memory via Recursive Language Models (RLM)

Instead of cramming an entire codebase into a single prompt, Memfun uses the
**Recursive Language Model** pattern from DSPy. The agent gets a sandboxed Python
REPL where variable state lives *outside* the token window. It can read files,
parse ASTs, run searches, store intermediate results in variables, and make
recursive sub-LLM calls -- all without exceeding context limits.

This means the agent's effective memory is **unbounded**. It can explore a million-
line codebase by navigating it programmatically -- reading what it needs, storing
what matters, and reasoning over the accumulated state. The token window becomes a
*working register*, not a hard ceiling.

The same mechanism powers persistent learning. After every conversation turn, the
agent extracts reusable knowledge (your preferences, project patterns, technical
details) and stores it in a searchable database with TF-IDF retrieval. Before each
turn, relevant memories are retrieved and injected as highest-priority context. The
agent genuinely learns from you over time -- it won't ask you what port to use
twice.

```
Traditional Agent:           Memfun RLM Agent:
┌─────────────────┐         ┌─────────────────┐
│  Context Window  │         │  Context Window  │ <- working register
│  (128k tokens)   │         │  (128k tokens)   │
│                  │         ├─────────────────┤
│  [entire codebase│         │  REPL Variables  │ <- unbounded state
│   crammed in]    │         │  (files, ASTs,   │
│                  │         │   search results,│
│  [truncated...]  │         │   intermediate   │
│                  │         │   computations)  │
│                  │         ├─────────────────┤
│                  │         │  Persistent DB   │ <- infinite memory
│                  │         │  (learnings,     │
│                  │         │   preferences,   │
│                  │         │   project facts) │
└─────────────────┘         └─────────────────┘
```

### 2. Scalable Multi-Agent Orchestration

A single agent isn't enough for complex software engineering. You need an architect
to plan, a coder to implement, a reviewer to catch bugs, and a tester to validate.
Memfun provides a **pluggable runtime** with four deployment tiers that let you
scale from a single laptop to a distributed cluster without changing code:

| Tier | Backend | Use Case |
|------|---------|----------|
| T0 | In-Process (asyncio) | Unit tests, CI, instant startup |
| T1 | SQLite | Single developer, local projects (default) |
| T2 | Redis | Team environments, shared state |
| T3 | NATS JetStream | Production: distributed, fault-tolerant, multi-node |

All four tiers implement the same 8 protocol interfaces (event bus, state store,
task queue, agent registry, session manager, sandbox adapter, knowledge base,
tool registry). Your agents, skills, and tools work identically across all tiers.

Agents are defined as **AGENT.md** files -- human-readable markdown documents that
specify capabilities, constraints, and delegation rules. The orchestrator
coordinates agents dynamically, routing tasks to the right specialist.

```
┌─────────────────────────────────────────────────┐
│                  Orchestrator                     │
│    (routes tasks, manages agent lifecycle)        │
├──────────┬──────────┬──────────┬────────────────┤
│ Architect│  Coder   │ Reviewer │   Planner      │
│ (plans   │ (RLM +   │ (code    │  (decomposes   │
│  design) │  REPL)   │  review) │   tasks)       │
├──────────┴──────────┴──────────┴────────────────┤
│            Memfun Runtime (T0-T3)                │
│  Event Bus │ State Store │ Task Queue │ Registry │
└─────────────────────────────────────────────────┘
```

### What Makes This SOTA?

Most coding agents are thin wrappers around a single LLM call. The ambitious ones
add tool calling or RAG. Memfun goes further on every axis:

| Capability | Typical Agents | Memfun |
|-----------|---------------|--------|
| **Context handling** | Truncate to fit window | RLM: navigate programmatically, unbounded |
| **Memory** | None (stateless) or basic RAG | Persistent TF-IDF DB + auto-extraction after every turn |
| **Multi-agent** | Sequential chains | True orchestration with event bus, delegation, agent registry |
| **Scaling** | Single process | 4 tiers: in-process to distributed NATS cluster |
| **Tool interface** | Custom per-agent | MCP standard (FastMCP 3.0), portable across tools |
| **Self-improvement** | None | Trace analysis, MIPROv2 optimization, agent/skill synthesis |
| **Skills** | Hard-coded capabilities | Agent Skills standard (agentskills.io), portable across 20+ AI tools |

The combination of **RLM for infinite context** + **pluggable multi-agent runtime** +
**persistent learning memory** + **self-optimization from traces** is, to our
knowledge, unique in open-source coding agents.

## Features

- **RLM Architecture** -- DSPy Recursive Language Models handle codebases 100x
  beyond native context windows via sandboxed REPL with recursive sub-LLM calls
- **Persistent Learning** -- Agent automatically extracts and remembers your
  preferences, project patterns, and technical details across sessions
- **Pluggable Backends** -- Four tiers (In-Process, SQLite, Redis, NATS JetStream)
  with identical protocol interfaces; swap at config time
- **Multi-Agent Orchestration** -- AGENT.md definitions, delegation system, and
  built-in agents (architect, orchestrator, reviewer, planner)
- **Agent Skills** -- 8 built-in portable skills following the agentskills.io
  open standard; synthesize new skills from execution traces
- **MCP Tool Integration** -- FastMCP 3.0 tool server with filesystem, search,
  git, web fetch, and web search; agents and skills exposed as MCP tools
- **Interactive Chat** -- Rich terminal UI with progress display, timer/token
  tracking, slash commands, web search, and conversation history
- **Self-Optimization** -- Trace collection, MIPROv2 optimization, agent synthesis,
  and skill effectiveness tracking
- **Security by Design** -- Parameterized SQL, SSRF prevention, sandbox isolation,
  trust tiers, path traversal protection, and secret management

## Quick Start

```bash
# Install with uv (recommended)
uv tool install memfun-cli

# Or install from source
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync
uv tool install packages/memfun-cli

# Initialize (first-time setup: LLM provider + project config)
memfun init

# Start chatting
memfun
```

On first run, `memfun init` walks you through:
1. **LLM provider** -- Anthropic (Claude), OpenAI, Ollama, or custom endpoint
2. **API key** -- securely stored in `~/.memfun/credentials.json`
3. **Project backend** -- SQLite (default), in-memory, Redis, or NATS
4. **Sandbox** -- local (default), Docker, or Modal

Then just run `memfun` to start an interactive chat session.

### Interactive Chat

```
$ memfun

 ███╗   ███╗███████╗███╗   ███╗███████╗██╗   ██╗███╗   ██╗
 ████╗ ████║██╔════╝████╗ ████║██╔════╝██║   ██║████╗  ██║
 ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██║   ██║██╔██╗ ██║
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║   ██║██║╚██╗██║
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║     ╚██████╔╝██║ ╚████║
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═══╝

memfun > build me a REST API with user auth

  Plan:
    ✓ 1. Analyze project structure  (thought for 3.2s)
    ✓ 2. Create Flask app with JWT auth  (thought for 8.7s)
    ✓ 3. Add user model and routes  (thought for 12.1s)
    ✓ 4. Write tests  (thought for 6.4s)

  Created app.py, models.py, routes/auth.py, tests/test_auth.py

  1m 23s  •  14.2k tokens

memfun > I prefer port 8080 for dev servers

  Remembered (project, Preferences): I prefer port 8080 for dev servers

memfun > now add a /health endpoint

  # Agent automatically uses port 8080 (learned from memory)
  ...
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/remember <text>` | Remember a preference or fact |
| `/memory` | View current memory contents |
| `/forget <target>` | Forget a memory entry (by number or text) |
| `/context` | Rescan project files |
| `/traces` | List recent execution traces |
| `/model` | Show or switch LLM model |
| `/clear` | Clear conversation history |
| `/exit` | Exit (or Ctrl+D) |

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

```
┌─────────────────────────────────────────────────────────┐
│                      memfun-cli                          │
│  Interactive chat, slash commands, setup wizard, CLI      │
├────────┬──────────┬──────────┬──────────┬───────────────┤
│memfun- │ memfun-  │ memfun-  │ memfun-  │   memfun-     │
│agent   │ skills   │ tools    │optimizer │   runtime     │
│        │          │          │          │               │
│ RLM    │ Discover │ MCP      │ Trace    │ 8 Protocol    │
│ coding │ Load     │ server   │ analysis │ interfaces    │
│ agent  │ Execute  │ Code+Web │ Agent    │ 4 backend     │
│ Traces │ Synth.   │ Gateway  │ synth.   │ tiers         │
│        │          │          │ MIPROv2  │ BaseAgent     │
├────────┴──────────┴──────────┴──────────┴───────────────┤
│                      memfun-core                         │
│         Config, types, errors, logging                    │
└─────────────────────────────────────────────────────────┘
```

### Packages

| Package | Description |
|---------|-------------|
| **memfun-core** | Shared types, config (`memfun.toml`), logging, errors |
| **memfun-runtime** | Pluggable runtime: 8 protocol interfaces, 4 backend tiers, BaseAgent |
| **memfun-agent** | RLM coding agent, DSPy signatures, MCP Tool Bridge, trace collection |
| **memfun-skills** | Agent Skills: discovery, loading, execution, synthesis, marketplace |
| **memfun-tools** | MCP tool server (FastMCP 3.0): code, git, filesystem, web tools |
| **memfun-optimizer** | Self-optimization: trace analysis, agent synthesis, MIPROv2, persistent memory |
| **memfun-cli** | CLI application: interactive chat, setup wizard, all commands |

### How RLM Works

The Recursive Language Model pattern separates the agent's *reasoning* from its
*memory*. Instead of one massive prompt, the agent operates in iterations:

```python
# Simplified RLM loop (actual implementation in memfun-agent)
for iteration in range(max_iterations):
    # Agent sees: task + REPL state (variables, not raw files)
    code = llm.generate_code(task, repl_state)

    # Execute in sandboxed REPL -- results stored as variables
    output = sandbox.execute(code)
    # e.g., code: files = read_file("src/auth.py")
    #       code: ast = parse_ast(files)
    #       code: issues = search("TODO|FIXME", "src/")

    # Agent can make sub-LLM calls for complex reasoning
    # e.g., code: analysis = sub_llm("analyze this function", func_code)

    # Check if task is complete
    if task_complete(output):
        return output
```

Each iteration, the agent can read files, run searches, parse code, call sub-LLMs,
and accumulate state in REPL variables. The context window only needs to hold the
current iteration's reasoning -- not the entire codebase. This is what enables
Memfun to handle repositories that are orders of magnitude larger than any model's
native context window.

### How Persistent Memory Works

After every conversation turn, the agent runs a learning extraction pipeline:

1. **Extract**: DSPy `LearningExtraction` signature analyzes the conversation and
   identifies reusable knowledge (preferences, patterns, technical details)
2. **Store**: Learnings are persisted to a SQLite-backed MemoryStore with TF-IDF
   indexing for efficient retrieval
3. **Retrieve**: Before each turn, relevant memories are retrieved via TF-IDF
   search and injected as highest-priority context
4. **Visible layer**: Learnings also appear in `.memfun/MEMORY.md` -- a human-
   readable, editable file you can manage with `/remember` and `/forget`

The database-backed memory scales to thousands of entries. The agent doesn't just
remember what you told it -- it learns patterns from how you work and applies them
automatically.

## Configuration

Memfun reads configuration from `memfun.toml` in the project root:

```toml
[project]
name = "my-project"

[llm]
provider = "anthropic"              # anthropic | openai | ollama | custom
model = "claude-sonnet-4-5-20250514"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.0
max_tokens = 8192

[backend]
tier = "sqlite"                     # in-process | sqlite | redis | nats
sqlite_path = ".memfun/memfun.db"

[sandbox]
backend = "local"                   # local | docker | modal
timeout_seconds = 30

[web]
search_backend = "duckduckgo"       # duckduckgo | brave | tavily | searxng
```

### Backend Tiers

**T0 -- In-Process** (asyncio queues, zero dependencies)
Best for: unit tests, CI pipelines, quick experiments.

**T1 -- SQLite** (WAL-mode, single-file, default)
Best for: individual developers, local projects. Zero infrastructure.

**T2 -- Redis** (pub/sub, streams, shared state)
Best for: team environments where multiple developers share agent state.

**T3 -- NATS JetStream** (distributed, fault-tolerant, multi-node)
Best for: production deployments. Horizontal scaling, persistence, clustering.
NATS is a single binary with zero external dependencies -- no ZooKeeper, no Kafka,
no Pulsar. Apache 2.0 licensed.

## Built-in Skills

Memfun ships with 8 portable skills following the [Agent Skills](https://agentskills.io) standard:

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

Skills are defined as `SKILL.md` files and are portable across Claude Code, Codex
CLI, Cursor, Gemini CLI, and 20+ other AI tools that support the Agent Skills
standard.

## Built-in Agents

| Agent | Role |
|-------|------|
| `architect` | Plans system design and technical approach |
| `orchestrator` | Coordinates multi-agent workflows |
| `reviewer` | Reviews code for correctness and quality |
| `planner` | Decomposes complex tasks into steps |

Define custom agents as `AGENT.md` files in `.memfun/agents/` or `agents/`.

## Development

```bash
# Clone and setup
git clone https://github.com/indoor47/memfun.git
cd memfun
uv sync

# Run tests (304 pass, 88 skip for Redis/NATS without servers)
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run pyright

# Run the CLI in development (editable install)
uv run memfun
```

## Tech Stack

- **Python 3.12+** with asyncio, `typing.Protocol`, dataclasses
- **DSPy 2.6+** -- RLM module, MIPROv2 optimizer, structured signatures
- **FastMCP 3.0** -- MCP server framework with composition and proxy
- **aiosqlite** -- async SQLite for T1 backend
- **redis** -- async Redis for T2 backend
- **nats-py** -- NATS JetStream for T3 backend
- **Typer + Rich** -- CLI framework with terminal UI
- **tree-sitter + ast-grep** -- AST parsing and code search
- **DuckDuckGo Search** -- zero-config web search (no API key required)
- **Ruff** -- linting and formatting
- **pytest + pytest-asyncio** -- testing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code conventions,
and the pull request process.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and security practices.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.
