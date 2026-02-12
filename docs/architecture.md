# Architecture

This document describes the package structure, backend tiers, and core design
patterns that make up Memfun.

## Package Overview

Memfun is organized as a monorepo managed by uv, with 7 packages under the
`packages/` directory:

```
memfun/
  packages/
    memfun-core/          # Shared types, config, errors, logging
    memfun-runtime/       # Protocol interfaces, backends (T0-T3), BaseAgent
    memfun-agent/         # RLM agent, MCP Tool Bridge, trace collection
    memfun-skills/        # Agent Skills runtime
    memfun-tools/         # MCP tool server (FastMCP)
    memfun-optimizer/     # Self-optimization pipeline
    memfun-cli/           # CLI application (Typer + Rich)
  skills/                 # Built-in Agent Skills definitions
  agents/                 # Built-in agent definitions (AGENT.md)
  docker/                 # Docker Compose for NATS/Redis, agent Dockerfile
  docs/                   # Documentation (MkDocs)
  tests/                  # Integration and end-to-end tests
```

## Dependency Graph

```
+-----------------------------------------------------+
|                    memfun-cli                        |
|        (Typer + Rich terminal interface)             |
+--------+----------+-----------+----------+-----------+
         |          |           |          |
+--------v--+ +-----v------+ +-v--------+ |
| memfun-   | | memfun-    | | memfun-  | |
| agent     | | skills     | | tools    | |
| (RLM,     | | (discover, | | (MCP     | |
|  DSPy,    | |  load,     | |  server, | |
|  traces)  | |  execute)  | |  code +  | |
+---------+-+ +-----+------+ |  web)    | |
          |         |         +----+-----+ |
          |         |              |        |
     +----v---------v--------------v--------v----------+
     |               memfun-runtime                    |
     |  (Protocols, BaseAgent, T0/T1/T2/T3 backends)   |
     +------------------------+------------------------+
                              |
     +------------------------v------------------------+
     |               memfun-core                       |
     |     (Config, types, errors, logging)            |
     +--------------------+----------------------------+
                          |
     +--------------------v----------------------------+
     |           memfun-optimizer                      |
     |  (Trace analysis, agent synthesis, MIPROv2)     |
     +-------------------------------------------------+
```

## Package Details

### memfun-core

The foundation layer. Provides:

- **MemfunConfig** -- Top-level configuration parsed from `memfun.toml`.
  Includes sections for LLM, backend, sandbox, and web tool settings.
- **Type definitions** -- Frozen dataclasses for messages, sessions, agent
  status, health, execution results, and sandbox handles.
- **Error hierarchy** -- A structured exception tree rooted at `MemfunError`
  with specific subclasses for backend, agent, session, sandbox, and skill
  errors.
- **Logging** -- Centralized logging setup using Python's standard library.

### memfun-runtime

The pluggable infrastructure layer. Defines 8 `typing.Protocol` interfaces
that abstract away the deployment backend. Any backend that implements these
protocols can be swapped in via configuration.

**Backend Tiers:**

| Tier | Backend | Use Case |
|------|---------|----------|
| T0 | In-Process | Zero-dependency local development, testing |
| T1 | SQLite | Single-user, single-machine (default) |
| T2 | Redis | Multi-user, shared state, pub/sub |
| T3 | NATS JetStream | Distributed, multi-agent production clusters |

Key components:

- **BaseAgent** -- Abstract base class for all agents.
- **RuntimeContext** -- Provides agents with access to runtime services.
- **RuntimeBuilder** -- Fluent API for constructing configured runtimes.
- **AgentManager** -- Lifecycle management for agent instances.
- **Orchestrator** -- Multi-agent coordination and delegation.

### memfun-agent

The intelligence layer. Implements the RLM (Recursive Language Model) coding
agent using DSPy.

- **RLM Agent** -- Uses DSPy modules to recursively explore codebases through
  a sandboxed Python REPL. Variable state is maintained outside the token
  window, enabling reasoning over contexts 100x larger than native limits.
- **MCP Tool Bridge** -- Connects the agent to MCP tool servers, translating
  agent actions into tool calls.
- **Trace Collector** -- Records execution traces for later analysis and
  optimization.

### memfun-skills

Implements the Agent Skills standard (agentskills.io):

- **Skill Discovery** -- Scans directories and registries for skill
  definitions.
- **Skill Loading** -- Parses YAML frontmatter and validates skill manifests.
- **Skill Execution** -- Runs skills through the agent runtime with proper
  sandboxing.
- **Skill Synthesis** -- Generates new skills from execution traces.

### memfun-tools

MCP tool server built on FastMCP:

- **Code Tools** -- Filesystem operations, code search (ripgrep/ast-grep), git
  operations, and repository map generation.
- **Web Tools** -- URL fetching with HTML-to-markdown conversion, web search
  with pluggable backends (DuckDuckGo, Brave, Tavily, SearXNG).
- **Gateway** -- Composes all tools into a unified MCP server. Exposes agents
  and skills as MCP tools.

### memfun-optimizer

Self-optimization pipeline:

- **Trace Analysis** -- Analyzes execution traces to identify patterns, find
  inefficiencies, and extract reusable strategies.
- **Agent Synthesis** -- Generates new agent definitions (AGENT.md format) from
  successful trace patterns.
- **MIPROv2 Optimization** -- Uses DSPy's MIPROv2 optimizer for prompt
  tuning and module optimization.

### memfun-cli

Terminal interface built on Typer and Rich:

- **Setup Wizard** -- Interactive `memfun init` that guides configuration.
- **Agent Commands** -- `analyze`, `fix`, `review`, `explain`, `ask`.
- **Skill Commands** -- `skill list`, `skill info`, `skill invoke`,
  `skill search`, `skill create`.
- **Agent Management** -- `agent list`, `agent info`.
- **Interactive Chat** -- Rich terminal UI for conversational sessions.
- **Slash Commands** -- `/skill-name` shorthand for skill invocation.

## The RLM Pattern

The Recursive Language Model pattern is the core innovation that enables Memfun
to handle large codebases. Traditional LLMs are limited by their context
window -- typically 128K-200K tokens. The RLM pattern works around this by:

1. **Separating variable space from token space.** The agent maintains Python
   variables in a sandboxed REPL. These variables can hold arbitrarily large
   data structures without consuming tokens.

2. **Recursive exploration.** The agent can spawn sub-LLM calls to analyze
   specific sections of code, summarize findings into variables, and then
   reason over those summaries at a higher level.

3. **Tool-mediated access.** The agent reads files, searches code, and
   navigates repositories through MCP tools rather than loading everything
   into the context window.

This approach enables reasoning over codebases that are 100x larger than what
fits in a single context window.

## Agent Definitions (AGENT.md)

Agents are defined using AGENT.md files -- Markdown documents with YAML
frontmatter that specify the agent's capabilities, tools, and behavior. The
`agents/` directory contains built-in agent definitions. The system supports:

- Agent discovery and registration
- Delegation between agents
- Agent-as-MCP-tool exposure
- Trace-driven agent synthesis

## Security Model

Memfun implements security at multiple layers:

- **Sandbox isolation** for code execution (Local, Docker, Modal)
- **Trust tiers** for agents and skills
- **SSRF prevention** in web tools
- **Parameterized SQL** in all database operations
- **Secret management** via environment variable references
