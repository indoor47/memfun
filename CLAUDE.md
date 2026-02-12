# Memfun - Autonomous Agent System

## Project Overview

Memfun is an open-source autonomous coding/reasoning agent that combines three key technologies:

1. **Memfun Runtime** - A pluggable, backend-agnostic infrastructure layer for deploying, coordinating, and scaling AI agents. Provides event-driven runtime, unified event bus, state management, MCP integration, and agent registry with tiered backends: In-Process (T0), SQLite (T1, default), Redis (T2), NATS JetStream (T3).

2. **DSPy RLM (Recursive Language Models)** - A DSPy module that lets LLMs explore large contexts through a sandboxed Python REPL with recursive sub-LLM calls. Handles contexts 2 orders of magnitude beyond native windows by separating variable space from token space.

3. **Agent Skills (agentskills.io)** - The open standard for portable AI agent capabilities. Memfun discovers, loads, activates, and executes skills; packages its own capabilities as portable skills; and synthesizes new skills from execution traces. Skills work across Claude Code, Codex CLI, Cursor, Gemini CLI, and 20+ other AI tools.

The vision: Build a production-ready autonomous agent that uses RLM patterns for massive codebase handling, a pluggable runtime for multi-agent orchestration, Agent Skills as the user-facing capability layer, and self-optimizes by discovering agent architectures and skills from RLM execution traces.

## Project Phase

This project is in the **production hardening** phase. Phases 1-4 are complete. Current focus: **Phase 5 (Production & Release)**.

Completed:
- Phase 1: Core foundation (memfun-core, protocols, T0+T1 backends, RuntimeBuilder, CI)
- Phase 2: Single-agent (RLM agent, LocalSandbox, MCP tools+gateway, CLI, 4 skills, trace collector)
- Phase 3: Multi-agent (T2 Redis + T3 NATS backends, AgentManager + Orchestrator, AGENT.md system, skill execution pipeline, Docker+Modal sandboxes, 4 more skills, 4 built-in agents, skills+agents as MCP tools, multi-agent CLI, conformance tests)
- Phase 4: Self-optimization (trace analysis pipeline, agent synthesis, MIPROv2 optimization, persistent memory, skill effectiveness tracking, marketplace foundation, interactive chat CLI)

In progress:
- Phase 5: Production & release (tests for Phase 4 modules, documentation, Docker, CI/CD, security audit, packaging)

## Monorepo Structure

```
memfun/
  packages/
    memfun-core/          # Shared types, config (memfun.toml), logging, errors
    memfun-tools/         # MCP tool server (FastMCP), repo map, code/git/fs/web tools
    memfun-agent/         # RLM agent, MCP Tool Bridge, trace collection
    memfun-runtime/       # Pluggable runtime: protocols (8), backends (T0-T3), BaseAgent, @agent
    memfun-optimizer/     # Self-optimization: trace analysis, agent synthesis, MIPROv2
    memfun-skills/        # Agent Skills runtime: discovery, loading, execution, synthesis
    memfun-cli/           # CLI app (Typer): setup wizard, commands, skill invocation, display
  skills/                 # Built-in Agent Skills (analyze-code, review-code, security-audit, etc.)
  agents/                 # Built-in agent definitions as AGENT.md (agent-architect, orchestrator, etc.)
  docker/                 # Docker Compose for NATS/Redis, agent Dockerfile
  docs/                   # MkDocs documentation site
  .github/workflows/      # CI/CD pipelines
```

## Agent Ecosystem

This project uses 18 specialized implementation agents organized into 9 groups:

### Core Infrastructure
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `runtime-implementer` | Protocol interfaces, RuntimeContext, BaseAgent, @agent, T0 backend | `/runtime-implementer` |
| `sqlite-backend` | T1 SQLite backend: WAL-mode event bus, KV store, sessions, registry | `/sqlite-backend` |
| `event-bus-implementer` | T2 Redis and T3 NATS backends, migration system | `/event-bus-implementer` |

### Security
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `security-auditor` | Security review, trust tiers, RBAC, audit trail, secret management | `/security-auditor` |
| `sandbox-implementer` | SandboxAdapter: Local/Docker/Modal sandbox backends, isolation | `/sandbox-implementer` |

### Agent System
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `agent-system` | Agent definition format (AGENT.md), discovery, registration, delegation, orchestration | `/agent-system` |

### Intelligence Layer
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `rlm-engine` | RLM coding agent, MCP Tool Bridge, trace collection, DSPy signatures | `/rlm-engine` |
| `trace-pipeline` | Self-optimization: trace analysis, agent synthesis, MIPROv2, A/B testing | `/trace-pipeline` |

### MCP & Tools
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `mcp-gateway` | FastMCP 3.0 gateway, providers, transforms, agent-as-tool, skills-as-tool | `/mcp-gateway` |
| `code-tools` | MCP tools: filesystem, search (ripgrep/ast-grep), git, repo map | `/code-tools` |
| `web-tools` | MCP tools: web_fetch (URL->markdown), web_search (pluggable backends), SSRF prevention | `/web-tools` |

### Skills
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `skills-implementer` | Agent Skills runtime: discovery, loading, execution, synthesis, marketplace | `/skills-implementer` |

### CLI & UX
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `cli-implementer` | Typer CLI: setup wizard, analyze/fix/review/ask, skill invocation, agent mgmt | `/cli-implementer` |

### Quality & Testing
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `backend-conformance-tester` | Protocol conformance tests parameterized across all 4 backends | `/backend-conformance-tester` |
| `integration-tester` | End-to-end tests: agent lifecycle, RLM flow, multi-agent, security | `/integration-tester` |

### Cross-Cutting
| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `code-reviewer` | Code quality review: correctness, consistency, types, errors, docs | `/code-reviewer` |
| `debug-detective` | Bug diagnosis: async issues, race conditions, backend bugs, perf | `/debug-detective` |
| `devops-deployer` | CI/CD, Docker, NATS deployment, PyPI packaging, monitoring | `/devops-deployer` |

## Agent Interaction Map

```
                    cli-implementer
                      /    |    \
              skill invoc  |  agent mgmt commands
                    |      |       |
             skills-impl   |   agent-system
                    |      |    /    |    \
             (skill reg) mcp-gateway |  (AGENT.md format, delegation, orchestration)
                         /  |  \    |
                   code-tools | (agent-as-tool + skills-as-tool)
                   web-tools  |
                    |         |
                    +--- rlm-engine ---+
                        /         \
              sandbox-implementer  trace-pipeline
                                    |
                              (synthesizes agents + skills via agent-system format)

  runtime-implementer <-- defines 8 Protocol interfaces for:
       |          |         |           |
  sqlite-backend  |   event-bus-impl  skills-implementer
                  |
          (all backends tested by)
                  |
  backend-conformance-tester + integration-tester

  agent-system -- owns AGENT.md definition format, agent discovery,
                  DefinedAgent runtime, delegation system, orchestrator
                  (builds on runtime-implementer's BaseAgent + @agent)

  security-auditor -- reviews all code (including agent + skill security)
  code-reviewer -- reviews all code
  debug-detective -- diagnoses all bugs
  devops-deployer -- CI/CD for everything
```

## Workflow

### Phase 1: Foundation (Weeks 1-3)
1. `/runtime-implementer` -- implement Protocol interfaces, RuntimeContext, BaseAgent, T0 backend
2. `/sqlite-backend` -- implement T1 SQLite backend (MVP default)
3. `/code-tools` -- implement MCP tools (filesystem, search, git) + repo map engine
4. `/web-tools` -- implement web_fetch and web_search MCP tools with SSRF prevention
5. `/mcp-gateway` -- compose tools into FastMCP gateway
6. `/cli-implementer` -- build CLI skeleton with `memfun init` wizard
7. `/backend-conformance-tester` -- write conformance tests for T0 and T1
8. `/devops-deployer` -- set up CI/CD pipeline

### Phase 2: Single Agent (Weeks 4-7)
1. `/rlm-engine` -- build RLM coding agent with MCP Tool Bridge
2. `/sandbox-implementer` -- implement Local sandbox (Deno/WASM)
3. `/cli-implementer` -- add analyze, fix, review, ask commands
4. `/skills-implementer` -- implement memfun-skills package foundation (types, registry, loader, validator)
5. `/skills-implementer` -- implement 4 built-in skills (analyze-code, fix-bugs, review-code, explain-code)
6. `/cli-implementer` -- add skill invocation (`memfun /skill-name`) and `memfun skill list/info`
7. `/security-auditor` -- security review of all Phase 1-2 code, including skill validation
8. `/integration-tester` -- end-to-end tests for single-agent flow

### Phase 3: Multi-Agent (Weeks 8-12)
1. `/event-bus-implementer` -- implement T2 Redis and T3 NATS backends
2. `/runtime-implementer` -- agent lifecycle management, orchestrator pattern
3. `/agent-system` -- AGENT.md format, parser, discovery, DefinedAgent, delegation system
4. `/agent-system` -- built-in agents (agent-architect, orchestrator) in `agents/` directory
5. `/skills-implementer` -- implement skill router, activator, executor, and tool name mapping
6. `/skills-implementer` -- implement 4 more built-in skills (generate-tests, security-audit, refactor, ask)
7. `/mcp-gateway` -- skills-as-MCP-tools exposure via `skills.*` namespace, agents-as-tools from AGENT.md
8. `/cli-implementer` -- multi-agent CLI commands, `memfun agent` commands, `memfun skill install/search/create/validate`
9. `/sandbox-implementer` -- Docker and Modal sandbox backends
10. `/backend-conformance-tester` -- extend tests for T2 and T3

### Phase 4: Self-Optimization (Weeks 13-17)
1. `/trace-pipeline` -- trace analysis, agent synthesis (outputs AGENT.md via agent-system format), MIPROv2 optimization
2. `/trace-pipeline` -- skill synthesis from execution traces (SkillSynthesizer)
3. `/agent-system` -- agent-architect trace-driven optimization loop, synthesized agent validation
4. `/skills-implementer` -- skill effectiveness tracking and marketplace foundation
5. `/cli-implementer` -- trace, optimization, `memfun agent` management, and `memfun skill stats/export` CLI commands

### Phase 5: Production & Release (Weeks 18-22)
1. `/devops-deployer` -- production hardening, packaging, release automation
2. `/skills-implementer` -- full marketplace (search, install from registry), community skills support
3. `/security-auditor` -- full security audit (including skills threat model)
4. `/integration-tester` -- complete end-to-end test coverage, cross-agent skill portability tests

## Key Technologies

- **Python 3.12+** with asyncio, typing.Protocol, dataclasses
- **DSPy 2.6+** -- RLM module, MIPROv2/GEPA optimizers, signatures
- **FastMCP 3.0** -- MCP server framework with composition/proxy
- **aiosqlite** -- async SQLite for T1 backend
- **redis[hiredis]** -- async Redis for T2 backend
- **nats-py** -- NATS JetStream for T3 backend
- **Typer + Rich** -- CLI framework with terminal UI
- **tree-sitter** -- AST parsing for repo map
- **ripgrep / ast-grep** -- code search tools
- **httpx** -- async HTTP client for web fetch/search tools
- **markdownify + selectolax** -- HTML-to-markdown conversion and content extraction
- **duckduckgo-search** -- zero-config web search backend (no API key)
- **PyYAML** -- YAML frontmatter parsing for Agent Skills and Agent Definitions (yaml.safe_load only)
- **Ruff** -- linting and formatting
- **Pyright** -- type checking
- **pytest + pytest-asyncio** -- testing

## Conventions

- All code follows Python 3.12+ conventions with `from __future__ import annotations`
- Type annotations on every function signature; Pyright strict mode
- Ruff for linting/formatting; conventional commits for git
- `typing.Protocol` with `@runtime_checkable` for all interfaces
- `dataclasses(frozen=True, slots=True)` for immutable value types
- Parameterized queries only for SQL (NEVER string concatenation)
- API keys stored as env var names only, never raw values
- All async operations properly awaited with timeout handling
