# Memfun

**Autonomous coding agent: pluggable runtime + DSPy RLM + MCP + Agent Skills**

---

## Overview

Memfun is an open-source autonomous coding agent that combines a pluggable,
backend-agnostic runtime with DSPy Recursive Language Models (RLM) and the
Agent Skills open standard. It is designed to analyze, fix, review, and explain
codebases of any size -- including those that exceed the native context window
of modern LLMs by orders of magnitude.

At its core, Memfun uses the RLM pattern to let language models explore large
code contexts through a sandboxed Python REPL with recursive sub-LLM calls.
Variable state lives outside the token window, so the agent can reason over
entire repositories without hitting context limits. The pluggable runtime
supports four deployment tiers -- from zero-dependency in-process execution all
the way to distributed NATS JetStream clusters -- letting teams scale from a
single developer laptop to a multi-agent production environment without
changing application code.

Memfun exposes its capabilities through the Agent Skills standard
(agentskills.io), making every built-in action portable across Claude Code,
Codex CLI, Cursor, Gemini CLI, and 20+ other AI tools. It ships with eight
ready-to-use skills and can synthesize new ones from execution traces, closing
the loop on self-improvement.

## Key Features

- **RLM Pattern** -- Recursive Language Model architecture via DSPy; handles
  codebases 100x beyond native context windows.
- **Pluggable Backends** -- Four tiers: In-Process (T0), SQLite (T1),
  Redis (T2), NATS JetStream (T3). Same protocol interfaces, swap at config
  time.
- **Agent Skills** -- 8 built-in portable skills following the agentskills.io
  open standard; discover, load, execute, and synthesize skills at runtime.
- **MCP Tool Integration** -- FastMCP-based tool server with filesystem,
  search, git, web fetch, and web search tools; agents and skills exposed as
  MCP tools.
- **Interactive Chat** -- Rich terminal UI for conversational coding sessions
  with full tool access.
- **Self-Optimization** -- Trace collection, MIPROv2 optimization, agent
  synthesis, and skill effectiveness tracking.
- **Multi-Agent Orchestration** -- Agent definitions via AGENT.md, delegation
  system, and built-in orchestrator.
- **Security by Design** -- Parameterized SQL, SSRF prevention, sandbox
  isolation, trust tiers, and secret management.

## Quick Start

```bash
# Install from PyPI
pip install memfun

# Initialize a new project
memfun init

# Start an interactive chat session
memfun chat
```

See the [Getting Started](getting-started.md) guide for a complete walkthrough.

## Project Structure

Memfun is organized as a monorepo with 7 packages:

| Package | Description |
|---------|-------------|
| `memfun-core` | Shared types, config (`memfun.toml`), logging, errors |
| `memfun-runtime` | Pluggable runtime: 8 protocol interfaces, 4 backend tiers, BaseAgent |
| `memfun-agent` | RLM coding agent, MCP Tool Bridge, trace collection |
| `memfun-skills` | Agent Skills runtime: discovery, loading, execution, synthesis |
| `memfun-tools` | MCP tool server (FastMCP): code, git, filesystem, web tools |
| `memfun-optimizer` | Self-optimization: trace analysis, agent synthesis, MIPROv2 |
| `memfun-cli` | CLI application: setup wizard, commands, skill invocation |

## Built-in Skills

Memfun ships with 8 Agent Skills that follow the agentskills.io standard:

| Skill | Description |
|-------|-------------|
| `analyze-code` | Analyze code structure, quality, and potential issues |
| `fix-bugs` | Identify and fix bugs based on a description |
| `review-code` | Provide structured code review feedback |
| `explain-code` | Explain how code works in plain language |
| `generate-tests` | Generate test cases for existing code |
| `security-audit` | Audit code for security vulnerabilities |
| `refactor` | Suggest and apply refactoring improvements |
| `ask` | Answer questions about a codebase |

## Links

- [Getting Started](getting-started.md) -- Installation and first run
- [Architecture](architecture.md) -- Package structure and design
- [CLI Reference](cli-reference.md) -- All commands with examples
- [Configuration](configuration.md) -- Full `memfun.toml` reference
- [Contributing](https://github.com/memfun/memfun/blob/main/CONTRIBUTING.md)
- [Security Policy](https://github.com/memfun/memfun/blob/main/SECURITY.md)
- [License (Apache 2.0)](https://github.com/memfun/memfun/blob/main/LICENSE)
