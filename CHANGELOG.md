# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.1 (2026-02-12)

### Added

- **Persistent learning memory**: Dual-layer system with MEMORY.md (visible,
  editable file) and MemoryStore (SQLite-backed TF-IDF searchable database).
  Agent auto-extracts learnings after every conversation turn.
- **Memory slash commands**: `/remember`, `/memory`, `/forget` for explicit
  memory management during chat sessions.
- **Timer and token display**: Formatted elapsed time ("2m 40s"), token count
  ("12.5k tokens"), and per-step "thought for Xs" timing in chat UI.
- **Learning diagnostic**: `/debug-learning` command to test the extraction
  pipeline and diagnose issues.
- Starter MEMORY.md created automatically during `memfun init`.

### Fixed

- Learning extraction failures now log at WARNING level (was DEBUG), making
  pipeline issues visible during development.
- Improved `_normalize_list()` to handle DSPy output formats: JSON arrays,
  newline-separated text, and string representations.
- Fixed `_insert_under_section()` to append at end of section (was inserting
  at beginning), preserving chronological order in MEMORY.md.

### Changed

- UI spacing improvements: vertical breathing room between all sections in
  progress display and operations summary.

## 0.1.0 (2026-02-11)

### Added

- Core runtime with 4 backend tiers: In-Process (T0), SQLite (T1), Redis (T2),
  NATS JetStream (T3). All backends conform to the same 8 protocol interfaces.
- RLM coding agent powered by DSPy, enabling recursive language model
  exploration of codebases that exceed native context windows by orders of
  magnitude.
- 8 built-in Agent Skills following the agentskills.io open standard:
  analyze-code, fix-bugs, review-code, explain-code, generate-tests,
  security-audit, refactor, and ask.
- Interactive chat CLI with Rich terminal UI, slash-command skill invocation,
  and full MCP tool access.
- Self-optimization pipeline: trace collection, trace analysis, agent
  synthesis via AGENT.md format, and MIPROv2 prompt optimization.
- MCP tool integration via FastMCP: filesystem, search, git, repo map,
  web fetch, and web search tools. Agents and skills are also exposed as
  MCP tools.
- Multi-agent orchestration with AGENT.md definitions, agent discovery,
  delegation system, and built-in orchestrator and architect agents.
- Sandbox isolation with Local, Docker, and Modal backends.
- Configuration via `memfun.toml` with sensible defaults.
- Monorepo workspace managed by uv with 7 packages: memfun-core,
  memfun-runtime, memfun-agent, memfun-skills, memfun-tools,
  memfun-optimizer, memfun-cli.
