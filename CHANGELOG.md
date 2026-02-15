# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.6 (2026-02-15)

### Added

- **Code map (repo structure index)**: Aider-style compact index of classes,
  functions, and methods with signatures extracted from source files.  Uses
  Python `ast` module for `.py` files, regex for JS/TS/Go/Rust/Java.
  Auto-injected into ContextPlanner, WorkflowEngine, and chat triage context
  so the LLM sees codebase structure without reading every file.
- **35 new tests**: Full coverage for code map extraction (Python, JS/TS,
  Go, Rust, Java), build_code_map, code_map_to_string formatting and
  truncation.

### Changed

- ContextPlanner now receives code definitions (classes, functions, methods)
  instead of just file paths and sizes, dramatically improving file selection
  accuracy for large projects.
- Chat triage context (`_scan_cwd_context`) now includes a compact code map
  and reads 5 source files (down from 10) since the code map provides
  structural overview.

## 0.1.4 (2026-02-13)

### Added

- **Context-First Solver**: Replaces the wasteful RLM iteration loop with a
  2-call pipeline: gather context (pure I/O) then solve in one shot (1 LLM call).
  Falls back to RLM on failure.  Projects under 200 KB get all files read
  with zero planning calls.
- **Verification/lint loop**: After file operations, auto-detects project
  linters (ruff, eslint, go vet, cargo check) and feeds errors back to the
  LLM for up to 2 fix cycles.
- **5 new specialist agents**: WebSearchAgent, WebFetchAgent, PlannerAgent,
  DebugAgent, SecurityAgent (9 total).
- **Shared context gathering**: WorkflowEngine gathers project context
  ONCE before fan-out, so specialists don't waste iterations rediscovering
  files independently.
- **Per-agent display transparency**: Multi-agent results show per-agent
  breakdown (iterations, ops, duration) in the chat UI.
- **Token tracking**: Context-first solver captures DSPy token usage and
  displays it in the result metadata.

### Fixed

- **Double-triage bug**: Chat CLI now triages ONCE and passes the result
  to the coding agent via payload, eliminating redundant LLM calls.
- **Triage classification**: QueryTriage signature rewritten to aggressively
  classify code-change requests as "task" (triggering multi-agent workflow)
  vs read-only "project" exploration.
- **Version banner**: Fixed version display in chat CLI banner.

## 0.1.3 (2026-02-13)

### Added

- **Multi-agent coordination system (Phase 6)**: Task decomposition into
  parallel sub-tasks with shared specifications, specialist agents
  (FileAgent, CoderAgent, TestAgent, ReviewAgent), WorkflowEngine with
  decompose -> fan-out -> review -> revise -> merge pipeline.
- **QueryResolver**: Resolves deictic references ("fix this", "2", "do it")
  using conversation history before triage dispatch.
- **TaskDecomposer**: DSPy-powered DAG decomposition with cycle detection,
  parallelism group inference, and single-task fallback.
- **SharedSpec store**: Cross-agent shared specification for alignment,
  file ownership tracking, and conflict detection.
- **4 specialist agents**: FileAgent (read-only analysis), CoderAgent
  (code generation), TestAgent (test writing), ReviewAgent (quality review).
  All registered via `@agent` decorator.
- **WorkflowEngine**: Orchestrates multi-agent workflows with parallel
  execution via `asyncio.gather()`, review/revision loops (up to 2 rounds),
  and graceful single-agent fallback.
- **Chat CLI integration**: `/agents` and `/workflow` slash commands,
  workflow-aware triage routing, `WorkflowResult` display.
- **87 new tests**: Full coverage for query_resolver, decomposer,
  shared_spec, specialists, and workflow modules.
- **RLM action pressure**: Iteration budget signals and stall detection
  to prevent the agent from endlessly re-reading files without acting.

### Changed

- Orchestrator default timeout bumped from 30s to 120s for RLM-based agents.
- `memfun-agent` `__init__.py` exports all Phase 6 modules.

## 0.1.2 (2026-02-12)

### Fixed

- Triage no longer misclassifies short follow-ups ("2", "yes", "option 1") as
  standalone questions. QueryTriage now receives recent conversation history.
- Short queries (<20 chars) with conversation history automatically route to
  the project/RLM path instead of direct answer.
- Last assistant response context increased from 400 to 2000 chars so numbered
  option lists are not truncated before reaching the triage or agent.

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
