# Contributing to Memfun

Thank you for your interest in contributing to Memfun. This guide covers
everything you need to get started.

## Prerequisites

- **Python 3.12+** -- Memfun requires Python 3.12 or later.
- **uv** -- We use [uv](https://docs.astral.sh/uv/) for dependency management
  and workspace orchestration. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/memfun/memfun.git
cd memfun

# Install all dependencies (including dev tools)
uv sync

# Verify everything works
uv run pytest

# Run the linter
uv run ruff check .

# Run the type checker
uv run pyright
```

The project is organized as a uv workspace with 7 packages under `packages/`.
Running `uv sync` at the root installs all packages in development mode.

## Code Conventions

All code in this project follows these conventions:

- **Future annotations** -- Every module begins with
  `from __future__ import annotations`.
- **Type annotations** -- Every function signature has full type annotations.
  Pyright runs in strict mode.
- **Ruff** -- Linting and formatting. Run `uv run ruff check .` and
  `uv run ruff format .`. The configuration lives in the root `pyproject.toml`.
- **Frozen dataclasses** -- Value types use
  `@dataclass(frozen=True, slots=True)` for immutability and performance.
- **Protocol interfaces** -- All runtime interfaces use `typing.Protocol` with
  `@runtime_checkable`.
- **Parameterized SQL** -- Never use string concatenation for SQL queries.
  Always use parameterized queries.
- **Async by default** -- IO-bound operations are async. All async operations
  use proper `await` with timeout handling.
- **Conventional commits** -- Commit messages follow the
  [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Project Structure

```
memfun/
  packages/
    memfun-core/       # Shared types, config, errors, logging
    memfun-runtime/    # Protocol interfaces, backends (T0-T3), BaseAgent
    memfun-agent/      # RLM agent, MCP Tool Bridge, trace collection
    memfun-skills/     # Agent Skills runtime
    memfun-tools/      # MCP tool server (FastMCP)
    memfun-optimizer/  # Self-optimization pipeline
    memfun-cli/        # CLI application (Typer + Rich)
  skills/              # Built-in Agent Skills definitions
  agents/              # Built-in agent definitions (AGENT.md)
  tests/               # Integration and end-to-end tests
  docs/                # Documentation (MkDocs)
```

## Testing

```bash
# Run the full test suite
uv run pytest

# Run tests for a specific package
uv run pytest packages/memfun-core/tests/

# Run with coverage
uv run pytest --cov

# Run conformance tests (parameterized across all backends)
uv run pytest tests/ -k conformance
```

Tests live in two places:

- `packages/<name>/tests/` -- Unit tests for each package.
- `tests/` -- Integration and end-to-end tests.

When adding new functionality, include tests that cover both the success path
and error cases.

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name
   (e.g. `feat/add-skill-caching` or `fix/sqlite-wal-lock`).
2. **Make your changes** following the code conventions above.
3. **Add or update tests** to cover your changes.
4. **Run the full check suite** before pushing:
   ```bash
   uv run ruff check .
   uv run pyright
   uv run pytest
   ```
5. **Open a pull request** against `main` with a clear description of what
   changed and why.
6. **Address review feedback** -- maintainers may request changes before
   merging.

All pull requests must pass CI (lint, type check, tests) before merging.

## Reporting Issues

Open a GitHub issue with a clear description, reproduction steps, and the
relevant environment details (OS, Python version, backend tier in use).

## License

By contributing to Memfun, you agree that your contributions will be licensed
under the Apache License 2.0.
