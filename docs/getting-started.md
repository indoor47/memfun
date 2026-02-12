# Getting Started

This guide walks you through installing Memfun, initializing a project, and
running your first coding session.

## Prerequisites

- **Python 3.12 or later** -- Memfun requires Python 3.12+.
- **An LLM API key** -- By default, Memfun uses Anthropic's Claude. Set the
  `ANTHROPIC_API_KEY` environment variable before running.

## Installation

### From PyPI

```bash
pip install memfun
```

### From Source (for development)

```bash
git clone https://github.com/memfun/memfun.git
cd memfun
pip install uv
uv sync
```

## Verify Installation

```bash
memfun version
```

This should print the installed version (e.g. `memfun 0.1.0`).

## Initialize a Project

Navigate to the root of your codebase and run:

```bash
memfun init
```

The interactive wizard will:

1. Ask for a project name (defaults to the directory name).
2. Prompt you to select an LLM provider and model.
3. Choose a backend tier (SQLite is the default).
4. Configure sandbox settings.
5. Write a `memfun.toml` configuration file.

After initialization, a `.memfun/` directory is created to store the local
database and runtime state.

## First Run

### Interactive Chat

Launch the interactive chat interface for a conversational coding session:

```bash
memfun chat
```

You can ask the agent questions, request code changes, and invoke skills
directly from the chat prompt.

### One-Shot Commands

Run a single command without entering the interactive session:

```bash
# Analyze code at a path
memfun analyze src/

# Review code
memfun review src/auth/

# Ask a question
memfun ask "How does the authentication flow work?"

# Fix a bug
memfun fix "NoneType error in parse_config when config file is missing"

# Explain code
memfun explain src/core/engine.py
```

### Skill Invocation

Invoke a skill directly using the slash-command syntax:

```bash
memfun /review-code
memfun /security-audit
memfun /generate-tests
```

Or use the `skill` subcommand:

```bash
memfun skill list
memfun skill info review-code
memfun skill invoke review-code
```

## Configuration

The `memfun init` wizard creates a `memfun.toml` file. You can also edit it
manually:

```toml
[project]
name = "my-project"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[backend]
tier = "sqlite"

[sandbox]
backend = "local"
timeout_seconds = 30
```

See the [Configuration](configuration.md) reference for all available options.

## Next Steps

- Read the [Architecture](architecture.md) guide to understand how the
  packages fit together.
- Explore the [CLI Reference](cli-reference.md) for all available commands.
- Check the [Configuration](configuration.md) guide for advanced options
  like Redis/NATS backends and Docker sandboxes.
