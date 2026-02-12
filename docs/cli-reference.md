# CLI Reference

Memfun provides a command-line interface built on Typer with Rich terminal
output. This page documents all available commands.

## Global Behavior

When invoked without a subcommand, `memfun` launches the interactive chat
interface (equivalent to `memfun chat`).

```bash
memfun
```

## Commands

### memfun init

Initialize a new Memfun project in the current directory.

```bash
memfun init
```

Runs an interactive wizard that creates a `memfun.toml` configuration file and
a `.memfun/` directory for local state. The wizard prompts for:

- Project name
- LLM provider and model
- Backend tier
- Sandbox configuration

### memfun chat

Launch the interactive chat interface.

```bash
memfun chat
```

Opens a Rich terminal UI for conversational coding sessions. You can ask
questions, request code changes, and invoke skills from the chat prompt.

### memfun ask

Ask the agent a question or assign it a task.

```bash
memfun ask "How does the authentication middleware work?"
memfun ask "What are the main entry points in this project?"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `question` | Yes | The question or task for the agent |

### memfun analyze

Analyze code at a path for structure, quality, and issues.

```bash
memfun analyze
memfun analyze src/
memfun analyze src/auth/handler.py
```

| Argument | Required | Description |
|----------|----------|-------------|
| `path` | No | File or directory to analyze (defaults to current directory) |

### memfun fix

Fix a bug based on its description.

```bash
memfun fix "TypeError in user authentication when token is expired"
memfun fix "Race condition in cache invalidation"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `description` | Yes | Description of the bug to fix |

### memfun review

Review code at a path and provide structured feedback.

```bash
memfun review
memfun review src/core/
memfun review src/api/routes.py
```

| Argument | Required | Description |
|----------|----------|-------------|
| `path` | No | File or directory to review (defaults to current directory) |

### memfun explain

Explain how code at a path works.

```bash
memfun explain
memfun explain src/engine/
memfun explain src/parser.py
```

| Argument | Required | Description |
|----------|----------|-------------|
| `path` | No | File or directory to explain (defaults to current directory) |

### memfun version

Show the installed Memfun version.

```bash
memfun version
```

### memfun config

View or edit the project configuration.

```bash
memfun config
```

## Skill Commands

### memfun skill list

List all available skills (built-in and installed).

```bash
memfun skill list
```

### memfun skill info

Show details about a specific skill.

```bash
memfun skill info review-code
memfun skill info security-audit
```

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Name of the skill |

### memfun skill search

Search for skills by keyword.

```bash
memfun skill search "testing"
memfun skill search "security"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `query` | Yes | Search query |

### memfun skill invoke

Invoke a skill by name.

```bash
memfun skill invoke review-code
memfun skill invoke generate-tests
```

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Name of the skill to invoke |

### memfun skill create

Create a new skill from a template.

```bash
memfun skill create
```

Launches an interactive wizard to scaffold a new Agent Skill definition.

## Slash Commands

Skills can be invoked directly using the slash-command shorthand:

```bash
memfun /review-code
memfun /security-audit
memfun /generate-tests
memfun /analyze-code
memfun /fix-bugs
memfun /explain-code
memfun /refactor
memfun /ask
```

This is equivalent to `memfun skill invoke <name>` but more concise.

## Agent Management Commands

### memfun agent list

List all registered agent definitions.

```bash
memfun agent list
```

### memfun agent info

Show details about a specific agent definition.

```bash
memfun agent info orchestrator
memfun agent info agent-architect
```

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Name of the agent |
