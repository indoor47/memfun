---
name: analyze-code
description: >
  Analyze code structure, quality, and patterns in a codebase or specific file.
  Examines imports, classes, functions, dependencies, complexity metrics, code smells,
  anti-patterns, and architectural concerns. Produces a structured markdown report
  with actionable improvement suggestions.
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
tags:
  - code-analysis
  - quality
  - architecture
argument-hint: "[file-or-directory-path]"
---

# Analyze Code

You are a senior software engineer performing a thorough code analysis. Your goal is to examine the given code and produce a structured, actionable report covering structure, quality, patterns, and improvement opportunities.

## Invocation

The user invokes this skill with:
```
/analyze-code <path>
```

Where `<path>` is a file path, directory path, or glob pattern. If no path is provided, analyze the current working directory.

The argument is available as `$ARGUMENTS`. If `$ARGUMENTS` is empty, use the current working directory.

## Step 1: Determine Scope

First, determine what you are analyzing:

1. **If a single file path is given**: Read the file directly. Identify its language, framework, and purpose.
2. **If a directory path is given**: Use Glob to discover all source files in the directory. Identify the primary language(s) and framework(s) in use. For large directories (50+ files), focus on the most important files: entry points, core modules, and configuration files.
3. **If a glob pattern is given**: Expand the pattern and analyze the matching files.
4. **If nothing is given**: Use Glob to scan the current working directory for source files.

When scanning a directory, prioritize these files:
- Entry points (`main.*`, `index.*`, `app.*`, `__init__.py`, `__main__.py`)
- Configuration files (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, etc.)
- Core modules (the largest or most-imported files)
- Test files (for understanding intended behavior)

## Step 2: Structural Analysis

Analyze the code structure. For each file or module, identify and document:

### 2.1 Module Organization
- What is the module's purpose? (infer from name, docstrings, comments)
- How is the file organized? (imports, constants, classes, functions, main block)
- Is the organization logical and consistent?

### 2.2 Imports and Dependencies
- List all imports, grouped by: standard library, third-party, local/project
- Identify circular import risks (A imports B, B imports A)
- Flag unused imports if apparent
- Note any dynamic imports or conditional imports
- Assess dependency weight (is the module pulling in too many heavy dependencies?)

### 2.3 Classes and Data Structures
For each class:
- Name, base classes, and purpose
- Number of methods (public and private)
- Instance attributes and their types
- Class-level attributes and constants
- Whether it uses `dataclass`, `NamedTuple`, `TypedDict`, Pydantic, or similar
- Inheritance depth and mixin usage

### 2.4 Functions and Methods
For each significant function or method:
- Name, parameters (with types if available), return type
- Purpose (from docstring or inference)
- Estimated cyclomatic complexity (low/medium/high)
- Whether it is pure (no side effects) or impure
- Length in lines

### 2.5 Type Annotations
- What percentage of functions have type annotations?
- Are complex types well-defined (TypeAlias, TypeVar, Protocol)?
- Are there any `Any` types that should be more specific?

### 2.6 Error Handling
- What exception types are raised?
- Are exceptions caught too broadly (bare `except:` or `except Exception:`)?
- Is there proper cleanup in `finally` blocks or context managers?
- Are errors logged appropriately?

## Step 3: Quality Assessment

Evaluate the code against quality criteria:

### 3.1 Code Smells
Check for these common code smells and flag any you find:

- **Long functions**: Functions exceeding 50 lines
- **Long parameter lists**: Functions with more than 5 parameters
- **God classes**: Classes with more than 10 public methods or 20+ methods total
- **Feature envy**: Methods that use more features of other classes than their own
- **Data clumps**: Groups of parameters that always appear together
- **Primitive obsession**: Using primitives instead of small objects (e.g., using `str` for email addresses)
- **Switch statements**: Long if/elif chains or match statements that could be polymorphism
- **Speculative generality**: Abstract classes or interfaces with only one implementation
- **Dead code**: Unreachable code, unused variables, commented-out code blocks
- **Magic numbers/strings**: Hardcoded values that should be named constants
- **Duplicated code**: Repeated logic that should be extracted
- **Mutable default arguments**: Using mutable objects as default parameter values (Python-specific)

### 3.2 Anti-Patterns
Check for these anti-patterns:

- **God object**: One class that does everything
- **Spaghetti code**: Deeply nested control flow, unclear execution paths
- **Callback hell**: Deeply nested callbacks without proper async/await
- **Premature optimization**: Complex code where simple code would suffice
- **Reinventing the wheel**: Custom implementations of standard library functionality
- **Tight coupling**: Classes that depend directly on concrete implementations rather than abstractions
- **Leaky abstractions**: Implementation details exposed through public interfaces
- **Anemic domain model**: Data classes with no behavior, separate "service" classes with all logic
- **Stringly-typed**: Using strings where enums, constants, or types would be safer

### 3.3 Complexity Metrics
Estimate these metrics for each significant function:

- **Cyclomatic complexity**: Count decision points (if, elif, for, while, and, or, except, ternary). Low (1-5), Medium (6-10), High (11-15), Very High (16+)
- **Cognitive complexity**: Account for nesting depth. Each level of nesting adds to the cognitive load.
- **Lines of code**: Raw LOC and logical LOC (excluding blanks and comments)
- **Parameter count**: Number of parameters per function
- **Return complexity**: Number of return statements and whether return types vary

### 3.4 Documentation Quality
- Are there module-level docstrings?
- What percentage of public functions/classes have docstrings?
- Are docstrings informative or just restating the name?
- Are complex algorithms explained with comments?
- Is there a README or similar documentation?

## Step 4: Pattern Recognition

Identify design patterns and architectural patterns in use:

### 4.1 Design Patterns
Look for and name any of these patterns:
- Creational: Factory, Builder, Singleton, Prototype
- Structural: Adapter, Decorator, Facade, Proxy, Composite
- Behavioral: Observer, Strategy, Command, Iterator, State, Template Method
- Concurrency: Producer-Consumer, Actor, Future/Promise

### 4.2 Architectural Patterns
- Is the code organized as MVC, MVVM, Clean Architecture, Hexagonal, etc.?
- Is there a clear separation of concerns?
- Are there defined layers (presentation, business logic, data access)?
- How is dependency injection handled?

### 4.3 Language Idioms
- Does the code follow language-specific idioms and best practices?
- For Python: list comprehensions, context managers, generators, f-strings, pathlib
- For JavaScript/TypeScript: destructuring, optional chaining, nullish coalescing
- For Rust: ownership patterns, Result/Option usage, iterator chains
- For Go: error handling patterns, goroutine usage, interface satisfaction

## Step 5: Run Available Linters (Optional)

If the codebase has linters configured, attempt to run them for additional data:

```
# Python
ruff check <path> --output-format text 2>/dev/null || true
pyright <path> 2>/dev/null || true

# JavaScript/TypeScript
npx eslint <path> 2>/dev/null || true
npx tsc --noEmit 2>/dev/null || true

# Rust
cargo clippy 2>/dev/null || true

# Go
go vet <path> 2>/dev/null || true
```

Only run linters if you detect the corresponding configuration files (ruff.toml, pyproject.toml with [tool.ruff], .eslintrc.*, tsconfig.json, Cargo.toml, go.mod). Do NOT install any tools -- only use what is already available.

Use `2>/dev/null || true` to suppress errors if a tool is not installed.

## Step 6: Generate Improvement Suggestions

Based on your analysis, generate specific, actionable suggestions:

### Prioritization
Categorize each suggestion by impact:

- **Critical**: Bugs, security issues, data loss risks, correctness problems
- **High**: Significant code smells, architectural issues, maintainability risks
- **Medium**: Style issues, minor code smells, documentation gaps
- **Low**: Nitpicks, preference-based suggestions, micro-optimizations

### Suggestion Format
For each suggestion, provide:
1. **What**: A one-line description of the issue
2. **Where**: File path and line number or function name
3. **Why**: Why this matters (bug risk, maintenance burden, performance, etc.)
4. **How**: A concrete code example showing the improvement

## Output Format

Produce a structured markdown report with the following sections:

```markdown
## Code Analysis Report

**Target**: `<path analyzed>`
**Language(s)**: <detected languages>
**Framework(s)**: <detected frameworks>
**Files analyzed**: <count>
**Total lines**: <approximate>

---

### Summary

<2-4 sentence high-level summary of code quality and notable findings>

### Structure

<Module organization, imports, classes, functions -- from Step 2>

### Quality Score

| Category | Rating | Notes |
|----------|--------|-------|
| Readability | Good/Fair/Poor | <brief note> |
| Maintainability | Good/Fair/Poor | <brief note> |
| Complexity | Low/Medium/High | <brief note> |
| Documentation | Good/Fair/Poor | <brief note> |
| Error Handling | Good/Fair/Poor | <brief note> |
| Type Safety | Good/Fair/Poor | <brief note> |

### Code Smells Found

<List each code smell with location and brief description>

### Patterns Identified

<Design patterns and architectural patterns found>

### Improvements

#### Critical
<Numbered list of critical improvements>

#### High Priority
<Numbered list of high-priority improvements>

#### Medium Priority
<Numbered list of medium-priority improvements>

#### Low Priority
<Numbered list of low-priority improvements>

### Dependency Analysis

<Import graph summary, notable dependencies, risks>

### Metrics Summary

| Metric | Value |
|--------|-------|
| Avg function length | <lines> |
| Max cyclomatic complexity | <value> (in <function>) |
| Type annotation coverage | <percentage> |
| Docstring coverage | <percentage> |
| Test coverage indicator | <has tests / no tests found> |
```

## Important Guidelines

- Be specific. Reference actual file paths, line numbers, function names, and variable names.
- Be constructive. Every criticism should come with a concrete suggestion.
- Be honest. If the code is good, say so. Do not manufacture issues.
- Be proportional. A 20-line utility script does not need the same depth as a 5000-line core module.
- Respect context. A prototype has different quality expectations than production code.
- Avoid false positives. If you are unsure whether something is an issue, note your uncertainty.
- Do NOT modify any code. This skill is read-only analysis. If the user wants fixes, direct them to `/fix-bugs` or `/review-code`.
- For very large codebases (100+ files), provide a high-level architectural analysis rather than file-by-file detail. Focus on the most impactful findings.
- If the target path does not exist or contains no source files, report that clearly rather than guessing.
