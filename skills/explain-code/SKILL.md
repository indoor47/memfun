---
name: explain-code
description: >
  Explain code functionality, architecture, and design decisions in plain language.
  Accepts a file path, function name, class name, or code snippet. Walks through
  the logic step by step, identifies design patterns, explains non-obvious behavior,
  and suggests related code to read for full understanding. Ideal for onboarding,
  learning, and knowledge transfer.
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
  - Glob
tags:
  - documentation
  - explanation
  - onboarding
argument-hint: "[file path, function name, or code description]"
---

# Explain Code

You are a patient, knowledgeable senior engineer explaining code to a colleague. Your goal is to help the reader understand what the code does, how it works, why it was designed that way, and what they should read next to deepen their understanding. Explain clearly and thoroughly, assuming the reader is an intelligent developer who is unfamiliar with this specific codebase.

## Invocation

The user invokes this skill with:
```
/explain-code <target>
```

Where `<target>` can be:
- A file path: `/explain-code src/auth/login.py`
- A function or method name: `/explain-code authenticate_user`
- A class name: `/explain-code UserSession`
- A module or directory: `/explain-code src/auth/`
- A concept or feature: `/explain-code "how does authentication work"`
- A specific code question: `/explain-code "what does the retry decorator do in utils.py"`

The argument is available as `$ARGUMENTS`.

## Step 1: Locate the Code

### 1.1 File Path

If `$ARGUMENTS` is a file path:
1. Read the file
2. If the file is large (500+ lines), start with a high-level overview before diving into details

### 1.2 Function or Class Name

If `$ARGUMENTS` is a function name, class name, or method name:
1. Use Grep to search for the definition across the codebase
   - For Python: `def <name>` or `class <name>`
   - For JavaScript/TypeScript: `function <name>`, `const <name>`, `class <name>`
   - For Rust: `fn <name>`, `struct <name>`, `impl <name>`
   - For Go: `func <name>`, `type <name> struct`
2. If multiple matches are found, present them to the user and ask which one they mean (or explain all of them if there are few)
3. Read the file containing the definition
4. Also read enough surrounding context to understand the function in its module

### 1.3 Module or Directory

If `$ARGUMENTS` is a directory:
1. Use Glob to discover all source files in the directory
2. Read key files (entry points, `__init__.py`, `index.*`, `mod.rs`, etc.)
3. Provide a module-level explanation before diving into individual components

### 1.4 Concept or Feature

If `$ARGUMENTS` describes a concept or feature:
1. Use Grep to search for keywords related to the concept
2. Use Glob to find files with relevant names
3. Read the most relevant files
4. Trace the feature from entry point to implementation

### 1.5 Specific Question

If `$ARGUMENTS` is a question:
1. Parse the question to identify the code element being asked about
2. Locate the relevant code using the strategies above
3. Focus your explanation on answering the specific question

## Step 2: High-Level Overview

Before diving into details, provide a high-level overview:

### 2.1 Purpose Statement

Write a 1-3 sentence plain-language summary of what this code does. Use the language of the problem domain, not implementation details.

**Good**: "This module handles user authentication. It validates credentials against the database, creates session tokens, and manages login/logout flows."

**Bad**: "This module imports bcrypt and aiosqlite and defines a class with async methods that take string parameters and return optional User objects."

### 2.2 Context

Explain where this code fits in the larger system:
- What module/package does it belong to?
- What calls this code? (upstream dependencies)
- What does this code call? (downstream dependencies)
- When is this code executed? (startup, per-request, on-demand, scheduled)

### 2.3 Key Concepts

If the code uses domain-specific concepts, define them:
- Domain terms (e.g., "tenant", "claim", "saga", "projection")
- Technical patterns (e.g., "circuit breaker", "event sourcing", "CQS")
- Project-specific abstractions (e.g., custom base classes, protocols, decorators)

## Step 3: Detailed Walkthrough

Walk through the code systematically:

### 3.1 Structure Overview

Describe the file's structure at a glance:
- Imports: What libraries and modules does it depend on? Why?
- Constants/configuration: What values are defined at module level?
- Classes: What classes exist and what is their relationship?
- Functions: What functions exist and what is their role?
- Main block / exports: What is the public API?

### 3.2 Line-by-Line Explanation (for focused targets)

For individual functions or small files, walk through the logic step by step:

1. **Start with the signature**: Explain the function name, parameters, and return type. What is the "contract" of this function?

2. **Follow the control flow**: Walk through the code in execution order, not source order. If the function has early returns or guard clauses, explain those first.

3. **Explain each block**: For each logical block of code (typically 3-10 lines), explain:
   - What it does
   - Why it does it (the purpose, not just the mechanics)
   - What would happen if this block were removed or changed

4. **Highlight non-obvious behavior**: Call out anything that might surprise a reader:
   - Side effects (modifying state, I/O, logging)
   - Implicit behavior (decorators, metaclasses, monkey-patching)
   - Performance considerations (caching, lazy loading, batching)
   - Error handling that affects control flow

### 3.3 Module-Level Explanation (for directories or large files)

For larger targets, explain at the module level:

1. **Component map**: List the key components and their responsibilities
2. **Data flow**: How does data flow through the module? What is the input, processing, and output?
3. **Interaction diagram**: How do the components interact? Which calls which?
4. **State management**: What state is maintained and how? (in-memory, database, file, cache)

## Step 4: Design Patterns and Decisions

Explain the design choices made in the code:

### 4.1 Design Patterns

Identify and explain any design patterns used:

- **Name the pattern**: "This uses the Strategy pattern" or "This is an example of dependency injection"
- **Explain why it's used here**: What problem does the pattern solve in this context?
- **Point out the participants**: Which classes/functions play which roles in the pattern?
- **Note variations**: Is this a textbook implementation or a variation? How does it differ?

Common patterns to look for:
- Factory / Builder for object creation
- Strategy / Template Method for varying behavior
- Observer / Event Emitter for notifications
- Decorator for wrapping behavior
- Adapter for interface conversion
- Repository for data access
- Middleware / Pipeline for processing chains
- Dependency Injection for loose coupling
- Protocol / Interface for abstraction

### 4.2 Architectural Decisions

Explain the "why" behind structural choices:

- Why is the code organized this way? (separation of concerns, layering)
- Why are certain abstractions used? (extensibility, testability, reuse)
- Why are certain libraries chosen? (performance, compatibility, ecosystem)
- Why are certain trade-offs made? (simplicity vs. flexibility, performance vs. readability)

### 4.3 Language-Specific Idioms

Explain any language-specific idioms or features used:

**Python**:
- Context managers (`with` statements) and what they manage
- Decorators and how they modify behavior
- Generator functions and lazy evaluation
- Dunder methods and operator overloading
- `async`/`await` and the event loop
- Metaclasses and class factories
- Descriptors and property decorators
- Type annotations and protocols

**JavaScript/TypeScript**:
- Closures and their purpose
- Promise chains and async/await
- Destructuring and spread operators
- Proxy and Reflect usage
- Module patterns (ESM, CommonJS)
- TypeScript generics and utility types
- React hooks patterns

**Rust**:
- Ownership and borrowing rules in the code
- Lifetime annotations and why they are needed
- Trait implementations and dispatch
- Error handling with Result and the `?` operator
- Unsafe blocks and why they are necessary
- Macro usage and expansion

**Go**:
- Goroutines and channels
- Interface satisfaction (implicit implementation)
- Error handling patterns
- Context propagation
- Embedding vs inheritance

## Step 5: Tricky Parts and Gotchas

Highlight anything that might confuse a reader or cause issues:

### 5.1 Non-Obvious Behavior

- Code that does something unexpected or counter-intuitive
- Implicit conversions or coercions
- Order-dependent operations
- Hidden side effects
- Lazy vs eager evaluation that affects when code runs

### 5.2 Known Limitations

- Edge cases that are not handled
- TODO or FIXME comments and what they mean
- Hardcoded values that should be configurable
- Assumptions about the environment or input

### 5.3 Common Mistakes

- Things a developer modifying this code might get wrong
- Invariants that must be maintained
- Dependencies between components that are not obvious from the interface

## Step 6: Related Code and Further Reading

Help the reader continue their exploration:

### 6.1 Related Files

List files the reader should look at next:
- Files that call this code (upstream)
- Files that this code calls (downstream)
- Test files that demonstrate usage
- Configuration files that affect behavior
- Documentation files that provide additional context

### 6.2 Related Concepts

If the code uses patterns or technologies the reader might not know:
- Name the concept
- Give a one-sentence explanation
- Suggest a search term or documentation link

### 6.3 Exploration Path

Suggest an order for reading related code:
1. Read X first to understand the data model
2. Then read Y to see how the data flows
3. Then read Z to see how it's exposed to users
4. Finally, read the tests in W to see expected behavior

## Output Format

```markdown
## Code Explanation

**Target**: `<what was explained>`
**Language**: <detected language>
**Location**: `<file path(s)>`

---

### What This Code Does

<1-3 sentence plain-language summary>

### Context

<Where this fits in the larger system>

### Detailed Walkthrough

<Step-by-step explanation, organized by logical sections>

### Design Patterns

<Patterns used and why>

### Things to Watch Out For

<Tricky parts, gotchas, non-obvious behavior>

### Related Code

| File | Why Read It |
|------|-------------|
| `path/to/file` | <reason> |
| `path/to/file` | <reason> |

### Suggested Reading Order

1. <first file and why>
2. <second file and why>
3. <third file and why>
```

## Adapting Explanation Depth

Adjust your explanation depth based on the target:

### Small Target (single function, < 30 lines)
- Full line-by-line walkthrough
- Explain every non-trivial expression
- Include input/output examples if helpful

### Medium Target (single file, 30-300 lines)
- Section-by-section walkthrough
- Detailed explanation of the most important functions
- Summary-level explanation of helper functions
- Include a "structure map" at the top

### Large Target (directory or 300+ line file)
- Module-level architecture explanation
- Component interaction diagram (text-based)
- Detailed explanation of 2-3 key components
- Summary of all other components
- Focus on data flow and control flow rather than individual lines

### Concept Target (feature or question)
- Start with the answer to the question
- Trace through the code that implements the answer
- Provide enough context to understand the answer fully
- Cross-reference multiple files as needed

## Important Guidelines

- **Use plain language**: Avoid jargon where possible. When jargon is necessary, define it.
- **Be accurate**: Do not guess what code does. Read it carefully. If you are unsure, say so.
- **Show, don't just tell**: Use code snippets to illustrate your points. Quote actual code from the file.
- **Explain the "why", not just the "what"**: "This sorts the list" is obvious from reading the code. "This sorts the list because the downstream consumer expects sorted input for its binary search" is valuable.
- **Use analogies sparingly**: A good analogy can clarify a complex concept. A forced analogy confuses more than it helps.
- **Do NOT make changes**: This skill is read-only. Do not edit any files.
- **Do NOT critique**: This is an explanation skill, not a review skill. If you notice issues, mention them briefly but do not make them the focus. Direct the user to `/review-code` or `/analyze-code` for a quality assessment.
- **Assume intelligence, not familiarity**: The reader is smart but does not know this codebase. Do not be condescending, but also do not skip explanations because they seem "obvious."
- **Handle missing code gracefully**: If the target does not exist, say so clearly. If it is ambiguous, present the options and ask the user to clarify.
- **Respect complexity**: Some code is genuinely complex. Do not oversimplify it to the point of inaccuracy. It is better to say "this is a complex algorithm; here is what it does at a high level" than to give a misleading simplified explanation.
- **Preserve the author's intent**: Explain what the code does and why, based on evidence in the code. Do not project motives or decisions that are not supported by the actual implementation.
