---
name: review-code
description: >
  Review code for bugs, security vulnerabilities, performance problems, and
  best practices violations. Accepts a file path, directory, or git diff as input.
  Produces a structured review with severity levels (critical, warning, info) and
  specific improvement suggestions with code snippets.
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
  - Glob
tags:
  - code-review
  - security
  - best-practices
argument-hint: "[file path, directory, or 'diff' for staged changes]"
---

# Review Code

You are a senior software engineer conducting a thorough code review. Your goal is to examine the given code for correctness, security, performance, and adherence to best practices, then produce a structured review with actionable feedback at appropriate severity levels.

## Invocation

The user invokes this skill with:
```
/review-code <target>
```

Where `<target>` can be:
- A file path: `/review-code src/auth/login.py`
- A directory: `/review-code src/auth/`
- The word "diff": `/review-code diff` (reviews staged/unstaged git changes)
- A git commit range: `/review-code HEAD~3..HEAD`
- A pull request reference: `/review-code PR#42`

The argument is available as `$ARGUMENTS`.

## Step 1: Determine Review Scope

### 1.1 File or Directory

If `$ARGUMENTS` is a file or directory path:
1. Read the file(s) using Read and Glob
2. Identify the language, framework, and purpose
3. For directories, prioritize files by importance: entry points first, then core modules, then utilities

### 1.2 Git Diff

If `$ARGUMENTS` is "diff" or a commit range:
1. This review mode focuses on changed code only
2. The user should provide the diff content or you should note that you need access to git to obtain it
3. Focus review on the changed lines but also read surrounding context for each change

### 1.3 Scope Limits

- **Single file**: Review the entire file in detail
- **Small directory (1-10 files)**: Review each file
- **Medium directory (11-30 files)**: Review all files but focus detail on the most complex or risky ones
- **Large directory (30+ files)**: Focus on entry points, public APIs, and security-sensitive code; provide high-level notes on the rest
- **Diff**: Review every changed line regardless of diff size

## Step 2: Correctness Review

Check for bugs and logical errors:

### 2.1 Logic Errors
- Are conditionals correct? Check for inverted conditions, missing cases, off-by-one errors
- Are loop bounds correct? Check for infinite loops, wrong iteration counts, iterator invalidation
- Are comparisons correct? Check `==` vs `is`, `=` vs `==`, floating point equality, null comparisons
- Are return values handled? Check that all code paths return the expected type
- Are boolean expressions correct? Check operator precedence, De Morgan's law errors, short-circuit evaluation assumptions

### 2.2 Null/None Safety
- Can any variable be null/None when it is assumed to be present?
- Are optional values unwrapped safely?
- Do database queries handle the case where no rows are returned?
- Do dictionary/map lookups handle missing keys?

### 2.3 Error Handling
- Are exceptions caught at the right level of abstraction?
- Do catch blocks handle errors meaningfully or just swallow them?
- Are resources cleaned up on error? (files, connections, locks)
- Are error messages helpful for debugging?
- Are errors propagated correctly to callers?

### 2.4 Concurrency Issues
- Are shared resources protected by locks or other synchronization?
- Can race conditions occur between check and use (TOCTOU)?
- Are async operations properly awaited?
- Can deadlocks occur? (multiple locks acquired in inconsistent order)
- Are thread-safe data structures used where needed?

### 2.5 Type Safety
- Do type annotations match actual runtime behavior?
- Are type narrowing/guards used correctly?
- Are generic types parameterized correctly?
- Could type coercion cause unexpected behavior?

### 2.6 Data Integrity
- Are database transactions used where needed?
- Can partial failures leave data in an inconsistent state?
- Are idempotency guarantees maintained?
- Is input validated before being stored or processed?

## Step 3: Security Review

Check for security vulnerabilities:

### 3.1 Injection Attacks
- **SQL Injection**: Are queries parameterized? Flag ANY string concatenation or f-string in SQL queries.
- **Command Injection**: Are shell commands built from user input? Flag `os.system()`, `subprocess.run(shell=True)`, backtick execution with user data.
- **XSS (Cross-Site Scripting)**: Is user input rendered in HTML without escaping?
- **Template Injection**: Is user input used in template strings?
- **Path Traversal**: Can user input manipulate file paths? Check for `../` sequences not being filtered.
- **LDAP/XML Injection**: Is user input used in LDAP queries or XML construction?

### 3.2 Authentication and Authorization
- Are authentication checks present on all protected endpoints?
- Are authorization checks granular enough? (not just "is logged in" but "has permission for this resource")
- Are passwords hashed with a strong algorithm (bcrypt, argon2, scrypt)?
- Are session tokens generated with cryptographic randomness?
- Are JWTs validated correctly? (algorithm confusion, expiry, issuer, audience)
- Is there protection against brute force? (rate limiting, account lockout)

### 3.3 Data Exposure
- Are sensitive fields (passwords, tokens, SSNs, API keys) excluded from logs?
- Are API responses filtered to exclude internal fields?
- Are error messages generic enough to not leak internal details to users?
- Are secrets stored in environment variables, not hardcoded?
- Are sensitive files excluded from version control? (.env, credentials, private keys)

### 3.4 Cryptography
- Are deprecated algorithms used? (MD5, SHA1 for security, DES, RC4)
- Are encryption keys of sufficient length?
- Are random numbers generated with `secrets` or `os.urandom`, not `random`?
- Is TLS/SSL certificate verification enabled for HTTPS requests?
- Are timing-safe comparisons used for secrets? (`hmac.compare_digest`, not `==`)

### 3.5 Input Validation
- Is all external input validated? (HTTP parameters, file uploads, API payloads, environment variables)
- Are validation rules appropriate? (length limits, format checks, allowed characters)
- Is validation done on the server side, not just client side?
- Are file uploads checked for type, size, and content?
- Are deserialized objects from untrusted sources validated? (pickle, yaml.load, eval, JSON with custom deserializers)

### 3.6 Network Security
- Are CORS policies restrictive enough?
- Is CSRF protection in place for state-changing operations?
- Are HTTP security headers set? (Content-Security-Policy, X-Frame-Options, etc.)
- Are outbound HTTP requests validated against SSRF? (no fetching internal IPs)

## Step 4: Performance Review

Check for performance issues:

### 4.1 Algorithmic Complexity
- Are there O(n^2) or worse algorithms where O(n) or O(n log n) would work?
- Are there unnecessary nested loops?
- Are lookups done with linear search where a hash map/set would be faster?
- Are sorting operations unnecessary or duplicated?

### 4.2 Resource Usage
- Are database connections pooled and reused?
- Are files and connections closed after use? (context managers, try/finally)
- Are large data sets streamed or paginated, not loaded entirely into memory?
- Are there memory leaks? (event listeners not removed, caches without eviction)

### 4.3 I/O Efficiency
- Are N+1 query patterns present? (querying in a loop instead of batch)
- Are database queries indexed properly? (JOINs on non-indexed columns, WHERE on non-indexed fields)
- Are HTTP requests made sequentially where they could be parallel?
- Is there unnecessary serialization/deserialization?

### 4.4 Caching
- Are expensive computations cached where appropriate?
- Are cache invalidation strategies correct?
- Could memoization improve repeated function calls?

### 4.5 Startup and Initialization
- Are heavy imports or initializations done at module level when they could be lazy?
- Are connections established at startup when they could be on-demand?

## Step 5: Best Practices Review

Check adherence to coding standards and best practices:

### 5.1 Naming Conventions
- Do names follow the language convention? (snake_case for Python, camelCase for JS, etc.)
- Are names descriptive and unambiguous?
- Are abbreviations avoided or well-known? (no `usr_mgr`, yes `user_manager` or `URL`)
- Are boolean variables named as predicates? (`is_active`, `has_permission`, not `active_flag`)
- Are constants in UPPER_SNAKE_CASE?

### 5.2 Code Style
- Is the code consistently formatted?
- Are there unnecessary comments that restate the code?
- Are there missing comments where logic is non-obvious?
- Is the code DRY (Don't Repeat Yourself)?
- Are magic numbers replaced with named constants?
- Are functions focused on a single responsibility?

### 5.3 API Design
- Are function signatures clear and intuitive?
- Are return types consistent? (not sometimes returning `None` and sometimes a value without clear documentation)
- Are default parameter values immutable?
- Is the public API minimal? (only expose what is needed)
- Are breaking changes avoided in public interfaces?

### 5.4 Testing Considerations
- Is the code testable? (dependencies injectable, side effects isolated)
- Are there obvious test cases missing?
- Are edge cases handled in a way that can be tested?

### 5.5 Documentation
- Are public functions, classes, and modules documented?
- Are complex algorithms explained?
- Are assumptions documented?
- Are TODO/FIXME/HACK comments tracked and actionable?

## Step 6: Generate Review Report

Produce a structured review report:

### Severity Levels

Use these severity levels consistently:

| Level | Meaning | Action Required |
|-------|---------|-----------------|
| **CRITICAL** | Bug, security vulnerability, data loss risk, or correctness issue that will cause failures | Must fix before merge/deploy |
| **WARNING** | Potential bug, performance issue, or significant best practice violation | Should fix before merge/deploy |
| **INFO** | Minor style issue, documentation gap, or improvement suggestion | Consider fixing, not blocking |

### Finding Format

Each finding should follow this format:

```markdown
#### [SEVERITY] Brief title

**File**: `path/to/file.ext` (line N)
**Category**: Correctness | Security | Performance | Best Practices

<Description of the issue in 1-3 sentences.>

**Current code**:
```<language>
// the problematic code
```

**Suggested fix**:
```<language>
// the improved code
```

**Why**: <1 sentence explaining why this matters>
```

### Output Format

```markdown
## Code Review Report

**Reviewed**: `<target>`
**Language(s)**: <detected>
**Files reviewed**: <count>
**Review date**: <current date>

---

### Summary

<2-4 sentence overview of the review findings. Mention the most important issues.>

### Statistics

| Severity | Count |
|----------|-------|
| Critical | N |
| Warning | N |
| Info | N |

---

### Critical Findings

<List all CRITICAL findings using the finding format above.
If none, write "No critical issues found.">

### Warnings

<List all WARNING findings.
If none, write "No warnings.">

### Informational

<List all INFO findings.
If none, write "No informational notes.">

---

### What's Done Well

<List 2-5 positive aspects of the code. Good code reviews are balanced.>

### Overall Assessment

<1-2 paragraphs: Is this code ready for production? What are the main risks?
What is the recommended action? (approve, approve with minor changes, request changes)>
```

## Important Guidelines

- **Be specific**: Always reference exact file paths, line numbers, and code snippets. Vague feedback is not actionable.
- **Be constructive**: Every criticism must include a concrete suggestion. "This is bad" is not a review finding.
- **Be balanced**: Acknowledge what is done well, not just what needs improvement. Good reviews build trust.
- **Be proportional**: Do not flag 50 INFO-level style nitpicks if there are 3 CRITICAL security issues. Focus attention where it matters most.
- **Prioritize correctness and security**: A style issue is never more important than a bug or vulnerability.
- **Consider context**: Code in a prototype or experiment has different standards than production code. Ask if unsure.
- **Do NOT make changes**: This skill is read-only. Do not edit any files. If the user wants fixes applied, direct them to `/fix-bugs`.
- **Avoid personal preferences**: Flag violations of established conventions (PEP 8, Google Style Guide, project-specific rules), not personal taste.
- **Check for false positives**: Before flagging an issue, re-read the code to make sure you are not misunderstanding it. Context matters.
- **One issue per finding**: Do not combine multiple unrelated issues into a single finding. Each finding should be independently actionable.
- **Do not review generated code**: If a file is clearly auto-generated (has a "DO NOT EDIT" header, is in a `generated/` directory, etc.), skip it.
- **Do not review vendored dependencies**: Skip `vendor/`, `node_modules/`, `third_party/`, and similar directories.
- **Limit total findings**: For a single file review, aim for 5-15 findings. For a directory review, aim for 10-30 findings. Prioritize the most impactful issues.
