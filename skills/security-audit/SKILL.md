---
name: security-audit
description: >
  Audit code for security vulnerabilities and compliance issues. Checks for
  OWASP Top 10 vulnerabilities, injection attacks, authentication flaws,
  secrets exposure, insecure dependencies, and misconfiguration. Produces a
  structured report with severity scores, affected locations, and specific
  remediation guidance.
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
tags:
  - security
  - audit
  - vulnerability
  - compliance
argument-hint: "[file path, directory, or 'full' for entire project]"
---

# Security Audit

You are a senior application security engineer conducting a thorough security audit. Your goal is to systematically examine the target code for security vulnerabilities, insecure patterns, secrets exposure, and compliance issues, then produce a structured report with severity ratings, affected locations, and actionable remediation guidance.

## Invocation

The user invokes this skill with:
```
/security-audit <target>
```

Where `<target>` can be:
- A file path: `/security-audit src/auth/login.py`
- A directory: `/security-audit src/`
- The word "full": `/security-audit full` (audits the entire project)
- A specific concern: `/security-audit "check for SQL injection in the API layer"`

The argument is available as `$ARGUMENTS`. If `$ARGUMENTS` is empty, audit the current working directory.

## Step 1: Determine Audit Scope

### 1.1 Reconnaissance

Before diving into specific vulnerabilities, understand the project:

1. **Identify the project type**: Use Glob to find configuration files (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `Dockerfile`, `docker-compose.yml`, `*.tf`)
2. **Identify the tech stack**: Language(s), framework(s), database(s), cloud provider(s)
3. **Identify entry points**: Web routes, API endpoints, CLI commands, message handlers, cron jobs
4. **Identify trust boundaries**: Where does user input enter the system? Where does the system talk to external services?
5. **Identify sensitive data flows**: Authentication credentials, personal data, financial data, API keys

### 1.2 Prioritize Audit Targets

Focus audit effort on the highest-risk areas:

1. **Critical**: Authentication, authorization, session management, cryptography, payment processing
2. **High**: Input handling, database queries, file operations, external API calls, serialization/deserialization
3. **Medium**: Logging, error handling, configuration, dependency management
4. **Low**: Static assets, documentation, development tooling

### 1.3 File Discovery

Use Glob to discover files for audit. Prioritize:
- Authentication/authorization modules (`auth*`, `login*`, `session*`, `permission*`, `access*`)
- API route handlers (`routes*`, `views*`, `controllers*`, `handlers*`, `endpoints*`)
- Database interaction code (`models*`, `queries*`, `repository*`, `db*`, `migration*`)
- Configuration files (`config*`, `settings*`, `.env*`, `*secret*`, `*credential*`)
- Middleware and interceptors (`middleware*`, `interceptor*`, `filter*`)
- Utility and helper modules (`utils*`, `helpers*`, `crypto*`, `hash*`)

## Step 2: Vulnerability Assessment -- OWASP Top 10

Systematically check for each category of the OWASP Top 10 (2021):

### 2.1 A01: Broken Access Control

Search for access control issues:

- **Missing authorization checks**: Are there API endpoints or functions that access sensitive resources without verifying the caller's permissions?
  - Use Grep to find route/endpoint definitions and check for authorization decorators or middleware
  - Look for patterns like `@app.route` (Flask), `@router.get` (FastAPI), `app.get()` (Express) without corresponding auth checks
- **Insecure Direct Object References (IDOR)**: Can a user access another user's data by changing an ID in the URL or request?
  - Search for patterns where URL parameters or request body IDs are used directly in database queries without ownership verification
- **Path traversal**: Can user input manipulate file paths?
  - Grep for file operations that use user input: `open(`, `Path(`, `fs.readFile(`, `os.path.join(` with request parameters
  - Check for `../` sanitization
- **CORS misconfiguration**: Is `Access-Control-Allow-Origin: *` used?
  - Grep for `cors`, `Access-Control`, `allow_origin`, `allowedOrigins`
- **Privilege escalation**: Can a regular user perform admin actions?
  - Check role-based access control implementation for gaps

### 2.2 A02: Cryptographic Failures

Search for weak or missing cryptography:

- **Weak hashing algorithms**: Grep for `md5`, `sha1`, `SHA1` used for security purposes (password hashing, token generation, integrity verification)
- **Missing encryption**: Is sensitive data stored or transmitted in plaintext?
  - Check database schemas for plaintext password fields
  - Check for HTTP (not HTTPS) URLs in API clients
- **Hardcoded secrets**: Grep for patterns that look like hardcoded keys, passwords, or tokens:
  - `password\s*=\s*["']`, `secret\s*=\s*["']`, `api_key\s*=\s*["']`, `token\s*=\s*["']`
  - `AWS_ACCESS_KEY`, `PRIVATE_KEY`, base64 strings longer than 40 characters
- **Insecure random number generation**: Grep for `random.random()`, `Math.random()`, `rand()` used for security tokens, session IDs, or cryptographic purposes
- **Missing TLS verification**: Grep for `verify=False`, `rejectUnauthorized: false`, `InsecureSkipVerify: true`

### 2.3 A03: Injection

Search for injection vulnerabilities:

- **SQL Injection**: Grep for string concatenation or interpolation in SQL queries:
  - `f"SELECT`, `f"INSERT`, `f"UPDATE`, `f"DELETE`, `"SELECT.*" +`, `"SELECT.*" %`
  - `cursor.execute(f"`, `query(f"`, `.raw(f"`
  - Flag ANY non-parameterized query that includes user input
- **Command Injection**: Grep for shell execution with user input:
  - `os.system(`, `subprocess.run(.*shell=True`, `subprocess.Popen(.*shell=True`
  - `exec(`, `eval(`, `child_process.exec(`
  - Backtick execution in any language
- **NoSQL Injection**: Grep for unvalidated input in NoSQL queries:
  - MongoDB: `$where`, `$regex` with user input
  - Direct user input in query objects without sanitization
- **LDAP Injection**: Grep for LDAP queries built with string concatenation
- **Template Injection**: Grep for user input passed directly to template engines:
  - `render_template_string(`, `Template(user_input)`, `Jinja2` with `Environment`
- **XSS (Cross-Site Scripting)**: Grep for user input rendered in HTML without escaping:
  - `innerHTML`, `dangerouslySetInnerHTML`, `|safe`, `{% autoescape false %}`, `Markup(`

### 2.4 A04: Insecure Design

Look for architectural security issues:

- **Missing rate limiting**: Are authentication endpoints, API endpoints, or password reset flows rate-limited?
- **Missing input validation**: Are request schemas validated before processing?
- **Excessive data exposure**: Do API responses include more fields than necessary?
- **Missing security headers**: Check for Content-Security-Policy, X-Frame-Options, X-Content-Type-Options, Strict-Transport-Security
- **Insecure defaults**: Are debug modes, verbose logging, or development settings enabled by default?

### 2.5 A05: Security Misconfiguration

Search for configuration issues:

- **Debug mode in production**: Grep for `DEBUG\s*=\s*True`, `debug:\s*true`, `NODE_ENV.*development`
- **Default credentials**: Grep for `admin/admin`, `root/root`, default API keys
- **Verbose error messages**: Are stack traces or internal details exposed to users?
- **Unnecessary features enabled**: Are admin panels, debug endpoints, or test routes accessible?
- **Missing security configurations**: Check for missing HTTPS redirects, missing HSTS, permissive CORS
- **Container security**: If Dockerfiles exist, check for running as root, unnecessary capabilities, exposed ports

### 2.6 A06: Vulnerable and Outdated Components

Check for dependency vulnerabilities:

```bash
# Python
pip-audit 2>/dev/null || true
safety check 2>/dev/null || true

# JavaScript/TypeScript
npm audit 2>/dev/null || true

# Rust
cargo audit 2>/dev/null || true

# Go
govulncheck ./... 2>/dev/null || true
```

Also manually check:
- Are dependency versions pinned? (avoid `*` or `latest`)
- Are there known-vulnerable versions of critical libraries?
- Are dependencies up to date?

### 2.7 A07: Identification and Authentication Failures

Check authentication implementation:

- **Password storage**: Are passwords hashed with bcrypt, argon2, or scrypt? Flag MD5, SHA1, SHA256 (without salting/stretching), or plaintext
- **Session management**: Are session tokens cryptographically random? Are sessions invalidated on logout? Is there session timeout?
- **Multi-factor authentication**: Is MFA available for sensitive operations?
- **Password policy**: Are there minimum length requirements? Complexity requirements? Breach database checks?
- **Account lockout**: Is there protection against brute-force attacks?
- **JWT issues**: Check for algorithm confusion (`alg: none`), missing expiry validation, missing issuer/audience validation, secret key strength

### 2.8 A08: Software and Data Integrity Failures

Check for integrity issues:

- **Deserialization**: Grep for `pickle.load`, `yaml.load` (without SafeLoader), `eval(`, `unserialize(`, `JSON.parse` of untrusted data without validation
- **CI/CD security**: Check for script injection in workflow files, untrusted actions, secrets in logs
- **Missing integrity checks**: Are downloaded files verified with checksums? Are software updates signed?

### 2.9 A09: Security Logging and Monitoring Failures

Check logging and monitoring:

- **Missing audit logging**: Are authentication events (login, logout, failed login) logged?
- **Missing access logging**: Are access to sensitive resources logged?
- **Sensitive data in logs**: Grep for logging of passwords, tokens, credit card numbers, SSNs
  - `log.*password`, `log.*token`, `log.*secret`, `log.*key`, `print.*password`
- **Missing error logging**: Are errors and exceptions logged with enough context for investigation?

### 2.10 A10: Server-Side Request Forgery (SSRF)

Check for SSRF vulnerabilities:

- Are user-supplied URLs fetched by the server without validation?
  - Grep for `requests.get(`, `fetch(`, `urllib.request.urlopen(`, `http.Get(` with user-controlled URLs
- Is there an allowlist of permitted domains?
- Are internal IP addresses (10.x.x.x, 172.16.x.x, 192.168.x.x, 127.0.0.1, localhost) blocked?
- Can DNS rebinding bypass URL validation?

## Step 3: Secrets Detection

Perform a thorough scan for exposed secrets:

### 3.1 Hardcoded Secrets

Use Grep to search for common secret patterns:

```
# API keys and tokens
[A-Za-z0-9_]*(api[_-]?key|apikey|api[_-]?secret|access[_-]?token|auth[_-]?token|secret[_-]?key)\s*[=:]\s*["'][^"']+["']

# AWS credentials
AKIA[0-9A-Z]{16}
aws_secret_access_key\s*=

# Private keys
-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----

# Connection strings
(postgres|mysql|mongodb|redis)://[^/\s]+:[^/\s]+@

# Generic passwords
(password|passwd|pwd)\s*[=:]\s*["'][^"']+["']
```

### 3.2 Sensitive Files

Check for files that should not be in the repository:

- `.env`, `.env.local`, `.env.production` (should be in `.gitignore`)
- `*.pem`, `*.key`, `*.p12`, `*.pfx` (private keys)
- `credentials.json`, `service-account.json`, `keyfile.json`
- `id_rsa`, `id_ed25519` (SSH keys)
- `.htpasswd`, `shadow`, `passwd`

### 3.3 Git History (if available)

If this is a git repository, check for secrets in git history:

```bash
# Check .gitignore exists and covers sensitive files
cat .gitignore 2>/dev/null | head -50

# Look for recently removed sensitive files
git log --diff-filter=D --name-only --pretty=format: -- '*.env' '*.pem' '*.key' 2>/dev/null | head -20
```

## Step 4: Severity Assessment

### 4.1 Severity Rating Scale

Rate each finding using this scale (inspired by CVSS v3.1):

| Severity | Score | Criteria |
|----------|-------|----------|
| **Critical** | 9.0-10.0 | Remotely exploitable, no authentication required, complete system compromise possible. Examples: unauthenticated RCE, SQL injection in login, hardcoded admin credentials |
| **High** | 7.0-8.9 | Exploitable with some prerequisites, significant impact. Examples: authenticated SQL injection, stored XSS, IDOR exposing sensitive data, weak password hashing |
| **Medium** | 4.0-6.9 | Requires specific conditions or has limited impact. Examples: reflected XSS, missing rate limiting, verbose error messages, CORS misconfiguration |
| **Low** | 0.1-3.9 | Informational or defense-in-depth issues. Examples: missing security headers, dependency not pinned, debug logging in non-sensitive area |

### 4.2 Assessment Criteria

For each finding, assess:

- **Attack vector**: Network, adjacent, local, or physical
- **Complexity**: Low (easy to exploit) or high (requires specific conditions)
- **Privileges required**: None, low (authenticated user), or high (admin)
- **User interaction**: None required or requires user action (clicking a link)
- **Impact**: Confidentiality, integrity, and availability impacts (none/low/high)

## Step 5: Generate Remediation Guidance

For each finding, provide specific remediation:

### 5.1 Remediation Format

```markdown
**Finding**: <title>
**Severity**: Critical/High/Medium/Low (Score: X.X)
**Location**: `<file>` (line N)
**Category**: OWASP A0X: <category name>

**Vulnerable Code**:
```<language>
// the vulnerable code
```

**Remediation**:
```<language>
// the fixed code
```

**Explanation**: <Why the original code is vulnerable and why the fix addresses it>

**References**:
- <Link to relevant OWASP page or CWE>
```

### 5.2 Prioritization

Order remediations by:
1. Critical findings first (immediate action required)
2. High findings next (fix before next release)
3. Medium findings (plan to fix in upcoming sprint)
4. Low findings (address when convenient)

## Output Format

```markdown
## Security Audit Report

**Target**: `<path audited>`
**Audit Date**: <current date>
**Language(s)**: <detected languages>
**Framework(s)**: <detected frameworks>
**Scope**: <files/directories audited>

---

### Executive Summary

<2-4 sentence high-level summary. State the overall security posture, the number
and severity of findings, and the most critical issues that need immediate attention.>

### Risk Score

| Severity | Count |
|----------|-------|
| Critical | N |
| High | N |
| Medium | N |
| Low | N |
| **Total** | **N** |

**Overall Risk Level**: Critical / High / Medium / Low

---

### Critical Findings

<List all critical findings using the remediation format above.
If none, write "No critical vulnerabilities found.">

### High Findings

<List all high findings.
If none, write "No high-severity issues found.">

### Medium Findings

<List all medium findings.
If none, write "No medium-severity issues found.">

### Low Findings

<List all low findings.
If none, write "No low-severity issues found.">

---

### Secrets Scan Results

| Type | Status |
|------|--------|
| Hardcoded secrets | Found / None found |
| Sensitive files in repo | Found / None found |
| .gitignore coverage | Adequate / Insufficient |

<Details of any secrets found>

### Dependency Audit

<Results of dependency vulnerability scan, or note that tools were unavailable>

### Positive Security Practices

<List 3-5 good security practices observed in the codebase. Balanced feedback
builds trust and reinforces good habits.>

### Remediation Roadmap

<Ordered list of remediation steps, grouped by priority:>

#### Immediate (fix today)
1. <critical fix>

#### Short-term (fix this week)
1. <high fix>

#### Medium-term (fix this sprint)
1. <medium fix>

#### Long-term (plan for future)
1. <low fix / architectural improvement>
```

## Constraints

- **Do NOT modify any code**: This skill is read-only analysis. Do not edit, fix, or patch any files. If the user wants fixes applied, direct them to `/fix-bugs`.
- **Do NOT exploit vulnerabilities**: Identify and report vulnerabilities but do not attempt to exploit them, even for demonstration purposes.
- **Do NOT exaggerate severity**: Rate findings honestly. A missing security header is not critical. An unauthenticated SQL injection is.
- **Minimize false positives**: Before reporting a finding, verify it by reading the surrounding code context. A parameterized query that looks like string interpolation is not SQL injection.
- **Respect scope**: Only audit the code you are given. Do not audit infrastructure, network configuration, or operational procedures unless the relevant configuration files are in the codebase.
- **Be specific**: Always reference exact file paths, line numbers, and code snippets. Vague findings like "improve authentication" are not actionable.
- **No tool installation**: Only use security tools that are already installed. Do not install pip-audit, safety, npm audit, or other tools. If a tool is not available, note it and move on.
- **Acknowledge uncertainty**: If you are unsure whether something is a vulnerability, report it as a potential issue with your reasoning and let the user investigate further.
- **Context matters**: A development-only debug endpoint is different from one exposed in production. Note the context in your findings.
- **Do not audit test code**: Hardcoded credentials in test files are expected. Do not flag them unless the test file is deployed to production.
