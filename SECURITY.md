# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Memfun, please report it
responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

- **Email**: Send a detailed report to security@memfun.dev
- **GitHub**: Use the [Security Advisories](https://github.com/memfun/memfun/security/advisories) feature to report privately.

Include the following in your report:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for a fix
within 7 days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes (current development) |

## Security Practices

Memfun is designed with security as a core concern. The following practices are
enforced across the codebase:

### SQL Injection Prevention

All database queries use parameterized statements. String concatenation for SQL
is never used. This applies to all backend tiers that involve SQL (T1 SQLite).

### SSRF Prevention

The web tools (web_fetch, web_search) implement SSRF prevention measures:

- URL validation and scheme allowlisting (HTTPS only by default)
- Domain allowlist and blocklist support via configuration
- Private/internal IP range blocking
- Response size limits to prevent resource exhaustion

### Sandbox Isolation

Code execution is sandboxed through multiple backend options:

- **Local sandbox** -- Restricted execution environment
- **Docker sandbox** -- Container-based isolation with resource limits
- **Modal sandbox** -- Cloud-based ephemeral execution environments

Each sandbox enforces timeout limits and memory caps configured via
`memfun.toml`.

### Secret Management

- API keys are stored as environment variable names in configuration, never as
  raw values.
- The `api_key_env` pattern ensures secrets are resolved at runtime from the
  environment.
- Configuration files (`memfun.toml`) contain no sensitive data by design.

### Trust Tiers

Agents and skills operate under a trust tier system. Built-in components have
higher trust than community-contributed ones. The trust tier determines which
tools and capabilities an agent can access.

### Dependency Management

- Dependencies are pinned via `uv.lock` for reproducible builds.
- Only well-maintained, widely-used packages are included as dependencies.
