# Learnings

A versioned, shared record of non-obvious findings discovered while building memfun. Maintained by the [reliability-monitor agent](../agents/reliability-monitor/AGENT.md) and contributors. Cross-referenced from `CLAUDE.md`.

> **Format**: each entry is a short H3 with a date, a one-line **What**, a one-line **Why it matters**, and the source (commit, PR, issue, or experiment). Append-only — don't edit historical entries unless they're flat wrong; instead add a follow-up that supersedes them.

## 2026-05-06 — Architecture audit + Experiment A

### Concurrent worktree edits to overlapping files conflict at ~60% regardless of model size
- **What**: With 5 parallel agents independently rewriting a shared file (`middleware.py` in the test fixture), `git merge-tree` reports 6/10 pairwise conflicts — the same rate at Qwen 2.5 Coder 3B and 7B.
- **Why it matters**: A stronger model does **not** produce more textually-mergeable edits. The architectural priority must be preventing the overlap at decomposition time, not repairing it after.
- **Source**: Experiment A, runs in `/tmp/exp-a-results/` and `/tmp/exp-a-results-7b/`. Issue #6 tracker.

### `temperature=0` + identical prompt = identical output, which falsely "merges clean"
- **What**: All 5 agents in S3 (full-codebase rename) produced byte-identical diffs (md5 `5379cfab06ad1716665494db53c333f2`). Pairwise conflicts read as 0/10, but this is a measurement artifact, not a validation of the merge model.
- **Why it matters**: Future fan-out experiments must inject per-agent variance (different scope, different system prompt) to produce meaningful merge data.
- **Source**: Experiment A scenario S3 on Qwen 7B.

### `AgentManager._agents` is keyed by name, not instance — caps in-process parallelism at "one per agent type"
- **What**: `lifecycle.py:72` uses `dict[str, _ManagedAgent]` keyed by name. `start_agent("coder-agent")` raises `RuntimeError` on the second call.
- **Why it matters**: Today's effective ceiling is ~3-5 concurrent specialists in T0/T1, regardless of how many sub-tasks the decomposer emits. Issue #12.
- **Source**: Architecture audit, 2026-05-06.

### `SharedSpec.file_registry` silently loses ownership on second writer
- **What**: `dict[path, task_id]` (`shared_spec.py:67`) — the second `self.file_registry[path] = task_id` overwrites the first; `get_conflict_files` then operates on the dict it just clobbered.
- **Why it matters**: Conflict detection is structurally broken even before any code runs. Must change to `dict[path, list[task_id]]` (issue #14).
- **Source**: Architecture audit, 2026-05-06.

### Hetzner CCX33 needs `libgomp1` to load llama.cpp CPU backends
- **What**: Stock Ubuntu 24.04 minimal install on Hetzner CCX dedicated boxes does NOT ship `libgomp.so.1`. llama.cpp's `libggml-cpu-*.so` plugins all link against it; without it, `srv load_model: failed to load model: make_cpu_buft_list: no CPU backend found`. CX33 (Ubuntu 22.04) had it pre-installed, masking the dependency.
- **Why it matters**: Any setup script or docker recipe that targets Hetzner cloud needs an explicit `apt-get install libgomp1`.
- **Source**: 14B/7B box bring-up, 2026-05-06.

### Volumes are disk, not RAM — to run bigger models you must rescale or reprovision
- **What**: Hetzner Cloud volumes give block storage. They do not add RAM. Server type fixes RAM. To go from 8GB → 32GB you either rescale the existing server (brief stop) or provision a new bigger type.
- **Why it matters**: Model RAM budgets dictate server type. CX33 (8GB) tops out at ~3B Q4. CCX33 (32GB) handles 7-14B comfortably. CCX53 (128GB) needed for 70B-class. CCX63 (192GB) for V4-Flash.
- **Source**: server provisioning discussion, 2026-05-06.

### memfun's `provider="openai" + base_url` already supports any OpenAI-compatible endpoint without code changes
- **What**: `LLMConfig` (`memfun-core/config.py:35-42`) carries `provider`, `model`, `base_url`, `api_key_env`. The DSPy bridge (`chat.py:1976-1984`) maps `provider="openai"` to LiteLLM's `openai/<model>` and forwards `api_base`. Setting `provider="openai"`, `base_url="http://127.0.0.1:8089/v1"`, `model="qwen-coder-7b"`, and any non-empty API key value works end-to-end.
- **Why it matters**: There's no need for new "ollama" / "openai-compat" / "vllm" provider classes. The wizard just needs to expose this path (issue #15).
- **Source**: end-to-end test of `memfun ask` against local Qwen, 2026-05-06.

### Wizard never asks for a model name on OpenAI/Custom paths → falls back to `claude-opus-4-6`
- **What**: `init.py:93-105` collects provider but not model. The default in `LLMConfig` (`config.py:37`) is `claude-opus-4-6`. So picking "OpenAI" in the wizard writes `provider=openai, model=claude-opus-4-6` which OpenAI rejects.
- **Why it matters**: This is why "GPT didn't work" historically. Must ask for model per-provider. Issue #15.
- **Source**: wizard code read, 2026-05-06.

### Stale credentials fail silently with no health check anywhere
- **What**: A 3-month-stale Anthropic API key in `~/.memfun/credentials.json` was returning 401 on every call, with no surface indication anywhere in the CLI. Discovered only when an external experiment ran a status-aware probe.
- **Why it matters**: Need `memfun doctor` (issue #7) to do a 4-token ping at startup or on demand.
- **Source**: Experiment A blocker, 2026-05-06.

### llama.cpp `--parallel N` controls KV-cache slots; 5 simultaneous 2K-token prompts on 4 slots → thrash
- **What**: Default `--parallel 4` with 8K context = 4 × 8K KV cache slots. Hitting 5 concurrent requests with 2K-token prompts each makes `decode: failed to find a memory slot for batch of size 1`; the server purges and retries, latency goes from 10s → 600s+. Fix: `--parallel 1` (single-inference) and serialize requests at the client.
- **Why it matters**: Concurrency on a small CPU box is gated by KV cache, not CPU. 1×8 cores on one inference beats 5×1.6 cores fighting for cache.
- **Source**: Experiment A first run, 2026-05-06.

## 2026-05-06 — Post-merge sweep after #16 / #17

### Pre-existing CI failures on `main` block every PR — fix on `main` first, not in the feature PR
- **What**: When PRs #16 and #17 reached the merge queue, three CI jobs failed for reasons unrelated to either PR: ruff `I001` (unsorted imports in `packages/memfun-cli/src/memfun_cli/main.py:107`), pyright strict-mode noise from `evals/swebench_harness.py` (57 errors driven by untyped `dspy`/`swebench`/`datasets`), and a missing `pip-audit` in dev deps. Resolved by `39a683d` directly on `main` (lint auto-fix + `pyproject.toml` `pyright.exclude = ["evals", ".claude/worktrees", ".memfun/worktrees"]` + `dev` dep).
- **Why it matters**: When CI breaks on `main`, every downstream PR inherits red status. The right move is a small, behavior-neutral "ci: unblock pipeline" commit on `main` rather than dragging unrelated fixes into the feature PR — keeps the audit trail clean and unblocks the queue immediately.
- **Source**: Commit `39a683d` (ci: unblock pipeline), 2026-05-06.

### Pyright strict mode is unusable on directories that import untyped third-party libs — exclude, don't try to type
- **What**: `evals/swebench_harness.py` produced 57 strict-mode errors purely because `dspy`, `swebench`, and `datasets` ship without type stubs. Adding `# type: ignore` everywhere is noise; the durable fix is `[tool.pyright].exclude = ["evals", ...]` in `pyproject.toml`. Production packages stay strict; eval/research scripts are excluded.
- **Why it matters**: Future agents tempted to "fix the pyright errors in evals/" will burn a turn on it. Don't — the directory is intentionally excluded from CI typecheck.
- **Source**: `pyproject.toml` `[tool.pyright]` exclude list, commit `39a683d`.

### `WorktreeManager.cleanup_worktree(path)` is unscoped — gate by `base_dir` even though current callers are well-behaved
- **What**: Security audit on PR #17 (HIGH finding) — `cleanup_worktree(Path("/tmp/user-wt"))` will succeed and wipe a user-created worktree if a buggy caller passes its path. Branch deletion *is* gated by `_branch_from_path`, but the worktree removal itself isn't. `WorkflowEngine` only ever calls cleanup with paths it stored from `make_worktree`, so this is not exploitable today, but the API contract is unsafe for future callers and inconsistent with `list_worktrees` (which IS scoped).
- **Why it matters**: General pattern — when an API takes a `path` argument and performs a destructive op, scope it to a known base dir at the entry point. Don't rely on every future caller getting it right. Tracked as issue #18.
- **Source**: PR #17 security-auditor (Opus 4.7) audit comment; issue #18 filed, 2026-05-06.

### `pip-audit --strict` exits non-zero on workspace-internal packages
- **What**: `pip-audit --strict` errors with `memfun-agent: Dependency not found on PyPI and could not be audited: memfun-agent (0.2.0)` because workspace-internal packages aren't published to PyPI. The exit code is 0 in this case (no actual CVEs), but the stderr line looks alarming and could be misread as a finding.
- **Why it matters**: Don't grep pip-audit output for "ERROR" to gate CI — read the structured findings list. The internal-package "not found on PyPI" message is expected for any uv workspace and is not a vulnerability.
- **Source**: Reliability-monitor sweep, 2026-05-06.

### `README.md` tests-passed badge and `tests/` line drift after every test-adding PR
- **What**: README claims "597 tests" in three places (badge, repo tree, Make target comment); actual collected count after #16+#17 is 722. CLAUDE.md doesn't claim a test count, but MEMORY.md auto-memory says "597 tests" and "8 packages" (7 packages on disk). Test counts and package counts in narrative docs go stale fast.
- **Why it matters**: Hard-coded counts in README are pure tech debt. Either replace with a CI-generated badge that reads pytest collect, or stop printing exact counts and use ranges ("700+ tests"). The package-count drift suggests MEMORY.md needs a periodic rewrite as the project grows.
- **Source**: Reliability-monitor sweep, 2026-05-06 (README:10, README:444, README:472).
