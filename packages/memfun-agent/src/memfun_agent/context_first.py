"""Context-First Solver: gather context, then solve in one shot.

Instead of iterating through the RLM loop (which wastes calls on context
discovery), this module:

1. **Plans** what context is needed (1 LLM call, or skip for small projects)
2. **Gathers** files and search results (pure I/O, no LLM)
3. **Solves** in a single pass with full context (1 LLM call)
4. **Executes** the resulting file operations mechanically

Falls back to the existing RLM loop when the single-shot approach fails.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import dspy
from memfun_core.logging import get_logger

from memfun_agent.signatures import (
    ContextPlanning,
    SingleShotSolving,
    VerificationFix,
)

logger = get_logger("agent.context_first")


def _get_dspy_token_usage() -> int:
    """Read cumulative token usage from the DSPy LM history.

    DSPy stores per-call history on the thread-local LM.  We read
    the total length of the history list — each entry corresponds
    to one API call.  The actual token counts are in
    ``entry['usage']`` when available.
    """
    try:
        lm = dspy.settings.lm
        if lm is None:
            return 0
        history = getattr(lm, "history", None)
        if not history:
            return 0
        total = 0
        for entry in history:
            usage = entry.get("usage", {}) if isinstance(entry, dict) else {}
            total += usage.get("total_tokens", 0)
        return total
    except Exception:
        return 0


# ── Configuration ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextFirstConfig:
    """Configuration for the Context-First Solver.

    Attributes:
        max_context_bytes: Skip planner and dump ALL files if the total
            project source is below this threshold.  200 KB ≈ 57K tokens.
        max_gather_bytes: Maximum bytes to gather (for large projects
            where the planner selects files).
        max_files: Maximum number of files to read.
        enable_planner: When False, always dump everything (fast path).
        verify_commands: Shell commands to run after operations execute.
            Each command is run in order; if any returns non-zero, the
            errors are fed back to the LLM for a fix attempt.
            Defaults to auto-detecting the project's linter.
        max_fix_attempts: Maximum number of verify → fix cycles.
    """

    max_context_bytes: int = 200_000
    max_gather_bytes: int = 400_000
    max_files: int = 50
    enable_planner: bool = True
    verify_commands: tuple[str, ...] = ()
    max_fix_attempts: int = 2


# ── Result types ──────────────────────────────────────────────


@dataclass
class ContextFirstResult:
    """Result from the Context-First Solver."""

    answer: str
    reasoning: str
    ops: list[tuple[str, str, Any]]
    files_created: list[str]
    success: bool
    method: str  # "context_first_fast" or "context_first_planned"
    total_tokens: int = 0

    def to_result_dict(self) -> dict[str, Any]:
        """Convert to the same dict format ``_handle_rlm()`` returns."""
        return {
            "answer": self.answer,
            "task_type": "ask",
            "method": self.method,
            "iterations": 1,
            "trajectory_length": 0,
            "final_reasoning": self.reasoning,
            "total_tokens": self.total_tokens,
            "ops": [
                {"type": o[0], "target": o[1], "detail": o[2]}
                for o in self.ops
            ],
            "files_created": self.files_created,
        }


@dataclass
class PlanResult:
    """Output of the ContextPlanner."""

    files_to_read: list[str]
    search_patterns: list[str]
    reasoning: str


@dataclass
class SolveResult:
    """Output of the SingleShotSolver."""

    reasoning: str
    answer: str
    operations: list[dict[str, Any]]


# ── File manifest helpers ─────────────────────────────────────

# Extensions to include in the manifest (source code only).
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
    ".md", ".txt", ".rst", ".sh", ".bash", ".zsh",
    ".sql", ".graphql", ".proto",
    ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".r", ".jl",
    ".vue", ".svelte", ".astro",
    ".dockerfile", ".xml", ".csv",
})

# Directories to always skip.
_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
    ".next", ".nuxt", ".output",
})


def build_file_manifest(
    project_root: str | Path,
) -> list[tuple[str, int]]:
    """Scan *project_root* and return ``(relative_path, size)`` pairs.

    Only includes source-code files, skipping binary blobs,
    ``node_modules``, ``__pycache__``, ``.git``, etc.
    """
    root = Path(project_root).resolve()
    manifest: list[tuple[str, int]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place.
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]

        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            suffix = fpath.suffix.lower()

            # Dockerfile has no extension
            if fname.lower() in ("dockerfile", "makefile", "rakefile"):
                suffix = ".dockerfile"

            if suffix not in _SOURCE_EXTENSIONS:
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            # Skip very large single files (> 1 MB likely generated).
            if size > 1_000_000:
                continue

            rel = str(fpath.relative_to(root))
            manifest.append((rel, size))

    return sorted(manifest, key=lambda x: x[0])


def manifest_to_string(manifest: list[tuple[str, int]]) -> str:
    """Format a manifest as a newline-separated string for the LLM."""
    lines = [f"{path} ({size} bytes)" for path, size in manifest]
    return "\n".join(lines)


# ── Context Planner ───────────────────────────────────────────


def _normalize_list(value: Any) -> list[Any]:
    """Coerce an LLM output to a Python list (defensive parsing)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        if "\n" in text:
            return [
                line.strip()
                for line in text.split("\n")
                if line.strip()
            ]
        if text:
            return [text]
    return []


class ContextPlanner(dspy.Module):
    """Decide what files need to be read to solve a task (1 LLM call)."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ContextPlanning)

    async def aplan(
        self,
        query: str,
        file_manifest: str,
        project_summary: str,
    ) -> PlanResult:
        """Plan which files to gather."""
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            file_manifest=file_manifest,
            project_summary=project_summary,
        )

        files = _normalize_list(
            getattr(result, "files_to_read", [])
        )
        patterns = _normalize_list(
            getattr(result, "search_patterns", [])
        )
        reasoning = str(getattr(result, "reasoning", ""))

        return PlanResult(
            files_to_read=[str(f) for f in files],
            search_patterns=[str(p) for p in patterns],
            reasoning=reasoning,
        )


# ── Context Gatherer ──────────────────────────────────────────


class ContextGatherer:
    """Read files and assemble context (pure I/O, no LLM)."""

    def __init__(
        self,
        max_bytes: int = 400_000,
        max_files: int = 50,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_files = max_files

    async def agather(
        self,
        files: list[str],
        project_root: str | Path,
        search_patterns: list[str] | None = None,
    ) -> str:
        """Read *files* and return assembled context string.

        Each file is prefixed with ``=== FILE: <path> ===`` so the
        LLM can identify boundaries.  Stops reading when
        ``max_bytes`` is reached.
        """
        root = Path(project_root).resolve()
        parts: list[str] = []
        total = 0
        read_count = 0

        for rel_path in files:
            if read_count >= self.max_files:
                break
            if total >= self.max_bytes:
                break

            fpath = root / rel_path
            if not fpath.is_file():
                # Try as absolute path
                fpath = Path(rel_path)
                if not fpath.is_file():
                    continue

            try:
                content = await asyncio.to_thread(
                    fpath.read_text, "utf-8"
                )
            except (OSError, UnicodeDecodeError):
                continue

            # Budget check.
            if total + len(content) > self.max_bytes:
                # Include a truncated version.
                remaining = self.max_bytes - total
                content = content[:remaining] + "\n... (truncated)"

            header = f"=== FILE: {rel_path} ===\n"
            parts.append(header + content)
            total += len(content)
            read_count += 1

        # Search patterns — match against already-read content.
        if search_patterns:
            combined = "\n".join(parts)
            matches: list[str] = []
            for pat in search_patterns[:5]:
                try:
                    for m in re.finditer(pat, combined):
                        # Show match with surrounding context.
                        start = max(0, m.start() - 100)
                        end = min(len(combined), m.end() + 100)
                        snippet = combined[start:end]
                        matches.append(
                            f"Pattern '{pat}' match: ...{snippet}..."
                        )
                        if len(matches) > 10:
                            break
                except re.error:
                    continue
            if matches:
                parts.append(
                    "=== SEARCH RESULTS ===\n"
                    + "\n".join(matches[:10])
                )

        return "\n\n".join(parts)

    def read_all_files(
        self,
        manifest: list[tuple[str, int]],
        project_root: str | Path,
    ) -> str:
        """Read ALL files from *manifest* (fast path for small projects)."""
        root = Path(project_root).resolve()
        parts: list[str] = []
        total = 0

        for rel_path, _size in manifest:
            if total >= self.max_bytes:
                break

            fpath = root / rel_path
            try:
                content = fpath.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            header = f"=== FILE: {rel_path} ===\n"
            parts.append(header + content)
            total += len(content)

        return "\n\n".join(parts)


# ── Single-Shot Solver ────────────────────────────────────────


class SingleShotSolver(dspy.Module):
    """Solve a coding task in one LLM call given full context."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(SingleShotSolving)

    async def asolve(
        self,
        query: str,
        full_context: str,
    ) -> SolveResult:
        """Solve the task and return structured operations."""
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            full_context=full_context,
        )

        reasoning = str(getattr(result, "reasoning", ""))
        answer = str(getattr(result, "answer", ""))
        raw_ops = str(getattr(result, "operations", "[]"))

        operations = _parse_operations(raw_ops)

        return SolveResult(
            reasoning=reasoning,
            answer=answer,
            operations=operations,
        )


def _parse_operations(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON operations string from LLM output."""
    text = raw.strip()

    # Try direct JSON parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                op for op in parsed
                if isinstance(op, dict) and "op" in op
            ]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON array from markdown code block.
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return [
                    op for op in parsed
                    if isinstance(op, dict) and "op" in op
                ]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try line-by-line JSON objects.
    ops: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "op" in obj:
                    ops.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
    return ops


# ── Operation Executor ────────────────────────────────────────


class OperationExecutor:
    """Execute file operations produced by the SingleShotSolver.

    Tracks operations in the same ``(type, target, detail)`` tuple
    format used by the RLM so the chat UI can display them.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.ops: list[tuple[str, str, Any]] = []
        self.files_created: list[str] = []
        self.attempted: int = 0  # total operations attempted
        self.failed: int = 0  # operations that silently failed

    async def execute(
        self, operations: list[dict[str, Any]]
    ) -> None:
        """Execute all operations sequentially."""
        for op in operations:
            op_type = op.get("op", "")
            if op_type in ("write_file", "edit_file"):
                self.attempted += 1
            try:
                if op_type == "write_file":
                    await self._write_file(op)
                elif op_type == "edit_file":
                    await self._edit_file(op)
                elif op_type == "run_cmd":
                    await self._run_cmd(op)
                else:
                    logger.warning("Unknown operation: %s", op_type)
            except Exception as exc:
                self.failed += 1
                logger.warning(
                    "Operation %s failed: %s", op_type, exc
                )

    async def _write_file(self, op: dict[str, Any]) -> None:
        path_str = op.get("path", "")
        content = op.get("content", "")
        if not path_str:
            self.failed += 1
            return

        path = Path(os.path.abspath(path_str))
        path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(path.write_text, content, "utf-8")

        self.ops.append(("write", str(path), len(content)))
        self.files_created.append(str(path))
        logger.info("Wrote %s (%d chars)", path, len(content))

    async def _edit_file(self, op: dict[str, Any]) -> None:
        path_str = op.get("path", "")
        old_text = op.get("old", "")
        new_text = op.get("new", "")
        if not path_str or not old_text:
            return

        path = Path(os.path.abspath(path_str))
        if not path.is_file():
            logger.warning("edit_file: %s does not exist", path)
            self.failed += 1
            return

        content = await asyncio.to_thread(path.read_text, "utf-8")
        if old_text not in content:
            logger.warning("edit_file: old text not found in %s", path)
            self.failed += 1
            return

        new_content = content.replace(old_text, new_text, 1)
        await asyncio.to_thread(
            path.write_text, new_content, "utf-8"
        )

        self.ops.append(("edit", str(path), len(new_content)))
        self.files_created.append(str(path))
        logger.info("Edited %s", path)

    async def _run_cmd(self, op: dict[str, Any]) -> None:
        import subprocess

        cmd = op.get("cmd", "")
        if not cmd:
            return

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(self.project_root),
        )

        self.ops.append(("cmd", cmd, result.returncode))
        if result.returncode != 0:
            logger.warning(
                "Command '%s' returned %d: %s",
                cmd,
                result.returncode,
                result.stderr[:500],
            )


# ── Verification ──────────────────────────────────────────────


def _detect_verify_commands(
    project_root: Path,
) -> list[str]:
    """Auto-detect verification commands for the project.

    Looks for common linter config files and returns appropriate
    commands.  Returns an empty list if nothing is detected.
    """
    cmds: list[str] = []

    # Python: ruff (fast, preferred)
    has_python = (
        (project_root / "pyproject.toml").is_file()
        or any(project_root.glob("*.py"))
    )
    has_ruff_config = (
        (project_root / ".ruff.toml").is_file()
        or (project_root / "pyproject.toml").is_file()
    )
    if has_python and has_ruff_config:
        cmds.append("ruff check .")

    # JavaScript/TypeScript: eslint
    has_eslint = (
        (project_root / ".eslintrc.json").is_file()
        or (project_root / ".eslintrc.js").is_file()
        or (project_root / "eslint.config.js").is_file()
    )
    if (project_root / "package.json").is_file() and has_eslint:
        cmds.append("npx eslint .")

    # Go
    if (project_root / "go.mod").is_file():
        cmds.append("go vet ./...")

    # Rust
    if (project_root / "Cargo.toml").is_file():
        cmds.append("cargo check")

    return cmds


@dataclass
class VerifyResult:
    """Result of running verification commands."""

    passed: bool
    errors: str  # Combined stderr/stdout from failing commands
    commands_run: list[str]


class Verifier:
    """Run verification commands (lint, type check, tests) on a project."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    async def averify(
        self,
        commands: list[str],
    ) -> VerifyResult:
        """Run *commands* and collect any errors.

        Returns a :class:`VerifyResult` with ``passed=True`` if all
        commands succeed (exit code 0), or ``passed=False`` with the
        combined error output.
        """
        import subprocess

        errors: list[str] = []
        commands_run: list[str] = []

        for cmd in commands:
            commands_run.append(cmd)
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self.project_root),
                )
                if result.returncode != 0:
                    output = ""
                    if result.stdout:
                        output += result.stdout[:3000]
                    if result.stderr:
                        output += "\n" + result.stderr[:3000]
                    errors.append(
                        f"$ {cmd} (exit {result.returncode})\n"
                        f"{output.strip()}"
                    )
            except subprocess.TimeoutExpired:
                errors.append(f"$ {cmd} (timed out after 120s)")
            except Exception as exc:
                errors.append(f"$ {cmd} (error: {exc})")

        return VerifyResult(
            passed=len(errors) == 0,
            errors="\n\n".join(errors),
            commands_run=commands_run,
        )


class FixSolver(dspy.Module):
    """Fix verification errors in one LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(VerificationFix)

    async def afix(
        self,
        query: str,
        full_context: str,
        verification_errors: str,
    ) -> list[dict[str, Any]]:
        """Produce fix operations for the given verification errors."""
        result = await asyncio.to_thread(
            self.predictor,
            query=query,
            full_context=full_context,
            verification_errors=verification_errors,
        )

        raw_ops = str(getattr(result, "operations", "[]"))
        return _parse_operations(raw_ops)


# ── Main Orchestrator ─────────────────────────────────────────


class ContextFirstSolver:
    """Orchestrates the context-first solving pipeline.

    Usage::

        solver = ContextFirstSolver(project_root="/path/to/project")
        result = await solver.asolve(query="Fix the bug in auth.py")
        if result.success:
            print(result.answer)
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        config: ContextFirstConfig | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = config or ContextFirstConfig()
        self._on_status = on_status

        self.planner = ContextPlanner()
        self.gatherer = ContextGatherer(
            max_bytes=self.config.max_gather_bytes,
            max_files=self.config.max_files,
        )
        self.solver = SingleShotSolver()
        self.fix_solver = FixSolver()
        self.verifier = Verifier(self.project_root)

    def _status(self, msg: str) -> None:
        """Emit a status update."""
        logger.info(msg)
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(msg)

    async def asolve(
        self,
        query: str,
        context: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> ContextFirstResult:
        """Run the context-first pipeline.

        Args:
            query: The user's task/question (already enriched by
                ``_build_rlm_query``).
            context: The assembled context from chat.py (project state,
                history summary, etc.).  Used for project_root discovery
                and as fallback context.
            conversation_history: Optional conversation history entries.

        Returns:
            A :class:`ContextFirstResult`.  Check ``success`` to decide
            whether to fall back to the RLM.
        """
        try:
            return await self._solve_internal(
                query, context, conversation_history
            )
        except Exception as exc:
            logger.warning(
                "Context-first solver failed: %s", exc
            )
            return ContextFirstResult(
                answer="",
                reasoning=f"Context-first failed: {exc}",
                ops=[],
                files_created=[],
                success=False,
                method="context_first_error",
            )

    async def _solve_internal(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, Any]] | None,
    ) -> ContextFirstResult:
        tokens_before = _get_dspy_token_usage()

        # 1. Build file manifest.
        self._status("Scanning project files...")
        manifest = build_file_manifest(self.project_root)

        if not manifest:
            logger.info("No source files found, skipping context-first")
            return ContextFirstResult(
                answer="",
                reasoning="No source files found in project",
                ops=[],
                files_created=[],
                success=False,
                method="context_first_empty",
            )

        total_size = sum(size for _, size in manifest)
        manifest_str = manifest_to_string(manifest)

        # Inject conversation context as extra info for the solver.
        extra_context = ""
        if context:
            # Include the chat-assembled context (learnings, recent turn).
            extra_context = (
                f"\n\n=== PROJECT CONTEXT ===\n{context[:4000]}"
            )

        # 2. Decide strategy: fast path vs planner.
        if (
            total_size <= self.config.max_context_bytes
            or not self.config.enable_planner
        ):
            # Fast path: project is small enough to dump everything.
            self._status(
                f"Reading all {len(manifest)} files "
                f"({total_size // 1024} KB)..."
            )
            full_context = self.gatherer.read_all_files(
                manifest, self.project_root
            )
            full_context += extra_context
            method = "context_first_fast"
        else:
            # Large project: use planner to select files.
            self._status("Planning context gathering...")
            project_summary = (
                f"Project at {self.project_root}\n"
                f"{len(manifest)} source files, "
                f"{total_size // 1024} KB total"
            )
            plan = await self.planner.aplan(
                query=query,
                file_manifest=manifest_str,
                project_summary=project_summary,
            )
            self._status(
                f"Reading {len(plan.files_to_read)} selected files..."
            )
            full_context = await self.gatherer.agather(
                files=plan.files_to_read,
                project_root=self.project_root,
                search_patterns=plan.search_patterns or None,
            )
            full_context += extra_context
            method = "context_first_planned"

        if not full_context.strip():
            return ContextFirstResult(
                answer="",
                reasoning="No context could be gathered",
                ops=[],
                files_created=[],
                success=False,
                method=method,
            )

        # 3. Single-shot solve.
        self._status("Solving in single shot...")
        solve_result = await self.solver.asolve(
            query=query,
            full_context=full_context,
        )

        # 4. Execute operations.
        executor = OperationExecutor(self.project_root)
        if solve_result.operations:
            self._status(
                f"Executing {len(solve_result.operations)} operations..."
            )
            await executor.execute(solve_result.operations)

        # 5. Verify & fix loop.
        verify_cmds = self._get_verify_commands()
        if verify_cmds and executor.files_created:
            await self._verify_and_fix(
                query=query,
                full_context=full_context,
                executor=executor,
                verify_cmds=verify_cmds,
            )

        tokens_after = _get_dspy_token_usage()
        total_tokens = max(0, tokens_after - tokens_before)

        # Determine success: if operations were attempted but ALL
        # failed (e.g. every edit_file had "old text not found"),
        # report failure so the caller can escalate.
        success = True
        if executor.attempted > 0 and executor.failed >= executor.attempted:
            logger.warning(
                "All %d file operations failed — marking as unsuccessful",
                executor.attempted,
            )
            success = False

        return ContextFirstResult(
            answer=solve_result.answer,
            reasoning=solve_result.reasoning,
            ops=executor.ops,
            files_created=executor.files_created,
            success=success,
            method=method,
            total_tokens=total_tokens,
        )

    def _get_verify_commands(self) -> list[str]:
        """Get verification commands: explicit config or auto-detected."""
        if self.config.verify_commands:
            return list(self.config.verify_commands)
        return _detect_verify_commands(self.project_root)

    async def _verify_and_fix(
        self,
        query: str,
        full_context: str,
        executor: OperationExecutor,
        verify_cmds: list[str],
    ) -> None:
        """Run verification and attempt fixes if it fails.

        Mutates *executor* by appending fix operations and files.
        """
        for attempt in range(1, self.config.max_fix_attempts + 1):
            self._status(
                f"Verifying (attempt {attempt}/"
                f"{self.config.max_fix_attempts})..."
            )
            vr = await self.verifier.averify(verify_cmds)

            if vr.passed:
                self._status("Verification passed")
                return

            logger.info(
                "Verification failed (attempt %d): %s",
                attempt,
                vr.errors[:200],
            )

            if attempt >= self.config.max_fix_attempts:
                self._status(
                    "Verification failed, max fix attempts reached"
                )
                logger.warning(
                    "Verification still failing after %d fix attempts",
                    self.config.max_fix_attempts,
                )
                return

            # Re-read affected files to get current state.
            self._status("Fixing verification errors...")
            current_context = await self._read_affected_files(
                executor.files_created
            )
            if not current_context:
                current_context = full_context

            try:
                fix_ops = await self.fix_solver.afix(
                    query=query,
                    full_context=current_context,
                    verification_errors=vr.errors,
                )
            except Exception as exc:
                logger.warning("Fix solver failed: %s", exc)
                return

            if not fix_ops:
                logger.info("Fix solver produced no operations")
                return

            self._status(
                f"Applying {len(fix_ops)} fix operations..."
            )
            await executor.execute(fix_ops)

    async def _read_affected_files(
        self, files: list[str]
    ) -> str:
        """Re-read files that were created/modified for fix context."""
        parts: list[str] = []
        for fpath_str in files:
            fpath = Path(fpath_str)
            if not fpath.is_file():
                continue
            try:
                content = await asyncio.to_thread(
                    fpath.read_text, "utf-8"
                )
                rel = str(fpath.relative_to(self.project_root))
            except (OSError, ValueError):
                rel = fpath_str
                try:
                    content = await asyncio.to_thread(
                        fpath.read_text, "utf-8"
                    )
                except (OSError, UnicodeDecodeError):
                    continue
            parts.append(f"=== FILE: {rel} ===\n{content}")
        return "\n\n".join(parts)
