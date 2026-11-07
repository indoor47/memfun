"""``memfun swarm`` — force multi-agent workflow execution (bypass triage).

This command is the user-facing entry-point that wires together the
parallel-agent infrastructure that landed in the parallel-agent epic
but had no direct invocation path:

* :class:`WorkflowEngine` — decompose-then-fan-out orchestration.
* :class:`WorktreeManager` — per-task isolated git worktrees so
  parallel writers don't clobber each other.
* :meth:`AgentOrchestrator.fan_out_to_pool` — per-instance pools so
  N sub-tasks of the same agent type can truly run in parallel.

Unlike ``memfun ask``/``memfun chat``, this command does **not** invoke
:class:`QueryTriage` — it sends the spec directly to the
:class:`WorkflowEngine`.  Use it when you know up-front that the task
naturally decomposes into parallel sub-tasks (e.g. *"refactor X across
modules A, B, C"*, *"build a CRUD endpoint with handler + tests +
docs"*).

Example::

    memfun swarm "Add a /health endpoint with handler, tests, and docs"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from memfun_core.config import MemfunConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from memfun_agent.workflow import SubTaskStatus, WorkflowResult

logger = logging.getLogger("cli.swarm")
console = Console()


# ── Helpers ────────────────────────────────────────────────────────


def _project_is_git_repo(project_root: Path) -> bool:
    """Return ``True`` if *project_root* is a git working tree.

    Uses ``git rev-parse --is-inside-work-tree`` rather than just
    looking for ``.git/`` because the latter is wrong for both
    submodules (``.git`` is a file) and worktrees (``.git`` is a
    file pointing at the real ``gitdir``).
    """
    import subprocess

    if not project_root.exists():
        return False
    try:
        proc = subprocess.run(
            ("git", "rev-parse", "--is-inside-work-tree"),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``Xm YYs`` or ``Xs``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:04.1f}s"


def _on_sub_task_status_factory(
    status_by_tid: dict[str, str],
) -> Any:
    """Build a callback that streams sub-task transitions to stdout.

    The callback is invoked by :class:`WorkflowEngine` every time a
    sub-task changes state (``pending`` → ``running`` → ``completed`` /
    ``failed``).  We print *only* on state changes so a single line
    is emitted per transition rather than spamming every poll.
    """

    def on_sub_task_status(tid: str, status: SubTaskStatus) -> None:
        prev = status_by_tid.get(tid)
        if prev == status.status:
            return
        status_by_tid[tid] = status.status

        st = status.status
        symbol = {
            "pending": "[dim]·[/dim]",
            "running": "[yellow]>[/yellow]",
            "completed": "[green]+[/green]",
            "failed": "[red]x[/red]",
            "revision": "[magenta]~[/magenta]",
        }.get(st, "?")
        agent_label = status.agent_name or "?"
        desc = (status.sub_task.description or "")[:80]
        console.print(f"  {symbol} [cyan]{tid}[/cyan] [dim]({agent_label})[/dim] {st:<10} {desc}")

    return on_sub_task_status


def _display_final_summary(
    result: WorkflowResult,
    wall_seconds: float,
) -> None:
    """Render the post-workflow summary panel."""
    succeeded = sum(1 for r in result.sub_task_results.values() if r.success)
    failed = sum(1 for r in result.sub_task_results.values() if not r.success)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value")
    table.add_row("workflow_id", result.workflow_id)
    table.add_row("success", str(result.success))
    table.add_row("sub_tasks", f"{succeeded} passed, {failed} failed")
    table.add_row("review_rounds", str(result.review_rounds))
    table.add_row(
        "files_created",
        ", ".join(result.files_created) or "(none)",
    )
    table.add_row("wall_time", _format_duration(wall_seconds))
    table.add_row(
        "workflow_duration",
        _format_duration(result.total_duration_ms / 1000.0),
    )
    if result.no_edits:
        table.add_row(
            "no_edits",
            f"[red]{result.no_edit_subtasks} code sub-task(s) produced no file changes[/red]",
        )
    if result.error:
        table.add_row("error", f"[red]{result.error}[/red]")

    # Auto-commit / auto-push outcome (issue #37).  Always shown
    # when the engine returned a CommitResult so the user knows
    # whether a commit landed without having to `git log`.
    cr = result.commit_result
    if cr is not None:
        if cr.committed and cr.commit_sha:
            short_sha = cr.commit_sha[:8]
            table.add_row(
                "commit",
                f"[green]{short_sha}[/green] on "
                f"[cyan]{cr.branch or '?'}[/cyan]"
                f" ({len(cr.files_committed)} file(s))",
            )
            if cr.pushed:
                table.add_row(
                    "push",
                    "[green]pushed to origin[/green]",
                )
            elif cr.push_error:
                table.add_row(
                    "push",
                    f"[yellow]not pushed: {cr.push_error}[/yellow]",
                )
        elif cr.skipped:
            table.add_row(
                "commit",
                f"[dim]skipped ({cr.error or 'unknown reason'})[/dim]",
            )
        elif cr.error:
            table.add_row(
                "commit",
                f"[red]failed: {cr.error}[/red]",
            )

    truncated = _truncated_responses()
    if truncated:
        table.add_row(
            "truncated",
            f"[red]LLM RESPONSE TRUNCATED x{truncated}: raise [llm] max_tokens "
            "(or [llm.roles.<role>].max_tokens)[/red]",
        )

    border = "green" if result.success and not truncated else "red"
    console.print()
    console.print(
        Panel(
            table,
            title="Swarm result",
            border_style=border,
        )
    )

    if result.answer:
        console.print()
        console.print(
            Panel(
                result.answer[:5000]
                + ("\n\n[dim]...truncated...[/dim]" if len(result.answer) > 5000 else ""),
                title="Merged answer",
                border_style="cyan",
            )
        )


# ── Opt-in machine-readable result (MEMFUN_SWARM_RESULT_JSON) ───────

#: When set, ``memfun swarm`` writes a JSON summary of the run to this path.
RESULT_JSON_ENV = "MEMFUN_SWARM_RESULT_JSON"
RESULT_JSON_SCHEMA_VERSION = 1
_MAX_OPS_PER_TASK = 2000
_MAX_DETAIL_CHARS = 200


def _normalize_op(op: Any) -> dict[str, str]:
    """``{type, target, detail}`` from an op dict or ``(type, target, detail)`` tuple."""
    if isinstance(op, dict):
        raw: dict[str, Any] = op  # pyright: ignore[reportUnknownVariableType]
        typ, target, detail = raw.get("type"), raw.get("target"), raw.get("detail")
    elif isinstance(op, (list, tuple)):
        seq: list[Any] = list(op)  # pyright: ignore[reportUnknownArgumentType]
        typ, target, detail = [*seq, None, None, None][:3]
    else:
        typ, target, detail = "unknown", op, None
    return {
        "type": str(typ or "unknown")[:_MAX_DETAIL_CHARS],
        "target": "" if target is None else str(target)[:_MAX_DETAIL_CHARS],
        "detail": "" if detail is None else str(detail)[:_MAX_DETAIL_CHARS],
    }


def _truncated_responses() -> int:
    """Truncated LLM responses in this process (all roles); 0 if telemetry is unavailable."""
    try:
        from memfun_agent.llm_telemetry import total_truncations

        return total_truncations()
    except Exception:
        return 0


def _reasoning_tokens() -> int:
    """Reasoning tokens the model server reported in this process; 0 if unavailable."""
    try:
        from memfun_agent.llm_telemetry import total_reasoning_tokens

        return total_reasoning_tokens()
    except Exception:
        return 0


def _swarm_result_payload(
    result: WorkflowResult | None,
    wall_seconds: float,
    sub_tasks_seen: dict[str, SubTaskStatus],
    error: str | None = None,
) -> dict[str, Any]:
    """JSON-safe summary of a swarm run.  Ops are commands/paths only, no prompts."""
    sub_results: dict[str, Any] = dict(result.sub_task_results) if result is not None else {}
    decomposition = result.decomposition if result is not None else None
    sub_tasks: dict[str, Any] = {}
    ops_total: dict[str, int] = {}
    for tid in sorted(set(sub_results) | set(sub_tasks_seen)):
        res = sub_results.get(tid)
        status = sub_tasks_seen.get(tid)
        data: dict[str, Any] = {}
        if res is not None and isinstance(getattr(res, "result", None), dict):
            data = res.result
        raw_ops: list[Any] = list(data.get("ops") or [])
        ops = [_normalize_op(o) for o in raw_ops[:_MAX_OPS_PER_TASK]]
        for op in [_normalize_op(o) for o in raw_ops]:
            ops_total[op["type"]] = ops_total.get(op["type"], 0) + 1
        iterations = data.get("iterations")
        sub_tasks[tid] = {
            "agent": (status.agent_name if status is not None else "")
            or (getattr(res, "agent_id", "") if res is not None else ""),
            "status": status.status if status is not None else None,
            "success": bool(res.success) if res is not None else None,
            "error": (
                str(res.error)[:_MAX_DETAIL_CHARS] if res is not None and res.error else None
            ),
            "iterations": iterations if isinstance(iterations, int) else None,
            "ops": ops,
            "ops_count": len(raw_ops),
            "truncated": len(raw_ops) > _MAX_OPS_PER_TASK,
            "files_created": [str(f) for f in cast("list[Any]", data.get("files_created") or [])][
                :500
            ],
            "started_at": status.started_at if status is not None else None,
            "completed_at": status.completed_at if status is not None else None,
            # Per-task review verdicts live in engine state, not on WorkflowResult.
            "review": None,
        }
    error_text = (result.error if result is not None else error) or None
    success = bool(result.success) if result is not None else False
    truncated = _truncated_responses()
    if truncated:
        success = False
        if not error_text or "truncated" not in error_text.lower():
            advice = (
                f"{truncated} LLM response(s) truncated (finish_reason=length): "
                "raise [llm] max_tokens (or [llm.roles.<role>].max_tokens)"
            )
            error_text = f"{error_text}; {advice}" if error_text else advice
    return {
        "schema_version": RESULT_JSON_SCHEMA_VERSION,
        "workflow_id": result.workflow_id if result is not None else None,
        "success": success,
        "error": error_text,
        "truncated_responses": truncated,
        "reasoning_tokens": _reasoning_tokens(),
        "unaccepted_tasks": list(result.unaccepted_tasks) if result is not None else [],
        "review_errors": result.review_errors if result is not None else 0,
        "review_rounds": result.review_rounds if result is not None else 0,
        "no_edits": bool(result.no_edits) if result is not None else False,
        "no_edit_subtasks": result.no_edit_subtasks if result is not None else 0,
        "files_created": list(result.files_created) if result is not None else [],
        "decomposition": None
        if decomposition is None
        else {
            "n_sub_tasks": len(decomposition.sub_tasks),
            "is_single_task": bool(decomposition.is_single_task),
            "groups": [list(g) for g in decomposition.parallelism_groups],
        },
        "sub_tasks": sub_tasks,
        "ops_total": ops_total,
        "wall_seconds": round(wall_seconds, 3),
    }


def _write_swarm_result_json(payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``$MEMFUN_SWARM_RESULT_JSON`` if set; never raises.

    The file is written to an exclusive no-follow temp file and then hard-linked into
    place, so it appears atomically and an existing entry at the path (a planted file
    or symlink included) is refused rather than overwritten or followed.
    """
    target = os.environ.get(RESULT_JSON_ENV, "").strip()
    if not target:
        return
    path = Path(target)
    tmp = path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, default=str)
            os.link(tmp, path)
        finally:
            os.unlink(tmp)
    except Exception as exc:
        logger.warning("Could not write %s=%s: %s", RESULT_JSON_ENV, target, exc)


# ── Async runner ───────────────────────────────────────────────────


async def _run_swarm(
    spec: str,
    *,
    max_pool: int,
    use_worktrees: bool,
    base_sha: str | None,
    timeout: float,
    auto_commit: bool | None,
    auto_push: bool | None,
    auto_branch: str | None,
    gh_pr: bool | None,
    decompose: bool = True,
    single_task_solver: str = "context-first",
) -> int:
    """Execute the swarm workflow.  Returns the desired exit code."""
    # Lazy imports — keep CLI startup cheap when other commands run.
    from memfun_agent.git_committer import create_swarm_branch
    from memfun_agent.workflow import WorkflowEngine
    from memfun_runtime.builder import RuntimeBuilder
    from memfun_runtime.lifecycle import AgentManager
    from memfun_runtime.orchestrator import (
        AgentOrchestrator,
        OrchestratorConfig,
        RetryPolicy,
    )
    from memfun_runtime.worktree import WorktreeManager

    # Specialist agents must be imported so the @agent decorator
    # populates the global registry before AgentManager.start_agent
    # is called.  Same import as chat.py.
    try:
        import memfun_agent.specialists as _specialists

        _ = _specialists  # quiet "unused import" without dropping side effect
    except Exception:
        logger.debug("Specialist agents unavailable", exc_info=True)

    # Configure DSPy from the same helper used by chat / agent cmds.
    # These helpers are intentionally underscore-prefixed but shared
    # across CLI commands — same pattern as chat.py.
    from memfun_cli.commands.agent_commands import (
        _configure_dspy,
        _load_credentials,
    )

    _load_credentials()
    config = MemfunConfig.load()
    _configure_dspy(config)

    # Truncation must be visible in swarm: install the callback AND the
    # callback-independent log hook explicitly (``_configure_dspy`` returns
    # early without an API key, and the callback alone missed real runs).
    try:
        from memfun_agent.llm_telemetry import (
            install_dspy_callback,
            install_truncation_hook,
            reset_truncations,
        )

        reset_truncations()
        install_truncation_hook()
        install_dspy_callback()
    except Exception:
        logger.debug("Failed to install truncation detection", exc_info=True)

    project_root = Path.cwd()

    # Resolve auto-commit settings: CLI flags override memfun.toml.
    # The CLI flag default for ``auto_commit`` is ``True`` so the
    # user gets a commit by default — but we honour
    # ``--no-commit`` to suppress it explicitly.
    wf_cfg = config.workflow
    effective_auto_commit = auto_commit if auto_commit is not None else wf_cfg.auto_commit
    effective_auto_push = auto_push if auto_push is not None else wf_cfg.auto_push
    effective_auto_branch = auto_branch if auto_branch is not None else wf_cfg.auto_branch
    effective_gh_pr = gh_pr if gh_pr is not None else wf_cfg.gh_pr

    # ── Optional pre-workflow branch creation (issue #37) ────
    if effective_auto_branch:
        if not _project_is_git_repo(project_root):
            console.print("[yellow]--branch ignored: not a git repository.[/yellow]")
        else:
            branch_name, branch_err = create_swarm_branch(
                project_root,
                effective_auto_branch,
            )
            if branch_err:
                console.print(
                    Panel(
                        f"[red]Could not create branch '{branch_name}': {branch_err}[/red]",
                        title="Branch error",
                        border_style="red",
                    )
                )
                return 1
            console.print(
                f"[dim]Branch:[/dim] [green]{branch_name}[/green] [dim](created off HEAD)[/dim]"
            )

    # Worktree manager (requires a git repo).
    worktree_manager: WorktreeManager | None = None
    if use_worktrees:
        if not _project_is_git_repo(project_root):
            console.print(
                Panel(
                    f"[red]--use-worktrees requires a git repository, "
                    f"but {project_root} is not one.[/red]\n\n"
                    "Re-run with [bold]--no-use-worktrees[/bold] or "
                    "initialise a git repo first (`git init`).",
                    title="Swarm error",
                    border_style="red",
                )
            )
            return 2
        worktree_manager = WorktreeManager(project_root=project_root)
        if base_sha is not None:
            # Stash the override on the manager for later — but since
            # WorktreeManager.make_worktree takes base_sha per-call,
            # we wire it through via a closure-based wrapper below.
            console.print(f"[dim]Worktrees rooted at HEAD ref:[/dim] {base_sha}")

    # Wrap the worktree manager if a custom base_sha was provided so
    # every make_worktree call pulls from that ref.
    if worktree_manager is not None and base_sha is not None:
        original_make = worktree_manager.make_worktree

        def _make_with_base(
            workflow_id: str,
            task_id: str,
            base_sha_arg: str | None = None,
        ) -> Path:
            return original_make(
                workflow_id,
                task_id,
                base_sha_arg or base_sha,
            )

        worktree_manager.make_worktree = _make_with_base  # type: ignore[method-assign]

    # Build runtime.
    runtime = await RuntimeBuilder(config).build()

    # Bind the runtime for per-agent LLM telemetry (best-effort).
    try:
        from memfun_agent.llm_telemetry import set_active_runtime

        set_active_runtime(runtime)
    except Exception:
        logger.debug("Failed to bind telemetry runtime", exc_info=True)

    manager = AgentManager(runtime)

    # Start rlm-coder as the single-task fallback path (matches chat.py).
    if not manager.is_running("rlm-coder"):
        try:
            await manager.start_agent("rlm-coder")
        except Exception as exc:
            logger.warning("Could not start rlm-coder: %s", exc)

    orch_config = OrchestratorConfig(
        default_timeout_seconds=timeout,
        retry_policy=RetryPolicy.FIXED,
        max_retries=1,
        max_concurrency=max(max_pool, 1),
    )
    orchestrator = AgentOrchestrator(
        runtime,
        manager,
        config=orch_config,
    )

    status_by_tid: dict[str, str] = {}
    print_sub_task = _on_sub_task_status_factory(status_by_tid)
    sub_tasks_seen: dict[str, SubTaskStatus] = {}

    def on_sub_task(tid: str, status: SubTaskStatus) -> None:
        sub_tasks_seen[tid] = status
        print_sub_task(tid, status)

    engine = WorkflowEngine(
        context=runtime,
        orchestrator=orchestrator,
        manager=manager,
        on_sub_task_status=on_sub_task,
        worktree_manager=worktree_manager,
        auto_commit=effective_auto_commit,
        auto_push=effective_auto_push,
        project_root=project_root,
        decompose=decompose,
        single_task_solver=single_task_solver,
        gh_pr=effective_gh_pr,
    )
    # Expose the per-instance pool cap so _execute_group can read it.
    engine._max_pool = max_pool  # type: ignore[attr-defined]

    # Auto-start the embedded dashboard (mirrors chat.py).  Reuses an
    # existing dashboard for this project if one is already running.
    dashboard_port: int | None = None
    try:
        from memfun_cli.commands.chat import _start_dashboard_background

        dashboard_port, _is_owner = await _start_dashboard_background(runtime)
    except Exception as exc:
        logger.debug("Dashboard auto-start skipped: %s", exc)

    banner = (
        f"[bold]Spec:[/bold] {spec}\n"
        f"[dim]project:[/dim] {project_root}\n"
        f"[dim]max_pool:[/dim] {max_pool}  "
        f"[dim]use_worktrees:[/dim] {use_worktrees}  "
        f"[dim]base_sha:[/dim] {base_sha or 'HEAD'}"
    )
    if dashboard_port:
        banner += f"\n[dim]Dashboard:[/dim] [cyan]http://localhost:{dashboard_port}[/cyan]"
    console.print(
        Panel(
            banner,
            title="memfun swarm",
            border_style="cyan",
        )
    )
    console.print()
    console.print("[bold]Sub-task progress:[/bold]")

    start = time.perf_counter()
    try:
        from memfun_agent.role_models import role_scope

        # Orchestrator role: any LLM call the engine makes outside an
        # agent / decomposer binding runs on [llm.roles.orchestrator].
        with role_scope("orchestrator"):
            result = await engine.execute_workflow(
                task_description=spec,
                project_context="",  # WorkflowEngine gathers shared context internally
                conversation_history=None,
                project=project_root.name,
            )
    except Exception as exc:
        console.print(
            Panel(
                f"[red]{type(exc).__name__}: {exc}[/red]",
                title="Swarm error",
                border_style="red",
            )
        )
        _write_swarm_result_json(
            _swarm_result_payload(
                None,
                time.perf_counter() - start,
                sub_tasks_seen,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return 1
    finally:
        # Best-effort: stop all managed agents so the process can exit.
        try:
            await manager.stop_all()
        except Exception:
            logger.debug("manager.stop_all failed", exc_info=True)

    wall = time.perf_counter() - start
    _display_final_summary(result, wall)
    _write_swarm_result_json(_swarm_result_payload(result, wall, sub_tasks_seen))
    return 0 if result.success and not _truncated_responses() else 1


# ── Typer entrypoint ───────────────────────────────────────────────


def swarm_command(
    spec: str = typer.Argument(
        ...,
        help="What to build / refactor.  Passed straight to the WorkflowEngine — no triage gate.",
    ),
    max_pool: int = typer.Option(
        5,
        "--max-pool",
        min=1,
        max=100,
        help="Maximum number of agent instances spun up per agent type. "
        "Drives the orchestrator semaphore and per-pool fan-out.",
    ),
    decompose: bool = typer.Option(
        True,
        "--decompose/--no-decompose",
        help="Let the LLM planner split the spec into sub-tasks (default).  "
        "--no-decompose skips the planner and runs the whole spec as ONE coding task.",
    ),
    single_task_solver: str = typer.Option(
        "context-first",
        "--single-task-solver",
        help="How a single (undecomposed) task is solved: 'context-first' (one-shot solver "
        "with a verify->fix loop, default) or 'rlm' (the iterative agent loop that sees "
        "tool and test output).",
    ),
    use_worktrees: bool = typer.Option(
        True,
        "--use-worktrees/--no-use-worktrees",
        help="Run each sub-task in its own git worktree (recommended for parallel safety).",
    ),
    base_sha: str | None = typer.Option(
        None,
        "--base-sha",
        help="Base SHA / ref for worktrees (defaults to HEAD).",
    ),
    timeout: int = typer.Option(
        1800,
        "--timeout",
        min=1,
        help="Per-task timeout in seconds.",
    ),
    commit: bool | None = typer.Option(
        None,
        "--commit/--no-commit",
        help="Auto-commit produced files after a successful swarm "
        "(issue #37).  Unset → read [workflow] auto_commit from "
        "memfun.toml (default ON).  Pass --no-commit to suppress.",
    ),
    push: bool | None = typer.Option(
        None,
        "--push/--no-push",
        help="After auto-commit, push the commit to ``origin``.  "
        "Unset → read [workflow] auto_push from memfun.toml "
        "(default OFF).",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Create a new branch off the current HEAD BEFORE the "
        "swarm runs.  Pass 'auto' for a timestamped "
        "``memfun/swarm-<ts>`` name, or any other string to use that "
        "name verbatim.",
    ),
    gh_pr: bool | None = typer.Option(
        None,
        "--gh-pr/--no-gh-pr",
        help="Open a GitHub PR per sub-task and route each one "
        "through the pr-reviewer agent (issue #38). "
        "Requires `gh` authenticated with the project's remote. "
        "Unset → read [workflow] gh_pr from memfun.toml (default OFF).",
    ),
) -> None:
    """Force the multi-agent WorkflowEngine path, bypassing triage.

    Useful when you have a task that naturally decomposes into parallel
    sub-tasks and want them dispatched as a swarm with per-task worktree
    isolation, rather than handled in one shot by the context-first
    solver.
    """
    if single_task_solver not in ("context-first", "rlm"):
        raise typer.BadParameter(
            f"--single-task-solver must be 'context-first' or 'rlm', got {single_task_solver!r}"
        )
    exit_code = asyncio.run(
        _run_swarm(
            spec,
            max_pool=max_pool,
            use_worktrees=use_worktrees,
            base_sha=base_sha,
            timeout=float(timeout),
            auto_commit=commit,
            auto_push=push,
            auto_branch=branch,
            gh_pr=gh_pr,
            decompose=decompose,
            single_task_solver=single_task_solver,
        )
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
