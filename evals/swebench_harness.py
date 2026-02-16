"""SWE-bench evaluation harness for Memfun.

Runs Memfun's context-first solver (with automatic escalation to
multi-agent workflow) on SWE-bench instances and produces a predictions
JSONL file compatible with `swebench.harness.run_evaluation`.

Usage:
    python evals/swebench_harness.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --output evals/predictions.jsonl \
        --max-instances 10

Prerequisites:
    pip install swebench datasets
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("memfun.eval.swebench")


# ---------------------------------------------------------------------------
# DSPy / LLM configuration
# ---------------------------------------------------------------------------


def _configure_dspy() -> None:
    """Configure DSPy LM from memfun.toml or environment variables."""
    import dspy
    from memfun_core.config import MemfunConfig

    config = MemfunConfig.load()
    provider = config.llm.provider
    model = config.llm.model
    api_key = os.environ.get(config.llm.api_key_env, "")

    # Also check ~/.memfun/credentials.json (memfun init stores keys there)
    if not api_key:
        creds_path = Path.home() / ".memfun" / "credentials.json"
        if creds_path.exists():
            try:
                creds = json.loads(creds_path.read_text())
                api_key = creds.get(config.llm.api_key_env, "")
                if api_key:
                    os.environ[config.llm.api_key_env] = api_key
                    logger.info("Loaded API key from %s", creds_path)
            except (json.JSONDecodeError, OSError):
                pass

    if not api_key and provider != "ollama":
        logger.error(
            "No API key found. Set %s or configure via memfun.toml",
            config.llm.api_key_env,
        )
        sys.exit(1)

    if provider == "anthropic":
        lm_model = f"anthropic/{model}"
    elif provider == "openai":
        lm_model = f"openai/{model}"
    elif provider == "ollama":
        lm_model = f"ollama_chat/{model}"
        api_key = "ollama"
    else:
        lm_model = model

    kwargs: dict[str, object] = {
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if config.llm.base_url:
        kwargs["api_base"] = config.llm.base_url

    lm = dspy.LM(lm_model, **kwargs)
    dspy.configure(lm=lm)
    logger.info("DSPy configured: %s (max_tokens=%s)", lm_model, config.llm.max_tokens)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _clone_repo(repo: str, base_commit: str, dest: Path) -> bool:
    """Clone a repo at a specific commit into dest directory.

    Returns True on success.
    """
    # repo format: "owner/name" -> https://github.com/owner/name.git
    url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", url, str(dest)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", base_commit],
            check=True,
            capture_output=True,
            cwd=str(dest),
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.error("Failed to clone %s@%s: %s", repo, base_commit[:8], exc)
        return False


def _collect_diff(repo_dir: Path) -> str:
    """Collect a unified diff of all changes in the repo."""
    try:
        # Stage everything so we capture new files too
        subprocess.run(
            ["git", "add", "-A"],
            check=True,
            capture_output=True,
            cwd=str(repo_dir),
        )
        result = subprocess.run(
            ["git", "diff", "--cached"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        return result.stdout
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to collect diff: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Memfun solver
# ---------------------------------------------------------------------------


async def _solve_instance(
    instance_id: str,
    problem_statement: str,
    repo_dir: Path,
) -> str:
    """Run Memfun on a single SWE-bench instance.

    Returns the unified diff produced by Memfun.
    """
    from memfun_agent.context_first import ContextFirstConfig, ContextFirstSolver

    logger.info("[%s] Solving with context-first...", instance_id)

    solver = ContextFirstSolver(
        project_root=repo_dir,
        config=ContextFirstConfig(
            max_context_bytes=400_000,
            max_gather_bytes=600_000,
            max_files=80,
            enable_planner=True,
            max_fix_attempts=2,
            enable_edit_retry=True,
            enable_consistency_review=True,
        ),
    )

    # Tier 1: Context-first
    result = await solver.asolve(
        query=problem_statement,
        category="task",
    )

    if result.success and result.files_created:
        logger.info(
            "[%s] Context-first succeeded: %d files modified",
            instance_id,
            len(result.files_created),
        )
        return _collect_diff(repo_dir)

    # Tier 2: Multi-agent workflow
    logger.info(
        "[%s] Context-first %s — escalating to workflow",
        instance_id,
        result.method,
    )
    try:
        diff = await _solve_with_workflow(instance_id, problem_statement, repo_dir)
        if diff:
            return diff
    except Exception as exc:
        logger.warning("[%s] Workflow failed: %s", instance_id, exc)

    # If context-first produced partial output, use that
    if result.files_created:
        return _collect_diff(repo_dir)

    logger.warning("[%s] No solution produced", instance_id)
    return ""


async def _solve_with_workflow(
    instance_id: str,
    problem_statement: str,
    repo_dir: Path,
) -> str:
    """Run the multi-agent workflow on a single instance."""
    from memfun_agent.workflow import WorkflowEngine
    from memfun_core.config import MemfunConfig
    from memfun_runtime.builder import RuntimeBuilder
    from memfun_runtime.lifecycle import AgentManager
    from memfun_runtime.orchestrator import AgentOrchestrator, OrchestratorConfig

    config = MemfunConfig.load()
    runtime = await RuntimeBuilder(config).build()
    manager = AgentManager(runtime)
    await manager.start_agent("rlm-coder")

    orchestrator = AgentOrchestrator(
        runtime,
        manager,
        config=OrchestratorConfig(default_timeout_seconds=600.0),
    )

    engine = WorkflowEngine(
        context=runtime,
        orchestrator=orchestrator,
        manager=manager,
    )

    # Build minimal project context
    project_context = f"Project root: {repo_dir}\n"

    wf_result = await engine.execute_workflow(
        task_description=problem_statement,
        project_context=project_context,
    )

    if wf_result.success and wf_result.files_created:
        logger.info(
            "[%s] Workflow succeeded: %d files, %d ops",
            instance_id,
            len(wf_result.files_created),
            len(wf_result.ops),
        )
        return _collect_diff(repo_dir)

    return ""


# ---------------------------------------------------------------------------
# Main harness loop
# ---------------------------------------------------------------------------


def _load_dataset(dataset_name: str, split: str = "test") -> list[dict]:
    """Load SWE-bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        sys.exit(1)

    logger.info("Loading dataset %s (split=%s)...", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    return list(ds)


async def _run_instance(
    instance: dict,
    work_dir: Path,
    model_name: str,
) -> dict | None:
    """Process a single SWE-bench instance.

    Returns a prediction dict or None on failure.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem = instance["problem_statement"]

    logger.info("=" * 60)
    logger.info("[%s] Starting", instance_id)
    logger.info("  repo: %s  commit: %s", repo, base_commit[:8])

    # Clone repo
    repo_dir = work_dir / instance_id.replace("/", "__")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    if not _clone_repo(repo, base_commit, repo_dir):
        return None

    # Solve
    start = time.monotonic()
    try:
        # Change to repo dir so Memfun sees it as project root
        original_cwd = os.getcwd()
        os.chdir(str(repo_dir))

        diff = await _solve_instance(instance_id, problem, repo_dir)
    except Exception as exc:
        logger.error("[%s] Solver exception: %s", instance_id, exc)
        diff = ""
    finally:
        os.chdir(original_cwd)

    elapsed = time.monotonic() - start
    logger.info(
        "[%s] Done in %.1fs — diff: %d chars",
        instance_id,
        elapsed,
        len(diff),
    )

    # Clean up repo to save disk
    shutil.rmtree(repo_dir, ignore_errors=True)

    if not diff:
        return None

    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": diff,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Memfun on SWE-bench and produce predictions JSONL",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name (default: SWE-bench_Lite)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--output",
        default="evals/predictions.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model-name",
        default="memfun-v0.2.0",
        help="Model name for predictions",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Max instances to process (0 = all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel instances (keep low, each uses LLM API)",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Working directory for repo clones (default: temp dir)",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="*",
        default=None,
        help="Specific instance IDs to run (default: all)",
    )

    args = parser.parse_args()

    # Configure LLM
    _configure_dspy()

    # Load dataset
    instances = _load_dataset(args.dataset, args.split)
    logger.info("Loaded %d instances", len(instances))

    # Filter
    if args.instance_ids:
        id_set = set(args.instance_ids)
        instances = [i for i in instances if i["instance_id"] in id_set]
        logger.info("Filtered to %d instances", len(instances))

    if args.max_instances > 0:
        instances = instances[: args.max_instances]
        logger.info("Limited to %d instances", len(instances))

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="memfun-swebench-"))
        cleanup_work = True

    logger.info("Work dir: %s", work_dir)

    # Process instances
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    solved = 0
    failed = 0

    if args.max_workers <= 1:
        # Sequential
        for instance in instances:
            pred = await _run_instance(instance, work_dir, args.model_name)
            if pred:
                predictions.append(pred)
                solved += 1
            else:
                failed += 1
    else:
        # Parallel with semaphore
        sem = asyncio.Semaphore(args.max_workers)

        async def _bounded(inst: dict) -> dict | None:
            async with sem:
                return await _run_instance(inst, work_dir, args.model_name)

        results = await asyncio.gather(
            *[_bounded(inst) for inst in instances],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, dict):
                predictions.append(r)
                solved += 1
            else:
                if isinstance(r, Exception):
                    logger.error("Instance exception: %s", r)
                failed += 1

    # Write predictions
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info("=" * 60)
    logger.info("RESULTS: %d solved, %d failed, %d total", solved, failed, len(instances))
    logger.info("Predictions written to: %s", output_path)

    if cleanup_work:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
