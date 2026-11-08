#!/usr/bin/env python3
"""Harness-benchmark arms: run ONE coding agent on ONE episode repo (stdlib only).

``run_smoke.py --arm <id>`` runs this file in the episode repo (cwd) in place of
``memfun swarm``.  Every arm gets the SAME task text (``--spec-file``, byte-identical to
what memfun gets), the same visible tests (the runner's ``.memfun/verify.json``), the same
sampling and the same OpenAI-compatible endpoint; only the agent harness differs.  The
hidden grader copy is never reachable from here: grading is the runner's job.

Arms (``ARMS``):

``single-shot``    one chat call: task, target file(s) and the visible tests in, whole new
                   file content out, written to disk.  No tools, no iteration (lower bound).
``mini-swe-agent`` the bash-only agent (``mini`` CLI, text-based actions, litellm).
``aider``          aider-chat, its own edit format, ``--auto-test`` on the visible tests.
``claude-code``    ``claude -p`` through an Anthropic-Messages-speaking base URL
                   (e.g. a LiteLLM proxy in front of the model server).

On exit it writes the runner's result file (``MEMFUN_SWARM_RESULT_JSON``, the same
schema the runner reads from ``memfun swarm``) with ``no_edits``, token usage and a
``failure_class``.  Exit code 0 unless the harness itself crashed or timed out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ARMS = ("single-shot", "mini-swe-agent", "aider", "claude-code")
RESULT_ENV = "MEMFUN_SWARM_RESULT_JSON"
RESULT_SCHEMA = 1
#: Visible-test files shown to the single-shot arm are capped (bytes) so one huge test
#: module cannot blow the window.
MAX_TEST_BYTES = 60_000
MAX_FILE_BYTES = 200_000
CONTEXT_LENGTH_RE = re.compile(
    r"context_length_exceeded|context length|context window|maximum context|too many tokens|"
    r"prompt is too long|reduce the length",
    re.I,
)
TRUNCATED_RE = re.compile(r"finish_reason['\": ]+length|response truncated|max_tokens", re.I)


class ArmError(Exception):
    """The harness itself failed (not the model): reported as a harness crash."""


# ── Shared helpers ──────────────────────────────────────────────────────────────


def visible_command(repo: Path) -> str | None:
    """The visible test command the runner wrote (``.memfun/verify.json``), if any."""
    try:
        doc = json.loads((repo / ".memfun" / "verify.json").read_text("utf-8"))
        cmds = [c for c in doc.get("commands", []) if isinstance(c, str)]
    except (OSError, ValueError, AttributeError):
        return None
    return cmds[0] if cmds else None


def visible_test_files(repo: Path, cmd: str | None) -> list[str]:
    """Test files named by the visible test command (relative paths that exist)."""
    if not cmd:
        return []
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return []
    out: list[str] = []
    for tok in tokens:
        path = tok.split("::", 1)[0]
        if path.endswith(".py") and (repo / path).is_file() and path not in out:
            out.append(path)
    return out


def _safe_rel(repo: Path, rel: str) -> Path:
    """*rel* resolved inside *repo*; anything escaping it, or a symlink, is refused."""
    p = repo / rel
    if p.is_symlink() or repo.resolve() not in p.resolve().parents:
        raise ArmError(f"refusing path outside the repo or a symlink: {rel}")
    return p


def changed_owned(repo: Path, owned: list[str]) -> bool:
    """Whether any owned file differs from the start commit (new files count)."""
    if not owned:
        return False
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *owned],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return bool(proc.stdout.strip())


def git_version() -> str:
    with contextlib.suppress(Exception):
        return subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    return ""


def run_group(cmd: list[str], env: dict[str, str], timeout: float, log: Path) -> tuple[int, bool]:
    """Run *cmd* in its own process group with a hard timeout; ``(returncode, timed_out)``."""
    with log.open("ab") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -9, True


def write_result(path: Path | None, doc: dict[str, Any]) -> None:
    if path is None:
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)


def token_count(text: str) -> int | None:
    """``12.5k`` / ``812`` as an int (aider's token lines)."""
    m = re.fullmatch(r"([\d.,]+)\s*([kKmM]?)", text.strip())
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    return int(n * {"": 1, "k": 1_000, "m": 1_000_000}[m.group(2).lower()])


# ── Arm: single-shot ────────────────────────────────────────────────────────────

SINGLE_SYSTEM = (
    "You are an expert Python engineer. You solve the task in ONE reply: you cannot run "
    "code or see test output. Reply with the COMPLETE new content of every file you must "
    "change, each as:\n=== FILE: <path> ===\n```python\n<entire file>\n```\n"
    "Output nothing else of substance. Never shorten a file with '...' or 'unchanged' markers."
)
FILE_BLOCK_RE = re.compile(
    r"=== FILE: (?P<path>[^\n=]+?) ===\s*\n```[a-zA-Z]*\n(?P<body>.*?)\n```", re.S
)


def build_single_shot_prompt(spec: str, repo: Path, owned: list[str], tests: list[str]) -> str:
    parts = [spec.strip(), ""]
    for rel in owned:
        p = repo / rel
        text = p.read_text("utf-8", "replace")[:MAX_FILE_BYTES] if p.is_file() else ""
        parts += [f"=== CURRENT FILE: {rel} ===", "```python", text, "```", ""]
    budget = MAX_TEST_BYTES
    for rel in tests:
        text = (repo / rel).read_text("utf-8", "replace")[:budget]
        budget -= len(text)
        parts += [f"=== VISIBLE TEST FILE: {rel} ===", "```python", text, "```", ""]
        if budget <= 0:
            break
    parts.append(
        "Reply with the complete new content of the file(s) to change, in the format required."
    )
    return "\n".join(parts)


def chat_call(
    base_url: str, model: str, system: str, user: str, a: argparse.Namespace
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
    }
    if a.top_p is not None:
        body["top_p"] = a.top_p
    if a.repetition_penalty is not None:
        body["repetition_penalty"] = a.repetition_penalty
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(req, timeout=a.timeout) as resp:
        return json.load(resp)


def run_single_shot(a: argparse.Namespace, repo: Path, spec: str, res: dict[str, Any]) -> None:
    owned = list(a.owned)
    tests = visible_test_files(repo, visible_command(repo))
    prompt = build_single_shot_prompt(spec, repo, owned, tests)
    res["calls"] = 1
    try:
        out = chat_call(a.base_url, a.model, SINGLE_SYSTEM, prompt, a)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:2000]
        if exc.code == 400 and CONTEXT_LENGTH_RE.search(text):
            res["failure_class"] = "context_length_exceeded"
            return
        raise ArmError(f"HTTP {exc.code}: {text[:300]}") from exc
    usage = out.get("usage") or {}
    res["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
    res["completion_tokens"] = int(usage.get("completion_tokens") or 0)
    choice = (out.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        res["truncated_responses"] = 1
        res["failure_class"] = "truncated"
        return
    content = (choice.get("message") or {}).get("content") or ""
    blocks = FILE_BLOCK_RE.findall(content)
    if not blocks and len(owned) == 1:  # a single fenced block with no header
        m = re.search(r"```[a-zA-Z]*\n(.*?)\n```", content, re.S)
        if m:
            blocks = [(owned[0], m.group(1))]
    written = 0
    for rel, body in blocks:
        rel = rel.strip()
        if rel not in owned:
            continue
        _safe_rel(repo, rel).write_text(body + "\n", "utf-8")
        written += 1
    if not written:
        res["failure_class"] = "format_error"


# ── Arm: mini-swe-agent ─────────────────────────────────────────────────────────

MINI_CONFIG = "mini_textbased.yaml"
MINI_MODEL_CLASS = "litellm_textbased"


def mini_command(a: argparse.Namespace, spec: str, traj: Path) -> list[str]:
    """``mini`` (mini-swe-agent v2) with the text-based config: no server-side tool parser."""
    mk = "model.model_kwargs"
    cfg = [
        f"model.model_name=openai/{a.model}",
        f"{mk}.api_base={a.base_url}",
        f"{mk}.api_key=dummy",
        f"{mk}.temperature={a.temperature}",
        f"{mk}.max_tokens={a.max_tokens}",
        "agent.mode=yolo",
        "agent.cost_limit=0",
    ]
    if a.top_p is not None:
        cfg.append(f"{mk}.top_p={a.top_p}")
    if a.repetition_penalty is not None:
        cfg.append(f"{mk}.extra_body.repetition_penalty={a.repetition_penalty}")
    cmd = [*shlex.split(a.bin or "mini"), "-c", MINI_CONFIG]
    for c in cfg:
        cmd += ["-c", c]
    return [
        *cmd,
        "--model-class",
        MINI_MODEL_CLASS,
        "-t",
        spec,
        "-y",
        "--exit-immediately",
        "-o",
        str(traj),
    ]


def mini_usage(traj: Path) -> dict[str, int]:
    """Token totals and call count from mini's trajectory file (best effort)."""
    prompt = completion = calls = 0
    try:
        doc = json.loads(traj.read_text("utf-8"))
    except (OSError, ValueError):
        return {}

    def walk(node: Any) -> None:
        nonlocal prompt, completion, calls
        if isinstance(node, dict):
            usage = node.get("usage")
            if isinstance(usage, dict) and "prompt_tokens" in usage:
                prompt += int(usage.get("prompt_tokens") or 0)
                completion += int(usage.get("completion_tokens") or 0)
                calls += 1
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "calls": calls}


def run_mini(a: argparse.Namespace, repo: Path, spec: str, res: dict[str, Any]) -> None:
    home = Path(os.environ["HOME"])
    traj = home / "mini-trajectory.json"
    env = {
        **os.environ,
        "MSWEA_GLOBAL_CONFIG_DIR": str(home / "mswea"),  # never the user's config
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_COST_TRACKING": "ignore_errors",  # a local model has no price table
        "MSWEA_CONFIGURED": "true",  # skip the first-run setup wizard
        "OPENAI_API_KEY": "dummy",
        "PAGER": "cat",
    }
    log = Path(a.log)
    rc, timed_out = run_group(mini_command(a, spec, traj), env, a.timeout, log)
    res.update(mini_usage(traj))
    res["harness_returncode"] = rc
    if timed_out:
        res["failure_class"] = "timeout"
        return
    text = log.read_text("utf-8", "replace") if log.is_file() else ""
    if CONTEXT_LENGTH_RE.search(text) and not changed_owned(repo, list(a.owned)):
        res["failure_class"] = "context_length_exceeded"
    elif rc != 0:
        res["failure_class"] = "harness_crash"


# ── Arm: aider ──────────────────────────────────────────────────────────────────

AIDER_TOKENS_RE = re.compile(
    r"Tokens: ([\d.,]+k?) sent(?:, [\d.,k]+ cache \w+)*, ([\d.,]+k?) received"
)


def aider_files(repo: Path, a: argparse.Namespace) -> tuple[Path, Path]:
    """Model settings (sampling, edit format) and metadata (window) files for aider."""
    home = Path(os.environ["HOME"])
    name = f"openai/{a.model}"
    extra: dict[str, Any] = {"temperature": a.temperature, "max_tokens": a.max_tokens}
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    if a.repetition_penalty is not None:
        extra["extra_body"] = {"repetition_penalty": a.repetition_penalty}
    settings = home / "aider-model-settings.json"
    # aider reads YAML; JSON is valid YAML.
    settings.write_text(
        json.dumps([{"name": name, "edit_format": a.edit_format, "extra_params": extra}]), "utf-8"
    )
    meta = home / "aider-model-metadata.json"
    meta.write_text(
        json.dumps(
            {
                name: {
                    "max_input_tokens": a.context_window,
                    "max_output_tokens": a.max_tokens,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
        ),
        "utf-8",
    )
    return settings, meta


def aider_command(a: argparse.Namespace, repo: Path, spec_file: Path) -> list[str]:
    settings, meta = aider_files(repo, a)
    cmd = [
        *shlex.split(a.bin or "aider"),
        "--model",
        f"openai/{a.model}",
        "--openai-api-base",
        a.base_url,
        "--openai-api-key",
        "dummy",
        "--model-settings-file",
        str(settings),
        "--model-metadata-file",
        str(meta),
        "--edit-format",
        a.edit_format,
        "--yes-always",
        "--no-auto-commits",
        "--no-dirty-commits",
        "--no-stream",
        "--no-pretty",
        "--no-fancy-input",
        "--no-check-update",
        "--no-analytics",
        "--no-show-release-notes",
        "--no-show-model-warnings",
        "--no-suggest-shell-commands",
        "--no-gitignore",
        "--chat-history-file",
        str(Path(os.environ["HOME"]) / "aider.chat.history.md"),
        "--input-history-file",
        str(Path(os.environ["HOME"]) / "aider.input.history"),
        "--message-file",
        str(spec_file),
    ]
    vcmd = visible_command(repo)
    if vcmd:  # the way Commit0's aider agent iterates: auto-test on the visible tests
        cmd += ["--test-cmd", vcmd, "--auto-test"]
    return [*cmd, *a.owned]


def run_aider(a: argparse.Namespace, repo: Path, spec: str, res: dict[str, Any]) -> None:
    env = {**os.environ, "OPENAI_API_KEY": "dummy", "AIDER_ANALYTICS": "false"}
    log = Path(a.log)
    rc, timed_out = run_group(aider_command(a, repo, Path(a.spec_file)), env, a.timeout, log)
    text = log.read_text("utf-8", "replace") if log.is_file() else ""
    sent = recv = 0
    for m in AIDER_TOKENS_RE.finditer(text):
        sent += token_count(m.group(1)) or 0
        recv += token_count(m.group(2)) or 0
    res.update(
        prompt_tokens=sent,
        completion_tokens=recv,
        calls=len(AIDER_TOKENS_RE.findall(text)),
        usage_note="parsed from aider's 'Tokens:' lines (rounded); the server counters are exact",
        harness_returncode=rc,
    )
    if timed_out:
        res["failure_class"] = "timeout"
    elif CONTEXT_LENGTH_RE.search(text) and not changed_owned(repo, list(a.owned)):
        res["failure_class"] = "context_length_exceeded"
    elif rc != 0:
        res["failure_class"] = "harness_crash"


# ── Arm: claude-code ────────────────────────────────────────────────────────────


def run_claude_code(a: argparse.Namespace, repo: Path, spec: str, res: dict[str, Any]) -> None:
    """``claude -p`` against an Anthropic-Messages base URL, in an isolated config dir."""
    if not a.anthropic_base_url:
        raise ArmError("claude-code needs --anthropic-base-url (a /v1/messages translation URL)")
    home = Path(os.environ["HOME"])
    cfg = home / "claude-config"
    cfg.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CLAUDE_CONFIG_DIR": str(cfg),  # never the user's own Claude config
        "ANTHROPIC_BASE_URL": a.anthropic_base_url,
        "ANTHROPIC_AUTH_TOKEN": "dummy",
        "ANTHROPIC_MODEL": a.model,
        "ANTHROPIC_SMALL_FAST_MODEL": a.model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(a.max_tokens, 32_000)),
    }
    cmd = [
        *shlex.split(a.bin or "claude"),
        "-p",
        spec,
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        a.model,
    ]
    log = Path(a.log)
    rc, timed_out = run_group(cmd, env, a.timeout, log)
    text = log.read_text("utf-8", "replace") if log.is_file() else ""
    with contextlib.suppress(ValueError):
        doc = json.loads(text[text.index("{") :])
        usage = doc.get("usage") or {}
        res.update(
            prompt_tokens=int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            calls=int(doc.get("num_turns") or 0),
        )
    res["harness_returncode"] = rc
    if timed_out:
        res["failure_class"] = "timeout"
    elif CONTEXT_LENGTH_RE.search(text) and not changed_owned(repo, list(a.owned)):
        res["failure_class"] = "context_length_exceeded"
    elif rc != 0:
        res["failure_class"] = "harness_crash"


RUNNERS = {
    "single-shot": run_single_shot,
    "mini-swe-agent": run_mini,
    "aider": run_aider,
    "claude-code": run_claude_code,
}


# ── Entry point ─────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("--spec-file", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--owned", action="append", default=[])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--context-window", type=int, default=262144)
    ap.add_argument("--timeout", type=float, default=1500.0, help="wall seconds (25 min)")
    ap.add_argument("--bin", default="", help="the harness executable (shell-split)")
    ap.add_argument("--edit-format", default="diff", help="aider edit format (whole | diff)")
    ap.add_argument("--anthropic-base-url", default="")
    ap.add_argument("--log", default="arm.log", help="harness output (inside the episode HOME)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    repo = Path.cwd()
    spec = a.spec_file.read_text("utf-8")
    t0 = time.time()
    res: dict[str, Any] = {"failure_class": ""}
    try:
        if a.log == "arm.log":
            a.log = str(Path(os.environ.get("HOME", ".")) / "arm.log")
        RUNNERS[a.arm](a, repo, spec, res)
        crashed = False
    except (ArmError, OSError, urllib.error.URLError, TimeoutError) as exc:
        res["failure_class"] = "harness_crash"
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
        crashed = True
    edited = changed_owned(repo, list(a.owned))
    fc = res.get("failure_class") or ""
    if not edited and not fc:
        fc = "no_edits"
    doc: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "workflow_id": None,
        "arm": a.arm,
        "success": edited and not crashed and fc == "",
        "no_edits": not edited and fc in ("no_edits", "format_error", "truncated"),
        "context_length_exceeded": fc == "context_length_exceeded",
        "truncated_responses": int(res.get("truncated_responses", 0)),
        "reasoning_tokens": 0,
        "failure_class": fc,
        "usage": {k: res.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "calls")},
        "usage_note": res.get("usage_note", ""),
        "wall_s": round(time.time() - t0, 2),
        "error": res.get("error", ""),
        "harness_returncode": res.get("harness_returncode"),
        "sub_tasks": {
            "T1": {
                "agent": a.arm,
                "status": "completed" if edited else "failed",
                "success": edited,
                "iterations": res.get("calls", 0),
                "ops": [],
                "truncated": bool(res.get("truncated_responses")),
            }
        },
        "settings": {
            "temperature": a.temperature,
            "top_p": a.top_p,
            "repetition_penalty": a.repetition_penalty,
            "max_tokens": a.max_tokens,
            "timeout_s": a.timeout,
            "edit_format": a.edit_format if a.arm == "aider" else None,
            "git": git_version(),
        },
    }
    result_path = os.environ.get(RESULT_ENV)
    write_result(Path(result_path) if result_path else None, doc)
    print(f"arm={a.arm} edited={edited} failure_class={fc or '-'} wall={doc['wall_s']}s")
    return 1 if crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
