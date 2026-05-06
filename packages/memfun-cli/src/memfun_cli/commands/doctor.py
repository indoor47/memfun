"""``memfun doctor`` — credential and LLM-endpoint health check.

This command performs a fast, low-cost ping against the configured LLM
endpoint to verify that:

1. The TOML config loads successfully.
2. ``~/.memfun/credentials.json`` (if present) is readable and well-formed.
3. The configured API key actually authenticates against the provider.
4. The endpoint is reachable within the timeout budget.

Each check is independent and short-circuits on the first hard failure
that prevents the next from running.  The LLM ping is a single 4-token
``"ping"`` completion — cost is well under $0.0001 per check on Anthropic
or OpenAI pricing.

Doctor is intentionally light on dependencies so it can be the first
diagnostic users run when something fails.  In particular, it does not
import ``dspy`` — only ``litellm`` directly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import typer
from memfun_core.config import LLMConfig, MemfunConfig
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a key safe to print to a terminal.

    Shows only the last four characters and the total length, never the
    leading bytes (which often encode a service identifier such as
    ``sk-ant-`` and would help an attacker confirm that a leaked tail
    came from a particular provider).
    """
    if not key:
        return "(empty)"
    if len(key) <= 4:
        return "*" * len(key)
    return f"...{key[-4:]} ({len(key)} chars)"


def _resolve_lite_llm_model(provider: str, model: str) -> tuple[str, str | None]:
    """Map a memfun ``(provider, model)`` to a LiteLLM model string.

    Returns ``(lite_llm_model, fallback_api_key)``.  ``fallback_api_key`` is
    set when the provider does not require a real key (e.g. Ollama) but
    LiteLLM still demands a non-empty value.

    Mirrors the resolution in ``chat.py:_configure_dspy`` so doctor pings
    the same endpoint the chat will later use.
    """
    if provider == "anthropic":
        return f"anthropic/{model}", None
    if provider == "openai":
        return f"openai/{model}", None
    if provider == "openai-compat":
        # OpenAI-compatible local server (llama.cpp / vLLM / OpenRouter etc.).
        # LiteLLM treats it as openai with a custom api_base; we need the prefix.
        return f"openai/{model}", "sk-local"
    if provider == "ollama":
        return f"ollama_chat/{model}", "ollama"
    # Custom: assume the model is already a fully-qualified LiteLLM identifier.
    return model, None


def _resolve_api_key(cfg: LLMConfig) -> str:
    """Look up the API key from the environment.

    The wizard exports ``credentials.json`` entries into ``os.environ`` at
    chat startup (see ``chat.py``); doctor skips that step but reads the
    file directly so a user who hasn't yet started chat can still
    diagnose.
    """
    env_key = os.environ.get(cfg.api_key_env, "").strip()
    if env_key:
        return env_key

    # Fall back to credentials.json — but only as a *display* convenience.
    # We do NOT export it into os.environ here; doctor's job is to report
    # what the next ``memfun ask`` will see.
    creds_path = Path.home() / ".memfun" / "credentials.json"
    if creds_path.exists():
        try:
            raw_data: object = json.loads(creds_path.read_text())
        except (OSError, json.JSONDecodeError):
            return ""
        if isinstance(raw_data, dict):
            value = raw_data.get(cfg.api_key_env)  # type: ignore[arg-type]
            if isinstance(value, str):
                return value.strip()
    return ""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_config() -> MemfunConfig | None:
    """Load and print the resolved config, or report failure."""
    try:
        cfg = MemfunConfig.load()
    except Exception as exc:
        console.print(
            f"  [red]✗[/red] Failed to load config: [red]{type(exc).__name__}: {exc}[/red]"
        )
        return None

    llm = cfg.llm
    if not llm.provider:
        console.print(
            "  [red]✗[/red] No LLM provider set in config (run [bold]memfun init[/bold])"
        )
        return None

    console.print(f"  Provider:    [cyan]{llm.provider}[/cyan]")
    console.print(f"  Model:       [cyan]{llm.model}[/cyan]")
    console.print(f"  Base URL:    [cyan]{llm.base_url or '(provider default)'}[/cyan]")
    console.print(f"  API key env: [cyan]{llm.api_key_env}[/cyan]")
    return cfg


def _check_credentials(cfg: MemfunConfig, *, verbose: bool) -> str:
    """Print credential discovery + return the resolved API key (may be empty)."""
    creds_path = Path.home() / ".memfun" / "credentials.json"
    env_value = os.environ.get(cfg.llm.api_key_env, "").strip()

    if creds_path.exists():
        try:
            raw = creds_path.read_text()
            data: object = json.loads(raw)
            if isinstance(data, dict) and cfg.llm.api_key_env in data:
                key_in_file: object = data[cfg.llm.api_key_env]  # type: ignore[index]
                if isinstance(key_in_file, str) and key_in_file.strip():
                    console.print(
                        f"  Credentials: [green]✓[/green] {creds_path} "
                        f"({_mask_key(key_in_file.strip())})"
                    )
                else:
                    console.print(
                        f"  Credentials: [yellow]![/yellow] {creds_path} "
                        f"missing entry for {cfg.llm.api_key_env}"
                    )
            else:
                console.print(
                    f"  Credentials: [yellow]![/yellow] {creds_path} "
                    f"has no key {cfg.llm.api_key_env}"
                )
        except (OSError, json.JSONDecodeError) as exc:
            console.print(
                f"  Credentials: [red]✗[/red] {creds_path} unreadable: {type(exc).__name__}"
            )
            if verbose:
                console.print(f"    [dim]{exc}[/dim]")
    else:
        console.print(f"  Credentials: [dim]{creds_path} not present[/dim]")

    if env_value:
        console.print(
            f"  API key:     [green]✓[/green] {cfg.llm.api_key_env} "
            f"set in env ({_mask_key(env_value)})"
        )
    else:
        console.print(f"  API key:     [yellow]![/yellow] {cfg.llm.api_key_env} not in env")

    return _resolve_api_key(cfg.llm)


def _ping_llm(
    cfg: MemfunConfig,
    api_key: str,
    *,
    timeout: float,
    verbose: bool,
) -> bool:
    """Send a 4-token completion and report the result."""
    provider = cfg.llm.provider

    if not api_key and provider != "ollama":
        console.print(
            f"  [red]✗[/red] No API key available — set "
            f"[bold]{cfg.llm.api_key_env}[/bold] or rerun "
            f"[bold]memfun init[/bold]"
        )
        return False

    lm_model, fallback_key = _resolve_lite_llm_model(provider, cfg.llm.model)
    effective_key = api_key or fallback_key or ""

    kwargs: dict[str, Any] = {
        "model": lm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": 0,
        "timeout": timeout,
    }
    if effective_key:
        kwargs["api_key"] = effective_key
    if cfg.llm.base_url:
        kwargs["api_base"] = cfg.llm.base_url

    # Import inside the function so doctor stays cheap when only earlier
    # checks were needed.
    try:
        import litellm
        from litellm.exceptions import (
            APIConnectionError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            Timeout,
        )
    except ImportError as exc:
        console.print(f"  [red]✗[/red] litellm not installed ({exc}); reinstall memfun")
        return False

    started = time.monotonic()
    try:
        response = litellm.completion(**kwargs)  # type: ignore[no-untyped-call]
    except AuthenticationError as exc:
        console.print(
            f"  [red]✗[/red] Authentication failed (key may be revoked "
            f"or rotated). Check [bold]{cfg.llm.api_key_env}[/bold] or "
            f"[bold]~/.memfun/credentials.json[/bold]."
        )
        if verbose:
            console.print(f"    [dim]{type(exc).__name__}: {exc}[/dim]")
        return False
    except Timeout as exc:
        elapsed = time.monotonic() - started
        console.print(
            f"  [red]✗[/red] Request timed out after {elapsed:.1f}s. Endpoint may be slow or down."
        )
        if verbose:
            console.print(f"    [dim]{type(exc).__name__}: {exc}[/dim]")
        return False
    except APIConnectionError as exc:
        target = cfg.llm.base_url or f"{provider} default endpoint"
        console.print(
            f"  [red]✗[/red] Endpoint unreachable: [bold]{target}[/bold]. Is the server running?"
        )
        if verbose:
            console.print(f"    [dim]{type(exc).__name__}: {exc}[/dim]")
        return False
    except (NotFoundError, BadRequestError) as exc:
        console.print(f"  [red]✗[/red] {type(exc).__name__}: {exc}")
        console.print(
            f"    [dim]Model [bold]{cfg.llm.model}[/bold] may not exist "
            f"on {provider}. Check the spelling.[/dim]"
        )
        return False
    except TimeoutError as exc:
        # Stdlib TimeoutError surfaces from asyncio paths; treat the same.
        elapsed = time.monotonic() - started
        console.print(f"  [red]✗[/red] Request timed out after {elapsed:.1f}s (asyncio): {exc}")
        return False
    except Exception as exc:
        console.print(f"  [red]✗[/red] {type(exc).__name__}: {exc}")
        return False

    elapsed_ms = (time.monotonic() - started) * 1000.0

    # LiteLLM responses follow the OpenAI shape; usage is optional.
    usage_str = ""
    usage: object = getattr(response, "usage", None)
    if usage is not None:
        completion_tokens: object = getattr(usage, "completion_tokens", None)
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")  # type: ignore[arg-type]
        if completion_tokens is not None:
            usage_str = f", {completion_tokens} response tokens"

    console.print(
        f"  [green]✓[/green] LLM responded in [bold]{elapsed_ms:.0f}ms[/bold]{usage_str}"
    )
    return True


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def doctor_command(
    timeout: float = typer.Option(
        5.0,
        "--timeout",
        help="Per-check timeout in seconds.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print full request/response details on failure.",
    ),
) -> None:
    """Health-check the memfun installation.

    Verifies config, credentials, and that the configured LLM endpoint
    actually accepts a 4-token authenticated request.  Exits non-zero on
    any failure so it can be wired into setup scripts and CI smoke tests.
    """
    console.print()
    console.print("[bold]memfun doctor[/bold]")
    console.print()

    console.print("[bold]Config[/bold]")
    cfg = _check_config()
    if cfg is None:
        console.print()
        console.print("[red]✗ doctor failed: config did not load[/red]")
        raise typer.Exit(code=1)

    console.print()
    console.print("[bold]Credentials[/bold]")
    api_key = _check_credentials(cfg, verbose=verbose)

    console.print()
    console.print(f"[bold]LLM ping[/bold] [dim](timeout={timeout:.1f}s)[/dim]")
    ok = _ping_llm(cfg, api_key, timeout=timeout, verbose=verbose)

    console.print()
    if ok:
        console.print("[green]✓ doctor: all checks passed[/green]")
        raise typer.Exit(code=0)
    console.print("[red]✗ doctor: at least one check failed[/red]")
    raise typer.Exit(code=1)
