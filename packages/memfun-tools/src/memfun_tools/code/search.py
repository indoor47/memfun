from __future__ import annotations

import asyncio
import shutil

from fastmcp import FastMCP

search_server = FastMCP("memfun-search-tools")


async def _run_command(cmd: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    """Run a command and return (stdout, stderr, returncode)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        proc.returncode or 0,
    )


@search_server.tool()
async def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_insensitive: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
) -> str:
    """Search file contents using ripgrep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory or file to search.
        glob: Glob pattern to filter files (e.g., '*.py').
        case_insensitive: Enable case-insensitive matching.
        context_lines: Lines of context before and after match.
        max_results: Maximum number of matches to return.
    """
    rg = shutil.which("rg")
    if not rg:
        raise RuntimeError(
            "ripgrep (rg) not found. Install it:"
            " https://github.com/BurntSushi/ripgrep"
        )

    # Clamp user-controllable numeric parameters
    max_results = max(1, min(max_results, 10_000))
    context_lines = max(0, min(context_lines, 20))

    cmd = [
        rg, "--no-heading", "--line-number",
        "--color=never", f"--max-count={max_results}",
    ]
    if case_insensitive:
        cmd.append("--ignore-case")
    if context_lines > 0:
        cmd.append(f"--context={context_lines}")
    if glob:
        cmd.extend(["--glob", glob])
    # Use -- separator to prevent pattern/path being
    # interpreted as flags
    cmd.extend(["--", pattern, path])

    stdout, stderr, rc = await _run_command(cmd)
    if rc == 1:
        return "(no matches)"
    if rc != 0:
        raise RuntimeError(f"ripgrep error: {stderr}")
    return stdout[:50000]  # Truncate very long output


@search_server.tool()
async def ast_grep(
    pattern: str,
    path: str = ".",
    language: str | None = None,
) -> str:
    """Search code using ast-grep for structural patterns.

    Args:
        pattern: AST pattern to search for (e.g., 'def $NAME($$$ARGS)').
        path: Directory to search.
        language: Language filter (python, javascript, typescript, etc.).
    """
    sg = shutil.which("sg") or shutil.which("ast-grep")
    if not sg:
        raise RuntimeError(
            "ast-grep (sg) not found. Install it:"
            " https://ast-grep.github.io"
        )

    # Validate language to prevent flag injection
    allowed_langs = frozenset({
        "python", "javascript", "typescript", "tsx",
        "jsx", "go", "rust", "java", "c", "cpp",
        "csharp", "ruby", "swift", "kotlin", "scala",
        "html", "css", "json", "yaml", "toml",
    })

    cmd = [
        sg, "run", "--pattern", pattern, path, "--json",
    ]
    if language:
        lang = language.lower().strip()
        if lang not in allowed_langs:
            raise ValueError(
                f"Unsupported language: {language!r}"
            )
        cmd.extend(["--lang", lang])

    stdout, stderr, rc = await _run_command(cmd)
    if rc != 0:
        return f"(no matches or error: {stderr})"
    return stdout[:50000]
