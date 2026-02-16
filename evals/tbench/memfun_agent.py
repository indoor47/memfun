"""Memfun agent for Terminal-Bench evaluation.

A BaseAgent subclass that uses Memfun's ContextFirst-inspired approach:
explore the environment, plan, execute step-by-step, and verify.

Usage:
    tb run --agent-import-path memfun_agent:MemfunAgent \
           --model anthropic/claude-opus-4-6 \
           --dataset terminal-bench-core==0.1.1 \
           --output-path /root/tbench-runs \
           --run-id memfun-v1
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import litellm
from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode

if TYPE_CHECKING:
    from terminal_bench.terminal.tmux_session import TmuxSession

logger = logging.getLogger("memfun.tbench")

# ── Configuration ─────────────────────────────────────────────────

MAX_ITERATIONS = 100  # generous budget per task
OUTPUT_TRUNCATE_CHARS = 8000  # truncate long command outputs
SUMMARY_THRESHOLD = 60_000  # summarize conversation at this char count
DONE_MARKER = "<<MEMFUN_DONE>>"
SUMMARY_MARKER = "<<MEMFUN_SUMMARY>>"

SYSTEM_PROMPT = """\
You are Memfun, an autonomous terminal agent. You solve tasks by running \
shell commands in a Linux terminal.

## Approach: Context-First

1. **Explore**: Before acting, understand the environment. Check the OS, \
installed tools, directory contents, running services. Read any relevant files.
2. **Plan**: Think step-by-step about the solution. Identify what needs to \
happen and in what order. Consider edge cases.
3. **Execute**: Run commands one at a time. After each command, observe the \
output before deciding the next action.
4. **Verify**: After implementing, verify the solution works. Run tests if \
available. Check that expected files/services/state exist.

## Tools

You have one tool: `bash`. Use it to execute a shell command in the terminal. \
The command output will be returned to you.

When you call the bash tool, provide the command to execute. The output will \
be captured and returned.

## Important Rules

- Run ONE command at a time. Wait for output before the next command.
- If a command produces no output for a while, it may be waiting for input \
or running a long process. Consider using timeout, background (&), or Ctrl-C.
- For long-running processes (servers, builds), start them in the background.
- If you encounter an error, read it carefully and adapt your approach.
- NEVER give up. If one approach fails, try another.
- When you are CERTAIN the task is fully completed and verified, respond \
with ONLY the text: """ + DONE_MARKER + """
- Do NOT say """ + DONE_MARKER + """ until you have actually verified the solution.

## Output Handling

- If a command produces very long output, focus on the relevant parts.
- Use grep, head, tail, or awk to filter output when useful.
- Avoid commands that produce unbounded output (e.g., `find /` without limits).
"""

# ── Helpers ───────────────────────────────────────────────────────


def _load_credentials() -> None:
    """Load API keys from ~/.memfun/credentials.json into environment."""
    for creds_path in [
        Path.home() / ".memfun" / "credentials.json",
        Path.cwd() / ".memfun" / "credentials.json",
    ]:
        if not creds_path.exists():
            continue
        try:
            creds = json.loads(creds_path.read_text())
            if isinstance(creds, dict):
                for key, value in creds.items():
                    if isinstance(key, str) and isinstance(value, str) and value:
                        os.environ.setdefault(key, value)
        except Exception:
            pass


def _truncate(text: str, max_chars: int = OUTPUT_TRUNCATE_CHARS) -> str:
    """Truncate long output, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n"
        + text[-half:]
    )


def _conversation_size(messages: list[dict]) -> int:
    """Approximate char count of the conversation."""
    return sum(
        len(str(m.get("content", "")))
        for m in messages
    )


# ── Tool definitions ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command in the terminal. "
                "Returns the command output (stdout + stderr)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ── Agent ─────────────────────────────────────────────────────────


class MemfunAgent(BaseAgent):
    """Memfun autonomous agent for Terminal-Bench.

    Uses a ContextFirst-inspired approach: explore → plan → execute → verify.
    Powered by litellm for LLM calls and tmux for terminal interaction.
    """

    @staticmethod
    def name() -> str:
        return "memfun"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_iterations: int = MAX_ITERATIONS,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        _load_credentials()
        self._model_name = model_name or os.environ.get(
            "MEMFUN_MODEL",
            "anthropic/claude-sonnet-4-5-20250929",
        )
        self._max_iterations = max_iterations
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ── Core loop ─────────────────────────────────────────────

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """Run the agent on a Terminal-Bench task."""
        instruction = self._render_instruction(instruction)
        markers: list[tuple[float, str]] = []
        failure_mode = FailureMode.NONE
        t0 = time.time()

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task:\n{instruction}"},
        ]

        done = False
        iteration = 0

        while not done and iteration < self._max_iterations:
            iteration += 1
            logger.info("Iteration %d / %d", iteration, self._max_iterations)

            # Summarize if conversation is getting long
            if _conversation_size(messages) > SUMMARY_THRESHOLD:
                messages = self._summarize_conversation(messages)

            # Call the LLM
            try:
                response = litellm.completion(
                    model=self._model_name,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=16384,
                )
            except litellm.exceptions.RateLimitError:
                logger.warning("Rate limited, sleeping 30s")
                time.sleep(30)
                continue
            except litellm.exceptions.ContextWindowExceededError:
                logger.warning("Context window exceeded, summarizing")
                messages = self._summarize_conversation(messages, force=True)
                continue
            except Exception as e:
                logger.error("LLM error: %s", e)
                failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                break

            # Track tokens
            usage = getattr(response, "usage", None)
            if usage:
                self._total_input_tokens += getattr(
                    usage, "prompt_tokens", 0
                )
                self._total_output_tokens += getattr(
                    usage, "completion_tokens", 0
                )

            # Process the response
            choice = response.choices[0]
            message = choice.message

            # Check for finish conditions
            if choice.finish_reason == "stop" and not message.tool_calls:
                text = message.content or ""
                messages.append({"role": "assistant", "content": text})

                if DONE_MARKER in text:
                    done = True
                    markers.append(
                        (time.time() - t0, "agent_declared_done")
                    )
                    logger.info("Agent declared done at iteration %d", iteration)
                else:
                    # Agent responded with text but no tool call and no DONE
                    # Nudge it to keep working or declare done
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue working on the task. "
                            "If you are done, respond with: "
                            + DONE_MARKER
                        ),
                    })
                continue

            # Handle tool calls
            if message.tool_calls:
                # Add assistant message with tool calls
                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    fn = tool_call.function
                    if fn.name != "bash":
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {fn.name}",
                        })
                        continue

                    # Parse command
                    try:
                        args = json.loads(fn.arguments)
                        command = args["command"]
                    except (json.JSONDecodeError, KeyError):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Error: invalid arguments",
                        })
                        continue

                    # Execute via tmux
                    logger.info("Executing: %s", command[:200])
                    output = self._execute_command(session, command)
                    output = _truncate(output)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output if output else "(no output)",
                    })

                    # Also check text content for DONE marker
                    if message.content and DONE_MARKER in message.content:
                        done = True
                        markers.append(
                            (time.time() - t0, "agent_declared_done")
                        )
                        break

        if iteration >= self._max_iterations and not done:
            failure_mode = FailureMode.AGENT_TIMEOUT
            logger.warning("Agent hit max iterations (%d)", self._max_iterations)

        return AgentResult(
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            failure_mode=failure_mode,
            timestamped_markers=markers,
        )

    # ── Command execution ─────────────────────────────────────

    def _execute_command(
        self,
        session: TmuxSession,
        command: str,
    ) -> str:
        """Send a command to the tmux session and capture output.

        Uses non-blocking send to avoid issues with heredocs and
        long-running commands that break the tmux blocking mechanism.
        """
        # Reset incremental tracking to get clean diff
        session.get_incremental_output()

        # Send command non-blocking (block=True breaks heredocs because
        # "; tmux wait -S done" is appended after EOF, breaking the delimiter)
        session.send_keys(
            [command, "Enter"],
            block=False,
            min_timeout_sec=8.0,  # wait at least 8s for output
        )

        # Get just the new output since the command was sent
        output = session.get_incremental_output()
        return output.strip()

    # ── Conversation management ───────────────────────────────

    def _summarize_conversation(
        self,
        messages: list[dict],
        *,
        force: bool = False,
    ) -> list[dict]:
        """Compress the conversation to stay within context limits.

        Keeps the system prompt and last few exchanges, summarizes the rest.
        """
        if len(messages) <= 6 and not force:
            return messages

        # Keep system + first user message + last N messages
        system = messages[0]
        first_user = messages[1]
        # Keep last 8 messages (4 exchanges)
        keep_recent = 8
        to_summarize = messages[2:-keep_recent] if len(messages) > keep_recent + 2 else []
        recent = messages[-keep_recent:] if len(messages) > keep_recent + 2 else messages[2:]

        if not to_summarize:
            return messages

        # Build summary text from the middle messages
        summary_parts = []
        for msg in to_summarize:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                summary_parts.append(f"[{role}] {content[:500]}")

        summary_text = "\n".join(summary_parts)

        try:
            summary_response = litellm.completion(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following agent conversation "
                            "history concisely. Focus on: what was attempted, "
                            "what succeeded, what failed, current state. "
                            "Be brief but complete."
                        ),
                    },
                    {"role": "user", "content": summary_text[:30000]},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            usage = getattr(summary_response, "usage", None)
            if usage:
                self._total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self._total_output_tokens += getattr(usage, "completion_tokens", 0)

            summary = summary_response.choices[0].message.content
        except Exception:
            # Fallback: just describe what was trimmed
            summary = f"[{len(to_summarize)} earlier messages trimmed]"

        compressed = [
            system,
            first_user,
            {
                "role": "user",
                "content": (
                    f"Summary of previous work:\n{summary}\n\n"
                    "Continue from where you left off."
                ),
            },
            *recent,
        ]

        logger.info(
            "Summarized conversation: %d → %d messages",
            len(messages),
            len(compressed),
        )
        return compressed
