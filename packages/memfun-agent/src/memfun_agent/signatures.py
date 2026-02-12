"""DSPy signatures for RLM coding agent tasks.

Each signature defines the input/output contract for a specific coding task.
These are used both by the RLM module (for complex tasks) and by direct
dspy.Predict / dspy.ChainOfThought calls (for simpler tasks).
"""
from __future__ import annotations

import dspy

# ── Code Analysis ──────────────────────────────────────────────


class CodeAnalysis(dspy.Signature):
    """Analyze code structure, quality, and patterns.

    Given source code and a specific analysis query, produce a structured
    analysis covering architecture, patterns, potential issues, and
    improvement suggestions.
    """

    code: str = dspy.InputField(
        desc="Source code to analyze (may be a single file or concatenated files)"
    )
    query: str = dspy.InputField(
        desc="Specific analysis question or focus area"
    )
    analysis: str = dspy.OutputField(
        desc="Structured analysis of the code addressing the query"
    )
    issues: list[str] = dspy.OutputField(
        desc="List of identified issues or concerns"
    )
    suggestions: list[str] = dspy.OutputField(
        desc="List of actionable improvement suggestions"
    )


# ── Bug Fix ────────────────────────────────────────────────────


class BugFix(dspy.Signature):
    """Fix a bug described in the context of the given code.

    Given source code and a bug description, identify the root cause,
    produce corrected code, and explain the fix.
    """

    code: str = dspy.InputField(
        desc="Source code containing the bug"
    )
    bug_description: str = dspy.InputField(
        desc="Description of the bug, including symptoms and reproduction steps"
    )
    root_cause: str = dspy.OutputField(
        desc="Explanation of the root cause of the bug"
    )
    fixed_code: str = dspy.OutputField(
        desc="Corrected source code with the bug fixed"
    )
    explanation: str = dspy.OutputField(
        desc="Explanation of what was changed and why"
    )


# ── Code Review ────────────────────────────────────────────────


class CodeReview(dspy.Signature):
    """Review code and provide structured feedback.

    Given source code (typically a diff or complete file), produce a
    code review with categorized findings and an overall assessment.
    """

    code: str = dspy.InputField(
        desc="Source code or diff to review"
    )
    context: str = dspy.InputField(
        desc=(
            "Additional context: PR description, related files, "
            "coding standards"
        )
    )
    summary: str = dspy.OutputField(
        desc="Brief overall assessment of the code quality"
    )
    findings: list[str] = dspy.OutputField(
        desc=(
            "List of findings, each prefixed with severity "
            "[critical/major/minor/nit]"
        )
    )
    approved: bool = dspy.OutputField(
        desc="Whether the code is approved for merge"
    )


# ── Code Explanation ───────────────────────────────────────────


class CodeExplanation(dspy.Signature):
    """Explain code functionality in clear, accessible language.

    Given source code and a target audience level, produce an
    explanation covering purpose, logic flow, and key concepts.
    """

    code: str = dspy.InputField(
        desc="Source code to explain"
    )
    audience: str = dspy.InputField(
        desc=(
            "Target audience level: beginner, intermediate, "
            "or expert"
        ),
    )
    explanation: str = dspy.OutputField(
        desc="Clear explanation of what the code does and how"
    )
    key_concepts: list[str] = dspy.OutputField(
        desc="Key concepts or patterns used in the code"
    )


# ── RLM Exploration (used internally by the RLM loop) ──────────────


class RLMExploration(dspy.Signature):
    """Generate Python code to accomplish a task in a REPL.

    You are a fully autonomous coding agent. You must do EVERYTHING
    yourself — write files, install dependencies, run commands,
    configure, test, deploy. Never tell the user to do something
    you can do. The user expects a finished, working result.

    Available tools:
    - write_file(path, content): Create/overwrite a file
    - read_file(path): Read file contents
    - run_cmd(cmd): Run a shell command, returns stdout
    - edit_file(path, old_text, new_text): Replace text in a file
    - list_files(path='.'): List files recursively
    - llm_query(question, context): Ask sub-LM a question
    - web_search(query, max_results=5): Search the web (DuckDuckGo)
    - web_fetch(url, max_length=50000): Fetch URL content as markdown

    Quality standards:
    - Write production-quality code, not placeholders or TODOs
    - Use proper structure, error handling, and clean style
    - Frontend: use modern CSS, responsive design, good UX
    - Backend: proper error handling, security, clean APIs
    - Always install deps yourself: run_cmd('pip install ...')
    - Always verify your work: check files, run the app, test
    - If deployment is requested, do it (ngrok, etc.)

    Rules:
    - Use write_file() not open(). Use run_cmd() not subprocess.
    - Set state['FINAL'] = answer when ALL work is complete.
    - For questions/analysis: state['FINAL'] must contain the FULL
      detailed answer with specific findings, data, and evidence
      from the files you read. NEVER set FINAL to just a summary
      like "Analyzed files" — include the actual information found.
    - For coding tasks: state['FINAL'] must list all files created
      and actions taken.
    - Do not stop early. Finish every step of the task.
    """

    context_metadata: str = dspy.InputField(
        desc=(
            "Metadata about the context: type, length, preview, "
            "and available helper functions"
        )
    )
    query: str = dspy.InputField(
        desc="The user's question or task to accomplish"
    )
    code_history: str = dspy.InputField(
        desc=(
            "History of previously executed code and their "
            "outputs (truncated)"
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Think step by step: What needs to be done next? "
            "Is there a higher-quality way to do it? Should I "
            "handle this myself or is user action truly needed?"
        )
    )
    next_code: str = dspy.OutputField(
        desc=(
            "Python code to execute. Use write_file(), "
            "run_cmd(), edit_file(). Always verify work. "
            "Set state['FINAL'] = summary when done."
        )
    )


# ── Query Classification ───────────────────────────────────────


class QueryTriage(dspy.Signature):
    """Classify a user query to determine the best handling strategy.

    The LLM decides whether the query needs project context, web search,
    or can be answered directly from general knowledge.

    IMPORTANT: Short follow-up messages like "2", "yes", "the second one",
    "do it", "go ahead" are CONTINUATIONS of the previous conversation.
    They should be classified as 'project' or 'task' (matching the previous
    turn's category), NEVER as 'direct'. Check recent_conversation to
    understand what the user is referring to.

    Use 'web' when the query asks about current information, external
    APIs, documentation, prices, news, weather, or anything that
    requires up-to-date internet data. Also use 'web' when the user
    explicitly asks to search, fetch, or look something up online.
    """

    query: str = dspy.InputField(
        desc="The user's question or request"
    )
    recent_conversation: str = dspy.InputField(
        desc=(
            "Recent conversation history (last 1-2 turns). "
            "CRITICAL: If the query is short (e.g. '2', 'yes', "
            "'option 1'), this context tells you what the user "
            "is referring to. A short query following a detailed "
            "agent response with numbered options is a follow-up, "
            "NOT a general knowledge question."
        )
    )
    project_summary: str = dspy.InputField(
        desc="Brief summary of available project context"
    )
    category: str = dspy.OutputField(
        desc=(
            "One of: 'direct' (general knowledge, simple questions "
            "with NO prior conversation context), "
            "'web' (needs web search or URL fetching for current info), "
            "'project' (needs codebase exploration, or is a follow-up "
            "to a previous project discussion), "
            "'task' (multi-step coding task needing planning, or is a "
            "follow-up to a previous task)"
        )
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for the classification"
    )


# ── Task Planning ──────────────────────────────────────────────


class TaskPlanning(dspy.Signature):
    """Create an execution plan for a fully autonomous coding agent.

    The agent does EVERYTHING itself — writes code, installs
    dependencies, runs commands, deploys, and verifies. Plans
    must include self-contained steps that never ask the user
    to do anything. The last step must always be verification
    (check files, run the app, test that it works).

    Quality: every step should produce production-ready output,
    not placeholders. Include dependency installation, proper
    configuration, and deployment if requested.
    """

    query: str = dspy.InputField(
        desc="The user's request or question"
    )
    context_metadata: str = dspy.InputField(
        desc="Project context summary and available tools"
    )
    steps: list[str] = dspy.OutputField(
        desc=(
            "Ordered list of 3-7 concrete action steps. "
            "Include dependency installation and verification. "
            "Last step must verify the result works."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Approach reasoning: what's the highest-quality "
            "way to accomplish this? What should the agent "
            "handle vs. what truly needs the user?"
        )
    )


# ── Learning Extraction ──────────────────────────────────────


class LearningExtraction(dspy.Signature):
    """Extract persistent learnings from a conversation turn.

    Given a user message and the agent's response, identify 0-3
    actionable learnings that should be remembered for future
    conversations. Focus on:

    - User preferences (port numbers, coding style, frameworks,
      naming conventions, UI choices)
    - Project-specific patterns (directory structure, build system,
      deploy targets, test strategy)
    - Technical details (API endpoints, credentials patterns,
      specific library versions, config formats)
    - Workflow preferences (test first, specific tools, review
      process, deployment flow)

    Only extract genuinely reusable knowledge. Do NOT extract:
    - One-time facts about the current task
    - Generic programming knowledge everyone knows
    - Information already obvious from project config files
    - Temporary state or in-progress details
    """

    user_message: str = dspy.InputField(
        desc="The user's message or request in this turn"
    )
    agent_response: str = dspy.InputField(
        desc=(
            "The agent's response (may be truncated). "
            "Look for implicit preferences revealed by "
            "user corrections or explicit instructions."
        )
    )
    existing_learnings: str = dspy.InputField(
        desc=(
            "Previously stored learnings (may be empty). "
            "Check for conflicts — if the user's new "
            "message contradicts an existing learning, "
            "the old one should be listed in 'updates'."
        )
    )
    learnings: list[str] = dspy.OutputField(
        desc=(
            "List of 0-3 learnings to persist. Each must be "
            "a concise, self-contained statement prefixed "
            "with its category: "
            "'[preference] User prefers port 8080' or "
            "'[technical] Project uses PostgreSQL 16' or "
            "'[pattern] Components go in src/components/' or "
            "'[workflow] Always run tests before committing'. "
            "Return empty list if nothing worth learning."
        )
    )
    updates: list[str] = dspy.OutputField(
        desc=(
            "List of existing learnings that are now "
            "outdated and should be REPLACED. Copy the "
            "exact content text of the old learning. "
            "Empty list if no conflicts found."
        )
    )
