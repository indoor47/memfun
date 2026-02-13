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
    - BEFORE setting FINAL for any question: you MUST have called
      read_file() on the main source files (not just list_files).
      If conversation_history exists, search it with multiple
      keywords. Cross-check code vs history — they may differ.
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
            "Pure Python code to execute — NO markdown fences, "
            "NO ```python blocks, NO explanatory text. "
            "Just raw Python. Use write_file(), "
            "run_cmd(), edit_file(). Always verify work. "
            "Set state['FINAL'] = detailed_answer when done."
        )
    )


# ── Query Classification ───────────────────────────────────────


class QueryTriage(dspy.Signature):
    """Classify a user query to determine the best handling strategy.

    Classification guide:
    - 'direct': Pure knowledge questions with no project context.
      "What is a mutex?", "Explain async/await".
    - 'web': Needs current internet data. "What's the latest React version?"
    - 'project': READ-ONLY codebase exploration. "How does auth work here?",
      "Explain the database schema", "What files handle routing?"
    - 'task': ANY request that creates, modifies, or generates code.
      "Add tests", "Fix the login bug", "Build a REST API", "Refactor models".
      When in doubt between 'project' and 'task', prefer 'task' — the
      system will fall back to single-agent if the task is simple.

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
            "One of: 'direct' (general knowledge, no project needed), "
            "'web' (needs internet search/fetch), "
            "'project' (READ-ONLY: explore, analyze, explain existing "
            "code without changing it), "
            "'task' (WRITE/CHANGE: create files, fix bugs, refactor, "
            "add features, implement, build, test, deploy — any request "
            "that will modify or create code. When in doubt between "
            "'project' and 'task', prefer 'task'.)"
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


# ── Query Resolution ──────────────────────────────────────


class QueryResolution(dspy.Signature):
    """Resolve deictic references and expand short queries into full descriptions.

    Given a potentially ambiguous user query and recent conversation history,
    produce an unambiguous, fully-specified task description.  If the query
    is already clear and self-contained, return it unchanged.

    Examples of resolution:
    - "fix this" + history about a bug  -> "Fix the TypeError in auth.py..."
    - "2" + history with numbered options -> "Implement option 2: use Redis..."
    - "do it" + history proposing refactor -> "Refactor the UserService class..."
    - "yes, but use FastAPI" + history -> "Build the REST API using FastAPI..."
    """

    query: str = dspy.InputField(
        desc="The user's raw query (may be short, ambiguous, or reference prior context)"
    )
    conversation_context: str = dspy.InputField(
        desc=(
            "Last 1-2 turns of conversation: the user's previous message "
            "and the agent's previous response (including any numbered "
            "options, file lists, proposed plans, or code suggestions)"
        )
    )
    resolved_query: str = dspy.OutputField(
        desc=(
            "The fully resolved, unambiguous task description. If the "
            "original query was already clear, return it verbatim. "
            "Otherwise, expand all references ('this', 'that', 'it', "
            "numbered options) into explicit descriptions using the "
            "conversation context."
        )
    )
    was_resolved: bool = dspy.OutputField(
        desc="True if the query was modified/expanded, False if returned verbatim"
    )


# ── Task Decomposition ───────────────────────────────────────


class TaskDecomposition(dspy.Signature):
    """Decompose a complex coding task into a DAG of parallelisable sub-tasks.

    Given a fully-resolved task description and project context, produce
    a structured decomposition with:
    - Sub-tasks that can be independently assigned to specialist agents
    - Input/output contracts (what files/data each sub-task reads/writes)
    - A shared specification (interfaces, naming conventions, patterns)
    - A dependency graph and parallelism groups

    IMPORTANT: Not every task should be decomposed.  If the task is
    simple enough for a single agent (< 5 RLM iterations), return a
    single sub-task.  Only decompose when there are genuinely
    independent work streams.
    """

    task_description: str = dspy.InputField(
        desc="The fully resolved task description"
    )
    project_context: str = dspy.InputField(
        desc=(
            "Project structure, existing files, conventions, "
            "and any relevant code snippets"
        )
    )
    sub_tasks: list[str] = dspy.OutputField(
        desc=(
            "JSON array of sub-task objects. Each object has: "
            "'id' (str, e.g. 'T1'), "
            "'description' (str, clear actionable description), "
            "'agent_type' (str, one of: 'file', 'coder', 'test', 'review', "
            "'web_search', 'web_fetch', 'planner', 'debug', 'security'), "
            "'inputs' (list[str], files/data this task reads), "
            "'outputs' (list[str], files/data this task produces), "
            "'depends_on' (list[str], IDs of tasks that must complete first), "
            "'max_iterations' (int, suggested RLM iterations 3-15). "
            "Agent type guide: "
            "'file' = read/analyze files (no writes), "
            "'coder' = write production code, "
            "'test' = write and run tests, "
            "'review' = general code review and QA, "
            "'web_search' = search the web for information/docs/APIs, "
            "'web_fetch' = fetch and extract content from specific URLs, "
            "'planner' = decompose sub-problems and create plans (no code), "
            "'debug' = diagnose errors, read logs, trace issues, "
            "'security' = security-focused vulnerability detection. "
            "Example: [{'id':'T1','description':'Analyze existing API routes',"
            "'agent_type':'file','inputs':['src/routes/'],'outputs':['analysis'],"
            "'depends_on':[],'max_iterations':5}]"
        )
    )
    shared_spec: str = dspy.OutputField(
        desc=(
            "A shared specification that all agents must follow: "
            "naming conventions, interface contracts, file structure "
            "conventions, import patterns, error handling approach. "
            "This ensures all agents produce compatible output."
        )
    )
    parallelism_groups: list[str] = dspy.OutputField(
        desc=(
            "JSON array of arrays. Each inner array is a group of task IDs "
            "that can execute in parallel. Example: "
            "[['T1','T2'],['T3'],['T4','T5']] "
            "means T1+T2 run in parallel, then T3, then T4+T5 in parallel."
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


# ── Context-First Solving ──────────────────────────────────


class ContextPlanning(dspy.Signature):
    """Decide what context an autonomous coding agent needs to solve a task.

    Given a user's query and a manifest of project files (paths + sizes),
    determine which files should be read to provide sufficient context for
    solving the task in a single pass.  Prioritise the most relevant files
    first so that, if a budget limit is reached, the most important context
    is always included.
    """

    query: str = dspy.InputField(
        desc="The user's task or question"
    )
    file_manifest: str = dspy.InputField(
        desc=(
            "Newline-separated list of project files with byte sizes, "
            "e.g. 'src/app.py (1234 bytes)'.  Covers all source files "
            "in the project."
        )
    )
    project_summary: str = dspy.InputField(
        desc="Brief description of the project structure and purpose"
    )

    files_to_read: list[str] = dspy.OutputField(
        desc=(
            "File paths to read, ordered by relevance (most important "
            "first).  Include every file likely needed to understand "
            "and solve the task.  Use exact paths from the manifest."
        )
    )
    search_patterns: list[str] = dspy.OutputField(
        desc=(
            "Optional grep/regex patterns to search across the codebase "
            "for additional context (e.g. function names, error strings). "
            "Return empty list if the files_to_read are sufficient."
        )
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why these files are needed"
    )


class SingleShotSolving(dspy.Signature):
    """Solve a coding task given complete source context in a single pass.

    You receive the full source code of all relevant files concatenated
    together, plus the user's task.  Produce a complete solution including:
    1. Step-by-step reasoning about the problem
    2. A human-readable summary of what you did
    3. A JSON array of file operations to execute

    IMPORTANT: The operations array must be valid JSON.  Each operation
    is an object with an 'op' field and operation-specific fields.
    """

    query: str = dspy.InputField(
        desc="The user's task or question"
    )
    full_context: str = dspy.InputField(
        desc=(
            "All relevant source code, concatenated with file headers "
            "like '=== FILE: path/to/file.py ==='"
        )
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step analysis: what is the problem, what is the "
            "root cause, what changes are needed and why"
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "Human-readable summary of what was done (for the user). "
            "Include file names and key changes."
        )
    )
    operations: str = dspy.OutputField(
        desc=(
            'JSON array of operations to execute. Supported ops: '
            '{"op":"write_file","path":"...","content":"..."} — '
            'create or overwrite a file; '
            '{"op":"edit_file","path":"...","old":"...","new":"..."} — '
            'replace first occurrence of old text with new text; '
            '{"op":"run_cmd","cmd":"..."} — run a shell command. '
            'Return [] if no file changes are needed (answer-only query).'
        )
    )


class VerificationFix(dspy.Signature):
    """Fix code that failed linting or verification checks.

    After the initial solution was applied, verification commands (linters,
    type checkers, test runners) reported errors.  Given the current file
    contents and the error output, produce ONLY the operations needed to
    fix those specific errors.  Do not rewrite entire files — use
    edit_file operations to make targeted fixes.
    """

    query: str = dspy.InputField(
        desc="The original user task (for context)"
    )
    full_context: str = dspy.InputField(
        desc=(
            "Current source code of all affected files, "
            "concatenated with file headers"
        )
    )
    verification_errors: str = dspy.InputField(
        desc=(
            "Combined stdout+stderr from the failed verification "
            "commands (lint errors, type errors, test failures)"
        )
    )

    reasoning: str = dspy.OutputField(
        desc="Analysis of each error and how to fix it"
    )
    operations: str = dspy.OutputField(
        desc=(
            'JSON array of fix operations. Prefer edit_file for '
            'targeted fixes. Same format as SingleShotSolving operations.'
        )
    )
