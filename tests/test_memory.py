"""Tests for the file-based MEMORY.md memory module."""
from __future__ import annotations

from pathlib import Path

import pytest
from memfun_cli.memory import (
    _classify_section,
    _is_duplicate,
    _read_memory_file,
    append_learning,
    create_starter_memory,
    forget,
    get_memory_display,
    load_memory_context,
    remember,
)

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect Path.home() to a temp directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home" / ".memfun").mkdir(parents=True)
    return tmp_path / "home"


@pytest.fixture()
def fake_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect Path.cwd() to a temp directory."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / ".memfun").mkdir()
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    return cwd


# ── load_memory_context ─────────────────────────────────────


class TestLoadMemoryContext:
    def test_no_files(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns empty string when no memory files."""
        assert load_memory_context() == ""

    def test_global_only(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Loads global memory when only global file exists."""
        gpath = fake_home / ".memfun" / "MEMORY.md"
        gpath.write_text(
            "# Memfun Memory\n## Preferences\n- Use port 8080\n"
        )
        result = load_memory_context()
        assert "GLOBAL MEMORY" in result
        assert "port 8080" in result
        assert "PROJECT MEMORY" not in result

    def test_project_only(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Loads project memory when only project file exists."""
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        ppath.write_text(
            "# Memfun Memory\n## Technical\n- PostgreSQL 16\n"
        )
        result = load_memory_context()
        assert "PROJECT MEMORY" in result
        assert "PostgreSQL 16" in result

    def test_both_files(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Loads both with correct ordering."""
        gpath = fake_home / ".memfun" / "MEMORY.md"
        gpath.write_text(
            "# Memory\n## Preferences\n- Global pref\n"
        )
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        ppath.write_text(
            "# Memory\n## Preferences\n- Project pref\n"
        )
        result = load_memory_context()
        assert "GLOBAL MEMORY" in result
        assert "PROJECT MEMORY" in result
        # Global appears before project
        global_pos = result.index("GLOBAL MEMORY")
        project_pos = result.index("PROJECT MEMORY")
        assert global_pos < project_pos

    def test_strips_comments(
        self, fake_home: Path, fake_cwd: Path
    ):
        """HTML comments are stripped from context."""
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        ppath.write_text(
            "# Memory\n"
            "## Preferences\n"
            "<!-- This is a comment -->\n"
            "- Visible line\n"
        )
        result = load_memory_context()
        assert "Visible line" in result
        assert "comment" not in result

    def test_enforces_line_limit(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Truncates to 50 content lines."""
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        # 1 header + 100 entries = 101 lines
        lines = ["# Memory"] + [
            f"- Entry {i}" for i in range(100)
        ]
        ppath.write_text("\n".join(lines))
        result = load_memory_context()
        # 50-line cap: header + entries 0..48
        assert "Entry 0" in result
        assert "Entry 48" in result
        assert "Entry 50" not in result

    def test_always_follow_directive(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Context includes the ALWAYS FOLLOW directive."""
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        ppath.write_text("# Memory\n- Something\n")
        result = load_memory_context()
        assert "ALWAYS FOLLOW" in result


# ── _read_memory_file ────────────────────────────────────────


class TestReadMemoryFile:
    def test_nonexistent(self, tmp_path: Path):
        """Returns empty for missing file."""
        assert _read_memory_file(tmp_path / "nope.md") == ""

    def test_char_limit(self, tmp_path: Path):
        """Truncates to char limit."""
        path = tmp_path / "big.md"
        path.write_text("x" * 3000)
        result = _read_memory_file(path)
        assert len(result) <= 2010  # 2000 + "..."


# ── remember ─────────────────────────────────────────────────


class TestRemember:
    def test_adds_to_preferences(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Preference keywords route to ## Preferences."""
        msg = remember("Use port 8080")
        assert "Remembered" in msg
        assert "Preferences" in msg
        content = (
            fake_cwd / ".memfun" / "MEMORY.md"
        ).read_text()
        assert "- Use port 8080" in content

    def test_adds_to_technical(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Technical keywords route to ## Technical."""
        msg = remember("API endpoint is /v2/users")
        assert "Technical" in msg

    def test_prevents_exact_duplicates(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Exact duplicate text is rejected."""
        remember("Use port 8080")
        msg = remember("Use port 8080")
        assert "Already remembered" in msg

    def test_prevents_fuzzy_duplicates(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Similar text is rejected."""
        remember("Use port 8080 for dev server")
        msg = remember("Use port 8080 for dev servers")
        assert "Already remembered" in msg

    def test_creates_file_if_missing(
        self, fake_home: Path, fake_cwd: Path
    ):
        """MEMORY.md is created if it doesn't exist."""
        mem_path = fake_cwd / ".memfun" / "MEMORY.md"
        assert not mem_path.exists()
        remember("Something new")
        assert mem_path.exists()

    def test_respects_line_limit(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns error when at capacity."""
        # Directly write 50 content lines to the file
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        lines = ["# Memory", "## Preferences"]
        for i in range(50):
            lines.append(f"- {chr(65 + i % 26)}{i:04d} x")
        ppath.write_text("\n".join(lines))
        msg = remember("One more unique entry")
        assert "full" in msg.lower()

    def test_global_flag(
        self, fake_home: Path, fake_cwd: Path
    ):
        """project=False writes to global file."""
        msg = remember("Global pref", project=False)
        assert "global" in msg
        content = (
            fake_home / ".memfun" / "MEMORY.md"
        ).read_text()
        assert "- Global pref" in content


# ── forget ───────────────────────────────────────────────────


class TestForget:
    def test_by_line_number(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Removes the nth content line."""
        remember("First entry")
        remember("Second entry unique")
        msg = forget("1")
        assert "First entry" in msg
        content = (
            fake_cwd / ".memfun" / "MEMORY.md"
        ).read_text()
        assert "First entry" not in content
        assert "Second entry" in content

    def test_by_text_match(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Removes line containing matching text."""
        remember("Use port 8080")
        msg = forget("port 8080")
        assert "Forgot" in msg
        content = (
            fake_cwd / ".memfun" / "MEMORY.md"
        ).read_text()
        assert "port 8080" not in content

    def test_nonexistent(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns 'not found' for unknown text."""
        remember("Existing")
        msg = forget("nonexistent thing")
        assert "No matching" in msg

    def test_line_number_out_of_range(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns error for out-of-range number."""
        remember("Only entry")
        msg = forget("5")
        assert "out of range" in msg


# ── append_learning ──────────────────────────────────────────


class TestAppendLearning:
    def test_adds_line(
        self, fake_home: Path, fake_cwd: Path
    ):
        """New learning is added."""
        result = append_learning("User prefers Flask")
        assert result is True
        content = (
            fake_cwd / ".memfun" / "MEMORY.md"
        ).read_text()
        assert "User prefers Flask" in content

    def test_deduplicates(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns False for duplicate content."""
        append_learning("User prefers Flask")
        result = append_learning("User prefers Flask")
        assert result is False

    def test_respects_limit(
        self, fake_home: Path, fake_cwd: Path
    ):
        """Returns False when at capacity."""
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        lines = ["# Memory", "## Preferences"]
        for i in range(50):
            lines.append(f"- {chr(65 + i % 26)}{i:04d} z")
        ppath.write_text("\n".join(lines))
        result = append_learning("One more unique item")
        assert result is False


# ── _classify_section ────────────────────────────────────────


class TestClassifySection:
    def test_preference_keywords(self):
        assert _classify_section("Always use port 8080") == "Preferences"
        assert _classify_section("Prefer Flask") == "Preferences"

    def test_technical_keywords(self):
        assert _classify_section("API version 2") == "Technical"
        assert _classify_section("database config") == "Technical"

    def test_pattern_keywords(self):
        assert _classify_section("directory structure uses src") == "Patterns"

    def test_workflow_keywords(self):
        assert _classify_section("test before deploy") == "Workflow"

    def test_default_to_preferences(self):
        assert _classify_section("random stuff") == "Preferences"


# ── _is_duplicate ────────────────────────────────────────────


class TestIsDuplicate:
    def test_exact_match(self):
        lines = ["# Header", "- Use port 8080"]
        assert _is_duplicate("Use port 8080", lines) is True

    def test_fuzzy_match_above_threshold(self):
        lines = ["- Use port 8080 for servers"]
        assert _is_duplicate(
            "Use port 8080 for server", lines
        ) is True

    def test_no_match_below_threshold(self):
        lines = ["- Use Flask for web apps"]
        assert _is_duplicate(
            "Deploy to Kubernetes", lines
        ) is False

    def test_ignores_non_content_lines(self):
        lines = ["# Use port 8080", "## Preferences"]
        assert _is_duplicate("Use port 8080", lines) is False


# ── create_starter_memory ────────────────────────────────────


class TestCreateStarterMemory:
    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / ".memfun" / "MEMORY.md"
        create_starter_memory(path)
        assert path.exists()
        content = path.read_text()
        assert "## Preferences" in content
        assert "## Technical" in content

    def test_idempotent(self, tmp_path: Path):
        path = tmp_path / ".memfun" / "MEMORY.md"
        create_starter_memory(path)
        path.read_text()  # verify it's readable
        # Write custom content
        path.write_text("Custom content")
        create_starter_memory(path)
        # Should NOT overwrite
        assert path.read_text() == "Custom content"


# ── get_memory_display ───────────────────────────────────────


class TestGetMemoryDisplay:
    def test_no_files(
        self, fake_home: Path, fake_cwd: Path
    ):
        display = get_memory_display()
        assert "No global memory" in display
        assert "No project memory" in display

    def test_with_files(
        self, fake_home: Path, fake_cwd: Path
    ):
        gpath = fake_home / ".memfun" / "MEMORY.md"
        gpath.write_text("# Global\n- G entry\n")
        ppath = fake_cwd / ".memfun" / "MEMORY.md"
        ppath.write_text("# Project\n- P entry\n")
        display = get_memory_display()
        assert "Global memory" in display
        assert "Project memory" in display
        assert "G entry" in display
        assert "P entry" in display
