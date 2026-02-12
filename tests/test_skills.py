"""Tests for the memfun-skills package: parser, loader, validator, registry bridge."""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from memfun_core.errors import SkillValidationError
from memfun_runtime.backends.memory import InProcessSkillRegistry
from memfun_skills.loader import SkillLoader
from memfun_skills.parser import parse_skill_md
from memfun_skills.registry import SkillRegistryBridge
from memfun_skills.types import SkillDefinition
from memfun_skills.validator import SkillValidator

if TYPE_CHECKING:
    from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────


def _write_skill_md(skill_dir: Path, content: str) -> Path:
    """Write a SKILL.md into the given directory and return its path."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


_VALID_SKILL_MD = textwrap.dedent("""\
    ---
    name: analyze-code
    description: Analyze code quality and suggest improvements
    version: 1.2.0
    user-invocable: true
    model-invocable: true
    allowed-tools:
      - Read
      - Bash
    tags:
      - code-analysis
      - quality
    ---

    # Analyze Code

    You are a code analysis skill.  When activated, analyze the code
    in the current working directory and produce a quality report.
""")


# ── Parser Tests ─────────────────────────────────────────────────────


class TestParseSkillMd:
    """Tests for parse_skill_md."""

    def test_parse_valid_skill(self, tmp_path: Path) -> None:
        """Parse a valid SKILL.md with full frontmatter and body."""
        skill_file = _write_skill_md(tmp_path / "analyze-code", _VALID_SKILL_MD)

        skill = parse_skill_md(skill_file)

        assert skill.name == "analyze-code"
        assert skill.description == "Analyze code quality and suggest improvements"
        assert skill.version == "1.2.0"
        assert skill.user_invocable is True
        assert skill.model_invocable is True
        assert skill.allowed_tools == ["Read", "Bash"]
        assert skill.tags == ["code-analysis", "quality"]
        assert "code analysis skill" in skill.instructions
        assert skill.source_path == (tmp_path / "analyze-code")

    def test_parse_minimal_skill(self, tmp_path: Path) -> None:
        """Parse a SKILL.md with only required fields."""
        content = textwrap.dedent("""\
            ---
            name: minimal-skill
            description: A minimal skill
            ---

            Do something.
        """)
        skill_file = _write_skill_md(tmp_path / "minimal-skill", content)

        skill = parse_skill_md(skill_file)

        assert skill.name == "minimal-skill"
        assert skill.description == "A minimal skill"
        assert skill.version == "0.1.0"
        assert skill.allowed_tools == []
        assert skill.tags == []
        assert "Do something." in skill.instructions

    def test_parse_missing_name(self, tmp_path: Path) -> None:
        """Verify error when 'name' field is missing."""
        content = textwrap.dedent("""\
            ---
            description: No name skill
            ---

            Instructions here.
        """)
        skill_file = _write_skill_md(tmp_path / "bad-skill", content)

        with pytest.raises(SkillValidationError, match="missing required field 'name'"):
            parse_skill_md(skill_file)

    def test_parse_missing_description(self, tmp_path: Path) -> None:
        """Verify error when 'description' field is missing."""
        content = textwrap.dedent("""\
            ---
            name: no-desc
            ---

            Instructions here.
        """)
        skill_file = _write_skill_md(tmp_path / "no-desc", content)

        with pytest.raises(
            SkillValidationError, match="missing required field 'description'"
        ):
            parse_skill_md(skill_file)

    def test_parse_malformed_yaml(self, tmp_path: Path) -> None:
        """Verify error when frontmatter YAML is malformed."""
        content = textwrap.dedent("""\
            ---
            name: [invalid yaml
            description: this will fail
            ---

            Body.
        """)
        skill_file = _write_skill_md(tmp_path / "malformed", content)

        with pytest.raises(SkillValidationError, match="Invalid YAML"):
            parse_skill_md(skill_file)

    def test_parse_no_frontmatter(self, tmp_path: Path) -> None:
        """Verify error when file has no YAML frontmatter."""
        content = "Just a plain markdown file.\n"
        skill_file = _write_skill_md(tmp_path / "plain", content)

        with pytest.raises(SkillValidationError, match="missing YAML frontmatter"):
            parse_skill_md(skill_file)

    def test_parse_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            parse_skill_md(tmp_path / "nonexistent" / "SKILL.md")

    def test_parse_resolves_scripts_dir(self, tmp_path: Path) -> None:
        """If a scripts/ directory exists, it is attached to the definition."""
        skill_dir = tmp_path / "with-scripts"
        _write_skill_md(skill_dir, _VALID_SKILL_MD)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("#!/bin/bash\necho hi\n")

        skill = parse_skill_md(skill_dir / "SKILL.md")

        assert skill.scripts_dir == skill_dir / "scripts"

    def test_parse_resolves_references_dir(self, tmp_path: Path) -> None:
        """If a references/ directory exists, it is attached to the definition."""
        skill_dir = tmp_path / "with-refs"
        _write_skill_md(skill_dir, _VALID_SKILL_MD)
        (skill_dir / "references").mkdir()

        skill = parse_skill_md(skill_dir / "SKILL.md")

        assert skill.references_dir == skill_dir / "references"


# ── Loader Tests ─────────────────────────────────────────────────────


class TestSkillLoader:
    """Tests for SkillLoader.discover and load_skill."""

    def test_discover_skills(self, tmp_path: Path) -> None:
        """Create a temp directory tree with multiple skills; verify discovery."""
        skills_root = tmp_path / "skills"

        # Skill 1: top-level
        _write_skill_md(
            skills_root / "alpha",
            textwrap.dedent("""\
                ---
                name: alpha
                description: Alpha skill
                ---

                Alpha instructions.
            """),
        )

        # Skill 2: nested
        _write_skill_md(
            skills_root / "group" / "beta",
            textwrap.dedent("""\
                ---
                name: beta
                description: Beta skill
                ---

                Beta instructions.
            """),
        )

        loader = SkillLoader()
        skills = loader.discover([skills_root])

        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_discover_skips_malformed(self, tmp_path: Path) -> None:
        """Malformed SKILL.md files are skipped without crashing discovery."""
        skills_root = tmp_path / "skills"

        # Good skill
        _write_skill_md(
            skills_root / "good",
            textwrap.dedent("""\
                ---
                name: good-skill
                description: A good skill
                ---

                Instructions.
            """),
        )

        # Bad skill (missing description)
        _write_skill_md(
            skills_root / "bad",
            textwrap.dedent("""\
                ---
                name: bad-skill
                ---

                No description.
            """),
        )

        loader = SkillLoader()
        skills = loader.discover([skills_root])

        assert len(skills) == 1
        assert skills[0].name == "good-skill"

    def test_discover_skips_duplicates(self, tmp_path: Path) -> None:
        """If two directories define the same skill name, only the first is kept."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        content = textwrap.dedent("""\
            ---
            name: same-name
            description: Same skill in two places
            ---

            Body.
        """)
        _write_skill_md(root_a / "same-name", content)
        _write_skill_md(root_b / "same-name", content)

        loader = SkillLoader()
        skills = loader.discover([root_a, root_b])

        assert len(skills) == 1
        assert skills[0].name == "same-name"

    def test_discover_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent discovery paths are silently skipped."""
        loader = SkillLoader()
        skills = loader.discover([tmp_path / "does-not-exist"])

        assert skills == []

    def test_load_skill(self, tmp_path: Path) -> None:
        """Load a single skill by directory path."""
        skill_dir = tmp_path / "my-skill"
        _write_skill_md(skill_dir, _VALID_SKILL_MD)

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)

        assert skill.name == "analyze-code"


# ── Validator Tests ──────────────────────────────────────────────────


class TestSkillValidator:
    """Tests for SkillValidator."""

    def test_validate_valid_skill(self) -> None:
        """A well-formed skill should produce no errors."""
        skill = SkillDefinition(
            name="review-code",
            description="Review code for quality issues",
            version="1.0.0",
            allowed_tools=["Read", "Bash"],
            tags=["code-review"],
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert errors == []

    def test_validate_invalid_name_uppercase(self) -> None:
        """Name with uppercase letters should be rejected."""
        skill = SkillDefinition(
            name="ReviewCode",
            description="A valid description",
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("lowercase" in e for e in errors)

    def test_validate_name_too_long(self) -> None:
        """Name longer than 64 characters should be rejected."""
        skill = SkillDefinition(
            name="a" * 65,
            description="A valid description",
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("64" in e for e in errors)

    def test_validate_missing_description(self) -> None:
        """Empty description should be rejected."""
        skill = SkillDefinition(
            name="valid-name",
            description="",
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("description" in e.lower() for e in errors)

    def test_validate_description_too_long(self) -> None:
        """Description longer than 1024 characters should be rejected."""
        skill = SkillDefinition(
            name="valid-name",
            description="x" * 1025,
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("1024" in e for e in errors)

    def test_validate_bad_semver(self) -> None:
        """Non-semver version string should be rejected."""
        skill = SkillDefinition(
            name="valid-name",
            description="A valid description",
            version="not-a-version",
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("SemVer" in e for e in errors)

    def test_validate_scripts_without_allowed_tools(self, tmp_path: Path) -> None:
        """Scripts dir present but no allowed-tools should produce a warning."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()

        skill = SkillDefinition(
            name="with-scripts",
            description="Has scripts",
            scripts_dir=scripts_dir,
            allowed_tools=[],
        )
        validator = SkillValidator()
        errors = validator.validate(skill)
        assert any("allowed-tools" in e for e in errors)

    def test_validate_strict_raises(self) -> None:
        """validate_strict should raise SkillValidationError on invalid skills."""
        skill = SkillDefinition(
            name="BAD NAME!",
            description="",
        )
        validator = SkillValidator()
        with pytest.raises(SkillValidationError):
            validator.validate_strict(skill)

    def test_validate_strict_passes(self) -> None:
        """validate_strict should not raise for valid skills."""
        skill = SkillDefinition(
            name="good-skill",
            description="A perfectly fine skill",
            version="0.1.0",
        )
        validator = SkillValidator()
        validator.validate_strict(skill)  # Should not raise


# ── Registry Bridge Tests ────────────────────────────────────────────


class TestSkillRegistryBridge:
    """Tests for SkillRegistryBridge.sync_skills."""

    async def test_sync_discovered_skills(self, tmp_path: Path) -> None:
        """Discover skills from disk and sync them to an in-memory registry."""
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "analyze-code",
            textwrap.dedent("""\
                ---
                name: analyze-code
                description: Analyze code quality
                version: 1.0.0
                allowed-tools:
                  - Read
                ---

                Analyze the code.
            """),
        )
        _write_skill_md(
            skills_root / "fix-bugs",
            textwrap.dedent("""\
                ---
                name: fix-bugs
                description: Fix bugs automatically
                version: 0.2.0
                allowed-tools:
                  - Read
                  - Write
                ---

                Fix the bugs.
            """),
        )

        loader = SkillLoader(extra_paths=[skills_root])
        registry = InProcessSkillRegistry()
        bridge = SkillRegistryBridge()

        # Use explicit paths to avoid picking up real ./skills/ dir
        synced = await bridge.sync_skills(
            loader, registry, paths=[skills_root]
        )

        assert len(synced) == 2

        # Verify registered in runtime registry
        registered = await registry.list_skills()
        names = {s.name for s in registered}
        assert names == {"analyze-code", "fix-bugs"}

        # Verify metadata mapping
        info = await registry.get_skill("analyze-code")
        assert info is not None
        assert info.description == "Analyze code quality"
        assert info.source_path == str(skills_root / "analyze-code")
        assert info.allowed_tools == ["Read"]
        assert info.metadata["version"] == "1.0.0"

    async def test_sync_deregisters_stale_skills(self, tmp_path: Path) -> None:
        """Skills that no longer exist on disk are removed from the registry."""
        from memfun_runtime.protocols.skill_registry import SkillInfo

        registry = InProcessSkillRegistry()

        # Pre-register a skill that will not exist on disk
        await registry.register_skill(
            SkillInfo(
                name="old-skill",
                description="No longer on disk",
                source_path="/gone",
            )
        )

        # Create only one skill on disk
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "new-skill",
            textwrap.dedent("""\
                ---
                name: new-skill
                description: A new skill
                ---

                New skill body.
            """),
        )

        loader = SkillLoader(extra_paths=[skills_root])
        bridge = SkillRegistryBridge()

        await bridge.sync_skills(loader, registry, paths=[skills_root])

        # old-skill should be gone
        assert await registry.get_skill("old-skill") is None
        # new-skill should be present
        assert await registry.get_skill("new-skill") is not None
