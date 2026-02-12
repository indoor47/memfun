"""Tests for skill effectiveness tracking and marketplace (packaging/importing).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from memfun_skills.effectiveness import SkillEffectivenessTracker
from memfun_skills.marketplace import SkillImporter, SkillPackage, SkillPackager

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit testing
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
tags:
  - testing
  - sample
---

# Test Skill

This is a test skill used in unit tests.

## Instructions

1. Do the thing
2. Report results
"""


# ── 1. SkillEffectivenessTracker Tests ───────────────────────


class TestSkillEffectivenessTracker:
    """Tests for SkillEffectivenessTracker using in-memory mode."""

    async def test_record_and_get_stats(self) -> None:
        """Record executions and retrieve aggregated stats."""
        tracker = SkillEffectivenessTracker()

        await tracker.record_execution("analyze", success=True, duration_ms=100.0)
        await tracker.record_execution("analyze", success=True, duration_ms=200.0)
        await tracker.record_execution(
            "analyze", success=False, duration_ms=300.0, error="timeout"
        )

        stats = await tracker.get_stats("analyze")
        assert stats is not None
        assert stats.skill_name == "analyze"
        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert stats.success_rate == pytest.approx(2 / 3)
        assert stats.avg_duration_ms == pytest.approx(200.0)
        assert stats.error_count == 1
        assert "timeout" in stats.common_errors

    async def test_get_stats_nonexistent(self) -> None:
        """Getting stats for an untracked skill returns None."""
        tracker = SkillEffectivenessTracker()
        stats = await tracker.get_stats("nonexistent")
        assert stats is None

    async def test_get_stats_with_user_ratings(self) -> None:
        """Avg user rating is computed from non-None ratings."""
        tracker = SkillEffectivenessTracker()

        await tracker.record_execution(
            "review", success=True, duration_ms=100.0, user_rating=4.0
        )
        await tracker.record_execution(
            "review", success=True, duration_ms=150.0, user_rating=5.0
        )
        await tracker.record_execution(
            "review", success=True, duration_ms=120.0  # no rating
        )

        stats = await tracker.get_stats("review")
        assert stats is not None
        assert stats.avg_user_rating == pytest.approx(4.5)

    async def test_get_stats_no_user_ratings(self) -> None:
        """Avg user rating is None when no ratings are provided."""
        tracker = SkillEffectivenessTracker()
        await tracker.record_execution("fix", success=True, duration_ms=100.0)

        stats = await tracker.get_stats("fix")
        assert stats is not None
        assert stats.avg_user_rating is None

    async def test_get_all_stats(self) -> None:
        """get_all_stats returns stats for every tracked skill."""
        tracker = SkillEffectivenessTracker()

        await tracker.record_execution("skill-a", success=True, duration_ms=100.0)
        await tracker.record_execution("skill-b", success=False, duration_ms=200.0)
        await tracker.record_execution("skill-c", success=True, duration_ms=300.0)

        all_stats = await tracker.get_all_stats()
        assert len(all_stats) == 3
        names = {s.skill_name for s in all_stats}
        assert names == {"skill-a", "skill-b", "skill-c"}

    async def test_get_all_stats_empty(self) -> None:
        """get_all_stats returns empty list when nothing is tracked."""
        tracker = SkillEffectivenessTracker()
        assert await tracker.get_all_stats() == []

    async def test_get_underperforming(self) -> None:
        """get_underperforming finds skills below the success threshold."""
        tracker = SkillEffectivenessTracker()

        # Good skill: 5/5 success
        for _ in range(5):
            await tracker.record_execution("good", success=True, duration_ms=100.0)

        # Bad skill: 1/5 success
        for i in range(5):
            await tracker.record_execution(
                "bad", success=(i == 0), duration_ms=100.0
            )

        # Mediocre skill: 3/5 success (60%)
        for i in range(5):
            await tracker.record_execution(
                "mediocre", success=(i < 3), duration_ms=100.0
            )

        underperforming = await tracker.get_underperforming(
            min_executions=5, max_success_rate=0.7
        )

        names = {s.skill_name for s in underperforming}
        assert "bad" in names
        assert "mediocre" in names
        assert "good" not in names

    async def test_get_underperforming_sorted_worst_first(self) -> None:
        """Underperforming skills are sorted by success rate ascending."""
        tracker = SkillEffectivenessTracker()

        # 40% success
        for i in range(5):
            await tracker.record_execution(
                "forty", success=(i < 2), duration_ms=100.0
            )

        # 20% success
        for i in range(5):
            await tracker.record_execution(
                "twenty", success=(i < 1), duration_ms=100.0
            )

        underperforming = await tracker.get_underperforming(
            min_executions=5, max_success_rate=0.5
        )
        assert len(underperforming) == 2
        assert underperforming[0].skill_name == "twenty"
        assert underperforming[1].skill_name == "forty"

    async def test_get_underperforming_respects_min_executions(self) -> None:
        """Skills with fewer executions than min_executions are excluded."""
        tracker = SkillEffectivenessTracker()

        # Only 2 executions (below min_executions=5)
        await tracker.record_execution("few", success=False, duration_ms=100.0)
        await tracker.record_execution("few", success=False, duration_ms=100.0)

        underperforming = await tracker.get_underperforming(min_executions=5)
        assert underperforming == []

    async def test_record_multiple_errors(self) -> None:
        """Common errors are tracked and deduplicated in stats."""
        tracker = SkillEffectivenessTracker()

        await tracker.record_execution(
            "buggy", success=False, duration_ms=100.0, error="timeout"
        )
        await tracker.record_execution(
            "buggy", success=False, duration_ms=100.0, error="timeout"
        )
        await tracker.record_execution(
            "buggy", success=False, duration_ms=100.0, error="oom"
        )

        stats = await tracker.get_stats("buggy")
        assert stats is not None
        assert stats.error_count == 3
        # "timeout" appears twice, "oom" once, so timeout should be listed first
        assert stats.common_errors[0] == "timeout"

    async def test_last_execution_timestamp(self) -> None:
        """last_execution reflects the most recent record."""
        tracker = SkillEffectivenessTracker()

        await tracker.record_execution("ts-test", success=True, duration_ms=100.0)
        await tracker.record_execution("ts-test", success=True, duration_ms=200.0)

        stats = await tracker.get_stats("ts-test")
        assert stats is not None
        assert stats.last_execution > 0


# ── 2. SkillPackager Tests ───────────────────────────────────


class TestSkillPackager:
    """Tests for SkillPackager: package() and export_to_file()."""

    def test_package_basic_skill(self, tmp_path: Path) -> None:
        """Package a minimal skill directory with just SKILL.md."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")

        packager = SkillPackager()
        package = packager.package(skill_dir)

        assert package.name == "test-skill"
        assert package.version == "1.0.0"
        assert "test skill" in package.description.lower()
        assert package.skill_md == SAMPLE_SKILL_MD
        assert package.scripts == {}
        assert package.references == {}

    def test_package_with_scripts(self, tmp_path: Path) -> None:
        """Package a skill directory containing scripts."""
        skill_dir = tmp_path / "scripted-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("print('hello')", encoding="utf-8")
        (scripts_dir / "utils.sh").write_text("echo hi", encoding="utf-8")

        packager = SkillPackager()
        package = packager.package(skill_dir)

        assert len(package.scripts) == 2
        assert "helper.py" in package.scripts
        assert "utils.sh" in package.scripts
        assert package.scripts["helper.py"] == "print('hello')"

    def test_package_with_references(self, tmp_path: Path) -> None:
        """Package a skill directory containing references."""
        skill_dir = tmp_path / "ref-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")

        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "guide.txt").write_text("Reference content", encoding="utf-8")

        packager = SkillPackager()
        package = packager.package(skill_dir)

        assert len(package.references) == 1
        assert "guide.txt" in package.references

    def test_package_missing_skill_md(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when SKILL.md is missing."""
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()

        packager = SkillPackager()
        with pytest.raises(FileNotFoundError, match=r"SKILL\.md"):
            packager.package(skill_dir)

    def test_package_malformed_frontmatter(self, tmp_path: Path) -> None:
        """Raises ValueError when SKILL.md has no frontmatter."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# No frontmatter here\nJust content.",
            encoding="utf-8",
        )

        packager = SkillPackager()
        with pytest.raises(ValueError, match="frontmatter"):
            packager.package(skill_dir)

    def test_package_metadata_fields(self, tmp_path: Path) -> None:
        """Package metadata includes expected fields."""
        skill_dir = tmp_path / "meta-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")

        packager = SkillPackager()
        package = packager.package(skill_dir)

        assert "packaged_at" in package.metadata
        assert isinstance(package.metadata["has_scripts"], bool)
        assert isinstance(package.metadata["has_references"], bool)

    def test_export_to_file(self, tmp_path: Path) -> None:
        """export_to_file writes valid JSON and returns the resolved path."""
        skill_dir = tmp_path / "export-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")

        packager = SkillPackager()
        package = packager.package(skill_dir)

        output_path = tmp_path / "output" / "test.skill"
        result_path = packager.export_to_file(package, output_path)

        assert result_path.exists()

        # Verify JSON is valid and contains expected data
        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert data["name"] == "test-skill"
        assert data["version"] == "1.0.0"
        assert data["skill_md"] == SAMPLE_SKILL_MD

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export_to_file creates parent directories if needed."""
        package = SkillPackage(
            name="simple",
            version="1.0.0",
            description="Simple test",
            skill_md=SAMPLE_SKILL_MD,
            scripts={},
            references={},
            metadata={},
        )

        output_path = tmp_path / "deep" / "nested" / "dir" / "skill.json"
        packager = SkillPackager()
        result = packager.export_to_file(package, output_path)
        assert result.exists()


# ── 3. SkillImporter Tests ───────────────────────────────────


class TestSkillImporter:
    """Tests for SkillImporter: import_from_file and import_package."""

    def _create_package_file(
        self,
        path: Path,
        *,
        name: str = "imported-skill",
        scripts: dict[str, str] | None = None,
        references: dict[str, str] | None = None,
    ) -> Path:
        """Create a .skill JSON file for testing import."""
        data = {
            "name": name,
            "version": "1.0.0",
            "description": "An imported test skill",
            "skill_md": SAMPLE_SKILL_MD,
            "scripts": scripts or {},
            "references": references or {},
            "metadata": {"source": "test"},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_import_from_file_basic(self, tmp_path: Path) -> None:
        """Import a minimal skill package and verify directory structure."""
        pkg_file = self._create_package_file(tmp_path / "test.skill")
        target_dir = tmp_path / "skills"
        target_dir.mkdir()

        importer = SkillImporter()
        skill_dir = importer.import_from_file(pkg_file, target_dir)

        assert skill_dir.is_dir()
        assert skill_dir.name == "imported-skill"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == SAMPLE_SKILL_MD

    def test_import_from_file_with_scripts(self, tmp_path: Path) -> None:
        """Import a package with scripts and verify they are created."""
        pkg_file = self._create_package_file(
            tmp_path / "scripted.skill",
            scripts={"run.py": "print('running')", "util.sh": "echo done"},
        )
        target_dir = tmp_path / "skills"
        target_dir.mkdir()

        importer = SkillImporter()
        skill_dir = importer.import_from_file(pkg_file, target_dir)

        scripts_dir = skill_dir / "scripts"
        assert scripts_dir.is_dir()
        assert (scripts_dir / "run.py").read_text(encoding="utf-8") == "print('running')"
        assert (scripts_dir / "util.sh").read_text(encoding="utf-8") == "echo done"

    def test_import_from_file_with_references(self, tmp_path: Path) -> None:
        """Import a package with references and verify they are created."""
        pkg_file = self._create_package_file(
            tmp_path / "refs.skill",
            references={"guide.md": "# Guide\nReference text"},
        )
        target_dir = tmp_path / "skills"
        target_dir.mkdir()

        importer = SkillImporter()
        skill_dir = importer.import_from_file(pkg_file, target_dir)

        refs_dir = skill_dir / "references"
        assert refs_dir.is_dir()
        assert (refs_dir / "guide.md").read_text(encoding="utf-8") == "# Guide\nReference text"

    def test_import_from_file_nonexistent_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for nonexistent package file."""
        importer = SkillImporter()
        with pytest.raises(FileNotFoundError, match="not found"):
            importer.import_from_file(
                tmp_path / "nonexistent.skill", tmp_path
            )

    def test_import_from_file_malformed_raises(self, tmp_path: Path) -> None:
        """Raises ValueError for malformed package file."""
        bad_file = tmp_path / "bad.skill"
        bad_file.write_text('{"invalid": true}', encoding="utf-8")

        importer = SkillImporter()
        with pytest.raises(ValueError, match="Invalid skill package"):
            importer.import_from_file(bad_file, tmp_path)

    def test_import_package_directly(self, tmp_path: Path) -> None:
        """import_package works with a SkillPackage object directly."""
        package = SkillPackage(
            name="direct-import",
            version="2.0.0",
            description="Directly imported",
            skill_md=SAMPLE_SKILL_MD,
            scripts={"helper.py": "# help"},
            references={},
            metadata={},
        )

        target_dir = tmp_path / "skills"
        target_dir.mkdir()

        importer = SkillImporter()
        skill_dir = importer.import_package(package, target_dir)

        assert skill_dir.name == "direct-import"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "scripts" / "helper.py").exists()

    def test_round_trip_package_export_import(self, tmp_path: Path) -> None:
        """Full round trip: create skill dir -> package -> export -> import."""
        # Step 1: Create original skill directory
        original_dir = tmp_path / "original-skill"
        original_dir.mkdir()
        (original_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        scripts_dir = original_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "analyze.py").write_text("# analyzer", encoding="utf-8")

        # Step 2: Package it
        packager = SkillPackager()
        package = packager.package(original_dir)

        # Step 3: Export to file
        export_path = tmp_path / "exported.skill"
        packager.export_to_file(package, export_path)

        # Step 4: Import into new directory
        import_dir = tmp_path / "imported-skills"
        import_dir.mkdir()

        importer = SkillImporter()
        imported_skill_dir = importer.import_from_file(export_path, import_dir)

        # Verify round-trip fidelity
        assert imported_skill_dir.is_dir()
        imported_md = (imported_skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert imported_md == SAMPLE_SKILL_MD
        imported_script = (
            imported_skill_dir / "scripts" / "analyze.py"
        ).read_text(encoding="utf-8")
        assert imported_script == "# analyzer"
