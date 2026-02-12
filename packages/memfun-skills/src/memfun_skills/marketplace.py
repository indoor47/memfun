"""Skill marketplace foundation: packaging, importing, and exporting skills.

Provides utilities to package skills (SKILL.md + scripts + references)
into portable bundles and import them into new environments.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Path is used at runtime for path operations
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("skills.marketplace")

# Valid skill/file name pattern (no path separators, no leading dots)
_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_MAX_SKILL_MD_SIZE = 1_000_000  # 1 MB
_MAX_SCRIPT_SIZE = 500_000  # 500 KB
_MAX_PACKAGE_FILES = 100


def _validate_package_name(name: str) -> None:
    """Validate a skill package name to prevent path traversal.

    Raises:
        ValueError: If the name is invalid or contains path traversal.
    """
    if not name:
        msg = "Skill package name cannot be empty"
        raise ValueError(msg)
    if ".." in name or "/" in name or "\\" in name:
        msg = f"Skill package name contains path traversal: {name!r}"
        raise ValueError(msg)
    if not _SAFE_NAME_RE.match(name):
        msg = f"Skill package name is invalid: {name!r}"
        raise ValueError(msg)


def _validate_filename(filename: str) -> None:
    """Validate a filename within a skill package.

    Prevents path traversal via filenames like '../../../etc/crontab'.

    Raises:
        ValueError: If the filename is unsafe.
    """
    if not filename:
        msg = "Filename cannot be empty"
        raise ValueError(msg)
    # Reject absolute paths and path traversal
    if filename.startswith("/") or filename.startswith("\\"):
        msg = f"Filename must be relative: {filename!r}"
        raise ValueError(msg)
    if ".." in filename.split("/"):
        msg = f"Filename contains path traversal: {filename!r}"
        raise ValueError(msg)
    # Validate each path component
    for part in filename.split("/"):
        if part.startswith("."):
            msg = f"Filename component starts with dot: {filename!r}"
            raise ValueError(msg)
        if not _SAFE_FILENAME_RE.match(part):
            msg = f"Filename contains invalid characters: {filename!r}"
            raise ValueError(msg)


def _assert_within_directory(path: Path, parent: Path) -> None:
    """Assert that a resolved path is within the expected parent directory.

    Prevents symlink escapes and path traversal.

    Raises:
        ValueError: If path resolves outside parent.
    """
    try:
        resolved = path.resolve()
        parent_resolved = parent.resolve()
        resolved.relative_to(parent_resolved)
    except ValueError:
        msg = f"Path escapes target directory: {path} resolves to {path.resolve()}"
        raise ValueError(msg) from None


@dataclass(frozen=True, slots=True)
class SkillPackage:
    """A portable skill package containing all required artifacts.

    Includes the SKILL.md definition, any associated scripts,
    reference files, and metadata about the skill.
    """

    name: str
    version: str
    description: str
    skill_md: str  # Full SKILL.md content
    scripts: dict[str, str]  # filename -> content
    references: dict[str, str]  # filename -> content
    metadata: dict[str, Any]  # Additional metadata (author, license, etc.)


class SkillPackager:
    """Packages skills into portable bundles.

    Reads a skill directory (containing SKILL.md and optionally
    scripts/ and references/ subdirectories) and creates a
    SkillPackage that can be exported or shared.

    Example usage::

        packager = SkillPackager()
        package = packager.package(Path("skills/analyze-code"))
        output_path = packager.export_to_file(package, Path("analyze-code.skill"))
    """

    def package(self, skill_path: Path) -> SkillPackage:
        """Package a skill directory into a SkillPackage.

        Args:
            skill_path: Path to the skill directory containing SKILL.md.

        Returns:
            A SkillPackage with all skill artifacts.

        Raises:
            FileNotFoundError: If SKILL.md is not found in skill_path.
            ValueError: If SKILL.md is malformed or missing required fields.
        """
        skill_md_path = skill_path / "SKILL.md"
        if not skill_md_path.exists():
            msg = f"SKILL.md not found in {skill_path}"
            raise FileNotFoundError(msg)

        # Read SKILL.md
        skill_md_content = skill_md_path.read_text(encoding="utf-8")

        # Extract metadata from SKILL.md frontmatter
        name, version, description = self._extract_metadata(skill_md_content, skill_path)

        # Package scripts if they exist
        scripts = self._package_directory(skill_path / "scripts")

        # Package references if they exist
        references = self._package_directory(skill_path / "references")

        # Build package metadata
        metadata = {
            "packaged_at": skill_path.as_posix(),
            "has_scripts": len(scripts) > 0,
            "has_references": len(references) > 0,
        }

        logger.info(
            "Packaged skill '%s' (version=%s, scripts=%d, references=%d)",
            name,
            version,
            len(scripts),
            len(references),
        )

        return SkillPackage(
            name=name,
            version=version,
            description=description,
            skill_md=skill_md_content,
            scripts=scripts,
            references=references,
            metadata=metadata,
        )

    def export_to_file(
        self,
        package: SkillPackage,
        output_path: Path,
    ) -> Path:
        """Export a SkillPackage to a JSON file.

        Args:
            package: The skill package to export.
            output_path: Destination file path.

        Returns:
            The resolved output path where the package was written.
        """
        package_data = {
            "name": package.name,
            "version": package.version,
            "description": package.description,
            "skill_md": package.skill_md,
            "scripts": package.scripts,
            "references": package.references,
            "metadata": package.metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(package_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Exported skill package to %s", output_path)
        return output_path.resolve()

    def _extract_metadata(
        self, skill_md: str, skill_path: Path
    ) -> tuple[str, str, str]:
        """Extract name, version, and description from SKILL.md frontmatter.

        Returns:
            Tuple of (name, version, description).

        Raises:
            ValueError: If required fields are missing.
        """
        # Simple frontmatter extraction (matches parser.py logic)
        if not skill_md.lstrip("\n").startswith("---"):
            msg = f"SKILL.md missing frontmatter in {skill_path}"
            raise ValueError(msg)

        # Find frontmatter section
        lines = skill_md.split("\n")
        frontmatter_lines = []
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if in_frontmatter:
                    break  # End of frontmatter
                in_frontmatter = True
                continue
            if in_frontmatter:
                frontmatter_lines.append(line)

        # Parse key fields (simple YAML parsing)
        name = None
        version = "0.1.1"  # default
        description = None

        i = 0
        while i < len(frontmatter_lines):
            line = frontmatter_lines[i]
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("version:"):
                version = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                desc_value = line.split(":", 1)[1].strip()
                # Handle multi-line description with ">"
                if desc_value == ">":
                    desc_lines = []
                    i += 1
                    while i < len(frontmatter_lines) and frontmatter_lines[i].startswith("  "):
                        desc_lines.append(frontmatter_lines[i].strip())
                        i += 1
                    description = " ".join(desc_lines)
                    i -= 1  # Back up one since outer loop will increment
                else:
                    description = desc_value
            i += 1

        if not name:
            msg = f"SKILL.md missing 'name' field in {skill_path}"
            raise ValueError(msg)
        if not description:
            msg = f"SKILL.md missing 'description' field in {skill_path}"
            raise ValueError(msg)

        return name, version, description

    def _package_directory(self, dir_path: Path) -> dict[str, str]:
        """Package all files in a directory into a filename->content dict.

        Args:
            dir_path: Path to directory to package.

        Returns:
            Dict mapping relative filename to file content.
        """
        if not dir_path.is_dir():
            return {}

        packaged = {}
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                relative_name = file_path.relative_to(dir_path).as_posix()
                try:
                    content = file_path.read_text(encoding="utf-8")
                    packaged[relative_name] = content
                except UnicodeDecodeError:
                    # Skip binary files or use base64 encoding if needed
                    logger.warning("Skipping binary file: %s", file_path)

        return packaged


class SkillImporter:
    """Imports skill packages into a target directory.

    Unpacks a SkillPackage (or reads from a .skill JSON file) and
    writes the SKILL.md and any associated files to the target
    directory, creating subdirectories as needed.

    Example usage::

        importer = SkillImporter()
        skill_path = importer.import_from_file(
            Path("analyze-code.skill"),
            Path("~/.memfun/imported-skills/"),
        )
    """

    def import_from_file(
        self,
        package_path: Path,
        target_dir: Path,
    ) -> Path:
        """Import a skill package from a JSON file.

        Args:
            package_path: Path to the .skill JSON file.
            target_dir: Directory where the skill should be created.

        Returns:
            Path to the created skill directory.

        Raises:
            FileNotFoundError: If package_path does not exist.
            ValueError: If the package file is malformed.
        """
        if not package_path.exists():
            msg = f"Skill package not found: {package_path}"
            raise FileNotFoundError(msg)

        # Load package data
        raw_text = package_path.read_text(encoding="utf-8")
        if len(raw_text) > _MAX_SKILL_MD_SIZE * 10:  # 10 MB total package limit
            msg = f"Skill package file too large: {len(raw_text)} bytes"
            raise ValueError(msg)

        package_data = json.loads(raw_text)

        # Validate required fields
        if "name" not in package_data or "skill_md" not in package_data:
            msg = f"Invalid skill package: {package_path}"
            raise ValueError(msg)

        # Validate package name (prevents path traversal on import)
        _validate_package_name(package_data["name"])

        # Validate all filenames in scripts and references
        scripts = package_data.get("scripts", {})
        references = package_data.get("references", {})

        if len(scripts) + len(references) > _MAX_PACKAGE_FILES:
            msg = f"Package contains too many files: {len(scripts) + len(references)}"
            raise ValueError(msg)

        for filename in scripts:
            _validate_filename(filename)
        for filename in references:
            _validate_filename(filename)

        # Validate sizes
        if len(package_data.get("skill_md", "")) > _MAX_SKILL_MD_SIZE:
            msg = "SKILL.md content exceeds size limit"
            raise ValueError(msg)
        for fname, content in scripts.items():
            if len(content) > _MAX_SCRIPT_SIZE:
                msg = f"Script {fname!r} exceeds size limit"
                raise ValueError(msg)
        for fname, content in references.items():
            if len(content) > _MAX_SCRIPT_SIZE:
                msg = f"Reference {fname!r} exceeds size limit"
                raise ValueError(msg)

        # Reconstruct SkillPackage
        package = SkillPackage(
            name=package_data["name"],
            version=package_data.get("version", "0.1.0"),
            description=package_data.get("description", ""),
            skill_md=package_data["skill_md"],
            scripts=scripts,
            references=references,
            metadata=package_data.get("metadata", {}),
        )

        return self.import_package(package, target_dir)

    def import_package(
        self,
        package: SkillPackage,
        target_dir: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Import a SkillPackage into the target directory.

        Args:
            package: The skill package to import.
            target_dir: Directory where the skill should be created.
            overwrite: If False (default), refuse to overwrite existing files.

        Returns:
            Path to the created skill directory.

        Raises:
            ValueError: If package name is invalid or paths escape target dir.
            FileExistsError: If skill already exists and overwrite is False.
        """
        # Validate package name to prevent path traversal
        _validate_package_name(package.name)

        skill_dir = target_dir / package.name

        # Check for existing skill (prevent silent overwrite)
        if skill_dir.exists() and not overwrite:
            msg = (
                f"Skill '{package.name}' already exists at {skill_dir}. "
                f"Pass overwrite=True to replace it."
            )
            raise FileExistsError(msg)

        skill_dir.mkdir(parents=True, exist_ok=True)

        # Verify skill_dir is actually within target_dir (symlink protection)
        _assert_within_directory(skill_dir, target_dir)

        # Write SKILL.md
        skill_md_path = skill_dir / "SKILL.md"
        # Refuse to write if target is a symlink pointing outside
        if skill_md_path.is_symlink():
            msg = f"SKILL.md target is a symlink: {skill_md_path}"
            raise ValueError(msg)
        skill_md_path.write_text(package.skill_md, encoding="utf-8")

        # Write scripts
        if package.scripts:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for filename, content in package.scripts.items():
                _validate_filename(filename)
                script_path = scripts_dir / filename
                script_path.parent.mkdir(parents=True, exist_ok=True)
                # Verify the resolved path is within the scripts directory
                _assert_within_directory(script_path, scripts_dir)
                if script_path.is_symlink():
                    msg = f"Script target is a symlink: {script_path}"
                    raise ValueError(msg)
                script_path.write_text(content, encoding="utf-8")

        # Write references
        if package.references:
            references_dir = skill_dir / "references"
            references_dir.mkdir(exist_ok=True)
            for filename, content in package.references.items():
                _validate_filename(filename)
                ref_path = references_dir / filename
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                # Verify the resolved path is within the references directory
                _assert_within_directory(ref_path, references_dir)
                if ref_path.is_symlink():
                    msg = f"Reference target is a symlink: {ref_path}"
                    raise ValueError(msg)
                ref_path.write_text(content, encoding="utf-8")

        logger.info(
            "Imported skill '%s' (version=%s) to %s",
            package.name,
            package.version,
            skill_dir,
        )

        return skill_dir
