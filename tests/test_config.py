from __future__ import annotations

import tempfile
from pathlib import Path

from memfun_core.config import MemfunConfig


class TestConfig:
    def test_default_config(self):
        config = MemfunConfig()
        assert config.backend.tier == "sqlite"
        assert config.llm.provider == "anthropic"
        assert config.sandbox.backend == "local"

    def test_from_toml_missing_file(self):
        config = MemfunConfig.from_toml("/nonexistent/path/memfun.toml")
        assert config.backend.tier == "sqlite"  # Returns defaults

    def test_from_toml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('''
[project]
name = "test-project"

[llm]
provider = "openai"
model = "gpt-4"

[backend]
tier = "memory"

[sandbox]
backend = "docker"
''')
            f.flush()
            config = MemfunConfig.from_toml(f.name)

        assert config.project_name == "test-project"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"
        assert config.backend.tier == "memory"
        assert config.sandbox.backend == "docker"

        Path(f.name).unlink()
