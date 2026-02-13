"""Unit tests for specialist agents.

Tests specialist agent registration and type mapping.
Does NOT test handle() method (requires full RLM setup).
"""
from __future__ import annotations

from memfun_agent.specialists import (
    AGENT_TYPE_MAP,
    agent_name_for_type,
)

# ── Type mapping tests ────────────────────────────────────────


def test_agent_name_for_type_file():
    """file type maps to file-agent."""
    assert agent_name_for_type("file") == "file-agent"


def test_agent_name_for_type_coder():
    """coder type maps to coder-agent."""
    assert agent_name_for_type("coder") == "coder-agent"


def test_agent_name_for_type_test():
    """test type maps to test-agent."""
    assert agent_name_for_type("test") == "test-agent"


def test_agent_name_for_type_review():
    """review type maps to review-agent."""
    assert agent_name_for_type("review") == "review-agent"


def test_agent_name_for_type_web_search():
    """web_search type maps to web-search-agent."""
    assert agent_name_for_type("web_search") == "web-search-agent"


def test_agent_name_for_type_web_fetch():
    """web_fetch type maps to web-fetch-agent."""
    assert agent_name_for_type("web_fetch") == "web-fetch-agent"


def test_agent_name_for_type_planner():
    """planner type maps to planner-agent."""
    assert agent_name_for_type("planner") == "planner-agent"


def test_agent_name_for_type_debug():
    """debug type maps to debug-agent."""
    assert agent_name_for_type("debug") == "debug-agent"


def test_agent_name_for_type_security():
    """security type maps to security-agent."""
    assert agent_name_for_type("security") == "security-agent"


def test_agent_name_for_type_unknown():
    """Unknown type defaults to coder-agent."""
    assert agent_name_for_type("unknown") == "coder-agent"
    assert agent_name_for_type("invalid") == "coder-agent"
    assert agent_name_for_type("") == "coder-agent"


def test_agent_type_map_completeness():
    """AGENT_TYPE_MAP has all 9 expected entries."""
    assert AGENT_TYPE_MAP == {
        "file": "file-agent",
        "coder": "coder-agent",
        "test": "test-agent",
        "review": "review-agent",
        "web_search": "web-search-agent",
        "web_fetch": "web-fetch-agent",
        "planner": "planner-agent",
        "debug": "debug-agent",
        "security": "security-agent",
    }


# ── Agent registration tests ──────────────────────────────────


def test_all_specialists_registered():
    """All 9 specialist agents are registered via @agent decorator."""
    # Import the module to trigger decorator registration
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()

    # Check all 9 agents are registered
    assert "file-agent" in registry
    assert "coder-agent" in registry
    assert "test-agent" in registry
    assert "review-agent" in registry
    assert "web-search-agent" in registry
    assert "web-fetch-agent" in registry
    assert "planner-agent" in registry
    assert "debug-agent" in registry
    assert "security-agent" in registry


def test_file_agent_metadata():
    """FileAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["file-agent"]

    # Check metadata stored in _agent_meta
    meta = agent_class._agent_meta
    assert meta["name"] == "file-agent"
    assert meta["version"] == "1.0"
    assert "file-analysis" in meta["capabilities"]
    assert "codebase-exploration" in meta["capabilities"]


def test_coder_agent_metadata():
    """CoderAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["coder-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "coder-agent"
    assert meta["version"] == "1.0"
    assert "code-generation" in meta["capabilities"]
    assert "file-creation" in meta["capabilities"]


def test_test_agent_metadata():
    """TestAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["test-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "test-agent"
    assert meta["version"] == "1.0"
    assert "test-generation" in meta["capabilities"]
    assert "test-execution" in meta["capabilities"]


def test_review_agent_metadata():
    """ReviewAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["review-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "review-agent"
    assert meta["version"] == "1.0"
    assert "code-review" in meta["capabilities"]
    assert "quality-assurance" in meta["capabilities"]


def test_web_search_agent_metadata():
    """WebSearchAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["web-search-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "web-search-agent"
    assert meta["version"] == "1.0"
    assert "web-search" in meta["capabilities"]
    assert "information-retrieval" in meta["capabilities"]


def test_web_fetch_agent_metadata():
    """WebFetchAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["web-fetch-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "web-fetch-agent"
    assert meta["version"] == "1.0"
    assert "web-fetch" in meta["capabilities"]
    assert "content-extraction" in meta["capabilities"]


def test_planner_agent_metadata():
    """PlannerAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["planner-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "planner-agent"
    assert meta["version"] == "1.0"
    assert "planning" in meta["capabilities"]
    assert "decomposition" in meta["capabilities"]


def test_debug_agent_metadata():
    """DebugAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["debug-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "debug-agent"
    assert meta["version"] == "1.0"
    assert "debugging" in meta["capabilities"]
    assert "error-diagnosis" in meta["capabilities"]


def test_security_agent_metadata():
    """SecurityAgent has correct metadata."""
    import memfun_agent.specialists  # noqa: F401
    from memfun_runtime.agent import get_agent_registry

    registry = get_agent_registry()
    agent_class = registry["security-agent"]

    meta = agent_class._agent_meta
    assert meta["name"] == "security-agent"
    assert meta["version"] == "1.0"
    assert "security-review" in meta["capabilities"]
    assert "vulnerability-detection" in meta["capabilities"]


def test_specialist_classes_exist():
    """All 9 specialist classes can be imported."""
    from memfun_agent.specialists import (
        CoderAgent,
        DebugAgent,
        FileAgent,
        PlannerAgent,
        ReviewAgent,
        SecurityAgent,
        TestAgent,
        WebFetchAgent,
        WebSearchAgent,
    )

    assert FileAgent is not None
    assert CoderAgent is not None
    assert TestAgent is not None
    assert ReviewAgent is not None
    assert WebSearchAgent is not None
    assert WebFetchAgent is not None
    assert PlannerAgent is not None
    assert DebugAgent is not None
    assert SecurityAgent is not None


def test_specialist_type_constants():
    """Original specialist classes have correct type constants."""
    from memfun_agent.specialists import (
        CoderAgent,
        FileAgent,
        ReviewAgent,
        TestAgent,
    )

    assert FileAgent._SPECIALIST_TYPE == "file"
    assert CoderAgent._SPECIALIST_TYPE == "coder"
    assert TestAgent._SPECIALIST_TYPE == "test"
    assert ReviewAgent._SPECIALIST_TYPE == "review"


def test_new_specialist_type_constants():
    """New specialist classes have correct type constants."""
    from memfun_agent.specialists import (
        DebugAgent,
        PlannerAgent,
        SecurityAgent,
        WebFetchAgent,
        WebSearchAgent,
    )

    assert WebSearchAgent._SPECIALIST_TYPE == "web_search"
    assert WebFetchAgent._SPECIALIST_TYPE == "web_fetch"
    assert PlannerAgent._SPECIALIST_TYPE == "planner"
    assert DebugAgent._SPECIALIST_TYPE == "debug"
    assert SecurityAgent._SPECIALIST_TYPE == "security"


def test_specialist_max_iterations():
    """Original specialist classes have appropriate iteration limits."""
    from memfun_agent.specialists import (
        CoderAgent,
        FileAgent,
        ReviewAgent,
        TestAgent,
    )

    # File/review agents need fewer iterations (read-only)
    assert FileAgent._MAX_ITERATIONS == 8
    assert ReviewAgent._MAX_ITERATIONS == 8

    # Test agents need moderate iterations
    assert TestAgent._MAX_ITERATIONS == 10

    # Coder agents need most iterations
    assert CoderAgent._MAX_ITERATIONS == 15


def test_new_specialist_max_iterations():
    """New specialist classes have appropriate iteration limits."""
    from memfun_agent.specialists import (
        DebugAgent,
        PlannerAgent,
        SecurityAgent,
        WebFetchAgent,
        WebSearchAgent,
    )

    assert WebSearchAgent._MAX_ITERATIONS == 8
    assert WebFetchAgent._MAX_ITERATIONS == 8
    assert PlannerAgent._MAX_ITERATIONS == 6
    assert DebugAgent._MAX_ITERATIONS == 12
    assert SecurityAgent._MAX_ITERATIONS == 8


def test_specialist_system_prefixes_exist():
    """Original specialists have non-empty system prefixes."""
    from memfun_agent.specialists import (
        CoderAgent,
        FileAgent,
        ReviewAgent,
        TestAgent,
    )

    assert len(FileAgent._SYSTEM_PREFIX) > 50
    assert len(CoderAgent._SYSTEM_PREFIX) > 50
    assert len(TestAgent._SYSTEM_PREFIX) > 50
    assert len(ReviewAgent._SYSTEM_PREFIX) > 50


def test_new_specialist_system_prefixes_exist():
    """New specialists have non-empty system prefixes."""
    from memfun_agent.specialists import (
        DebugAgent,
        PlannerAgent,
        SecurityAgent,
        WebFetchAgent,
        WebSearchAgent,
    )

    assert len(WebSearchAgent._SYSTEM_PREFIX) > 50
    assert len(WebFetchAgent._SYSTEM_PREFIX) > 50
    assert len(PlannerAgent._SYSTEM_PREFIX) > 50
    assert len(DebugAgent._SYSTEM_PREFIX) > 50
    assert len(SecurityAgent._SYSTEM_PREFIX) > 50


def test_file_agent_prefix_content():
    """FileAgent prefix emphasizes read-only behavior."""
    from memfun_agent.specialists import FileAgent

    prefix = FileAgent._SYSTEM_PREFIX.lower()
    assert "read" in prefix or "analy" in prefix
    assert "do not write" in prefix or "never" in prefix


def test_coder_agent_prefix_content():
    """CoderAgent prefix emphasizes code generation."""
    from memfun_agent.specialists import CoderAgent

    prefix = CoderAgent._SYSTEM_PREFIX.lower()
    assert "write" in prefix or "generat" in prefix or "code" in prefix


def test_test_agent_prefix_content():
    """TestAgent prefix emphasizes testing."""
    from memfun_agent.specialists import TestAgent

    prefix = TestAgent._SYSTEM_PREFIX.lower()
    assert "test" in prefix
    assert "run" in prefix or "execut" in prefix


def test_review_agent_prefix_content():
    """ReviewAgent prefix emphasizes review."""
    from memfun_agent.specialists import ReviewAgent

    prefix = ReviewAgent._SYSTEM_PREFIX.lower()
    assert "review" in prefix
    assert "quality" in prefix or "consistency" in prefix


def test_web_search_agent_prefix_content():
    """WebSearchAgent prefix emphasizes web search."""
    from memfun_agent.specialists import WebSearchAgent

    prefix = WebSearchAgent._SYSTEM_PREFIX.lower()
    assert "search" in prefix
    assert "web" in prefix


def test_web_fetch_agent_prefix_content():
    """WebFetchAgent prefix emphasizes fetching URLs."""
    from memfun_agent.specialists import WebFetchAgent

    prefix = WebFetchAgent._SYSTEM_PREFIX.lower()
    assert "fetch" in prefix
    assert "url" in prefix


def test_planner_agent_prefix_content():
    """PlannerAgent prefix emphasizes planning without code writing."""
    from memfun_agent.specialists import PlannerAgent

    prefix = PlannerAgent._SYSTEM_PREFIX.lower()
    assert "plan" in prefix
    assert "do not write" in prefix or "do not" in prefix


def test_debug_agent_prefix_content():
    """DebugAgent prefix emphasizes debugging and root cause analysis."""
    from memfun_agent.specialists import DebugAgent

    prefix = DebugAgent._SYSTEM_PREFIX.lower()
    assert "debug" in prefix or "diagnos" in prefix
    assert "root cause" in prefix


def test_security_agent_prefix_content():
    """SecurityAgent prefix emphasizes security vulnerabilities."""
    from memfun_agent.specialists import SecurityAgent

    prefix = SecurityAgent._SYSTEM_PREFIX.lower()
    assert "security" in prefix
    assert "vulnerabilit" in prefix
