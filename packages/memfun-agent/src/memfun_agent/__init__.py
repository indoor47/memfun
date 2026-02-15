"""Memfun Agent: RLM-powered autonomous coding agent.

Public API:
- ``RLMCodingAgent`` -- The main coding agent (BaseAgent subclass).
- ``RLMModule`` / ``RLMConfig`` / ``RLMResult`` -- Core RLM module.
- ``ContextFirstSolver`` / ``ContextFirstConfig`` -- Context-first solving.
- ``build_code_map`` / ``code_map_to_string`` -- Code structure indexing.
- ``MCPToolBridge`` / ``create_tool_bridge`` -- MCP tool integration.
- ``TraceCollector`` / ``ExecutionTrace`` -- Trace collection.
- DSPy signatures: ``CodeAnalysis``, ``BugFix``, ``CodeReview``,
  ``CodeExplanation``, ``RLMExploration``, ``QueryResolution``,
  ``TaskDecomposition``, ``ContextPlanning``, ``SingleShotSolving``.
- Agent definitions (AGENT.md): ``AgentDefinition``, ``AgentLoader``,
  ``AgentValidator``, ``AgentRegistryBridge``, ``DefinedAgent``,
  ``parse_agent_md``.
- Multi-agent: ``QueryResolver``, ``TaskDecomposer``, ``SubTask``,
  ``DecompositionResult``, ``SharedSpec``, ``SharedSpecStore``,
  ``WorkflowEngine``, ``WorkflowResult``, ``WorkflowState``.
- Specialist agents: ``FileAgent``, ``CoderAgent``, ``TestAgent``,
  ``ReviewAgent``, ``WebSearchAgent``, ``WebFetchAgent``,
  ``PlannerAgent``, ``DebugAgent``, ``SecurityAgent``.
"""
from __future__ import annotations

from memfun_agent.code_map import (
    Definition,
    FileMap,
    build_code_map,
    code_map_to_string,
)
from memfun_agent.coding_agent import RLMCodingAgent
from memfun_agent.context_first import (
    ConsistencyResult,
    ConsistencyReviewer,
    ContextFirstConfig,
    ContextFirstResult,
    ContextFirstSolver,
    EditDiagnostic,
)
from memfun_agent.decomposer import (
    DecompositionResult,
    SubTask,
    TaskDecomposer,
)
from memfun_agent.definitions import (
    AgentDefinition,
    AgentLoader,
    AgentManifest,
    AgentRegistryBridge,
    AgentValidator,
    DefinedAgent,
    agent_to_metadata,
    parse_agent_md,
)
from memfun_agent.query_resolver import QueryResolver
from memfun_agent.rlm import (
    LocalREPL,
    RLMConfig,
    RLMModule,
    RLMResult,
    build_context_metadata,
)
from memfun_agent.shared_spec import SharedSpec, SharedSpecStore
from memfun_agent.signatures import (
    BugFix,
    CodeAnalysis,
    CodeExplanation,
    CodeReview,
    ConsistencyReview,
    ContextPlanning,
    LearningExtraction,
    QueryResolution,
    RLMExploration,
    SingleShotSolving,
    TaskDecomposition,
    VerificationFix,
)
from memfun_agent.specialists import (
    AGENT_ACTIVITY,
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
from memfun_agent.tool_bridge import (
    MCPToolBridge,
    ToolResult,
    create_tool_bridge,
)
from memfun_agent.traces import (
    ExecutionTrace,
    TokenUsage,
    TraceCollector,
    TraceStep,
)
from memfun_agent.workflow import (
    WorkflowEngine,
    WorkflowResult,
    WorkflowState,
)

__all__ = [
    "AGENT_ACTIVITY",
    "AgentDefinition",
    "AgentLoader",
    "AgentManifest",
    "AgentRegistryBridge",
    "AgentValidator",
    "BugFix",
    "CodeAnalysis",
    "CodeExplanation",
    "CodeReview",
    "CoderAgent",
    "ConsistencyResult",
    "ConsistencyReview",
    "ConsistencyReviewer",
    "ContextFirstConfig",
    "ContextFirstResult",
    "ContextFirstSolver",
    "ContextPlanning",
    "DebugAgent",
    "DecompositionResult",
    "DefinedAgent",
    "Definition",
    "EditDiagnostic",
    "ExecutionTrace",
    "FileAgent",
    "FileMap",
    "LearningExtraction",
    "LocalREPL",
    "MCPToolBridge",
    "PlannerAgent",
    "QueryResolution",
    "QueryResolver",
    "RLMCodingAgent",
    "RLMConfig",
    "RLMExploration",
    "RLMModule",
    "RLMResult",
    "ReviewAgent",
    "SecurityAgent",
    "SharedSpec",
    "SharedSpecStore",
    "SingleShotSolving",
    "SubTask",
    "TaskDecomposer",
    "TaskDecomposition",
    "TestAgent",
    "TokenUsage",
    "ToolResult",
    "TraceCollector",
    "TraceStep",
    "VerificationFix",
    "WebFetchAgent",
    "WebSearchAgent",
    "WorkflowEngine",
    "WorkflowResult",
    "WorkflowState",
    "agent_to_metadata",
    "build_code_map",
    "build_context_metadata",
    "code_map_to_string",
    "create_tool_bridge",
    "parse_agent_md",
]
