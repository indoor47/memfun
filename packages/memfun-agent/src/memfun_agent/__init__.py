"""Memfun Agent: RLM-powered autonomous coding agent.

Public API:
- ``RLMCodingAgent`` -- The main coding agent (BaseAgent subclass).
- ``RLMModule`` / ``RLMConfig`` / ``RLMResult`` -- Core RLM module.
- ``MCPToolBridge`` / ``create_tool_bridge`` -- MCP tool integration.
- ``TraceCollector`` / ``ExecutionTrace`` -- Trace collection.
- DSPy signatures: ``CodeAnalysis``, ``BugFix``, ``CodeReview``,
  ``CodeExplanation``, ``RLMExploration``.
- Agent definitions (AGENT.md): ``AgentDefinition``, ``AgentLoader``,
  ``AgentValidator``, ``AgentRegistryBridge``, ``DefinedAgent``,
  ``parse_agent_md``.
"""
from __future__ import annotations

from memfun_agent.coding_agent import RLMCodingAgent
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
from memfun_agent.rlm import (
    LocalREPL,
    RLMConfig,
    RLMModule,
    RLMResult,
    build_context_metadata,
)
from memfun_agent.signatures import (
    BugFix,
    CodeAnalysis,
    CodeExplanation,
    CodeReview,
    LearningExtraction,
    RLMExploration,
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

__all__ = [
    # Agent definitions (AGENT.md)
    "AgentDefinition",
    "AgentLoader",
    "AgentManifest",
    "AgentRegistryBridge",
    "AgentValidator",
    # Signatures
    "BugFix",
    "CodeAnalysis",
    "CodeExplanation",
    "CodeReview",
    # Defined agent
    "DefinedAgent",
    # Traces
    "ExecutionTrace",
    "LearningExtraction",
    # RLM core
    "LocalREPL",
    # Tool bridge
    "MCPToolBridge",
    # Agent
    "RLMCodingAgent",
    "RLMConfig",
    "RLMExploration",
    "RLMModule",
    "RLMResult",
    "TokenUsage",
    "ToolResult",
    "TraceCollector",
    "TraceStep",
    "agent_to_metadata",
    "build_context_metadata",
    "create_tool_bridge",
    "parse_agent_md",
]
