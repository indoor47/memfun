"""Memfun Optimizer: Self-optimization via trace analysis and agent synthesis.

This package provides tools for:
- Trace analysis and pattern discovery
- Agent synthesis from discovered execution patterns
- MIPROv2 optimization of synthesized agents
- A/B testing framework for comparing agent performance
- Skill synthesis from execution patterns

Main Components:
    Phase 1 - Discovery:
        - DiscoveryPipeline: Orchestrates trace-to-agent discovery
        - StrategyExtractor: Classifies trace steps into strategy signatures
        - PatternAnalyzer: Identifies recurring patterns across signatures

    Phase 2 - Synthesis:
        - AgentSynthesizer: Converts DiscoveredAgent specs to executable code
        - SkillSynthesizer: Generates SKILL.md definitions from patterns

    Phase 3 - Optimization:
        - AgentOptimizer: MIPROv2 wrapper for refining agent modules

    Phase 4 - Evaluation:
        - ComparisonRunner: A/B testing for agent evaluation
"""
from __future__ import annotations

from memfun_optimizer.comparison import (
    AgentRunResult,
    ComparisonReport,
    ComparisonRunner,
)
from memfun_optimizer.discovery import DiscoveryPipeline
from memfun_optimizer.optimizer import (
    AgentOptimizer,
    OptimizationResult,
)
from memfun_optimizer.patterns import PatternAnalyzer
from memfun_optimizer.signatures import StrategyExtractor
from memfun_optimizer.skill_synthesis import SkillSynthesizer, SynthesizedSkill
from memfun_optimizer.synthesizer import (
    AgentSynthesizer,
    SynthesizedAgent,
)
from memfun_optimizer.types import (
    DiscoveredAgent,
    StepKind,
    StrategyPattern,
    StrategySignature,
    StrategyStep,
)

__all__ = [
    "AgentOptimizer",
    "AgentRunResult",
    "AgentSynthesizer",
    "ComparisonReport",
    "ComparisonRunner",
    "DiscoveredAgent",
    "DiscoveryPipeline",
    "OptimizationResult",
    "PatternAnalyzer",
    "SkillSynthesizer",
    "StepKind",
    "StrategyExtractor",
    "StrategyPattern",
    "StrategySignature",
    "StrategyStep",
    "SynthesizedAgent",
    "SynthesizedSkill",
]
