"""Tests for memfun-optimizer package: signatures, patterns, discovery,
synthesizer, skill synthesis, memory store, and memory search.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from memfun_agent.traces import ExecutionTrace, TokenUsage, TraceStep
from memfun_optimizer.discovery import DiscoveryPipeline
from memfun_optimizer.memory.search import MemorySearch
from memfun_optimizer.memory.store import MemoryStore
from memfun_optimizer.memory.types import MemoryEntry, MemoryQuery
from memfun_optimizer.patterns import PatternAnalyzer
from memfun_optimizer.signatures import StrategyExtractor
from memfun_optimizer.skill_synthesis import SkillSynthesizer
from memfun_optimizer.synthesizer import AgentSynthesizer
from memfun_optimizer.types import (
    DiscoveredAgent,
    StepKind,
    StrategyPattern,
    StrategySignature,
    StrategyStep,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_trace_step(
    iteration: int,
    code: str,
    *,
    reasoning: str = "thinking",
    output: str = "",
) -> TraceStep:
    return TraceStep(
        iteration=iteration,
        reasoning=reasoning,
        code=code,
        output=output,
        duration_ms=10.0,
    )


def _make_trace(
    *,
    task_type: str = "analyze",
    query: str = "test query",
    steps: list[TraceStep] | None = None,
    success: bool = True,
    duration_ms: float = 100.0,
    total_tokens: int = 500,
) -> ExecutionTrace:
    return ExecutionTrace(
        task_type=task_type,
        query=query,
        trajectory=steps or [],
        success=success,
        duration_ms=duration_ms,
        token_usage=TokenUsage(total_tokens=total_tokens),
    )


def _make_signature(
    *,
    step_kinds: list[StepKind],
    trace_id: str = "",
    task_type: str = "analyze",
    success: bool = True,
    duration_ms: float = 100.0,
    token_cost: int = 500,
) -> StrategySignature:
    steps = [
        StrategyStep(
            kind=kind,
            iteration=i + 1,
            confidence=0.9,
            raw_code="",
            raw_reasoning="",
        )
        for i, kind in enumerate(step_kinds)
    ]
    return StrategySignature(
        trace_id=trace_id or f"trace-{id(steps)}",
        task_type=task_type,
        steps=steps,
        total_iterations=len(steps),
        success=success,
        duration_ms=duration_ms,
        token_cost=token_cost,
    )


def _make_pattern(
    *,
    step_sequence: list[StepKind] | None = None,
    frequency: int = 5,
    confidence: float = 0.85,
    success_rate: float = 0.9,
) -> StrategyPattern:
    seq = step_sequence or [StepKind.PEEK, StepKind.GREP, StepKind.SUBMIT]
    return StrategyPattern(
        pattern_id="test-pattern-001",
        step_sequence=seq,
        frequency=frequency,
        confidence=confidence,
        avg_duration_ms=120.0,
        avg_token_cost=450.0,
        success_rate=success_rate,
        example_trace_ids=["t1", "t2"],
    )


def _make_discovered_agent(
    *,
    name: str = "test-agent",
    task_type: str = "analyze",
    stages: list[dict] | None = None,
    pattern: StrategyPattern | None = None,
) -> DiscoveredAgent:
    return DiscoveredAgent(
        name=name,
        description=f"A test agent for {task_type}",
        task_type=task_type,
        source_pattern=pattern or _make_pattern(),
        stages=stages or [
            {"name": "Read", "kind": "PEEK", "description": "Read source files"},
            {"name": "Query", "kind": "QUERY", "description": "Analyze with LLM"},
        ],
        estimated_speedup=2.5,
        estimated_cost_reduction=0.5,
    )


# ── 1. StrategyExtractor Tests ───────────────────────────────


class TestStrategyExtractor:
    """Tests for StrategyExtractor: classify_step and extract."""

    def test_classify_step_peek(self) -> None:
        """Code containing open() is classified as PEEK."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'f = open("file.txt")')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.PEEK
        assert result.confidence >= 0.85

    def test_classify_step_peek_read_text(self) -> None:
        """Code with .read_text() is classified as PEEK."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'content = path.read_text()')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.PEEK

    def test_classify_step_grep(self) -> None:
        """Code with re.findall() is classified as GREP."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'matches = re.findall(r"\\w+", text)')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.GREP
        assert result.confidence >= 0.85

    def test_classify_step_grep_re_search(self) -> None:
        """Code with re.search() is classified as GREP."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'match = re.search(r"pattern", text)')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.GREP

    def test_classify_step_search_glob(self) -> None:
        """Code with glob() is classified as SEARCH."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'files = glob("*.py")')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.SEARCH

    def test_classify_step_search_os_walk(self) -> None:
        """Code with os.walk() is classified as SEARCH."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'for root, dirs, files in os.walk("."): pass')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.SEARCH

    def test_classify_step_query(self) -> None:
        """Code with llm_query() is classified as QUERY."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'result = llm_query("what is this code?")')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.QUERY

    def test_classify_step_batch_query(self) -> None:
        """Code with llm_query_batched() is classified as BATCH_QUERY."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'results = llm_query_batched(items)')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.BATCH_QUERY

    def test_classify_step_submit(self) -> None:
        """Code setting state['FINAL'] is classified as SUBMIT."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, "state['FINAL'] = 'the answer'")
        result = extractor.classify_step(step)
        assert result.kind == StepKind.SUBMIT
        assert result.confidence >= 0.9

    def test_classify_step_partition(self) -> None:
        """Code with .split() is classified as PARTITION."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'parts = text.split("\\n")')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.PARTITION

    def test_classify_step_aggregate(self) -> None:
        """Code with .append() is classified as AGGREGATE."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'results.append(item)')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.AGGREGATE

    def test_classify_step_transform(self) -> None:
        """Code with json.dumps() is classified as TRANSFORM."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'output = json.dumps(data)')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.TRANSFORM

    def test_classify_step_other(self) -> None:
        """Unrecognized code is classified as OTHER with low confidence."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 'x = 42')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.OTHER
        assert result.confidence == pytest.approx(0.1)

    def test_classify_step_preserves_iteration(self) -> None:
        """Classified step keeps the iteration number from the trace step."""
        extractor = StrategyExtractor()
        step = _make_trace_step(7, 'f = open("x.py")')
        result = extractor.classify_step(step)
        assert result.iteration == 7

    def test_classify_step_preserves_raw_fields(self) -> None:
        """Classified step keeps raw_code and raw_reasoning."""
        extractor = StrategyExtractor()
        step = _make_trace_step(1, 're.findall("a", text)', reasoning="searching")
        result = extractor.classify_step(step)
        assert result.raw_code == 're.findall("a", text)'
        assert result.raw_reasoning == "searching"

    def test_classify_step_highest_confidence_wins(self) -> None:
        """When multiple patterns match, the highest confidence one wins."""
        extractor = StrategyExtractor()
        # llm_query has 0.95 confidence, higher than .replace which is 0.6
        step = _make_trace_step(1, 'result = llm_query(text.replace("x", "y"))')
        result = extractor.classify_step(step)
        assert result.kind == StepKind.QUERY

    def test_extract_full_trace(self) -> None:
        """Extract produces a StrategySignature from a complete trace."""
        extractor = StrategyExtractor()
        steps = [
            _make_trace_step(1, 'f = open("file.txt")'),
            _make_trace_step(2, 'matches = re.findall(r"def", content)'),
            _make_trace_step(3, "state['FINAL'] = 'found 5 functions'"),
        ]
        trace = _make_trace(steps=steps, total_tokens=700)

        sig = extractor.extract(trace)
        assert sig.trace_id == trace.trace_id
        assert sig.task_type == "analyze"
        assert sig.total_iterations == 3
        assert sig.success is True
        assert sig.token_cost == 700
        assert len(sig.steps) == 3
        assert sig.steps[0].kind == StepKind.PEEK
        assert sig.steps[1].kind == StepKind.GREP
        assert sig.steps[2].kind == StepKind.SUBMIT

    def test_extract_empty_trace(self) -> None:
        """Extract handles a trace with no steps gracefully."""
        extractor = StrategyExtractor()
        trace = _make_trace(steps=[])
        sig = extractor.extract(trace)
        assert sig.total_iterations == 0
        assert sig.steps == []

    def test_extract_batch(self) -> None:
        """extract_batch produces one signature per trace."""
        extractor = StrategyExtractor()
        traces = [
            _make_trace(steps=[_make_trace_step(1, 'open("a")')]),
            _make_trace(steps=[_make_trace_step(1, 're.search("p", t)')]),
        ]
        sigs = extractor.extract_batch(traces)
        assert len(sigs) == 2
        assert sigs[0].steps[0].kind == StepKind.PEEK
        assert sigs[1].steps[0].kind == StepKind.GREP


# ── 2. PatternAnalyzer Tests ─────────────────────────────────


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer: analyze, find_dominant, rank."""

    def test_analyze_empty(self) -> None:
        """Analyze returns empty list for no signatures."""
        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze([])
        assert patterns == []

    def test_analyze_single_signature(self) -> None:
        """A single signature produces a single pattern with frequency 1."""
        analyzer = PatternAnalyzer()
        sig = _make_signature(step_kinds=[StepKind.PEEK, StepKind.SUBMIT])
        patterns = analyzer.analyze([sig])
        assert len(patterns) == 1
        assert patterns[0].frequency == 1

    def test_analyze_clusters_similar_signatures(self) -> None:
        """Similar signatures are clustered into the same pattern."""
        analyzer = PatternAnalyzer(similarity_threshold=0.7)
        # Create 3 signatures with identical step sequences
        sigs = [
            _make_signature(
                step_kinds=[StepKind.PEEK, StepKind.GREP, StepKind.SUBMIT],
                trace_id=f"t-{i}",
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(sigs)
        assert len(patterns) == 1
        assert patterns[0].frequency == 3

    def test_analyze_separates_different_signatures(self) -> None:
        """Dissimilar signatures form separate clusters."""
        analyzer = PatternAnalyzer(similarity_threshold=0.9)
        sig_a = _make_signature(
            step_kinds=[StepKind.PEEK, StepKind.GREP, StepKind.SUBMIT],
            trace_id="t-a",
        )
        sig_b = _make_signature(
            step_kinds=[StepKind.QUERY, StepKind.BATCH_QUERY, StepKind.AGGREGATE],
            trace_id="t-b",
        )
        patterns = analyzer.analyze([sig_a, sig_b])
        assert len(patterns) == 2

    def test_analyze_computes_success_rate(self) -> None:
        """Pattern success rate reflects constituent signatures."""
        analyzer = PatternAnalyzer()
        sigs = [
            _make_signature(
                step_kinds=[StepKind.PEEK, StepKind.SUBMIT],
                trace_id=f"t-{i}",
                success=(i < 3),  # 3 successes, 1 failure
            )
            for i in range(4)
        ]
        patterns = analyzer.analyze(sigs)
        assert len(patterns) == 1
        assert patterns[0].success_rate == pytest.approx(0.75)

    def test_analyze_computes_avg_duration(self) -> None:
        """Pattern avg_duration_ms averages constituent durations."""
        analyzer = PatternAnalyzer()
        sigs = [
            _make_signature(
                step_kinds=[StepKind.PEEK],
                trace_id=f"t-{i}",
                duration_ms=float(100 * (i + 1)),
            )
            for i in range(4)
        ]
        patterns = analyzer.analyze(sigs)
        assert len(patterns) == 1
        assert patterns[0].avg_duration_ms == pytest.approx(250.0)

    def test_find_dominant_filters_by_frequency(self) -> None:
        """find_dominant excludes patterns below min_frequency."""
        analyzer = PatternAnalyzer()
        patterns = [
            _make_pattern(frequency=5, confidence=0.9),
            _make_pattern(frequency=1, confidence=0.9),
        ]
        dominant = analyzer.find_dominant(patterns, min_frequency=3)
        assert len(dominant) == 1
        assert dominant[0].frequency == 5

    def test_find_dominant_filters_by_confidence(self) -> None:
        """find_dominant excludes patterns below min_confidence."""
        analyzer = PatternAnalyzer()
        patterns = [
            _make_pattern(frequency=5, confidence=0.9),
            _make_pattern(frequency=5, confidence=0.3),
        ]
        dominant = analyzer.find_dominant(patterns, min_confidence=0.5)
        assert len(dominant) == 1
        assert dominant[0].confidence == 0.9

    def test_find_dominant_empty_when_none_qualify(self) -> None:
        """find_dominant returns empty list when nothing qualifies."""
        analyzer = PatternAnalyzer()
        patterns = [_make_pattern(frequency=1, confidence=0.1)]
        dominant = analyzer.find_dominant(
            patterns, min_frequency=10, min_confidence=0.9
        )
        assert dominant == []

    def test_rank_by_composite_score(self) -> None:
        """rank sorts patterns by frequency * confidence * success_rate."""
        analyzer = PatternAnalyzer()
        p_low = _make_pattern(frequency=2, confidence=0.5, success_rate=0.5)
        p_high = _make_pattern(frequency=10, confidence=0.9, success_rate=0.95)
        ranked = analyzer.rank([p_low, p_high])
        assert ranked[0] is p_high
        assert ranked[1] is p_low

    def test_rank_empty(self) -> None:
        """rank handles an empty list gracefully."""
        analyzer = PatternAnalyzer()
        assert analyzer.rank([]) == []


# ── 3. DiscoveryPipeline Tests ───────────────────────────────


class TestDiscoveryPipeline:
    """Tests for DiscoveryPipeline with mock TraceCollector."""

    async def test_run_insufficient_traces(self) -> None:
        """Returns empty when fewer traces than min_traces."""
        collector = AsyncMock()
        collector.list_trace_ids = AsyncMock(return_value=["t1"])
        collector.load = AsyncMock(return_value=_make_trace())

        pipeline = DiscoveryPipeline(collector, min_traces=5)
        discovered = await pipeline.run()
        assert discovered == []

    async def test_run_with_matching_traces(self) -> None:
        """Discovers agents when enough similar traces are available."""
        # Build 5 traces with the same PEEK -> GREP -> SUBMIT pattern
        traces = [
            _make_trace(
                task_type="code_analysis",
                steps=[
                    _make_trace_step(1, 'f = open("file.py")'),
                    _make_trace_step(2, 're.findall(r"class", content)'),
                    _make_trace_step(3, "state['FINAL'] = 'done'"),
                ],
            )
            for _ in range(5)
        ]
        trace_ids = [t.trace_id for t in traces]
        trace_map = dict(zip(trace_ids, traces, strict=True))

        collector = AsyncMock()
        collector.list_trace_ids = AsyncMock(return_value=trace_ids)
        collector.load = AsyncMock(side_effect=lambda tid: trace_map.get(tid))

        pipeline = DiscoveryPipeline(
            collector,
            min_traces=3,
            min_frequency=2,
            min_confidence=0.3,
        )
        discovered = await pipeline.run(task_type="code_analysis")
        assert len(discovered) >= 1
        assert all(isinstance(a, DiscoveredAgent) for a in discovered)
        assert all(a.task_type == "code_analysis" for a in discovered)

    async def test_run_filters_by_task_type(self) -> None:
        """Only traces matching the task_type filter are used."""
        traces = [
            _make_trace(task_type="analyze"),
            _make_trace(task_type="review"),
            _make_trace(task_type="analyze"),
        ]
        trace_ids = [t.trace_id for t in traces]
        trace_map = dict(zip(trace_ids, traces, strict=True))

        collector = AsyncMock()
        collector.list_trace_ids = AsyncMock(return_value=trace_ids)
        collector.load = AsyncMock(side_effect=lambda tid: trace_map.get(tid))

        pipeline = DiscoveryPipeline(collector, min_traces=3)
        # Only 2 "analyze" traces, so below min_traces=3
        discovered = await pipeline.run(task_type="analyze")
        assert discovered == []

    async def test_run_no_dominant_patterns(self) -> None:
        """Returns empty when patterns exist but none are dominant."""
        # All traces have different step sequences
        traces = [
            _make_trace(
                steps=[_make_trace_step(1, 'open("a")')],
            ),
            _make_trace(
                steps=[_make_trace_step(1, 're.findall("x", t)')],
            ),
            _make_trace(
                steps=[_make_trace_step(1, 'glob("*.py")')],
            ),
        ]
        trace_ids = [t.trace_id for t in traces]
        trace_map = dict(zip(trace_ids, traces, strict=True))

        collector = AsyncMock()
        collector.list_trace_ids = AsyncMock(return_value=trace_ids)
        collector.load = AsyncMock(side_effect=lambda tid: trace_map.get(tid))

        pipeline = DiscoveryPipeline(
            collector,
            min_traces=3,
            min_frequency=3,  # Needs 3 traces in same cluster
        )
        discovered = await pipeline.run()
        assert discovered == []

    async def test_discovered_agent_has_stages(self) -> None:
        """Discovered agents have stage definitions from the pattern."""
        traces = [
            _make_trace(
                task_type="fix",
                steps=[
                    _make_trace_step(1, 'f = open("bug.py")'),
                    _make_trace_step(2, 'f = open("bug.py")'),
                    _make_trace_step(3, "state['FINAL'] = 'fixed'"),
                ],
            )
            for _ in range(5)
        ]
        trace_ids = [t.trace_id for t in traces]
        trace_map = dict(zip(trace_ids, traces, strict=True))

        collector = AsyncMock()
        collector.list_trace_ids = AsyncMock(return_value=trace_ids)
        collector.load = AsyncMock(side_effect=lambda tid: trace_map.get(tid))

        pipeline = DiscoveryPipeline(
            collector, min_traces=3, min_frequency=2, min_confidence=0.3
        )
        discovered = await pipeline.run(task_type="fix")
        assert len(discovered) >= 1
        agent = discovered[0]
        assert len(agent.stages) > 0
        assert agent.estimated_speedup > 1.0
        assert agent.estimated_cost_reduction > 0.0


# ── 4. AgentSynthesizer Tests ────────────────────────────────


class TestAgentSynthesizer:
    """Tests for AgentSynthesizer: synthesize and helpers."""

    def test_synthesize_produces_source_and_md(self) -> None:
        """synthesize() returns SynthesizedAgent with source_code and agent_md."""
        synthesizer = AgentSynthesizer()
        agent = _make_discovered_agent()
        result = synthesizer.synthesize(agent)

        assert result.name == "test-agent"
        assert result.task_type == "analyze"
        assert "class TestAgent" in result.source_code
        assert "dspy.Module" in result.source_code
        assert "---" in result.agent_md
        assert "name:" in result.agent_md
        assert "test-agent" in result.agent_md

    def test_synthesize_source_has_stages(self) -> None:
        """Generated source code contains stage methods."""
        synthesizer = AgentSynthesizer()
        agent = _make_discovered_agent(stages=[
            {"name": "Read", "kind": "PEEK", "description": "Read files"},
            {"name": "Analyze", "kind": "QUERY", "description": "Run LLM"},
            {"name": "Format", "kind": "TRANSFORM", "description": "Format output"},
        ])
        result = synthesizer.synthesize(agent)

        assert "_stage_1" in result.source_code
        assert "_stage_2" in result.source_code
        assert "_stage_3" in result.source_code

    def test_synthesize_agent_md_has_frontmatter(self) -> None:
        """Generated AGENT.md contains valid YAML frontmatter."""
        synthesizer = AgentSynthesizer()
        agent = _make_discovered_agent()
        result = synthesizer.synthesize(agent)

        assert result.agent_md.startswith("---")
        assert "version: 0.1.0" in result.agent_md
        assert "synthesized" in result.agent_md

    def test_synthesize_raises_on_empty_name(self) -> None:
        """synthesize() raises ValueError when name is empty."""
        synthesizer = AgentSynthesizer()
        agent = MagicMock()
        agent.name = ""
        agent.task_type = "analyze"

        with pytest.raises(ValueError, match="name and task_type"):
            synthesizer.synthesize(agent)

    def test_synthesize_raises_on_no_stages(self) -> None:
        """synthesize() raises ValueError when agent has no stages."""
        synthesizer = AgentSynthesizer()
        agent = MagicMock()
        agent.name = "test"
        agent.task_type = "fix"
        agent.stages = []

        with pytest.raises(ValueError, match="no stages"):
            synthesizer.synthesize(agent)

    def test_class_name_conversion(self) -> None:
        """_class_name converts hyphenated names to PascalCase."""
        assert AgentSynthesizer._class_name("code-analyzer") == "CodeAnalyzer"
        assert AgentSynthesizer._class_name("my-agent") == "MyAgent"
        assert AgentSynthesizer._class_name("single") == "Single"

    def test_synthesize_discovered_from_reference(self) -> None:
        """SynthesizedAgent carries reference to source DiscoveredAgent."""
        synthesizer = AgentSynthesizer()
        agent = _make_discovered_agent()
        result = synthesizer.synthesize(agent)
        assert result.discovered_from is agent


# ── 5. SkillSynthesizer Tests ────────────────────────────────


class TestSkillSynthesizer:
    """Tests for SkillSynthesizer: synthesize from execution traces."""

    def _make_skill_traces(
        self,
        count: int = 5,
        *,
        success: bool = True,
    ) -> list[ExecutionTrace]:
        """Build a list of traces suitable for skill synthesis."""
        return [
            _make_trace(
                task_type="code-review",
                query="review the code for bugs and style issues",
                steps=[
                    _make_trace_step(
                        1,
                        'content = read("main.py")',
                        reasoning="Reading the source file to review",
                    ),
                    _make_trace_step(
                        2,
                        'matches = grep(r"TODO", content)',
                        reasoning="Searching for TODO comments",
                    ),
                    _make_trace_step(
                        3,
                        "state['FINAL'] = 'Review complete'",
                        reasoning="Submitting final review",
                    ),
                ],
                success=success,
            )
            for _ in range(count)
        ]

    def test_synthesize_returns_skill_when_pattern_exists(self) -> None:
        """synthesize returns a SynthesizedSkill for consistent traces."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        traces = self._make_skill_traces(5)
        skill = synth.synthesize(traces, task_type="code-review")

        assert skill is not None
        assert skill.task_type == "code-review"
        assert "code-review" in skill.name
        assert skill.confidence > 0.0
        assert skill.confidence <= 1.0
        assert len(skill.source_trace_ids) == 5

    def test_synthesize_returns_none_insufficient_traces(self) -> None:
        """Returns None when trace count is below min_pattern_frequency."""
        synth = SkillSynthesizer(min_pattern_frequency=10)
        traces = self._make_skill_traces(3)
        skill = synth.synthesize(traces, task_type="code-review")
        assert skill is None

    def test_synthesize_returns_none_low_success_rate(self) -> None:
        """Returns None when success rate is too low for dominant pattern."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        traces = self._make_skill_traces(5, success=False)
        skill = synth.synthesize(traces, task_type="code-review")
        assert skill is None

    def test_synthesize_skill_md_has_frontmatter(self) -> None:
        """Generated skill MD has YAML frontmatter with required fields."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        traces = self._make_skill_traces(5)
        skill = synth.synthesize(traces, task_type="code-review")

        assert skill is not None
        assert skill.skill_md_content.lstrip().startswith("---")
        assert "name:" in skill.skill_md_content
        assert "description:" in skill.skill_md_content
        assert "version:" in skill.skill_md_content

    def test_synthesize_skill_md_contains_instructions(self) -> None:
        """Generated skill MD body contains synthesized instructions."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        traces = self._make_skill_traces(5)
        skill = synth.synthesize(traces, task_type="code-review")

        assert skill is not None
        assert "Synthesized Strategy" in skill.skill_md_content
        assert "success rate" in skill.skill_md_content.lower()

    def test_synthesize_confidence_increases_with_traces(self) -> None:
        """More traces should yield higher confidence (all else equal)."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        skill_5 = synth.synthesize(
            self._make_skill_traces(5), task_type="code-review"
        )
        skill_15 = synth.synthesize(
            self._make_skill_traces(15), task_type="code-review"
        )
        assert skill_5 is not None
        assert skill_15 is not None
        assert skill_15.confidence >= skill_5.confidence

    def test_synthesize_returns_none_no_common_tools(self) -> None:
        """Returns None when traces lack recognizable tool patterns."""
        synth = SkillSynthesizer(min_pattern_frequency=3)
        # Traces with no tool keywords in code/reasoning
        traces = [
            _make_trace(
                task_type="misc",
                query="do something completely abstract",
                steps=[
                    _make_trace_step(
                        1, "x = 42", reasoning="calculating"
                    ),
                ],
                success=True,
            )
            for _ in range(5)
        ]
        skill = synth.synthesize(traces, task_type="misc")
        assert skill is None


# ── 6. MemoryStore Tests ─────────────────────────────────────


class TestMemoryStore:
    """Tests for MemoryStore using in-memory mode (no backend)."""

    async def test_add_and_get(self) -> None:
        """Add an entry and retrieve it by ID."""
        store = MemoryStore()
        entry_id = await store.add(
            topic="bugs",
            content="Null pointer in async handler",
            source="trace",
            confidence=0.9,
            tags=("async", "bugs"),
        )

        entry = await store.get(entry_id)
        assert entry is not None
        assert entry.topic == "bugs"
        assert entry.content == "Null pointer in async handler"
        assert entry.source == "trace"
        assert entry.confidence == pytest.approx(0.9)
        assert "async" in entry.tags
        assert "bugs" in entry.tags

    async def test_get_nonexistent(self) -> None:
        """Getting a nonexistent entry returns None."""
        store = MemoryStore()
        assert await store.get("deadbeef0123456789abcdef01234567") is None

    async def test_update_content(self) -> None:
        """Update only the content field of an entry."""
        store = MemoryStore()
        entry_id = await store.add(
            topic="perf", content="Original content"
        )
        updated = await store.update(entry_id, content="Updated content")

        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.topic == "perf"  # unchanged

    async def test_update_confidence(self) -> None:
        """Update only the confidence field."""
        store = MemoryStore()
        entry_id = await store.add(topic="a", content="b", confidence=0.5)
        updated = await store.update(entry_id, confidence=0.95)

        assert updated is not None
        assert updated.confidence == pytest.approx(0.95)

    async def test_update_tags(self) -> None:
        """Update only the tags field."""
        store = MemoryStore()
        entry_id = await store.add(
            topic="a", content="b", tags=("old",)
        )
        updated = await store.update(entry_id, tags=("new", "tag"))

        assert updated is not None
        assert updated.tags == ("new", "tag")

    async def test_update_nonexistent(self) -> None:
        """Updating a nonexistent entry returns None."""
        store = MemoryStore()
        assert await store.update("deadbeef0123456789abcdef01234567", content="x") is None

    async def test_update_refreshes_updated_at(self) -> None:
        """update() refreshes the updated_at timestamp."""
        store = MemoryStore()
        entry_id = await store.add(topic="a", content="b")
        original = await store.get(entry_id)
        assert original is not None

        updated = await store.update(entry_id, content="new")
        assert updated is not None
        assert updated.updated_at >= original.updated_at

    async def test_delete(self) -> None:
        """Delete an existing entry."""
        store = MemoryStore()
        entry_id = await store.add(topic="temp", content="will delete")
        assert await store.delete(entry_id) is True
        assert await store.get(entry_id) is None

    async def test_delete_nonexistent(self) -> None:
        """Deleting a nonexistent entry returns False."""
        store = MemoryStore()
        assert await store.delete("deadbeef0123456789abcdef01234567") is False

    async def test_list_entries_all(self) -> None:
        """list_entries returns all entries without a filter."""
        store = MemoryStore()
        await store.add(topic="a", content="one")
        await store.add(topic="b", content="two")
        await store.add(topic="a", content="three")

        entries = await store.list_entries()
        assert len(entries) == 3

    async def test_list_entries_filter_by_topic(self) -> None:
        """list_entries filters by topic when specified."""
        store = MemoryStore()
        await store.add(topic="bugs", content="bug 1")
        await store.add(topic="perf", content="perf note")
        await store.add(topic="bugs", content="bug 2")

        entries = await store.list_entries(topic="bugs")
        assert len(entries) == 2
        assert all(e.topic == "bugs" for e in entries)

    async def test_list_entries_respects_limit(self) -> None:
        """list_entries respects the limit parameter."""
        store = MemoryStore()
        for i in range(10):
            await store.add(topic="t", content=f"entry {i}")

        entries = await store.list_entries(limit=3)
        assert len(entries) == 3

    async def test_search_basic(self) -> None:
        """search() finds entries matching keywords in content."""
        store = MemoryStore()
        await store.add(
            topic="bugs",
            content="Null pointer dereference in async handler",
            tags=("async",),
        )
        await store.add(
            topic="perf",
            content="Cache invalidation is slow",
        )

        query = MemoryQuery(query="async null pointer")
        results = await store.search(query)
        assert len(results) >= 1
        assert results[0].entry.topic == "bugs"

    async def test_has_backend_false_without_store(self) -> None:
        """has_backend is False when no state store is provided."""
        store = MemoryStore()
        assert store.has_backend is False

    async def test_has_backend_true_with_store(self) -> None:
        """has_backend is True when a state store is provided."""
        mock_store = MagicMock()
        store = MemoryStore(state_store=mock_store)
        assert store.has_backend is True


# ── 7. MemorySearch Tests ────────────────────────────────────


class TestMemorySearch:
    """Tests for MemorySearch: TF-IDF scoring, stopwords, topic/tag matching."""

    def _make_entries(self) -> list[MemoryEntry]:
        """Build a set of memory entries for search testing."""
        return [
            MemoryEntry(
                id="e1",
                topic="bugs",
                content="Null pointer dereference in async handler",
                tags=("async", "bugs"),
                confidence=0.9,
            ),
            MemoryEntry(
                id="e2",
                topic="performance",
                content="Cache invalidation causes slow response times",
                tags=("cache", "performance"),
                confidence=0.8,
            ),
            MemoryEntry(
                id="e3",
                topic="bugs",
                content="Race condition in concurrent map access",
                tags=("concurrency", "bugs"),
                confidence=0.85,
            ),
            MemoryEntry(
                id="e4",
                topic="patterns",
                content="Observer pattern for event handling",
                tags=("design", "events"),
                confidence=0.7,
            ),
        ]

    def test_search_returns_relevant_results(self) -> None:
        """Search for 'null pointer' returns the bug entry."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="null pointer")
        results = search.search(query, entries)
        assert len(results) >= 1
        assert results[0].entry.id == "e1"

    def test_search_topic_match_boost(self) -> None:
        """Entries whose topic matches the query get a score boost."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="bugs")
        results = search.search(query, entries)
        # Both bug entries should be ranked high
        result_ids = [r.entry.id for r in results]
        assert "e1" in result_ids
        assert "e3" in result_ids

    def test_search_tag_filter(self) -> None:
        """Tag filter restricts results to entries with matching tags."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="slow", tags=["cache"])
        results = search.search(query, entries)
        assert len(results) >= 1
        assert all(
            any(t in r.entry.tags for t in ["cache"])
            for r in results
        )

    def test_search_topic_filter(self) -> None:
        """Topic filter restricts results to entries in specified topics."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="async", topics=["bugs"])
        results = search.search(query, entries)
        assert all(r.entry.topic == "bugs" for r in results)

    def test_search_min_confidence_filter(self) -> None:
        """Entries below min_confidence are excluded."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        # Only entries with confidence >= 0.85
        query = MemoryQuery(query="bugs", min_confidence=0.85)
        results = search.search(query, entries)
        assert all(r.entry.confidence >= 0.85 for r in results)

    def test_search_respects_limit(self) -> None:
        """Results are limited to the query's limit parameter."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="bugs handler pattern", limit=2)
        results = search.search(query, entries)
        assert len(results) <= 2

    def test_search_stopwords_ignored(self) -> None:
        """Stopwords in the query are ignored during scoring."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        # "the" and "in" and "a" are stopwords
        query_with_stops = MemoryQuery(query="the null pointer in a handler")
        query_without = MemoryQuery(query="null pointer handler")

        results_with = search.search(query_with_stops, entries)
        results_without = search.search(query_without, entries)

        # Both should return same top result
        assert len(results_with) >= 1
        assert len(results_without) >= 1
        assert results_with[0].entry.id == results_without[0].entry.id

    def test_search_tfidf_weighting(self) -> None:
        """Rarer terms get higher IDF weight, boosting unique matches."""
        # Entry with a unique term should score higher on that term
        entries = [
            MemoryEntry(
                id="common",
                topic="general",
                content="common word appears everywhere here",
                confidence=0.9,
            ),
            MemoryEntry(
                id="rare",
                topic="general",
                content="common word plus unique_identifier_xyz",
                confidence=0.9,
            ),
        ]
        search = MemorySearch()
        search.index(entries)

        # "unique_identifier_xyz" only appears in one entry
        query = MemoryQuery(query="unique_identifier_xyz")
        results = search.search(query, entries)
        assert len(results) >= 1
        assert results[0].entry.id == "rare"

    def test_search_no_results_for_unrelated_query(self) -> None:
        """A query with no matching terms returns empty results."""
        entries = self._make_entries()
        search = MemorySearch()
        search.index(entries)

        query = MemoryQuery(query="zzzznonexistent xyzabc")
        results = search.search(query, entries)
        assert results == []

    def test_search_confidence_weighting(self) -> None:
        """Score is multiplied by entry confidence, so higher confidence ranks first."""
        entries = [
            MemoryEntry(
                id="low_conf",
                topic="caching",
                content="important finding about caching strategies",
                confidence=0.3,
            ),
            MemoryEntry(
                id="high_conf",
                topic="caching",
                content="important finding about caching strategies",
                confidence=0.95,
            ),
            MemoryEntry(
                id="unrelated",
                topic="other",
                content="something completely different here",
                confidence=0.9,
            ),
        ]
        search = MemorySearch()
        search.index(entries)

        # "caching" matches the topic, giving a +10 boost to the two caching entries
        query = MemoryQuery(query="caching")
        results = search.search(query, entries)
        assert len(results) >= 2
        # high_conf entry should rank above low_conf because score is weighted by confidence
        caching_results = [r for r in results if "caching" in r.entry.topic]
        assert len(caching_results) == 2
        assert caching_results[0].entry.id == "high_conf"

    def test_index_then_search_empty_entries(self) -> None:
        """Indexing and searching empty entries list returns no results."""
        search = MemorySearch()
        search.index([])
        results = search.search(MemoryQuery(query="anything"), [])
        assert results == []

    def test_tokenize_removes_short_tokens(self) -> None:
        """Tokens of length 1 are filtered out."""
        search = MemorySearch()
        tokens = search._tokenize("I a am in the loop x")
        # "I", "a", "x" are length 1 and should be removed
        # "am", "in", "the" are stopwords
        assert "loop" in tokens
        assert "i" not in tokens
        assert "a" not in tokens
        assert "x" not in tokens
