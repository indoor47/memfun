"""Unit tests for QueryResolver.

Tests deictic reference resolution for short/ambiguous user queries.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from memfun_agent.query_resolver import QueryResolver


@pytest.mark.asyncio
async def test_long_query_short_circuits():
    """Long queries (>150 chars) bypass resolution."""
    resolver = QueryResolver()
    long_query = "x" * 151
    history = [
        {"role": "user", "content": "previous message"},
        {"role": "assistant", "content": "previous response"},
    ]

    resolved, was_resolved = await resolver.aresolve(long_query, history)

    assert resolved == long_query
    assert was_resolved is False


@pytest.mark.asyncio
async def test_empty_history_short_circuits():
    """Queries with <2 history entries bypass resolution."""
    resolver = QueryResolver()
    query = "fix this"

    # Empty history
    resolved, was_resolved = await resolver.aresolve(query, [])
    assert resolved == query
    assert was_resolved is False

    # Single history entry
    resolved, was_resolved = await resolver.aresolve(query, [{"role": "user", "content": "test"}])
    assert resolved == query
    assert was_resolved is False


@pytest.mark.asyncio
async def test_resolution_with_mock():
    """Resolution calls DSPy with conversation context."""
    resolver = QueryResolver()
    query = "do it"
    history = [
        {"role": "user", "content": "Should I refactor the UserService class?"},
        {"role": "assistant", "content": "Yes, I recommend extracting the validation logic."},
    ]

    # Mock the DSPy Predict call
    mock_result = MagicMock()
    mock_result.resolved_query = "Refactor the UserService class by extracting validation logic"
    mock_result.was_resolved = True

    async def mock_to_thread(func, *args, **kwargs):
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        resolved, was_resolved = await resolver.aresolve(query, history)

    assert resolved == "Refactor the UserService class by extracting validation logic"
    assert was_resolved is True


@pytest.mark.asyncio
async def test_empty_resolved_fallback():
    """Empty resolved_query from LLM falls back to original."""
    resolver = QueryResolver()
    query = "test query"
    history = [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "response"},
    ]

    # Mock LLM returning empty string
    mock_result = MagicMock()
    mock_result.resolved_query = ""
    mock_result.was_resolved = True

    with patch.object(resolver.resolver, "__call__", return_value=mock_result):
        resolved, was_resolved = await resolver.aresolve(query, history)

    assert resolved == query
    assert was_resolved is False


@pytest.mark.asyncio
async def test_resolution_handles_exception():
    """LLM exceptions fall back to original query."""
    resolver = QueryResolver()
    query = "test"
    history = [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "response"},
    ]

    with patch.object(resolver.resolver, "__call__", side_effect=RuntimeError("LLM failed")):
        resolved, was_resolved = await resolver.aresolve(query, history)

    assert resolved == query
    assert was_resolved is False


@pytest.mark.asyncio
async def test_history_formatting_with_files():
    """History includes files_created from assistant responses."""
    resolver = QueryResolver()
    query = "test"
    history = [
        {"role": "user", "content": "Create a config file"},
        {
            "role": "assistant",
            "content": "Created config.yaml",
            "files_created": ["config.yaml", "settings.py"],
        },
    ]

    mock_result = MagicMock()
    mock_result.resolved_query = "resolved"
    mock_result.was_resolved = True

    captured_kwargs = {}

    async def mock_to_thread(func, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        await resolver.aresolve(query, history)

        # Check conversation_context includes files
        context = captured_kwargs["conversation_context"]
        assert "[files: config.yaml, settings.py]" in context


@pytest.mark.asyncio
async def test_history_formatting_with_ops():
    """History includes ops count from assistant responses."""
    resolver = QueryResolver()
    query = "test"
    history = [
        {"role": "user", "content": "Refactor the code"},
        {
            "role": "assistant",
            "content": "Refactored",
            "ops": [("write", "file.py", "content"), ("run", "pytest", "")],
        },
    ]

    mock_result = MagicMock()
    mock_result.resolved_query = "resolved"
    mock_result.was_resolved = True

    captured_kwargs = {}

    async def mock_to_thread(func, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        await resolver.aresolve(query, history)

        # Check conversation_context includes ops
        context = captured_kwargs["conversation_context"]
        assert "[ops: 2 operations]" in context


@pytest.mark.asyncio
async def test_history_truncation():
    """Long history entries are truncated to 2000 chars."""
    resolver = QueryResolver()
    query = "test"
    long_content = "x" * 3000
    history = [
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "short"},
    ]

    mock_result = MagicMock()
    mock_result.resolved_query = "resolved"
    mock_result.was_resolved = True

    captured_kwargs = {}

    async def mock_to_thread(func, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        await resolver.aresolve(query, history)

        # Check content was truncated
        context = captured_kwargs["conversation_context"]
        assert len(context) < len(long_content)
        assert "x" * 2000 in context


@pytest.mark.asyncio
async def test_last_four_history_entries_used():
    """Only last 4 history entries are used for context."""
    resolver = QueryResolver()
    query = "test"
    history = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "resp3"},
    ]

    mock_result = MagicMock()
    mock_result.resolved_query = "resolved"
    mock_result.was_resolved = True

    captured_kwargs = {}

    async def mock_to_thread(func, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        await resolver.aresolve(query, history)

        # Check only last 4 entries included
        context = captured_kwargs["conversation_context"]
        assert "msg1" not in context
        assert "resp1" not in context
        assert "msg2" in context
        assert "resp2" in context
        assert "msg3" in context
        assert "resp3" in context


@pytest.mark.asyncio
async def test_whitespace_stripping():
    """Query is stripped before length check."""
    resolver = QueryResolver()
    query = "  \n\tshort query\n  "
    history = [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "response"},
    ]

    mock_result = MagicMock()
    mock_result.resolved_query = "resolved short query"
    mock_result.was_resolved = True

    async def mock_to_thread(func, *args, **kwargs):
        return mock_result

    with patch("asyncio.to_thread", side_effect=mock_to_thread):
        resolved, was_resolved = await resolver.aresolve(query, history)

    assert resolved == "resolved short query"
    assert was_resolved is True
