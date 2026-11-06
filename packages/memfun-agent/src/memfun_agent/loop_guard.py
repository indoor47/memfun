"""Abort an LLM call early when its output degenerates into a repetition loop.

A greedy-decoded 27B model can fall into an endless loop (``re, reall, re_first, re, ...``
for 85 000 characters) and only stop at ``max_tokens``: ten minutes of GPU for a response
that is garbage.  This module watches the streamed text and cuts the call as soon as the
tail is the same block repeated many times.

How it hooks in: :func:`install_loop_guard` wraps ``litellm.completion`` /
``litellm.acompletion`` (what DSPy's LiteLLM engine calls).  For OpenAI-compatible calls
to a custom ``api_base`` that did not ask for streaming, the wrapper streams, feeds every
chunk to a :class:`LoopDetector`, and returns the assembled ``ModelResponse``.  An aborted
response is returned with ``finish_reason="length"`` so the truncation machinery
(``llm_telemetry``, ``DecompositionTruncatedError``) treats it as the visible failure it
is, an ERROR ``LLM LOOP ABORTED`` line is logged and :func:`loop_abort_counts` counts it.
Everything else (Anthropic, streaming callers, no ``api_base``) passes through untouched.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("agent.loop_guard")

#: Check the tail every time this many new characters arrived.
CHECK_EVERY_CHARS = 512
#: The repeated region must be at least this long (~500 tokens) ...
MIN_REPEAT_CHARS = 2000
#: ... made of at least this many copies of one block, whose length is in this range.
MIN_COPIES = 8
MIN_BLOCK = 2
MAX_BLOCK = 600
_TAIL_CHARS = MIN_REPEAT_CHARS + MAX_BLOCK * 2

ENV_DISABLE = "MEMFUN_LOOP_GUARD"  # "0"/"false"/"off" disables; set per LM by [llm] loop_guard


class LoopDetector:
    """Incremental repetition detector over a growing text."""

    def __init__(
        self,
        min_repeat_chars: int = MIN_REPEAT_CHARS,
        min_copies: int = MIN_COPIES,
        check_every: int = CHECK_EVERY_CHARS,
    ) -> None:
        self.min_repeat_chars = min_repeat_chars
        self.min_copies = min_copies
        self.check_every = check_every
        self._tail = ""
        self._since_check = 0
        self.block: str = ""

    def feed(self, text: str) -> bool:
        """Add *text*; ``True`` once the output ends in a long repeated block."""
        if not text:
            return False
        self._tail = (self._tail + text)[-(self.min_repeat_chars + MAX_BLOCK * 2) :]
        self._since_check += len(text)
        if self._since_check < self.check_every:
            return False
        self._since_check = 0
        return self._looping()

    def _looping(self) -> bool:
        tail = self._tail
        n = len(tail)
        for p in range(MIN_BLOCK, min(MAX_BLOCK, n // self.min_copies) + 1):
            unit = tail[n - p :]
            # count how many copies of ``unit`` end the text
            copies = 1
            pos = n - 2 * p
            while pos >= 0 and tail[pos : pos + p] == unit:
                copies += 1
                pos -= p
            # a block made of one repeated character is a legitimate run (padding,
            # separators): only blocks with some variety count
            if (
                copies >= self.min_copies
                and copies * p >= self.min_repeat_chars
                and len(set(unit)) >= 2
                and unit.strip()
            ):
                self.block = unit
                return True
        return False


_COUNTS: dict[str, int] = {}
_LOCK = threading.Lock()


def loop_abort_counts() -> dict[str, int]:
    """Calls aborted for repetition by this process, by role."""
    with _LOCK:
        return dict(_COUNTS)


def _record(role: str) -> None:
    with _LOCK:
        _COUNTS[role] = _COUNTS.get(role, 0) + 1


def enabled() -> bool:
    """The ``MEMFUN_LOOP_GUARD`` env switch: ``0``/``false``/``off`` turns the guard off."""
    return os.environ.get(ENV_DISABLE, "").lower() not in ("0", "false", "off", "no")


def _guardable(kwargs: dict[str, Any]) -> bool:
    model = kwargs.get("model")
    return (
        isinstance(model, str)
        and model.startswith("openai/")
        and bool(kwargs.get("api_base"))
        and not kwargs.get("stream")
        and enabled()
    )


def _chunk_text(chunk: Any) -> str:
    try:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None) or ""
        reasoning = getattr(delta, "reasoning_content", None) or ""
        return f"{reasoning}{text}"
    except (AttributeError, IndexError, TypeError):
        return ""


def _finish(litellm: Any, chunks: list[Any], kwargs: dict[str, Any], aborted: str) -> Any:
    response = litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
    if aborted:
        from memfun_agent.llm_telemetry import current_role

        role = current_role()
        _record(role)
        logger.error(
            "LLM LOOP ABORTED (role=%s, model=%s): output repeated %r for >= %d chars; "
            "call cut after %d chunks",
            role,
            kwargs.get("model"),
            aborted[:60],
            MIN_REPEAT_CHARS,
            len(chunks),
        )
        # Report it as a cut-off response: every truncation path then fails visibly.
        for choice in response.choices:
            choice.finish_reason = "length"
    return response


def _stream_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    out["stream"] = True
    opts = dict(out.get("stream_options") or {})
    opts.setdefault("include_usage", True)
    out["stream_options"] = opts
    # DSPy passes litellm cache/retry options; streaming ignores caching, keep retries.
    out.pop("cache", None)
    return out


def _guarded_completion(litellm: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
    if args or not _guardable(kwargs):
        return original(*args, **kwargs)
    detector = LoopDetector()
    chunks: list[Any] = []
    aborted = ""
    stream = original(**_stream_kwargs(kwargs))
    try:
        for chunk in stream:
            chunks.append(chunk)
            if detector.feed(_chunk_text(chunk)):
                aborted = detector.block
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("closing aborted stream failed", exc_info=True)
    return _finish(litellm, chunks, kwargs, aborted)


async def _guarded_acompletion(litellm: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
    if args or not _guardable(kwargs):
        return await original(*args, **kwargs)
    detector = LoopDetector()
    chunks: list[Any] = []
    aborted = ""
    stream = await original(**_stream_kwargs(kwargs))
    try:
        async for chunk in stream:
            chunks.append(chunk)
            if detector.feed(_chunk_text(chunk)):
                aborted = detector.block
                break
    finally:
        aclose = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if callable(aclose):
            try:
                res = aclose()
                if hasattr(res, "__await__"):
                    await res
            except Exception:  # pragma: no cover - best effort
                logger.debug("closing aborted stream failed", exc_info=True)
    return _finish(litellm, chunks, kwargs, aborted)


_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def install_loop_guard() -> bool:
    """Wrap ``litellm.completion`` / ``acompletion`` once; ``True`` if newly installed."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return False
        import litellm

        orig = litellm.completion
        aorig = litellm.acompletion

        @functools.wraps(orig)
        def completion(*args: Any, **kwargs: Any) -> Any:
            return _guarded_completion(litellm, orig, *args, **kwargs)

        @functools.wraps(aorig)
        async def acompletion(*args: Any, **kwargs: Any) -> Any:
            return await _guarded_acompletion(litellm, aorig, *args, **kwargs)

        litellm.completion = completion
        litellm.acompletion = acompletion
        _INSTALLED = True
        return True
