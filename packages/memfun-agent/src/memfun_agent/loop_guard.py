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

import asyncio
import functools
import os
import threading
import time
from typing import Any

from memfun_core.logging import get_logger

logger = get_logger("agent.loop_guard")

#: Check the tail every time this many new characters arrived.
CHECK_EVERY_CHARS = 512
#: The repeated region must be at least this long (~1500 tokens): legitimate output can
#: repeat a short line a few hundred times (``| 0 | 0 |`` table rows, ``pass`` lines,
#: ``0,`` padding), a loop never stops, so waiting costs only seconds ...
MIN_REPEAT_CHARS = 6000
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
                and any(ch.isalnum() for ch in unit)
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


#: (litellm model string, api_base) pairs of the LMs built with ``[llm] loop_guard = true``.
#: The litellm wrapper is process-wide but only touches calls to these exact endpoints.
_GUARDED: set[tuple[str, str]] = set()


def register_guarded(model: str, api_base: str) -> None:
    """Mark calls to ``model`` at ``api_base`` as guarded (idempotent)."""
    with _LOCK:
        _GUARDED.add((model, api_base))


def _guardable(kwargs: dict[str, Any]) -> bool:
    model, api_base = kwargs.get("model"), kwargs.get("api_base")
    return (
        isinstance(model, str)
        and model.startswith("openai/")
        and isinstance(api_base, str)
        and (model, api_base) in _GUARDED
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


#: A connection error mid-stream is retried from scratch (litellm's own retries only cover
#: the request phase), at most this many times, with a short backoff.
MAX_STREAM_RETRIES = 3
RETRY_BACKOFF_S = 1.0


#: litellm's own streaming guard raises this after ~100 identical chunks in a row: that IS a
#: loop (single-token repetition), so it is an abort, not a connection error to retry.
LITELLM_REPEAT_MSG = "repeating the same chunk"


def _is_litellm_repeat(exc: BaseException) -> bool:
    return LITELLM_REPEAT_MSG in str(exc)


def _repeat_block(chunks: list[Any]) -> str:
    return (_chunk_text(chunks[-1]).strip() if chunks else "") or LITELLM_REPEAT_MSG


def _retries(kwargs: dict[str, Any]) -> int:
    n = kwargs.get("num_retries")
    return max(0, min(int(n), MAX_STREAM_RETRIES)) if isinstance(n, int) else MAX_STREAM_RETRIES


def _guarded_completion(litellm: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
    if args or not _guardable(kwargs):
        return original(*args, **kwargs)
    attempt = 0
    while True:
        detector = LoopDetector()
        chunks: list[Any] = []
        aborted = ""
        stream = None
        try:
            stream = original(**_stream_kwargs(kwargs))
            for chunk in stream:
                chunks.append(chunk)
                if detector.feed(_chunk_text(chunk)):
                    aborted = detector.block
                    break
        except Exception as exc:
            if _is_litellm_repeat(exc) and chunks:
                aborted = _repeat_block(chunks)
            else:
                if attempt >= _retries(kwargs):
                    raise
                attempt += 1
                logger.warning(
                    "LLM stream failed mid-call (%s); retry %d", type(exc).__name__, attempt
                )
                time.sleep(RETRY_BACKOFF_S * attempt)
                continue
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("closing stream failed", exc_info=True)
        return _finish(litellm, chunks, kwargs, aborted)


async def _guarded_acompletion(litellm: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
    if args or not _guardable(kwargs):
        return await original(*args, **kwargs)
    attempt = 0
    while True:
        detector = LoopDetector()
        chunks: list[Any] = []
        aborted = ""
        stream = None
        try:
            stream = await original(**_stream_kwargs(kwargs))
            async for chunk in stream:
                chunks.append(chunk)
                if detector.feed(_chunk_text(chunk)):
                    aborted = detector.block
                    break
        except Exception as exc:
            if _is_litellm_repeat(exc) and chunks:
                aborted = _repeat_block(chunks)
            else:
                if attempt >= _retries(kwargs):
                    raise
                attempt += 1
                logger.warning(
                    "LLM stream failed mid-call (%s); retry %d", type(exc).__name__, attempt
                )
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
                continue
        finally:
            aclose = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if callable(aclose):
                try:
                    res = aclose()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:  # pragma: no cover - best effort
                    logger.debug("closing stream failed", exc_info=True)
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
