"""SystemExit from agent-written REPL code must not escape LocalREPL.

Regression for the live swarm freeze: argparse's ``sys.exit(2)`` on a usage
error escaped ``LocalREPL.execute`` (``except Exception`` only), propagated
out of ``asyncio.run`` and killed every coroutine in the episode.
"""

from __future__ import annotations

import asyncio

import pytest
from memfun_agent.rlm import LocalREPL, RLMModule, build_context_metadata

_ARGPARSE_SETUP = (
    "import argparse\np = argparse.ArgumentParser(prog='x')\np.add_argument('nums', nargs='+')\n"
)


@pytest.mark.parametrize(
    ("code", "code_repr"),
    [
        ("import sys\nsys.exit(2)", "2"),
        ("import sys\nsys.exit()", "None"),
        ("raise SystemExit('msg')", "'msg'"),
        (_ARGPARSE_SETUP + "p.parse_args([])", "2"),
        (_ARGPARSE_SETUP + "p.parse_args(['--help'])", "0"),
    ],
    ids=["exit2", "exit-noarg", "raise-msg", "argparse-usage", "argparse-help"],
)
def test_system_exit_is_reported_not_raised(code: str, code_repr: str) -> None:
    repl = LocalREPL()
    result = repl.execute(code)  # must not raise
    assert result.success is False
    assert "SystemExit" in result.stderr
    assert f"Your code called sys.exit({code_repr})" in result.stderr
    assert "it was blocked" in result.stderr
    assert "argparse" in result.stderr
    assert result.duration_ms >= 0.0


def test_output_before_exit_is_kept_and_namespace_survives() -> None:
    repl = LocalREPL()
    result = repl.execute("x = 41\nprint('before')\nimport sys\nsys.exit(1)")
    assert result.success is False
    assert result.stdout == "before\n"
    follow = repl.execute("print(x + 1)")
    assert follow.success is True
    assert follow.stdout == "42\n"


def test_argparse_usage_message_still_captured() -> None:
    result = LocalREPL().execute(_ARGPARSE_SETUP + "p.parse_args([])")
    # argparse writes its usage error to stderr before exiting.
    assert "usage: x" in result.stderr
    assert "required: nums" in result.stderr


def test_normal_exception_unchanged() -> None:
    result = LocalREPL().execute("print('hi')\n1 / 0")
    assert result.success is False
    assert result.stdout == "hi\n"
    assert "ZeroDivisionError" in result.stderr
    assert "sys.exit" not in result.stderr


def test_success_unchanged() -> None:
    result = LocalREPL().execute("print('ok')")
    assert result.success is True
    assert result.stdout == "ok\n"
    assert result.stderr == ""


def test_keyboard_interrupt_still_propagates() -> None:
    repl = LocalREPL()
    with pytest.raises(KeyboardInterrupt):
        repl.execute("raise KeyboardInterrupt")


def test_keyboard_interrupt_subclass_still_propagates() -> None:
    """Signal-to-exception adapters (e.g. a TerminateRequestedError that
    subclasses KeyboardInterrupt) must still stop the program."""

    class TerminateRequestedError(KeyboardInterrupt):
        pass

    repl = LocalREPL()
    repl.namespace["TerminateRequestedError"] = TerminateRequestedError
    with pytest.raises(TerminateRequestedError):
        repl.execute("raise TerminateRequestedError('signal 15')")


def test_event_loop_survives_agent_sys_exit() -> None:
    """Swarm-style: one agent's REPL code exits; the sibling still finishes."""
    module = RLMModule()
    repl = LocalREPL()

    async def agent() -> bool:
        await asyncio.sleep(0)
        out = await module._execute_code(  # pyright: ignore[reportPrivateUsage]
            repl, _ARGPARSE_SETUP + "p.parse_args([])"
        )
        return out.success

    async def other() -> str:
        for _ in range(5):
            await asyncio.sleep(0)
        return "other-done"

    async def episode() -> list[object]:
        return list(await asyncio.gather(asyncio.wait_for(agent(), 30), other()))

    assert asyncio.run(episode()) == [False, "other-done"]


def test_prompt_mentions_sys_exit_is_caught() -> None:
    meta = build_context_metadata("ctx", root="/tmp/x")
    assert "sys.exit() (incl. argparse errors/--help) is caught" in meta
    assert "__main__" in meta
