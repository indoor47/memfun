from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memfun_core.types import Session, SessionConfig


@runtime_checkable
class SessionManager(Protocol):
    """User session management."""

    async def create_session(
        self, user_id: str, config: SessionConfig | None = None,
    ) -> Session: ...
    async def get_session(self, session_id: str) -> Session | None: ...
    async def update_session(self, session_id: str, data: dict) -> None: ...
    async def end_session(self, session_id: str) -> None: ...
