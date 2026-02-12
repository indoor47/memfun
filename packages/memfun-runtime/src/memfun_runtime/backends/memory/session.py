from __future__ import annotations

import time
import uuid

from memfun_core.errors import SessionNotFoundError
from memfun_core.types import Session, SessionConfig


class InProcessSessionManager:
    """T0 session manager: in-memory sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._configs: dict[str, SessionConfig] = {}

    async def create_session(self, user_id: str, config: SessionConfig | None = None) -> Session:
        config = config or SessionConfig()
        session_id = uuid.uuid4().hex
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
        )
        self._sessions[session_id] = session
        self._configs[session_id] = config
        return session

    async def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        config = self._configs.get(session_id)
        if config and (time.time() - session.created_at) > config.ttl_seconds:
            del self._sessions[session_id]
            del self._configs[session_id]
            return None
        return session

    async def update_session(self, session_id: str, data: dict) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        session.data.update(data)

    async def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._configs.pop(session_id, None)
