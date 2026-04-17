from __future__ import annotations

import asyncio
from typing import Optional, Protocol

from app.models import SessionState


class SessionStore(Protocol):
    async def get(self, session_id: str) -> Optional[SessionState]:
        ...

    async def upsert(self, state: SessionState) -> None:
        ...

    async def delete(self, session_id: str) -> None:
        ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            state = self._data.get(session_id)
            return state.model_copy(deep=True) if state else None

    async def upsert(self, state: SessionState) -> None:
        async with self._lock:
            self._data[state.session_id] = state.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)
