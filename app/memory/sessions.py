"""
In-memory session store.

For production, replace with PostgreSQL-backed storage.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from app.state import AgentState, AgentPhase


class Session:
    def __init__(self, session_id: str, goal: str) -> None:
        self.id = session_id
        self.goal = goal
        self.state = AgentState(goal=goal, phase=AgentPhase.IDLE)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_updated = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "phase": self.state["phase"],
            "tasks": self.state.get("tasks", []),
            "current_task_id": self.state.get("current_task_id"),
            "pending_clarification": self.state.get("pending_clarification"),
            "execution_history": self.state.get("execution_history", []),
            "clarification_history": self.state.get("clarification_history", []),
            "error": self.state.get("error"),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, goal: str) -> Session:
        with self._lock:
            session = Session(session_id, goal)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def update_state(self, session_id: str, updates: dict) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            for k, v in updates.items():
                session.state[k] = v
            session.last_updated = datetime.now(timezone.utc).isoformat()

    def list_all(self) -> list[dict]:
        return [s.to_dict() for s in self._sessions.values()]


store = SessionStore()
