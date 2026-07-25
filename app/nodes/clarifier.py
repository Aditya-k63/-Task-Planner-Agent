"""
Clarifier node — handles the 3-option clarification flow.

When user responds, the answer feeds back into executor context.
"""

from __future__ import annotations

import logging

from app.state import AgentState, AgentPhase, Clarification

logger = logging.getLogger(__name__)


def clarifier_resume(state: AgentState, answer: str) -> dict:
    """
    Called when user answers a clarification question.
    Merges the answer into state and resumes execution.
    """
    logger.info(f"CLARIFIER: User answered — {answer}")

    # Get clarification safely (state may be plain dict from LangGraph)
    clar_raw = state.get("pending_clarification") if isinstance(state, dict) else getattr(state, "pending_clarification", None)
    if clar_raw is None:
        return {"error": "No pending clarification"}

    clarification = Clarification(**clar_raw) if isinstance(clar_raw, dict) else clar_raw

    # Record the answer
    clarification_data = clarification.model_dump()
    clarification_data["answer"] = answer

    # Find the matching option label if it's a single letter
    if len(answer) == 1:
        for opt in clarification.options:
            if opt.id.lower() == answer.lower():
                clarification_data["answer"] = f"{opt.label}: {opt.description}"
                break

    from datetime import datetime, timezone
    clarification_data["answered_at"] = datetime.now(timezone.utc).isoformat()

    # Add to history, clear pending
    history = list(state.get("clarification_history", []) if isinstance(state, dict) else getattr(state, "clarification_history", []))
    history.append(clarification_data)

    return {
        "pending_clarification": None,
        "clarification_history": history,
        "phase": AgentPhase.EXECUTING.value,
    }


def get_pending_question(state: AgentState) -> dict | None:
    """Returns the pending clarification for the frontend to render."""
    clar_raw = state.get("pending_clarification") if isinstance(state, dict) else getattr(state, "pending_clarification", None)
    if clar_raw is None:
        return None
    c = Clarification(**clar_raw) if isinstance(clar_raw, dict) else clar_raw
    return {
        "id": c.id,
        "task_id": c.task_id,
        "question": c.question,
        "context": c.context,
        "options": [{"id": o.id, "label": o.label, "description": o.description} for o in c.options],
    }
