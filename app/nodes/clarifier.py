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

    clarification = state.pending_clarification
    if clarification is None:
        return {"error": "No pending clarification"}

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
    history = list(state.get("clarification_history", []))
    history.append(clarification_data)

    return {
        "pending_clarification": None,
        "clarification_history": history,
        "phase": AgentPhase.EXECUTING.value,
    }


def get_pending_question(state: AgentState) -> dict | None:
    """Returns the pending clarification for the frontend to render."""
    c = state.pending_clarification
    if c is None:
        return None
    return {
        "id": c.id,
        "task_id": c.task_id,
        "question": c.question,
        "context": c.context,
        "options": [{"id": o.id, "label": o.label, "description": o.description} for o in c.options],
    }
