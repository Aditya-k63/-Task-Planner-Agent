import pytest
from app.nodes.clarifier import get_pending_question, clarifier_resume
from app.state import AgentState, Clarification, ClarificationOption, AgentPhase


def _make_clarify_state():
    options = [
        ClarificationOption(id="a", label="Option A", description="First choice"),
        ClarificationOption(id="b", label="Option B", description="Second choice"),
        ClarificationOption(id="c", label="Option C", description="Third choice"),
    ]
    c = Clarification(
        task_id="task1",
        question="Which approach?",
        context="Need to decide",
        options=options,
    )
    return AgentState(
        goal="test goal",
        pending_clarification=c.model_dump(),
        phase=AgentPhase.CLARIFYING,
    )


def test_get_pending_question():
    state = _make_clarify_state()
    q = get_pending_question(state)
    assert q is not None
    assert q["question"] == "Which approach?"
    assert len(q["options"]) == 3
    assert q["options"][0]["id"] == "a"


def test_get_pending_question_none():
    state = AgentState(goal="test")
    q = get_pending_question(state)
    assert q is None


def test_clarifier_resume_with_letter():
    state = _make_clarify_state()
    updates = clarifier_resume(state, "b")
    assert updates["pending_clarification"] is None
    assert len(updates["clarification_history"]) == 1
    assert "Option B" in updates["clarification_history"][0]["answer"]


def test_clarifier_resume_with_custom():
    state = _make_clarify_state()
    updates = clarifier_resume(state, "Use Redis instead")
    assert updates["pending_clarification"] is None
    assert updates["clarification_history"][0]["answer"] == "Use Redis instead"


def test_clarifier_resume_no_pending():
    state = AgentState(goal="test")
    updates = clarifier_resume(state, "a")
    assert "error" in updates
