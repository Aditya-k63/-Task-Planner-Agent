import pytest
from app.state import AgentState, Task, TaskStatus, TaskPriority, Clarification, ClarificationOption, AgentPhase


def test_agent_state_defaults():
    state = AgentState(goal="Build a website")
    assert state.goal == "Build a website"
    assert state.tasks == []
    assert state.current_task_id is None
    assert state.phase == AgentPhase.IDLE
    assert state.error is None
    assert state.retry_count == 0


def test_task_model():
    task = Task(description="Set up database", priority=TaskPriority.HIGH)
    assert task.status == TaskStatus.PENDING
    assert task.priority == TaskPriority.HIGH
    assert task.id is not None
    assert len(task.id) == 8


def test_task_with_dependencies():
    t1 = Task(description="Create schema")
    t2 = Task(description="Run migrations", dependencies=[t1.id])
    assert t1.id in t2.dependencies


def test_clarification_model():
    options = [
        ClarificationOption(id="a", label="SQLite", description="Lightweight DB"),
        ClarificationOption(id="b", label="PostgreSQL", description="Production DB"),
        ClarificationOption(id="c", label="MongoDB", description="NoSQL"),
    ]
    c = Clarification(
        task_id="abc123",
        question="Which database?",
        context="Need to pick one",
        options=options,
    )
    assert len(c.options) == 3
    assert c.answer is None


def test_agent_state_get_current_task():
    t1 = Task(description="Task 1")
    t2 = Task(description="Task 2")
    state = AgentState(
        goal="test",
        tasks=[t1.model_dump(), t2.model_dump()],
        current_task_id=t1.id,
    )
    current = state.get_current_task()
    assert current is not None
    assert current.id == t1.id


def test_agent_state_pending_tasks():
    t1 = Task(description="Done", status=TaskStatus.COMPLETED)
    t2 = Task(description="Pending", status=TaskStatus.PENDING)
    t3 = Task(description="In progress", status=TaskStatus.IN_PROGRESS)
    state = AgentState(
        goal="test",
        tasks=[t1.model_dump(), t2.model_dump(), t3.model_dump()],
    )
    pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
    assert len(pending) == 1
    assert pending[0].description == "Pending"


def test_agent_state_all_completed():
    t1 = Task(description="Done", status=TaskStatus.COMPLETED)
    t2 = Task(description="Also done", status=TaskStatus.COMPLETED)
    state = AgentState(goal="test", tasks=[t1.model_dump(), t2.model_dump()])
    assert state.all_completed() is True

    t3 = Task(description="Pending")
    state2 = AgentState(goal="test", tasks=[t1.model_dump(), t3.model_dump()])
    assert state2.all_completed() is False
