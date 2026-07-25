"""
Router node — decides what happens next based on current state.

Returns one of: execute, replan, done
"""

from __future__ import annotations

import logging

from app.state import AgentState, TaskStatus, AgentPhase

logger = logging.getLogger(__name__)


def router_node(state: AgentState) -> dict:
    """Route to next action. Returns partial state update."""
    tasks = state.tasks

    if not tasks:
        logger.info("ROUTER: No tasks — done")
        return {"phase": AgentPhase.DONE.value}

    # Find next pending task whose dependencies are all completed
    pending = [t for t in tasks if t.status == TaskStatus.PENDING]
    completed_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}

    for task in pending:
        deps_met = all(dep in completed_ids for dep in task.dependencies)
        if deps_met:
            logger.info(f"ROUTER: Next task — {task.description[:50]}")
            return {
                "current_task_id": task.id,
                "phase": AgentPhase.EXECUTING.value,
                "retry_count": 0,
            }

    # Check if any tasks are in progress
    in_progress = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
    if in_progress:
        logger.info("ROUTER: Tasks still in progress, waiting")
        return {"phase": AgentPhase.EXECUTING.value}

    # All tasks resolved
    all_done = all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in tasks)
    if all_done:
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        logger.info(f"ROUTER: All tasks done — {completed} completed, {failed} failed")
        return {"phase": AgentPhase.DONE.value}

    # Some tasks blocked — try replanning
    logger.info("ROUTER: Some tasks blocked, triggering replan")
    return {"phase": AgentPhase.PLANNING.value}
