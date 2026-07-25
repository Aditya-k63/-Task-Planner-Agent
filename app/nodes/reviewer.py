"""
Reviewer node — verifies task completion.

Checks if the task's output satisfies the goal and marks it completed or failed.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import AgentState, Task, TaskStatus, AgentPhase, Step
from app.llm import get_llm

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM = """You are a task reviewer. Evaluate whether a task was completed successfully.

You will receive:
- The original goal
- The task description
- The execution history (tool calls and outputs)

Evaluate: Did the task accomplish what it was supposed to?

RESPOND WITH ONLY valid JSON:
{{
  "pass": true/false,
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "suggestion": "if failed, what should be tried differently"
}}

Be lenient — if the task is substantially complete, mark it as passing.
Only fail if the task clearly did not achieve its purpose."""


def reviewer_node(state: AgentState) -> dict:
    """Review the current task's completion. Returns partial state update."""
    logger.info("REVIEWER: Reviewing task completion")
    llm = get_llm()

    current_task = state.get_current_task()
    if current_task is None:
        return {"phase": AgentPhase.ERROR.value, "error": "No current task to review"}

    # Gather execution history for this task
    task_steps = [s for s in state["execution_history"] if s.get("task_id") == current_task.id]
    steps_text = "\n".join(
        f"  [{s['action']}] tool={s.get('tool', '-')} input={s.get('input', '-')} output={s.get('output', '-')[:200]}"
        for s in task_steps
    )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""Goal: {state['goal']}

Task: {current_task.description}

Execution history:
{steps_text if steps_text else '(no tool calls — task may not have been executed)'}

Did this task succeed? Respond with JSON only."""),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        raw = raw.strip()

    try:
        data = json.loads(raw)
        passed = data.get("pass", False)
        score = data.get("score", 0.0)
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        logger.error(f"Reviewer returned invalid JSON: {raw[:200]}")
        passed = False
        score = 0.0
        reason = f"Review parse error: {raw[:200]}"

    logger.info(f"REVIEWER: pass={passed}, score={score:.2f}, reason={reason[:100]}")

    # Update task status
    tasks = [t.model_dump() if hasattr(t, 'model_dump') else t for t in state["tasks"]]
    for i, t in enumerate(tasks):
        if t["id"] == current_task.id:
            if passed:
                tasks[i]["status"] = TaskStatus.COMPLETED.value
                tasks[i]["result"] = reason
            else:
                tasks[i]["status"] = TaskStatus.FAILED.value
                tasks[i]["error"] = reason
            break

    return {
        "tasks": tasks,
        "execution_history": state["execution_history"] + [
            Step(
                task_id=current_task.id,
                action="review",
                output=f"pass={passed} score={score:.2f} reason={reason}",
                success=passed,
            ).model_dump()
        ],
    }
