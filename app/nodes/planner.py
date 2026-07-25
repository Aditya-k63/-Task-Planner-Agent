"""
Planner node — breaks a high-level goal into a dependency DAG of tasks.

LLM receives the goal + available tools, returns a structured task plan.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState, Task, TaskStatus, TaskPriority, AgentPhase
from app.llm import get_llm

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a task planner. Given a high-level goal, break it into concrete,
ordered subtasks that an AI agent can execute using available tools.

RULES:
1. Each task must be a single, concrete, actionable step (not vague).
2. Define dependencies explicitly — which tasks must complete before others.
3. Prioritize tasks (low/medium/high/critical).
4. If the goal requires information you don't have, add a task that uses the web_search tool.
5. Keep task count reasonable: 2-8 tasks for most goals.
6. Do NOT ask clarifying questions — just make reasonable assumptions.

RESPOND WITH ONLY valid JSON (no markdown, no explanation):
{
  "tasks": [
    {
      "description": "string — concrete action",
      "priority": "low|medium|high|critical",
      "dependencies": ["task_id_1", ...],
      "substeps": ["step1", "step2"]
    }
  ]
}

Available tools: {tools}
"""

PLANNER_RESPONSE_SCHEMA = """{
  "tasks": [
    {
      "description": "string",
      "priority": "low|medium|high|critical",
      "dependencies": [],
      "substeps": []
    }
  ]
}"""


def planner_node(state: AgentState) -> dict:
    """Break goal into tasks. Returns partial state update."""
    logger.info("PLANNER: Breaking goal into tasks")
    llm = get_llm()

    from app.tools.registry import registry
    tools = ", ".join(t.name for t in registry.list_tools())

    system_msg = PLANNER_SYSTEM.format(tools=tools)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Goal: {state['goal']}\n\nCreate a task plan. Respond with JSON only."),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        raw = raw.strip()

    try:
        data = json.loads(raw)
        tasks_raw = data.get("tasks", [])
    except json.JSONDecodeError:
        logger.error(f"Planner returned invalid JSON: {raw[:200]}")
        return {
            "phase": AgentPhase.ERROR.value,
            "error": f"Planner returned invalid JSON: {raw[:200]}",
        }

    tasks: list[Task] = []
    for i, t in enumerate(tasks_raw):
        priority_str = t.get("priority", "medium")
        try:
            priority = TaskPriority(priority_str)
        except ValueError:
            priority = TaskPriority.MEDIUM

        task = Task(
            description=t["description"],
            priority=priority,
            dependencies=t.get("dependencies", []),
            substeps=t.get("substeps", []),
            status=TaskStatus.PENDING,
        )
        tasks.append(task)

    if not tasks:
        return {
            "phase": AgentPhase.DONE.value,
            "error": "Planner produced no tasks",
        }

    first_task = tasks[0]
    logger.info(f"PLANNER: Created {len(tasks)} tasks, starting with '{first_task.description[:50]}'")

    return {
        "tasks": [t.model_dump() for t in tasks],
        "current_task_id": first_task.id,
        "phase": AgentPhase.EXECUTING.value,
        "retry_count": 0,
    }
