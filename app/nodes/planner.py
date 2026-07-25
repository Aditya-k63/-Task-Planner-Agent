"""
Planner node — breaks a high-level goal into a dependency DAG of tasks.

LLM receives the goal + available tools, returns a structured task plan.
"""

from __future__ import annotations

import json
import re
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
    try:
        return _planner_node_inner(state)
    except Exception as e:
        logger.error(f"PLANNER: Unhandled exception: {type(e).__name__}: {e}")
        return {
            "phase": AgentPhase.ERROR.value,
            "error": f"Planner crashed: {type(e).__name__}: {e}",
        }


def _planner_node_inner(state: AgentState) -> dict:
    llm = get_llm()

    from app.tools.registry import registry
    tools = ", ".join(t.name for t in registry.list_tools())

    system_msg = PLANNER_SYSTEM.format(tools=tools)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Goal: {state['goal']}\n\nCreate a task plan. Respond with JSON only."),
    ]

    response = llm.invoke(messages)
    # Handle both string and list content formats
    content = response.content
    if isinstance(content, list):
        # Extract text from content blocks
        raw = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    else:
        raw = str(content).strip()
    logger.info(f"PLANNER raw response (first 500): {raw[:500]}")

    # Strip markdown code fences if present
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()
            if raw.startswith(("json", "JSON")):
                raw = raw[4:].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        logger.info(f"PLANNER after fence strip: {raw[:300]}")

    # Extract JSON object if surrounded by other text
    start = raw.find("{")
    end = raw.rfind("}")
    logger.info(f"PLANNER JSON search: start={start}, end={end}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
        logger.info(f"PLANNER extracted JSON: {raw[:300]}")

    try:
        data = json.loads(raw)
        tasks_raw = data.get("tasks", [])
    except json.JSONDecodeError:
        logger.error(f"Planner returned invalid JSON: {raw[:300]}")
        # Fallback: try to extract tasks array via regex
        tasks_match = re.search(r'"tasks"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if tasks_match:
            try:
                tasks_raw = json.loads(tasks_match.group(1))
                logger.info(f"PLANNER: Extracted tasks via regex fallback")
            except json.JSONDecodeError:
                return {
                    "phase": AgentPhase.ERROR.value,
                    "error": f"Planner returned invalid JSON: {raw[:300]}",
                }
        else:
            return {
                "phase": AgentPhase.ERROR.value,
                "error": f"Planner returned invalid JSON: {raw[:300]}",
            }

    tasks: list[Task] = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            logger.warning(f"PLANNER: Skipping non-dict task item: {type(t)}: {str(t)[:100]}")
            continue
        desc = t.get("description", "")
        if not desc or not isinstance(desc, str):
            logger.warning(f"PLANNER: Skipping task with no valid description: {t}")
            continue
        priority_str = t.get("priority", "medium")
        try:
            priority = TaskPriority(priority_str)
        except ValueError:
            priority = TaskPriority.MEDIUM

        task = Task(
            description=desc,
            priority=priority,
            dependencies=[d for d in t.get("dependencies", []) if isinstance(d, str)],
            substeps=[s for s in t.get("substeps", []) if isinstance(s, str)],
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
