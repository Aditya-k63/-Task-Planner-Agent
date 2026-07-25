"""
Executor node — executes a single task using tool calls.

Detects when LLM requests clarification and transitions to CLARIFY phase.
Supports multi-step tool use within a single task.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.state import AgentState, Task, TaskStatus, AgentPhase, Clarification, ClarificationOption, Step
from app.llm import get_llm

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM = """You are a task executor. You execute one task at a time using available tools.

AVAILABLE TOOLS:
{tool_schemas}

RULES:
1. Execute the task described below using the tools above.
2. Use tools by responding with EXACTLY this JSON format (one per line, or multiple lines):
   TOOL_CALL: {{"tool": "tool_name", "input": "input for tool"}}
3. When the task is done, respond with:
   DONE: <summary of what was accomplished>
4. If you need clarification from the user to proceed, respond with:
   CLARIFY: {{"question": "...", "context": "...", "options": [{{"id": "a", "label": "...", "description": "..."}}, ...]}}
   - Provide exactly 3 options
   - Each option must be a concrete, actionable choice
5. If the task cannot be completed, respond with:
   FAILED: <reason>

Current task: {task_description}
Goal context: {goal}
"""

CLARIFY_SYSTEM = """You are a clarification agent. The executor encountered ambiguity.

Generate exactly 3 concrete options for the user to choose from.
Each option should represent a valid, actionable path forward.

RESPOND WITH ONLY valid JSON:
{{
  "question": "clear question text",
  "context": "why this clarification is needed",
  "options": [
    {{"id": "a", "label": "short label", "description": "detailed description"}},
    {{"id": "b", "label": "short label", "description": "detailed description"}},
    {{"id": "c", "label": "short label", "description": "detailed description"}}
  ]
}}"""


def executor_node(state: AgentState) -> dict:
    """Execute the current task. Returns partial state update."""
    logger.info("EXECUTOR: Starting task execution")
    llm = get_llm()
    from app.tools.registry import registry

    current_task = state.get_current_task()
    if current_task is None:
        return {"phase": AgentPhase.ERROR.value, "error": "No current task to execute"}

    tool_schemas = json.dumps(registry.list_schemas(), indent=2)
    system_msg = EXECUTOR_SYSTEM.format(
        tool_schemas=tool_schemas,
        task_description=current_task.description,
        goal=state["goal"],
    )

    messages = [SystemMessage(content=system_msg)]

    # Add execution history for context
    history = state["execution_history"]
    if history:
        recent = history[-5:]
        for step in recent:
            messages.append(AIMessage(content=f"[Previous] {step['action']}: {step.get('output', '')[:200]}"))

    # Add clarification history if any
    clarifications = state.get("clarification_history", [])
    if clarifications:
        for c in clarifications:
            if c.get("answer"):
                messages.append(AIMessage(content=f"[User clarified] Q: {c['question']}\nA: {c['answer']}"))

    # Multi-step execution loop (max 5 tool calls per invocation)
    max_steps = 5
    steps_taken = 0

    while steps_taken < max_steps:
        messages.append(HumanMessage(content=f"Execute the task. ({steps_taken + 1}/{max_steps} steps remaining)"))
        response = llm.invoke(messages)
        raw = response.content.strip()

        logger.info(f"EXECUTOR LLM: {raw[:200]}")

        # Check for DONE
        if raw.upper().startswith("DONE:"):
            summary = raw[5:].strip()
            return {
                "execution_history": state["execution_history"] + [
                    Step(task_id=current_task.id, action="completed", output=summary).model_dump()
                ],
            }

        # Check for FAILED
        if raw.upper().startswith("FAILED:"):
            reason = raw[7:].strip()
            logger.warning(f"EXECUTOR: Task failed — {reason}")
            updated_task = current_task.model_dump()
            updated_task["status"] = TaskStatus.FAILED.value
            updated_task["error"] = reason
            return {
                "tasks": [t.model_dump() if hasattr(t, 'model_dump') else t for t in state["tasks"] if t.id != current_task.id] + [updated_task],
                "execution_history": state["execution_history"] + [
                    Step(task_id=current_task.id, action="failed", output=reason, success=False).model_dump()
                ],
            }

        # Check for CLARIFY
        if raw.upper().startswith("CLARIFY:"):
            clarify_json = raw[8:].strip()
            try:
                clarify_data = json.loads(clarify_json)
                options = [ClarificationOption(**o) for o in clarify_data.get("options", [])]
                clarification = Clarification(
                    task_id=current_task.id,
                    question=clarify_data.get("question", ""),
                    context=clarify_data.get("context", ""),
                    options=options,
                )
                logger.info(f"EXECUTOR: Requesting clarification — {clarification.question[:80]}")
                return {
                    "pending_clarification": clarification.model_dump(),
                    "phase": AgentPhase.CLARIFYING.value,
                }
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to parse CLARIFY: {e}")
                messages.append(AIMessage(content=f"Invalid CLARIFY format. Use valid JSON. Error: {e}"))
                continue

        # Check for TOOL_CALL
        if raw.upper().startswith("TOOL_CALL:"):
            try:
                tool_json = raw[10:].strip()
                tool_data = json.loads(tool_json)
                tool_name = tool_data["tool"]
                tool_input = tool_data.get("input", "")

                logger.info(f"EXECUTOR: Calling tool '{tool_name}' with input: {tool_input[:100]}")
                result = registry.execute(tool_name, tool_input)
                logger.info(f"EXECUTOR: Tool result: {result[:200]}")

                messages.append(AIMessage(content=raw))
                messages.append(ToolMessage(content=result, tool_call_id=f"call_{steps_taken}"))

                steps_taken += 1

                state["execution_history"] = state["execution_history"] + [
                    Step(task_id=current_task.id, action="tool_call", tool=tool_name, input=tool_input, output=result[:500]).model_dump()
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse TOOL_CALL: {e}")
                messages.append(AIMessage(content=f"Invalid tool call format. Use: TOOL_CALL: {{\"tool\": \"name\", \"input\": \"arg\"}}"))
                continue
        else:
            # LLM gave a text response — treat as clarification request or task completion
            messages.append(AIMessage(content=raw))
            steps_taken += 1

    # Exhausted max steps
    logger.warning("EXECUTOR: Max steps reached")
    return {
        "execution_history": state["execution_history"] + [
            Step(task_id=current_task.id, action="max_steps_reached", output="Execution limit reached for this task").model_dump()
        ],
    }
