"""
Executor node — executes a single task using tool calls.

Detects when LLM requests clarification and transitions to CLARIFY phase.
Supports multi-step tool use within a single task.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.state import Task, TaskStatus, AgentPhase, Clarification, ClarificationOption, Step
from app.llm import get_llm

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM = """You are a task executor. You execute one task at a time using available tools.

AVAILABLE TOOLS:
{tool_schemas}

RULES:
1. Execute the task described below using the tools above.
2. Use tools by responding with EXACTLY this JSON format:
   TOOL_CALL: {{"tool": "tool_name", "input": "input for tool"}}
3. When the task is done, respond with:
   DONE: <summary of what was accomplished>
4. If you need clarification from the user to proceed, respond with:
   CLARIFY: {{"question": "...", "context": "...", "options": [{{"id": "a", "label": "...", "description": "..."}}, ...]}}
   - Provide exactly 3 options
   - Each option must be a concrete, actionable choice
5. If the task cannot be completed, respond with:
   FAILED: <reason>

CLARIFICATION RULES — READ CAREFULLY:
- NEVER ask for clarification on simple factual or calculation tasks (e.g. math, unit conversion, definitions, lookups). Just answer or use a tool.
- NEVER ask for clarification when you already have enough information to proceed. Just execute.
- ONLY use CLARIFY when the task is genuinely ambiguous, has multiple valid interpretations, or is missing critical information that prevents you from taking ANY action.
- If you can make a reasonable assumption, do so — don't ask.
- Default to acting, not asking.

Current task: {task_description}
Goal context: {goal}
"""


def executor_node(state: dict) -> dict:
    """Execute the current task. Returns state update."""
    logger.info("EXECUTOR: Starting task execution")
    try:
        return _executor_inner(state)
    except Exception as e:
        logger.error(f"EXECUTOR: Unhandled exception: {type(e).__name__}: {e}", exc_info=True)
        return {"phase": "error", "error": f"Executor crashed: {type(e).__name__}: {e}"}


def _executor_inner(state: dict) -> dict:
    from app.tools.registry import registry
    llm = get_llm()

    tasks_raw = state.get("tasks", [])
    current_task_id = state.get("current_task_id")
    goal = state.get("goal", "unknown")
    history = state.get("execution_history", [])

    logger.info(f"EXECUTOR: tasks={len(tasks_raw)}, current_task_id={current_task_id}")

    # Find current task
    current_task = None
    for t in tasks_raw:
        tid = t.get("id") if isinstance(t, dict) else t.id
        if tid == current_task_id:
            current_task = Task(**t) if isinstance(t, dict) else t
            break

    if current_task is None:
        logger.error(f"Task lookup failed. Available: {[t.get('id') if isinstance(t, dict) else t.id for t in tasks_raw]}, looking for: {current_task_id}")
        state["phase"] = "error"
        state["error"] = f"No task found with id={current_task_id}"
        return state

    tool_schemas = json.dumps(registry.list_schemas(), indent=2)
    system_msg = EXECUTOR_SYSTEM.format(
        tool_schemas=tool_schemas,
        task_description=current_task.description,
        goal=goal,
    )

    messages = [SystemMessage(content=system_msg)]

    # Add execution history for context
    if history:
        for step in history[-5:]:
            messages.append(AIMessage(content=f"[Previous] {step.get('action', '')}: {step.get('output', '')[:200]}"))

    # Add clarification history if any
    for c in state.get("clarification_history", []):
        if c.get("answer"):
            messages.append(AIMessage(content=f"[User clarified] Q: {c.get('question', '')}\nA: {c['answer']}"))

    # Multi-step execution loop (max 5 tool calls per invocation)
    max_steps = 5
    steps_taken = 0
    new_history = list(history)

    while steps_taken < max_steps:
        messages.append(HumanMessage(content=f"Execute the task. ({steps_taken + 1}/{max_steps} steps remaining)"))
        response = llm.invoke(messages)
        raw = response.content
        # Handle list content
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        raw = str(raw).strip()

        logger.info(f"EXECUTOR LLM: {raw[:300]}")

        # Check for DONE
        if raw.upper().startswith("DONE:"):
            summary = raw[5:].strip()
            new_history.append(Step(task_id=current_task.id, action="completed", output=summary).model_dump())
            state["execution_history"] = new_history
            return state

        # Check for FAILED
        if raw.upper().startswith("FAILED:"):
            reason = raw[7:].strip()
            logger.warning(f"EXECUTOR: Task failed — {reason}")
            updated_task = current_task.model_dump()
            updated_task["status"] = TaskStatus.FAILED.value
            updated_task["error"] = reason
            updated_tasks = [t for t in tasks_raw if (t.get("id") if isinstance(t, dict) else t.id) != current_task.id]
            updated_tasks.append(updated_task)
            new_history.append(Step(task_id=current_task.id, action="failed", output=reason, success=False).model_dump())
            state["tasks"] = updated_tasks
            state["execution_history"] = new_history
            return state

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
                state["pending_clarification"] = clarification.model_dump()
                state["phase"] = AgentPhase.CLARIFYING.value
                return state
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to parse CLARIFY: {e}")
                messages.append(AIMessage(content=f"Invalid CLARIFY format. Use valid JSON. Error: {e}"))
                continue

        # Check for TOOL_CALL
        if raw.upper().startswith("TOOL_CALL:"):
            try:
                tool_json = raw[10:].strip()
                # Handle LLM wrapping in extra quotes or newlines
                tool_json = tool_json.strip('"').strip("'")
                tool_data = json.loads(tool_json)
                tool_name = tool_data["tool"]
                tool_input = tool_data.get("input", "")

                logger.info(f"EXECUTOR: Calling tool '{tool_name}' with input: {tool_input[:100]}")
                result = registry.execute(tool_name, tool_input)
                logger.info(f"EXECUTOR: Tool result: {result[:200]}")

                messages.append(AIMessage(content=raw))
                messages.append(ToolMessage(content=result, tool_call_id=f"call_{steps_taken}"))

                steps_taken += 1
                new_history.append(Step(task_id=current_task.id, action="tool_call", tool=tool_name, input=tool_input, output=result[:500]).model_dump())
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse TOOL_CALL: {e}")
                messages.append(AIMessage(content=f"Invalid format. Use: TOOL_CALL: {{\"tool\": \"name\", \"input\": \"arg\"}}"))
                continue
        else:
            # LLM gave text response — ask it to use tools
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content="Use a TOOL_CALL to execute the task, or DONE/FAILED/CLARIFY."))
            steps_taken += 1

    logger.warning("EXECUTOR: Max steps reached")
    new_history.append(Step(task_id=current_task.id, action="max_steps_reached", output="Execution limit reached").model_dump())
    state["execution_history"] = new_history
    return state
