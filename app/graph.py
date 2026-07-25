"""
LangGraph StateGraph — wires all nodes into an agent loop.

Flow:
  START → planner → router → (executor → reviewer → router)* → done
                                        ↓
                                  clarifier → executor
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.state import AgentState, AgentPhase
from app.nodes.planner import planner_node
from app.nodes.executor import executor_node
from app.nodes.reviewer import reviewer_node
from app.nodes.router import router_node

logger = logging.getLogger(__name__)


# --- Edge conditions ---

def after_planner(state: AgentState) -> Literal["router", "__end__"]:
    if state["phase"] == AgentPhase.ERROR.value:
        return "__end__"
    return "router"


def after_router(state: AgentState) -> Literal["executor", "planner", "__end__"]:
    phase = AgentPhase(state["phase"])
    if phase == AgentPhase.DONE:
        return "__end__"
    if phase == AgentPhase.ERROR:
        return "__end__"
    if phase == AgentPhase.PLANNING:
        return "planner"
    return "executor"


def after_executor(state: AgentState) -> Literal["reviewer", "__end__"]:
    phase = AgentPhase(state["phase"])
    if phase == AgentPhase.CLARIFYING:
        # Graph pauses — clarifier is handled externally via API
        return "__end__"
    if phase == AgentPhase.ERROR:
        return "__end__"
    return "reviewer"


def after_reviewer(state: AgentState) -> Literal["router", "__end__"]:
    if state["phase"] == AgentPhase.ERROR.value:
        return "__end__"
    return "router"


def after_clarifier(state: AgentState) -> Literal["executor", "__end__"]:
    if state["phase"] == AgentPhase.ERROR.value:
        return "__end__"
    return "executor"


# --- Build graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("router", router_node)

    # Edges
    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", after_planner, {"router": "router", "__end__": END})
    graph.add_conditional_edges("router", after_router, {"executor": "executor", "planner": "planner", "__end__": END})
    graph.add_conditional_edges("executor", after_executor, {"reviewer": "reviewer", "__end__": END})
    graph.add_conditional_edges("reviewer", after_reviewer, {"router": "router", "__end__": END})

    # Clarifier is handled via API — user answers, then executor is re-invoked
    # We don't wire clarifier into the graph as a node — it's an external interrupt
    # The graph pauses at executor (phase=CLARIFYING), API resumes it

    return graph


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph
