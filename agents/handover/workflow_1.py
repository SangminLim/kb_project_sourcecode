from __future__ import annotations

from typing import Any, Dict, List

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    StateGraph = None
    START = "__start__"
    END = "__end__"

from .models import AgentResult, AgentWorkflowState


def build_handover_workflow(agent: Any):
    """Handover LangGraph workflow orchestration."""

    if StateGraph is None:
        raise RuntimeError(
            "LangGraph가 설치되어 있지 않습니다. "
            "pip install langgraph 후 사용하세요."
        )

    workflow = StateGraph(AgentWorkflowState)

    workflow.add_node("prepare", agent._graph_prepare_node)
    workflow.add_node("rewrite", agent._graph_rewrite_node)
    workflow.add_node("resolve", agent._graph_resolve_node)
    workflow.add_node("guard", agent._graph_guard_node)
    workflow.add_node("retrieve", agent._graph_retrieve_node)
    workflow.add_node("respond", agent._graph_respond_node)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "rewrite")
    workflow.add_edge("rewrite", "resolve")
    workflow.add_edge("resolve", "guard")

    workflow.add_conditional_edges(
        "guard",
        agent._graph_after_guard,
        {
            "end": END,
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


def run_handover_graph(
    agent: Any,
    question: str,
    chat_history: List[Dict[str, str]],
    top_k: int,
) -> AgentResult:
    """LangGraph workflow 실행."""

    compiled = build_handover_workflow(agent)

    initial_state: AgentWorkflowState = {
        "question": question,
        "chat_history": chat_history or [],
        "top_k": top_k,
        "debug_logs": [],
    }

    final_state = compiled.invoke(initial_state)

    result = final_state.get("result")

    if not result:
        raise RuntimeError("LangGraph workflow did not return AgentResult.")

    return result
