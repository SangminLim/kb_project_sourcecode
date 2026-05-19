from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class AgentResult:
    original_question: str
    normalized_question: str
    rewritten_question: str
    system_id: Optional[str]
    intent: str
    answer: str
    render_type: str = "text"
    graph_data: Optional[Dict[str, Any]] = None
    query_meta: Optional[Dict[str, Any]] = None
    realtime_mode: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    debug_logs: List[str] = field(default_factory=list)


class AgentWorkflowState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    top_k: int

    raw_question: str
    normalized_question: str
    rewritten_question: str
    rewrite_history: List[Dict[str, str]]

    debug_logs: List[str]

    intent_hint_before_rewrite: str
    intent_hint_is_confident: bool
    intent_fallback_allowed: bool
    rewrite_intent_hint: str

    system_id_hint: Optional[str]
    system_id: Optional[str]
    intent: str

    render_type: str
    graph_data: Optional[Dict[str, Any]]
    query_meta: Optional[Dict[str, Any]]
    realtime_mode: Optional[str]
    structured_data: Optional[Dict[str, Any]]

    where: Optional[Dict[str, Any]]
    search_query: str
    search_result: Dict[str, Any]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    source_rows: List[Dict[str, Any]]

    result: AgentResult


@dataclass(frozen=True)
class ResponseRoute:
    name: str
    render_type: str
    realtime_mode: Optional[str] = None


def resolve_response_route(
    intent: str,
    render_type: str,
    has_graph: bool,
    has_query_meta: bool,
    query_meta: Optional[Dict[str, Any]] = None,
) -> ResponseRoute:
    """응답 라우팅을 render_type/query_meta 중심으로 결정한다."""
    query_meta = query_meta or {}

    if render_type == "graph" and has_graph:
        return ResponseRoute(name="graph", render_type="graph")

    if render_type == "chart" and has_query_meta:
        return ResponseRoute(
            name="chart",
            render_type="chart",
            realtime_mode=query_meta.get("realtime_mode") or "chart_only",
        )

    if render_type == "table" and has_query_meta:
        return ResponseRoute(
            name="table",
            render_type="table",
            realtime_mode=query_meta.get("realtime_mode"),
        )

    return ResponseRoute(name="llm_text", render_type=render_type)
