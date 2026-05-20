from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TypedDict

from ..context import *
from ..state import build_chat_history
from ..services.realtime_payload import enrich_result_with_realtime_payload
from ..sql.sql_analysis_ui import run_sql_analysis_request
from ..batch.batch_ui import run_batch_development, run_batch_llm_validation, run_batch_sql_improvement
from agents.batch_dev.classifier.request_classifier import detect_structured_request_type

def unique_preserve_order(items: List[Any]) -> List[str]:
    """순서를 유지하면서 중복 값을 제거한다.

    workflow 레이어에서 graph_trace/debug_logs 중복 제거에 사용한다.
    renderers 모듈에 의존하지 않도록 workflow 내부에 작은 순수 함수로 둔다.
    """
    result: List[str] = []
    seen = set()
    for item in items or []:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_general_fallback_chat_config() -> Any:
    """일반 질문 fallback용 ChatConfig를 만든다.

    app.py가 특정 LLM provider에 묶이지 않도록 프로젝트 공통 ChatConfig를 우선 사용하고,
    생성자 차이가 있으면 SimpleNamespace로 안전하게 대체한다.
    """
    candidates = [
        {
            "model": GENERAL_FALLBACK_MODEL,
            "temperature": GENERAL_FALLBACK_TEMPERATURE,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
            "max_tokens": GENERAL_FALLBACK_MAX_TOKENS,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
            "temperature": GENERAL_FALLBACK_TEMPERATURE,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
        },
    ]
    for kwargs in candidates:
        try:
            return ChatConfig(**kwargs)
        except TypeError:
            continue
    return SimpleNamespace(
        model=GENERAL_FALLBACK_MODEL,
        temperature=GENERAL_FALLBACK_TEMPERATURE,
        timeout=GENERAL_FALLBACK_TIMEOUT,
        max_tokens=GENERAL_FALLBACK_MAX_TOKENS,
    )


def build_out_of_scope_message(question: str = "") -> str:
    """업무 범위 밖 질문 안내 문구를 한 곳에서 관리한다."""
    return """
현재 질문은 업무 인수인계 지원 범위를 벗어난 일반 질문으로 판단되었습니다.

이 에이전트는 현재 카드업무 인수인계 및 운영 지원 기능에 특화되어 있습니다.

지원 기능 예시:
- 업무 개요 조회
- 배치 프로세스 조회
- 배치 흐름도 조회
- 테이블 리니지 조회
- 청구 이용내역서 월별 금액 조회
- 오늘 장애현황 조회
- 배치 개발 요청
- SQL 분석 및 개선 검토

예시 질문:
- "소득공제 업무 개요 알려줘"
- "청구 배치 흐름도 보여줘"
- "오늘 장애현황 조회"
- "가맹점 월정산 배치 개발 요청"
- "이 SQL 문제점 분석해줘"

일반 질문 응답을 사용하려면 GENERAL_FALLBACK_USE_LLM=true 로 설정하세요.
""".strip()


def _looks_like_out_of_scope_answer(answer: str) -> bool:
    """기존 Agent가 반환한 범위 밖 안내 문구를 감지한다.

    llm.py 내부 문구가 일부 바뀌어도 핵심 표현 기준으로 보수적으로 감지한다.
    """
    text = str(answer or "")
    patterns = [
        "업무 인수인계 범위",
        "지원 범위",
        "현재 업무 인수인계",
        "질문만 지원",
        "지원하지 않는 질문",
    ]
    return any(pattern in text for pattern in patterns)


def run_general_fallback(user_question: str, chat_history: Optional[List[Dict[str, str]]] = None, reason: str = "out_of_scope") -> Any:
    """업무 범위 밖 일반 질문을 처리한다.

    - 설정으로 LLM fallback을 끌 수 있다.
    - LLM 호출 실패 시에도 사용자에게 딱딱한 차단 문구가 아니라 자연스러운 안내를 보여준다.
    - 업무 전용 RAG/DB/배치 로직과 분리해 운영 영향도를 낮춘다.
    """
    normalized_question = apply_dictionary_rewrite(user_question)
    answer = build_out_of_scope_message(user_question)
    generated_by = "guide"
    error_message = None

    if GENERAL_FALLBACK_USE_LLM:
        system_prompt = """
너는 업무 인수인계 에이전트의 일반 질문 fallback assistant다.
사용자가 업무 범위 밖의 일반 질문을 하면 한국어로 간단하고 실용적으로 답한다.
단, 회사 내부 데이터/DB/문서가 필요한 척 추측하지 않는다.
최신 정보, 날씨, 주가, 법률/의료/금융 투자판단처럼 실시간성 또는 전문성이 필요한 내용은 확인 필요성을 명확히 말한다.
""".strip()
        history_text = ""
        for item in (chat_history or [])[-6:]:
            role = item.get("role", "")
            content = str(item.get("content", "")).strip()
            if role and content:
                history_text += f"{role}: {content}\n"
        user_prompt = f"""
최근 대화:
{history_text or "(없음)"}

사용자 질문:
{user_question}

답변 조건:
- 업무 인수인계 범위 밖 질문임을 길게 반복하지 말 것
- 가능한 경우 바로 답변할 것
- 실시간 정보가 필요한 질문이면 현재 시스템에서는 실시간 조회가 필요하다고 말하고, 확인 방법을 짧게 안내할 것
""".strip()
        try:
            raw = ollama_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                config=_build_general_fallback_chat_config(),
            )
            if isinstance(raw, dict):
                answer = str(raw.get("answer") or raw.get("content") or raw)
            else:
                answer = str(raw or "").strip() or answer
            generated_by = "llm_fallback"
        except Exception as exc:
            logger.exception("General fallback generation failed")
            error_message = str(exc)
            answer = build_out_of_scope_message(user_question)

    return SimpleNamespace(
        answer=answer,
        intent="general_fallback",
        render_type="text",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=normalized_question,
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            f"[GENERAL_FALLBACK 1] reason={reason}",
            f"[GENERAL_FALLBACK 2] enabled={GENERAL_FALLBACK_USE_LLM}",
            f"[GENERAL_FALLBACK 3] generated_by={generated_by}",
            f"[GENERAL_FALLBACK 4] error={error_message}" if error_message else "[GENERAL_FALLBACK 4] error=None",
        ],
    )

class HandoverGraphState(TypedDict, total=False):
    """LangGraph에서 공유하는 상태.

    Streamlit 화면 상태와 분리하고, 질문 처리에 필요한 값만 그래프 상태로 전달한다.
    """
    question: str
    chat_history: List[Dict[str, str]]
    force_sql_analysis: bool
    normalized_question: str
    initial_intent: str
    route: str
    result: Any
    batch_dev_raw: Any
    graph_trace: List[str]
    error: str


def _append_graph_trace(state: HandoverGraphState, message: str) -> List[str]:
    trace = list(state.get("graph_trace") or [])
    trace.append(message)
    return trace


def _agent_graph_routes() -> List[Dict[str, Any]]:
    """agent_graph_policy.json 기반 라우팅 정책을 반환한다.

    intent가 추가되면 코드 수정 없이 conf/agent_graph_policy.json의 routes만 확장하면 된다.
    """
    routes = AGENT_GRAPH_POLICY.get("routes", []) if isinstance(AGENT_GRAPH_POLICY, dict) else []
    return [route for route in routes if isinstance(route, dict)]


def _resolve_route_by_intent(intent: str) -> str:
    """intent를 실행 노드명으로 변환한다.

    1. agent_graph_policy.json routes 우선
    2. intent 값이 실제 그래프 실행 노드명과 같으면 직접 라우팅
    3. 그 외 fallback
    """
    normalized_intent = str(intent or "default").strip()

    for route in _agent_graph_routes():
        node = str(route.get("node") or route.get("name") or "").strip()
        intents = [str(item).strip() for item in route.get("intents", []) if str(item).strip()]
        if node and normalized_intent in intents:
            return node

    policy_nodes = set()
    graph_nodes = AGENT_GRAPH_POLICY.get("graph_nodes") if isinstance(AGENT_GRAPH_POLICY, dict) else None
    if isinstance(graph_nodes, dict):
        policy_nodes.update(str(node).strip() for node in graph_nodes.keys() if str(node).strip())

    for route in _agent_graph_routes():
        node = str(route.get("node") or route.get("name") or "").strip()
        if node:
            policy_nodes.add(node)

    # build_handover_graph에서 실제 add_node로 등록하는 실행 노드명.
    # 업무명/테이블명/배치명 하드코딩이 아니라 그래프 노드명 안전망이다.
    runtime_nodes = {
        "sql_analysis",
        "batch_development",
        "handover_agent",
        "general_fallback",
    }

    if normalized_intent in policy_nodes or normalized_intent in runtime_nodes:
        return normalized_intent

    return str(AGENT_GRAPH_POLICY.get("fallback_node") or "general_fallback")


def _env_flag(name: str, default: bool = True) -> bool:
    """정책 파일에서 지정한 환경변수 flag를 bool로 읽는다."""
    if not name:
        return default
    return os.getenv(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "y"}


def _graph_node_enabled(node_name: str) -> bool:
    """agent_graph_policy.json의 node_options에 따라 노드 실행 여부를 판단한다."""
    options = {}
    if isinstance(AGENT_GRAPH_POLICY, dict):
        options = (AGENT_GRAPH_POLICY.get("node_options") or {}).get(node_name, {}) or {}
    if not isinstance(options, dict):
        return True
    if "enabled" in options:
        return bool(options.get("enabled"))
    enabled_env = str(options.get("enabled_env") or "").strip()
    if enabled_env:
        return _env_flag(enabled_env, default=False)
    return True


def _post_nodes_for_route(route: str) -> List[str]:
    """실행 노드 이후에 수행할 후처리 노드 목록을 정책 기반으로 반환한다."""
    if not isinstance(AGENT_GRAPH_POLICY, dict):
        return ["finalize"]
    post_nodes = AGENT_GRAPH_POLICY.get("post_nodes") or {}
    if not isinstance(post_nodes, dict):
        return ["finalize"]
    route_key = str(route or "default").strip()
    nodes = post_nodes.get(route_key, post_nodes.get("default", ["finalize"]))
    if not isinstance(nodes, list):
        return ["finalize"]
    return [str(node).strip() for node in nodes if str(node).strip()]


def _first_post_node(state: HandoverGraphState) -> str:
    """현재 route의 첫 번째 후처리 노드를 반환한다."""
    for node_name in _post_nodes_for_route(str(state.get("route") or "")):
        if _graph_node_enabled(node_name):
            return node_name
    return "finalize"


def _next_post_node(state: HandoverGraphState, current_node: str) -> str:
    """후처리 체인에서 다음 노드를 반환한다."""
    nodes = _post_nodes_for_route(str(state.get("route") or ""))
    try:
        start_idx = nodes.index(current_node) + 1
    except ValueError:
        start_idx = 0
    for node_name in nodes[start_idx:]:
        if _graph_node_enabled(node_name):
            return node_name
    return "finalize"


def _attach_graph_trace_to_result(result: Any, trace: List[str]) -> Any:
    """그래프 trace를 결과 객체 debug_logs에 중복 없이 붙인다."""
    if result is None:
        return result
    debug_logs = list(getattr(result, "debug_logs", []) or [])
    debug_logs.extend(trace or [])
    setattr(result, "debug_logs", unique_preserve_order(debug_logs))
    return result


def graph_classify_node(state: HandoverGraphState) -> HandoverGraphState:
    """질문을 표준화하고 1차 intent를 판별한다.

    구조화된 배치 요청서는 일반 자연어 intent 분류보다 먼저 감지한다.
    """
    question = str(state.get("question") or "").strip()
    normalized_question = apply_dictionary_rewrite(question)

    structured_intent = detect_structured_request_type(question)
    detected_intent = detect_intent(normalized_question)
    initial_intent = structured_intent or detected_intent

    trace = _append_graph_trace(
        state,
        f"[GRAPH 1] classify intent={initial_intent}",
    )
    trace.append(f"[GRAPH 1-1] structured_intent={structured_intent}")
    trace.append(f"[GRAPH 1-2] detected_intent={detected_intent}")

    return {
        **state,
        "question": question,
        "normalized_question": normalized_question,
        "initial_intent": initial_intent,
        "graph_trace": trace,
    }


def graph_route_node(state: HandoverGraphState) -> HandoverGraphState:
    """강제 SQL 분석 여부와 intent 정책에 따라 실행 노드를 결정한다."""
    if state.get("force_sql_analysis"):
        route = str(AGENT_GRAPH_POLICY.get("force_sql_analysis_node") or "sql_analysis")
    else:
        route = _resolve_route_by_intent(str(state.get("initial_intent") or "default"))

    trace = _append_graph_trace(state, f"[GRAPH 2] route={route}")
    trace.append(f"[GRAPH 2-1] initial_intent={state.get('initial_intent')}")

    return {
        **state,
        "route": route,
        "graph_trace": trace,
    }


def graph_next_node(state: HandoverGraphState) -> str:
    """LangGraph conditional edge에서 사용할 실행 노드명.

    구조화 배치요청서에서 판별된 route가
    agent_graph_policy.json 누락 때문에 fallback 되지 않도록
    실제 실행 가능한 graph node 기준으로 허용한다.
    """
    route = str(state.get("route") or "general_fallback").strip()

    allowed_nodes = {
        "sql_analysis",
        "batch_development",
        "handover_agent",
        "general_fallback",
    }

    return route if route in allowed_nodes else "general_fallback"


def graph_after_execute_node(state: HandoverGraphState) -> str:
    """실행 노드 이후 첫 후처리 노드명."""
    return _first_post_node(state)


def graph_after_batch_validation_node(state: HandoverGraphState) -> str:
    """batch_validation 이후 다음 후처리 노드명."""
    return _next_post_node(state, "batch_validation")


def graph_after_sql_improvement_node(state: HandoverGraphState) -> str:
    """sql_improvement 이후 다음 후처리 노드명."""
    return _next_post_node(state, "sql_improvement")


def graph_sql_analysis_node(state: HandoverGraphState) -> HandoverGraphState:
    result = run_sql_analysis_request(str(state.get("question") or ""))
    debug_logs = list(getattr(result, "debug_logs", []) or [])
    debug_logs.extend(state.get("graph_trace") or [])
    setattr(result, "debug_logs", debug_logs)
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 3] executed=sql_analysis"),
    }


def graph_batch_development_node(state: HandoverGraphState) -> HandoverGraphState:
    """배치 개발 파일 생성 노드.

    검증/개선은 별도 LangGraph 노드에서 수행한다.
    """
    user_question = str(state.get("question") or "")
    dev_result = BatchDevAgent().run(user_question)
    answer = dev_result.message
    if dev_result.errors:
        answer = "배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요."

    result = SimpleNamespace(
        answer=answer,
        intent="batch_development",
        render_type="batch_dev",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=apply_dictionary_rewrite(user_question),
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            "[BATCH_DEV 1] intent=batch_development",
            f"[BATCH_DEV 2] success={dev_result.success}",
            f"[BATCH_DEV 3] created_files={len(dev_result.created_files)}",
        ],
        batch_dev_result={
            "batch_spec": dev_result.batch_spec,
            "created_files": dev_result.created_files,
            "warnings": dev_result.warnings,
            "errors": dev_result.errors,
            "message": dev_result.message,
            "success": dev_result.success,
            "validation_report": None,
            "sql_improvement": None,
        },
    )
    return {
        **state,
        "result": result,
        "batch_dev_raw": dev_result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 3] executed=batch_development.generate"),
    }


def graph_batch_validation_node(state: HandoverGraphState) -> HandoverGraphState:
    """배치 생성 결과 검증 노드."""
    result = state.get("result")
    dev_result = state.get("batch_dev_raw")
    if result is None or dev_result is None:
        return {
            **state,
            "graph_trace": _append_graph_trace(state, "[GRAPH 4] skipped=batch_validation:no_result"),
        }

    payload = getattr(result, "batch_dev_result", {}) or {}
    if not payload.get("success"):
        return {
            **state,
            "graph_trace": _append_graph_trace(state, "[GRAPH 4] skipped=batch_validation:generation_failed"),
        }

    validation_report = run_batch_llm_validation(str(state.get("question") or ""), dev_result)
    payload["validation_report"] = validation_report
    setattr(result, "batch_dev_result", payload)
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 4] executed=batch_validation"),
    }


def graph_sql_improvement_node(state: HandoverGraphState) -> HandoverGraphState:
    """배치 SQL 개선 제안 노드."""
    result = state.get("result")
    dev_result = state.get("batch_dev_raw")
    if result is None or dev_result is None:
        return {
            **state,
            "graph_trace": _append_graph_trace(state, "[GRAPH 5] skipped=sql_improvement:no_result"),
        }

    payload = getattr(result, "batch_dev_result", {}) or {}
    if not payload.get("success"):
        return {
            **state,
            "graph_trace": _append_graph_trace(state, "[GRAPH 5] skipped=sql_improvement:generation_failed"),
        }

    sql_improvement = run_batch_sql_improvement(dev_result)
    payload["sql_improvement"] = sql_improvement
    setattr(result, "batch_dev_result", payload)
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 5] executed=sql_improvement"),
    }


def graph_finalize_node(state: HandoverGraphState) -> HandoverGraphState:
    """공통 최종 정리 노드.

    모든 route의 마지막에서 trace를 result.debug_logs에 합쳐 평가 패널에서 확인 가능하게 한다.
    """
    result = _attach_graph_trace_to_result(state.get("result"), state.get("graph_trace") or [])
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH FINAL] finalized"),
    }


def graph_handover_agent_node(state: HandoverGraphState) -> HandoverGraphState:
    agent = get_agent()
    result = agent.answer_question(
        question=str(state.get("question") or ""),
        chat_history=state.get("chat_history") or [],
    )
    result = enrich_result_with_realtime_payload(result)
    if _looks_like_out_of_scope_answer(getattr(result, "answer", "")):
        result = run_general_fallback(
            str(state.get("question") or ""),
            state.get("chat_history") or [],
            reason="agent_out_of_scope_message",
        )
    debug_logs = list(getattr(result, "debug_logs", []) or [])
    debug_logs.extend(state.get("graph_trace") or [])
    setattr(result, "debug_logs", debug_logs)
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 3] executed=handover_agent"),
    }


def graph_general_fallback_node(state: HandoverGraphState) -> HandoverGraphState:
    result = run_general_fallback(
        str(state.get("question") or ""),
        state.get("chat_history") or [],
        reason=f"unsupported_intent:{state.get('initial_intent')}",
    )
    debug_logs = list(getattr(result, "debug_logs", []) or [])
    debug_logs.extend(state.get("graph_trace") or [])
    setattr(result, "debug_logs", debug_logs)
    return {
        **state,
        "result": result,
        "graph_trace": _append_graph_trace(state, "[GRAPH 3] executed=general_fallback"),
    }


def build_handover_graph() -> Any:
    """업무 인수인계 질문 처리 그래프를 생성한다.

    노드 추가/intent 추가는 agent_graph_policy.json과 노드 함수 확장으로 처리한다.
    LangGraph가 설치되지 않았거나 비활성화된 경우 None을 반환하고 legacy runner로 fallback한다.
    """
    if not LANGGRAPH_ENABLED or StateGraph is None:
        return None

    workflow = StateGraph(HandoverGraphState)
    workflow.add_node("classify", graph_classify_node)
    workflow.add_node("route", graph_route_node)
    workflow.add_node("sql_analysis", graph_sql_analysis_node)
    workflow.add_node("batch_development", graph_batch_development_node)
    workflow.add_node("batch_validation", graph_batch_validation_node)
    workflow.add_node("sql_improvement", graph_sql_improvement_node)
    workflow.add_node("handover_agent", graph_handover_agent_node)
    workflow.add_node("general_fallback", graph_general_fallback_node)
    workflow.add_node("finalize", graph_finalize_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "route")
    workflow.add_conditional_edges(
        "route",
        graph_next_node,
        {
            "sql_analysis": "sql_analysis",
            "batch_development": "batch_development",
            "handover_agent": "handover_agent",
            "general_fallback": "general_fallback",
        },
    )

    for node_name in ["sql_analysis", "batch_development", "handover_agent", "general_fallback"]:
        workflow.add_conditional_edges(
            node_name,
            graph_after_execute_node,
            {
                "batch_validation": "batch_validation",
                "sql_improvement": "sql_improvement",
                "finalize": "finalize",
            },
        )

    workflow.add_conditional_edges(
        "batch_validation",
        graph_after_batch_validation_node,
        {
            "sql_improvement": "sql_improvement",
            "finalize": "finalize",
        },
    )
    workflow.add_conditional_edges(
        "sql_improvement",
        graph_after_sql_improvement_node,
        {"finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()


@st.cache_resource
def get_handover_graph() -> Any:
    return build_handover_graph()


def run_legacy_handover_flow(
    user_question: str,
    chat_history: List[Dict[str, str]],
    force_sql_analysis: bool = False,
) -> Any:
    """LangGraph 비활성/미설치 시 기존 처리 흐름을 그대로 수행한다."""
    normalized_for_intent = apply_dictionary_rewrite(user_question)
    structured_intent = detect_structured_request_type(user_question)
    initial_intent = structured_intent or detect_intent(normalized_for_intent)

    if force_sql_analysis:
        return run_sql_analysis_request(user_question)
    if initial_intent == "batch_development":
        return run_batch_development(user_question)
    if not is_supported_intent(initial_intent):
        return run_general_fallback(user_question, chat_history, reason=f"unsupported_intent:{initial_intent}")

    agent = get_agent()
    result = agent.answer_question(question=user_question, chat_history=chat_history)
    result = enrich_result_with_realtime_payload(result)
    if _looks_like_out_of_scope_answer(getattr(result, "answer", "")):
        result = run_general_fallback(user_question, chat_history, reason="agent_out_of_scope_message")
    return result


def run_handover_graph(
    user_question: str,
    chat_history: List[Dict[str, str]],
    force_sql_analysis: bool = False,
) -> Any:
    """Streamlit main에서 호출하는 단일 진입점."""
    graph = get_handover_graph()
    if graph is None:
        result = run_legacy_handover_flow(user_question, chat_history, force_sql_analysis)
        debug_logs = list(getattr(result, "debug_logs", []) or [])
        debug_logs.append("[GRAPH 0] LangGraph disabled or not installed; legacy flow used")
        setattr(result, "debug_logs", debug_logs)
        return result

    state = graph.invoke(
        {
            "question": user_question,
            "chat_history": chat_history,
            "force_sql_analysis": force_sql_analysis,
            "graph_trace": [],
        }
    )
    result = state.get("result")
    if result is None:
        return run_general_fallback(user_question, chat_history, reason="graph_empty_result")
    debug_logs = list(getattr(result, "debug_logs", []) or [])
    debug_logs.extend(state.get("graph_trace") or [])
    setattr(result, "debug_logs", unique_preserve_order(debug_logs))
    return result
