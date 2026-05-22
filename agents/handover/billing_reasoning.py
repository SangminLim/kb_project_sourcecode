from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .models import AgentResult, AgentWorkflowState
from .response_builder import build_chart_answer


def _as_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _contains_any_keyword(text: str, keywords: Sequence[Any]) -> bool:
    normalized = _text(text).lower()
    return any(_text(keyword).lower() in normalized for keyword in keywords if _text(keyword))


DEFAULT_BILLING_PLANNER_POLICY: Dict[str, Any] = {
    "enabled": True,
    "default_steps": [
        "prepare_realtime_query",
        "validate_chart_fields",
        "fetch_monthly_billing",
        "build_chart",
        "summarize_trend",
    ],
    "step_rules": [
        {
            "step": "summarize_trend",
            "include_when_any": ["요약", "정리", "추이", "흐름", "증감", "변동"],
            "reason": "사용자가 청구 데이터 요약 또는 추이 설명을 요청함",
        },
        {
            "step": "build_chart",
            "include_when_any": ["그래프", "차트", "시각화", "보여줘"],
            "reason": "사용자가 청구 데이터를 그래프로 확인하길 요청함",
        },
    ],
    "step_labels": {
        "prepare_realtime_query": "청구 데이터 조회 준비",
        "validate_chart_fields": "차트 컬럼 검증",
        "fetch_monthly_billing": "월별 청구 금액 조회",
        "build_chart": "월별 금액 그래프 생성",
        "summarize_trend": "증감 흐름 요약",
    },
}


def merge_billing_planner_policy(query_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """청구 그래프/요약 workflow 실행 계획 정책을 병합한다.

    기본값은 chart + summary의 안전한 표준 절차이고,
    업무별 단계/라벨/키워드는 realtime_queries[].planner_policy에서 덮어쓸 수 있다.
    """
    policy: Dict[str, Any] = dict(DEFAULT_BILLING_PLANNER_POLICY)
    policy["default_steps"] = list(DEFAULT_BILLING_PLANNER_POLICY.get("default_steps", []))
    policy["step_rules"] = [dict(rule) for rule in DEFAULT_BILLING_PLANNER_POLICY.get("step_rules", [])]
    policy["step_labels"] = dict(DEFAULT_BILLING_PLANNER_POLICY.get("step_labels", {}))

    if not query_meta:
        return policy

    override = query_meta.get("planner_policy") if isinstance(query_meta, Mapping) else None
    if not isinstance(override, Mapping):
        return policy

    for key, value in override.items():
        if key == "step_labels" and isinstance(value, Mapping):
            merged_labels = dict(policy.get("step_labels", {}))
            merged_labels.update(dict(value))
            policy[key] = merged_labels
        elif key == "step_rules" and isinstance(value, list):
            policy[key] = [dict(rule) for rule in value if isinstance(rule, Mapping)]
        elif key == "default_steps" and isinstance(value, list):
            policy[key] = [str(step) for step in value if str(step).strip()]
        else:
            policy[key] = value
    return policy


def build_billing_plan(question: str, query_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """질문과 설정 기반으로 청구 그래프/요약 workflow 실행 계획을 만든다.

    LLM이 임의로 도구를 선택하는 구조가 아니라,
    운영 설정 기반의 제한형 planner로 차트 생성과 요약 단계를 명시한다.
    """
    policy = merge_billing_planner_policy(query_meta)
    default_steps = [str(step) for step in policy.get("default_steps", []) if str(step).strip()]
    steps: List[str] = list(dict.fromkeys(default_steps))
    reasons: List[str] = []

    if not policy.get("enabled", True):
        return {
            "enabled": False,
            "steps": steps,
            "step_labels": policy.get("step_labels", {}),
            "reasons": ["planner_policy.enabled=false"],
        }

    for rule in policy.get("step_rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        step = _text(rule.get("step"))
        if not step:
            continue
        keywords = _as_list(rule.get("include_when_any"))
        include_by_default = bool(rule.get("enabled_by_default", False))
        include_by_keyword = _contains_any_keyword(question, keywords)
        if include_by_default or include_by_keyword:
            if step not in steps:
                steps.append(step)
            reason = _text(rule.get("reason")) or f"{step} 조건 충족"
            reasons.append(reason)

    required_order = [
        "prepare_realtime_query",
        "validate_chart_fields",
        "fetch_monthly_billing",
        "build_chart",
        "summarize_trend",
    ]
    for required_step in required_order:
        if required_step not in steps:
            steps.append(required_step)

    steps = [step for step in required_order if step in steps] + [
        step for step in steps if step not in required_order
    ]

    return {
        "enabled": True,
        "steps": list(dict.fromkeys(steps)),
        "step_labels": policy.get("step_labels", {}),
        "reasons": reasons or ["기본 청구 그래프/요약 절차 적용"],
    }


def billing_planner_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-B 0] node = billing_planner")

    query_meta = dict(state.get("query_meta") or {})
    question = (
        state.get("rewritten_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    plan = build_billing_plan(str(question), query_meta)

    query_meta["execution_plan"] = plan
    query_meta.setdefault("post_process", "billing_graph_reasoning")

    planned_steps = plan.get("steps", [])
    planner_reasons = plan.get("reasons", [])
    requires_chart = "build_chart" in planned_steps
    requires_summary = "summarize_trend" in planned_steps

    debug_logs.append(f"[BILLING PLAN 1] detected_request = {state.get('intent', 'billing_chart')}")
    debug_logs.append(f"[BILLING PLAN 2] selected_steps = {planned_steps}")
    debug_logs.append(f"[BILLING PLAN 3] requires_chart = {requires_chart}")
    debug_logs.append(f"[BILLING PLAN 4] requires_summary = {requires_summary}")
    debug_logs.append(f"[BILLING PLAN 5] planner_reasons = {planner_reasons}")
    debug_logs.append("[BILLING PLAN 6] selected_workflow = billing_graph_reasoning_flow")

    return {
        **state,
        "query_meta": query_meta,
        "billing_plan": plan,
        "debug_logs": debug_logs,
    }


def billing_prepare_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-B 1] node = billing_prepare")

    query_meta = dict(state.get("query_meta") or {})
    query_meta.setdefault("post_process", "billing_graph_reasoning")

    plan = state.get("billing_plan") or query_meta.get("execution_plan") or {}
    planned_steps = set(_as_list(plan.get("steps"))) if isinstance(plan, Mapping) else set()

    x_field = _text(query_meta.get("x_field") or "billing_month")
    y_field = _text(query_meta.get("y_field") or "amount")
    query_meta.setdefault("x_field", x_field)
    query_meta.setdefault("y_field", y_field)

    debug_logs.append(f"[BILLING PLAN 7] chart_fields = x:{x_field}, y:{y_field}")
    if "fetch_monthly_billing" in planned_steps:
        debug_logs.append("[BILLING PLAN 8] data_fetch = delegated_to_realtime_payload")
    if "build_chart" in planned_steps:
        debug_logs.append("[BILLING PLAN 9] chart_rendering = delegated_to_streamlit_renderer")
    if "summarize_trend" in planned_steps:
        debug_logs.append("[BILLING PLAN 10] trend_summary = delegated_to_realtime_summary")

    return {
        **state,
        "query_meta": query_meta,
        "debug_logs": debug_logs,
    }


def billing_respond_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-B 2] node = billing_respond")

    query_meta = dict(state.get("query_meta") or {})
    answer = build_chart_answer(query_meta)

    result = AgentResult(
        original_question=state.get("question", ""),
        normalized_question=state.get("normalized_question", ""),
        rewritten_question=state.get("rewritten_question", ""),
        system_id=state.get("system_id"),
        intent=state.get("intent", "billing_chart"),
        answer=answer,
        render_type=str(query_meta.get("render_type") or state.get("render_type") or "chart"),
        graph_data=None,
        query_meta=query_meta,
        realtime_mode=str(query_meta.get("realtime_mode") or "billing_graph_reasoning"),
        structured_data=None,
        sources=state.get("source_rows", []) or [],
        debug_logs=debug_logs,
    )
    return {**state, "query_meta": query_meta, "result": result, "debug_logs": debug_logs}
