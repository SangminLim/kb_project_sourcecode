from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import AgentResult, AgentWorkflowState
from .response_builder import build_table_answer


def _as_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


DEFAULT_INCIDENT_PLANNER_POLICY: Dict[str, Any] = {
    "enabled": True,
    "default_steps": ["prepare_realtime_query", "status_reasoning", "summarize_table"],
    "step_rules": [
        {
            "step": "action_guide",
            "include_when_any": ["조치", "조치방법", "해결", "원인", "담당", "대응"],
            "reason": "사용자가 원인/조치/담당자 확인을 요청함",
        },
        {
            "step": "impact_analysis",
            "include_when_any": ["영향", "고객영향", "후속", "연계", "후행", "영향도"],
            "reason": "사용자가 영향도 또는 후속 배치 확인을 요청함",
        },
    ],
    "step_labels": {
        "prepare_realtime_query": "장애현황 조회 준비",
        "status_reasoning": "상태/우선순위 판단",
        "action_guide": "조치 가이드 보강",
        "impact_analysis": "영향도 판단 보강",
        "summarize_table": "표 요약 답변 생성",
    },
}


DEFAULT_INCIDENT_REASONING_POLICY: Dict[str, Any] = {
    "enabled": True,
    "open_statuses": ["FAIL", "FAILED", "ERROR", "OPEN", "RUNNING", "처리중", "미처리", "실패", "오류"],
    "closed_statuses": ["SUCCESS", "DONE", "CLOSED", "RESOLVED", "종료", "완료", "조치완료", "정상"],
    "critical_severities": ["CRITICAL", "HIGH", "긴급", "높음", "심각"],
    "high_elapsed_minutes": 30,
    "medium_elapsed_minutes": 10,
    "retry_warn_count": 2,
    "column_aliases": {
        "batch_name": ["batch_name", "배치명", "job_name", "JOB명", "프로그램명"],
        "job_id": ["job_id", "배치ID", "JOB_ID", "jobId"],
        "system_name": ["system_name", "시스템명", "시스템", "SYSTEM_NAME"],
        "status": ["status", "상태", "STATUS"],
        "severity": ["severity", "심각도", "SEVERITY", "등급"],
        "error_code": ["error_code", "오류코드", "ERROR_CODE"],
        "error_message": ["error_message", "오류메시지", "ERROR_MESSAGE", "에러메시지"],
        "start_time": ["start_time", "오류발생시간", "발생시간", "START_TIME", "created_at"],
        "end_time": ["end_time", "종료시간", "END_TIME", "resolved_at"],
        "elapsed_minutes": ["elapsed_minutes", "경과분", "ELAPSED_MINUTES", "elapsed_min"],
        "retry_count": ["retry_count", "재시도횟수", "RETRY_COUNT"],
        "action_detail": ["action_detail", "조치내용", "ACTION_DETAIL"],
        "action_owner": ["action_owner", "담당자", "ACTION_OWNER", "owner"],
        "impact_yn": ["impact_yn", "영향여부", "고객영향", "customer_impact", "CUSTOMER_IMPACT"],
    },
    "output_columns": {
        "status_reason": "상태판단",
        "impact_level": "영향도",
        "priority": "우선순위",
        "long_running": "장기미처리여부",
        "has_action_record": "운영조치존재여부",
        "needs_additional_action": "추가조치필요여부",
        "reasoning_note": "판단근거",
        "recommended_action": "권장조치",
    },
}


def merge_incident_planner_policy(query_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """장애 workflow 실행 계획 정책을 병합한다.

    기본값은 안전한 운영 절차이고, 실제 업무별 키워드/단계는
    realtime_queries[].planner_policy에서 덮어쓸 수 있다.
    """
    policy: Dict[str, Any] = dict(DEFAULT_INCIDENT_PLANNER_POLICY)
    policy["default_steps"] = list(DEFAULT_INCIDENT_PLANNER_POLICY.get("default_steps", []))
    policy["step_rules"] = [dict(rule) for rule in DEFAULT_INCIDENT_PLANNER_POLICY.get("step_rules", [])]
    policy["step_labels"] = dict(DEFAULT_INCIDENT_PLANNER_POLICY.get("step_labels", {}))

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


def _contains_any_keyword(text: str, keywords: Sequence[Any]) -> bool:
    normalized = _text(text).lower()
    return any(_text(keyword).lower() in normalized for keyword in keywords if _text(keyword))


def build_incident_plan(question: str, query_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """질문과 설정 기반으로 장애현황 workflow 실행 계획을 만든다.

    LLM이 임의로 도구를 고르는 구조가 아니라, 운영 설정 기반의 제한형 planner다.
    그래서 재현성과 운영 안정성을 유지하면서도 질문에 따라 조치/영향도 단계를 확장할 수 있다.
    """
    policy = merge_incident_planner_policy(query_meta)
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
                # 답변 생성은 항상 마지막에 둔다.
                if "summarize_table" in steps:
                    steps.insert(max(0, len(steps) - 1), step)
                else:
                    steps.append(step)
            reason = _text(rule.get("reason")) or f"{step} 조건 충족"
            reasons.append(reason)

    if "prepare_realtime_query" not in steps:
        steps.insert(0, "prepare_realtime_query")
    if "summarize_table" not in steps:
        steps.append("summarize_table")

    return {
        "enabled": True,
        "steps": steps,
        "step_labels": policy.get("step_labels", {}),
        "reasons": reasons or ["기본 장애현황 조회 절차 적용"],
    }


def merge_incident_policy(query_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """query_meta.reasoning_policy를 기본 정책에 병합한다.

    운영 기준값은 코드가 아니라 realtime_queries[].reasoning_policy에서 바꿀 수 있다.
    """
    policy: Dict[str, Any] = dict(DEFAULT_INCIDENT_REASONING_POLICY)
    policy["column_aliases"] = dict(DEFAULT_INCIDENT_REASONING_POLICY["column_aliases"])
    policy["output_columns"] = dict(DEFAULT_INCIDENT_REASONING_POLICY["output_columns"])

    if not query_meta:
        return policy

    override = query_meta.get("reasoning_policy") if isinstance(query_meta, Mapping) else None
    if not isinstance(override, Mapping):
        return policy

    for key, value in override.items():
        if key in {"column_aliases", "output_columns"} and isinstance(value, Mapping):
            nested = dict(policy.get(key, {}))
            nested.update(dict(value))
            policy[key] = nested
        else:
            policy[key] = value
    return policy


def _get_by_alias(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in row:
            return row.get(key)
    lowered = {str(k).lower(): k for k in row.keys()}
    for key in aliases:
        found = lowered.get(str(key).lower())
        if found is not None:
            return row.get(found)
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _number(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    text = _text(value)
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M%S",
        "%Y%m%d %H%M%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00").replace("+09:00", ""))
    except Exception:
        return None


def _elapsed_minutes(row: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    aliases = policy.get("column_aliases", {})
    explicit = _get_by_alias(row, aliases.get("elapsed_minutes", []))
    if explicit not in (None, ""):
        return max(0, _number(explicit, 0))

    start = _parse_datetime(_get_by_alias(row, aliases.get("start_time", [])))
    end = _parse_datetime(_get_by_alias(row, aliases.get("end_time", []))) or datetime.now()
    if not start:
        return 0
    return max(0, int((end - start).total_seconds() // 60))


def _is_in(value: Any, candidates: Iterable[Any]) -> bool:
    norm = _upper(value)
    return any(norm == _upper(item) for item in candidates)


def _has_value(value: Any) -> bool:
    return bool(_text(value))


def _normalize_row(row: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    aliases = policy.get("column_aliases", {})
    normalized = dict(row)
    for canonical, key_aliases in aliases.items():
        normalized[f"__{canonical}"] = _get_by_alias(row, _as_list(key_aliases))
    normalized["__elapsed_minutes"] = _elapsed_minutes(row, policy)
    return normalized


def reason_incident_row(row: Mapping[str, Any], policy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """장애 1건에 상태/영향도/우선순위 판단을 붙인다.

    판단 기준은 query_meta.reasoning_policy로 바꿀 수 있고, LLM이 임의로 판정을 바꾸지 않게
    deterministic rule로 먼저 산정한다.
    """
    policy = dict(policy or DEFAULT_INCIDENT_REASONING_POLICY)
    nrow = _normalize_row(row, policy)

    status = nrow.get("__status")
    severity = nrow.get("__severity")
    action_detail = nrow.get("__action_detail")
    action_owner = nrow.get("__action_owner")
    impact_yn = nrow.get("__impact_yn")
    retry_count = _number(nrow.get("__retry_count"), 0)
    elapsed = _number(nrow.get("__elapsed_minutes"), 0)

    is_open = _is_in(status, policy.get("open_statuses", [])) or not _is_in(status, policy.get("closed_statuses", []))
    is_critical = _is_in(severity, policy.get("critical_severities", []))
    has_impact = _upper(impact_yn) in {"Y", "YES", "TRUE", "1", "있음", "영향"}
    # 운영 조치 여부와 추가 조치 필요 여부는 의미가 다르므로 분리한다.
    # - 운영조치존재여부: 조치내용 또는 담당자가 하나라도 등록되어 있는지
    # - 추가조치필요여부: 미종료 장애인데 조치내용/담당자 중 하나라도 누락되었는지
    has_action_detail = _has_value(action_detail)
    has_action_owner = _has_value(action_owner)
    has_action_record = has_action_detail or has_action_owner
    action_info_incomplete = not (has_action_detail and has_action_owner)
    needs_additional_action = is_open and action_info_incomplete
    long_running = is_open and elapsed >= _number(policy.get("high_elapsed_minutes"), 30)
    medium_running = is_open and elapsed >= _number(policy.get("medium_elapsed_minutes"), 10)
    retry_warn = retry_count >= _number(policy.get("retry_warn_count"), 2)

    reason_parts: List[str] = []
    if is_open:
        reason_parts.append("미종료 상태")
    if long_running:
        reason_parts.append(f"경과 {elapsed}분")
    elif medium_running:
        reason_parts.append(f"경과 {elapsed}분")
    if is_critical:
        reason_parts.append(f"심각도 {severity}")
    if has_impact:
        reason_parts.append("영향도 표시 있음")
    if retry_warn:
        reason_parts.append(f"재시도 {retry_count}회")
    if not has_action_record:
        reason_parts.append("운영 조치정보 미등록")
    elif action_info_incomplete:
        reason_parts.append("운영 조치정보 일부 누락")
    else:
        reason_parts.append("운영 조치정보 등록됨")

    if is_open and (is_critical or has_impact or long_running):
        priority = "P1"
        impact_level = "높음"
    elif is_open and (medium_running or retry_warn or needs_additional_action):
        priority = "P2"
        impact_level = "중간"
    elif is_open:
        priority = "P3"
        impact_level = "낮음"
    else:
        priority = "P4"
        impact_level = "종료/확인"

    if not is_open:
        status_reason = "종료 장애"
        recommended_action = "완료 여부와 재발 방지 조치만 확인"
    elif priority == "P1":
        status_reason = "즉시 조치 필요"
        recommended_action = "담당자 확인 후 우선 조치 및 영향 업무 공지"
    elif priority == "P2":
        status_reason = "우선 확인 필요"
        recommended_action = "경과시간/조치내역 확인 후 담당자 배정"
    else:
        status_reason = "모니터링 필요"
        recommended_action = "상태 변화와 재시도 결과 확인"

    output_columns = policy.get("output_columns", {})
    enriched = dict(row)
    enriched[str(output_columns.get("status_reason", "상태판단"))] = status_reason
    enriched[str(output_columns.get("impact_level", "영향도"))] = impact_level
    enriched[str(output_columns.get("priority", "우선순위"))] = priority
    enriched[str(output_columns.get("long_running", "장기미처리여부"))] = long_running
    enriched[str(output_columns.get("has_action_record", "운영조치존재여부"))] = has_action_record
    enriched[str(output_columns.get("needs_additional_action", "추가조치필요여부"))] = needs_additional_action

    # 기존 화면/이력 호환용 영문 key도 함께 제공한다.
    enriched["is_long_running"] = long_running
    enriched["has_action_record"] = has_action_record
    enriched["needs_additional_action"] = needs_additional_action

    enriched[str(output_columns.get("reasoning_note", "판단근거"))] = ", ".join(reason_parts) if reason_parts else "추가 위험 신호 없음"
    enriched[str(output_columns.get("recommended_action", "권장조치"))] = recommended_action
    return enriched


def apply_incident_reasoning(rows: Sequence[Mapping[str, Any]], query_meta: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    policy = merge_incident_policy(query_meta)
    if not policy.get("enabled", True):
        return [dict(row) for row in rows]
    return [reason_incident_row(row, policy) for row in rows]


def summarize_incident_rows(rows: Sequence[Mapping[str, Any]], query_meta: Optional[Mapping[str, Any]] = None) -> str:
    if not rows:
        title = "장애현황"
        if query_meta:
            title = str(query_meta.get("title") or title)
        return f"{title} 조회 결과가 없습니다."

    policy = merge_incident_policy(query_meta)
    cols = policy.get("output_columns", {})
    priority_col = str(cols.get("priority", "우선순위"))
    status_col = str(cols.get("status_reason", "상태판단"))

    counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for row in rows:
        priority = _text(row.get(priority_col) or "미분류")
        status = _text(row.get(status_col) or "미분류")
        counts[priority] = counts.get(priority, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    priority_order = ["P1", "P2", "P3", "P4", "미분류"]
    count_text = ", ".join([f"{key} {counts[key]}건" for key in priority_order if key in counts])
    status_text = ", ".join([f"{key} {value}건" for key, value in status_counts.items()])

    first_p1 = next((row for row in rows if _text(row.get(priority_col)) == "P1"), None)
    highlight = ""
    if first_p1:
        batch = _text(first_p1.get("배치명") or first_p1.get("batch_name") or first_p1.get("JOB명"))
        reason = _text(first_p1.get(str(cols.get("reasoning_note", "판단근거"))))
        highlight = f" 우선 확인 대상은 {batch or 'P1 장애'}이며, 판단근거는 {reason or '상세 확인 필요'}입니다."

    return f"총 {len(rows)}건의 장애를 확인했습니다. 우선순위는 {count_text or '미분류'}입니다. 상태판단은 {status_text or '미분류'}입니다.{highlight}"


def _extract_incident_rows(state: AgentWorkflowState) -> List[Mapping[str, Any]]:
    for key in ("incident_rows", "realtime_rows", "query_result_rows", "rows"):
        value = state.get(key)  # type: ignore[arg-type]
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def incident_planner_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-I 0] node = incident_planner")

    query_meta = dict(state.get("query_meta") or {})
    question = (
        state.get("rewritten_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    plan = build_incident_plan(str(question), query_meta)

    query_meta["execution_plan"] = plan

    planned_steps = plan.get("steps", [])
    planner_reasons = plan.get("reasons", [])
    requires_reasoning = any(
        step in planned_steps
        for step in ["status_reasoning", "impact_analysis", "action_guide"]
    )
    requires_action_summary = "action_guide" in planned_steps
    requires_impact_analysis = "impact_analysis" in planned_steps

    debug_logs.append(f"[PLAN 1] detected_request = {state.get('intent', 'incident_status')}")
    debug_logs.append(f"[PLAN 2] selected_steps = {planned_steps}")
    debug_logs.append(f"[PLAN 3] requires_reasoning = {requires_reasoning}")
    debug_logs.append(f"[PLAN 4] requires_action_summary = {requires_action_summary}")
    debug_logs.append(f"[PLAN 5] requires_impact_analysis = {requires_impact_analysis}")
    debug_logs.append(f"[PLAN 6] planner_reasons = {planner_reasons}")
    debug_logs.append("[PLAN 7] selected_workflow = incident_reasoning_flow")

    return {
        **state,
        "query_meta": query_meta,
        "incident_plan": plan,
        "debug_logs": debug_logs,
    }


def incident_prepare_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-I 1] node = incident_prepare")
    query_meta = dict(state.get("query_meta") or {})
    query_meta.setdefault("post_process", "incident_reasoning")
    query_meta.setdefault("reasoning_policy", merge_incident_policy(query_meta))
    return {**state, "query_meta": query_meta, "incident_policy": query_meta.get("reasoning_policy"), "debug_logs": debug_logs}


def incident_reason_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-I 2] node = incident_reason")
    rows = _extract_incident_rows(state)

    # Streamlit realtime renderer가 DB/샘플 rows를 별도 계층에서 주입하는 구조에서는
    # LangGraph state에 rows가 없을 수 있다. 이 경우 그래프는 실행계획만 query_meta에 싣고,
    # 실제 표/요약/멀티 reasoning 렌더링은 기존 realtime renderer에 위임한다.
    if not rows:
        debug_logs.append("[PLAN 8] row_reasoning = delegated_to_realtime_renderer")
        return {
            **state,
            "incident_rows": [],
            "incident_reasoned_rows": [],
            "incident_summary": "",
            "debug_logs": debug_logs,
        }

    plan = state.get("incident_plan") or (state.get("query_meta") or {}).get("execution_plan") or {}
    planned_steps = set(_as_list(plan.get("steps"))) if isinstance(plan, Mapping) else set()
    if "status_reasoning" in planned_steps or "impact_analysis" in planned_steps or "action_guide" in planned_steps:
        reasoned_rows = apply_incident_reasoning(rows, state.get("query_meta") or {})
        summary = summarize_incident_rows(reasoned_rows, state.get("query_meta") or {})
        debug_logs.append("[INCIDENT 2] status_reasoning = applied")
        if "impact_analysis" in planned_steps:
            debug_logs.append("[INCIDENT 2-1] impact_analysis = included_by_plan")
        if "action_guide" in planned_steps:
            debug_logs.append("[INCIDENT 2-2] action_guide = included_by_plan")
    else:
        reasoned_rows = [dict(row) for row in rows]
        summary = summarize_incident_rows(reasoned_rows, state.get("query_meta") or {})
        debug_logs.append("[INCIDENT 2] status_reasoning = skipped_by_plan")
    debug_logs.append(f"[INCIDENT 1] rows = {len(rows)}")
    debug_logs.append("[INCIDENT 3] priority_ranking = applied_when_planned")
    return {
        **state,
        "incident_rows": [dict(row) for row in rows],
        "incident_reasoned_rows": reasoned_rows,
        "incident_summary": summary,
        "debug_logs": debug_logs,
    }


def incident_respond_node(agent: Any, state: AgentWorkflowState) -> AgentWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[LG-I 3] node = incident_respond")

    query_meta = dict(state.get("query_meta") or {})
    reasoned_rows = state.get("incident_reasoned_rows") or []

    # 중복 응답 방지:
    # LangGraph incident planner는 execution_plan/post_process/reasoning_policy를 query_meta에 싣는 역할만 한다.
    # 실제 장애현황 표, 요약, Multi Reasoning Step은 기존 realtime renderer가 담당하므로
    # 여기서 별도 summary/plan_text를 answer에 붙이지 않는다.
    if reasoned_rows:
        query_meta["rows"] = reasoned_rows

    answer = build_table_answer(query_meta)

    result = AgentResult(
        original_question=state.get("question", ""),
        normalized_question=state.get("normalized_question", ""),
        rewritten_question=state.get("rewritten_question", ""),
        system_id=state.get("system_id"),
        intent=state.get("intent", "incident_status"),
        answer=answer,
        render_type=str(query_meta.get("render_type") or state.get("render_type") or "table"),
        graph_data=None,
        query_meta=query_meta,
        realtime_mode=str(query_meta.get("realtime_mode") or "incident_status"),
        structured_data=None,
        sources=state.get("source_rows", []) or [],
        debug_logs=debug_logs,
    )
    return {**state, "query_meta": query_meta, "result": result, "debug_logs": debug_logs}
