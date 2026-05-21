from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import AgentResult, AgentWorkflowState


def _as_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


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
        "reasoning_note": "판단근거",
        "recommended_action": "권장조치",
    },
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
    action_missing = not _has_value(action_detail) or not _has_value(action_owner)
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
    if action_missing:
        reason_parts.append("조치정보 미흡")

    if is_open and (is_critical or has_impact or long_running):
        priority = "P1"
        impact_level = "높음"
    elif is_open and (medium_running or retry_warn or action_missing):
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
    if not rows:
        debug_logs.append("[INCIDENT 1] rows = empty_or_not_yet_loaded")
        return {**state, "incident_rows": [], "incident_reasoned_rows": [], "incident_summary": "", "debug_logs": debug_logs}

    reasoned_rows = apply_incident_reasoning(rows, state.get("query_meta") or {})
    summary = summarize_incident_rows(reasoned_rows, state.get("query_meta") or {})
    debug_logs.append(f"[INCIDENT 1] rows = {len(rows)}")
    debug_logs.append("[INCIDENT 2] status_reasoning = applied")
    debug_logs.append("[INCIDENT 3] priority_ranking = applied")
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
    summary = str(state.get("incident_summary") or "").strip()
    if reasoned_rows:
        query_meta["rows"] = reasoned_rows
        answer = summary or summarize_incident_rows(reasoned_rows, query_meta)
    else:
        answer = str(query_meta.get("title") or "장애현황") + "을 표 형태로 조회하고, 조회 결과에 상태판단/영향도/우선순위를 후처리하도록 준비했습니다."

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
