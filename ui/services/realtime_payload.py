from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from services.realtime_query_service import RealtimeQueryService

from ..context import (
    DATABASE_URL,
    REALTIME_MAX_ROWS,
    logger,
    get_realtime_policy,
)
from ..summaries.realtime_summaries import generate_realtime_summary


@st.cache_resource
def get_realtime_service() -> RealtimeQueryService | None:
    """RealtimeQueryService를 ui.context에 의존하지 않고 여기서 생성한다.

    리팩토링 후 ui/context.py에 get_realtime_service가 없더라도
    realtime payload 계층이 독립적으로 DB 조회 서비스를 사용할 수 있게 한다.
    """
    if not DATABASE_URL:
        return None
    return RealtimeQueryService(DATABASE_URL)


def dataframe_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    safe_df = df.where(pd.notnull(df), None)
    return {
        "columns": safe_df.columns.tolist(),
        "rows": safe_df.to_dict(orient="records"),
    }


def payload_to_dataframe(payload: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("rows", [])
    columns = payload.get("columns", [])
    return pd.DataFrame(rows, columns=columns)


def fetch_realtime_dataframe(query_meta: Dict[str, Any]) -> pd.DataFrame:
    service = get_realtime_service()
    if service is None:
        raise RuntimeError(
            "DB 접속 정보가 설정되지 않았습니다. "
            ".env에 DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SERVICE를 설정하세요."
        )

    df = service.fetch_dataframe(query_meta)

    if REALTIME_MAX_ROWS > 0 and len(df) > REALTIME_MAX_ROWS:
        logger.warning(
            "Realtime query result truncated: query_id=%s rows=%s limit=%s",
            (query_meta or {}).get("query_id"),
            len(df),
            REALTIME_MAX_ROWS,
        )
        return df.head(REALTIME_MAX_ROWS).copy()

    return df




def _get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """DataFrame에서 후보 컬럼 중 실제 존재하는 첫 컬럼명을 찾는다.

    SQL alias가 한글/영문으로 바뀌어도 reasoning 로직을 유지하기 위한 유틸이다.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _safe_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _policy_list(policy: Dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = policy.get(key)
    if isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
        return values or default
    if isinstance(value, str) and value.strip():
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


def _policy_int(policy: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(policy.get(key, default))
    except Exception:
        return default


def _build_incident_reasoning_results(df: pd.DataFrame, policy: Dict[str, Any]) -> tuple[list[Dict[str, Any]], list[str]]:
    """장애현황 조회 결과에 rule 기반 multi reasoning 결과를 붙인다.

    원칙:
    - SQL 컬럼명은 한글/영문 alias를 모두 허용한다.
    - 운영 기준값은 realtime policy에서 조정 가능하게 한다.
    - LLM이 우선순위/상태를 임의 판단하지 않도록 deterministic rule로 계산한다.
    """
    debug_logs: list[str] = []
    debug_logs.append(f"[INCIDENT 1] source_rows = {len(df)}")

    if df.empty:
        debug_logs.append("[INCIDENT 2] skipped = empty_dataframe")
        return [], debug_logs

    batch_col = _get_first_existing_column(df, ["batch_name", "배치명", "job_name", "JOB명", "작업명"])
    status_col = _get_first_existing_column(df, ["status", "상태", "incident_status"])
    error_code_col = _get_first_existing_column(df, ["error_code", "오류코드", "err_code"])
    error_msg_col = _get_first_existing_column(df, ["error_message", "오류메시지", "오류내용", "err_msg"])
    start_time_col = _get_first_existing_column(df, ["start_time", "오류발생시간", "발생시간", "occurred_at"])
    end_time_col = _get_first_existing_column(df, ["end_time", "종료시간", "완료시간", "resolved_at"])
    elapsed_col = _get_first_existing_column(df, ["elapsed_min", "elapsed_minutes", "경과분", "경과시간분"])
    severity_col = _get_first_existing_column(df, ["severity", "심각도", "등급"])
    action_detail_col = _get_first_existing_column(df, ["action_detail", "조치내용", "조치방법"])
    action_owner_col = _get_first_existing_column(df, ["action_owner", "담당자", "조치담당자"])

    unresolved_statuses = set(_policy_list(
        policy,
        "incident_unresolved_statuses",
        ["FAIL", "ERROR", "FAILED", "처리중", "미처리", "장애", "실패"],
    ))
    closed_statuses = set(_policy_list(
        policy,
        "incident_closed_statuses",
        ["SUCCESS", "DONE", "COMPLETED", "CLOSED", "RESOLVED", "종료", "완료", "정상"],
    ))
    high_severities = set(_policy_list(
        policy,
        "incident_high_severities",
        ["HIGH", "CRITICAL", "P1", "P2", "상", "긴급", "심각"],
    ))
    long_running_minutes = _policy_int(policy, "incident_long_running_minutes", 30)

    debug_logs.append("[INCIDENT 2] column_mapping = " + str({
        "batch": batch_col,
        "status": status_col,
        "error_code": error_code_col,
        "elapsed": elapsed_col,
        "severity": severity_col,
        "action_detail": action_detail_col,
        "action_owner": action_owner_col,
    }))
    debug_logs.append(f"[INCIDENT 3] policy.long_running_minutes = {long_running_minutes}")

    results: list[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        batch_name = _safe_text(row.get(batch_col) if batch_col else None, f"ROW_{idx + 1}")
        status = _safe_text(row.get(status_col) if status_col else None, "상태없음")
        status_upper = status.upper()
        severity = _safe_text(row.get(severity_col) if severity_col else None, "")
        severity_upper = severity.upper()
        error_code = _safe_text(row.get(error_code_col) if error_code_col else None, "")
        error_message = _safe_text(row.get(error_msg_col) if error_msg_col else None, "")
        start_time = _safe_text(row.get(start_time_col) if start_time_col else None, "")
        end_time = _safe_text(row.get(end_time_col) if end_time_col else None, "")
        action_detail = _safe_text(row.get(action_detail_col) if action_detail_col else None, "")
        action_owner = _safe_text(row.get(action_owner_col) if action_owner_col else None, "")
        elapsed_min = _safe_number(row.get(elapsed_col) if elapsed_col else None, 0.0)

        reasons: list[str] = []

        is_closed = status in closed_statuses or status_upper in closed_statuses
        is_unresolved = status in unresolved_statuses or status_upper in unresolved_statuses
        if not is_unresolved and not is_closed:
            # 종료 컬럼이 있는데 비어 있으면 보수적으로 미해결 후보로 본다.
            is_unresolved = bool(end_time_col and not end_time)

        if is_unresolved:
            reasons.append(f"상태가 '{status}'로 미해결/장애 상태에 해당합니다.")
        elif is_closed:
            reasons.append(f"상태가 '{status}'로 종료 상태에 해당합니다.")
        else:
            reasons.append(f"상태 '{status}'는 정책에 명시되지 않아 확인 필요로 분류했습니다.")

        is_long_running = bool(is_unresolved and elapsed_col and elapsed_min >= long_running_minutes)
        if is_long_running:
            reasons.append(f"경과분 {int(elapsed_min)}분이 장기화 기준 {long_running_minutes}분 이상입니다.")
        elif is_unresolved and elapsed_col:
            reasons.append(f"경과분 {int(elapsed_min)}분은 장기화 기준 {long_running_minutes}분 미만입니다.")
        elif is_unresolved:
            reasons.append("경과분 컬럼이 없어 장기화 여부는 보수적으로 확인 필요입니다.")

        has_action = bool(action_detail)
        if has_action:
            reasons.append("등록된 조치내용이 있습니다.")
        else:
            reasons.append("등록된 조치내용이 없어 조치 확인이 필요합니다.")

        is_high_severity = bool(severity and (severity in high_severities or severity_upper in high_severities))
        if is_high_severity:
            reasons.append(f"심각도 '{severity}'가 높은 등급 정책에 해당합니다.")

        if is_unresolved and (is_high_severity or is_long_running):
            priority = "P1"
        elif is_unresolved:
            priority = "P2"
        elif not has_action:
            priority = "P3"
        else:
            priority = "P4"

        status_reasoning = "미해결" if is_unresolved else "종료" if is_closed else "확인필요"
        action_needed = bool(is_unresolved and not has_action)

        results.append({
            "배치명": batch_name,
            "상태": status,
            "오류코드": error_code,
            "오류메시지": error_message,
            "오류발생시간": start_time,
            "종료시간": end_time,
            "경과분": int(elapsed_min) if elapsed_col else None,
            "심각도": severity,
            "담당자": action_owner or "담당자 미지정",
            "상태판단": status_reasoning,
            "장기미처리여부": is_long_running,
            "조치필요여부": action_needed,
            "우선순위": priority,
            "판단근거": reasons,
        })

    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    results.sort(key=lambda item: (priority_order.get(str(item.get("우선순위")), 99), str(item.get("배치명", ""))))

    debug_logs.append(f"[INCIDENT 4] reasoning_results = {len(results)}")
    debug_logs.append("[INCIDENT 5] priority_ranking = completed")
    return results, debug_logs


def _should_build_incident_reasoning(query_meta: Dict[str, Any], realtime_mode: Optional[str], policy: Dict[str, Any]) -> bool:
    query_id = str((query_meta or {}).get("query_id") or "").strip()
    mode = str(realtime_mode or (query_meta or {}).get("realtime_mode") or "").strip()
    summary_type = str(policy.get("summary_type") or policy.get("summary_handler") or "").strip()
    return (
        query_id == "today_incidents"
        or mode == "incident_table_with_summary"
        or summary_type == "incident_summary"
    )


def build_realtime_payload(
    query_meta: Dict[str, Any],
    render_type: str,
    realtime_mode: Optional[str] = None,
) -> Dict[str, Any]:
    policy = get_realtime_policy(query_meta)
    summary_type = policy.get("summary_type")

    payload: Dict[str, Any] = {
        "query_id": query_meta.get("query_id"),
        "render_type": render_type,
        "summary": None,
        "summary_type": summary_type,
        "dataframe": None,
        "empty_message": None,
        "error": None,
        "reasoning_results": None,
        "debug_logs": [],
    }

    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as exc:
        logger.exception(
            "Realtime query failed: query_id=%s",
            (query_meta or {}).get("query_id"),
        )
        payload["error"] = str(exc)
        return payload

    payload["dataframe"] = dataframe_to_payload(df)

    if _should_build_incident_reasoning(query_meta, realtime_mode, policy):
        reasoning_results, incident_debug_logs = _build_incident_reasoning_results(df, policy)
        payload["reasoning_results"] = reasoning_results
        payload["debug_logs"] = incident_debug_logs

    if df.empty:
        payload["empty_message"] = str(policy.get("empty_message") or "조회 결과가 없습니다.")
        return payload

    try:
        payload["summary"] = generate_realtime_summary(query_meta, df, policy)
    except Exception as exc:
        logger.exception(
            "Realtime summary generation failed: query_id=%s",
            (query_meta or {}).get("query_id"),
        )
        payload["summary_error"] = str(exc)

    return payload


def enrich_result_with_realtime_payload(result: Any) -> Any:
    if getattr(result, "render_type", None) not in {"table", "chart"}:
        return result
    if not getattr(result, "query_meta", None):
        return result

    realtime_payload = build_realtime_payload(
        query_meta=result.query_meta,
        render_type=result.render_type,
        realtime_mode=getattr(result, "realtime_mode", None),
    )
    setattr(result, "realtime_payload", realtime_payload)
    return result
