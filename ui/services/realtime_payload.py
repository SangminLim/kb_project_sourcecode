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


def _policy_float(policy: Dict[str, Any], key: str, default: float) -> float:
    """정책값을 float로 안전하게 읽는다.

    운영 기준값은 realtime policy 또는 query_meta.reasoning_policy에서 조정할 수 있게 한다.
    """
    try:
        return float(policy.get(key, default))
    except Exception:
        return default


def _policy_bool(policy: Dict[str, Any], key: str, default: bool = True) -> bool:
    value = policy.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
    impact_col = _get_first_existing_column(df, ["impact_yn", "영향여부", "고객영향", "customer_impact", "CUSTOMER_IMPACT"])
    downstream_col = _get_first_existing_column(df, ["downstream_jobs", "후속배치", "후행배치", "연계배치", "후속업무"])

    # realtime policy와 query_meta.reasoning_policy가 서로 다른 key 체계를 쓸 수 있어 둘 다 흡수한다.
    unresolved_statuses = set(_policy_list(
        policy,
        "incident_unresolved_statuses",
        _policy_list(policy, "open_statuses", ["FAIL", "ERROR", "FAILED", "처리중", "미처리", "장애", "실패"]),
    ))
    closed_statuses = set(_policy_list(
        policy,
        "incident_closed_statuses",
        _policy_list(policy, "closed_statuses", ["SUCCESS", "DONE", "COMPLETED", "CLOSED", "RESOLVED", "종료", "완료", "정상"]),
    ))
    high_severities = set(_policy_list(
        policy,
        "incident_high_severities",
        _policy_list(policy, "critical_severities", ["HIGH", "CRITICAL", "P1", "P2", "상", "긴급", "심각"]),
    ))
    long_running_minutes = _policy_int(policy, "incident_long_running_minutes", _policy_int(policy, "high_elapsed_minutes", 30))

    debug_logs.append("[INCIDENT 2] column_mapping = " + str({
        "batch": batch_col,
        "status": status_col,
        "error_code": error_code_col,
        "elapsed": elapsed_col,
        "severity": severity_col,
        "action_detail": action_detail_col,
        "action_owner": action_owner_col,
        "impact_yn": impact_col,
        "downstream": downstream_col,
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
        impact_raw = _safe_text(row.get(impact_col) if impact_col else None, "")
        downstream = _safe_text(row.get(downstream_col) if downstream_col else None, "")
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

        has_action_detail = bool(action_detail)
        has_action_owner = bool(action_owner)
        has_action_record = has_action_detail or has_action_owner
        action_info_incomplete = not (has_action_detail and has_action_owner)
        needs_additional_action = bool(is_unresolved and action_info_incomplete)
        if not has_action_record:
            reasons.append("등록된 운영 조치정보가 없어 조치 확인이 필요합니다.")
        elif action_info_incomplete:
            reasons.append("운영 조치정보가 일부 등록되어 있으나 조치내용/담당자 중 일부 확인이 필요합니다.")
        else:
            reasons.append("등록된 조치내용과 담당자 정보가 있습니다.")

        has_impact = impact_raw.upper() in {"Y", "YES", "TRUE", "1", "있음", "영향"}
        if has_impact:
            reasons.append("원천 데이터에 고객/업무 영향 표시가 있습니다.")

        is_high_severity = bool(severity and (severity in high_severities or severity_upper in high_severities))
        if is_high_severity:
            reasons.append(f"심각도 '{severity}'가 높은 등급 정책에 해당합니다.")

        if is_unresolved and (is_high_severity or is_long_running or has_impact):
            priority = "P1"
        elif is_unresolved and needs_additional_action:
            priority = "P2"
        elif is_unresolved:
            priority = "P2"
        elif not has_action_record:
            priority = "P3"
        else:
            priority = "P4"

        status_reasoning = "미해결" if is_unresolved else "종료" if is_closed else "확인필요"
        if priority == "P1":
            impact_level = "높음"
        elif priority == "P2":
            impact_level = "중간"
        elif priority == "P3":
            impact_level = "낮음"
        else:
            impact_level = "종료/확인"

        if not is_unresolved:
            recommended_action = "완료 여부와 재발 방지 조치만 확인"
        elif needs_additional_action:
            recommended_action = "조치내용/담당자를 보강하고 담당자 배정 후 재처리 여부 확인"
        elif priority == "P1":
            recommended_action = "담당자 확인 후 우선 조치하고 후속 배치/고객 영향 여부를 공지"
        else:
            recommended_action = "경과시간과 재시도 결과를 확인하고 상태 변화를 모니터링"

        downstream_check = downstream or ("후속 배치 영향 확인 필요" if is_unresolved and (priority == "P1" or is_long_running) else "후속 영향 낮음")

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
            "조치내용": action_detail or "조치내용 미등록",
            "영향여부": impact_raw or "영향 정보 없음",
            "후속영향확인": downstream_check,
            "상태판단": status_reasoning,
            "영향도판단": impact_level,
            "장기미처리여부": is_long_running,
            "운영조치존재여부": has_action_record,
            "추가조치필요여부": needs_additional_action,
            # 구버전 화면 호환용 key도 유지한다.
            "조치필요여부": needs_additional_action,
            "has_action_record": has_action_record,
            "needs_additional_action": needs_additional_action,
            "is_long_running": is_long_running,
            "우선순위": priority,
            "권장조치": recommended_action,
            "판단근거": reasons,
        })

    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    results.sort(key=lambda item: (priority_order.get(str(item.get("우선순위")), 99), str(item.get("배치명", ""))))

    debug_logs.append(f"[INCIDENT 4] reasoning_results = {len(results)}")
    debug_logs.append("[INCIDENT 5] priority_ranking = completed")
    return results, debug_logs


def _build_billing_reasoning_results(
    df: pd.DataFrame,
    query_meta: Dict[str, Any],
    policy: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], list[str]]:
    """청구 월별 금액 차트/요약에 대한 rule 기반 multi reasoning 결과를 만든다.

    원칙:
    - query_id를 기준으로 분기하지 않는다.
    - x_field/y_field는 query_meta에서 읽고, 없으면 DataFrame 컬럼을 보수적으로 사용한다.
    - threshold/표시 정책은 realtime policy 또는 query_meta.reasoning_policy로 조정 가능하게 둔다.
    """
    debug_logs: list[str] = []
    debug_logs.append(f"[BILLING 1] source_rows = {len(df)}")

    plan = (query_meta or {}).get("execution_plan") or {}
    planned_steps = plan.get("steps", []) if isinstance(plan, dict) else []
    debug_logs.append(f"[BILLING 2] planned_steps = {planned_steps}")

    if df.empty:
        debug_logs.append("[BILLING 3] skipped = empty_dataframe")
        return [], debug_logs

    reasoning_policy = {}
    if isinstance((query_meta or {}).get("reasoning_policy"), dict):
        reasoning_policy.update((query_meta or {}).get("reasoning_policy") or {})
    if isinstance(policy, dict):
        reasoning_policy = {**policy, **reasoning_policy}

    configured_x = str((query_meta or {}).get("x_field") or "").strip()
    configured_y = str((query_meta or {}).get("y_field") or "").strip()

    x_field = configured_x if configured_x in df.columns else None
    y_field = configured_y if configured_y in df.columns else None

    if x_field is None and len(df.columns) >= 1:
        x_field = str(df.columns[0])
    if y_field is None and len(df.columns) >= 2:
        y_field = str(df.columns[1])

    debug_logs.append("[BILLING 3] column_mapping = " + str({
        "x_field": x_field,
        "y_field": y_field,
        "configured_x_field": configured_x,
        "configured_y_field": configured_y,
    }))

    if not x_field or not y_field or x_field not in df.columns or y_field not in df.columns:
        return [
            {
                "step": "validate_chart_fields",
                "title": "차트 컬럼 검증",
                "status": "확인 필요",
                "details": [
                    f"x_field={configured_x or '(미지정)'}",
                    f"y_field={configured_y or '(미지정)'}",
                    "DataFrame에서 차트 생성에 필요한 컬럼을 찾지 못했습니다.",
                ],
            }
        ], debug_logs + ["[BILLING 4] validation = failed"]

    work_df = df[[x_field, y_field]].copy()
    work_df[y_field] = pd.to_numeric(work_df[y_field], errors="coerce").fillna(0)
    work_df = work_df.sort_values(by=x_field).reset_index(drop=True)

    row_count = int(len(work_df))
    total_amount = float(work_df[y_field].sum())
    max_row = work_df.loc[work_df[y_field].idxmax()]
    min_row = work_df.loc[work_df[y_field].idxmin()]
    latest_row = work_df.iloc[-1]
    prev_row = work_df.iloc[-2] if row_count >= 2 else None

    change_rate_pct = None
    change_direction = "단일 구간"
    if prev_row is not None and float(prev_row[y_field]) != 0:
        change_rate_pct = round(
            ((float(latest_row[y_field]) - float(prev_row[y_field])) / float(prev_row[y_field])) * 100,
            2,
        )
        if change_rate_pct > 0:
            change_direction = "증가"
        elif change_rate_pct < 0:
            change_direction = "감소"
        else:
            change_direction = "동일"

    threshold = _policy_float(reasoning_policy, "billing_change_threshold_pct", 10.0)
    anomaly_threshold = _policy_float(reasoning_policy, "billing_anomaly_threshold_pct", threshold)
    anomaly_check_enabled = _policy_bool(reasoning_policy, "billing_anomaly_check_enabled", True)

    volatility_note = "증감률 판단 생략"
    if change_rate_pct is not None:
        if abs(float(change_rate_pct)) >= threshold:
            volatility_note = f"전월 대비 변동률 {change_rate_pct}%가 기준 {threshold}% 이상입니다."
        else:
            volatility_note = f"전월 대비 변동률 {change_rate_pct}%가 기준 {threshold}% 미만입니다."

    anomaly_status = "판단 불가"
    anomaly_level = "확인 필요"
    anomaly_reason = "비교 가능한 이전 구간이 없어 이상징후 판단을 생략했습니다."
    anomaly_check_items = _policy_list(
        reasoning_policy,
        "billing_anomaly_check_items",
        [
            "대형 승인/취소 거래 반영 여부 확인",
            "청구 마감 또는 미청구 데이터 누락 여부 확인",
            "월별 집계 기준 및 원천 데이터 적재 건수 확인",
        ],
    )

    if change_rate_pct is not None:
        abs_change_rate = abs(float(change_rate_pct))
        if abs_change_rate >= anomaly_threshold:
            anomaly_status = "급증" if change_rate_pct > 0 else "급감" if change_rate_pct < 0 else "변동 없음"
            anomaly_level = "주의"
            anomaly_reason = f"전월 대비 변동률 {change_rate_pct}%가 이상징후 기준 {anomaly_threshold}% 이상입니다."
        else:
            anomaly_status = "정상 범위"
            anomaly_level = "정상"
            anomaly_reason = f"전월 대비 변동률 {change_rate_pct}%가 이상징후 기준 {anomaly_threshold}% 미만입니다."

    planned_step_set = {str(step) for step in planned_steps}

    def step_selected(step: str) -> bool:
        # execution_plan이 없는 과거 이력/직접 호출은 기존처럼 전체 reasoning을 표시한다.
        # execution_plan이 있으면 planner가 선택한 step만 payload로 만든다.
        return not planned_step_set or step in planned_step_set

    results: list[Dict[str, Any]] = []

    if step_selected("prepare_realtime_query"):
        results.append({
            "step": "prepare_realtime_query",
            "title": "청구 데이터 조회 준비",
            "status": "완료",
            "details": [
                f"조회 대상: {(query_meta or {}).get('title') or (query_meta or {}).get('query_id') or '청구 데이터'}",
                f"렌더링 유형: {(query_meta or {}).get('render_type') or 'chart'}",
            ],
        })

    if step_selected("validate_chart_fields"):
        results.append({
            "step": "validate_chart_fields",
            "title": "차트 컬럼 검증",
            "status": "완료",
            "details": [
                f"x_field={x_field}",
                f"y_field={y_field}",
                f"조회 행 수={row_count}건",
            ],
        })

    if step_selected("fetch_monthly_billing"):
        results.append({
            "step": "fetch_monthly_billing",
            "title": "월별 청구 금액 조회",
            "status": "완료",
            "details": [
                f"총 {row_count}개 구간",
                f"총액={total_amount:,.0f}",
                f"최고 구간={max_row[x_field]} / {float(max_row[y_field]):,.0f}",
                f"최저 구간={min_row[x_field]} / {float(min_row[y_field]):,.0f}",
            ],
        })

    if step_selected("build_chart"):
        results.append({
            "step": "build_chart",
            "title": "월별 금액 그래프 생성",
            "status": "완료",
            "details": [
                f"차트 유형={(query_meta or {}).get('chart_type') or 'bar'}",
                f"X축={x_field}",
                f"Y축={y_field}",
            ],
        })

    if step_selected("summarize_trend"):
        results.append({
            "step": "summarize_trend",
            "title": "증감 흐름 요약",
            "status": "완료",
            "details": [
                f"최근 구간={latest_row[x_field]} / {float(latest_row[y_field]):,.0f}",
                f"전월 대비 증감률={change_rate_pct if change_rate_pct is not None else '계산 불가'}",
                f"흐름 판단={change_direction}",
                volatility_note,
            ],
        })

    if anomaly_check_enabled and step_selected("detect_billing_anomaly"):
        results.append({
            "step": "detect_billing_anomaly",
            "title": "청구 이상징후 판단",
            "status": anomaly_level,
            "details": [
                f"이상징후판단={anomaly_status}",
                anomaly_reason,
                f"판단기준={anomaly_threshold}%",
                "확인필요사항=" + " / ".join(anomaly_check_items),
            ],
        })

    debug_logs.append(f"[BILLING 4] reasoning_results = {len(results)}")
    if step_selected("summarize_trend"):
        debug_logs.append(f"[BILLING 5] trend_direction = {change_direction}")
    if step_selected("detect_billing_anomaly"):
        debug_logs.append(f"[BILLING 5-1] anomaly_status = {anomaly_status}")
    debug_logs.append("[BILLING 6] chart_reasoning = completed")
    return results, debug_logs


def _should_build_incident_reasoning(query_meta: Dict[str, Any], realtime_mode: Optional[str], policy: Dict[str, Any]) -> bool:
    post_process = str((query_meta or {}).get("post_process") or "").strip()
    mode = str(realtime_mode or (query_meta or {}).get("realtime_mode") or "").strip().lower()
    summary_type = str(policy.get("summary_type") or policy.get("summary_handler") or "").strip()
    return (
        post_process == "incident_reasoning"
        or "incident" in mode
        or summary_type == "incident_summary"
    )


def _should_build_billing_plan_logs(query_meta: Dict[str, Any], realtime_mode: Optional[str], policy: Dict[str, Any]) -> bool:
    post_process = str((query_meta or {}).get("post_process") or "").strip()
    mode = str(realtime_mode or (query_meta or {}).get("realtime_mode") or "").strip().lower()
    summary_type = str(policy.get("summary_type") or policy.get("summary_handler") or "").strip()
    return (
        post_process == "billing_graph_reasoning"
        or "billing" in mode
        or summary_type == "timeseries_amount_summary"
    )


def _build_plan_debug_logs(query_meta: Dict[str, Any], target_domain: str) -> list[str]:
    plan = (query_meta or {}).get("execution_plan") or {}
    post_process = str((query_meta or {}).get("post_process") or "")
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    reasons = plan.get("reasons", []) if isinstance(plan, dict) else []
    return [
        f"[PLAN 1] execution_plan_loaded = {bool(plan)}",
        f"[PLAN 2] selected_steps = {steps}",
        f"[PLAN 3] post_process = {post_process}",
        f"[PLAN 4] target_domain = {target_domain}",
        f"[PLAN 5] planner_reasons = {reasons}",
    ]


def _should_generate_summary(query_meta: Dict[str, Any], realtime_mode: Optional[str], policy: Dict[str, Any]) -> bool:
    """요약 생성 여부를 planner step과 정책으로 결정한다.

    billing 계열은 summarize_trend step이 선택됐을 때만 상단 데이터 요약을 생성한다.
    단, 과거 payload/직접 호출처럼 execution_plan이 없거나 policy에서 강제하면 기존 동작을 유지한다.
    """
    if not _should_build_billing_plan_logs(query_meta, realtime_mode, policy):
        return True

    if bool(policy.get("billing_summary_always_enabled", False)):
        return True

    plan = (query_meta or {}).get("execution_plan") or {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    planned_step_set = {str(step) for step in steps}
    if not planned_step_set:
        return True
    return "summarize_trend" in planned_step_set


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
        "billing_reasoning_results": None,
        "debug_logs": [],
        "execution_plan": (query_meta or {}).get("execution_plan"),
        "reasoning_policy": (query_meta or {}).get("reasoning_policy"),
        "query_meta": query_meta,
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
        incident_policy = dict(policy or {})
        if isinstance((query_meta or {}).get("reasoning_policy"), dict):
            incident_policy.update((query_meta or {}).get("reasoning_policy") or {})
        reasoning_results, incident_debug_logs = _build_incident_reasoning_results(df, incident_policy)
        payload["reasoning_results"] = reasoning_results
        payload["debug_logs"] = _build_plan_debug_logs(query_meta, "incident") + incident_debug_logs

    if _should_build_billing_plan_logs(query_meta, realtime_mode, policy):
        billing_reasoning_results, billing_debug_logs = _build_billing_reasoning_results(df, query_meta, policy)
        payload["billing_reasoning_results"] = billing_reasoning_results
        payload["debug_logs"] = (
            _build_plan_debug_logs(query_meta, "billing")
            + billing_debug_logs
            + list(payload.get("debug_logs") or [])
        )

    if df.empty:
        payload["empty_message"] = str(policy.get("empty_message") or "조회 결과가 없습니다.")
        return payload

    if _should_generate_summary(query_meta, realtime_mode, policy):
        try:
            payload["summary"] = generate_realtime_summary(query_meta, df, policy)
            if _should_build_billing_plan_logs(query_meta, realtime_mode, policy):
                payload["debug_logs"] = list(payload.get("debug_logs") or []) + [
                    "[BILLING 7] trend_summary = generated" if payload.get("summary") else "[BILLING 7] trend_summary = skipped"
                ]
        except Exception as exc:
            logger.exception(
                "Realtime summary generation failed: query_id=%s",
                (query_meta or {}).get("query_id"),
            )
            payload["summary_error"] = str(exc)
    elif _should_build_billing_plan_logs(query_meta, realtime_mode, policy):
        payload["debug_logs"] = list(payload.get("debug_logs") or []) + [
            "[BILLING 7] trend_summary = skipped_by_plan"
        ]

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
