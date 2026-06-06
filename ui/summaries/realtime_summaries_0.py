from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd


def _get_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_cell(row: pd.Series, column: Optional[str], default: str = "") -> str:
    if not column:
        return default
    value = row.get(column)
    if pd.isna(value) or value is None:
        return default
    text = str(value).strip()
    return text if text else default


def generate_incident_summary(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    batch_col = _get_first_existing_column(df, ["batch_name", "배치명"])
    error_code_col = _get_first_existing_column(df, ["error_code", "오류코드"])
    error_msg_col = _get_first_existing_column(df, ["error_message", "오류메시지", "오류내용"])
    action_detail_col = _get_first_existing_column(df, ["action_detail", "조치방법", "조치내용"])
    action_owner_col = _get_first_existing_column(df, ["action_owner", "담당자", "조치담당자"])

    action_lines = []
    batch_names = []
    error_messages = []
    owners = []

    for _, row in df.iterrows():
        batch_name = _safe_cell(row, batch_col, "배치명 없음")
        error_code = _safe_cell(row, error_code_col, "오류코드 없음")
        error_message = _safe_cell(row, error_msg_col, "오류 메시지 없음")
        action_detail = _safe_cell(row, action_detail_col, "등록된 조치 방법 없음")
        action_owner = _safe_cell(row, action_owner_col, "담당자 미지정")

        batch_names.append(batch_name)
        error_messages.append(error_message)
        action_lines.append(f"- {batch_name} ({error_code}): {action_detail}")
        owners.append(action_owner)

    unique_error_messages = list(dict.fromkeys(error_messages))
    unique_owners = [owner for owner in dict.fromkeys(owners) if owner != "담당자 미지정"]

    lines = [
        "장애 현황 조회 결과",
        "",
        f"전체 장애 건수: {len(df)}건",
        "",
        "장애 배치명 목록:",
        *[f"- {name}" for name in batch_names],
        "",
        "주요 오류 원인 요약:",
        f"- {', '.join(unique_error_messages)}",
        "",
        "조치 방법 요약:",
        *action_lines,
        "",
        f"담당자: {', '.join(unique_owners) if unique_owners else '담당자 미지정'}",
        "",
        "확인 필요사항: 조치 후 배치 재실행 여부와 후속 배치 영향도를 확인하세요.",
    ]
    return "\n".join(lines)


def summarize_billing_dataframe(df: pd.DataFrame, x_field: str, y_field: str) -> Dict[str, Any]:
    if df.empty:
        return {"row_count": 0}
    work_df = df.copy()
    work_df[y_field] = pd.to_numeric(work_df[y_field], errors="coerce").fillna(0)
    work_df = work_df.sort_values(by=x_field).reset_index(drop=True)

    max_row = work_df.loc[work_df[y_field].idxmax()]
    min_row = work_df.loc[work_df[y_field].idxmin()]
    latest_row = work_df.iloc[-1]
    prev_row = work_df.iloc[-2] if len(work_df) >= 2 else None

    change_rate_pct = None
    if prev_row is not None and float(prev_row[y_field]) != 0:
        change_rate_pct = round(
            ((float(latest_row[y_field]) - float(prev_row[y_field])) / float(prev_row[y_field])) * 100,
            2,
        )

    return {
        "row_count": int(len(work_df)),
        "total_amount": float(work_df[y_field].sum()),
        "max_period": str(max_row[x_field]),
        "max_amount": float(max_row[y_field]),
        "min_period": str(min_row[x_field]),
        "min_amount": float(min_row[y_field]),
        "latest_period": str(latest_row[x_field]),
        "latest_amount": float(latest_row[y_field]),
        "previous_period": str(prev_row[x_field]) if prev_row is not None else None,
        "previous_amount": float(prev_row[y_field]) if prev_row is not None else None,
        "change_rate_pct": change_rate_pct,
    }


def format_krw(amount: Any) -> str:
    amount_int = int(round(float(amount or 0)))
    if amount_int != 0 and amount_int % 10000 == 0:
        return f"{amount_int // 10000:,}만 원"
    return f"{amount_int:,}원"


def format_billing_month(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{6}", text):
        return f"{text[:4]}년 {int(text[4:6])}월"
    return text


def _billing_pattern_text(summary: Dict[str, Any]) -> str:
    change_rate_pct = summary.get("change_rate_pct")
    if change_rate_pct is None:
        return "데이터 패턴은 단일 구간이므로 증감 판단은 생략합니다."
    if change_rate_pct > 0:
        return "데이터 패턴은 최근 구간에서 증가하는 흐름입니다."
    if change_rate_pct < 0:
        return "데이터 패턴은 최근 구간에서 감소하는 흐름입니다."
    return "데이터 패턴은 최근 구간에서 전월과 동일한 흐름입니다."


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get_nested_value(source: Dict[str, Any], key_candidates: List[str]) -> Any:
    """query_meta/policy 안의 설정값을 여러 명칭으로 안전하게 찾는다.

    운영 설정 JSON의 필드명이 조금 달라도 코드 수정 없이 threshold 계열 값을
    찾기 위한 보조 함수다.
    """
    if not isinstance(source, dict):
        return None

    for key in key_candidates:
        if key in source:
            return source.get(key)

    for nested_key in ("anomaly_policy", "reasoning_policy", "planner_policy"):
        nested = source.get(nested_key)
        if isinstance(nested, dict):
            found = _get_nested_value(nested, key_candidates)
            if found is not None:
                return found

    return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _billing_anomaly_requested(query_meta: Dict[str, Any], policy: Optional[Dict[str, Any]]) -> bool:
    """상단 요약에 이상징후를 붙일지 판단한다.

    특정 질문 문구를 다시 하드코딩하지 않고, Planner가 만든 execution_plan의
    step 목록 또는 정책 설정을 기준으로 판단한다.
    """
    query_meta = query_meta or {}
    policy = policy or {}

    execution_plan = query_meta.get("execution_plan")
    if isinstance(execution_plan, dict):
        steps = [str(step).strip() for step in _as_list(execution_plan.get("steps"))]
        if "detect_billing_anomaly" in steps:
            return True

    policy_steps = []
    for key in ("steps", "default_steps", "mandatory_steps"):
        policy_steps.extend([str(step).strip() for step in _as_list(policy.get(key))])
    if "detect_billing_anomaly" in policy_steps:
        return True

    return bool(policy.get("requires_anomaly_check") or query_meta.get("requires_anomaly_check"))


def _billing_anomaly_threshold_pct(query_meta: Dict[str, Any], policy: Optional[Dict[str, Any]]) -> float:
    """이상징후 기준값을 설정에서 읽는다.

    우선순위:
    1) policy / query_meta / 하위 policy에 명시된 threshold
    2) 없으면 실무 기본값 10.0 사용

    기본값은 마지막 fallback일 뿐이며, 운영에서는 realtime_queries[].summary_policy
    또는 reasoning_policy에 threshold를 넣어 조정할 수 있다.
    """
    key_candidates = [
        "anomaly_threshold_pct",
        "threshold_pct",
        "change_rate_threshold_pct",
        "change_threshold_pct",
    ]

    for source in (policy or {}, query_meta or {}):
        value = _get_nested_value(source, key_candidates)
        parsed = _safe_float(value)
        if parsed is not None:
            return abs(parsed)

    return 10.0


def _extract_existing_billing_anomaly(query_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """이미 계산된 이상징후 결과가 query_meta에 있으면 우선 사용한다.

    realtime_payload 또는 reasoning 단계가 결과를 query_meta에 넣도록 확장된 경우,
    여기서는 재계산하지 않고 해당 결과를 상단 요약에 표시한다.
    """
    if not isinstance(query_meta, dict):
        return None

    direct_candidates = [
        query_meta.get("billing_anomaly"),
        query_meta.get("anomaly_result"),
        query_meta.get("anomaly_summary"),
    ]

    for item in direct_candidates:
        if isinstance(item, dict):
            return item

    for container_key in ("reasoning_results", "analysis_results", "step_results"):
        container = query_meta.get(container_key)
        for item in _as_list(container):
            if not isinstance(item, dict):
                continue
            step = str(item.get("step") or item.get("step_id") or "").strip()
            if step == "detect_billing_anomaly":
                return item

    return None


def _format_existing_anomaly_result(result: Dict[str, Any]) -> Optional[str]:
    status = str(result.get("status") or result.get("result_status") or "").strip()
    anomaly_type = str(
        result.get("anomaly_type")
        or result.get("anomaly")
        or result.get("judgement")
        or result.get("판단")
        or ""
    ).strip()
    reason = str(
        result.get("reason")
        or result.get("reasoning_note")
        or result.get("message")
        or result.get("판단근거")
        or ""
    ).strip()
    required_checks = result.get("required_checks") or result.get("check_items") or result.get("확인필요사항")

    lines = ["⚠ 이상징후 분석"]

    headline_parts = []
    if status:
        headline_parts.append(f"상태: {status}")
    if anomaly_type:
        headline_parts.append(f"판단: {anomaly_type}")
    if headline_parts:
        lines.append(" / ".join(headline_parts))

    if reason:
        lines.append(f"판단 근거: {reason}")

    check_items = [str(item).strip() for item in _as_list(required_checks) if str(item).strip()]
    if check_items:
        lines.append("확인 필요사항:")
        lines.extend([f"- {item}" for item in check_items])

    return "\n".join(lines) if len(lines) > 1 else None


def _build_billing_anomaly_summary(
    query_meta: Dict[str, Any],
    summary: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """청구 이상징후 상단 요약을 만든다.

    - Planner가 detect_billing_anomaly를 선택한 경우에만 표시한다.
    - 이미 계산된 이상징후 결과가 있으면 그 값을 우선 사용한다.
    - 없으면 현재 요약값의 전월 대비 변동률과 설정 기준값으로 표시용 판단을 만든다.
    """
    if not _billing_anomaly_requested(query_meta, policy):
        return None

    existing = _extract_existing_billing_anomaly(query_meta)
    if existing:
        formatted = _format_existing_anomaly_result(existing)
        if formatted:
            return formatted

    change_rate_pct = summary.get("change_rate_pct")
    if change_rate_pct is None:
        return "⚠ 이상징후 분석\n전월 비교 대상이 부족하여 이상징후 판단은 보류합니다."

    threshold = _billing_anomaly_threshold_pct(query_meta, policy)
    change_rate = float(change_rate_pct)
    abs_change_rate = abs(change_rate)

    if abs_change_rate >= threshold:
        anomaly_type = "급증" if change_rate > 0 else "급감"
        status = "주의"
        reason = f"전월 대비 변동률 {change_rate:.2f}%가 이상징후 기준 {threshold:.1f}% 이상입니다."
    else:
        anomaly_type = "특이사항 없음"
        status = "정상"
        reason = f"전월 대비 변동률 {change_rate:.2f}%가 이상징후 기준 {threshold:.1f}% 미만입니다."

    lines = [
        "⚠ 이상징후 분석",
        f"상태: {status}",
        f"판단: {anomaly_type}",
        f"판단 근거: {reason}",
    ]

    if status == "주의":
        lines.extend([
            "확인 필요사항:",
            "- 대형 승인/취소 거래 반영 여부 확인",
            "- 청구 마감 또는 미청구 데이터 누락 여부 확인",
            "- 월별 집계 기준 및 원천 데이터 적재 건수 확인",
        ])

    return "\n".join(lines)


def generate_billing_summary(
    query_meta: Dict[str, Any],
    df: pd.DataFrame,
    policy: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if df.empty:
        return None

    summary = summarize_billing_dataframe(
        df,
        x_field=query_meta.get("x_field", "billing_month"),
        y_field=query_meta.get("y_field", "amount"),
    )

    lines = [
        f"전체 흐름 요약: 총 {summary['row_count']}개월치 조회 결과, 총액은 {format_krw(summary['total_amount'])}입니다.",
        f"최고 금액 구간: {format_billing_month(summary['max_period'])}, {format_krw(summary['max_amount'])}.",
        f"최저 금액 구간: {format_billing_month(summary['min_period'])}, {format_krw(summary['min_amount'])}.",
    ]

    if summary.get("change_rate_pct") is not None:
        lines.append(
            f"최근 구간: {format_billing_month(summary['latest_period'])} "
            f"{format_krw(summary['latest_amount'])}, 전월 대비 {summary['change_rate_pct']}% 변동."
        )

    lines.append(_billing_pattern_text(summary))

    anomaly_summary = _build_billing_anomaly_summary(query_meta, summary, policy)
    if anomaly_summary:
        lines.append(anomaly_summary)

    return "\n\n".join(lines)


SUMMARY_HANDLER_MAP = {
    "incident_summary": generate_incident_summary,
    "timeseries_amount_summary": generate_billing_summary,
}


def generate_realtime_summary(query_meta: Dict[str, Any], df: pd.DataFrame, policy: Dict[str, Any]) -> Optional[str]:
    handler_name = str(policy.get("summary_handler") or policy.get("summary_type") or "").strip()
    if not handler_name:
        return None

    handler = SUMMARY_HANDLER_MAP.get(handler_name)
    if handler is None:
        return f"등록되지 않은 요약 handler입니다: {handler_name}"

    if handler_name == "timeseries_amount_summary":
        return handler(query_meta, df, policy)
    return handler(df)
