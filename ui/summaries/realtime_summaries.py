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


def generate_billing_summary(query_meta: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
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
        return handler(query_meta, df)
    return handler(df)
