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
