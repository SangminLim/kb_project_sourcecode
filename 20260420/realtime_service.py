"""
realtime_service.py

설명
- DB 연결과 실시간 조회형 SQL을 담당하는 서비스 레이어
- query_id 기준으로 SQL 실행 책임을 분리
- streamlit.py는 UI 렌더링에 집중하고, llm.py는 질문 해석에 집중하도록 역할 분리
- SQL 하드코딩 분산을 피하기 위해 QUERY_REGISTRY에서 조회 정의를 관리
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


QUERY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "today_incidents": {
        "sql": """
            SELECT
                batch_name,
                status,
                error_code,
                error_message,
                start_time,
                end_time
            FROM TB_BATCH_INCIDENT
            WHERE DATE(start_time) = CURRENT_DATE
              AND status = :status
            ORDER BY start_time DESC
            LIMIT :limit_count
        """,
        "params": {
            "status": "FAIL",
            "limit_count": 100,
        },
    },
    "billing_monthly_amount": {
        "sql": """
            SELECT
                billing_month,
                amount
            FROM TB_BILLING_MONTHLY_AMOUNT
            ORDER BY billing_month
        """,
        "params": {},
    },
}


class RealtimeQueryService:
    def __init__(self, database_url: str) -> None:
        if not database_url or not database_url.strip():
            raise ValueError("database_url이 비어 있습니다.")
        self.engine: Engine = create_engine(database_url)

    def fetch_dataframe(self, query_meta: Dict[str, Any]) -> pd.DataFrame:
        query_id = (query_meta or {}).get("query_id", "").strip()
        if not query_id:
            raise ValueError("query_meta에 query_id가 없습니다.")

        query_config = QUERY_REGISTRY.get(query_id)
        if not query_config:
            raise ValueError(f"지원하지 않는 query_id 입니다: {query_id}")

        sql = query_config.get("sql", "").strip()
        if not sql:
            raise ValueError(f"{query_id}에 대한 SQL이 등록되어 있지 않습니다.")

        params = query_config.get("params", {}) or {}

        return pd.read_sql(
            text(sql),
            self.engine,
            params=params,
        )