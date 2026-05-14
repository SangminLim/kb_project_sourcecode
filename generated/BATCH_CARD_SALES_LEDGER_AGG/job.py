from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BATCH_ID = "BATCH_CARD_SALES_LEDGER_AGG"
BATCH_NAME = "소득공제 월별 통합 집계 배치"


def build_database_url(database_url: Optional[str] = None) -> str:
    """
    MariaDB 접속 URL 생성

    우선순위:
    1. --database-url 실행 파라미터
    2. .env 환경변수 조합

    .env 예시:
        DB_USER=smlim
        DB_PASSWORD=1111
        DB_HOST=localhost
        DB_PORT=3306
        DB_SERVICE=testDB
    """

    if database_url:
        return database_url

    load_dotenv()

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_service = os.getenv("DB_SERVICE")

    missing = []

    if not db_user:
        missing.append("DB_USER")
    if not db_password:
        missing.append("DB_PASSWORD")
    if not db_host:
        missing.append("DB_HOST")
    if not db_service:
        missing.append("DB_SERVICE")

    if missing:
        raise ValueError(
            f".env 환경변수가 누락되었습니다: {', '.join(missing)}"
        )

    return (
        f"mysql+pymysql://{db_user}:{db_password}"
        f"@{db_host}:{db_port}/{db_service}"
    )


def read_sql() -> str:
    return (
        Path(__file__).resolve().parent / "query.sql"
    ).read_text(encoding="utf-8").strip()


def run(database_url: Optional[str], base_ym: str) -> Dict[str, Any]:
    database_url = build_database_url(database_url)

    if not base_ym:
        raise ValueError("base_ym이 비어 있습니다.")

    print(f"[START] {BATCH_ID} {BATCH_NAME} base_ym={base_ym}")

    engine = create_engine(database_url)
    sql = read_sql()
    delete_sql = """DELETE FROM TB_DEDUCTION_MONTHLY_SUMMARY WHERE BASE_YM = :base_ym""".strip()

    with engine.begin() as conn:
        deleted_rows = 0
        inserted_rows = 0

        if delete_sql:
            result = conn.execute(text(delete_sql), {"base_ym": base_ym})
            deleted_rows = result.rowcount
            print(f"[DELETE] rows={deleted_rows}")

        result = conn.execute(text(sql), {"base_ym": base_ym})
        inserted_rows = result.rowcount
        print(f"[INSERT] rows={inserted_rows}")

    print(f"[END] {BATCH_ID} deleted={deleted_rows} inserted={inserted_rows}")

    return {
        "batch_id": BATCH_ID,
        "batch_name": BATCH_NAME,
        "base_ym": base_ym,
        "deleted_rows": deleted_rows,
        "inserted_rows": inserted_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=BATCH_NAME)

    parser.add_argument(
        "--database-url",
        required=False,
        default=None,
        help="직접 DB URL 입력 시 사용",
    )

    parser.add_argument("--base-ym", required=True)

    args = parser.parse_args()

    run(args.database_url, args.base_ym)
