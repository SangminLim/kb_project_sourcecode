from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BATCH_ID = "BATCH_TRAD_MARKET_MERCHANT_EXPORT"
BATCH_NAME = "전통시장 가맹점 파일 생성"
OUTPUT_FILE_PREFIX = "traditional_market_merchant"
OUTPUT_FORMAT = "csv"
OUTPUT_ENCODING = "utf-8-sig"


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


def build_output_path(output_dir: str, base_date: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    return path / f"{OUTPUT_FILE_PREFIX}_{base_date}.{OUTPUT_FORMAT}"


def validate_dataframe(df: pd.DataFrame) -> None:
    # 실무 확장 포인트:
    # 건수 검증 / NULL 검증 / 중복 검증 / 금액 합계 검증 등을 추가 가능
    if df is None:
        raise ValueError("조회 결과 DataFrame이 없습니다.")


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    if OUTPUT_FORMAT == "csv":
        df.to_csv(output_path, index=False, encoding=OUTPUT_ENCODING)
        return

    if OUTPUT_FORMAT == "txt":
        df.to_csv(
            output_path,
            index=False,
            sep="|",
            encoding=OUTPUT_ENCODING,
        )
        return

    if OUTPUT_FORMAT == "xlsx":
        df.to_excel(output_path, index=False)
        return

    raise ValueError(
        f"지원하지 않는 출력 형식입니다: {OUTPUT_FORMAT}"
    )


def run(
    database_url: Optional[str],
    base_date: str,
    output_dir: str = "./output"
) -> Dict[str, Any]:

    database_url = build_database_url(database_url)

    if not base_date:
        raise ValueError("base_date가 비어 있습니다.")

    print(f"[START] {BATCH_ID} {BATCH_NAME} base_date={base_date}")

    engine = create_engine(database_url)

    sql = read_sql()

    df = pd.read_sql(
        text(sql),
        engine,
        params={"base_date": base_date}
    )

    validate_dataframe(df)

    output_path = build_output_path(output_dir, base_date)

    write_output(df, output_path)

    result = {
        "batch_id": BATCH_ID,
        "batch_name": BATCH_NAME,
        "base_date": base_date,
        "row_count": int(len(df)),
        "output_file": str(output_path),
    }

    print(
        f"[END] {BATCH_ID} "
        f"rows={result['row_count']} "
        f"file={result['output_file']}"
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=BATCH_NAME)

    parser.add_argument(
        "--database-url",
        required=False,
        default=None,
        help="직접 DB URL 입력 시 사용",
    )

    parser.add_argument("--base-date", required=True)

    parser.add_argument(
        "--output-dir",
        default="./output"
    )

    args = parser.parse_args()

    run(
        args.database_url,
        args.base_date,
        args.output_dir,
    )
