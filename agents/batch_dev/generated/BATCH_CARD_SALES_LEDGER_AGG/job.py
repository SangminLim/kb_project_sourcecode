from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BATCH_ID = "BATCH_CARD_SALES_LEDGER_AGG"
BATCH_NAME = "매출원장테이블 월별 집계"
OUTPUT_FILE_PREFIX = "batch_output"
OUTPUT_FORMAT = "csv"
OUTPUT_ENCODING = "utf-8-sig"

# 생성된 job.py 위치 기준:
# batch_dev/generated/{BATCH_ID}/job.py
# parents[2] = batch_dev
# 기본 출력 위치 = batch_dev/output
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "output"
).resolve()

def build_database_url(database_url: Optional[str] = None) -> str:
    """
    MariaDB 접속 URL 생성

    우선순위:
    1. --database-url 실행 파라미터
    2. .env 환경변수 조합
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
        f"mysql+pymysql://{quote_plus(db_user)}:{quote_plus(db_password)}"
        f"@{db_host}:{db_port}/{db_service}"
    )

def read_sql() -> str:
    return (
        Path(__file__).resolve().parent / "query.sql"
    ).read_text(encoding="utf-8").strip()

def split_sql_statements(sql: str) -> List[str]:
    """세미콜론 기준으로 SQL 문장을 분리한다.

    실무 확장 포인트:
    - 현재는 템플릿 생성 SQL 기준의 단순 분리
    - 복잡한 프로시저/함수 DDL은 별도 SQL parser 적용 가능
    """
    statements = []
    for statement in re.split(r";\s*(?:\r?\n|$)", sql or ""):
        cleaned = statement.strip()
        if cleaned:
            statements.append(cleaned)
    return statements

def detect_sql_statement_type(sql: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", " ", sql or "", flags=re.DOTALL)
    cleaned = re.sub(r"--.*?$", " ", cleaned, flags=re.MULTILINE).strip()
    match = re.search(
        r"\b(WITH|SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else "UNKNOWN"

def resolve_execution_mode(sql: str) -> str:
    """SQL 패턴에 따라 실행 전략을 결정한다.

    - 마지막/단일 SQL이 SELECT/WITH이면 조회형: read_sql + 파일 생성
    - INSERT/UPDATE/DELETE/MERGE/DDL이 포함되면 실행형: execute + commit
    """
    statements = split_sql_statements(sql)
    if not statements:
        raise ValueError("query.sql이 비어 있습니다.")

    statement_types = [detect_sql_statement_type(stmt) for stmt in statements]

    write_types = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "CREATE",
        "ALTER",
        "DROP",
        "TRUNCATE",
    }

    if any(stmt_type in write_types for stmt_type in statement_types):
        return "execute"

    if statement_types[-1] in {"SELECT", "WITH"}:
        return "query"

    return "execute"

def resolve_runtime_value(
    base_date: Optional[str] = None,
    base_ym: Optional[str] = None,
) -> str:
    """배치 실행 기준값을 결정한다."""
    value = (base_ym or base_date or "").strip()
    if not value:
        raise ValueError("base_date 또는 base_ym 중 하나는 반드시 입력해야 합니다.")
    return value

def build_sql_params(
    base_date: Optional[str] = None,
    base_ym: Optional[str] = None,
) -> Dict[str, str]:
    """SQL 바인드 파라미터 생성.

    query.sql이 :base_date 또는 :base_ym 중 어느 쪽을 쓰더라도 실행 가능하도록
    동일 실행 기준값을 두 키에 모두 넣는다.
    """
    runtime_value = resolve_runtime_value(base_date=base_date, base_ym=base_ym)

    return {
        "base_date": (base_date or runtime_value).strip(),
        "base_ym": (base_ym or runtime_value).strip(),
    }

def build_output_path(output_dir: str, runtime_value: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    return path / f"{OUTPUT_FILE_PREFIX}_{runtime_value}.{OUTPUT_FORMAT}"

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

def execute_statements(engine: Any, sql: str, sql_params: Dict[str, str]) -> int:
    """INSERT/UPDATE/DELETE 등 실행형 SQL을 transaction으로 실행한다."""
    affected_rows = 0

    with engine.begin() as conn:
        for statement in split_sql_statements(sql):
            result = conn.execute(text(statement), sql_params)
            rowcount = getattr(result, "rowcount", None)
            if isinstance(rowcount, int) and rowcount > 0:
                affected_rows += rowcount

    return affected_rows

def run_query_to_file(
    engine: Any,
    sql: str,
    sql_params: Dict[str, str],
    output_dir: str,
    runtime_value: str,
) -> Dict[str, Any]:
    df = pd.read_sql(
        text(sql),
        engine,
        params=sql_params,
    )

    validate_dataframe(df)

    output_path = build_output_path(output_dir, runtime_value)

    write_output(df, output_path)

    return {
        "row_count": int(len(df)),
        "output_file": str(output_path),
    }

def run_execute_sql(
    engine: Any,
    sql: str,
    sql_params: Dict[str, str],
) -> Dict[str, Any]:
    affected_rows = execute_statements(engine, sql, sql_params)

    return {
        "row_count": int(affected_rows),
        "output_file": None,
    }

def run(
    database_url: Optional[str],
    base_date: Optional[str] = None,
    base_ym: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Dict[str, Any]:

    database_url = build_database_url(database_url)
    runtime_value = resolve_runtime_value(base_date=base_date, base_ym=base_ym)
    sql_params = build_sql_params(base_date=base_date, base_ym=base_ym)

    print(
        f"[START] {BATCH_ID} {BATCH_NAME} "
        f"runtime_value={runtime_value}"
    )

    engine = create_engine(database_url)

    sql = read_sql()
    execution_mode = resolve_execution_mode(sql)

    if execution_mode == "query":
        execution_result = run_query_to_file(
            engine=engine,
            sql=sql,
            sql_params=sql_params,
            output_dir=output_dir,
            runtime_value=runtime_value,
        )
    else:
        execution_result = run_execute_sql(
            engine=engine,
            sql=sql,
            sql_params=sql_params,
        )

    result = {
        "batch_id": BATCH_ID,
        "batch_name": BATCH_NAME,
        "base_date": sql_params.get("base_date"),
        "base_ym": sql_params.get("base_ym"),
        "runtime_value": runtime_value,
        "execution_mode": execution_mode,
        "row_count": execution_result["row_count"],
        "output_file": execution_result["output_file"],
    }

    print(
        f"[END] {BATCH_ID} "
        f"mode={result['execution_mode']} "
        f"rows={result['row_count']} "
        f"file={result['output_file'] or '-'}"
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

    parser.add_argument(
        "--base-date",
        required=False,
        default=None,
        help="일배치 기준일자. 예: 20260430",
    )

    parser.add_argument(
        "--base-ym",
        required=False,
        default=None,
        help="월배치 기준년월. 예: 202604",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    run(
        database_url=args.database_url,
        base_date=args.base_date,
        base_ym=args.base_ym,
        output_dir=args.output_dir,
    )
