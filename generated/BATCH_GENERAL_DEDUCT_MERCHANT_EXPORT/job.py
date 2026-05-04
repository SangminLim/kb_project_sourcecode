from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy import create_engine, text


BATCH_ID = "BATCH_GENERAL_DEDUCT_MERCHANT_EXPORT"
BATCH_NAME = "소득공제가맹점테이블 파일 생성"
OUTPUT_FILE_PREFIX = "general_deduct_merchant"
OUTPUT_FORMAT = "csv"
OUTPUT_ENCODING = "utf-8-sig"


def read_sql() -> str:
    return (Path(__file__).resolve().parent / "query.sql").read_text(encoding="utf-8").strip()


def build_output_path(output_dir: str, base_date: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{OUTPUT_FILE_PREFIX}_{base_date}.{OUTPUT_FORMAT}"


def validate_dataframe(df: pd.DataFrame) -> None:
    # 실무 확장 포인트: 건수, NULL, 중복, 금액 합계 검증을 여기에 추가한다.
    if df is None:
        raise ValueError("조회 결과 DataFrame이 없습니다.")


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    if OUTPUT_FORMAT == "csv":
        df.to_csv(output_path, index=False, encoding=OUTPUT_ENCODING)
        return
    if OUTPUT_FORMAT == "txt":
        df.to_csv(output_path, index=False, sep="|", encoding=OUTPUT_ENCODING)
        return
    if OUTPUT_FORMAT == "xlsx":
        df.to_excel(output_path, index=False)
        return
    raise ValueError(f"지원하지 않는 출력 형식입니다: {OUTPUT_FORMAT}")


def run(database_url: str, base_date: str, output_dir: str = "./output") -> Dict[str, Any]:
    if not database_url:
        raise ValueError("database_url이 비어 있습니다.")
    if not base_date:
        raise ValueError("base_date가 비어 있습니다.")

    print(f"[START] {BATCH_ID} {BATCH_NAME} base_date={base_date}")

    engine = create_engine(database_url)
    sql = read_sql()
    df = pd.read_sql(text(sql), engine, params={"base_date": base_date})

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
    print(f"[END] {BATCH_ID} rows={result['row_count']} file={result['output_file']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=BATCH_NAME)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--base-date", required=True)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()
    run(args.database_url, args.base_date, args.output_dir)
