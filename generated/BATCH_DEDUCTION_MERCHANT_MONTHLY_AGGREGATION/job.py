from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


BATCH_ID = "BATCH_DEDUCTION_MERCHANT_MONTHLY_AGGREGATION"


def load_spec() -> dict:
    spec_path = Path(__file__).resolve().parent / "batch_spec.json"
    with spec_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(database_url: str, base_ym: str | None = None, base_date: str | None = None) -> None:
    spec = load_spec()
    params = {"base_ym": base_ym, "base_date": base_date}
    params = {k: v for k, v in params.items() if v is not None}

    engine = create_engine(database_url)
    print(f"[START] {BATCH_ID} batch_type={spec.get('batch_type')} params={params}")

    with engine.begin() as conn:
        target = spec.get("target", {})
        if spec.get("batch_type") == "aggregation_to_table" and target.get("load_strategy") == "delete_insert":
            delete_sql = target.get("delete_sql")
            if delete_sql:
                result = conn.execute(text(delete_sql), params)
                print(f"[DELETE] target={target.get('table')} rows={result.rowcount}")

            result = conn.execute(text(spec["sql"]), params)
            print(f"[INSERT] target={target.get('table')} rows={result.rowcount}")

        elif spec.get("batch_type") == "db_to_file":
            df = pd.read_sql(text(spec["sql"]), conn, params=params)
            output = spec.get("target", {})
            output_dir = Path(output.get("output_dir", "./output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            file_pattern = output.get("output_file_pattern", f"{BATCH_ID}_{base_ym or base_date or 'result'}.csv")
            file_name = file_pattern.format(base_ym=base_ym, base_date=base_date)
            file_path = output_dir / file_name
            df.to_csv(file_path, index=False, encoding=output.get("encoding", "utf-8-sig"))
            print(f"[FILE] rows={len(df)} file={file_path}")
        else:
            raise ValueError(f"지원하지 않는 batch_type입니다: {spec.get('batch_type')}")

    print(f"[END] {BATCH_ID} success")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--base-ym", required=False)
    parser.add_argument("--base-date", required=False)
    args = parser.parse_args()
    run(args.database_url, args.base_ym, args.base_date)


if __name__ == "__main__":
    main()
