from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_table(meta: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    for table in meta.get("tables", []):
        if table.get("table_name") == table_name:
            return table
    raise KeyError(f"metadata table not found: {table_name}")


def find_batch(meta: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    for batch in meta.get("batch_definitions", []):
        if batch.get("batch_id") == batch_id:
            return batch
    raise KeyError(f"batch definition not found: {batch_id}")


def find_rule_set(meta: Dict[str, Any], rule_set_id: str) -> Dict[str, Any]:
    for rule_set in meta.get("rule_sets", []):
        if rule_set.get("rule_set_id") == rule_set_id:
            return rule_set
    raise KeyError(f"rule set not found: {rule_set_id}")


def meta_columns(meta: Dict[str, Any], table_name: str) -> List[str]:
    table = find_table(meta, table_name)
    return [col["column_name"] for col in table.get("columns", [])]


def db_columns(engine: Any, table_name: str) -> List[str]:
    inspector = inspect(engine)
    return [col["name"] for col in inspector.get_columns(table_name)]


def validate_columns(engine: Any, meta: Dict[str, Any], table_names: List[str], fail: bool = True) -> None:
    messages: List[str] = []
    for table_name in table_names:
        expected = set(meta_columns(meta, table_name))
        actual = set(db_columns(engine, table_name))
        missing = sorted(expected - actual)
        if missing:
            messages.append(f"{table_name} missing columns: {', '.join(missing)}")
    if messages:
        message = "\n".join(messages)
        if fail:
            raise RuntimeError(message)
        print(f"[WARN] metadata/db column mismatch\n{message}")


def build_matched_subquery(batch: Dict[str, Any], rule_set: Dict[str, Any]) -> str:
    source_filters = [item["condition"] for item in rule_set.get("source_filters", [])]
    merchant_filters = [item["condition"] for item in rule_set.get("merchant_filters", [])]
    where_clause = "\n      AND ".join(source_filters + merchant_filters)

    parts: List[str] = []
    for category in batch.get("deduction_categories", []):
        parts.append(f"""
        SELECT
            L.SALES_SEQ_NO,
            L.BASE_YM,
            L.CUSTOMER_ID,
            L.MERCHANT_ID,
            L.SALES_AMT,
            '{category['deduct_type_cd']}' AS DEDUCT_TYPE_CD,
            '{category['deduct_type_nm']}' AS DEDUCT_TYPE_NM
        FROM {batch['source_table']} L
        JOIN {category['merchant_table']} M
          ON L.MERCHANT_ID = M.MERCHANT_ID
        WHERE {where_clause}
        """.strip())
    return "\nUNION ALL\n".join(parts)


def execute_scalar(conn: Any, sql: str, params: Dict[str, Any]) -> int:
    value = conn.execute(text(sql), params).scalar()
    return int(value or 0)


def run(database_url: str, base_ym: str, meta_path: str, output_dir: str, dry_run: bool = False) -> None:
    meta = load_json(meta_path)
    config_path = Path(__file__).with_name("config.json")
    config = load_json(config_path)
    batch = find_batch(meta, config["batch_id"])
    rule_set = find_rule_set(meta, batch["rule_set_id"])
    run_id = str(uuid.uuid4())
    engine = create_engine(database_url, pool_pre_ping=True)

    table_names = [batch["source_table"], batch["target_table"], batch["error_table"], batch["log_table"]]
    table_names.extend([item["merchant_table"] for item in batch.get("deduction_categories", [])])
    validate_columns(engine, meta, table_names, fail=bool(config.get("fail_on_missing_columns", True)))

    matched_subquery = build_matched_subquery(batch, rule_set)
    source_count_sql = f"SELECT COUNT(*) FROM {batch['source_table']} L WHERE L.BASE_YM = :base_ym"
    matched_count_sql = f"SELECT COUNT(*) FROM ({matched_subquery}) matched"
    target_insert_sql = f"""
        INSERT INTO {batch['target_table']} (
            BASE_YM, CUSTOMER_ID, DEDUCT_TYPE_CD, DEDUCT_TYPE_NM,
            TXN_COUNT, SALES_AMT, DEDUCT_AMT, REG_DTM, UPD_DTM
        )
        SELECT
            BASE_YM,
            CUSTOMER_ID,
            DEDUCT_TYPE_CD,
            DEDUCT_TYPE_NM,
            COUNT(*) AS TXN_COUNT,
            SUM(SALES_AMT) AS SALES_AMT,
            SUM(SALES_AMT) AS DEDUCT_AMT,
            NOW() AS REG_DTM,
            NULL AS UPD_DTM
        FROM ({matched_subquery}) matched
        GROUP BY BASE_YM, CUSTOMER_ID, DEDUCT_TYPE_CD, DEDUCT_TYPE_NM
    """
    target_select_sql = f"SELECT * FROM {batch['target_table']} WHERE BASE_YM = :base_ym ORDER BY CUSTOMER_ID, DEDUCT_TYPE_CD"

    error_insert_cancel_sql = f"""
        INSERT INTO {batch['error_table']} (
            RUN_ID, SALES_SEQ_NO, BASE_YM, CUSTOMER_ID, MERCHANT_ID, SALES_AMT, ERROR_CD, ERROR_MSG, REG_DTM
        )
        SELECT :run_id, L.SALES_SEQ_NO, L.BASE_YM, L.CUSTOMER_ID, L.MERCHANT_ID, L.SALES_AMT,
               'CANCEL_TXN', '취소 거래 제외', NOW()
        FROM {batch['source_table']} L
        WHERE L.BASE_YM = :base_ym AND L.CANCEL_YN <> 'N'
    """
    error_insert_status_sql = f"""
        INSERT INTO {batch['error_table']} (
            RUN_ID, SALES_SEQ_NO, BASE_YM, CUSTOMER_ID, MERCHANT_ID, SALES_AMT, ERROR_CD, ERROR_MSG, REG_DTM
        )
        SELECT :run_id, L.SALES_SEQ_NO, L.BASE_YM, L.CUSTOMER_ID, L.MERCHANT_ID, L.SALES_AMT,
               'INVALID_STATUS', '정상 매출 상태 아님', NOW()
        FROM {batch['source_table']} L
        WHERE L.BASE_YM = :base_ym AND L.CANCEL_YN = 'N' AND L.SALES_STATUS_CD <> '01'
    """
    error_insert_unmatched_sql = f"""
        INSERT INTO {batch['error_table']} (
            RUN_ID, SALES_SEQ_NO, BASE_YM, CUSTOMER_ID, MERCHANT_ID, SALES_AMT, ERROR_CD, ERROR_MSG, REG_DTM
        )
        SELECT :run_id, L.SALES_SEQ_NO, L.BASE_YM, L.CUSTOMER_ID, L.MERCHANT_ID, L.SALES_AMT,
               'NO_DEDUCT_MERCHANT', '공제 대상 가맹점 매칭 실패', NOW()
        FROM {batch['source_table']} L
        LEFT JOIN ({matched_subquery}) matched
          ON L.SALES_SEQ_NO = matched.SALES_SEQ_NO
        WHERE L.BASE_YM = :base_ym
          AND L.CANCEL_YN = 'N'
          AND L.SALES_STATUS_CD = '01'
          AND matched.SALES_SEQ_NO IS NULL
    """

    print(f"[START] {config['batch_id']} base_ym={base_ym} run_id={run_id}")
    params = {"base_ym": base_ym, "run_id": run_id}
    if dry_run:
        print("[DRY_RUN] matched SQL")
        print(matched_subquery)
        return

    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {batch['log_table']} (RUN_ID, BATCH_ID, BASE_YM, STATUS_CD, START_DTM)
            VALUES (:run_id, :batch_id, :base_ym, 'RUNNING', NOW())
        """), {"run_id": run_id, "batch_id": config["batch_id"], "base_ym": base_ym})
        try:
            source_count = execute_scalar(conn, source_count_sql, {"base_ym": base_ym})
            matched_count = execute_scalar(conn, matched_count_sql, {"base_ym": base_ym})

            if config.get("truncate_target_month_before_insert", True):
                conn.execute(text(f"DELETE FROM {batch['target_table']} WHERE BASE_YM = :base_ym"), {"base_ym": base_ym})
            conn.execute(text(f"DELETE FROM {batch['error_table']} WHERE RUN_ID = :run_id"), {"run_id": run_id})

            conn.execute(text(target_insert_sql), {"base_ym": base_ym})
            conn.execute(text(error_insert_cancel_sql), params)
            conn.execute(text(error_insert_status_sql), params)
            conn.execute(text(error_insert_unmatched_sql), params)

            target_count = execute_scalar(conn, f"SELECT COUNT(*) FROM {batch['target_table']} WHERE BASE_YM = :base_ym", {"base_ym": base_ym})
            excluded_count = execute_scalar(conn, f"SELECT COUNT(*) FROM {batch['error_table']} WHERE RUN_ID = :run_id", {"run_id": run_id})
            conn.execute(text(f"""
                UPDATE {batch['log_table']}
                   SET STATUS_CD = 'SUCCESS', END_DTM = NOW(),
                       SOURCE_COUNT = :source_count,
                       EXCLUDED_COUNT = :excluded_count,
                       TARGET_COUNT = :target_count
                 WHERE RUN_ID = :run_id
            """), {"source_count": source_count, "excluded_count": excluded_count, "target_count": target_count, "run_id": run_id})
        except Exception as exc:
            conn.execute(text(f"""
                UPDATE {batch['log_table']}
                   SET STATUS_CD = 'FAIL', END_DTM = NOW(), ERROR_MESSAGE = :error_message
                 WHERE RUN_ID = :run_id
            """), {"error_message": str(exc)[:1000], "run_id": run_id})
            raise

    output_path = None
    if config.get("write_csv", True):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{config['export_file_prefix']}_{base_ym}.csv"
        df = pd.read_sql(text(target_select_sql), engine, params={"base_ym": base_ym})
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[END] {config['batch_id']} source={source_count} matched={matched_count} excluded={excluded_count} target={target_count} file={output_path}")


def main() -> None:
    default_config = load_json(Path(__file__).with_name("config.json"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--base-ym", required=True, help="YYYYMM")
    parser.add_argument("--meta-path", default=default_config.get("meta_path", "conf/erwin_meta_complex.json"))
    parser.add_argument("--output-dir", default=default_config.get("output_dir", "output"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.database_url, args.base_ym, args.meta_path, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
