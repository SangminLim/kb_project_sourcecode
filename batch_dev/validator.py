from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

SUPPORTED_BATCH_TYPES = {"db_to_file", "file_to_db", "db_to_db"}
SUPPORTED_OUTPUT_FORMATS = {"csv", "txt", "xlsx"}
BLOCKED_SQL_KEYWORDS = {
    "DELETE", "UPDATE", "INSERT", "MERGE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"
}


def _is_safe_name(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_\.]*$", value or ""))


def _find_blocked_keywords(sql: str) -> List[str]:
    tokens = set(re.findall(r"\b[A-Za-z]+\b", sql.upper()))
    return sorted(tokens & BLOCKED_SQL_KEYWORDS)


def validate_batch_spec(spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for key in ["batch_id", "batch_name", "batch_type"]:
        if not str(spec.get(key, "")).strip():
            errors.append(f"필수값 누락: {key}")

    batch_type = spec.get("batch_type")
    if batch_type not in SUPPORTED_BATCH_TYPES:
        errors.append(f"지원하지 않는 batch_type: {batch_type}")

    batch_id = spec.get("batch_id", "")
    if batch_id and not re.match(r"^BATCH_[A-Z0-9_]+$", batch_id):
        errors.append("batch_id는 BATCH_로 시작하고 영문 대문자/숫자/언더스코어만 사용하는 것을 권장합니다.")

    source = spec.get("source", {}) or {}
    table = source.get("table", "")
    if batch_type in {"db_to_file", "db_to_db"}:
        if not table or table == "TODO_SOURCE_TABLE":
            warnings.append("소스 테이블명이 확정되지 않았습니다. TODO_SOURCE_TABLE을 실제 테이블명으로 수정하세요.")
        elif not _is_safe_name(table):
            errors.append(f"테이블명 형식이 안전하지 않습니다: {table}")

        columns = source.get("columns", []) or []
        if not columns:
            warnings.append("컬럼 목록이 비어 있어 SELECT * 초안으로 생성됩니다. 운영 반영 전 명시 컬럼으로 수정하세요.")

        sql = str(spec.get("sql", "")).strip()
        if not sql:
            errors.append("SQL이 비어 있습니다.")
        else:
            blocked = _find_blocked_keywords(sql)
            if blocked:
                errors.append(f"읽기 배치 SQL에는 변경/DDL 키워드를 사용할 수 없습니다: {', '.join(blocked)}")
            if ":base_date" not in sql and "base_date" in str(spec.get("parameters", [])):
                warnings.append("SQL에 :base_date 바인드 변수가 없습니다. 기준일자 배치라면 조건을 확인하세요.")

    target = spec.get("target", {}) or {}
    output_format = target.get("output_format")
    if batch_type == "db_to_file":
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            errors.append(f"지원하지 않는 output_format: {output_format}")
        if not target.get("output_file_prefix"):
            errors.append("output_file_prefix가 필요합니다.")

    return errors, warnings
