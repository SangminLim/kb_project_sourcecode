from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple

from ..config import SUPPORTED_BATCH_TYPES, SUPPORTED_OUTPUT_FORMATS

READ_SQL_BLOCKED_KEYWORDS = {
    "DELETE", "UPDATE", "INSERT", "MERGE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"
}
DDL_BLOCKED_KEYWORDS = {"DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"}


def _is_safe_name(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_\.]*$", value or ""))


def _find_keywords(sql: str, keywords: set[str]) -> List[str]:
    tokens = set(re.findall(r"\b[A-Za-z]+\b", sql.upper()))
    return sorted(tokens & keywords)


def _extract_table_columns_from_erwin(erwin_meta: Mapping[str, Any], table_name: str) -> set[str]:
    target = str(table_name or "").upper()
    for table in erwin_meta.get("tables", []) or []:
        if str(table.get("table_name", "")).upper() != target:
            continue
        return {
            str(col.get("column_name", "")).upper()
            for col in table.get("columns", []) or []
            if col.get("column_name")
        }
    return set()


def _source_table(spec: Mapping[str, Any]) -> str:
    source = spec.get("source", {}) if isinstance(spec.get("source", {}), Mapping) else {}
    meta_source = spec.get("meta_source", {}) if isinstance(spec.get("meta_source", {}), Mapping) else {}
    resolved_tables = meta_source.get("resolved_tables", {}) if isinstance(meta_source.get("resolved_tables", {}), Mapping) else {}
    return str(source.get("table") or resolved_tables.get("base") or "").strip()


def _source_columns(spec: Mapping[str, Any]) -> List[str]:
    source = spec.get("source", {}) if isinstance(spec.get("source", {}), Mapping) else {}
    columns = source.get("columns") or spec.get("output_columns") or []
    if isinstance(columns, str):
        return [item.strip().upper() for item in re.split(r"[,\s]+", columns) if item.strip()]
    if isinstance(columns, list):
        return [str(item).strip().upper() for item in columns if str(item).strip()]
    return []


def validate_batch_spec(spec: Dict[str, Any], erwin_meta: Mapping[str, Any] | None = None) -> Tuple[List[str], List[str]]:
    """batch_spec 공통 검증.

    LLM이 만든 spec도 운영 반영 전에는 반드시 결정적 검증을 통과해야 한다.
    특정 업무명/테이블명을 하드코딩하지 않고 spec 구조, SQL 안전성, ERWin 메타 일치성만 확인한다.
    """
    errors: List[str] = []
    warnings: List[str] = []

    for key in ["batch_id", "batch_name", "batch_type"]:
        if not str(spec.get(key, "")).strip():
            errors.append(f"필수값 누락: {key}")

    batch_type = spec.get("batch_type")
    if batch_type not in SUPPORTED_BATCH_TYPES:
        errors.append(f"지원하지 않는 batch_type: {batch_type}")

    batch_id = str(spec.get("batch_id", ""))
    if batch_id and not re.match(r"^BATCH_[A-Z0-9_]+$", batch_id):
        warnings.append("batch_id는 BATCH_로 시작하고 영문 대문자/숫자/언더스코어만 사용하는 것을 권장합니다.")

    table = _source_table(spec)
    if batch_type in {"db_to_file", "db_to_db", "aggregation_to_table"}:
        if not table or table == "TODO_SOURCE_TABLE":
            warnings.append("소스 테이블명이 확정되지 않았습니다. TODO_SOURCE_TABLE을 실제 테이블명으로 수정하세요.")
        elif not _is_safe_name(table):
            errors.append(f"테이블명 형식이 안전하지 않습니다: {table}")

        columns = _source_columns(spec)
        if batch_type == "db_to_file" and (not columns or columns == ["*"]):
            warnings.append("컬럼 목록이 비어 있거나 SELECT * 구조입니다. 운영 반영 전 명시 컬럼으로 수정하세요.")

        if erwin_meta and table and table != "TODO_SOURCE_TABLE":
            meta_columns = _extract_table_columns_from_erwin(erwin_meta, table)
            if not meta_columns:
                warnings.append(f"ERWin 메타에서 소스 테이블을 확인하지 못했습니다: {table}")
            elif columns and columns != ["*"]:
                missing = [col for col in columns if col not in meta_columns]
                if missing:
                    errors.append(f"ERWin 메타에 없는 컬럼이 batch_spec에 포함되어 있습니다: {', '.join(missing)}")

        sql = str(spec.get("sql", "")).strip()
        if not sql:
            errors.append("SQL이 비어 있습니다.")
        else:
            ddl_blocked = _find_keywords(sql, DDL_BLOCKED_KEYWORDS)
            if ddl_blocked:
                errors.append(f"배치 SQL에는 DDL/권한 키워드를 사용할 수 없습니다: {', '.join(ddl_blocked)}")

            if batch_type == "db_to_file":
                blocked = _find_keywords(sql, READ_SQL_BLOCKED_KEYWORDS)
                if blocked:
                    errors.append(f"파일 생성 조회 SQL에는 변경/DDL 키워드를 사용할 수 없습니다: {', '.join(blocked)}")
                if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
                    errors.append("db_to_file 배치 SQL은 SELECT/WITH로 시작해야 합니다.")

            if batch_type == "aggregation_to_table":
                if re.search(r"\bDELETE\s+FROM\b", sql, flags=re.IGNORECASE) and not re.search(r"\bWHERE\b", sql, flags=re.IGNORECASE):
                    errors.append("DELETE 선처리 SQL에는 WHERE 조건이 필요합니다.")
                if not re.search(r"\b(INSERT\s+INTO|SELECT)\b", sql, flags=re.IGNORECASE):
                    warnings.append("aggregation_to_table SQL에서 INSERT/SELECT 패턴을 확인하지 못했습니다.")

            parameter_text = str(spec.get("parameters", []))
            if ":base_date" not in sql and "base_date" in parameter_text:
                warnings.append("SQL에 :base_date 바인드 변수가 없습니다. 기준일자 배치라면 조건을 확인하세요.")
            if ":base_ym" not in sql and "base_ym" in parameter_text:
                warnings.append("SQL에 :base_ym 바인드 변수가 없습니다. 기준년월 배치라면 조건을 확인하세요.")

    target = spec.get("target", {}) or {}
    if not isinstance(target, Mapping):
        target = {}
    output_format = target.get("output_format")
    if batch_type == "db_to_file":
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            errors.append(f"지원하지 않는 output_format: {output_format}")
        if not target.get("output_file_prefix"):
            errors.append("output_file_prefix가 필요합니다.")
        if not target.get("output_file_pattern"):
            warnings.append("output_file_pattern이 없어 파일명 규칙 확인이 필요합니다.")

    llm_source = spec.get("llm_spec_source")
    if isinstance(llm_source, Mapping) and llm_source.get("enabled") and not llm_source.get("used"):
        warnings.append(f"LLM batch_spec 초안 생성이 실패하여 rule/parser fallback을 사용했습니다: {llm_source.get('error')}")

    return errors, warnings
