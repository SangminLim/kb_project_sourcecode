from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DEFAULT_BATCH_TYPE,
    DEFAULT_OUTPUT_ENCODING,
    ERWIN_METADATA_PATH,
    SQL_TEMPLATE_DIR,
    DB_DIALECT,
)
from .request_classifier import load_request_schema
from .rule_engine import select_business_rule


ROLE_SYNONYMS = {
    "customer_id": {"CUSTOMER_ID", "CUST_ID", "MBR_ID"},
    "merchant_id": {"MERCHANT_ID", "MCHT_ID", "MER_ID"},
    "base_month": {"BASE_YM", "STD_YM", "YYYYMM"},
    "transaction_date": {"SALES_DT", "APPROVAL_DT", "TRX_DT", "USE_DT", "BASE_DATE"},
    "amount": {"SALES_AMT", "APPROVAL_AMT", "USE_AMT", "AMT"},
    "cancel_flag": {"CANCEL_YN", "CNCL_YN"},
    "use_flag": {"USE_YN", "VALID_YN"},
    "effective_start_date": {"APPLY_START_DT", "START_DT", "VALID_START_DT"},
    "effective_end_date": {"APPLY_END_DT", "END_DT", "VALID_END_DT"},
    "reg_datetime": {"REG_DTM", "REG_DT", "CREATED_AT"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _null_function() -> str:
    dialect = (DB_DIALECT or "mariadb").lower()
    if dialect in {"mariadb", "mysql"}:
        return "IFNULL"
    if dialect == "oracle":
        return "NVL"
    return "IFNULL"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_erwin_meta() -> Dict[str, Any]:
    candidate_paths = [
        ERWIN_METADATA_PATH,
        Path(__file__).resolve().parent / "metadata" / "erwin_meta.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return _load_json(path, {"tables": [], "relations": []})
    return {"tables": [], "relations": []}


def _table_map(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(t.get("table_name", "")).upper(): t for t in meta.get("tables", []) if t.get("table_name")}


def _column_names(table: Optional[Dict[str, Any]]) -> List[str]:
    if not table:
        return []
    return [str(c.get("column_name", "")).upper() for c in table.get("columns", []) if c.get("column_name")]


def _find_column_by_role(table: Optional[Dict[str, Any]], role: str) -> Optional[str]:
    if not table:
        return None
    for col in table.get("columns", []) or []:
        if str(col.get("role", "")).lower() == role.lower():
            return str(col.get("column_name", "")).upper()
    names = set(_column_names(table))
    for candidate in ROLE_SYNONYMS.get(role, set()):
        if candidate in names:
            return candidate
    return None


def _infer_table_role(table: Dict[str, Any]) -> str:
    role = str(table.get("table_role", "")).strip()
    if role:
        return role
    cols = set(_column_names(table))
    if {"CUSTOMER_ID", "MERCHANT_ID"}.issubset(cols) and ("SALES_AMT" in cols or "APPROVAL_AMT" in cols or "USE_AMT" in cols):
        return "transaction_ledger"
    if "MERCHANT_ID" in cols and {"APPLY_START_DT", "APPLY_END_DT"}.issubset(cols):
        return "classification_master"
    return "generic_table"


def _classification_value(table: Dict[str, Any]) -> str:
    value = str(table.get("classification_value", "")).strip()
    if value:
        return value
    name = str(table.get("table_name", "")).upper()
    name = re.sub(r"^TB_", "", name)
    name = re.sub(r"_?MERCHANT$", "", name)
    return name or "MATCHED"


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value or "").strip("_")
    return cleaned.upper() if cleaned else "NEW_BATCH"


def _table_base_name(table_name: str) -> str:
    return re.sub(r"^TB_", "", table_name.upper())


def _default_batch_id(table_name: str, suffix: str = "EXPORT") -> str:
    return f"BATCH_{_safe_identifier(_table_base_name(table_name))}_{suffix}"


def _default_file_prefix(table_name: str) -> str:
    return _table_base_name(table_name).lower()


def _batch_id_from_name(batch_name: str, fallback: str) -> str:
    """
    요청서 배치명을 기반으로 배치 ID를 생성한다.
    특정 업무명을 if문으로 하드코딩하지 않고, 한글/특수문자는 제거한 뒤 fallback을 사용한다.

    실무에서는 배치 ID를 요청서에 명시하거나, 별도 naming rule/config로 관리하는 것이 가장 안전하다.
    """
    cleaned = _safe_identifier(batch_name)
    if cleaned and cleaned != "NEW_BATCH":
        return f"BATCH_{cleaned}"
    return fallback


def _extract_labeled_values(text: str) -> Dict[str, str]:
    schema = load_request_schema()
    fields = schema.get("fields") or {}
    label_to_field: Dict[str, str] = {}
    for field_name, field_def in fields.items():
        for alias in field_def.get("aliases") or []:
            label_to_field[str(alias)] = str(field_name)

    labels_pattern = "|".join(re.escape(label) for label in sorted(label_to_field, key=len, reverse=True))
    if not labels_pattern:
        return {}

    pattern = re.compile(
        rf"(?:^|\n|\s)({labels_pattern})\s*:\s*(.*?)(?=(?:\n|\s)(?:{labels_pattern})\s*:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    values: Dict[str, str] = {}
    for match in pattern.finditer(text):
        label = match.group(1)
        value = re.sub(r"\s+", " ", match.group(2)).strip(" \n\t-")
        field_name = label_to_field.get(label)
        if field_name and value:
            values[field_name] = value
    return values


def _extract_explicit_table(text: str, values: Dict[str, str]) -> Optional[str]:
    if values.get("source_table"):
        raw = values["source_table"]
        match = re.search(r"\b([A-Za-z][A-Za-z0-9_\.]+)\b", raw)
        if match:
            return match.group(1).strip(". ,;:").upper()
    for pattern in [r"\b(TB_[A-Za-z0-9_\.]+)\b", r"(?:테이블|table)\s*(?:은|는|:)?\s*([A-Za-z][A-Za-z0-9_\.]+)"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(". ,;:").upper()
    return None


def _find_table(text: str, values: Dict[str, str], meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tables = meta.get("tables", [])
    explicit = _extract_explicit_table(text, values)
    if explicit:
        for table in tables:
            if str(table.get("table_name", "")).upper() == explicit:
                return table
        return {"table_name": explicit, "table_kor_name": explicit, "columns": []}

    normalized_text = _normalize(text).lower()
    best_table = None
    best_score = 0
    for table in tables:
        names = [table.get("table_name", ""), table.get("table_kor_name", ""), *(table.get("aliases") or [])]
        score = sum(1 for name in names if name and str(name).lower() in normalized_text)
        if _infer_table_role(table) == "transaction_ledger" and any(k in normalized_text for k in ["집계", "월별", "통합", "원장", "대상 거래", "거래 추출"]):
            score += 2
        if score > best_score:
            best_score = score
            best_table = table
    return best_table


def _parse_columns(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[,/\s]+", value)
    return [p.strip().upper() for p in parts if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", p.strip())]


def _output_format(values: Dict[str, str], text: str) -> str:
    raw = (values.get("output_format") or values.get("output_file") or text or "").lower()
    if "xlsx" in raw or "excel" in raw or "엑셀" in raw:
        return "xlsx"
    if "txt" in raw or "전문" in raw:
        return "txt"
    return "csv"


def _file_prefix(values: Dict[str, str], table_name: str) -> str:
    output_file = values.get("output_file") or ""
    if output_file:
        name = re.sub(r"\.(csv|txt|xlsx)$", "", output_file, flags=re.IGNORECASE)
        name = re.sub(r"_?YYYYMMDD|_?\{base_date\}|_?\{yyyymmdd\}", "", name, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", name).strip("_")
        if cleaned:
            return cleaned
    return _default_file_prefix(table_name)


def _base_date_column(
    values: Dict[str, str],
    rule: Optional[Dict[str, Any]],
    table: Optional[Dict[str, Any]],
) -> str:
    """
    배치 기준일 컬럼을 결정한다.

    우선순위:
    1. 요청서에 명시된 기준일자 컬럼
    2. rule defaults에 정의된 기준일자 컬럼
    3. ERWIN 메타의 table_role / column role 기반 자동 추론
    4. 최종 fallback

    특정 테이블명을 직접 비교하지 않고 table_role과 column role을 사용한다.
    따라서 전통시장 전용 하드코딩이 아니라 classification_master 계열
    마스터 테이블 전체에 재사용 가능한 방식이다.
    """
    raw = values.get("base_date_column") or ""
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", raw)
    if match:
        return match.group(1).upper()

    default_col = ((rule or {}).get("defaults") or {}).get("base_date_column")
    if default_col:
        return str(default_col).upper()

    table_role = _infer_table_role(table or {})

    # 유효기간을 가진 마스터성 테이블은 적용시작일을 기준일 컬럼으로 표시한다.
    if table_role == "classification_master":
        return (
            _find_column_by_role(table, "effective_start_date")
            or "APPLY_START_DT"
        )

    # 거래 원장성 테이블은 거래일자를 기준일 컬럼으로 표시한다.
    if table_role == "transaction_ledger":
        return (
            _find_column_by_role(table, "transaction_date")
            or "BASE_DATE"
        )

    # 그 외 테이블은 월 기준 컬럼을 우선하고, 없으면 거래일자/BASE_DATE 순으로 fallback한다.
    return (
        _find_column_by_role(table, "base_month")
        or _find_column_by_role(table, "transaction_date")
        or "BASE_DATE"
    )


def _render_string(template: str, context: Dict[str, Any]) -> str:
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def _render_sql_template(template_name: str, context: Dict[str, Any]) -> str:
    path = SQL_TEMPLATE_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"SQL 템플릿이 없습니다: {path}")
    return _render_string(path.read_text(encoding="utf-8"), context).strip()


def _build_conditions(rule: Optional[Dict[str, Any]], context: Dict[str, Any]) -> str:
    condition_defs = (rule or {}).get("conditions") or [{"template": "{{ base_date_column }} = :base_date"}]
    rendered = [_render_string(str(item.get("template", "")), context).strip() for item in condition_defs]
    rendered = [x for x in rendered if x]
    return "\n  AND ".join(rendered) if rendered else "1 = 1"


def _schedule_type(values: Dict[str, str], rule: Optional[Dict[str, Any]]) -> str:
    return str(values.get("schedule_type") or ((rule or {}).get("defaults") or {}).get("schedule_type") or "manual")


def _relations_from_base(meta: Dict[str, Any], base_table_name: str) -> List[Dict[str, Any]]:
    base = base_table_name.upper()
    return [r for r in meta.get("relations", []) if str(r.get("left_table", "")).upper() == base]


def _join_condition(base_alias: str, join_alias: str, rel: Dict[str, Any]) -> str:
    clauses = []
    for item in rel.get("join_columns", []) or []:
        clauses.append(f"{base_alias}.{item['left_column']} = {join_alias}.{item['right_column']}")
    eff = rel.get("effective_date") or {}
    if eff.get("transaction_date_column") and eff.get("start_column") and eff.get("end_column"):
        clauses.append(
            f"{base_alias}.{eff['transaction_date_column']} BETWEEN {join_alias}.{eff['start_column']} "
            f"AND {_null_function()}({join_alias}.{eff['end_column']}, '99991231')"
        )
    return " AND ".join(clauses) if clauses else "1 = 1"


def _select_join_tables_for_classification(meta: Dict[str, Any], base_table: Dict[str, Any], text: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    tables = _table_map(meta)
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    normalized = _normalize(text).lower()
    for rel in _relations_from_base(meta, str(base_table.get("table_name", ""))):
        right_name = str(rel.get("right_table", "")).upper()
        right_table = tables.get(right_name)
        if not right_table:
            continue
        if _infer_table_role(right_table) != "classification_master":
            continue
        aliases = [right_table.get("table_name", ""), right_table.get("table_kor_name", ""), *(right_table.get("aliases") or [])]
        mentioned = any(a and str(a).lower() in normalized for a in aliases)
        broad_request = any(k in normalized for k in ["가맹점", "분류", "유형", "소득공제", "통합"])
        if mentioned or broad_request:
            results.append((right_table, rel))
    return results


def _alias_map(join_items: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, str]:
    return {str(t.get("table_name", "")).upper(): f"J{idx}" for idx, (t, _) in enumerate(join_items, start=1)}


def _build_dynamic_classification_case(join_items: List[Tuple[Dict[str, Any], Dict[str, Any]]], aliases: Dict[str, str]) -> str:
    lines = ["CASE"]
    for table, rel in join_items:
        table_name = str(table.get("table_name", "")).upper()
        alias = aliases[table_name]
        right_col = None
        join_cols = rel.get("join_columns", []) or []
        if join_cols:
            right_col = join_cols[0].get("right_column")
        right_col = right_col or _find_column_by_role(table, "merchant_id") or _column_names(table)[0]
        lines.append(f"    WHEN {alias}.{right_col} IS NOT NULL THEN '{_classification_value(table)}'")
    lines.append("    ELSE 'UNKNOWN'")
    lines.append("END")
    return "\n".join(lines)


def _build_dynamic_aggregation_spec(user_request: str, values: Dict[str, str], meta: Dict[str, Any], rule: Dict[str, Any], base_table: Dict[str, Any]) -> Dict[str, Any]:
    base_alias = "A"
    base_table_name = str(base_table.get("table_name", "TODO_SOURCE_TABLE")).upper()

    customer_col = _find_column_by_role(base_table, "customer_id")
    base_month_col = _find_column_by_role(base_table, "base_month")
    amount_col = _find_column_by_role(base_table, "amount")
    cancel_col = _find_column_by_role(base_table, "cancel_flag")
    if not customer_col or not base_month_col or not amount_col:
        missing = [name for name, col in [("customer_id", customer_col), ("base_month", base_month_col), ("amount", amount_col)] if not col]
        raise ValueError(f"집계 배치 생성에 필요한 컬럼 역할을 찾지 못했습니다: {missing}")

    join_items = _select_join_tables_for_classification(meta, base_table, user_request)
    aliases = _alias_map(join_items)
    classification_case = _build_dynamic_classification_case(join_items, aliases) if join_items else "'UNKNOWN'"

    join_lines: List[str] = []
    matched_predicates: List[str] = []
    resolved_join_tables: List[str] = []

    for table, rel in join_items:
        table_name = str(table.get("table_name", "")).upper()
        resolved_join_tables.append(table_name)
        alias = aliases[table_name]
        conditions = [_join_condition(base_alias, alias, rel)]
        use_col = _find_column_by_role(table, "use_flag")
        if use_col:
            conditions.append(f"{alias}.{use_col} = 'Y'")
        join_lines.append(f"LEFT JOIN {table_name} {alias}\n  ON " + "\n AND ".join(conditions))
        join_cols = rel.get("join_columns", []) or []
        right_col = (join_cols[0].get("right_column") if join_cols else None) or _find_column_by_role(table, "merchant_id")
        if right_col:
            matched_predicates.append(f"{alias}.{right_col} IS NOT NULL")

    where_parts = [f"{base_alias}.{base_month_col} = :base_ym"]
    if cancel_col:
        where_parts.append(f"{base_alias}.{cancel_col} = 'N'")
    if matched_predicates:
        where_parts.append("(" + " OR ".join(matched_predicates) + ")")

    select_parts = [
        f"{base_alias}.{customer_col} AS CUSTOMER_ID",
        f"{base_alias}.{base_month_col} AS BASE_YM",
        f"{classification_case} AS MERCHANT_TYPE",
        f"SUM({base_alias}.{amount_col}) AS TOTAL_AMT",
        "COUNT(*) AS TXN_COUNT",
    ]
    group_by_parts = [f"{base_alias}.{customer_col}", f"{base_alias}.{base_month_col}", classification_case]

    select_sql = (
        "SELECT\n    " + ",\n    ".join(select_parts) +
        f"\nFROM {base_table_name} {base_alias}\n" +
        "\n".join(join_lines) +
        "\nWHERE " + "\n  AND ".join(where_parts) +
        "\nGROUP BY\n    " + ",\n    ".join(group_by_parts)
    )

    target_table = str((rule.get("target") or {}).get("table") or values.get("target_table") or "TODO_TARGET_TABLE").upper()
    target_columns = list((rule.get("target") or {}).get("columns") or ["CUSTOMER_ID", "BASE_YM", "MERCHANT_TYPE", "TOTAL_AMT", "TXN_COUNT", "REG_DTM"])
    insert_columns = [c for c in target_columns if c != "REG_DTM"]
    sql = (
        f"INSERT INTO {target_table} (\n    " + ",\n    ".join(target_columns) + "\n)\n"
        "SELECT\n    " + ",\n    ".join([f"S.{c}" for c in insert_columns]) + ",\n    NOW() AS REG_DTM\n"
        f"FROM (\n{select_sql}\n) S"
    )

    return {
        "batch_type": "aggregation_to_table",
        "batch_id": _default_batch_id(base_table_name, "AGG"),
        "parameters": [{"name": "base_ym", "required": True, "description": "기준년월(YYYYMM)"}],

        # source는 물리 테이블명이 아니라 업무 역할만 표현한다.
        "source": {
            "table_role": _infer_table_role(base_table),
            "column_roles": ["customer_id", "base_month", "amount"],
            "join_table_role": "classification_master",
            "dynamic_inference": True,
        },

        # 실제 ERWin 메타 추론 결과는 별도 영역에 둔다.
        "resolved": {
            "tables": {
                "base": base_table_name,
                "joins": resolved_join_tables,
            },
            "columns": {
                "customer_id": customer_col,
                "base_month": base_month_col,
                "amount": amount_col,
                "cancel_flag": cancel_col,
            },
        },

        "target": {
            "table": target_table,
            "load_strategy": "delete_insert",
            "delete_sql": f"DELETE FROM {target_table} WHERE BASE_YM = :base_ym" if target_table != "TODO_TARGET_TABLE" else "",
            "columns": target_columns,
        },
        "sql": sql,
        "validation_rules": {"min_rows": 0, "not_null_columns": ["CUSTOMER_ID", "BASE_YM"]},
    }



def _build_dynamic_ledger_extract_spec(
    user_request: str,
    values: Dict[str, str],
    meta: Dict[str, Any],
    rule: Dict[str, Any],
    base_table: Dict[str, Any],
) -> Dict[str, Any]:
    """
    transaction_ledger + classification_master 관계를 ERWIN 메타 기반으로 해석하여
    대상 거래 추출 SQL을 생성한다.

    업무별 테이블명을 박지 않고 table_role, column role, relations를 사용한다.
    """
    base_alias = "L"
    base_table_name = str(base_table.get("table_name", "TODO_SOURCE_TABLE")).upper()

    primary_keys = base_table.get("primary_keys") or []
    sales_seq_col = str(primary_keys[0]).upper() if primary_keys else None

    transaction_date_col = _find_column_by_role(base_table, "transaction_date")
    customer_col = _find_column_by_role(base_table, "customer_id")
    merchant_col = _find_column_by_role(base_table, "merchant_id")
    amount_col = _find_column_by_role(base_table, "amount")
    base_month_col = _find_column_by_role(base_table, "base_month")
    cancel_col = _find_column_by_role(base_table, "cancel_flag")

    required = {
        "transaction_date": transaction_date_col,
        "customer_id": customer_col,
        "merchant_id": merchant_col,
        "amount": amount_col,
        "base_month": base_month_col,
    }
    missing = [role for role, col in required.items() if not col]
    if missing:
        raise ValueError(f"거래 추출 배치 생성에 필요한 컬럼 역할을 찾지 못했습니다: {missing}")

    join_items = _select_join_tables_for_classification(meta, base_table, user_request)
    if not join_items:
        raise ValueError("거래 원장과 연결된 classification_master relation을 찾지 못했습니다.")

    aliases = _alias_map(join_items)
    join_lines: List[str] = []
    matched_predicates: List[str] = []
    resolved_join_tables: List[str] = []
    case_lines = ["CASE"]

    for table, rel in join_items:
        table_name = str(table.get("table_name", "")).upper()
        resolved_join_tables.append(table_name)
        alias = aliases[table_name]

        conditions = [_join_condition(base_alias, alias, rel)]
        use_col = _find_column_by_role(table, "use_flag")
        if use_col:
            conditions.append(f"{alias}.{use_col} = 'Y'")

        join_lines.append(f"LEFT JOIN {table_name} {alias}\n  ON " + "\n AND ".join(conditions))

        join_cols = rel.get("join_columns", []) or []
        right_col = (join_cols[0].get("right_column") if join_cols else None) or _find_column_by_role(table, "merchant_id")
        if right_col:
            matched_predicates.append(f"{alias}.{right_col} IS NOT NULL")
            case_lines.append(f"    WHEN {alias}.{right_col} IS NOT NULL THEN '{_classification_value(table)}'")

    case_lines.append("    ELSE 'UNKNOWN'")
    case_lines.append("END")
    merchant_type_case = "\n".join(case_lines)

    select_columns = []
    if sales_seq_col:
        select_columns.append(f"{base_alias}.{sales_seq_col} AS SALES_SEQ_NO")
    select_columns.extend([
        f"{base_alias}.{transaction_date_col} AS SALES_DT",
        f"{base_alias}.{customer_col} AS CUSTOMER_ID",
        f"{base_alias}.{merchant_col} AS MERCHANT_ID",
        f"{base_alias}.{amount_col} AS SALES_AMT",
        f"{base_alias}.{base_month_col} AS BASE_YM",
        f"{merchant_type_case} AS MERCHANT_TYPE",
    ])

    where_parts = [f"{base_alias}.{base_month_col} = :base_ym"]
    if cancel_col:
        where_parts.append(f"{base_alias}.{cancel_col} = 'N'")
    if matched_predicates:
        where_parts.append("(" + " OR ".join(matched_predicates) + ")")

    sql = (
        "SELECT\n    " + ",\n    ".join(select_columns) +
        f"\nFROM {base_table_name} {base_alias}\n" +
        "\n".join(join_lines) +
        "\nWHERE " + "\n  AND ".join(where_parts)
    )

    batch_name = values.get("batch_name") or "소득공제 대상 거래 추출 배치"
    batch_id = _batch_id_from_name(batch_name, _default_batch_id(base_table_name, "EXTRACT"))
    output_format = _output_format(values, user_request)
    output_prefix = values.get("output_file_prefix") or _default_file_prefix(base_table_name)

    return {
        "batch_type": "db_to_file",
        "batch_id": batch_id,
        "parameters": [{"name": "base_ym", "required": True, "description": "기준년월(YYYYMM)"}],
        "source": {
            "table_role": _infer_table_role(base_table),
            "column_roles": ["transaction_date", "customer_id", "merchant_id", "amount", "base_month"],
            "join_table_role": "classification_master",
            "dynamic_inference": True,
        },
        "resolved": {
            "tables": {"base": base_table_name, "joins": resolved_join_tables},
            "columns": {
                "sales_seq": sales_seq_col,
                "transaction_date": transaction_date_col,
                "customer_id": customer_col,
                "merchant_id": merchant_col,
                "amount": amount_col,
                "base_month": base_month_col,
                "cancel_flag": cancel_col,
            },
        },
        "target": {
            "output_format": output_format,
            "output_file_prefix": output_prefix,
            "output_file_pattern": f"{output_prefix}_{{base_ym}}.{output_format}",
            "output_dir": "./output",
            "encoding": DEFAULT_OUTPUT_ENCODING,
        },
        "sql": sql,
        "validation_rules": {
            "min_rows": 0,
            "not_null_columns": ["SALES_DT", "CUSTOMER_ID", "MERCHANT_ID", "BASE_YM", "MERCHANT_TYPE"],
        },
    }


def build_batch_spec(user_request: str) -> Dict[str, Any]:
    """
    사용자 요청서/자연어를 batch_spec으로 변환한다.

    설계 원칙:
    - 업무별 if문을 두지 않는다.
    - 테이블/컬럼은 ERWin 메타에서 읽는다.
    - Rule에는 처리 패턴만 둔다.
    - SQL은 메타 역할/관계 기반으로 생성한다.
    """
    text = _normalize(user_request)
    values = _extract_labeled_values(user_request)
    erwin_meta = _load_erwin_meta()
    table = _find_table(user_request, values, erwin_meta)

    rule = select_business_rule(user_request, table, erwin_meta)
    rule_type = str((rule or {}).get("rule_type") or (rule or {}).get("batch_type") or "")

    if rule_type in {"monthly_aggregation", "aggregation_to_table"} and table:
        dynamic = _build_dynamic_aggregation_spec(user_request, values, erwin_meta, rule or {}, table)
        batch_name = values.get("batch_name") or f"{table.get('table_kor_name', table.get('table_name'))} 월별 집계"
        resolved = dynamic.get("resolved") or {}
        return {
            "version": "1.0",
            "batch_id": dynamic["batch_id"],
            "batch_name": batch_name,
            "batch_type": dynamic["batch_type"],
            "description": text,
            "schedule_type": _schedule_type(values, rule),
            "parameters": dynamic["parameters"],
            "source": dynamic["source"],
            "target": dynamic["target"],
            "sql": dynamic["sql"],
            "validation_rules": dynamic["validation_rules"],
            "meta_source": {
                "type": "erwin_meta",
                "path": str(ERWIN_METADATA_PATH),
                "resolved_tables": resolved.get("tables", {}),
                "resolved_columns": resolved.get("columns", {}),
            },
            "rule_source": {
                "rule_id": (rule or {}).get("rule_id"),
                "path": (rule or {}).get("_path"),
                "mode": "dynamic_meta_inference",
                "template_type": (rule or {}).get("template_type"),
            },
        }

    if rule_type in {"ledger_extract", "ledger_extract_with_classification"}:
        # 거래추출 배치는 단일 마스터 테이블이 아니라 거래 원장 테이블이 기준이다.
        # _find_table()이 요청서의 참조 테이블명을 먼저 잡아 TB_BOOK_PERF_MERCHANT 같은
        # classification_master를 반환할 수 있으므로, 여기서는 ERWIN table_role 기준으로
        # transaction_ledger 테이블을 다시 선택한다.
        ledger_tables = [
            t for t in erwin_meta.get("tables", [])
            if _infer_table_role(t) == "transaction_ledger"
        ]

        if not ledger_tables:
            raise ValueError("ERWIN 메타에서 transaction_ledger 테이블을 찾지 못했습니다.")

        ledger_table = ledger_tables[0]

        dynamic = _build_dynamic_ledger_extract_spec(user_request, values, erwin_meta, rule or {}, ledger_table)
        batch_name = values.get("batch_name") or "소득공제 대상 거래 추출 배치"
        resolved = dynamic.get("resolved") or {}
        return {
            "version": "1.0",
            "batch_id": dynamic["batch_id"],
            "batch_name": batch_name,
            "batch_type": dynamic["batch_type"],
            "description": text,
            "schedule_type": _schedule_type(values, rule),
            "parameters": dynamic["parameters"],
            "source": dynamic["source"],
            "target": dynamic["target"],
            "sql": dynamic["sql"],
            "validation_rules": dynamic["validation_rules"],
            "meta_source": {
                "type": "erwin_meta",
                "path": str(ERWIN_METADATA_PATH),
                "resolved_tables": resolved.get("tables", {}),
                "resolved_columns": resolved.get("columns", {}),
            },
            "rule_source": {
                "rule_id": (rule or {}).get("rule_id"),
                "path": (rule or {}).get("_path"),
                "mode": "dynamic_meta_inference",
                "template_type": (rule or {}).get("template_type"),
            },
        }

    table_name = str((table or {}).get("table_name") or "TODO_SOURCE_TABLE").upper()
    table_kor_name = str((table or {}).get("table_kor_name") or table_name)
    table_columns = _column_names(table)
    requested_columns = _parse_columns(values.get("columns", ""))
    columns = requested_columns if requested_columns else table_columns
    if table_columns:
        columns = [c for c in columns if c in table_columns]
    if not columns:
        columns = ["*"]

    batch_type = str((rule or {}).get("batch_type") or DEFAULT_BATCH_TYPE)
    output_format = _output_format(values, user_request)
    output_file_prefix = _file_prefix(values, table_name)
    base_date_column = _base_date_column(values, rule, table)

    context = {
        "table_name": table_name,
        "columns": ",\n    ".join(columns),
        "base_date_column": base_date_column,
        "null_fn": _null_function(),
    }
    context["conditions"] = _build_conditions(rule, context)
    context["where_clause"] = context["conditions"]

    sql_template = str((rule or {}).get("sql_template") or "generic_export.sql.j2")
    sql = _render_sql_template(sql_template, context)

    batch_name = values.get("batch_name") or f"{table_kor_name} 파일 생성"
    batch_id = _default_batch_id(table_name)

    return {
        "version": "1.0",
        "batch_id": batch_id,
        "batch_name": batch_name,
        "batch_type": batch_type,
        "description": text,
        "schedule_type": _schedule_type(values, rule),
        "parameters": [{"name": "base_date", "required": True, "description": "기준일자(YYYYMMDD)"}],
        "source": {
            "table_role": _infer_table_role(table or {}),
            "column_roles": columns if columns == ["*"] else [],
            "base_date_column_role": "transaction_date",
            "dynamic_inference": bool(table),
        },
        "target": {
            "output_format": output_format,
            "output_file_prefix": output_file_prefix,
            "output_file_pattern": f"{output_file_prefix}_{{base_date}}.{output_format}",
            "output_dir": "./output",
            "encoding": DEFAULT_OUTPUT_ENCODING,
        },
        "sql": sql,
        "validation_rules": {"min_rows": 0, "not_null_columns": [c for c in ["MERCHANT_ID", base_date_column] if c in columns]},
        "meta_source": {
            "type": "erwin_meta",
            "path": str(ERWIN_METADATA_PATH),
            "resolved_tables": {"base": table_name} if table_name != "TODO_SOURCE_TABLE" else {},
            "resolved_columns": {"base_date": base_date_column},
        },
        "rule_source": {
            "rule_id": (rule or {}).get("rule_id"),
            "path": (rule or {}).get("_path"),
            "sql_template": sql_template,
            "template_type": (rule or {}).get("template_type"),
        },
    }
