from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_BATCH_TYPE,
    DEFAULT_OUTPUT_ENCODING,
    ERWIN_METADATA_PATH,
    SQL_TEMPLATE_DIR,
    DB_DIALECT,
)
from .request_classifier import load_request_schema
from .rule_engine import select_business_rule


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


def _column_names(table: Optional[Dict[str, Any]]) -> List[str]:
    if not table:
        return []
    return [str(c.get("column_name", "")).upper() for c in table.get("columns", []) if c.get("column_name")]


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value or "").strip("_")
    return cleaned.upper() if cleaned else "NEW_BATCH"


def _table_base_name(table_name: str) -> str:
    return re.sub(r"^TB_", "", table_name.upper())


def _default_batch_id(table_name: str) -> str:
    return f"BATCH_{_safe_identifier(_table_base_name(table_name))}_EXPORT"


def _default_file_prefix(table_name: str) -> str:
    return _table_base_name(table_name).lower()


def _extract_labeled_values(text: str) -> Dict[str, str]:
    """request_schema.json의 alias를 기준으로 명세형 TXT/채팅 값을 추출한다."""
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


def _base_date_column(values: Dict[str, str], rule: Optional[Dict[str, Any]], table_columns: List[str]) -> str:
    raw = values.get("base_date_column") or ""
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", raw)
    if match:
        return match.group(1).upper()
    default_col = ((rule or {}).get("defaults") or {}).get("base_date_column")
    if default_col:
        return str(default_col).upper()
    return "BASE_DATE" if "BASE_DATE" in table_columns else (table_columns[0] if table_columns else "BASE_DATE")


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


def build_batch_spec(user_request: str) -> Dict[str, Any]:
    """
    사용자 요청서/자연어를 batch_spec으로 변환한다.

    설계 원칙:
    - 업무별 if문을 두지 않는다.
    - 테이블/컬럼은 ERWIN 메타에서 읽는다.
    - 업무 규칙은 business_rules/*.json에서 읽는다.
    - SQL은 sql_templates/*.j2에서 만든다.
    """
    text = _normalize(user_request)
    values = _extract_labeled_values(user_request)
    erwin_meta = _load_erwin_meta()
    table = _find_table(user_request, values, erwin_meta)

    table_name = str((table or {}).get("table_name") or "TODO_SOURCE_TABLE").upper()
    table_kor_name = str((table or {}).get("table_kor_name") or table_name)
    table_columns = _column_names(table)

    requested_columns = _parse_columns(values.get("columns", ""))
    columns = requested_columns if requested_columns else table_columns
    if table_columns:
        columns = [c for c in columns if c in table_columns]
    if not columns:
        columns = ["*"]

    rule = select_business_rule(user_request, table)
    batch_type = str((rule or {}).get("batch_type") or DEFAULT_BATCH_TYPE)
    output_format = _output_format(values, user_request)
    output_file_prefix = _file_prefix(values, table_name)
    base_date_column = _base_date_column(values, rule, table_columns)

    context = {
        "table_name": table_name,
        "columns": ",\n    ".join(columns),
        "base_date_column": base_date_column,
        "null_fn": _null_function(),
    }
    context["conditions"] = _build_conditions(rule, context)

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
            "table": table_name,
            "columns": columns,
            "base_date_column": base_date_column,
        },
        "target": {
            "output_format": output_format,
            "output_file_prefix": output_file_prefix,
            "output_file_pattern": f"{output_file_prefix}_{{base_date}}.{output_format}",
            "output_dir": "./output",
            "encoding": DEFAULT_OUTPUT_ENCODING,
        },
        "sql": sql,
        "validation_rules": {
            "min_rows": 0,
            "not_null_columns": [c for c in ["MERCHANT_ID", base_date_column] if c in columns],
        },
        "meta_source": {
            "type": "erwin_meta",
            "path": str(ERWIN_METADATA_PATH),
            "tables": [table_name] if table_name != "TODO_SOURCE_TABLE" else [],
        },
        "rule_source": {
            "rule_id": (rule or {}).get("rule_id"),
            "path": (rule or {}).get("_path"),
            "sql_template": sql_template,
        },
    }
