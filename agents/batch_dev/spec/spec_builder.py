from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.batch_dev.config import (
    DEFAULT_BATCH_TYPE,
    DEFAULT_OUTPUT_ENCODING,
    ERWIN_METADATA_PATH,
    SQL_TEMPLATE_DIR,
    DB_DIALECT,
    BATCH_SPEC_USE_LLM,
)
from agents.batch_dev.classifier.request_classifier import load_request_schema
from agents.batch_dev.rule.rule_engine import infer_request_capabilities, select_business_rule

try:
    from agents.batch_dev.spec.llm_spec_builder import build_batch_spec_draft_with_llm
except Exception as exc:  # LLM draft 기능은 선택 기능이므로 import 실패 시 기존 rule/parser 흐름 유지
    print(f"[WARN] llm_spec_builder import failed: {type(exc).__name__}: {exc}")
    build_batch_spec_draft_with_llm = None


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
    "classification_type": {"MERCHANT_TYPE", "MERCHANT_CLASS", "MERCHANT_CATEGORY"},
    "amount_sum": {"TOTAL_AMT", "SUM_AMT", "TOTAL_AMOUNT"},
    "row_count": {"TXN_COUNT", "CNT", "COUNT"},
}

AGGREGATION_RESULT_ALIAS_BY_ROLE = {
    "customer_id": "CUSTOMER_ID",
    "base_month": "BASE_YM",
    "classification_type": "MERCHANT_TYPE",
    "amount_sum": "TOTAL_AMT",
    "row_count": "TXN_COUNT",
    "reg_datetime": "REG_DTM",
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
    if {"CUSTOMER_ID", "BASE_YM"}.issubset(cols) and ("TOTAL_AMT" in cols or "SUM_AMT" in cols or "TXN_COUNT" in cols):
        return "monthly_summary"
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
            alias_text = str(alias).strip()
            if alias_text:
                label_to_field[alias_text] = str(field_name)

    labels_pattern = "|".join(
        re.escape(label)
        for label in sorted(label_to_field, key=len, reverse=True)
    )

    values: Dict[str, str] = {}

    if labels_pattern:
        pattern = re.compile(
            rf"(?:^|\n|\s)({labels_pattern})\s*[:：]\s*(.*?)(?=(?:\n|\s)(?:{labels_pattern})\s*[:：]|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(text):
            label = match.group(1)
            value = re.sub(r"\s+", " ", match.group(2)).strip(" \n\t-")
            field_name = label_to_field.get(label)
            if field_name and value:
                values[field_name] = value

    # request_schema가 오래되었거나 alias가 누락되어도,
    # 핵심 라벨은 요청서에서 직접 보정한다.
    # 특정 업무명/테이블명을 하드코딩하지 않고 라벨 패턴만 일반화한다.
    if not str(values.get("target_table", "")).strip():
        target_table = _extract_target_table_from_text(text)
        if target_table:
            values["target_table"] = target_table

    if not str(values.get("batch_type", "")).strip():
        batch_type = _extract_batch_type_from_text(text)
        if batch_type:
            values["batch_type"] = batch_type

    if not str(values.get("template_type", "")).strip():
        template_type = _extract_template_type_from_text(text)
        if template_type:
            values["template_type"] = template_type

    return values


def _extract_target_table_from_text(text: str) -> str:
    """요청서 원문에서 target table을 추출한다.

    request_schema.json alias 기반 파싱이 우선이며,
    이 함수는 schema 누락/구버전 설정에 대한 안전망이다.
    """
    patterns = [
        r"(?:^|\n|\s)target_table\s*[:：]\s*([A-Za-z][A-Za-z0-9_\.]+)",
        r"(?:^|\n|\s)target\s*table\s*[:：]\s*([A-Za-z][A-Za-z0-9_\.]+)",
        r"(?:^|\n|\s)적재\s*테이블\s*[:：]\s*([A-Za-z][A-Za-z0-9_\.]+)",
        r"(?:^|\n|\s)대상\s*테이블\s*[:：]\s*([A-Za-z][A-Za-z0-9_\.]+)",
        r"(?:^|\n|\s)결과\s*테이블\s*[:：]\s*([A-Za-z][A-Za-z0-9_\.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(". ,;:").upper()

    return ""


def _extract_batch_type_from_text(text: str) -> str:
    """요청서 원문에서 batch_type을 추출한다.

    예)
    - batch_type: ledger_extract_with_classification
    - 배치 유형: ledger_extract_with_classification 실행 주기: 월배치
    - 처리 유형: aggregation_to_table
    """
    patterns = [
        r"(?:^|\n|\s)batch_type\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
        r"(?:^|\n|\s)배치\s*유형\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
        r"(?:^|\n|\s)처리\s*유형\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower().replace("-", "_")
    return ""


def _extract_template_type_from_text(text: str) -> str:
    """요청서 원문에서 template_type을 추출한다."""
    patterns = [
        r"(?:^|\n|\s)template_type\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
        r"(?:^|\n|\s)template\s*type\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
        r"(?:^|\n|\s)템플릿\s*유형\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
        r"(?:^|\n|\s)템플릿\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower().replace("-", "_")
    return ""


def _merge_llm_draft_values(parser_values: Dict[str, Any], llm_draft: Dict[str, Any]) -> Dict[str, Any]:
    """request_schema 파싱 결과와 LLM draft를 병합한다.

    우선순위는 명시적 파서 값 > LLM draft > fallback 이다.
    단, 기존 request_schema가 잡지 못하는 output_columns/conditions/target_table 같은
    구조화 값은 LLM draft를 그대로 보존한다.
    """
    merged: Dict[str, Any] = dict(parser_values or {})
    if not llm_draft:
        return merged

    mapping = {
        "batch_name": "batch_name",
        "source_table": "source_table",
        "target_table": "target_table",
        "output_format": "output_format",
        "output_file": "output_file",
        "output_file_prefix": "output_file_prefix",
        "base_date_column": "base_date_column",
        "schedule_type": "schedule_type",
        "batch_type": "batch_type",
        "template_type": "template_type",
        "base_parameter": "base_parameter",
    }
    for draft_key, value_key in mapping.items():
        value = llm_draft.get(draft_key)
        if value not in (None, "", [], {}) and not str(merged.get(value_key, "")).strip():
            merged[value_key] = value

    output_columns = llm_draft.get("output_columns") or llm_draft.get("columns")
    if output_columns and not str(merged.get("columns", "")).strip():
        if isinstance(output_columns, list):
            merged["columns"] = ", ".join(str(item).strip().upper() for item in output_columns if str(item).strip())
        else:
            merged["columns"] = str(output_columns)

    if llm_draft.get("conditions"):
        merged["conditions"] = llm_draft.get("conditions")
    if llm_draft.get("joins"):
        merged["joins"] = llm_draft.get("joins")
    if llm_draft.get("validation_rules"):
        merged["llm_validation_rules"] = llm_draft.get("validation_rules")
    if llm_draft.get("llm_notes"):
        merged["llm_notes"] = llm_draft.get("llm_notes")
    if llm_draft.get("capabilities"):
        merged["capabilities"] = llm_draft.get("capabilities")

    return merged


def _safe_condition_fragment(condition: Any, context: Dict[str, Any]) -> str:
    """LLM/request condition을 WHERE 절 fragment로 쓰기 전 최소 안전 검증/정규화한다."""
    text = _normalize(str(condition or ""))
    if not text:
        return ""

    base_date_column = str(context.get("base_date_column") or "BASE_DATE")
    start_col = str(context.get("effective_start_column") or "APPLY_START_DT")
    end_col = str(context.get("effective_end_column") or "APPLY_END_DT")
    use_col = str(context.get("use_flag_column") or "USE_YN")

    text = text.replace("기준일자", ":base_date")
    text = text.replace("기준일", ":base_date")
    text = text.replace("적용시작일자", start_col)
    text = text.replace("적용시작일", start_col)
    text = text.replace("적용종료일자", end_col)
    text = text.replace("적용종료일", end_col)
    text = text.replace("종료일자", end_col)
    text = text.replace("종료일", end_col)
    text = text.replace("사용여부", use_col)

    text = text.replace("{{ base_date_column }}", base_date_column)
    text = text.replace("{{base_date_column}}", base_date_column)

    upper = text.upper()
    blocked = {";", "--", "/*", "*/"}
    if any(token in text for token in blocked):
        return ""

    blocked_keywords = {
        "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "EXEC", "CALL"
    }
    tokens = set(re.findall(r"\b[A-Za-z]+\b", upper))
    if tokens & blocked_keywords:
        return ""

    if not re.fullmatch(r"[A-Za-z0-9_:.<>=!'\"()%\s+\-/]+", text):
        return ""

    return text


def _looks_like_effective_date_condition(values: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """요청서 조건이 유효기간 조건을 뜻하는지 판단한다.

    특정 테이블명을 보지 않고, 자연어 표현과 table_role/컬럼 역할만 사용한다.
    """
    raw = values.get("conditions") or values.get("condition") or ""
    if isinstance(raw, list):
        raw_text = " ".join(str(item) for item in raw)
    else:
        raw_text = str(raw)

    text = _normalize(raw_text).lower()
    request_hint = any(token in text for token in ["사이", "between", "유효", "기간", "시작", "종료"])
    has_effective_cols = bool(context.get("effective_start_column") and context.get("effective_end_column"))
    table_role = str(context.get("table_role") or "").lower()
    return has_effective_cols and (table_role == "classification_master" or request_hint)


def _build_effective_date_conditions(context: Dict[str, Any]) -> List[str]:
    """classification/effective-dated master 공통 조건 생성.

    APPLY_START_DT 같은 물리 컬럼을 직접 고정하지 않고,
    ERWin role 또는 ROLE_SYNONYMS로 추론된 컬럼명을 사용한다.
    """
    start_col = str(context.get("effective_start_column") or "").upper()
    end_col = str(context.get("effective_end_column") or "").upper()
    use_col = str(context.get("use_flag_column") or "").upper()

    if not start_col or not end_col:
        return []

    conditions: List[str] = []
    if use_col:
        conditions.append(f"{use_col} = 'Y'")

    conditions.append(f"{start_col} <= :base_date")
    conditions.append(f"({end_col} IS NULL OR {end_col} >= :base_date)")
    return conditions


def _extract_explicit_conditions(values: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    raw_conditions = values.get("conditions") or values.get("condition") or []
    if isinstance(raw_conditions, str):
        candidates = [raw_conditions]
    elif isinstance(raw_conditions, list):
        candidates = raw_conditions
    else:
        candidates = []

    # 요청서가 "기준일자가 적용시작일자와 종료일자 사이" 같은 자연어 조건이면,
    # 안전한 SQL fragment로 직접 변환한다.
    if _looks_like_effective_date_condition(values, context):
        effective_conditions = _build_effective_date_conditions(context)
        if effective_conditions:
            return effective_conditions

    rendered: List[str] = []
    for item in candidates:
        text = _safe_condition_fragment(item, context)
        if text and text not in rendered:
            rendered.append(text)
    return rendered


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



def _select_table_by_role(meta: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    for table in meta.get("tables", []) or []:
        if _infer_table_role(table) == role:
            return table
    return None


def _columns_by_roles(table: Optional[Dict[str, Any]], roles: List[str]) -> List[str]:
    """ERWin column role 순서로 실제 target 컬럼명을 찾는다.

    rule_catalog에는 물리 컬럼명을 두지 않고 역할만 둔다.
    실제 INSERT 대상 컬럼은 erwin_meta의 target table_role/column role에서 결정한다.
    """
    resolved: List[str] = []
    for role in roles or []:
        column = _find_column_by_role(table, role)
        if not column:
            raise ValueError(f"target 테이블에서 필요한 column role을 찾지 못했습니다: {role}")
        resolved.append(column)
    return resolved


def _resolve_target_table(rule: Dict[str, Any], values: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """요청서/rule/ERWin 기준으로 target 물리 테이블을 결정한다.

    우선순위:
    1. 요청서/LLM draft가 명시한 target_table
    2. rule.target.table_role에 해당하는 ERWin table

    코드에는 특정 업무/테이블명을 넣지 않는다.
    """
    target_rule = (rule or {}).get("target") or {}
    explicit = str(values.get("target_table") or "").strip()
    if explicit:
        table_map = _table_map(meta)
        target_table = table_map.get(explicit.upper()) or {"table_name": explicit.upper(), "columns": []}
        return explicit.upper(), target_table

    target_role = str(target_rule.get("table_role") or "").strip()
    if target_role:
        target_table = _select_table_by_role(meta, target_role)
        if not target_table:
            raise ValueError(f"ERWin 메타에서 target table_role을 찾지 못했습니다: {target_role}")
        return str(target_table.get("table_name", "")).upper(), target_table

    raise ValueError("집계 결과 target table을 결정하지 못했습니다. 요청서 target_table 또는 rule_catalog.target.table_role과 ERWin table_role을 지정해야 합니다.")


def _resolve_target_columns(rule: Dict[str, Any], target_table: Dict[str, Any]) -> List[str]:
    target_rule = (rule or {}).get("target") or {}
    column_roles = [str(role) for role in target_rule.get("column_roles", []) or [] if str(role).strip()]
    if column_roles:
        return _columns_by_roles(target_table, column_roles)

    columns = _column_names(target_table)
    if columns:
        return columns

    raise ValueError("target 컬럼을 결정하지 못했습니다. rule_catalog.target.column_roles 또는 ERWin target columns를 지정해야 합니다.")


def _resolve_partition_column(rule: Dict[str, Any], target_table: Dict[str, Any], target_columns: List[str]) -> str:
    target_rule = (rule or {}).get("target") or {}
    role = str(target_rule.get("partition_column_role") or "").strip()
    if role:
        column = _find_column_by_role(target_table, role)
        if not column:
            raise ValueError(f"target partition column role을 찾지 못했습니다: {role}")
        return column
    return _infer_partition_column(rule, target_columns, str(target_rule.get("partition_param") or "base_ym"))


def _select_alias_for_target_column(target_table: Dict[str, Any], target_column: str) -> str:
    role = ""
    for col in target_table.get("columns", []) or []:
        if str(col.get("column_name", "")).upper() == target_column.upper():
            role = str(col.get("role") or "").strip()
            break
    return AGGREGATION_RESULT_ALIAS_BY_ROLE.get(role, target_column.upper())


def _requires_aggregation_path(
    capabilities: Any,
    values: Optional[Dict[str, Any]] = None,
) -> bool:
    caps = set(capabilities or [])
    if {"aggregation", "group_by"}.issubset(caps):
        return True

    # LLM draft 또는 parser가 구조화한 batch_type도 함께 본다.
    batch_type = str((values or {}).get("batch_type") or "").strip().lower()
    if batch_type == "aggregation_to_table":
        return True

    return False


def _capability_list(
    user_request: str,
    table: Optional[Dict[str, Any]],
    erwin_meta: Dict[str, Any],
    values: Optional[Dict[str, Any]] = None,
) -> List[str]:
    capabilities = set(infer_request_capabilities(user_request, table, erwin_meta))
    for item in (values or {}).get("capabilities", []) or []:
        if str(item).strip():
            capabilities.add(str(item).strip())
    return sorted(capabilities)



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

    주의:
    - classification_master의 APPLY_START_DT는 기준일자 컬럼이 아니라 유효시작일 컬럼이다.
    - source metadata 호환을 위해 base_date_column에는 start column을 둘 수 있지만,
      SQL 조건은 _build_conditions에서 effective_start/end 역할을 사용해 기간 조건으로 만든다.
    """
    raw = values.get("base_date_column") or ""
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", raw)
    if match:
        return match.group(1).upper()

    default_col = ((rule or {}).get("defaults") or {}).get("base_date_column")
    if default_col:
        return str(default_col).upper()

    table_role = _infer_table_role(table or {})

    if table_role == "classification_master":
        return (
            _find_column_by_role(table, "effective_start_date")
            or "APPLY_START_DT"
        )

    if table_role == "transaction_ledger":
        return (
            _find_column_by_role(table, "transaction_date")
            or "BASE_DATE"
        )

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


def _build_conditions(
    rule: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    values: Optional[Dict[str, Any]] = None,
) -> str:
    """WHERE 조건 생성.

    우선순위:
    1. 요청서/LLM이 명시한 조건을 안전하게 변환
    2. classification_master/effective-dated table은 role 기반 유효기간 조건 생성
    3. business rule condition 사용
    4. 최후 fallback만 단일 기준일 equality 사용
    """
    explicit_conditions = _extract_explicit_conditions(values or {}, context)
    if explicit_conditions:
        return "\n  AND ".join(explicit_conditions)

    table_role = str(context.get("table_role") or "").lower()
    if table_role == "classification_master":
        effective_conditions = _build_effective_date_conditions(context)
        if effective_conditions:
            return "\n  AND ".join(effective_conditions)

    condition_defs = (rule or {}).get("conditions") or []
    rendered = [_render_string(str(item.get("template", "")), context).strip() for item in condition_defs]
    rendered = [x for x in rendered if x]
    if rendered:
        return "\n  AND ".join(rendered)

    # 최후 fallback. 단, 유효기간 컬럼이 있는 테이블은 위에서 이미 처리되어야 한다.
    return "{{ base_date_column }} = :base_date".replace("{{ base_date_column }}", str(context.get("base_date_column") or "BASE_DATE"))


def _parameter_to_default_column(parameter_name: str) -> str:
    """배치 파라미터명에서 기본 target 파티션 컬럼명을 추론한다."""
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(parameter_name or "")).strip("_").lower()
    if normalized == "base_ym":
        return "BASE_YM"
    if normalized in {"base_date", "base_dt"}:
        return "BASE_DATE"
    return normalized.upper() if normalized else ""


def _infer_partition_param(parameters: List[Dict[str, Any]]) -> str:
    names = [str(item.get("name", "")).strip() for item in parameters or [] if str(item.get("name", "")).strip()]
    if not names:
        return ""

    for preferred in ("base_ym", "base_date", "base_dt"):
        if preferred in names:
            return preferred

    return names[0]


def _infer_partition_column(
    rule: Optional[Dict[str, Any]],
    target_columns: List[str],
    partition_param: str,
) -> str:
    target_rule = (rule or {}).get("target") or {}

    explicit = (
        target_rule.get("partition_column")
        or target_rule.get("base_column")
        or target_rule.get("delete_key_column")
    )
    if explicit:
        return str(explicit).upper()

    normalized_target_columns = [str(col).upper() for col in target_columns or []]
    default_column = _parameter_to_default_column(partition_param)

    if default_column and default_column in normalized_target_columns:
        return default_column

    for candidate in ("BASE_YM", "BASE_DATE", "BASE_DT", "STD_YM", "YYYYMM"):
        if candidate in normalized_target_columns:
            return candidate

    return default_column


def _build_execution_strategy(
    rule: Optional[Dict[str, Any]],
    target_table: str,
    target_columns: List[str],
    parameters: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rule_target = (rule or {}).get("target") or {}
    rule_strategy = (
        (rule or {}).get("execution_strategy")
        or rule_target.get("execution_strategy")
        or {}
    )

    if isinstance(rule_strategy, dict) and rule_strategy:
        strategy = dict(rule_strategy)
        strategy.setdefault("target_table", target_table)
        strategy.setdefault("partition_param", _infer_partition_param(parameters))
        strategy.setdefault(
            "partition_column",
            _infer_partition_column(rule, target_columns, str(strategy.get("partition_param", ""))),
        )
        return strategy

    load_strategy = str(
        rule_target.get("load_strategy")
        or rule_target.get("write_mode")
        or "delete_insert"
    ).strip().lower()

    if load_strategy not in {"delete_insert", "replace_partition", "delete_then_insert"}:
        return {
            "type": load_strategy or "append_only",
            "target_table": target_table,
        }

    partition_param = _infer_partition_param(parameters)
    partition_column = _infer_partition_column(rule, target_columns, partition_param)

    return {
        "type": "replace_partition",
        "target_table": target_table,
        "partition_column": partition_column,
        "partition_param": partition_param,
    }


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

    target_rule = (rule or {}).get("target") or {}
    target_table, target_table_meta = _resolve_target_table(rule, values, meta)
    target_columns = _resolve_target_columns(rule, target_table_meta)

    select_expressions: List[str] = []
    for target_column in target_columns:
        alias = _select_alias_for_target_column(target_table_meta, target_column)
        if alias == "REG_DTM":
            select_expressions.append(f"NOW() AS {target_column}")
        else:
            select_expressions.append(f"S.{alias} AS {target_column}")

    sql = (
        f"INSERT INTO {target_table} (\n    " + ",\n    ".join(target_columns) + "\n)\n"
        "SELECT\n    " + ",\n    ".join(select_expressions) + "\n"
        f"FROM (\n{select_sql}\n) S"
    )

    parameters = [{"name": "base_ym", "required": True, "description": "기준년월(YYYYMM)"}]
    partition_column = _resolve_partition_column(rule, target_table_meta, target_columns)
    target_rule_for_strategy = dict(target_rule)
    target_rule_for_strategy["partition_column"] = partition_column
    rule_for_strategy = dict(rule or {})
    rule_for_strategy["target"] = target_rule_for_strategy
    execution_strategy = _build_execution_strategy(
        rule=rule_for_strategy,
        target_table=target_table,
        target_columns=target_columns,
        parameters=parameters,
    )

    return {
        "batch_type": "aggregation_to_table",
        "batch_id": _default_batch_id(base_table_name, "AGG"),
        "parameters": parameters,
        "execution_strategy": execution_strategy,
        "source": {
            "table": base_table_name,
            "table_role": _infer_table_role(base_table),
            "column_roles": ["customer_id", "base_month", "amount"],
            "join_table_role": "classification_master",
            "dynamic_inference": True,
        },
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
            "table_role": str((target_rule or {}).get("table_role") or _infer_table_role(target_table_meta)),
            "load_strategy": target_rule.get("load_strategy") or "delete_insert",
            "execution_strategy": execution_strategy,
            "delete_sql": (
                f"DELETE FROM {execution_strategy.get('target_table')} "
                f"WHERE {execution_strategy.get('partition_column')} = :{execution_strategy.get('partition_param')}"
                if execution_strategy.get("type") == "replace_partition"
                and execution_strategy.get("target_table")
                and execution_strategy.get("partition_column")
                and execution_strategy.get("partition_param")
                else ""
            ),
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
            "table": base_table_name,
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


def _source_date_metadata(table: Optional[Dict[str, Any]], base_date_column: str) -> Dict[str, Any]:
    """source 메타데이터의 날짜 관련 설명을 생성한다."""
    table_role = _infer_table_role(table or {})
    effective_start_col = _find_column_by_role(table, "effective_start_date")
    effective_end_col = _find_column_by_role(table, "effective_end_date")
    use_col = _find_column_by_role(table, "use_flag")

    if table_role == "classification_master" and effective_start_col and effective_end_col:
        return {
            "base_date_column_role": "effective_date_range",
            "base_date_column": base_date_column,
            "effective_date": {
                "start_column": effective_start_col,
                "end_column": effective_end_col,
                "parameter": "base_date",
                "use_flag_column": use_col,
            },
        }

    return {
        "base_date_column_role": "transaction_date",
        "base_date_column": base_date_column,
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
    parser_values = _extract_labeled_values(user_request)
    erwin_meta = _load_erwin_meta()

    llm_draft: Dict[str, Any] = {}
    if BATCH_SPEC_USE_LLM and build_batch_spec_draft_with_llm is not None:
        try:
            llm_draft = build_batch_spec_draft_with_llm(
                user_request,
                erwin_meta=erwin_meta,
                request_schema=load_request_schema(),
            )
        except Exception as exc:
            llm_draft = {
                "llm_error": f"{type(exc).__name__}: {exc}",
            }

    values = _merge_llm_draft_values(parser_values, llm_draft)
    table = _find_table(user_request, values, erwin_meta)

    capabilities = _capability_list(user_request, table, erwin_meta, values)
    rule = select_business_rule(user_request, table, erwin_meta, capabilities=capabilities)

    # 요청서에 명시된 batch_type/template_type을 우선한다.
    # 예: ledger_extract_with_classification, aggregation_to_table
    # 특정 업무명을 하드코딩하지 않고 처리 패턴명만 사용한다.
    requested_rule_type = str(
        values.get("template_type")
        or values.get("batch_type")
        or ""
    ).strip().lower().replace("-", "_")

    rule_type = requested_rule_type or str(
        (rule or {}).get("rule_type")
        or (rule or {}).get("batch_type")
        or ""
    ).strip().lower().replace("-", "_")

    if not rule_type and _requires_aggregation_path(capabilities, values):
        rule_type = "monthly_aggregation"

    if rule_type in {"monthly_aggregation", "aggregation_to_table"}:
        aggregation_table = table if _infer_table_role(table or {}) == "transaction_ledger" else _select_table_by_role(erwin_meta, "transaction_ledger")
        if not aggregation_table:
            raise ValueError("집계 배치 생성을 위한 transaction_ledger 역할 테이블을 찾지 못했습니다.")
        dynamic = _build_dynamic_aggregation_spec(user_request, values, erwin_meta, rule or {}, aggregation_table)
        batch_name = values.get("batch_name") or f"{aggregation_table.get('table_kor_name', aggregation_table.get('table_name'))} 월별 집계"
        resolved = dynamic.get("resolved") or {}
        return {
            "version": "1.0",
            "batch_id": dynamic["batch_id"],
            "batch_name": batch_name,
            "batch_type": dynamic["batch_type"],
            "description": text,
            "schedule_type": _schedule_type(values, rule),
            "parameters": dynamic["parameters"],
            "execution_strategy": dynamic.get("execution_strategy", {}),
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
            "llm_spec_source": {
                "enabled": bool(BATCH_SPEC_USE_LLM),
                "used": bool(llm_draft and not llm_draft.get("llm_error")),
                "error": llm_draft.get("llm_error") if isinstance(llm_draft, dict) else None,
                "draft_keys": sorted([str(key) for key in llm_draft.keys()]) if isinstance(llm_draft, dict) else [],
                "notes": llm_draft.get("llm_notes", []) if isinstance(llm_draft, dict) else [],
                "capabilities": capabilities,
            },
            "rule_source": {
                "rule_id": (rule or {}).get("rule_id"),
                "path": (rule or {}).get("_path"),
                "mode": "dynamic_meta_inference",
                "template_type": (rule or {}).get("template_type"),
            },
        }

    if rule_type in {"ledger_extract", "ledger_extract_with_classification"}:
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
            "llm_spec_source": {
                "enabled": bool(BATCH_SPEC_USE_LLM),
                "used": bool(llm_draft and not llm_draft.get("llm_error")),
                "error": llm_draft.get("llm_error") if isinstance(llm_draft, dict) else None,
                "draft_keys": sorted([str(key) for key in llm_draft.keys()]) if isinstance(llm_draft, dict) else [],
                "notes": llm_draft.get("llm_notes", []) if isinstance(llm_draft, dict) else [],
                "capabilities": capabilities,
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
    table_role = _infer_table_role(table or {})

    context = {
        "table_name": table_name,
        "columns": ",\n    ".join(columns),
        "base_date_column": base_date_column,
        "table_role": table_role,
        "effective_start_column": _find_column_by_role(table, "effective_start_date"),
        "effective_end_column": _find_column_by_role(table, "effective_end_date"),
        "use_flag_column": _find_column_by_role(table, "use_flag"),
        "null_fn": _null_function(),
    }
    context["conditions"] = _build_conditions(rule, context, values)
    context["where_clause"] = context["conditions"]

    sql_template = str((rule or {}).get("sql_template") or "generic_export.sql.j2")
    sql = _render_sql_template(sql_template, context)

    batch_name = values.get("batch_name") or f"{table_kor_name} 파일 생성"
    batch_id = _default_batch_id(table_name)
    source_date_meta = _source_date_metadata(table, base_date_column)

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
            "table_role": table_role,
            "column_roles": columns if columns == ["*"] else [],
            **source_date_meta,
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
        "validation_rules": {
            **({"min_rows": 0, "not_null_columns": [c for c in ["MERCHANT_ID", base_date_column] if c in columns]}),
            **(values.get("llm_validation_rules") if isinstance(values.get("llm_validation_rules"), dict) else {}),
        },
        "meta_source": {
            "type": "erwin_meta",
            "path": str(ERWIN_METADATA_PATH),
            "resolved_tables": {"base": table_name} if table_name != "TODO_SOURCE_TABLE" else {},
            "resolved_columns": {
                "base_date": base_date_column,
                "effective_start": context.get("effective_start_column"),
                "effective_end": context.get("effective_end_column"),
                "use_flag": context.get("use_flag_column"),
            },
        },
        "llm_spec_source": {
            "enabled": bool(BATCH_SPEC_USE_LLM),
            "used": bool(llm_draft and not llm_draft.get("llm_error")),
            "error": llm_draft.get("llm_error") if isinstance(llm_draft, dict) else None,
            "draft_keys": sorted([str(key) for key in llm_draft.keys()]) if isinstance(llm_draft, dict) else [],
            "notes": llm_draft.get("llm_notes", []) if isinstance(llm_draft, dict) else [],
                "capabilities": capabilities,
        },
        "rule_source": {
            "rule_id": (rule or {}).get("rule_id"),
            "path": (rule or {}).get("_path"),
            "sql_template": sql_template,
            "template_type": (rule or {}).get("template_type"),
        },
    }
