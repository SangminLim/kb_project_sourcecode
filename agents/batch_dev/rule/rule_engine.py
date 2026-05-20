from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import BUSINESS_RULE_DIR


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
}


def _as_rule_list(payload: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    """rule_catalog.json 형태와 개별 rule json 형태를 모두 지원한다."""
    if isinstance(payload.get("rules"), list):
        rules = []
        for item in payload.get("rules") or []:
            if isinstance(item, dict):
                rule = dict(item)
                rule["_path"] = str(path)
                rules.append(rule)
        return rules

    rule = dict(payload)
    rule["_path"] = str(path)
    return [rule]


def load_business_rules(rule_dir: Path = BUSINESS_RULE_DIR) -> List[Dict[str, Any]]:
    if not rule_dir.exists():
        return []

    rules: List[Dict[str, Any]] = []
    for path in sorted(rule_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rules.extend(_as_rule_list(payload, path))

    return sorted(rules, key=lambda r: int(r.get("priority", 0)), reverse=True)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _column_names(table: Optional[Dict[str, Any]]) -> Set[str]:
    if not table:
        return set()
    return {
        str(c.get("column_name", "")).upper()
        for c in table.get("columns", []) or []
        if c.get("column_name")
    }


def _has_column_role(table: Optional[Dict[str, Any]], role: str) -> bool:
    if not table:
        return False
    for col in table.get("columns", []) or []:
        if str(col.get("role", "")).lower() == role.lower():
            return True
    names = _column_names(table)
    return bool(names & ROLE_SYNONYMS.get(role, set()))


def _table_role(table: Optional[Dict[str, Any]]) -> str:
    if not table:
        return ""

    explicit_role = str(table.get("table_role", "")).strip()
    if explicit_role:
        return explicit_role

    columns = _column_names(table)
    if {"CUSTOMER_ID", "MERCHANT_ID"}.issubset(columns) and (
        "SALES_AMT" in columns or "APPROVAL_AMT" in columns or "USE_AMT" in columns
    ):
        return "transaction_ledger"

    if "MERCHANT_ID" in columns and {"APPLY_START_DT", "APPLY_END_DT"}.issubset(columns):
        return "classification_master"

    if {"CUSTOMER_ID", "BASE_YM"}.issubset(columns) and (
        "TOTAL_AMT" in columns or "SUM_AMT" in columns or "TXN_COUNT" in columns
    ):
        return "monthly_summary"

    return "generic_table"


def _available_roles(table: Optional[Dict[str, Any]], erwin_meta: Optional[Dict[str, Any]]) -> Set[str]:
    roles: Set[str] = set()
    role = _table_role(table)
    if role:
        roles.add(role)

    if erwin_meta:
        for item in erwin_meta.get("tables", []) or []:
            item_role = _table_role(item)
            if item_role:
                roles.add(item_role)
    return roles


def _tables_by_role(erwin_meta: Optional[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    if not erwin_meta:
        return []
    return [t for t in erwin_meta.get("tables", []) or [] if _table_role(t) == role]


def _relations_from_role(erwin_meta: Optional[Dict[str, Any]], left_role: str, right_role: str) -> List[Dict[str, Any]]:
    if not erwin_meta:
        return []
    tables = {str(t.get("table_name", "")).upper(): t for t in erwin_meta.get("tables", []) or []}
    results = []
    for rel in erwin_meta.get("relations", []) or []:
        left = tables.get(str(rel.get("left_table", "")).upper())
        right = tables.get(str(rel.get("right_table", "")).upper())
        if _table_role(left) == left_role and _table_role(right) == right_role:
            results.append(rel)
    return results


def infer_request_capabilities(
    text: str,
    table: Optional[Dict[str, Any]],
    erwin_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """요청서/ERWin 역할/관계에서 처리 capability를 추론한다.

    특정 배치명이나 특정 테이블명을 if문으로 보지 않고,
    출력 컬럼명, 처리 의미, table_role, relation 존재 여부를 조합한다.
    """
    normalized = _normalize_text(text)
    upper_text = str(text or "").upper()
    capabilities: Set[str] = set()

    roles = _available_roles(table, erwin_meta)
    current_role = _table_role(table)

    if current_role:
        capabilities.add(f"source_role:{current_role}")
    if "transaction_ledger" in roles:
        capabilities.add("aggregation_source")
    if "classification_master" in roles:
        capabilities.add("classification_master_available")

    if _relations_from_role(erwin_meta, "transaction_ledger", "classification_master"):
        capabilities.add("classification_relation_available")

    # 구조적 출력/측정값 단서
    measure_signals = {"TOTAL_AMT", "TXN_COUNT", "SUM_AMT", "COUNT"}
    if any(signal in upper_text for signal in measure_signals):
        capabilities.add("aggregation")
    if any(signal in upper_text for signal in {"CUSTOMER_ID", "BASE_YM", "MERCHANT_TYPE"}):
        capabilities.add("group_by")
    if "BASE_YM" in upper_text or "기준년월" in normalized:
        capabilities.add("base_month_parameter")
    if "MERCHANT_TYPE" in upper_text or "가맹점유형" in normalized or "가맹점분류" in normalized:
        capabilities.add("classification_join")
    if "APPLY_START_DT" in upper_text or "APPLY_END_DT" in upper_text or "유효기간" in normalized or "적용기간" in normalized:
        capabilities.add("effective_date_matching")
    if "CANCEL_YN" in upper_text or "취소" in normalized:
        capabilities.add("exclude_cancelled")
    if "USE_YN" in upper_text or "사용여부" in normalized:
        capabilities.add("use_flag_filter")
    if "deleteinsert" in normalized or "delete_insert" in normalized or "replacepartition" in normalized or "replace_partition" in normalized or "delete insert" in str(text or "").lower() or "재처리" in normalized:
        capabilities.add("replace_partition")
    if any(token in normalized for token in ["파일", "csv", "txt", "xlsx", "추출", "생성"]):
        capabilities.add("file_export")

    # table role 자체에서 보강
    if _has_column_role(table, "base_month"):
        capabilities.add("base_month_parameter")
    if _has_column_role(table, "cancel_flag"):
        capabilities.add("exclude_cancelled")

    return sorted(capabilities)


def _text_contains_any(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    normalized_text = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized_text for keyword in keywords)


def _text_contains_all(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    normalized_text = _normalize_text(text)
    return all(_normalize_text(keyword) in normalized_text for keyword in keywords)


def _text_contains_excluded(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    normalized_text = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized_text for keyword in keywords)


def _matches_required_columns(match: Dict[str, Any], table: Optional[Dict[str, Any]]) -> bool:
    table_columns = _column_names(table)
    required_columns = {str(c).upper() for c in match.get("required_columns", []) or []}
    return not required_columns or required_columns.issubset(table_columns)


def _matches_table_role(
    match: Dict[str, Any],
    table: Optional[Dict[str, Any]],
    erwin_meta: Optional[Dict[str, Any]],
) -> bool:
    current_role = _table_role(table)
    available_roles = _available_roles(table, erwin_meta)

    required_table_role = str(match.get("required_table_role") or "").strip()
    if required_table_role and current_role != required_table_role:
        return False

    required_table_roles = {
        str(role).strip()
        for role in match.get("required_table_roles", []) or []
        if str(role).strip()
    }
    if required_table_roles:
        if erwin_meta:
            if not required_table_roles.issubset(available_roles):
                return False
        elif current_role not in required_table_roles:
            return False

    table_role_any = {
        str(role).strip()
        for role in match.get("table_role_any", []) or []
        if str(role).strip()
    }
    if table_role_any and current_role not in table_role_any:
        return False

    return True


def _matches_capabilities(match: Dict[str, Any], capabilities: Set[str]) -> bool:
    required = {str(item).strip() for item in match.get("required_capabilities", []) or [] if str(item).strip()}
    if required and not required.issubset(capabilities):
        return False

    any_of = {str(item).strip() for item in match.get("capability_any", []) or [] if str(item).strip()}
    if any_of and not (any_of & capabilities):
        return False

    excluded = {str(item).strip() for item in match.get("exclude_capabilities", []) or [] if str(item).strip()}
    if excluded and (excluded & capabilities):
        return False

    return True


def _matches_table_name(match: Dict[str, Any], table: Optional[Dict[str, Any]]) -> bool:
    table_name_pattern = match.get("table_name_regex")
    if table_name_pattern and table:
        table_name = str(table.get("table_name", ""))
        if not re.search(str(table_name_pattern), table_name, flags=re.IGNORECASE):
            return False
    return True


def _matches_rule(
    rule: Dict[str, Any],
    *,
    text: str,
    table: Optional[Dict[str, Any]],
    erwin_meta: Optional[Dict[str, Any]] = None,
    capabilities: Optional[Set[str]] = None,
) -> bool:
    match = rule.get("match") or {}

    if _text_contains_excluded(text, list(match.get("exclude_any", []) or [])):
        return False

    request_any = list(match.get("request_any", []) or match.get("required_any", []) or [])
    if not _text_contains_any(text, request_any):
        return False

    request_all = list(match.get("request_all", []) or match.get("required_all", []) or [])
    if not _text_contains_all(text, request_all):
        return False

    if not _matches_required_columns(match, table):
        return False
    if not _matches_table_role(match, table, erwin_meta):
        return False
    if not _matches_capabilities(match, capabilities or set()):
        return False
    if not _matches_table_name(match, table):
        return False

    return True


def select_business_rule(
    text: str,
    table: Optional[Dict[str, Any]],
    erwin_meta: Optional[Dict[str, Any]] = None,
    capabilities: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """요청서/테이블/ERWin/capability 기반 rule 선택.

    - 업무명별 if문을 두지 않는다.
    - rule_catalog.json의 match 조건으로 패턴을 선택한다.
    - capability가 명시되지 않으면 infer_request_capabilities로 결정적으로 추론한다.
    """
    capability_set = set(capabilities or infer_request_capabilities(text, table, erwin_meta))

    for rule in load_business_rules():
        if _matches_rule(
            rule,
            text=text,
            table=table,
            erwin_meta=erwin_meta,
            capabilities=capability_set,
        ):
            rule = dict(rule)
            rule.setdefault("_matched_capabilities", sorted(capability_set))
            return rule

    return None
