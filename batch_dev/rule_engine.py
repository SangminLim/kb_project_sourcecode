from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import BUSINESS_RULE_DIR


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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _column_names(table: Optional[Dict[str, Any]]) -> Set[str]:
    if not table:
        return set()

    return {
        str(c.get("column_name", "")).upper()
        for c in table.get("columns", [])
        if c.get("column_name")
    }


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


def _text_contains_any(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True

    normalized_text = _normalize_text(text)
    return any(_normalize_text(str(keyword)) in normalized_text for keyword in keywords)


def _text_contains_all(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True

    normalized_text = _normalize_text(text)
    return all(_normalize_text(str(keyword)) in normalized_text for keyword in keywords)


def _text_contains_excluded(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False

    normalized_text = _normalize_text(text)
    return any(_normalize_text(str(keyword)) in normalized_text for keyword in keywords)


def _matches_required_columns(match: Dict[str, Any], table: Optional[Dict[str, Any]]) -> bool:
    table_columns = _column_names(table)
    required_columns = {str(c).upper() for c in match.get("required_columns", [])}

    if required_columns and not required_columns.issubset(table_columns):
        return False

    return True


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
        else:
            if current_role not in required_table_roles:
                return False

    table_role_any = {
        str(role).strip()
        for role in match.get("table_role_any", []) or []
        if str(role).strip()
    }
    if table_role_any and current_role not in table_role_any:
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
) -> bool:
    match = rule.get("match") or {}

    exclude_any = list(match.get("exclude_any", []) or [])
    if _text_contains_excluded(text, exclude_any):
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

    if not _matches_table_name(match, table):
        return False

    return True


def select_business_rule(
    text: str,
    table: Optional[Dict[str, Any]],
    erwin_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    요청서 텍스트, 기준 테이블, ERWIN 메타를 기준으로 업무 rule을 선택한다.

    핵심 방향:
    - 업무명별 if문을 두지 않는다.
    - rule_catalog.json의 match 조건으로 패턴을 선택한다.
    - table_role / required_table_roles / exclude_any를 지원한다.
    - erwin_meta를 넘기면 전체 메타 기준 역할 매칭도 가능하다.
    """
    for rule in load_business_rules():
        if _matches_rule(rule, text=text, table=table, erwin_meta=erwin_meta):
            return rule

    return None
