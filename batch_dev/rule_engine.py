from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BUSINESS_RULE_DIR


def load_business_rules(rule_dir: Path = BUSINESS_RULE_DIR) -> List[Dict[str, Any]]:
    if not rule_dir.exists():
        return []
    rules: List[Dict[str, Any]] = []
    for path in sorted(rule_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            rule = json.load(f)
            rule["_path"] = str(path)
            rules.append(rule)
    return sorted(rules, key=lambda r: int(r.get("priority", 0)), reverse=True)


def _column_names(table: Optional[Dict[str, Any]]) -> set[str]:
    if not table:
        return set()
    return {str(c.get("column_name", "")).upper() for c in table.get("columns", []) if c.get("column_name")}


def _text_contains_any(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    return any(str(k).lower() in text.lower() for k in keywords)


def _matches_rule(rule: Dict[str, Any], *, text: str, table: Optional[Dict[str, Any]]) -> bool:
    match = rule.get("match") or {}
    table_columns = _column_names(table)

    required_columns = {str(c).upper() for c in match.get("required_columns", [])}
    if required_columns and not required_columns.issubset(table_columns):
        return False

    if not _text_contains_any(text, list(match.get("request_any", []))):
        return False

    request_all = list(match.get("request_all", []))
    if request_all and not all(str(k).lower() in text.lower() for k in request_all):
        return False

    table_name_pattern = match.get("table_name_regex")
    if table_name_pattern and table:
        table_name = str(table.get("table_name", ""))
        if not re.search(str(table_name_pattern), table_name, flags=re.IGNORECASE):
            return False

    return True


def select_business_rule(text: str, table: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for rule in load_business_rules():
        if _matches_rule(rule, text=text, table=table):
            return rule
    return None
