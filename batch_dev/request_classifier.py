from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .config import REQUEST_SCHEMA_PATH


def load_request_schema(path: Path = REQUEST_SCHEMA_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"request_type": "batch_development", "minimum_matched_fields": 3, "fields": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _has_label(text: str, aliases: list[str]) -> bool:
    return any(re.search(rf"(?:^|\n|\s){re.escape(alias)}\s*:", text, flags=re.IGNORECASE) for alias in aliases)


def count_matched_fields(text: str, schema: Optional[Dict[str, Any]] = None) -> int:
    schema = schema or load_request_schema()
    matched = 0
    for field_def in (schema.get("fields") or {}).values():
        if _has_label(text, list(field_def.get("aliases") or [])):
            matched += 1
    return matched


def detect_structured_request_type(text: str) -> Optional[str]:
    schema = load_request_schema()
    minimum = int(schema.get("minimum_matched_fields", 3))
    if count_matched_fields(text, schema) >= minimum:
        return str(schema.get("request_type") or "batch_development")
    return None
