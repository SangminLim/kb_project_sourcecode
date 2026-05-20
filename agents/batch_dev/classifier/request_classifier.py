from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from ..config import REQUEST_SCHEMA_PATH, REQUEST_CLASSIFIER_EXTRA_SIGNALS


def load_request_schema() -> Dict[str, Any]:
    if REQUEST_SCHEMA_PATH.exists():
        with REQUEST_SCHEMA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    fallback = Path(__file__).resolve().parent / "request_schema.json"
    if fallback.exists():
        with fallback.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {"fields": {}}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _field_aliases(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    fields = schema.get("fields") or {}
    result: Dict[str, Set[str]] = {}

    for field_name, field_def in fields.items():
        aliases = set()
        if isinstance(field_def, dict):
            aliases.update(str(alias).strip() for alias in field_def.get("aliases") or [] if str(alias).strip())
        if str(field_name).strip():
            aliases.add(str(field_name).strip())
        result[str(field_name)] = aliases

    return result


def _has_labeled_value(text: str, aliases: Iterable[str]) -> bool:
    """요청서 안에 '라벨: 값' 형태가 있는지 확인한다.

    특정 문구를 코드에 직접 박지 않고 request_schema aliases를 사용한다.
    """
    for alias in sorted({str(a).strip() for a in aliases if str(a).strip()}, key=len, reverse=True):
        pattern = rf"(?:^|\n|\s){re.escape(alias)}\s*[:：]\s*\S+"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def detect_structured_request_type(text: str) -> str | None:
    """배치 요청서 형태의 입력이면 batch_development intent로 분류한다.

    판별 기준:
    1. request_schema.json의 fields[].aliases 중 '라벨: 값'으로 매칭된 field 수
    2. request_schema.minimum_matched_fields 이상이면 batch_development
    3. 보조 신호는 config.REQUEST_CLASSIFIER_EXTRA_SIGNALS에서만 관리
    """
    q = _normalize(text)
    if not q:
        return None

    schema = load_request_schema()
    request_type = str(schema.get("request_type") or "batch_development")
    try:
        minimum_matched_fields = int(schema.get("minimum_matched_fields", 3))
    except Exception:
        minimum_matched_fields = 3

    matched_fields = []
    for field_name, aliases in _field_aliases(schema).items():
        if _has_labeled_value(q, aliases):
            matched_fields.append(field_name)

    if len(matched_fields) >= max(1, minimum_matched_fields):
        return request_type

    extra_signals = {str(signal).strip() for signal in REQUEST_CLASSIFIER_EXTRA_SIGNALS if str(signal).strip()}
    if extra_signals and any(signal in q for signal in extra_signals):
        return request_type

    return None
