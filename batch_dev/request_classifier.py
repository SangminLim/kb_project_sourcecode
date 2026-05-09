from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import REQUEST_SCHEMA_PATH


def load_request_schema() -> Dict[str, Any]:
    if REQUEST_SCHEMA_PATH.exists():
        with REQUEST_SCHEMA_PATH.open('r', encoding='utf-8') as f:
            return json.load(f)
    fallback = Path(__file__).resolve().parent / 'request_schema.json'
    if fallback.exists():
        with fallback.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {'fields': {}}


def detect_structured_request_type(text: str) -> str | None:
    """배치 요청서 형태의 입력이면 batch_development intent로 분류한다."""
    q = (text or '').strip()
    if not q:
        return None
    signals = ['[배치 개발 요청서]', '배치명:', '대상 테이블:', '처리 내용:', '출력:']
    if any(signal in q for signal in signals):
        return 'batch_development'
    return None
