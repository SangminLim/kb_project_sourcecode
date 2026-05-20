from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

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


def _normalize_label(text: str) -> str:
    """라벨 비교용 정규화.

    예)
    - "기준 테이블" == "기준테이블"
    - "[업무목적]" == "업무 목적"
    - "배치명]" == "배치명"
    - "[출력컬럼" == "출력컬럼"
    """
    value = str(text or "").strip()
    value = re.sub(r"^[\[\(【]\s*", "", value)
    value = re.sub(r"\s*[\]\)】]$", "", value)
    value = value.rstrip(":：")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def _field_aliases(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    fields = schema.get("fields") or {}
    result: Dict[str, Set[str]] = {}

    for field_name, field_def in fields.items():
        aliases = set()
        if isinstance(field_def, dict):
            aliases.update(
                str(alias).strip()
                for alias in field_def.get("aliases") or []
                if str(alias).strip()
            )
        if str(field_name).strip():
            aliases.add(str(field_name).strip())
        result[str(field_name)] = aliases

    return result


def _alias_regex(alias: str) -> str:
    """alias 내부 공백 유무 차이를 허용하는 regex를 만든다.

    request_schema.json의 alias를 기준으로 동작하므로
    특정 배치명/업무명/테이블명을 코드에 하드코딩하지 않는다.
    """
    normalized = re.sub(r"\s+", "", str(alias or "").strip())
    if not normalized:
        return ""
    return r"\s*".join(re.escape(ch) for ch in normalized)


def _is_meaningful_value(value: str) -> bool:
    """라벨 뒤에 실제 값이 있는지 확인한다.

    참조테이블처럼 값이 '-'인 선택 항목은 매칭 필드에서 제외해도 된다.
    minimum_matched_fields 기준을 넘기면 요청서로 인정된다.
    """
    cleaned = str(value or "").strip()
    if not cleaned:
        return False
    if cleaned in {"-", "없음", "N/A", "n/a", "null", "NULL"}:
        return False
    return True


def _has_inline_labeled_value(text: str, aliases: Iterable[str]) -> bool:
    """인라인 라벨 값을 감지한다.

    지원 형식:
    - 배치명: 값
    - 배치명：값
    - [배치명] 값
    - 배치명] 값      # 여는 대괄호 누락
    - [배치명 값      # 닫는 대괄호 누락
    """
    for alias in sorted({str(a).strip() for a in aliases if str(a).strip()}, key=len, reverse=True):
        alias_pattern = _alias_regex(alias)
        if not alias_pattern:
            continue

        patterns = [
            rf"(?:^|\n|\s){alias_pattern}\s*[:：]\s*(?P<value>[^\n]+)",
            rf"(?:^|\n|\s)\[\s*{alias_pattern}\s*\]\s*(?P<value>[^\n]+)",
            rf"(?:^|\n|\s){alias_pattern}\s*\]\s*(?P<value>[^\n]+)",
            rf"(?:^|\n|\s)\[\s*{alias_pattern}\s+(?P<value>[^\n]+)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if _is_meaningful_value(match.group("value")):
                    return True

    return False


def _schema_label_keys(schema: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for aliases in _field_aliases(schema).values():
        for alias in aliases:
            key = _normalize_label(alias)
            if key:
                keys.add(key)
    return keys


def _extract_loose_sections(text: str, schema: Dict[str, Any]) -> Dict[str, str]:
    """섹션형 요청서를 느슨하게 추출한다.

    사람이 직접 쓰는 요청서는 대괄호가 깨지거나, 한 줄에 값이 붙을 수 있다.
    schema alias에 있는 라벨만 섹션 라벨로 인정한다.

    지원:
    - [배치명]
    - [배치명] 전통시장 가맹점 파일 생성
    - 배치명] 전통시장 가맹점 파일 생성
    - [배치명 전통시장 가맹점 파일 생성
    - 배치명: 전통시장 가맹점 파일 생성
    """
    valid_labels = _schema_label_keys(schema)
    sections: Dict[str, List[str]] = {}
    current_label: str | None = None

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        detected_label: str | None = None
        tail = ""

        # [라벨] 값 / 라벨] 값 / [라벨 값 / 라벨: 값 모두 느슨하게 처리
        match = re.match(
            r"^\s*\[?\s*(?P<label>[^\]\[:：\n]+?)\s*\]?\s*(?P<tail>.*)$",
            stripped,
        )
        if match:
            candidate = _normalize_label(match.group("label"))
            if candidate in valid_labels:
                detected_label = candidate
                tail = match.group("tail").strip()
                if tail.startswith(":") or tail.startswith("："):
                    tail = tail[1:].strip()

        if detected_label:
            current_label = detected_label
            sections.setdefault(current_label, [])
            if tail:
                sections[current_label].append(tail)
            continue

        if current_label:
            sections[current_label].append(line)

    return {label: "\n".join(lines).strip() for label, lines in sections.items()}


def _has_section_labeled_value(text: str, aliases: Iterable[str], schema: Dict[str, Any]) -> bool:
    """섹션형 라벨 다음의 값을 확인한다."""
    sections = _extract_loose_sections(text, schema)
    if not sections:
        return False

    alias_keys = {_normalize_label(alias) for alias in aliases if str(alias).strip()}
    for label, value in sections.items():
        if label in alias_keys and _is_meaningful_value(value):
            return True

    return False


def _has_labeled_value(text: str, aliases: Iterable[str], schema: Dict[str, Any]) -> bool:
    """요청서 안에 schema alias 기반 라벨 값이 있는지 확인한다."""
    return (
        _has_inline_labeled_value(text, aliases)
        or _has_section_labeled_value(text, aliases, schema)
    )


def _matched_fields(text: str, schema: Dict[str, Any]) -> List[str]:
    matched: List[str] = []
    for field_name, aliases in _field_aliases(schema).items():
        if _has_labeled_value(text, aliases, schema):
            matched.append(field_name)
    return matched


def detect_structured_request_type(text: str) -> str | None:
    """배치 요청서 형태의 입력이면 batch_development intent로 분류한다.

    판별 기준은 request_schema.json에 둔다.
    특정 배치명, 특정 테이블명, 특정 업무명을 코드에 하드코딩하지 않는다.

    1. request_schema.json의 fields[].aliases 기준으로 라벨을 찾는다.
    2. 라벨 형식은 '라벨: 값', '[라벨] 값', '[라벨]\\n값',
       '라벨] 값', '[라벨 값'을 지원한다.
    3. minimum_matched_fields 이상이면 request_type으로 분류한다.
    4. optional/required 설정은 schema에서 관리한다.
    """
    raw_text = str(text or "")
    q = _normalize(raw_text)
    if not q:
        return None

    schema = load_request_schema()
    request_type = str(schema.get("request_type") or "batch_development")

    try:
        minimum_matched_fields = int(schema.get("minimum_matched_fields", 3))
    except Exception:
        minimum_matched_fields = 3

    matched_fields = _matched_fields(raw_text, schema)

    if len(matched_fields) >= max(1, minimum_matched_fields):
        return request_type

    extra_signals = {
        str(signal).strip()
        for signal in REQUEST_CLASSIFIER_EXTRA_SIGNALS
        if str(signal).strip()
    }
    if extra_signals and any(signal in q for signal in extra_signals):
        return request_type

    return None
