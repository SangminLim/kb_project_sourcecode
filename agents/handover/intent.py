from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agents.batch_dev.classifier.request_classifier import detect_structured_request_type

from .config import (
    CANONICAL_QUESTION_TEMPLATES, CONVERSATION_POLICY, INTENT_PATTERNS,
    QUESTION_REPLACEMENTS, REALTIME_INTENT_REGISTRY, SYSTEM_NAME_BY_ID, SYSTEM_SPECS,
)
from .utils import normalize_whitespace

def replace_aliases(question: str) -> str:
    # dictionary 단계는 오타 보정이 아니라 시스템명/업무 용어 alias 표준화만 담당한다.
    normalized = normalize_whitespace(question)
    for spec in SYSTEM_SPECS:
        canonical_name = spec["canonical_name"]
        aliases = sorted(set(spec["aliases"]), key=len, reverse=True)
        for alias in aliases:
            normalized = re.sub(re.escape(alias), canonical_name, normalized)
        normalized = normalized.replace(f"{canonical_name}은행", canonical_name)
        normalized = normalized.replace(f"{canonical_name}증권", canonical_name)
        normalized = normalized.replace(f"{canonical_name}{canonical_name}", canonical_name)
    return normalize_whitespace(normalized)


def apply_dictionary_rewrite(question: str) -> str:
    q = replace_aliases(normalize_whitespace(question))
    for src, dst in QUESTION_REPLACEMENTS.items():
        if src in q:
            q = q.replace(src, dst)
    return normalize_whitespace(q)


def detect_system_id(question: str) -> Optional[str]:
    normalized = replace_aliases(question)
    for spec in SYSTEM_SPECS:
        if spec["canonical_name"] in normalized:
            return spec["system_id"]
    return None


def detect_system_id_with_history(question: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
    current_system_id = detect_system_id(question)
    if current_system_id:
        return current_system_id
    for item in reversed(chat_history):
        history_system_id = detect_system_id(item.get("content", ""))
        if history_system_id:
            return history_system_id
    return None


def get_intent_match_scores(question: str) -> List[Tuple[str, int, int, int, int]]:
    """intent_registry.json 기준으로 intent 후보 점수를 계산한다.

    기존 방식처럼 먼저 발견된 패턴으로 바로 결정하지 않는다.
    예를 들어 "배치 흐름도"에는 "배치"와 "흐름도"가 함께 들어갈 수 있으므로,
    더 구체적인 패턴이 포함된 intent가 우선되도록 점수화한다.

    점수 구성:
    - matched_count: 매칭된 패턴 수
    - max_pattern_len: 가장 긴 매칭 패턴 길이
    - total_pattern_len: 매칭 패턴 길이 합계
    - registry_order: intent_registry.json 선언 순서 보존용

    새 intent나 패턴을 추가해도 코드 수정 없이 intent_registry.json만 확장하면 된다.
    """
    q = normalize_whitespace(question)
    candidates: List[Tuple[str, int, int, int, int]] = []

    for order, (intent, patterns) in enumerate(INTENT_PATTERNS.items()):
        matched_patterns = [
            str(pattern).strip()
            for pattern in patterns
            if str(pattern).strip() and str(pattern).strip() in q
        ]
        if not matched_patterns:
            continue

        matched_count = len(matched_patterns)
        max_pattern_len = max(len(pattern) for pattern in matched_patterns)
        total_pattern_len = sum(len(pattern) for pattern in matched_patterns)
        candidates.append((intent, matched_count, max_pattern_len, total_pattern_len, order))

    candidates.sort(key=lambda item: (item[2], item[1], item[3], -item[4]), reverse=True)
    return candidates


def detect_intent(question: str) -> str:
    q = normalize_whitespace(question)

    # 배치 요청서처럼 구조화된 입력은 업무 키워드가 아니라 request_schema.json 기준으로 판별한다.
    if detect_structured_request_type is not None:
        structured_type = detect_structured_request_type(q)
        if structured_type:
            return structured_type

    candidates = get_intent_match_scores(q)
    if candidates:
        return candidates[0][0]
    return "default"


def is_confident_intent_hint(question: str, intent: str) -> bool:
    """rewrite 전에 intent를 고정해도 되는지 판단한다.

    rewrite 전 intent는 어디까지나 힌트다. 너무 짧거나 넓은 패턴으로 잡힌 intent를
    LLM rewrite에 고정하면 "배치 흐르도"가 batch_process로 굳어지는 문제가 생긴다.

    기준값은 conversation_policy.json에서 조정 가능하다.
    - intent_hint_min_pattern_length: 기본 3
    - confident_intent_hints: 강제로 신뢰할 intent 목록
    """
    if not intent or intent == "default":
        return False

    confident_intents = set(str(v) for v in CONVERSATION_POLICY.get("confident_intent_hints", []))
    if intent in confident_intents:
        return True

    try:
        min_pattern_len = int(CONVERSATION_POLICY.get("intent_hint_min_pattern_length", 3))
    except Exception:
        min_pattern_len = 3

    for candidate_intent, _count, max_pattern_len, _total_len, _order in get_intent_match_scores(question):
        if candidate_intent == intent:
            return max_pattern_len >= min_pattern_len
    return False


def is_followup_question(question: str) -> bool:
    """이전 답변/질문을 이어서 말하는 짧은 후속 질문인지 판단한다.

    실무 기준:
    - followup 판단 로직은 코드 하드코딩이 아니라 conversation_policy.json 기반으로 동작한다.
    - followup_signals: 일반 후속 표현
    - followup_reference_terms: 이전 시스템/업무를 참조하는 표현
    """
    q = normalize_whitespace(question)
    if not q:
        return False

    followup_signals = CONVERSATION_POLICY.get("followup_signals", [])
    followup_reference_terms = CONVERSATION_POLICY.get(
        "followup_reference_terms",
        [],
    )

    followup_terms = [
        str(term).strip()
        for term in (
            list(followup_signals)
            + list(followup_reference_terms)
        )
        if str(term).strip()
    ]

    return any(term in q for term in followup_terms)


def detect_previous_user_intent(chat_history: List[Dict[str, str]]) -> str:
    """직전 사용자 발화에서만 intent를 가져온다.

    assistant 답변까지 합쳐서 detect_intent를 돌리면 답변 문구의 키워드 때문에
    현재 질문 의도가 과도하게 끌려가는 문제가 생길 수 있다.
    """
    for item in reversed(chat_history):
        if item.get("role") != "user":
            continue
        previous_question = apply_dictionary_rewrite(item.get("content", ""))
        previous_intent = detect_intent(previous_question)
        if previous_intent != "default":
            return previous_intent
    return "default"


def build_canonical_question(question: str, resolved_system_id: Optional[str], intent: str) -> str:
    """intent별 표준 질문을 설정 파일 기반으로 만든다.

    업무명/산출물 문구는 code가 아니라 conf/canonical_question_templates.json 또는
    conf/realtime_intent_registry.json에서 관리한다.
    """
    system_name = SYSTEM_NAME_BY_ID.get(resolved_system_id, "")
    template = CANONICAL_QUESTION_TEMPLATES.get(intent)

    if not template:
        realtime_spec = REALTIME_INTENT_REGISTRY.get(intent, {})
        template = str(realtime_spec.get("canonical_question") or "").strip()

    if not template:
        return question

    system_required_intents = set(CONVERSATION_POLICY.get("system_required_intents", []))
    if intent in system_required_intents and not system_name:
        return question

    try:
        return template.format(
            system_name=system_name,
            system_id=resolved_system_id or "",
            question=question,
        ).strip()
    except Exception:
        return question
