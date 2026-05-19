from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .config import SYSTEM_NAME_BY_ID, SYSTEM_PROMPT_BY_INTENT
from .utils import history_to_text

def _get_prompt_entry(intent: str) -> Any:
    return SYSTEM_PROMPT_BY_INTENT.get(intent, SYSTEM_PROMPT_BY_INTENT["default"])


def get_system_prompt_for_intent(intent: str) -> str:
    entry = _get_prompt_entry(intent)
    if isinstance(entry, dict):
        return str(entry.get("system_prompt") or SYSTEM_PROMPT_BY_INTENT["default"]).strip()
    return str(entry).strip()


def get_answer_rules_for_intent(intent: str, system_id: Optional[str] = None) -> List[str]:
    """prompt_templates.json에 정의된 공통/intent별 답변 규칙을 가져온다."""
    rules: List[str] = []

    common_rules = SYSTEM_PROMPT_BY_INTENT.get("_common_answer_rules", [])
    if isinstance(common_rules, list):
        rules.extend([str(rule).strip() for rule in common_rules if str(rule).strip()])

    entry = _get_prompt_entry(intent)
    if isinstance(entry, dict):
        intent_rules = entry.get("answer_rules", [])
        if isinstance(intent_rules, list):
            rules.extend([str(rule).strip() for rule in intent_rules if str(rule).strip()])

    if system_id:
        system_name = SYSTEM_NAME_BY_ID.get(system_id, system_id)
        rules.append(f"system_id가 주어진 경우 반드시 {system_name} 시스템 정보만 사용하고, 다른 시스템 이름/내용은 절대 포함하지 않는다.")

    if not rules:
        rules = [
            "반드시 한국어로만 답변한다.",
            "검색 문맥에 있는 내용만 사용한다.",
            "문맥이 부족하면 부족하다고 말한다.",
        ]

    return list(dict.fromkeys(rules))


def build_answer_prompt(
    rewritten_question: str,
    intent: str,
    search_result: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    system_id: Optional[str] = None,
) -> Tuple[str, str]:
    system_prompt = get_system_prompt_for_intent(intent)
    documents = search_result.get("documents", [[]])[0][:3]
    metadatas = search_result.get("metadatas", [[]])[0][:3]

    context_lines: List[str] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        context_lines.append(
            f"[문서 {idx}] system={meta.get('system_name', '')} section={meta.get('section', '')} title={meta.get('title', '')}\n{doc}"
        )

    answer_rules = get_answer_rules_for_intent(intent, system_id)
    answer_rule_text = "\n".join([f"- {rule}" for rule in answer_rules])
    history_text = history_to_text(chat_history)
    prompt = f"""
        이전 대화:
        {history_text if history_text else '(없음)'}

        사용자 질문:
        {rewritten_question}

        검색 문맥:
        {chr(10).join(context_lines) if context_lines else '(없음)'}

        답변 규칙:
        {answer_rule_text}
        """.strip()
    return system_prompt, prompt
