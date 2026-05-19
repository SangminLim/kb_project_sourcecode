from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import FEW_SHOT_EXAMPLES, INTENT_PATTERNS, CONVERSATION_POLICY, SYSTEM_NAME_BY_ID, ChatConfig
from .intent import apply_dictionary_rewrite, build_canonical_question, detect_system_id
from .llm_client import _langchain_generate_text, ollama_generate
from .utils import history_to_text, normalize_whitespace

def build_rewrite_prompt(
    question: str,
    chat_history: List[Dict[str, str]],
    resolved_system_name: Optional[str],
    resolved_intent: str,
) -> Tuple[str, str]:
    """LLM rewrite 전용 프롬프트를 만든다.

    실무 원칙:
    - LLM은 답변하지 않고 검색 가능한 질문 1문장만 만든다.
    - dictionary는 업무 용어/시스템 alias 표준화까지만 담당한다.
    - 오타, 띄어쓰기, 구어체, 축약 표현은 LLM이 추론한다.
    - 의도가 이미 확정된 경우에는 의도를 바꾸지 않는다.
    - 의도가 default인 경우에는 LLM이 문맥상 가장 자연스러운 검색 질문으로 정리한다.
    """
    fixed_intent = resolved_intent if resolved_intent and resolved_intent != "default" else "(아직 확정되지 않음)"
    canonical_hint = ""
    if resolved_intent and resolved_intent != "default":
        canonical_hint = build_canonical_question(question, detect_system_id(question), resolved_intent)

    system_prompt = (
        "너는 사내 업무 질의 재작성기다. "
        "사용자의 오타, 띄어쓰기 오류, 축약어, 구어체를 검색 가능한 한국어 질문 1문장으로 바꾼다. "
        "절대로 답변하지 말고, 설명하지 말고, JSON을 출력하지 말고, 질문 1문장만 출력한다. "
        "시스템명, 업무명, 조회 대상, 사용자가 요청한 산출물의 의미를 임의로 바꾸지 않는다."
    )
    examples = "\n\n".join([f"[사용자]\n{ex['user']}\n[재작성]\n{ex['assistant']}" for ex in FEW_SHOT_EXAMPLES])
    history_text = history_to_text(chat_history)
    intent_names = ", ".join(INTENT_PATTERNS.keys())
    canonical_rule = f"- 가능한 경우 다음 표준 질문에 가깝게 정리한다: {canonical_hint}" if canonical_hint else "- 의도가 확정되지 않았으면 질문 의미를 보존한 채 검색 가능한 표현으로만 정리한다."

    prompt = f"""
            다음 규칙을 지켜라.
            1) 이미 결정된 시스템명: {resolved_system_name or '(없음)'}
            2) 이미 결정된 의도: {fixed_intent}
            3) 의도가 확정된 경우에는 그 의도를 바꾸지 않는다.
            4) 의도가 확정되지 않은 경우에는 가능한 의도 후보({intent_names}) 중 하나로 검색될 수 있게 표현만 정리한다.
            5) 시스템명이 결정된 경우에는 다른 시스템명을 새로 추정하거나 바꾸지 않는다.
            6) dictionary에서 표준화된 업무 용어는 유지한다.
            7) 오타, 띄어쓰기, 구어체, 축약 표현은 자연스럽게 보정한다.
            {canonical_rule}
            8) 반드시 한국어 질문 한 줄만 출력한다.

            예시:
            {examples if examples else '(없음)'}

            이전 대화:
            {history_text if history_text else '(없음)'}

            현재 질문:
            {question}
            """.strip()
    return system_prompt, prompt


def is_valid_rewritten_question(text: str, resolved_intent: Optional[str] = None) -> bool:
    q = normalize_whitespace(text)
    if not q or "\n" in q:
        return False

    # LLM이 답변/설명/JSON을 출력하면 rewrite 결과로 쓰지 않는다.
    blocked_prefixes = ("{", "[", "답변", "설명", "요약:", "재작성:")
    if q.startswith(blocked_prefixes):
        return False

    intent_conflict_keywords = CONVERSATION_POLICY.get("intent_conflict_keywords", {})
    conflict_keywords = intent_conflict_keywords.get(resolved_intent, []) if resolved_intent else []
    if any(str(keyword) in q for keyword in conflict_keywords if str(keyword).strip()):
        return False

    # 의도가 이미 정해진 경우에는 intent_registry 패턴 중 하나가 남아 있는지 확인한다.
    # 의도가 default이면 LLM rewrite 결과를 너무 강하게 버리지 않는다.
    if resolved_intent and resolved_intent != "default":
        patterns = INTENT_PATTERNS.get(resolved_intent, [])
        return any(str(pattern) in q for pattern in patterns if str(pattern).strip())

    return True


def rewrite_question(
    question: str,
    chat_history: List[Dict[str, str]],
    config: ChatConfig,
    resolved_system_id: Optional[str] = None,
    resolved_intent: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """사용자 질문을 검색 가능한 표준 질문으로 재작성한다.

    처리 순서:
    1. 최소 정리: 공백만 정리
    2. dictionary: 업무 용어/시스템 alias 표준화
    3. LLM rewrite: 오타/구어체/의도 표현 보정
    4. 실패 시 dictionary 결과로 fallback

    typo_normalization.json 같은 오타 사전은 더 이상 주 처리 경로에 사용하지 않는다.
    """
    debug_logs: List[str] = []
    raw_question = normalize_whitespace(question)
    dictionary_question = apply_dictionary_rewrite(raw_question)
    effective_intent = resolved_intent or "default"
    resolved_system_name = SYSTEM_NAME_BY_ID.get(resolved_system_id, "")

    debug_logs.append(f"[REWRITE 1] original_question = {question}")
    debug_logs.append(f"[REWRITE 2] whitespace_normalized = {raw_question}")
    debug_logs.append(f"[REWRITE 3] dictionary_standardized = {dictionary_question}")
    debug_logs.append(f"[REWRITE 4] resolved_intent_hint = {effective_intent}")

    system_prompt, prompt = build_rewrite_prompt(
        dictionary_question,
        chat_history,
        resolved_system_name,
        effective_intent,
    )
    debug_logs.append("[REWRITE 5] llm_rewrite = started")

    try:
        rewrite_engine = "requests"
        try:
            rewritten = _langchain_generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                config=config,
            )
            rewrite_engine = "langchain"
        except Exception as chain_exc:
            debug_logs.append(
                f"[REWRITE 5-1] langchain_rewrite = fallback_to_requests "
                f"({type(chain_exc).__name__}: {chain_exc})"
            )
            rewritten = ollama_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=config,
            )

        final_rewritten = normalize_whitespace(rewritten or dictionary_question)
        if not is_valid_rewritten_question(final_rewritten, effective_intent):
            debug_logs.append(f"[REWRITE 6] llm_rewrite_engine = {rewrite_engine}")
            debug_logs.append(f"[REWRITE 6] llm_rewrite = invalid_output ({final_rewritten})")
            debug_logs.append(f"[REWRITE 7] fallback_rewritten_question = {dictionary_question}")
            return dictionary_question, debug_logs

        debug_logs.append(f"[REWRITE 6] llm_rewrite_engine = {rewrite_engine}")
        debug_logs.append(f"[REWRITE 6] final_rewritten_question = {final_rewritten}")
        return final_rewritten, debug_logs
    except Exception as e:
        debug_logs.append(f"[REWRITE 6] llm_rewrite = failed ({type(e).__name__}: {e})")
        debug_logs.append(f"[REWRITE 7] fallback_rewritten_question = {dictionary_question}")
        return dictionary_question, debug_logs
