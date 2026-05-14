from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

# LangChain은 운영 환경에 설치되어 있으면 우선 사용하고,
# 없거나 오류가 발생하면 기존 requests 기반 Ollama 호출로 자동 fallback한다.
# 현재 requirements 조합(langchain==0.3.x, langchain-community==0.3.x)에 맞춘 import다.
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableBranch
except Exception:  # pragma: no cover - optional dependency
    PromptTemplate = None
    StrOutputParser = None
    JsonOutputParser = None
    RunnableLambda = None
    RunnableBranch = None

try:
    from langchain_community.llms import Ollama as LangChainOllama
except Exception:  # pragma: no cover - optional dependency
    LangChainOllama = None

try:
    from batch_dev.request_classifier import detect_structured_request_type
except Exception:
    detect_structured_request_type = None

if load_dotenv is not None:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CONF_DIR = Path(os.getenv("CONF_DIR", str(BASE_DIR / "conf")))

SYSTEM_REGISTRY_PATH = Path(os.getenv("SYSTEM_REGISTRY_PATH", str(CONF_DIR / "system_registry.json")))
QUESTION_DICTIONARY_PATH = Path(os.getenv("QUESTION_DICTIONARY_PATH", str(CONF_DIR / "question_dictionary.json")))
TYPO_NORMALIZATION_PATH = Path(os.getenv("TYPO_NORMALIZATION_PATH", str(CONF_DIR / "typo_normalization.json")))
INTENT_REGISTRY_PATH = Path(os.getenv("INTENT_REGISTRY_PATH", str(CONF_DIR / "intent_registry.json")))
PROMPT_TEMPLATE_PATH = Path(os.getenv("PROMPT_TEMPLATE_PATH", str(CONF_DIR / "prompt_templates.json")))
FEW_SHOT_PATH = Path(os.getenv("FEW_SHOT_PATH", str(CONF_DIR / "few_shot_examples.json")))
CONVERSATION_POLICY_PATH = Path(os.getenv("CONVERSATION_POLICY_PATH", str(CONF_DIR / "conversation_policy.json")))


def _load_required_json_file(path: Path, config_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{config_name} 설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_system_specs() -> List[Dict[str, Any]]:
    data = _load_required_json_file(SYSTEM_REGISTRY_PATH, "system_registry")
    if not isinstance(data, list) or not data:
        raise ValueError("system_registry.json 형식이 올바르지 않습니다. 비어 있지 않은 list여야 합니다.")
    return data


def load_question_replacements() -> Dict[str, str]:
    data = _load_required_json_file(QUESTION_DICTIONARY_PATH, "question_dictionary")
    if not isinstance(data, dict):
        raise ValueError("question_dictionary.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return {str(k): str(v) for k, v in data.items()}


def load_typo_normalization() -> Dict[str, str]:
    """Deprecated: typo_normalization.json은 더 이상 필수 설정이 아니다.

    실무 기준으로 오타/띄어쓰기/구어체 보정은 LLM rewrite가 담당한다.
    이 함수는 기존 환경 호환을 위해 파일이 있으면 읽되, 없으면 빈 dict를 반환한다.
    """
    if not TYPO_NORMALIZATION_PATH.exists():
        return {}
    data = _load_required_json_file(TYPO_NORMALIZATION_PATH, "typo_normalization")
    if not isinstance(data, dict):
        raise ValueError("typo_normalization.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return {str(k): str(v) for k, v in data.items()}


def load_intent_patterns() -> Dict[str, List[str]]:
    """conf/intent_registry.json에서 intent별 매칭 패턴을 읽는다.

    지원 형식:
    1) {"overview": {"patterns": ["개요"]}}
    2) {"overview": ["개요"]}
    """
    data = _load_required_json_file(INTENT_REGISTRY_PATH, "intent_registry")
    if not isinstance(data, dict) or not data:
        raise ValueError("intent_registry.json 형식이 올바르지 않습니다. 비어 있지 않은 dict여야 합니다.")

    loaded: Dict[str, List[str]] = {}
    for intent, value in data.items():
        if isinstance(value, dict):
            patterns = value.get("patterns", [])
        elif isinstance(value, list):
            patterns = value
        else:
            patterns = []
        loaded[str(intent)] = [str(item) for item in patterns if str(item).strip()]

    if not any(loaded.values()):
        raise ValueError("intent_registry.json에 사용할 수 있는 patterns가 없습니다.")
    return loaded


def load_prompt_templates() -> Dict[str, str]:
    """conf/prompt_templates.json에서 intent별 system prompt를 읽는다."""
    data = _load_required_json_file(PROMPT_TEMPLATE_PATH, "prompt_templates")
    if not isinstance(data, dict) or not data:
        raise ValueError("prompt_templates.json 형식이 올바르지 않습니다. 비어 있지 않은 dict여야 합니다.")

    loaded = {str(k): str(v) for k, v in data.items() if str(v).strip()}
    if "default" not in loaded:
        raise ValueError("prompt_templates.json에는 반드시 default 프롬프트가 있어야 합니다.")
    return loaded


def load_few_shot_examples() -> List[Dict[str, str]]:
    """conf/few_shot_examples.json에서 질문 재작성 few-shot 예시를 읽는다."""
    data = _load_required_json_file(FEW_SHOT_PATH, "few_shot_examples")
    if not isinstance(data, list):
        raise ValueError("few_shot_examples.json 형식이 올바르지 않습니다. list여야 합니다.")

    loaded: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user and assistant:
            loaded.append({"user": user, "assistant": assistant})
    return loaded

def load_conversation_policy() -> Dict[str, Any]:
    """conf/conversation_policy.json에서 대화 정책을 읽는다.

    운영 설정으로 관리할 항목:
    - followup_signals
    - intent_conflict_keywords
    - system_required_intents
    - out_of_scope_message
    - missing_system_message
    """
    data = _load_required_json_file(CONVERSATION_POLICY_PATH, "conversation_policy")
    if not isinstance(data, dict):
        raise ValueError("conversation_policy.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return data


SYSTEM_SPECS = load_system_specs()
QUESTION_REPLACEMENTS = load_question_replacements()
TYPO_NORMALIZATION = load_typo_normalization()
INTENT_PATTERNS = load_intent_patterns()
SYSTEM_PROMPT_BY_INTENT = load_prompt_templates()
FEW_SHOT_EXAMPLES = load_few_shot_examples()
CONVERSATION_POLICY = load_conversation_policy()
SYSTEM_NAME_BY_ID: Dict[str, str] = {spec["system_id"]: spec["canonical_name"] for spec in SYSTEM_SPECS}


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_input_typos(text: str) -> str:
    """Deprecated compatibility wrapper.

    오타 치환을 코드/JSON으로 관리하지 않는다.
    여기서는 최소 정리인 공백 정리만 수행하고, 실제 오타/구어체 보정은 LLM rewrite에서 처리한다.
    """
    return normalize_whitespace(text)


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


def history_to_text(chat_history: List[Dict[str, str]], max_turns: int = 4) -> str:
    recent = chat_history[-max_turns:]
    return "\n".join([f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent]).strip()


def is_followup_question(question: str) -> bool:
    """이전 답변/질문을 이어서 말하는 짧은 후속 질문인지 판단한다.

    followup 신호는 코드에 박지 않고 conf/conversation_policy.json에서 관리한다.
    """
    q = normalize_whitespace(question)
    if not q:
        return False

    followup_signals = CONVERSATION_POLICY.get("followup_signals", [])
    return any(str(signal) in q for signal in followup_signals if str(signal).strip())


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
    system_name = SYSTEM_NAME_BY_ID.get(resolved_system_id, "")
    templates = {
        "overview": "{system_name} 소득공제 업무 개요를 알려줘",
        "batch_process": "{system_name} 소득공제 배치 프로세스를 설명해줘",
        "batch_flow": "{system_name} 소득공제 배치 흐름도를 그려줘",
        "table_lineage": "{system_name} 소득공제 테이블 리니지를 보여줘",
        "billing_monthly_amount": "청구 이용내역서 월별 금액을 조회해서 그래프로 보여줘",
        "today_incidents": "오늘 장애현황 알려줘",
        "batch_development": question,
    }

    system_required_intents = set(CONVERSATION_POLICY.get("system_required_intents", []))
    if intent in system_required_intents:
        if system_name:
            return templates[intent].format(system_name=system_name)
        # 시스템명이 없으면 {system_name} 템플릿을 절대 반환하지 않는다.
        return question

    if intent in {"billing_monthly_amount", "today_incidents"}:
        return templates[intent]

    if intent == "batch_development":
        return question

    return question


@dataclass
class ChatConfig:
    """Upstage Chat 설정.

    기존 app.py 영향 최소화를 위해 ChatConfig 이름은 유지한다.
    .env에서 UPSTAGE_API_KEY, UPSTAGE_BASE_URL, UPSTAGE_CHAT_MODEL을 읽는다.
    """
    api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    model: str = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro3")
    timeout: int = int(os.getenv("UPSTAGE_CHAT_TIMEOUT", "120"))
    temperature: float = float(os.getenv("UPSTAGE_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("UPSTAGE_MAX_TOKENS", "2048"))


@dataclass
class EmbedConfig:
    """Upstage Embedding 설정.

    ChromaDB는 그대로 사용하고, query_texts 임베딩만 Upstage로 생성한다.
    기존 Ollama 임베딩으로 만든 컬렉션은 차원/모델이 다를 수 있으므로
    반드시 Upstage 임베딩으로 reset 후 재적재해야 한다.
    """
    api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    model: str = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large-query")
    timeout: int = int(os.getenv("UPSTAGE_EMBED_TIMEOUT", "60"))


class UpstageEmbeddingFunction:
    """ChromaDB에서 사용할 Upstage Embedding Function."""

    def __init__(self, config: EmbedConfig) -> None:
        self.config = config

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not self.config.api_key:
            raise ValueError("UPSTAGE_API_KEY가 비어 있습니다. .env에 UPSTAGE_API_KEY를 설정하세요.")

        resp = requests.post(
            f"{self.config.base_url.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "input": input,
            },
            timeout=self.config.timeout,
        )
        resp.raise_for_status()

        data = resp.json().get("data", [])
        vectors = [item.get("embedding") for item in data]

        if not vectors or any(not vector for vector in vectors):
            raise ValueError("Upstage Embedding 응답에 embedding 값이 없습니다.")

        return vectors


# 기존 ingest 코드가 OllamaEmbeddingFunction 이름을 import하고 있을 가능성을 고려한 호환 alias.
# 새 코드에서는 UpstageEmbeddingFunction 사용을 권장한다.
OllamaEmbeddingFunction = UpstageEmbeddingFunction


def _use_langchain() -> bool:
    """기존 설정 호환용 함수.

    Upstage는 OpenAI 호환 REST API를 직접 호출하므로 LangChain이 없어도 동작한다.
    기존 debug flag 구조를 깨지 않기 위해 함수명은 유지한다.
    """
    return os.getenv("LANGCHAIN_ENABLED", "false").lower() not in {"0", "false", "no", "n"}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "n"}


def get_langchain_feature_flags() -> Dict[str, bool]:
    """운영 중 기능을 켜고 끌 수 있는 확장 옵션."""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "upstage"),
        "langchain_enabled": _use_langchain(),
        "router_enabled": _env_flag("LANGCHAIN_ROUTER_ENABLED", "true"),
        "structured_parser_enabled": _env_flag("LANGCHAIN_STRUCTURED_PARSER_ENABLED", "true"),
        "retrieval_compression_enabled": _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false"),
    }


def _upstage_generate_requests(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    if not config.api_key:
        raise ValueError("UPSTAGE_API_KEY가 비어 있습니다. .env에 UPSTAGE_API_KEY를 설정하세요.")

    resp = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": False,
        },
        timeout=config.timeout,
    )
    resp.raise_for_status()

    payload = resp.json()
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("Upstage Chat 응답에 choices 값이 없습니다.")

    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


def ollama_generate(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    """답변 생성 진입점.

    기존 호출부가 ollama_generate를 사용하고 있어서 함수명은 유지하되,
    내부 구현은 Upstage Chat API 호출로 변경한다.
    """
    return _upstage_generate_requests(prompt=prompt, system_prompt=system_prompt, config=config)


# 새 이름도 함께 제공한다.
upstage_generate = ollama_generate


def get_llm_engine_name() -> str:
    return f"requests_upstage:{os.getenv('UPSTAGE_CHAT_MODEL', 'solar-pro3')}"


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
        rewritten = ollama_generate(prompt=prompt, system_prompt=system_prompt, config=config)
        final_rewritten = normalize_whitespace(rewritten or dictionary_question)
        if not is_valid_rewritten_question(final_rewritten, effective_intent):
            debug_logs.append(f"[REWRITE 6] llm_rewrite = invalid_output ({final_rewritten})")
            debug_logs.append(f"[REWRITE 7] fallback_rewritten_question = {dictionary_question}")
            return dictionary_question, debug_logs

        debug_logs.append(f"[REWRITE 6] final_rewritten_question = {final_rewritten}")
        return final_rewritten, debug_logs
    except Exception as e:
        debug_logs.append(f"[REWRITE 6] llm_rewrite = failed ({type(e).__name__}: {e})")
        debug_logs.append(f"[REWRITE 7] fallback_rewritten_question = {dictionary_question}")
        return dictionary_question, debug_logs


def get_system_by_id(payload: Dict[str, Any], system_id: str) -> Optional[Dict[str, Any]]:
    for domain in payload.get("domains", []):
        for system in domain.get("systems", []):
            if system.get("system_id") == system_id:
                return system
    return None


def get_realtime_query(payload: Dict[str, Any], query_id: str) -> Optional[Dict[str, Any]]:
    for item in payload.get("realtime_queries", []):
        if item.get("query_id") == query_id:
            return item
    return None



def _tokenize_for_score(text: str) -> List[str]:
    text = normalize_whitespace(text).lower()
    return [token for token in re.split(r"[^0-9a-zA-Z가-힣_]+", text) if len(token) >= 2]


def compress_search_result(
    search_result: Dict[str, Any],
    query: str,
    intent: str,
    top_k: int = 4,
) -> Dict[str, Any]:
    """검색 결과를 가볍게 재정렬/압축한다.

    LangChain의 Document Compressor/Rerank 개념을 현재 구조에 안전하게 얇게 붙인 것이다.
    - 기본은 OFF: LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED=true 일 때만 적용
    - 외부 reranker 모델 없이 동작하므로 CPU 환경에서도 안전하다.
    - where_filter 결과를 벗어나지 않고, 기존 Chroma 결과 안에서만 재정렬한다.
    """
    if not _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false"):
        return search_result

    docs = list(search_result.get("documents", [[]])[0] or [])
    metas = list(search_result.get("metadatas", [[]])[0] or [])
    distances = list(search_result.get("distances", [[]])[0] or [])
    ids = list(search_result.get("ids", [[]])[0] or [])

    if not docs:
        return search_result

    query_tokens = set(_tokenize_for_score(query))

    def score_item(item: Tuple[int, str, Dict[str, Any]]) -> Tuple[float, int]:
        idx, doc, meta = item
        doc_tokens = set(_tokenize_for_score(doc))
        overlap = len(query_tokens & doc_tokens)
        section_bonus = 3 if meta.get("section") == intent else 0
        title_bonus = 1 if any(token in str(meta.get("title", "")).lower() for token in query_tokens) else 0
        return (float(overlap + section_bonus + title_bonus), -idx)

    ranked = sorted(enumerate(zip(docs, metas)), key=lambda pair: score_item((pair[0], pair[1][0], pair[1][1])), reverse=True)
    keep_indexes = [idx for idx, _ in ranked[: max(1, top_k)]]

    def pick(values: List[Any]) -> List[Any]:
        return [values[i] for i in keep_indexes if i < len(values)]

    compressed = dict(search_result)
    compressed["documents"] = [pick(docs)]
    compressed["metadatas"] = [pick(metas)]
    if distances:
        compressed["distances"] = [pick(distances)]
    if ids:
        compressed["ids"] = [pick(ids)]
    return compressed


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """LLM 출력이 JSON이어야 하는 영역에서 사용할 수 있는 안전 parser.

    현재 일반 답변에는 적용하지 않는다. 배치 개발 요청 파싱처럼 JSON이 필요한 기능을
    추가할 때 재사용하도록 분리했다.
    """
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


@dataclass(frozen=True)
class ResponseRoute:
    name: str
    render_type: str
    realtime_mode: Optional[str] = None


def resolve_response_route(intent: str, render_type: str, has_graph: bool, has_query_meta: bool) -> ResponseRoute:
    """응답 라우팅을 intent별 if문 대신 공통 규칙으로 결정한다.

    새 intent를 추가할 때는 conf/intent_registry.json과 원본 JSON 메타를 먼저 확장하고,
    이 함수는 render_type 중심으로 최소한만 분기한다.
    """
    if render_type == "graph" and has_graph:
        return ResponseRoute(name="graph", render_type="graph")
    if render_type == "chart" and has_query_meta:
        return ResponseRoute(name="chart", render_type="chart", realtime_mode="chart_only")
    if render_type == "table" and has_query_meta:
        return ResponseRoute(name="table", render_type="table", realtime_mode="incident_table_with_summary")
    return ResponseRoute(name="llm_text", render_type=render_type)

def retrieve_docs(
    persist_dir: str,
    collection_name: str,
    query: str,
    top_k: int = 4,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=UpstageEmbeddingFunction(EmbedConfig()),
    )
    kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": top_k}
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def build_answer_prompt(
    rewritten_question: str,
    intent: str,
    search_result: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    system_id: Optional[str] = None,
) -> Tuple[str, str]:
    base_system_prompt = SYSTEM_PROMPT_BY_INTENT.get(
        intent,
        SYSTEM_PROMPT_BY_INTENT["default"],
    )
    system_guard = ""
    if system_id:
        system_name = SYSTEM_NAME_BY_ID.get(system_id, system_id)
        system_guard = f" 반드시 {system_name} 시스템 정보만 사용하고, 다른 시스템 정보는 절대 섞지 마라."
    system_prompt = base_system_prompt + system_guard
    documents = search_result.get("documents", [[]])[0][:3]
    metadatas = search_result.get("metadatas", [[]])[0][:3]

    context_lines: List[str] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        context_lines.append(
            f"[문서 {idx}] system={meta.get('system_name', '')} section={meta.get('section', '')} title={meta.get('title', '')}\n{doc}"
        )

    history_text = history_to_text(chat_history)
    prompt = f"""
        이전 대화:
        {history_text if history_text else '(없음)'}

        사용자 질문:
        {rewritten_question}

        검색 문맥:
        {chr(10).join(context_lines) if context_lines else '(없음)'}

        답변 규칙:
        - 반드시 한국어로만 답변한다.
        - 검색 문맥에 있는 내용만 사용한다.
        - system_id가 주어진 경우 해당 시스템 정보만 사용하고, 다른 시스템 이름/내용은 절대 포함하지 않는다.
        - overview 질문이면 핵심 요약 2~4문장으로 정리한다.
        - batch_process 질문이면 한 줄 흐름, STEP별 배치, 핵심 배치 순서로만 답변한다.
        - batch_process 답변에서 같은 STEP/배치 목록을 문장형 설명과 STEP 상세로 두 번 반복하지 않는다.
        - batch_process 답변에서 배치명은 검색 문맥에 있는 job_id만 사용한다.
        - 문맥이 부족하면 부족하다고 말한다.
        """.strip()
    return system_prompt, prompt


def build_graph_answer(graph_data: Dict[str, Any], intent: str) -> str:
    summary = graph_data.get("summary")
    if summary:
        return summary
    if intent == "batch_flow":
        return "배치 흐름의 시작, 핵심 처리, 종료 단계를 아래 흐름도에서 확인할 수 있습니다."
    return "원천 테이블부터 결과 테이블까지의 데이터 흐름을 아래 리니지에서 확인할 수 있습니다."


def build_chart_answer(query_meta: Dict[str, Any]) -> str:
    return f"{query_meta.get('title', '조회 결과')}를 시각화했습니다. 월별 추이와 분포를 바로 확인할 수 있습니다."


def build_table_answer(query_meta: Dict[str, Any]) -> str:
    return f"{query_meta.get('title', '조회 결과')}을 표 형태로 정리했습니다. 아래에서 세부 항목을 확인할 수 있습니다."


def build_overview_fallback(overview: Dict[str, Any]) -> str:
    """overview는 화면에서 구조화 렌더링하므로 answer에는 짧은 대표 문장만 담는다."""
    return str(
        overview.get("summary")
        or overview.get("content")
        or overview.get("title")
        or "업무 개요 정보가 없습니다."
    ).strip()


def _format_execution_label(execution: str) -> str:
    """배치 실행 방식을 사용자에게 보여줄 한글 라벨로 변환한다.

    실행 방식 값은 JSON 메타에서 오므로 새 값이 추가되어도 원문을 보존한다.
    """
    labels = {
        "parallel": "병렬",
        "sequential": "순차",
    }
    return labels.get(str(execution or "").strip().lower(), str(execution or "").strip())


def _format_list_inline(values: Any) -> str:
    """list/string 값을 화면용 한 줄 문자열로 변환한다."""
    if values is None:
        return ""
    if isinstance(values, list):
        return ", ".join([str(v).strip() for v in values if str(v).strip()])
    return str(values).strip()


def _format_duration_sec(value: Any) -> str:
    """초 단위 평균 수행시간을 보기 좋게 표시한다."""
    if value is None or str(value).strip() == "":
        return ""
    try:
        seconds = int(float(value))
    except Exception:
        return str(value).strip()

    if seconds < 60:
        return f"{seconds}초"

    minutes = seconds // 60
    remain = seconds % 60
    if remain:
        return f"{minutes}분 {remain}초"
    return f"{minutes}분"


def build_batch_process_fallback(batch_process: Dict[str, Any]) -> str:
    """배치 프로세스를 중복 없이 구조화해서 생성한다.

    원칙:
    - 하드코딩된 배치명/단계명 없이 JSON steps/jobs/key_jobs 기반으로 생성한다.
    - 같은 내용을 문장형 설명과 STEP 상세로 두 번 반복하지 않는다.
    - 한 줄 흐름 -> STEP 상세 -> 핵심 배치 순서로만 출력한다.
    - batch_process.jobs[]에 운영 메타가 있으면 함께 출력한다.
    - LangChain 실패 시 fallback으로 사용해도 그대로 사용자에게 보여줄 수 있는 품질을 유지한다.
    """
    title = batch_process.get("title", "배치 프로세스")
    steps = batch_process.get("steps", [])
    if not steps:
        return str(title or "배치 프로세스 정보가 없습니다.")

    lines: List[str] = [f"📌 {title}", ""]

    step_names = [str(step.get("name", "")).strip() for step in steps if str(step.get("name", "")).strip()]
    if step_names:
        lines.append("🔹 한 줄 흐름")
        lines.append(" → ".join(step_names))
        lines.append("")

    lines.append("🔹 단계별 배치 프로세스")
    for step in steps:
        step_no = step.get("step", "")
        step_name = str(step.get("name", "")).strip()
        execution_label = _format_execution_label(str(step.get("execution", "")))
        description = str(step.get("description", "")).strip()

        header_parts = [f"STEP {step_no}" if step_no != "" else "STEP", step_name]
        header = ". ".join([part for part in header_parts if part])
        if execution_label:
            header = f"{header} ({execution_label})"
        lines.append(f"{header}")

        if description:
            lines.append(f"👉 {description}")

        for job in step.get("jobs", []) or []:
            job_id = str(job.get("job_id", "")).strip()
            job_desc = str(job.get("description", "")).strip()
            job_name = str(job.get("job_name", "")).strip()

            display_name = job_name if job_name and job_name != job_id else ""
            if job_id and job_desc:
                if display_name:
                    lines.append(f"- {job_id} ({display_name}): {job_desc}")
                else:
                    lines.append(f"- {job_id}: {job_desc}")
            elif job_id:
                lines.append(f"- {job_id}" + (f" ({display_name})" if display_name else ""))
            elif job_desc:
                lines.append(f"- {job_desc}")

            operation_lines: List[str] = []
            schedule_type = str(job.get("schedule_type", "")).strip()
            execution_time = str(job.get("execution_time", "")).strip()
            avg_duration = _format_duration_sec(job.get("avg_duration_sec"))
            batch_file = str(job.get("batch_file", "")).strip()
            program_name = str(job.get("program_name", "")).strip()
            owner_team = str(job.get("owner_team", "")).strip()
            retry_count = str(job.get("retry_count", "")).strip()
            upstream_jobs = _format_list_inline(job.get("upstream_jobs"))
            downstream_jobs = _format_list_inline(job.get("downstream_jobs"))
            failure_action = str(job.get("failure_action", "")).strip()

            if schedule_type:
                operation_lines.append(f"  - 실행주기: {schedule_type}")
            if execution_time:
                operation_lines.append(f"  - 실행시간: {execution_time}")
            if avg_duration:
                operation_lines.append(f"  - 평균수행시간: {avg_duration}")
            if batch_file:
                operation_lines.append(f"  - 실행배치파일: {batch_file}")
            if program_name:
                operation_lines.append(f"  - 프로그램명: {program_name}")
            if owner_team:
                operation_lines.append(f"  - 담당팀: {owner_team}")
            if retry_count:
                operation_lines.append(f"  - 재시도횟수: {retry_count}")
            if upstream_jobs:
                operation_lines.append(f"  - 선행배치: {upstream_jobs}")
            if downstream_jobs:
                operation_lines.append(f"  - 후행배치: {downstream_jobs}")
            if failure_action:
                operation_lines.append(f"  - 장애조치방법: {failure_action}")

            lines.extend(operation_lines)

    key_jobs: List[str] = []
    for step in steps:
        for job_id in step.get("key_jobs", []) or []:
            job_id_str = str(job_id).strip()
            if job_id_str and job_id_str not in key_jobs:
                key_jobs.append(job_id_str)

    if key_jobs:
        lines.append("⭐ 핵심 배치")
        for job_id in key_jobs:
            lines.append(f"- {job_id}")

    return "".join(lines).strip()


def remove_repeated_step_sections(answer: str) -> str:
    """LLM이 문장형 단계 설명 뒤에 STEP 상세를 반복 출력한 경우 앞부분을 제거한다.

    하드코딩된 업무/배치명 기준이 아니라 STEP 패턴 반복 여부만 본다.
    이미 깔끔한 답변이면 원문을 그대로 반환한다.
    """
    text = str(answer or "").strip()
    if not text:
        return text

    # 'STEP 1.' 형태 상세 구간이 있으면 그 앞의 장황한 '1단계에서는...' 문장형 반복을 제거한다.
    step_match = re.search(r"(?im)^\s*STEP\s*1\s*[\.).]", text)
    if not step_match:
        return text

    prefix = text[: step_match.start()].strip()
    suffix = text[step_match.start() :].strip()

    # 앞부분에 단계형 문장과 배치명이 이미 있고, 뒤에 STEP 상세가 다시 있으면 중복으로 판단한다.
    prefix_has_stage_text = bool(re.search(r"[123]\s*단계|Step\s*[123]", prefix, flags=re.IGNORECASE))
    prefix_has_batch_id = bool(re.search(r"BATCH_\d+_", prefix))
    suffix_has_batch_id = bool(re.search(r"BATCH_\d+_", suffix))

    if prefix_has_stage_text and prefix_has_batch_id and suffix_has_batch_id:
        # 제목/핵심 배치/핵심 흐름 같은 짧은 헤더는 유지하고, 긴 문장형 단계 설명만 제거한다.
        header_lines: List[str] = []
        for line in prefix.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"[123]\s*단계|Step\s*[123]", stripped, flags=re.IGNORECASE):
                break
            header_lines.append(stripped)
        if header_lines:
            return "\n".join(header_lines + ["", suffix]).strip()
        return suffix

    return text


@dataclass
class AgentResult:
    original_question: str
    normalized_question: str
    rewritten_question: str
    system_id: Optional[str]
    intent: str
    answer: str
    render_type: str = "text"
    graph_data: Optional[Dict[str, Any]] = None
    query_meta: Optional[Dict[str, Any]] = None
    realtime_mode: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    debug_logs: List[str] = field(default_factory=list)


class HandoverAgent:
    def __init__(
        self,
        json_path: str,
        persist_dir: str = "./chroma",
        collection_name: str = "handover_agent",
        chat_config: Optional[ChatConfig] = None,
    ) -> None:
        self.payload = load_json(json_path)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chat_config = chat_config or ChatConfig()

    def _build_structured_payload(
        self,
        system_id: Optional[str],
        intent: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if system_id and intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
            system = get_system_by_id(self.payload, system_id)
            if not system:
                return "text", None, None, None
            if intent in {"overview", "batch_process"}:
                return "text", None, None, system.get(intent)
            return "graph", system.get(intent), None, None

        if intent == "billing_monthly_amount":
            return "chart", None, get_realtime_query(self.payload, "billing_monthly_amount"), None

        if intent == "today_incidents":
            return "table", None, get_realtime_query(self.payload, "today_incidents"), None

        return "text", None, None, None

    def _build_where_filter(self, system_id: Optional[str], intent: str) -> Optional[Dict[str, Any]]:
        if intent in {"overview", "batch_process", "batch_flow", "table_lineage"} and system_id:
            return {"$and": [{"system_id": system_id}, {"section": intent}]}

        if intent == "billing_monthly_amount":
            return {"$and": [{"section": "realtime_query"}, {"query_id": "billing_monthly_amount"}]}

        if intent == "today_incidents":
            return {"$and": [{"section": "realtime_query"}, {"query_id": "today_incidents"}]}

        return None

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
    ) -> AgentResult:
        chat_history = chat_history or []
        debug_logs: List[str] = []
        debug_logs.append(f"[LC 1] feature_flags = {get_langchain_feature_flags()}")

        # 실무 질의 처리 순서:
        # 1) 최소 정리: 공백만 정리
        # 2) dictionary: 업무 용어/시스템 alias 표준화
        # 3) LLM rewrite: 오타/구어체/의도 표현 보정
        # 4) rewrite 결과 기준 intent/system 재판별
        # 5) 검색
        raw_question = normalize_whitespace(question)
        normalized_question = apply_dictionary_rewrite(raw_question)
        debug_logs.append(f"[PREP 1] whitespace_normalized = {raw_question}")
        debug_logs.append(f"[PREP 2] dictionary_standardized = {normalized_question}")

        # rewrite 전 intent는 잠정 후보로만 기록한다.
        # 오타가 포함된 현재 질문에서 잡힌 약한 intent를 LLM rewrite에 고정하지 않는다.
        # 실제 라우팅 intent는 rewrite 이후 질문으로 다시 판별한다.
        intent_hint_before_rewrite = detect_intent(normalized_question)
        intent_hint_is_confident = is_confident_intent_hint(normalized_question, intent_hint_before_rewrite)
        intent_fallback_allowed = intent_hint_is_confident
        rewrite_intent_hint = "default"
        debug_logs.append(f"[PREP 3] raw_intent_hint_before_rewrite = {intent_hint_before_rewrite}")
        debug_logs.append(f"[PREP 3-0] intent_hint_is_confident = {intent_hint_is_confident}")
        debug_logs.append("[PREP 3-1] rewrite_intent_hint = default (intent is decided after rewrite)")

        direct_system_id = detect_system_id(normalized_question)
        if direct_system_id:
            system_id_hint = direct_system_id
            debug_logs.append("[PREP 3-2] system_id_hint_source = current_question")
        elif chat_history and (is_followup_question(normalized_question) or intent_hint_before_rewrite != "default"):
            # 시스템명은 현재 질문에 없거나 오타가 심할 수 있으므로 history 보조 사용을 허용한다.
            # 단, intent는 여기서 history로 끌어오지 않고 rewrite 이후 default일 때만 보조 적용한다.
            system_id_hint = detect_system_id_with_history(normalized_question, chat_history)
            debug_logs.append("[PREP 3-2] system_id_hint_source = history_context")
        else:
            system_id_hint = None
            debug_logs.append("[PREP 3-2] system_id_hint_source = none")

        debug_logs.append(f"[PREP 4] resolved_system_id_hint_before_rewrite = {system_id_hint}")
        debug_logs.append(f"[PREP 5] resolved_intent_hint_before_rewrite = {rewrite_intent_hint}")

        rewritten_question, rewrite_logs = rewrite_question(
            question=normalized_question,
            chat_history=chat_history,
            config=self.chat_config,
            resolved_system_id=system_id_hint,
            resolved_intent=rewrite_intent_hint,
        )
        debug_logs.extend(rewrite_logs)

        rewritten_standardized = apply_dictionary_rewrite(rewritten_question)
        if rewritten_standardized != rewritten_question:
            debug_logs.append(f"[REWRITE 8] post_dictionary_standardized = {rewritten_standardized}")
            rewritten_question = rewritten_standardized

        intent = detect_intent(rewritten_question)

        if intent == "default" and chat_history and is_followup_question(rewritten_question):
            previous_intent = detect_previous_user_intent(chat_history)
            if previous_intent != "default":
                intent = previous_intent
                debug_logs.append(f"[STEP 5-1] intent_fallback_to_history_followup = {previous_intent}")

        if intent == "default" and intent_fallback_allowed and intent_hint_before_rewrite != "default":
            intent = intent_hint_before_rewrite
            debug_logs.append(f"[STEP 5-2] intent_fallback_to_confident_current_hint = {intent_hint_before_rewrite}")

        system_id = detect_system_id(rewritten_question) or system_id_hint
        if not system_id and (intent != "default" or is_followup_question(rewritten_question)):
            system_id = detect_system_id_with_history(rewritten_question, chat_history)

        debug_logs.append(f"[STEP 5] detected_system_id = {system_id}")
        debug_logs.append(f"[STEP 6] detected_intent = {intent}")

        if intent == "default" and not system_id:
            debug_logs.append("[STEP 6-1] out_of_scope_detected = True")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=None,
                intent="out_of_scope",
                answer=str(CONVERSATION_POLICY.get(
                    "out_of_scope_message",
                    "현재 에이전트의 지원 범위를 벗어난 질문입니다."
                )),
                render_type="text",
                debug_logs=debug_logs,
            )

        system_required_intents = set(CONVERSATION_POLICY.get("system_required_intents", []))
        if intent in system_required_intents and not system_id:
            debug_logs.append("[STEP 6-2] missing_system_id = True")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=None,
                intent=intent,
                answer=str(CONVERSATION_POLICY.get(
                    "missing_system_message",
                    "어느 시스템 기준인지 확인할 수 없습니다. 시스템명을 포함해서 다시 질문해주세요."
                )),
                render_type="text",
                debug_logs=debug_logs,
            )

        render_type, graph_data, query_meta, structured_data = self._build_structured_payload(system_id, intent)
        where = self._build_where_filter(system_id, intent)
        debug_logs.append(f"[STEP 7] render_type = {render_type}")
        debug_logs.append(f"[STEP 8] where_filter = {where}")

        if intent == "overview" and system_id and structured_data:
            debug_logs.append("[STEP 9] overview_direct_render = true")
            debug_logs.append("[STEP 10] retrieval = skipped")
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_overview_renderer)")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=system_id,
                intent=intent,
                answer=build_overview_fallback(structured_data),
                render_type=render_type,
                graph_data=graph_data,
                query_meta=query_meta,
                realtime_mode=None,
                structured_data=structured_data,
                sources=[],
                debug_logs=debug_logs,
            )

        search_query = rewritten_question
        debug_logs.append(f"[STEP 9] search_query = {search_query}")

        search_result = retrieve_docs(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            query=search_query,
            top_k=top_k,
            where=where,
        )

        search_result = compress_search_result(
            search_result=search_result,
            query=search_query,
            intent=intent,
            top_k=top_k,
        )
        if _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false"):
            debug_logs.append("[STEP 9-1] retrieval_compression = enabled")
        else:
            debug_logs.append("[STEP 9-1] retrieval_compression = disabled")

        documents = search_result.get("documents", [[]])[0]
        metadatas = search_result.get("metadatas", [[]])[0]

        if not documents and system_id and intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
            debug_logs.append("[STEP 10] filtered_retrieval_empty = fallback_to_structured_payload")
            if intent == "overview" and structured_data:
                return AgentResult(
                    question,
                    normalized_question,
                    rewritten_question,
                    system_id,
                    intent,
                    build_overview_fallback(structured_data),
                    render_type,
                    graph_data,
                    query_meta,
                    None,
                    structured_data,
                    [],
                    debug_logs,
                )
            if intent == "batch_process" and structured_data:
                return AgentResult(
                    question,
                    normalized_question,
                    rewritten_question,
                    system_id,
                    intent,
                    build_batch_process_fallback(structured_data),
                    render_type,
                    graph_data,
                    query_meta,
                    None,
                    structured_data,
                    [],
                    debug_logs,
                )

        source_rows: List[Dict[str, Any]] = []
        debug_logs.append(f"[STEP 10] retrieved_doc_count = {len(documents)}")

        for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source_rows.append(
                {
                    "rank": rank,
                    "title": meta.get("title"),
                    "system_id": meta.get("system_id"),
                    "system_name": meta.get("system_name"),
                    "section": meta.get("section"),
                    "doc_level": meta.get("doc_level"),
                    "chunk_id": meta.get("chunk_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "step": meta.get("step"),
                    "job_id": meta.get("job_id"),
                    "preview": (doc[:300] + "...") if len(doc) > 300 else doc,
                }
            )

        route = resolve_response_route(
            intent=intent,
            render_type=render_type,
            has_graph=bool(graph_data),
            has_query_meta=bool(query_meta),
        )
        debug_logs.append(f"[STEP 11] response_route = {route.name}")

        if route.name == "graph" and graph_data:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_graph_answer(graph_data, intent),
                route.render_type,
                graph_data,
                None,
                None,
                None,
                source_rows,
                debug_logs,
            )

        if route.name == "chart" and query_meta:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_chart_answer(query_meta),
                route.render_type,
                None,
                query_meta,
                route.realtime_mode,
                None,
                source_rows,
                debug_logs,
            )

        if route.name == "table" and query_meta:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_table_answer(query_meta),
                route.render_type,
                None,
                query_meta,
                route.realtime_mode,
                None,
                source_rows,
                debug_logs,
            )

        if intent == "batch_process" and structured_data:
            answer = build_batch_process_fallback(structured_data)
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_batch_renderer)")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                answer,
                render_type,
                graph_data,
                query_meta,
                None,
                structured_data,
                source_rows,
                debug_logs,
            )

        system_prompt, prompt = build_answer_prompt(
            rewritten_question=rewritten_question,
            intent=intent,
            search_result=search_result,
            chat_history=chat_history,
            system_id=system_id,
        )

        try:
            debug_logs.append(f"[STEP 11] answer_generation_engine = {get_llm_engine_name()}")
            debug_logs.append("[STEP 12] answer_generation = started")
            answer = ollama_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.chat_config,
            )
            if intent == "batch_process":
                answer = remove_repeated_step_sections(answer)
                debug_logs.append("[STEP 12-1] duplicate_step_cleanup = applied")
            debug_logs.append("[STEP 13] answer_generation = success")
        except Exception as e:
            debug_logs.append(f"[STEP 13] answer_generation = failed ({type(e).__name__}: {e})")
            if intent == "overview" and structured_data:
                answer = build_overview_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = overview_structured_payload")
            elif intent == "batch_process" and structured_data:
                answer = build_batch_process_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = batch_process_structured_payload")
            elif documents:
                answer = documents[0]
                debug_logs.append("[STEP 14] fallback = first_retrieved_document")
            else:
                answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                debug_logs.append("[STEP 14] fallback = generic_error_message")

        return AgentResult(
            question,
            normalized_question,
            rewritten_question,
            system_id,
            intent,
            answer,
            render_type,
            graph_data,
            query_meta,
            None,
            structured_data,
            source_rows,
            debug_logs,
        )
