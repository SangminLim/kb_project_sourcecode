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
    from batch_dev.request_classifier import detect_structured_request_type
except Exception:
    detect_structured_request_type = None

BASE_DIR = Path(__file__).resolve().parent
CONF_DIR = Path(os.getenv("CONF_DIR", str(BASE_DIR / "conf")))

SYSTEM_REGISTRY_PATH = Path(os.getenv("SYSTEM_REGISTRY_PATH", str(CONF_DIR / "system_registry.json")))
QUESTION_DICTIONARY_PATH = Path(os.getenv("QUESTION_DICTIONARY_PATH", str(CONF_DIR / "question_dictionary.json")))
TYPO_NORMALIZATION_PATH = Path(os.getenv("TYPO_NORMALIZATION_PATH", str(CONF_DIR / "typo_normalization.json")))
INTENT_REGISTRY_PATH = Path(os.getenv("INTENT_REGISTRY_PATH", str(CONF_DIR / "intent_registry.json")))
PROMPT_TEMPLATE_PATH = Path(os.getenv("PROMPT_TEMPLATE_PATH", str(CONF_DIR / "prompt_templates.json")))
FEW_SHOT_PATH = Path(os.getenv("FEW_SHOT_PATH", str(CONF_DIR / "few_shot_examples.json")))


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

SYSTEM_SPECS = load_system_specs()
QUESTION_REPLACEMENTS = load_question_replacements()
TYPO_NORMALIZATION = load_typo_normalization()
INTENT_PATTERNS = load_intent_patterns()
SYSTEM_PROMPT_BY_INTENT = load_prompt_templates()
FEW_SHOT_EXAMPLES = load_few_shot_examples()
SYSTEM_NAME_BY_ID: Dict[str, str] = {spec["system_id"]: spec["canonical_name"] for spec in SYSTEM_SPECS}


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_input_typos(text: str) -> str:
    normalized = normalize_whitespace(text)
    for src, dst in sorted(TYPO_NORMALIZATION.items(), key=lambda x: len(x[0]), reverse=True):
        normalized = re.sub(re.escape(src), dst, normalized)
    return normalize_whitespace(normalized)


def replace_aliases(question: str) -> str:
    normalized = normalize_input_typos(question)
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


def detect_intent(question: str) -> str:
    q = normalize_whitespace(question)

    # 배치 요청서처럼 구조화된 입력은 업무 키워드가 아니라 request_schema.json 기준으로 판별한다.
    if detect_structured_request_type is not None:
        structured_type = detect_structured_request_type(q)
        if structured_type:
            return structured_type

    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return "default"


def history_to_text(chat_history: List[Dict[str, str]], max_turns: int = 4) -> str:
    recent = chat_history[-max_turns:]
    return "\n".join([f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent]).strip()


def is_followup_question(question: str) -> bool:
    """이전 답변/질문을 이어서 말하는 짧은 후속 질문인지 판단한다.

    특정 업무 문장을 하드코딩하지 않고, 대명사/재표현/출력형식 변경 신호만 본다.
    예: 다시 보여줘, 그거 표로, 테이블로 보여줘, 그래프로 그려줘 등
    """
    q = normalize_whitespace(question)
    if not q:
        return False

    followup_signals = {
        "다시", "방금", "이전", "아까", "위에",
        "그거", "그걸", "그럼", "그걸로", "이걸", "저걸",
        "표로", "테이블로", "그래프로", "차트로",
        "그려줘", "보여줘", "정리해줘", "요약해줘",
    }
    return any(signal in q for signal in followup_signals)


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
    if intent in {"overview", "batch_process", "batch_flow", "table_lineage"} and system_name:
        return templates[intent].format(system_name=system_name)
    if intent in templates:
        return templates[intent]
    return question


@dataclass
class ChatConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")
    timeout: int = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "120"))


@dataclass
class EmbedConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    timeout: int = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "60"))


class OllamaEmbeddingFunction:
    def __init__(self, config: EmbedConfig) -> None:
        self.config = config

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in input:
            resp = requests.post(
                f"{self.config.base_url}/api/embeddings",
                json={"model": self.config.model, "prompt": text},
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding")
            if not embedding:
                raise ValueError("Embedding 응답에 embedding 값이 없습니다.")
            vectors.append(embedding)
        return vectors


def ollama_generate(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    resp = requests.post(
        f"{config.base_url}/api/generate",
        json={
            "model": config.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=config.timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def build_rewrite_prompt(
    question: str,
    chat_history: List[Dict[str, str]],
    resolved_system_name: Optional[str],
    resolved_intent: str,
) -> Tuple[str, str]:
    system_prompt = (
        "너는 질문 재작성기다. 절대로 설명하지 말고, 답변하지 말고, 검색용 질문 1문장만 출력한다. "
        "출력은 반드시 한국어 한 줄이어야 한다. 이미 결정된 시스템명과 의도는 변경하지 말고 표현만 정리한다."
    )
    examples = "\n\n".join([f"[사용자]\n{ex['user']}\n[재작성]\n{ex['assistant']}" for ex in FEW_SHOT_EXAMPLES])
    history_text = history_to_text(chat_history)
    canonical_hint = build_canonical_question(question, detect_system_id(question), resolved_intent)
    prompt = f"""
다음 규칙을 지켜라.
1) 이미 결정된 시스템명은 {resolved_system_name or '(없음)'} 이다. 이 값은 절대 변경하지 않는다.
2) 이미 결정된 의도는 {resolved_intent} 이다. 이 의도는 절대 변경하지 않는다.
3) 시스템명이 결정된 경우, 다른 시스템명(KKK은행/BBB증권)을 새로 추정하거나 바꾸지 않는다.
4) 가능한 경우 아래 형태처럼 검색용 질문 1문장으로 정리한다.
- {canonical_hint}
5) 문맥상 이전 대화가 있으면 반영하되, 시스템명과 의도는 유지한다.
6) 반드시 한국어 질문 한 줄만 출력한다.

예시:
{examples}

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

    # rewrite 결과가 이미 결정된 intent와 충돌하면 버린다.
    # 예: batch_process인데 LLM이 임의로 "흐름도"를 붙이는 경우.
    intent_conflict_keywords: Dict[str, List[str]] = {
        "overview": ["배치 프로세스", "배치 흐름도", "테이블 리니지", "월별 금액", "장애현황"],
        "batch_process": ["흐름도", "리니지", "월별 금액", "장애현황", "그래프"],
        "batch_flow": ["리니지", "월별 금액", "장애현황"],
        "table_lineage": ["배치 프로세스", "배치 흐름도", "월별 금액", "장애현황"],
        "billing_monthly_amount": ["배치 프로세스", "배치 흐름도", "리니지", "장애현황"],
        "today_incidents": ["월별 금액", "배치 흐름도", "리니지"],
    }
    if resolved_intent in intent_conflict_keywords:
        if any(keyword in q for keyword in intent_conflict_keywords[resolved_intent]):
            return False

    valid_keywords = [
        "업무 개요",
        "배치 프로세스",
        "배치 흐름도",
        "테이블 리니지",
        "월별 금액",
        "장애현황",
        "배치 개발",
        "배치 생성",
        "배치 만들어",
    ]
    return any(k in q for k in valid_keywords)


def rewrite_question(
    question: str,
    chat_history: List[Dict[str, str]],
    config: ChatConfig,
    resolved_system_id: Optional[str] = None,
    resolved_intent: Optional[str] = None,
) -> Tuple[str, List[str]]:
    debug_logs: List[str] = []
    dict_rewritten = apply_dictionary_rewrite(question)
    effective_intent = resolved_intent or detect_intent(dict_rewritten)
    resolved_system_name = SYSTEM_NAME_BY_ID.get(resolved_system_id, "")

    debug_logs.append(f"[STEP 1] original_question = {question}")
    debug_logs.append(f"[STEP 2] dictionary_rewritten = {dict_rewritten}")

    if effective_intent != "default":
        canonical_question = build_canonical_question(
            question=dict_rewritten,
            resolved_system_id=resolved_system_id,
            intent=effective_intent,
        )
        if canonical_question != dict_rewritten:
            debug_logs.append("[STEP 3] canonical_rewrite = applied")
            debug_logs.append(f"[STEP 4] final_rewritten_question = {canonical_question}")
            return canonical_question, debug_logs

    if detect_system_id(dict_rewritten) and effective_intent != "default":
        debug_logs.append("[STEP 3] few_shot_rewrite = skipped (dictionary 결과가 이미 충분히 명확함)")
        debug_logs.append(f"[STEP 4] final_rewritten_question = {dict_rewritten}")
        return dict_rewritten, debug_logs

    system_prompt, prompt = build_rewrite_prompt(
        dict_rewritten,
        chat_history,
        resolved_system_name,
        effective_intent,
    )
    debug_logs.append("[STEP 3] few_shot_rewrite = started")
    try:
        rewritten = ollama_generate(prompt=prompt, system_prompt=system_prompt, config=config)
        final_rewritten = normalize_whitespace(rewritten or dict_rewritten)
        if not is_valid_rewritten_question(final_rewritten, effective_intent):
            fallback_question = build_canonical_question(dict_rewritten, resolved_system_id, effective_intent)
            debug_logs.append(f"[STEP 4] few_shot_rewrite = invalid_output ({final_rewritten})")
            debug_logs.append(f"[STEP 5] fallback_rewritten_question = {fallback_question}")
            return fallback_question, debug_logs
        debug_logs.append(f"[STEP 4] final_rewritten_question = {final_rewritten}")
        return final_rewritten, debug_logs
    except Exception as e:
        fallback_question = build_canonical_question(dict_rewritten, resolved_system_id, effective_intent)
        debug_logs.append(f"[STEP 4] few_shot_rewrite = failed ({type(e).__name__}: {e})")
        debug_logs.append(f"[STEP 5] fallback_rewritten_question = {fallback_question}")
        return fallback_question, debug_logs


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
        embedding_function=OllamaEmbeddingFunction(EmbedConfig()),
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
- batch_process 질문이면 핵심 단계와 핵심 배치를 먼저 요약한다.
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
    summary = overview.get("summary") or overview.get("content") or "업무 개요 정보가 없습니다."
    outputs = overview.get("outputs", [])
    if outputs:
        return f"{summary} 최종 산출물은 {', '.join(outputs)}입니다."
    return summary


def build_batch_process_fallback(batch_process: Dict[str, Any]) -> str:
    steps = batch_process.get("steps", [])
    parts = []
    for step in steps:
        execution = step.get("execution")
        execution_kr = "병렬" if execution == "parallel" else "순차" if execution == "sequential" else execution
        parts.append(f"{step.get('step')}단계 {step.get('name')}({execution_kr})")
    if parts:
        return "핵심 흐름은 " + " → ".join(parts) + " 순입니다."
    return batch_process.get("title", "배치 프로세스 정보가 없습니다.")


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

        normalized_question = apply_dictionary_rewrite(question)
        debug_logs.append(f"[PREP 1] normalized_question = {normalized_question}")

        system_id = detect_system_id_with_history(normalized_question, chat_history)
        debug_logs.append(f"[PREP 2] resolved_system_id_before_rewrite = {system_id}")

        initial_intent = detect_intent(normalized_question)
        if initial_intent == "default" and chat_history:
            if is_followup_question(normalized_question):
                previous_intent = detect_previous_user_intent(chat_history)
                if previous_intent != "default":
                    initial_intent = previous_intent
                    debug_logs.append(f"[PREP 3-1] followup_intent_inherited = {previous_intent}")
                else:
                    debug_logs.append("[PREP 3-1] followup_intent_inherited = skipped (previous intent 없음)")
            else:
                debug_logs.append("[PREP 3-1] history_intent_inherited = skipped (followup 아님)")
        debug_logs.append(f"[PREP 3] resolved_intent_before_rewrite = {initial_intent}")

        rewritten_question, rewrite_logs = rewrite_question(
            question=normalized_question,
            chat_history=chat_history,
            config=self.chat_config,
            resolved_system_id=system_id,
            resolved_intent=initial_intent,
        )
        debug_logs.extend(rewrite_logs)

        intent = detect_intent(rewritten_question)
        debug_logs.append(f"[STEP 5] detected_system_id = {system_id}")
        debug_logs.append(f"[STEP 6] detected_intent = {intent}")

        render_type, graph_data, query_meta, structured_data = self._build_structured_payload(system_id, intent)
        where = self._build_where_filter(system_id, intent)
        debug_logs.append(f"[STEP 7] render_type = {render_type}")
        debug_logs.append(f"[STEP 8] where_filter = {where}")

        search_query = rewritten_question
        debug_logs.append(f"[STEP 9] search_query = {search_query}")

        search_result = retrieve_docs(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            query=search_query,
            top_k=top_k,
            where=where,
        )

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

        if intent == "batch_flow" and graph_data:
            debug_logs.append("[STEP 11] structured_response = batch_flow")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_graph_answer(graph_data, intent),
                "graph",
                graph_data,
                None,
                None,
                None,
                source_rows,
                debug_logs,
            )

        if intent == "table_lineage" and graph_data:
            debug_logs.append("[STEP 11] structured_response = table_lineage")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_graph_answer(graph_data, intent),
                "graph",
                graph_data,
                None,
                None,
                None,
                source_rows,
                debug_logs,
            )

        if intent == "billing_monthly_amount" and query_meta:
            debug_logs.append("[STEP 11] structured_response = billing_monthly_amount")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_chart_answer(query_meta),
                "chart",
                None,
                query_meta,
                "chart_only",
                None,
                source_rows,
                debug_logs,
            )

        if intent == "today_incidents" and query_meta:
            debug_logs.append("[STEP 11] structured_response = today_incidents")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_table_answer(query_meta),
                "table",
                None,
                query_meta,
                "incident_table_with_summary",
                None,
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
            debug_logs.append("[STEP 11] answer_generation = started")
            answer = ollama_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.chat_config,
            )
            debug_logs.append("[STEP 12] answer_generation = success")
        except Exception as e:
            debug_logs.append(f"[STEP 12] answer_generation = failed ({type(e).__name__}: {e})")
            if intent == "overview" and structured_data:
                answer = build_overview_fallback(structured_data)
                debug_logs.append("[STEP 13] fallback = overview_structured_payload")
            elif intent == "batch_process" and structured_data:
                answer = build_batch_process_fallback(structured_data)
                debug_logs.append("[STEP 13] fallback = batch_process_structured_payload")
            elif documents:
                answer = documents[0]
                debug_logs.append("[STEP 13] fallback = first_retrieved_document")
            else:
                answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                debug_logs.append("[STEP 13] fallback = generic_error_message")

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
