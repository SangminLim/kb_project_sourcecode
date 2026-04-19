"""
llm.py (debug log 추가 버전)

설계 원칙
- 하드코딩 최소화
- dictionary 기반 질문 정규화
- LLM 기반 질문 재작성(rewrite)
- chat history 반영
- intent별 system prompt 분리
- few-shot 포함
- metadata filter 기반 검색
- LLM 실패 시 fallback 제공
- 답변은 반드시 한국어로 출력
- debug_logs 로 질문 해석 과정 추적 가능

LLM 사용 여부
- LLM 사용:
  * overview
  * batch_process
- LLM 생략:
  * batch_flow
  * table_lineage
  * billing_monthly_amount
  * today_incidents

환경 변수 (예시)
- OLLAMA_BASE_URL=http://127.0.0.1:11434
- OLLAMA_CHAT_MODEL=llama3:8b
- OLLAMA_EMBED_MODEL=nomic-embed-text
- OLLAMA_CHAT_TIMEOUT=120
- OLLAMA_EMBED_TIMEOUT=60
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_REGISTRY_PATH = Path(os.getenv("SYSTEM_REGISTRY_PATH", str(BASE_DIR / "system_registry.json")))
QUESTION_DICTIONARY_PATH = Path(os.getenv("QUESTION_DICTIONARY_PATH", str(BASE_DIR / "question_dictionary.json")))
TYPO_NORMALIZATION_PATH = Path(os.getenv("TYPO_NORMALIZATION_PATH", str(BASE_DIR / "typo_normalization.json")))

DEFAULT_SYSTEM_SPECS: List[Dict[str, Any]] = [
    {
        "system_id": "kkk_bank",
        "canonical_name": "KKK은행",
        "aliases": ["KKK은행", "케이케이케이은행", "KKK"],
    },
    {
        "system_id": "bbb_securities",
        "canonical_name": "BBB증권",
        "aliases": ["BBB증권", "비비비증권", "BBB"],
    },
]

DEFAULT_QUESTION_REPLACEMENTS = {
    "흐름 보여줘": "배치 흐름도를 그려줘",
    "흐름도 보여줘": "배치 흐름도를 그려줘",
    "플로우 보여줘": "배치 흐름도를 그려줘",
    "리니지 보여줘": "테이블 리니지를 보여줘",
    "리니지 알려줘": "테이블 리니지를 보여줘",
    "리니지는": "테이블 리니지를 보여줘",
    "리니지 좀": "테이블 리니지를 보여줘",
    "라인리지": "리니지",
    "업무 알려줘": "업무 개요를 알려줘",
    "프로세스 알려줘": "배치 프로세스를 설명해줘",
    "장애 알려줘": "오늘 장애현황 알려줘",
}

DEFAULT_TYPO_NORMALIZATION: Dict[str, str] = {
    "BBBK증권": "BBB증권",
    "BBBK": "BBB증권",
}


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_system_specs() -> List[Dict[str, Any]]:
    data = _load_json_file(SYSTEM_REGISTRY_PATH)
    if isinstance(data, list) and data:
        return data
    return DEFAULT_SYSTEM_SPECS


def load_question_replacements() -> Dict[str, str]:
    data = _load_json_file(QUESTION_DICTIONARY_PATH)
    if isinstance(data, dict) and data:
        return data
    return DEFAULT_QUESTION_REPLACEMENTS


def load_typo_normalization() -> Dict[str, str]:
    data = _load_json_file(TYPO_NORMALIZATION_PATH)
    if isinstance(data, dict) and data:
        return data
    return DEFAULT_TYPO_NORMALIZATION


SYSTEM_SPECS: List[Dict[str, Any]] = load_system_specs()
QUESTION_REPLACEMENTS: Dict[str, str] = load_question_replacements()
TYPO_NORMALIZATION: Dict[str, str] = load_typo_normalization()

INTENT_PATTERNS: Dict[str, List[str]] = {
    "overview": ["업무 개요", "업무개요", "개요", "업무 설명", "업무설명", "상세"],
    "batch_process": ["배치 프로세스", "배치프로세스", "배치 설명", "프로세스", "배치작업"],
    "batch_flow": ["배치 흐름도", "흐름도", "플로우", "flow"],
    "table_lineage": ["테이블 리니지", "리니지", "테이블 관계도", "라인리지", "lineage"],
    "billing_monthly_amount": ["청구 이용내역서", "월별 금액", "월별금액", "월별 그래프", "그래프로"],
    "today_incidents": ["오늘 장애현황", "장애현황", "장애 현황", "오류 현황", "배치 장애"],
}

SYSTEM_PROMPT_BY_INTENT: Dict[str, str] = {
    "overview": (
        "너는 금융 업무 인수인계 문서를 설명하는 실무형 어시스턴트다. "
        "반드시 한국어로만 답변하고, 과장하지 말고, 검색 결과에 있는 내용만 사용해 간결하고 정확하게 설명한다."
    ),
    "batch_process": (
        "너는 배치 운영/개발 담당자 관점에서 설명하는 어시스턴트다. "
        "반드시 한국어로만 답변하고, 단계별 처리 순서와 각 배치의 역할을 실무적으로 요약한다."
    ),
    "default": (
        "너는 금융 인수인계 RAG 어시스턴트다. "
        "반드시 한국어로만 답변하고, 질문 의도를 파악해 검색 결과 기반으로 답변한다."
    ),
}

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "user": "KKK 소득공제 설명해줘",
        "assistant": "KKK은행 소득공제 업무 개요를 알려줘",
    },
    {
        "user": "BBB증권 흐름도 보여줘",
        "assistant": "BBB증권 소득공제 배치 흐름도를 그려줘",
    },
    {
        "user": "이번 달 청구 월별금액 그래프로 보고싶어",
        "assistant": "청구 이용내역서 월별 금액을 조회해서 그래프로 보여줘",
    },
]


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


def detect_system_id_with_history(
    question: str,
    chat_history: List[Dict[str, str]],
) -> Optional[str]:
    current_system_id = detect_system_id(question)
    if current_system_id:
        return current_system_id

    for item in reversed(chat_history):
        content = item.get("content", "")
        history_system_id = detect_system_id(content)
        if history_system_id:
            return history_system_id

    return None


def detect_intent(question: str) -> str:
    q = normalize_whitespace(question)
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return "default"


def history_to_text(chat_history: List[Dict[str, str]], max_turns: int = 4) -> str:
    recent = chat_history[-max_turns:]
    lines: List[str] = []
    for item in recent:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


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
            data = resp.json()
            embedding = data.get("embedding")
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
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=config.timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def build_rewrite_prompt(question: str, chat_history: List[Dict[str, str]]) -> Tuple[str, str]:
    system_prompt = (
        "너는 질문 재작성기다. "
        "절대로 설명하지 말고, 답변하지 말고, 검색용 질문 1문장만 출력한다. "
        "출력은 반드시 한국어 평서/질문형 한 줄이어야 한다. "
        "다음 표현은 절대 출력하지 마라: "
        "'데이터를 찾았습니다', '외부 라이브러리로 렌더링하면 됩니다', "
        "'보여주는 것이 좋을 것 같습니다', '다시 보여주는 것이 좋을 것 같습니다'."
    )

    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(f"[사용자]\n{ex['user']}\n[재작성]\n{ex['assistant']}")
    example_text = "\n\n".join(examples)

    history_text = history_to_text(chat_history)

    prompt = f"""
    다음 규칙을 지켜라.
    1) 시스템명은 KKK은행 / BBB증권 중 하나로 표준화한다.
    2) 의도는 가능한 한 다음 중 하나의 질문 형태로만 바꾼다:
    - KKK은행 소득공제 업무 개요를 알려줘
    - KKK은행 소득공제 배치 프로세스를 설명해줘
    - KKK은행 소득공제 배치 흐름도를 그려줘
    - KKK은행 소득공제 테이블 리니지를 보여줘
    - 청구 이용내역서 월별 금액을 조회해서 그래프로 보여줘
    - 오늘 장애현황 알려줘
    3) 문맥상 이전 대화가 있으면 반영한다.
    4) 반드시 한국어 질문 한 줄만 출력한다.
    5) 설명문, 판단문, 답변문을 출력하지 않는다.
    6) 출력에 마침표를 붙이지 않는다.

    예시:
    {example_text}

    이전 대화:
    {history_text if history_text else "(없음)"}

    현재 질문:
    {question}
    """.strip()

    return system_prompt, prompt

def is_valid_rewritten_question(text: str) -> bool:
    q = normalize_whitespace(text)

    invalid_phrases = [
        "데이터를 찾았습니다",
        "외부 라이브러리로 렌더링하면 됩니다",
        "보여주는 것이 좋을 것 같습니다",
        "다시 보여주는 것이 좋을 것 같습니다",
        "이전 대화에서",
    ]
    if any(p in q for p in invalid_phrases):
        return False

    valid_keywords = [
        "업무 개요",
        "배치 프로세스",
        "배치 흐름도",
        "테이블 리니지",
        "월별 금액",
        "장애현황",
    ]
    if not any(k in q for k in valid_keywords):
        return False

    if "\n" in q:
        return False

    return True


def rewrite_question(
    question: str,
    chat_history: List[Dict[str, str]],
    config: ChatConfig,
) -> Tuple[str, List[str]]:
    debug_logs: List[str] = []

    dict_rewritten = apply_dictionary_rewrite(question)
    debug_logs.append(f"[STEP 1] original_question = {question}")
    debug_logs.append(f"[STEP 2] dictionary_rewritten = {dict_rewritten}")

    if detect_system_id(dict_rewritten) and detect_intent(dict_rewritten) != "default":
        debug_logs.append("[STEP 3] few_shot_rewrite = skipped (dictionary 결과가 이미 충분히 명확함)")
        debug_logs.append(f"[STEP 4] final_rewritten_question = {dict_rewritten}")
        return dict_rewritten, debug_logs

    system_prompt, prompt = build_rewrite_prompt(dict_rewritten, chat_history)
    debug_logs.append("[STEP 3] few_shot_rewrite = started")

    try:
        rewritten = ollama_generate(prompt=prompt, system_prompt=system_prompt, config=config)
        final_rewritten = normalize_whitespace(rewritten or dict_rewritten)

        if not is_valid_rewritten_question(final_rewritten):
            debug_logs.append(f"[STEP 4] few_shot_rewrite = invalid_output ({final_rewritten})")
            debug_logs.append(f"[STEP 5] fallback_rewritten_question = {dict_rewritten}")
            return dict_rewritten, debug_logs

        debug_logs.append(f"[STEP 4] final_rewritten_question = {final_rewritten}")
        return final_rewritten, debug_logs
    except Exception as e:
        debug_logs.append(f"[STEP 4] few_shot_rewrite = failed ({type(e).__name__}: {e})")
        debug_logs.append(f"[STEP 5] fallback_rewritten_question = {dict_rewritten}")
        return dict_rewritten, debug_logs


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

    kwargs: Dict[str, Any] = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def build_answer_prompt(
    rewritten_question: str,
    intent: str,
    search_result: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> Tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_BY_INTENT.get(intent, SYSTEM_PROMPT_BY_INTENT["default"])
    documents = search_result.get("documents", [[]])[0][:1]
    metadatas = search_result.get("metadatas", [[]])[0][:1]

    context_lines: List[str] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        title = meta.get("title", "")
        section = meta.get("section", "")
        system_name = meta.get("system_name", "")
        context_lines.append(
            f"[문서 {idx}] system={system_name} section={section} title={title}\n{doc}"
        )

    history_text = history_to_text(chat_history)
    prompt = f"""
이전 대화:
{history_text if history_text else "(없음)"}

사용자 질문:
{rewritten_question}

검색 문맥:
{chr(10).join(context_lines) if context_lines else "(없음)"}

답변 규칙:
- 반드시 한국어로만 답변한다.
- 영어로 답변하지 않는다.
- 검색 문맥에 있는 내용만 사용한다.
- 부족한 내용은 지어내지 않는다.
- 문맥이 부족하면 부족하다고 말한다.
- 배치 프로세스 질문이면 단계별로 나누어 설명한다.
""".strip()
    return system_prompt, prompt


def build_batch_process_fallback(batch_process: Dict[str, Any]) -> str:
    title = batch_process.get("title", "배치 프로세스")
    lines: List[str] = [f"{title}는 다음과 같습니다."]
    for step in batch_process.get("steps", []):
        step_no = step.get("step")
        step_name = step.get("name", "")
        execution = step.get("execution", "")
        execution_kr = "병렬" if execution == "parallel" else "순차" if execution == "sequential" else execution
        lines.append(f"\n{step_no}단계 {step_name} ({execution_kr})")
        for job in step.get("jobs", []):
            job_id = job.get("job_id", "")
            desc = job.get("description", "")
            lines.append(f"- {job_id}: {desc}")
    return "\n".join(lines).strip()


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
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if intent in {"batch_flow", "table_lineage"} and system_id:
            system = get_system_by_id(self.payload, system_id)
            if not system:
                return "graph", None, None
            return "graph", system.get(intent), None

        if intent == "billing_monthly_amount":
            return "chart", None, get_realtime_query(self.payload, "billing_monthly_amount")

        if intent == "today_incidents":
            return "table", None, get_realtime_query(self.payload, "today_incidents")

        return "text", None, None

    def _build_where_filter(self, system_id: Optional[str], intent: str) -> Optional[Dict[str, Any]]:
        if intent in {"overview", "batch_process", "batch_flow", "table_lineage"} and system_id:
            return {
                "$and": [
                    {"system_id": system_id},
                    {"section": intent},
                ]
            }

        if intent == "billing_monthly_amount":
            return {
                "$and": [
                    {"section": "realtime_query"},
                    {"query_id": "billing_monthly_amount"},
                ]
            }

        if intent == "today_incidents":
            return {
                "$and": [
                    {"section": "realtime_query"},
                    {"query_id": "today_incidents"},
                ]
            }

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

        rewritten_question, rewrite_logs = rewrite_question(
            question=normalized_question,
            chat_history=chat_history,
            config=self.chat_config,
        )
        debug_logs.extend(rewrite_logs)

        system_id = detect_system_id_with_history(
            rewritten_question,
            chat_history,
        )
        intent = detect_intent(rewritten_question)
        debug_logs.append(f"[STEP 5] detected_system_id = {system_id}")
        debug_logs.append(f"[STEP 6] detected_intent = {intent}")

        render_type, graph_data, query_meta = self._build_structured_payload(system_id, intent)
        where = self._build_where_filter(system_id, intent)
        debug_logs.append(f"[STEP 7] render_type = {render_type}")
        debug_logs.append(f"[STEP 8] where_filter = {where}")

        search_query = rewritten_question
        if system_id:
            search_query = f"{system_id} {rewritten_question}"
        debug_logs.append(f"[STEP 9] search_query = {search_query}")

        search_result = retrieve_docs(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            query=search_query,
            top_k=top_k,
            where=where,
        )

        source_rows: List[Dict[str, Any]] = []
        documents = search_result.get("documents", [[]])[0]
        metadatas = search_result.get("metadatas", [[]])[0]
        debug_logs.append(f"[STEP 10] retrieved_doc_count = {len(documents)}")

        for doc, meta in zip(documents, metadatas):
            source_rows.append(
                {
                    "title": meta.get("title"),
                    "section": meta.get("section"),
                    "system_name": meta.get("system_name"),
                    "preview": (doc[:160] + "...") if len(doc) > 160 else doc,
                }
            )

        # 구조형 질문은 LLM 호출 생략
        if intent == "batch_flow" and graph_data:
            debug_logs.append("[STEP 11] structured_response = batch_flow")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=system_id,
                intent=intent,
                answer=f"{graph_data.get('title')} 데이터를 찾았습니다. 외부 라이브러리로 렌더링하면 됩니다.",
                render_type="graph",
                graph_data=graph_data,
                query_meta=None,
                sources=source_rows,
                debug_logs=debug_logs,
            )

        if intent == "table_lineage" and graph_data:
            debug_logs.append("[STEP 11] structured_response = table_lineage")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=system_id,
                intent=intent,
                answer=f"{graph_data.get('title')} 데이터를 찾았습니다. 외부 라이브러리로 렌더링하면 됩니다.",
                render_type="graph",
                graph_data=graph_data,
                query_meta=None,
                sources=source_rows,
                debug_logs=debug_logs,
            )

        if intent == "billing_monthly_amount" and query_meta:
            debug_logs.append("[STEP 11] structured_response = billing_monthly_amount")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=system_id,
                intent=intent,
                answer=f"{query_meta.get('title')} 메타정보를 찾았습니다. DB 조회 후 그래프로 렌더링하면 됩니다.",
                render_type="chart",
                graph_data=None,
                query_meta=query_meta,
                sources=source_rows,
                debug_logs=debug_logs,
            )

        if intent == "today_incidents" and query_meta:
            debug_logs.append("[STEP 11] structured_response = today_incidents")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=system_id,
                intent=intent,
                answer=f"{query_meta.get('title')} 메타정보를 찾았습니다. DB 조회 후 표로 렌더링하면 됩니다.",
                render_type="table",
                graph_data=None,
                query_meta=query_meta,
                sources=source_rows,
                debug_logs=debug_logs,
            )

        system_prompt, prompt = build_answer_prompt(
            rewritten_question=rewritten_question,
            intent=intent,
            search_result=search_result,
            chat_history=chat_history,
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
            # fallback
            if intent == "batch_process" and system_id:
                system = get_system_by_id(self.payload, system_id)
                if system and system.get("batch_process"):
                    answer = build_batch_process_fallback(system["batch_process"])
                    debug_logs.append("[STEP 13] fallback = batch_process_structured_payload")
                elif documents:
                    answer = documents[0]
                    debug_logs.append("[STEP 13] fallback = first_retrieved_document")
                else:
                    answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                    debug_logs.append("[STEP 13] fallback = generic_error_message")
            elif documents:
                answer = documents[0]
                debug_logs.append("[STEP 13] fallback = first_retrieved_document")
            else:
                answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                debug_logs.append("[STEP 13] fallback = generic_error_message")

        return AgentResult(
            original_question=question,
            normalized_question=normalized_question,
            rewritten_question=rewritten_question,
            system_id=system_id,
            intent=intent,
            answer=answer,
            render_type=render_type,
            graph_data=graph_data,
            query_meta=query_meta,
            sources=source_rows,
            debug_logs=debug_logs,
        )
