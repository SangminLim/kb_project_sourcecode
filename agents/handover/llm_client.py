from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:
    _langsmith_traceable = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.embeddings import Embeddings
    from langchain_chroma import Chroma as LangChainChroma
except Exception:
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None
    Embeddings = None
    LangChainChroma = None

from .config import ChatConfig, EmbedConfig

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


class UpstageLangChainEmbeddings(Embeddings if Embeddings is not None else object):
    """LangChain VectorStore에서 사용할 Upstage Embeddings adapter.

    기존 Chroma ingest/query에 쓰던 UpstageEmbeddingFunction을 그대로 재사용한다.
    모델명, base_url, timeout은 EmbedConfig/.env에서 읽기 때문에 코드에 업무별 값을 박지 않는다.
    """

    def __init__(self, config: Optional[EmbedConfig] = None) -> None:
        self.config = config or EmbedConfig()
        self.embedding_function = UpstageEmbeddingFunction(self.config)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_function([str(text) for text in texts])

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embedding_function([str(text)])
        if not vectors:
            raise ValueError("Upstage embedding 결과가 비어 있습니다.")
        return vectors[0]


def _use_langchain() -> bool:
    """기존 설정 호환용 함수.

    Upstage는 OpenAI 호환 REST API를 직접 호출하므로 LangChain이 없어도 동작한다.
    기존 debug flag 구조를 깨지 않기 위해 함수명은 유지한다.
    """
    return os.getenv("LANGCHAIN_ENABLED", "false").lower() not in {"0", "false", "no", "n"}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "n"}




def _use_langsmith() -> bool:
    """LangSmith tracing 활성화 여부를 환경변수 기준으로 판단한다.

    LangChain/LangGraph 자동 tracing은 LANGCHAIN_TRACING_V2 또는 LANGSMITH_TRACING으로 켜지고,
    requests 기반 직접 호출은 이 함수와 _run_with_langsmith_trace를 통해 선택적으로 trace된다.
    """
    tracing_enabled = (
        _env_flag("LANGSMITH_TRACING", "false")
        or _env_flag("LANGCHAIN_TRACING_V2", "false")
        or _env_flag("LANGSMITH_TRACING_V2", "false")
    )
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return bool(tracing_enabled and has_key)


def _run_with_langsmith_trace(
    *,
    name: str,
    run_type: str,
    inputs: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    fn: Callable[[], Any],
) -> Any:
    """langsmith가 설치/설정된 경우에만 함수 호출을 trace한다.

    langsmith 미설치, API KEY 미설정, tracing 비활성 상태에서는 기존 동작을 그대로 유지한다.
    """
    if not _use_langsmith() or _langsmith_traceable is None:
        return fn()

    @_langsmith_traceable(name=name, run_type=run_type, metadata=metadata or {})
    def _wrapped(**kwargs: Any) -> Any:
        return fn()

    return _wrapped(**(inputs or {}))


def get_langchain_feature_flags() -> Dict[str, bool]:
    """운영 중 기능을 켜고 끌 수 있는 확장 옵션."""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "upstage"),
        "langchain_enabled": _use_langchain(),
        "rewrite_chain_enabled": _env_flag("LANGCHAIN_REWRITE_CHAIN_ENABLED", "true"),
        "retriever_enabled": _env_flag("LANGCHAIN_RETRIEVER_ENABLED", "true"),
        "router_enabled": _env_flag("LANGCHAIN_ROUTER_ENABLED", "true"),
        "structured_parser_enabled": _env_flag("LANGCHAIN_STRUCTURED_PARSER_ENABLED", "true"),
        "retrieval_compression_enabled": _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false"),
        "langsmith_tracing_enabled": _use_langsmith(),
    }


def _langchain_generate_text(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    """LangChain 기반 텍스트 생성 공통 함수.

    현재는 rewrite_question에서만 사용한다.
    실패 시 기존 requests 기반 호출로 fallback할 수 있도록 예외를 그대로 전달한다.
    """
    if not _use_langchain() or not _env_flag("LANGCHAIN_REWRITE_CHAIN_ENABLED", "true"):
        raise RuntimeError("LangChain rewrite chain is disabled.")

    if ChatOpenAI is None or ChatPromptTemplate is None or StrOutputParser is None:
        raise RuntimeError(
            "LangChain 패키지가 설치되어 있지 않습니다. "
            "pip install langchain langchain-core langchain-openai 를 확인하세요."
        )

    if not config.api_key:
        raise ValueError("UPSTAGE_API_KEY가 비어 있습니다. .env에 UPSTAGE_API_KEY를 설정하세요.")

    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url.rstrip("/"),
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
    )

    chain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{user_prompt}"),
        ]
    )

    chain = chain_prompt | llm | StrOutputParser()
    return str(chain.invoke({"user_prompt": prompt})).strip()


def _upstage_generate_requests(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    if not config.api_key:
        raise ValueError("UPSTAGE_API_KEY가 비어 있습니다. .env에 UPSTAGE_API_KEY를 설정하세요.")

    def _call_upstage() -> str:
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

    return _run_with_langsmith_trace(
        name="upstage_chat_generate",
        run_type="llm",
        inputs={"prompt": prompt, "system_prompt": system_prompt, "model": config.model},
        metadata={
            "provider": "upstage",
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "call_path": "requests",
        },
        fn=_call_upstage,
    )


def ollama_generate(prompt: str, system_prompt: str, config: ChatConfig) -> str:
    """답변 생성 진입점.

    기존 호출부가 ollama_generate를 사용하고 있어서 함수명은 유지하되,
    내부 구현은 Upstage Chat API 호출로 변경한다.
    """
    return _upstage_generate_requests(prompt=prompt, system_prompt=system_prompt, config=config)


def get_llm_engine_name() -> str:
    model = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro3")
    if _use_langchain() and _env_flag("LANGCHAIN_REWRITE_CHAIN_ENABLED", "true"):
        return f"langchain_upstage:{model}"
    return f"requests_upstage:{model}"


# 기존 ingest 코드 호환 alias
OllamaEmbeddingFunction = UpstageEmbeddingFunction
upstage_generate = ollama_generate
