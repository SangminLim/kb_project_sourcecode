from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from .config import EmbedConfig
from .llm_client import (
    Embeddings, LangChainChroma, UpstageEmbeddingFunction, UpstageLangChainEmbeddings,
    _env_flag, _use_langchain,
)
from .utils import normalize_whitespace

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


def _retrieve_docs_chromadb_direct(
    persist_dir: str,
    collection_name: str,
    query: str,
    top_k: int = 4,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """기존 ChromaDB 직접 조회 경로.

    LangChain retriever가 비활성화되었거나 실패할 때 안정적인 fallback으로 사용한다.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=UpstageEmbeddingFunction(EmbedConfig()),
    )
    kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": top_k}
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _retrieve_docs_langchain(
    persist_dir: str,
    collection_name: str,
    query: str,
    top_k: int = 4,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LangChain Retriever 기반 문서 조회.

    확장 포인트:
    - LANGCHAIN_RETRIEVER_SEARCH_TYPE: similarity, mmr 등 LangChain retriever search_type
    - LANGCHAIN_RETRIEVER_FETCH_K: MMR 등에서 후보 문서 수
    - where 필터는 기존 Chroma metadata filter를 그대로 전달한다.
    """
    if not _use_langchain() or not _env_flag("LANGCHAIN_RETRIEVER_ENABLED", "true"):
        raise RuntimeError("LangChain retriever is disabled.")

    if LangChainChroma is None or Embeddings is None:
        raise RuntimeError(
            "LangChain Chroma 패키지가 설치되어 있지 않습니다. "
            "pip install langchain-chroma langchain-core 를 확인하세요."
        )

    embeddings = UpstageLangChainEmbeddings(EmbedConfig())

    vectorstore = LangChainChroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    search_type = os.getenv("LANGCHAIN_RETRIEVER_SEARCH_TYPE", "similarity").strip() or "similarity"

    search_kwargs: Dict[str, Any] = {"k": top_k}
    if where:
        search_kwargs["filter"] = where

    fetch_k = os.getenv("LANGCHAIN_RETRIEVER_FETCH_K", "").strip()
    if fetch_k:
        try:
            search_kwargs["fetch_k"] = int(fetch_k)
        except Exception:
            pass

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    docs = retriever.invoke(query)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for idx, doc in enumerate(docs):
        documents.append(str(getattr(doc, "page_content", "") or ""))
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadatas.append(metadata)

        doc_id = (
            metadata.get("id")
            or metadata.get("chunk_id")
            or metadata.get("source")
            or f"langchain_doc_{idx + 1}"
        )
        ids.append(str(doc_id))

    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [[]],
    }


def retrieve_docs(
    persist_dir: str,
    collection_name: str,
    query: str,
    top_k: int = 4,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """문서 검색 진입점.

    기본은 LangChain Retriever를 사용하고, 실패하면 기존 ChromaDB 직접 조회로 fallback한다.
    업무별 조건은 코드에 박지 않고 where filter와 env 설정으로 주입한다.
    """
    if _use_langchain() and _env_flag("LANGCHAIN_RETRIEVER_ENABLED", "true"):
        try:
            return _retrieve_docs_langchain(
                persist_dir=persist_dir,
                collection_name=collection_name,
                query=query,
                top_k=top_k,
                where=where,
            )
        except Exception:
            # 호출부의 AgentResult/debug_logs 구조를 깨지 않기 위해 여기서는 조용히 fallback한다.
            # 검색 실패 원인까지 화면에 보여줘야 하면 answer_question에서 별도 wrapper를 두면 된다.
            pass

    return _retrieve_docs_chromadb_direct(
        persist_dir=persist_dir,
        collection_name=collection_name,
        query=query,
        top_k=top_k,
        where=where,
    )
