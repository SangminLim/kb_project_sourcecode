from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalPolicy:
    """
    intent별 검색 정책.
    - 여기만 바꾸면 llm.py 로직을 건드리지 않고 검색 전략을 조정할 수 있다.
    - Chroma where 필터는 ingest.py에서 저장한 metadata와 반드시 맞아야 한다.
    """
    section: Optional[str]
    top_k: int = 4
    prefer_doc_levels: List[str] = field(default_factory=list)
    query_id: Optional[str] = None


DEFAULT_RETRIEVAL_POLICIES: Dict[str, RetrievalPolicy] = {
    "overview": RetrievalPolicy(section="overview", top_k=5, prefer_doc_levels=["section", "chunk"]),
    "batch_process": RetrievalPolicy(section="batch_process", top_k=6, prefer_doc_levels=["section", "chunk"]),
    "batch_step": RetrievalPolicy(section="batch_process", top_k=4, prefer_doc_levels=["chunk", "detail"]),
    "batch_job": RetrievalPolicy(section="batch_job", top_k=4, prefer_doc_levels=["detail"]),
    "batch_flow": RetrievalPolicy(section="batch_flow", top_k=3, prefer_doc_levels=["structure"]),
    "table_lineage": RetrievalPolicy(section="table_lineage", top_k=3, prefer_doc_levels=["structure"]),
    "billing_monthly_amount": RetrievalPolicy(section="realtime_query", top_k=2, prefer_doc_levels=["structure"], query_id="billing_monthly_amount"),
    "today_incidents": RetrievalPolicy(section="realtime_query", top_k=2, prefer_doc_levels=["structure"], query_id="today_incidents"),
    "default": RetrievalPolicy(section=None, top_k=5),
}


@dataclass
class QueryPlan:
    original_question: str
    search_query: str
    intent: str
    retrieval_intent: str
    system_id: Optional[str]
    where: Optional[Dict[str, Any]]
    top_k: int
    step: Optional[int] = None
    job_id: Optional[str] = None
    debug: List[str] = field(default_factory=list)


def _and_filter(conditions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clean = [cond for cond in conditions if cond]
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    return {"$and": clean}


def extract_step(question: str) -> Optional[int]:
    """질문에서 '2단계', 'step 2', 'STEP2' 같은 표현을 추출한다."""
    q = question or ""
    patterns = [
        r"(?:step|STEP)\s*([0-9]+)",
        r"([0-9]+)\s*단계",
        r"([0-9]+)\s*번째\s*단계",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def extract_job_id(question: str) -> Optional[str]:
    """질문에서 BATCH_XX_... 형태의 배치ID를 추출한다."""
    match = re.search(r"\b(BATCH_[A-Z0-9_]+)\b", question or "", flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def resolve_retrieval_intent(intent: str, question: str) -> str:
    """
    사용자 의도보다 더 세밀한 검색 의도를 결정한다.
    예: intent는 batch_process지만 질문에 '2단계'가 있으면 batch_step으로 좁힌다.
    """
    if extract_job_id(question):
        return "batch_job"
    if intent == "batch_process" and extract_step(question) is not None:
        return "batch_step"
    return intent if intent in DEFAULT_RETRIEVAL_POLICIES else "default"


def build_retrieval_plan(
    *,
    original_question: str,
    rewritten_question: str,
    system_id: Optional[str],
    intent: str,
    default_top_k: int = 4,
) -> QueryPlan:
    retrieval_intent = resolve_retrieval_intent(intent, rewritten_question or original_question)
    policy = DEFAULT_RETRIEVAL_POLICIES.get(retrieval_intent, DEFAULT_RETRIEVAL_POLICIES["default"])

    step = extract_step(rewritten_question) or extract_step(original_question)
    job_id = extract_job_id(rewritten_question) or extract_job_id(original_question)

    conditions: List[Dict[str, Any]] = []
    if system_id and policy.section not in {"realtime_query"}:
        conditions.append({"system_id": system_id})
    if policy.section:
        conditions.append({"section": policy.section})
    if policy.query_id:
        conditions.append({"query_id": policy.query_id})
    if retrieval_intent == "batch_step" and step is not None:
        conditions.append({"step": step})
    if retrieval_intent == "batch_job" and job_id:
        conditions.append({"job_id": job_id})

    top_k = policy.top_k or default_top_k
    search_query = rewritten_question or original_question

    debug = [
        f"retrieval_intent={retrieval_intent}",
        f"step={step}",
        f"job_id={job_id}",
        f"top_k={top_k}",
    ]

    return QueryPlan(
        original_question=original_question,
        search_query=search_query,
        intent=intent,
        retrieval_intent=retrieval_intent,
        system_id=system_id,
        where=_and_filter(conditions),
        top_k=top_k,
        step=step,
        job_id=job_id,
        debug=debug,
    )


def should_retry_without_step(plan: QueryPlan) -> bool:
    """단계/배치ID 필터가 너무 강해서 결과가 0건일 때 완화 검색할지 판단한다."""
    return plan.retrieval_intent in {"batch_step", "batch_job"}


def relax_plan(plan: QueryPlan) -> QueryPlan:
    """강한 필터를 제거하고 section/system 수준으로 완화한다."""
    fallback_intent = "batch_process" if plan.retrieval_intent in {"batch_step", "batch_job"} else plan.intent
    policy = DEFAULT_RETRIEVAL_POLICIES.get(fallback_intent, DEFAULT_RETRIEVAL_POLICIES["default"])
    conditions: List[Dict[str, Any]] = []
    if plan.system_id and policy.section not in {"realtime_query"}:
        conditions.append({"system_id": plan.system_id})
    if policy.section:
        conditions.append({"section": policy.section})
    if policy.query_id:
        conditions.append({"query_id": policy.query_id})

    return QueryPlan(
        original_question=plan.original_question,
        search_query=plan.search_query,
        intent=plan.intent,
        retrieval_intent=fallback_intent,
        system_id=plan.system_id,
        where=_and_filter(conditions),
        top_k=policy.top_k,
        step=None,
        job_id=None,
        debug=plan.debug + ["relaxed_filter=true"],
    )
