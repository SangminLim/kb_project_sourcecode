from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import requests


@dataclass
class EmbedConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    timeout: int = 60


class OllamaEmbeddingFunction:
    """
    ChromaDB embedding function wrapper
    - Ollama /api/embeddings 사용
    """

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


def read_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def normalize_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, list):
        return [safe_text(v) for v in values if safe_text(v)]
    text = safe_text(values)
    return [text] if text else []


def join_labeled_list(label: str, values: Any) -> str:
    items = normalize_list(values)
    if not items:
        return ""
    return f"{label}: " + ", ".join(items)


def make_doc_id(parts: Iterable[str]) -> str:
    raw = "||".join([p for p in parts if p])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_overview_text(overview: Dict[str, Any]) -> str:
    parts: List[str] = []

    for field in ["summary", "content"]:
        value = safe_text(overview.get(field))
        if value:
            parts.append(value)

    labeled_sections = [
        ("주요 입력 데이터", overview.get("input_data")),
        ("주요 대상 거래", overview.get("target_transactions")),
        ("제외 및 보정 항목", overview.get("exclusions")),
        ("최종 산출물", overview.get("outputs")),
        ("핵심 포인트", overview.get("key_points")),
    ]
    for label, values in labeled_sections:
        text = join_labeled_list(label, values)
        if text:
            parts.append(text)

    return "\n".join(parts).strip()


def join_jobs(step: Dict[str, Any]) -> str:
    lines = [
        f"step={step.get('step')} name={step.get('name')} execution={step.get('execution')}"
    ]

    description = safe_text(step.get("description"))
    if description:
        lines.append(f"description={description}")

    key_jobs = normalize_list(step.get("key_jobs"))
    if key_jobs:
        lines.append("key_jobs=" + ", ".join(key_jobs))

    for job in step.get("jobs", []):
        job_id = safe_text(job.get("job_id"))
        job_desc = safe_text(job.get("description"))
        if job_id or job_desc:
            lines.append(f"- {job_id}: {job_desc}")

    return "\n".join(lines)


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    overview, batch_process 같은 설명형 텍스트만 보조적으로 청킹한다.
    - 너무 짧은 문서는 그대로 유지
    - 구조형 데이터(batch_flow, table_lineage, batch_step, batch_job)는 이 함수를 쓰지 않음
    """
    normalized = safe_text(text)
    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_flow_text(batch_flow: Dict[str, Any]) -> str:
    lines: List[str] = []

    summary = safe_text(batch_flow.get("summary"))
    if summary:
        lines.append(summary)

    highlight_nodes = normalize_list(batch_flow.get("highlight_nodes"))
    if highlight_nodes:
        lines.append("핵심 노드: " + ", ".join(highlight_nodes))

    start_nodes = normalize_list(batch_flow.get("start_nodes"))
    if start_nodes:
        lines.append("시작 노드: " + ", ".join(start_nodes))

    end_nodes = normalize_list(batch_flow.get("end_nodes"))
    if end_nodes:
        lines.append("종료 노드: " + ", ".join(end_nodes))

    for node in batch_flow.get("nodes", []):
        lines.append(
            f"node={node.get('id')} label={node.get('label')} type={node.get('type')} step={node.get('step')}"
        )

    for edge in batch_flow.get("edges", []):
        lines.append(f"{edge.get('from')} -> {edge.get('to')}")

    return "\n".join(lines).strip()


def build_lineage_text(table_lineage: Dict[str, Any]) -> str:
    lines: List[str] = []

    summary = safe_text(table_lineage.get("summary"))
    if summary:
        lines.append(summary)

    highlight_tables = normalize_list(table_lineage.get("highlight_tables"))
    if highlight_tables:
        lines.append("핵심 테이블: " + ", ".join(highlight_tables))

    for table in table_lineage.get("tables", []):
        lines.append(f"table={table.get('id')} layer={table.get('layer')}")

    for edge in table_lineage.get("edges", []):
        lines.append(f"{edge.get('from')} -> {edge.get('to')}")

    return "\n".join(lines).strip()


def flatten_system_docs(system: Dict[str, Any], domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    검색 품질을 위해:
    1) overview / batch_process / batch_flow / table_lineage를 별도 문서로 적재
    2) batch step / job 도 별도 문서로 적재
    3) 개선된 JSON 메타데이터(summary, input_data, outputs, description 등)를 함께 반영
    """
    docs: List[Dict[str, Any]] = []
    system_id = safe_text(system.get("system_id"))
    system_name = safe_text(system.get("system_name"))
    task_name = safe_text(system.get("task_name"))
    domain_id = safe_text(domain.get("domain_id"))
    domain_name = safe_text(domain.get("domain_name"))

    def append_doc(section: str, title: str, text: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        metadata = {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "system_id": system_id,
            "system_name": system_name,
            "task_name": task_name,
            "section": section,
            "title": title,
        }
        if extra_meta:
            metadata.update(extra_meta)
        docs.append(
            {
                "id": make_doc_id([system_id, section, title, text]),
                "document": text,
                "metadata": metadata,
            }
        )

    def append_chunked_doc(
        section: str,
        title: str,
        text: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        overlap: int = 80,
    ) -> None:
        chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return

        if len(chunks) == 1:
            append_doc(section=section, title=title, text=chunks[0], extra_meta=extra_meta)
            return

        for idx, chunk in enumerate(chunks, start=1):
            chunk_meta = {
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "is_chunked": True,
            }
            if extra_meta:
                chunk_meta.update(extra_meta)

            append_doc(
                section=section,
                title=f"{title} / chunk {idx}",
                text=chunk,
                extra_meta=chunk_meta,
            )

    overview = system.get("overview") or {}
    if overview:
        overview_text = build_overview_text(overview)
        append_chunked_doc(
            section="overview",
            title=safe_text(overview.get("title")),
            text=overview_text,
            extra_meta={
                "has_summary": bool(safe_text(overview.get("summary"))),
                "has_input_data": bool(normalize_list(overview.get("input_data"))),
                "has_outputs": bool(normalize_list(overview.get("outputs"))),
            },
            chunk_size=450,
            overlap=80,
        )

    batch_process = system.get("batch_process") or {}
    if batch_process:
        step_texts: List[str] = []
        for step in batch_process.get("steps", []):
            step_texts.append(join_jobs(step))
        batch_text = "\n\n".join([text for text in step_texts if text])
        append_chunked_doc(
            section="batch_process",
            title=safe_text(batch_process.get("title")),
            text=batch_text,
            chunk_size=600,
            overlap=100,
        )

        for step in batch_process.get("steps", []):
            append_doc(
                section="batch_step",
                title=f"{safe_text(batch_process.get('title'))} / step {step.get('step')}",
                text=join_jobs(step),
                extra_meta={
                    "step": step.get("step"),
                    "execution": safe_text(step.get("execution")),
                    "step_name": safe_text(step.get("name")),
                },
            )
            for job in step.get("jobs", []):
                job_id = safe_text(job.get("job_id"))
                job_desc = safe_text(job.get("description"))
                job_text_parts = [
                    job_id,
                    job_desc,
                    f"step={step.get('step')}",
                    f"step_name={safe_text(step.get('name'))}",
                    f"execution={safe_text(step.get('execution'))}",
                ]
                step_desc = safe_text(step.get("description"))
                if step_desc:
                    job_text_parts.append(f"step_description={step_desc}")

                append_doc(
                    section="batch_job",
                    title=job_id,
                    text="\n".join([part for part in job_text_parts if part]),
                    extra_meta={
                        "step": step.get("step"),
                        "job_id": job_id,
                    },
                )

    batch_flow = system.get("batch_flow") or {}
    if batch_flow:
        append_doc(
            section="batch_flow",
            title=safe_text(batch_flow.get("title")),
            text=build_flow_text(batch_flow),
        )

    table_lineage = system.get("table_lineage") or {}
    if table_lineage:
        append_doc(
            section="table_lineage",
            title=safe_text(table_lineage.get("title")),
            text=build_lineage_text(table_lineage),
        )

    return docs


def flatten_realtime_queries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for query in payload.get("realtime_queries", []):
        text = "\n".join(
            [
                safe_text(query.get("title")),
                safe_text(query.get("render_type")),
                safe_text(query.get("response_template")),
                safe_text(query.get("chart_type")),
                safe_text(query.get("series_name")),
                safe_text(query.get("x_field")),
                safe_text(query.get("y_field")),
                safe_text(query.get("columns")),
                safe_text(query.get("summary_prompt")),
                safe_text(query.get("data_source")),
            ]
        )
        docs.append(
            {
                "id": make_doc_id([safe_text(query.get("query_id")), text]),
                "document": text,
                "metadata": {
                    "domain_id": "realtime",
                    "domain_name": "realtime",
                    "system_id": "realtime",
                    "system_name": "realtime",
                    "task_name": safe_text(query.get("title")),
                    "section": "realtime_query",
                    "title": safe_text(query.get("title")),
                    "query_id": safe_text(query.get("query_id")),
                    "render_type": safe_text(query.get("render_type")),
                },
            }
        )
    return docs


def flatten_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for domain in payload.get("domains", []):
        for system in domain.get("systems", []):
            docs.extend(flatten_system_docs(system, domain))
    docs.extend(flatten_realtime_queries(payload))
    return docs


def upsert_documents(
    docs: List[Dict[str, Any]],
    persist_dir: str,
    collection_name: str,
    reset: bool,
    embed_config: EmbedConfig,
) -> None:
    client = chromadb.PersistentClient(path=persist_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] 기존 컬렉션 삭제: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=OllamaEmbeddingFunction(embed_config),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [d["id"] for d in docs]
    documents = [d["document"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    batch_size = 50
    for idx in range(0, len(docs), batch_size):
        collection.upsert(
            ids=ids[idx : idx + batch_size],
            documents=documents[idx : idx + batch_size],
            metadatas=metadatas[idx : idx + batch_size],
        )

    print("[INFO] 적재 완료")
    print(f"[INFO] persist_dir = {persist_dir}")
    print(f"[INFO] collection_name = {collection_name}")
    print(f"[INFO] document_count = {collection.count()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="handover JSON 경로")
    parser.add_argument("--persist_dir", default="./chroma", help="Chroma 저장 경로")
    parser.add_argument("--collection", default="handover_agent", help="컬렉션명")
    parser.add_argument("--reset", action="store_true", help="기존 컬렉션 삭제 후 재적재")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(args.json_path)
    docs = flatten_payload(payload)
    print(f"[INFO] flattened_docs = {len(docs)}")
    upsert_documents(
        docs=docs,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        reset=args.reset,
        embed_config=EmbedConfig(),
    )


if __name__ == "__main__":
    main()
