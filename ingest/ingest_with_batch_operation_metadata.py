from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbedConfig:
    api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    model: str = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large-query")
    timeout: int = int(os.getenv("UPSTAGE_EMBED_TIMEOUT", "60"))


class UpstageEmbeddingFunction:
    """
    ChromaDB embedding function wrapper
    - Upstage Embeddings API 사용
    - Chroma 저장소는 그대로 사용하고, 임베딩 생성만 Upstage로 변경한다.
    """

    def __init__(self, config: EmbedConfig) -> None:
        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "UPSTAGE_API_KEY가 비어 있습니다. .env 또는 OS 환경변수에 UPSTAGE_API_KEY를 설정하세요."
            )

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        for text in input:
            response = requests.post(
                f"{self.config.base_url.rstrip('/')}/embeddings",
                headers=headers,
                json={
                    "model": self.config.model,
                    "input": text,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()
            embedding = (data.get("data") or [{}])[0].get("embedding")

            if not embedding:
                raise ValueError("Upstage Embedding 응답에 embedding 값이 없습니다.")

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


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    범용 fallback 청킹
    - overview/batch_process를 의미 단위로 먼저 자른 뒤
      너무 긴 청크만 추가로 잘라내는 보조 용도
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


def split_long_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    normalized = safe_text(text)
    if not normalized:
        return []
    return split_text(normalized, chunk_size=chunk_size, overlap=overlap)


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


def build_overview_chunks(overview: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    2차 청킹: overview를 의미 단위로 분리
    - 기존 전체 overview 문서도 유지할 수 있도록 별도 청크를 추가 생성
    - 너무 세분화하지 않고 질문 대응에 필요한 정도만 분리
    """
    chunk_specs = [
        (
            "summary_content",
            [safe_text(overview.get("summary")), safe_text(overview.get("content"))],
        ),
        (
            "input_target",
            [
                join_labeled_list("주요 입력 데이터", overview.get("input_data")),
                join_labeled_list("주요 대상 거래", overview.get("target_transactions")),
            ],
        ),
        (
            "exclusions",
            [join_labeled_list("제외 및 보정 항목", overview.get("exclusions"))],
        ),
        (
            "outputs_key_points",
            [
                join_labeled_list("최종 산출물", overview.get("outputs")),
                join_labeled_list("핵심 포인트", overview.get("key_points")),
            ],
        ),
    ]

    chunks: List[Dict[str, Any]] = []
    logical_order = 1
    for chunk_type, parts in chunk_specs:
        text = "\n".join([part for part in parts if safe_text(part)]).strip()
        if not text:
            continue

        split_chunks = split_long_chunk(text, chunk_size=450, overlap=80)
        for sub_index, chunk_text in enumerate(split_chunks, start=1):
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_type": chunk_type,
                    "chunk_order": logical_order,
                    "sub_chunk_index": sub_index,
                    "sub_chunk_count": len(split_chunks),
                }
            )
        logical_order += 1

    return chunks


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

        operation_fields = [
            ("job_name", "배치명"),
            ("schedule_type", "실행주기"),
            ("execution_time", "실행시간"),
            ("avg_duration_sec", "평균수행시간초"),
            ("batch_file", "실행배치파일"),
            ("program_name", "프로그램명"),
            ("owner_team", "담당팀"),
            ("retry_count", "재시도횟수"),
            ("upstream_jobs", "선행배치"),
            ("downstream_jobs", "후행배치"),
            ("failure_action", "장애조치방법"),
            ("operation_note", "운영비고"),
        ]
        for field, label in operation_fields:
            value = job.get(field)
            if isinstance(value, list):
                values = normalize_list(value)
                if values:
                    lines.append(f"  {label}: " + ", ".join(values))
            else:
                text = safe_text(value)
                if text:
                    lines.append(f"  {label}: {text}")

    return "\n".join(lines)


def build_batch_process_chunks(batch_process: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    2차 청킹: batch_process는 step 단위가 핵심
    - 질문이 단계 중심으로 들어오므로 step 단위가 가장 실무적
    - 너무 긴 step은 내부적으로만 추가 분할
    """
    chunks: List[Dict[str, Any]] = []
    for step in batch_process.get("steps", []):
        step_no = step.get("step")
        step_text = join_jobs(step)
        if not step_text:
            continue

        split_chunks = split_long_chunk(step_text, chunk_size=600, overlap=100)
        for sub_index, chunk_text in enumerate(split_chunks, start=1):
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_type": "step",
                    "chunk_order": step_no if isinstance(step_no, int) else 0,
                    "step": step_no,
                    "step_name": safe_text(step.get("name")),
                    "execution": safe_text(step.get("execution")),
                    "sub_chunk_index": sub_index,
                    "sub_chunk_count": len(split_chunks),
                }
            )
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
    2) overview_chunk / batch_process_chunk 를 추가 적재 (2차 청킹)
    3) batch_step / batch_job 도 별도 문서로 적재
    4) 흐름도/리니지/실시간 구조는 기존처럼 유지
    """
    docs: List[Dict[str, Any]] = []
    system_id = safe_text(system.get("system_id"))
    system_name = safe_text(system.get("system_name"))
    task_name = safe_text(system.get("task_name"))
    domain_id = safe_text(domain.get("domain_id"))
    domain_name = safe_text(domain.get("domain_name"))

    def append_doc(section: str, title: str, text: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        if not safe_text(text):
            return

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
                "id": make_doc_id([system_id, section, title, text, safe_text(metadata.get("chunk_id"))]),
                "document": text,
                "metadata": metadata,
            }
        )

    overview = system.get("overview") or {}
    if overview:
        overview_title = safe_text(overview.get("title"))
        overview_text = build_overview_text(overview)

        # 기존 전체 overview 문서 유지
        append_doc(
            section="overview",
            title=overview_title,
            text=overview_text,
            extra_meta={
                "doc_level": "section",
                "has_summary": bool(safe_text(overview.get("summary"))),
                "has_input_data": bool(normalize_list(overview.get("input_data"))),
                "has_outputs": bool(normalize_list(overview.get("outputs"))),
            },
        )

        # 2차 청킹 문서 추가
        overview_chunks = build_overview_chunks(overview)
        for idx, chunk in enumerate(overview_chunks, start=1):
            append_doc(
                section="overview",
                title=f"{overview_title} / semantic chunk {idx}",
                text=chunk["chunk_text"],
                extra_meta={
                    "doc_level": "chunk",
                    "chunk_id": f"overview_chunk_{idx}",
                    "chunk_type": chunk["chunk_type"],
                    "chunk_order": chunk["chunk_order"],
                    "sub_chunk_index": chunk["sub_chunk_index"],
                    "sub_chunk_count": chunk["sub_chunk_count"],
                    "is_chunked": True,
                },
            )

    batch_process = system.get("batch_process") or {}
    if batch_process:
        batch_title = safe_text(batch_process.get("title"))
        step_texts: List[str] = []
        for step in batch_process.get("steps", []):
            step_texts.append(join_jobs(step))
        batch_text = "\n\n".join([text for text in step_texts if text])

        # 기존 전체 batch_process 문서 유지
        append_doc(
            section="batch_process",
            title=batch_title,
            text=batch_text,
            extra_meta={"doc_level": "section"},
        )

        # 2차 청킹 문서 추가
        process_chunks = build_batch_process_chunks(batch_process)
        for idx, chunk in enumerate(process_chunks, start=1):
            append_doc(
                section="batch_process",
                title=f"{batch_title} / step chunk {idx}",
                text=chunk["chunk_text"],
                extra_meta={
                    "doc_level": "chunk",
                    "chunk_id": f"batch_process_chunk_{idx}",
                    "chunk_type": chunk["chunk_type"],
                    "chunk_order": chunk["chunk_order"],
                    "step": chunk["step"],
                    "step_name": chunk["step_name"],
                    "execution": chunk["execution"],
                    "sub_chunk_index": chunk["sub_chunk_index"],
                    "sub_chunk_count": chunk["sub_chunk_count"],
                    "is_chunked": True,
                },
            )

        # 기존 세부 검색용 batch_step / batch_job 유지
        for step in batch_process.get("steps", []):
            append_doc(
                section="batch_step",
                title=f"{batch_title} / step {step.get('step')}",
                text=join_jobs(step),
                extra_meta={
                    "step": step.get("step"),
                    "execution": safe_text(step.get("execution")),
                    "step_name": safe_text(step.get("name")),
                    "doc_level": "detail",
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

                operation_fields = [
                    ("job_name", "job_name"),
                    ("schedule_type", "schedule_type"),
                    ("execution_time", "execution_time"),
                    ("avg_duration_sec", "avg_duration_sec"),
                    ("batch_file", "batch_file"),
                    ("program_name", "program_name"),
                    ("owner_team", "owner_team"),
                    ("retry_count", "retry_count"),
                    ("upstream_jobs", "upstream_jobs"),
                    ("downstream_jobs", "downstream_jobs"),
                    ("failure_action", "failure_action"),
                    ("operation_note", "operation_note"),
                ]
                operation_meta: Dict[str, Any] = {}
                for field, label in operation_fields:
                    value = job.get(field)
                    if isinstance(value, list):
                        values = normalize_list(value)
                        if values:
                            joined_value = ", ".join(values)
                            job_text_parts.append(f"{label}={joined_value}")
                            operation_meta[field] = joined_value
                    else:
                        text = safe_text(value)
                        if text:
                            job_text_parts.append(f"{label}={text}")
                            operation_meta[field] = text

                append_doc(
                    section="batch_job",
                    title=job_id,
                    text="\n".join([part for part in job_text_parts if part]),
                    extra_meta={
                        "step": step.get("step"),
                        "job_id": job_id,
                        "doc_level": "detail",
                        **operation_meta,
                    },
                )

    batch_flow = system.get("batch_flow") or {}
    if batch_flow:
        append_doc(
            section="batch_flow",
            title=safe_text(batch_flow.get("title")),
            text=build_flow_text(batch_flow),
            extra_meta={"doc_level": "structure"},
        )

    table_lineage = system.get("table_lineage") or {}
    if table_lineage:
        append_doc(
            section="table_lineage",
            title=safe_text(table_lineage.get("title")),
            text=build_lineage_text(table_lineage),
            extra_meta={"doc_level": "structure"},
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
                    "doc_level": "structure",
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
        embedding_function=UpstageEmbeddingFunction(embed_config),
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
    load_dotenv()
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
