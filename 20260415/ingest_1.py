import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings


# ==============================
# 설정
# ==============================
DEFAULT_JSON_PATH = "./handover_agent_ready.json"
PERSIST_DIR = "./chroma"
COLLECTION_NAME = "chroma-handover-agent-v1"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

UPSTAGE_MODEL = "solar-embedding-1-large"


# ==============================
# 유틸
# ==============================
def load_json(json_path: str) -> Dict[str, Any]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_text_block(title: str, lines: List[str]) -> str:
    cleaned = [line for line in lines if line and str(line).strip()]
    body = "\n".join(cleaned)
    return f"[title] {title}\n{body}"


# ==============================
# JSON -> LangChain Document 변환
# ==============================
def extract_intent_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    intents = data.get("intents", [])
    for item in intents:
        intent_name = item.get("intent_name", "")
        sample_questions = item.get("sample_questions", [])
        response_type = item.get("response_type", "")
        next_recommended_intents = item.get("next_recommended_intents", [])

        content = build_text_block(
            title=f"Intent 정의 - {intent_name}",
            lines=[
                f"[doc_type] intent_definition",
                f"[intent_name] {intent_name}",
                f"[response_type] {response_type}",
                f"[sample_questions] {', '.join(sample_questions)}",
                f"[next_recommended_intents] {', '.join(next_recommended_intents)}",
            ],
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_type": "intent_definition",
                    "intent_name": intent_name,
                    "response_type": response_type,
                    "source_section": "intents",
                },
            )
        )

    return docs


def extract_conversation_flow_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    flow_map = data.get("conversation_flow", {})
    for intent_name, flow_info in flow_map.items():
        answer_policy = flow_info.get("answer_policy", "")
        suggested_followups = flow_info.get("suggested_followups", [])

        content = build_text_block(
            title=f"대화 흐름 - {intent_name}",
            lines=[
                f"[doc_type] conversation_flow",
                f"[intent_name] {intent_name}",
                f"[answer_policy] {answer_policy}",
                f"[suggested_followups] {', '.join(suggested_followups)}",
            ],
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_type": "conversation_flow",
                    "intent_name": intent_name,
                    "source_section": "conversation_flow",
                },
            )
        )

    return docs


def extract_work_catalog_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    work_catalog = data.get("work_catalog", [])
    for item in work_catalog:
        domain = item.get("domain", "")
        customer = item.get("customer", "")
        title = item.get("title", "")
        source = item.get("source", "")
        summary = item.get("summary", "")

        # 1) 업무 요약 문서
        summary_content = build_text_block(
            title=title,
            lines=[
                f"[doc_type] work_summary",
                f"[domain] {domain}",
                f"[customer] {customer}",
                f"[source] {source}",
                f"[summary] {summary}",
            ],
        )
        docs.append(
            Document(
                page_content=summary_content,
                metadata={
                    "doc_type": "work_summary",
                    "domain": domain,
                    "customer": customer,
                    "title": title,
                    "source": source,
                },
            )
        )

        # 2) 배치 플로우 문서
        batch_flow = item.get("batch_flow", {})
        execution_rule = batch_flow.get("execution_rule", "")
        stages = batch_flow.get("stages", [])

        stage_lines = [
            f"[doc_type] batch_flow",
            f"[domain] {domain}",
            f"[customer] {customer}",
            f"[source] {source}",
            f"[execution_rule] {execution_rule}",
        ]

        for stage in stages:
            stage_name = stage.get("stage", "")
            name = stage.get("name", "")
            parallel = stage.get("parallel", False)
            jobs = stage.get("jobs", [])

            stage_lines.append(
                f"[stage] {stage_name} / {name} / parallel={parallel}"
            )

            for job in jobs:
                job_id = job.get("job_id", "")
                desc = job.get("description", "")
                stage_lines.append(f"  - [job_id] {job_id} / [description] {desc}")

                # 배치 작업별 개별 문서도 생성
                job_content = build_text_block(
                    title=f"{title} - {job_id}",
                    lines=[
                        f"[doc_type] batch_job",
                        f"[domain] {domain}",
                        f"[customer] {customer}",
                        f"[parent_title] {title}",
                        f"[job_id] {job_id}",
                        f"[description] {desc}",
                        f"[stage] {stage_name}",
                        f"[stage_name] {name}",
                        f"[execution_rule] {execution_rule}",
                    ],
                )
                docs.append(
                    Document(
                        page_content=job_content,
                        metadata={
                            "doc_type": "batch_job",
                            "domain": domain,
                            "customer": customer,
                            "title": title,
                            "job_id": job_id,
                            "stage": stage_name,
                            "source": source,
                        },
                    )
                )

        batch_content = build_text_block(
            title=f"{title} - 배치 플로우",
            lines=stage_lines,
        )
        docs.append(
            Document(
                page_content=batch_content,
                metadata={
                    "doc_type": "batch_flow",
                    "domain": domain,
                    "customer": customer,
                    "title": title,
                    "source": source,
                },
            )
        )

        # 3) 테이블 리니지 문서
        table_lineage = item.get("table_lineage", {})
        source_tables = table_lineage.get("source_tables", [])
        intermediate_tables = table_lineage.get("intermediate_tables", [])
        result_tables = table_lineage.get("result_tables", [])
        relations = table_lineage.get("relations", [])

        lineage_lines = [
            f"[doc_type] table_lineage",
            f"[domain] {domain}",
            f"[customer] {customer}",
            f"[source] {source}",
            f"[source_tables] {', '.join(source_tables)}",
            f"[intermediate_tables] {', '.join(intermediate_tables)}",
            f"[result_tables] {', '.join(result_tables)}",
        ]

        for rel in relations:
            src = rel.get("from", "")
            dst = rel.get("to", "")
            lineage_lines.append(f"[relation] {src} -> {dst}")

            # relation 단위 문서도 생성
            rel_content = build_text_block(
                title=f"{title} - 테이블 관계 {src} -> {dst}",
                lines=[
                    f"[doc_type] table_relation",
                    f"[domain] {domain}",
                    f"[customer] {customer}",
                    f"[parent_title] {title}",
                    f"[from] {src}",
                    f"[to] {dst}",
                ],
            )
            docs.append(
                Document(
                    page_content=rel_content,
                    metadata={
                        "doc_type": "table_relation",
                        "domain": domain,
                        "customer": customer,
                        "title": title,
                        "from_table": src,
                        "to_table": dst,
                        "source": source,
                    },
                )
            )

        lineage_content = build_text_block(
            title=f"{title} - 테이블 리니지",
            lines=lineage_lines,
        )
        docs.append(
            Document(
                page_content=lineage_content,
                metadata={
                    "doc_type": "table_lineage",
                    "domain": domain,
                    "customer": customer,
                    "title": title,
                    "source": source,
                },
            )
        )

    return docs


def extract_development_environment_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    env = data.get("development_environment", {})
    if not env:
        return docs

    title = env.get("title", "개발환경 구축")
    source = env.get("source", "")
    summary = env.get("summary", "")
    components = env.get("components", {})
    paths = env.get("paths", {})
    installation_steps = env.get("installation_steps", [])
    related_links = env.get("related_links", [])

    lines = [
        f"[doc_type] development_environment",
        f"[source] {source}",
        f"[summary] {summary}",
        f"[components] {safe_str(components)}",
        f"[paths] {safe_str(paths)}",
    ]

    for step in installation_steps:
        lines.append(
            f"[installation_step] {step.get('step', '')} / {step.get('name', '')}"
        )
        for detail in step.get("details", []):
            lines.append(f"  - {detail}")

    for link in related_links:
        lines.append(f"[related_link] {link.get('name', '')} = {link.get('url', '')}")

    content = build_text_block(title=title, lines=lines)

    docs.append(
        Document(
            page_content=content,
            metadata={
                "doc_type": "development_environment",
                "title": title,
                "source": source,
            },
        )
    )

    return docs


def extract_operational_query_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    operational_queries = data.get("operational_queries", {})
    for query_name, item in operational_queries.items():
        content = build_text_block(
            title=f"운영 조회 - {query_name}",
            lines=[
                f"[doc_type] operational_query",
                f"[query_name] {query_name}",
                f"[description] {item.get('description', '')}",
                f"[default_condition] {safe_str(item.get('default_condition', {}))}",
                f"[tables] {', '.join(item.get('tables', []))}",
                f"[columns] {', '.join(item.get('columns', []))}",
                f"[sql_example] {item.get('sql_example', '')}",
                f"[response_format] {item.get('response_format', '')}",
            ],
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_type": "operational_query",
                    "query_name": query_name,
                    "title": f"운영 조회 - {query_name}",
                },
            )
        )

    return docs


def extract_business_summary_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    for item in data.get("business_summary", []):
        title = item.get("title", "")
        summary = item.get("summary", "")

        domain = "기타"
        customer = "기타"

        if "소득공제" in title:
            domain = "소득공제"
        elif "청구" in title:
            domain = "청구"

        if "A은행" in title:
            customer = "A은행"
        elif "B증권" in title:
            customer = "B증권"

        content = build_text_block(
            title=title,
            lines=[
                f"[doc_type] business_summary",
                f"[domain] {domain}",
                f"[customer] {customer}",
                f"[summary] {summary}",
            ],
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_type": "business_summary",
                    "domain": domain,
                    "customer": customer,
                    "title": title,
                },
            )
        )

    return docs


def extract_agent_optimization_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    opt = data.get("agent_optimization", {})
    if not opt:
        return docs

    role_def = opt.get("job_role_definition", {})
    work_sequence = opt.get("work_sequence", [])
    incident_response = opt.get("incident_response_procedure", [])
    batch_table_mapping = opt.get("batch_table_mapping", [])
    query_parameters = opt.get("query_parameters", [])

    lines = [
        f"[doc_type] agent_optimization",
        f"[job_role_description] {role_def.get('description', '')}",
        f"[main_tasks] {', '.join(role_def.get('main_tasks', []))}",
        f"[work_sequence] {' > '.join(work_sequence)}",
        f"[incident_response_procedure] {' > '.join(incident_response)}",
    ]

    for item in batch_table_mapping:
        lines.append(
            f"[batch_table_mapping] {item.get('batch_group', '')} => {item.get('mapping', '')}"
        )

    for item in query_parameters:
        lines.append(
            f"[query_parameter] {item.get('name', '')} = {item.get('description', '')}"
        )

    content = build_text_block(title="에이전트 최적화 보강", lines=lines)

    docs.append(
        Document(
            page_content=content,
            metadata={
                "doc_type": "agent_optimization",
                "title": "에이전트 최적화 보강",
            },
        )
    )

    return docs


def json_to_documents(data: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []
    docs.extend(extract_intent_documents(data))
    docs.extend(extract_conversation_flow_documents(data))
    docs.extend(extract_work_catalog_documents(data))
    docs.extend(extract_development_environment_documents(data))
    docs.extend(extract_operational_query_documents(data))
    docs.extend(extract_business_summary_documents(data))
    docs.extend(extract_agent_optimization_documents(data))
    return docs


# ==============================
# 청크 분할
# ==============================
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# ==============================
# 벡터DB 적재
# ==============================
def build_vectorstore(documents: List[Document], persist_dir: str, collection_name: str):
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("환경변수 UPSTAGE_API_KEY 가 설정되지 않았습니다.")

    embedding = UpstageEmbeddings(
        model=UPSTAGE_MODEL,
        api_key=api_key,
    )

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    return vectordb


# ==============================
# 테스트 검색
# ==============================
def run_test_queries(vectordb):
    test_cases = [
        "내 업무는 어떤게 있어?",
        "소득공제 업무 상세 설명해줘",
        "A은행 소득공제 배치 플로우 알려줘",
        "A은행 청구 테이블 리니지 보여줘",
        "오늘 장애현황 알려줘",
        "개발환경 구축 알려줘",
        "은행에 전달할 청구현황 이용내역서 현황 뽑아줘",
    ]

    for i, query in enumerate(test_cases, start=1):
        print("\n" + "=" * 80)
        print(f"[TEST {i}] query = {query}")
        print("=" * 80)

        results = vectordb.similarity_search(query, k=3)
        for idx, doc in enumerate(results, start=1):
            print(f"\n--- result {idx} ---")
            print("metadata:", doc.metadata)
            print("content:", doc.page_content[:800])


# ==============================
# 메인
# ==============================
def main():
    parser = argparse.ArgumentParser(description="handover_agent_ready.json -> Chroma ingest")
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH, help="입력 JSON 파일 경로")
    parser.add_argument("--persist_dir", type=str, default=PERSIST_DIR, help="Chroma 저장 경로")
    parser.add_argument("--collection_name", type=str, default=COLLECTION_NAME, help="Chroma collection 이름")
    parser.add_argument("--test", action="store_true", help="적재 후 테스트 검색 실행")
    args = parser.parse_args()

    print(f"[INFO] JSON 로드 시작: {args.json_path}")
    data = load_json(args.json_path)

    raw_docs = json_to_documents(data)
    print(f"[INFO] 원본 Document 수: {len(raw_docs)}")

    split_docs = split_documents(raw_docs)
    print(f"[INFO] 청크 분할 후 Document 수: {len(split_docs)}")

    vectordb = build_vectorstore(
        documents=split_docs,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    print("[INFO] Chroma 적재 완료")
    print(f"[INFO] collection_name = {args.collection_name}")
    print(f"[INFO] persist_directory = {args.persist_dir}")

    if args.test:
        run_test_queries(vectordb)


if __name__ == "__main__":
    main()