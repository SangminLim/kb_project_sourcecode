import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_upstage import UpstageEmbeddings


# ==============================
# 설정
# ==============================
# 실행 예시
#
# 1) Ollama 사용
#    python ingest.py --json_path ./handover_agent_ready.json --use_ollama 1
# python testcase.py --use_ollama 1
#
# 2) Upstage 사용
#    set UPSTAGE_API_KEY=발급받은키
#    python ingest.py --json_path ./handover_agent_ready.json --use_ollama 0
#
# 이 파일의 역할
# - handover_agent_ready.json 파일을 읽는다.
# - JSON 안의 각 섹션을 LangChain Document 리스트로 변환한다.
# - 긴 문서는 chunk 단위로 분할한다.
# - 임베딩을 생성한다.
# - Chroma 벡터DB에 실제로 영구 저장한다.

DEFAULT_JSON_PATH = "./handover_agent_ready.json"
DEFAULT_PERSIST_DIR = "./chroma"
DEFAULT_COLLECTION_NAME = "chroma-handover-agent-v1"

# 너무 긴 문서는 검색 효율이 떨어질 수 있어 chunk로 분할한다.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# 임베딩 모델 기본값
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
UPSTAGE_MODEL = "solar-embedding-1-large"


# ==============================
# 유틸 함수
# ==============================
def load_json(json_path: str) -> Dict[str, Any]:
    """
    JSON 파일을 읽어서 dict로 반환한다.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value: Any) -> str:
    """
    값이 None / dict / list / 기타 타입일 때
    안전하게 문자열로 변환한다.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_text_block(title: str, lines: List[str]) -> str:
    """
    title + 본문 lines를 하나의 검색용 텍스트 블록으로 합친다.

    page_content에 들어갈 실제 본문을 만드는 함수다.
    """
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    body = "\n".join(cleaned)
    return f"[title] {title}\n{body}"


def guess_domain_customer_from_title(title: str) -> Tuple[str, str]:
    """
    title 문자열만 보고 domain/customer를 추정한다.
    business_summary처럼 domain/customer가 직접 없을 때 보조적으로 사용한다.
    """
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

    return domain, customer


# ==============================
# JSON -> Document 변환 함수들
# ==============================
def extract_intent_documents(data: Dict[str, Any]) -> List[Document]:
    """
    intents 섹션을 intent_definition 문서들로 변환한다.
    """
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
                "[doc_type] intent_definition",
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
    """
    conversation_flow 섹션을 conversation_flow 문서들로 변환한다.
    """
    docs: List[Document] = []

    flow_map = data.get("conversation_flow", {})
    for intent_name, flow_info in flow_map.items():
        answer_policy = flow_info.get("answer_policy", "")
        suggested_followups = flow_info.get("suggested_followups", [])

        content = build_text_block(
            title=f"대화 흐름 - {intent_name}",
            lines=[
                "[doc_type] conversation_flow",
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
    """
    work_catalog 섹션을 여러 종류의 문서로 변환한다.

    생성 문서 예시:
    - work_summary
    - batch_job
    - batch_flow
    - table_relation
    - table_lineage
    """
    docs: List[Document] = []

    work_catalog = data.get("work_catalog", [])
    for item in work_catalog:
        domain = item.get("domain", "")
        customer = item.get("customer", "")
        title = item.get("title", "")
        source = item.get("source", "")
        summary = item.get("summary", "")

        # 1) 업무 전체 요약
        summary_content = build_text_block(
            title=title,
            lines=[
                "[doc_type] work_summary",
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

        # 2) 배치 플로우 + 배치 job
        batch_flow = item.get("batch_flow", {})
        execution_rule = batch_flow.get("execution_rule", "")
        stages = batch_flow.get("stages", [])

        stage_lines = [
            "[doc_type] batch_flow",
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

            stage_lines.append(f"[stage] {stage_name} / {name} / parallel={parallel}")

            for job in jobs:
                job_id = job.get("job_id", "")
                desc = job.get("description", "")

                stage_lines.append(f"  - [job_id] {job_id} / [description] {desc}")

                # 개별 배치 작업 문서
                job_content = build_text_block(
                    title=f"{title} - {job_id}",
                    lines=[
                        "[doc_type] batch_job",
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

        # 3) 테이블 리니지 + 테이블 관계
        table_lineage = item.get("table_lineage", {})
        source_tables = table_lineage.get("source_tables", [])
        intermediate_tables = table_lineage.get("intermediate_tables", [])
        result_tables = table_lineage.get("result_tables", [])
        relations = table_lineage.get("relations", [])

        lineage_lines = [
            "[doc_type] table_lineage",
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

            # 개별 테이블 관계 문서
            rel_content = build_text_block(
                title=f"{title} - 테이블 관계 {src} -> {dst}",
                lines=[
                    "[doc_type] table_relation",
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
    """
    development_environment 섹션을 문서로 변환한다.
    """
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
        "[doc_type] development_environment",
        f"[source] {source}",
        f"[summary] {summary}",
        f"[components] {safe_str(components)}",
        f"[paths] {safe_str(paths)}",
    ]

    for step in installation_steps:
        lines.append(f"[installation_step] {step.get('step', '')} / {step.get('name', '')}")
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
    """
    operational_queries 섹션을 문서로 변환한다.
    """
    docs: List[Document] = []

    operational_queries = data.get("operational_queries", {})
    for query_name, item in operational_queries.items():
        content = build_text_block(
            title=f"운영 조회 - {query_name}",
            lines=[
                "[doc_type] operational_query",
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
    """
    business_summary 섹션을 문서로 변환한다.
    """
    docs: List[Document] = []

    for item in data.get("business_summary", []):
        title = item.get("title", "")
        summary = item.get("summary", "")

        domain, customer = guess_domain_customer_from_title(title)

        content = build_text_block(
            title=title,
            lines=[
                "[doc_type] business_summary",
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
    """
    agent_optimization 섹션을 문서로 변환한다.
    """
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
        "[doc_type] agent_optimization",
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
    """
    JSON 전체를 순회하면서 각 섹션별 문서를 모두 모아 반환한다.
    """
    docs: List[Document] = []
    docs.extend(extract_intent_documents(data))
    docs.extend(extract_conversation_flow_documents(data))
    docs.extend(extract_work_catalog_documents(data))
    docs.extend(extract_development_environment_documents(data))
    docs.extend(extract_operational_query_documents(data))
    docs.extend(extract_business_summary_documents(data))
    docs.extend(extract_agent_optimization_documents(data))
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    긴 문서를 chunk 단위로 분할한다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_embeddings(
    use_ollama: int = 1,
    ollama_model: str = OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    upstage_model: str = UPSTAGE_MODEL,
):
    """
    사용할 임베딩 객체를 생성한다.
    """
    if use_ollama == 1:
        print("[INFO] Embedding Provider = Ollama")
        return OllamaEmbeddings(
            model=ollama_model,
            base_url=ollama_base_url,
        )

    print("[INFO] Embedding Provider = Upstage")
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")

    return UpstageEmbeddings(
        model=upstage_model,
        api_key=api_key,
    )


def build_vectorstore(
    documents: List[Document],
    persist_dir: str,
    collection_name: str,
    use_ollama: int = 1,
    ollama_model: str = OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    upstage_model: str = UPSTAGE_MODEL,
) -> Chroma:
    """
    문서를 실제 Chroma 벡터DB에 저장한다.

    중요:
    - PersistentClient를 사용해 디스크에 실제 저장되도록 한다.
    - 기존 컬렉션이 있으면 삭제 후 다시 만든다.
      (테스트 중 중복 적재 방지용)
    """
    embeddings = get_embeddings(
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        upstage_model=upstage_model,
    )

    # 디스크에 실제 영구 저장되는 Chroma 클라이언트
    client = chromadb.PersistentClient(path=persist_dir)

    # 기존 동일 컬렉션이 있으면 삭제
    # 이유:
    # - 테스트할 때 중복 적재를 막기 위해
    # - 새 JSON 기준으로 깔끔하게 다시 적재하기 위해
    try:
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
            print(f"[INFO] 기존 collection 삭제: {collection_name}")
    except Exception as e:
        print(f"[WARN] 기존 collection 삭제 중 예외(무시 가능): {e}")

    # LangChain Chroma 래퍼 생성
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # 실제 문서 저장
    vectordb.add_documents(documents)

    # 저장 확인용 count 출력
    try:
        print(f"[DEBUG] 저장 후 collection count = {vectordb._collection.count()}")
    except Exception as e:
        print(f"[ERROR] 저장 후 count 확인 실패: {e}")

    return vectordb


def ingest_json_to_chroma(
    json_path: str = DEFAULT_JSON_PATH,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    use_ollama: int = 1,
    ollama_model: str = OLLAMA_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    upstage_model: str = UPSTAGE_MODEL,
) -> Dict[str, Any]:
    """
    JSON -> Document 변환 -> chunk 분할 -> Chroma 저장
    전체 ingest 과정을 한 번에 수행한다.
    """
    print(f"[INFO] JSON 로드 시작: {json_path}")
    data = load_json(json_path)

    raw_docs = json_to_documents(data)
    print(f"[INFO] 원본 Document 수: {len(raw_docs)}")

    split_docs = split_documents(raw_docs)
    print(f"[INFO] 청크 분할 후 Document 수: {len(split_docs)}")

    vectordb = build_vectorstore(
        documents=split_docs,
        persist_dir=persist_dir,
        collection_name=collection_name,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        upstage_model=upstage_model,
    )

    print("[INFO] Chroma 적재 완료")
    print(f"[INFO] collection_name = {collection_name}")
    print(f"[INFO] persist_directory = {persist_dir}")
    print(f"[INFO] use_ollama = {use_ollama}")

    if use_ollama == 1:
        print(f"[INFO] ollama_model = {ollama_model}")
        print(f"[INFO] ollama_base_url = {ollama_base_url}")
    else:
        print(f"[INFO] upstage_model = {upstage_model}")

    return {
        "vectordb": vectordb,
        "raw_docs": raw_docs,
        "split_docs": split_docs,
    }


def main() -> None:
    """
    CLI 실행 진입점.
    """
    parser = argparse.ArgumentParser(description="handover_agent_ready.json -> Chroma ingest")
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH, help="입력 JSON 파일 경로")
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR, help="Chroma 저장 경로")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME, help="Chroma collection 이름")
    parser.add_argument("--use_ollama", type=int, default=1, help="1이면 Ollama, 0이면 Upstage")
    parser.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL, help="Ollama embedding 모델명")
    parser.add_argument("--ollama_base_url", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--upstage_model", type=str, default=UPSTAGE_MODEL, help="Upstage embedding 모델명")
    args = parser.parse_args()

    ingest_json_to_chroma(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        upstage_model=args.upstage_model,
    )


if __name__ == "__main__":
    main()