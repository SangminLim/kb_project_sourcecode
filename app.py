from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph

load_dotenv()

from llm import ChatConfig, HandoverAgent, ollama_generate
from realtime_query_service import RealtimeQueryService

PAGE_TITLE = "업무 인수인계 에이전트"
PAGE_ICON = "🤖"

JSON_PATH = os.getenv("JSON_PATH", "ingest/handover_improved.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "handover_agent")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
BILLING_QUERY_ID = "billing_monthly_amount"
TODAY_INCIDENTS_QUERY_ID = "today_incidents"


@st.cache_resource
def get_agent() -> HandoverAgent:
    return HandoverAgent(
        json_path=JSON_PATH,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )


def init_session_state() -> None:
    if "message_list" not in st.session_state:
        st.session_state.message_list = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "evaluation_mode" not in st.session_state:
        st.session_state.evaluation_mode = True


def build_chat_history(message_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in message_list:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


def get_recent_questions(message_list: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    questions: List[str] = []
    seen = set()
    for item in reversed(message_list):
        if item.get("role") != "user":
            continue
        content = (item.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        questions.append(content)
        if len(questions) >= limit:
            break
    return questions


def shorten_text(text: str, max_len: int = 40) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def draw_graphviz_graph(graph_data: Dict[str, Any], graph_kind: str = "flow") -> None:
    dot = Digraph()
    title = graph_data.get("title", "")
    if title:
        st.subheader(title)
    if graph_kind == "lineage":
        for table in graph_data.get("tables", []):
            node_id = str(table.get("id", ""))
            dot.node(node_id, node_id)
        for edge in graph_data.get("edges", []):
            src = str(edge.get("from", ""))
            dst = str(edge.get("to", ""))
            if src and dst:
                dot.edge(src, dst)
    else:
        for node in graph_data.get("nodes", []):
            node_id = str(node.get("id", ""))
            node_label = str(node.get("label", "")).strip()
            label = f"{node_id}\n({node_label})" if node_label and node_label != node_id else node_id
            dot.node(node_id, label)
        for edge in graph_data.get("edges", []):
            src = str(edge.get("from", ""))
            dst = str(edge.get("to", ""))
            if src and dst:
                dot.edge(src, dst)
    st.graphviz_chart(dot, use_container_width=True)


@st.cache_resource
def get_realtime_service() -> RealtimeQueryService | None:
    if not DATABASE_URL:
        return None
    return RealtimeQueryService(DATABASE_URL)


def fetch_realtime_dataframe(query_meta: Dict[str, Any]) -> pd.DataFrame:
    service = get_realtime_service()
    if service is None:
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다. .env에 DATABASE_URL을 설정하세요.")
    return service.fetch_dataframe(query_meta)


def build_incident_summary_prompt(rows_json: str) -> str:
    return f"""
너는 금융 배치 운영 담당자를 돕는 실무형 어시스턴트다.
아래는 오늘 장애현황 조회 결과다.

[장애현황 데이터]
{rows_json}

규칙:
- 데이터에 있는 내용만 사용한다
- 반드시 한국어로만 답한다
- 아래 순서로만 간단히 정리한다
1. 전체 장애 건수
2. 장애 배치명 목록
3. 주요 오류 원인 요약
4. 확인 필요사항
- 데이터가 비어 있으면 '오늘 장애는 없습니다'라고 답한다
""".strip()


def summarize_billing_dataframe(df: pd.DataFrame, x_field: str, y_field: str) -> Dict[str, Any]:
    if df.empty:
        return {"row_count": 0}

    work_df = df.copy()
    work_df[y_field] = pd.to_numeric(work_df[y_field], errors="coerce").fillna(0)
    work_df = work_df.sort_values(by=x_field).reset_index(drop=True)

    max_row = work_df.loc[work_df[y_field].idxmax()]
    min_row = work_df.loc[work_df[y_field].idxmin()]
    latest_row = work_df.iloc[-1]
    prev_row = work_df.iloc[-2] if len(work_df) >= 2 else None

    change_rate_pct = None
    if prev_row is not None and float(prev_row[y_field]) != 0:
        change_rate_pct = round(((float(latest_row[y_field]) - float(prev_row[y_field])) / float(prev_row[y_field])) * 100, 2)

    return {
        "row_count": int(len(work_df)),
        "total_amount": float(work_df[y_field].sum()),
        "max_period": str(max_row[x_field]),
        "max_amount": float(max_row[y_field]),
        "min_period": str(min_row[x_field]),
        "min_amount": float(min_row[y_field]),
        "latest_period": str(latest_row[x_field]),
        "latest_amount": float(latest_row[y_field]),
        "previous_period": str(prev_row[x_field]) if prev_row is not None else None,
        "previous_amount": float(prev_row[y_field]) if prev_row is not None else None,
        "change_rate_pct": change_rate_pct,
    }


def build_billing_summary_prompt(query_meta: Dict[str, Any], summary_payload: Dict[str, Any]) -> str:
    title = query_meta.get("title", "조회 결과")
    summary_prompt = (query_meta.get("summary_prompt") or "").strip()
    payload_json = json.dumps(summary_payload, ensure_ascii=False)

    if summary_prompt:
        return f"""
너는 금융 데이터 분석을 돕는 실무형 어시스턴트다.

[조회 제목]
{title}

[요약 데이터]
{payload_json}

추가 지시:
{summary_prompt}

규칙:
- 요약 데이터에 있는 내용만 사용한다
- 반드시 한국어로만 답한다
- 4줄 이내로 간단히 정리한다
- 과장하거나 추측하지 않는다
""".strip()

    return f"""
너는 금융 데이터 분석을 돕는 실무형 어시스턴트다.

[조회 제목]
{title}

[요약 데이터]
{payload_json}

규칙:
- 요약 데이터에 있는 내용만 사용한다
- 반드시 한국어로만 답한다
- 아래 순서로만 4줄 이내로 정리한다
1. 전체 흐름 요약
2. 최고 금액 구간
3. 최저 금액 구간
4. 최근 구간의 증감 포인트
- 마지막 줄에는 데이터 패턴(증가/감소/변동성/특정 구간 집중)을 한 문장으로 해석한다
- change_rate_pct가 없으면 최근 증감은 생략 가능하다
- 추측하지 않는다
""".strip()


def generate_incident_summary(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    rows_json = df.to_json(orient="records", force_ascii=False)
    summary_prompt = build_incident_summary_prompt(rows_json)
    return ollama_generate(
        prompt=summary_prompt,
        system_prompt="반드시 한국어로만, 데이터 기반으로 요약해라.",
        config=ChatConfig(),
    )


def generate_billing_summary(query_meta: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    summary_payload = summarize_billing_dataframe(
        df,
        x_field=query_meta.get("x_field", "billing_month"),
        y_field=query_meta.get("y_field", "amount"),
    )
    summary_prompt = build_billing_summary_prompt(query_meta, summary_payload)
    return ollama_generate(
        prompt=summary_prompt,
        system_prompt="반드시 한국어로만, 요약 데이터 기반으로 간단히 해석해라.",
        config=ChatConfig(),
    )


def _render_bullets(items: List[str]) -> None:
    for item in items or []:
        st.markdown(f"- {item}")


def dataframe_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    safe_df = df.where(pd.notnull(df), None)
    return {
        "columns": safe_df.columns.tolist(),
        "rows": safe_df.to_dict(orient="records"),
    }


def payload_to_dataframe(payload: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("rows", [])
    columns = payload.get("columns", [])
    return pd.DataFrame(rows, columns=columns)


def build_realtime_payload(query_meta: Dict[str, Any], render_type: str, realtime_mode: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query_id": query_meta.get("query_id"),
        "render_type": render_type,
        "summary": None,
        "dataframe": None,
        "empty_message": None,
        "error": None,
    }

    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as e:
        payload["error"] = str(e)
        return payload

    payload["dataframe"] = dataframe_to_payload(df)

    if df.empty:
        if query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID:
            payload["empty_message"] = "오늘 장애는 없습니다."
        else:
            payload["empty_message"] = "조회 결과가 없습니다."
        return payload

    try:
        query_id = query_meta.get("query_id")
        if render_type == "table" and (realtime_mode == "incident_table_with_summary" or query_id == TODAY_INCIDENTS_QUERY_ID):
            payload["summary"] = generate_incident_summary(df)
        elif render_type == "chart" and query_id == BILLING_QUERY_ID:
            payload["summary"] = generate_billing_summary(query_meta, df)
    except Exception as e:
        payload["summary_error"] = str(e)

    return payload


def enrich_result_with_realtime_payload(result: Any) -> Any:
    if getattr(result, "render_type", None) not in {"table", "chart"} or not getattr(result, "query_meta", None):
        return result
    realtime_payload = build_realtime_payload(
        query_meta=result.query_meta,
        render_type=result.render_type,
        realtime_mode=getattr(result, "realtime_mode", None),
    )
    setattr(result, "realtime_payload", realtime_payload)
    return result


def render_overview_block(result: Any) -> None:
    data = getattr(result, "structured_data", None) or {}
    overview = data.get("overview", {}) if "overview" in data else data

    title = overview.get("title", "업무 개요")
    st.markdown(f"#### 📌 {title}")

    summary = (overview.get("summary") or "").strip()
    if summary:
        st.markdown("##### 🔹 핵심 요약")
        summary_lines = [s.strip() for s in summary.split(" / ") if s.strip()]
        if len(summary_lines) <= 1:
            st.markdown(f"- {summary}")
        else:
            _render_bullets(summary_lines)

    st.markdown("---")

    input_data = overview.get("input_data", [])
    if input_data:
        st.markdown("##### 🔹 주요 입력 데이터")
        _render_bullets(input_data)

    target_transactions = overview.get("target_transactions", [])
    if target_transactions:
        st.markdown("##### 🔹 주요 대상 거래")
        _render_bullets(target_transactions)

    exclusions = overview.get("exclusions", [])
    if exclusions:
        st.markdown("##### 🔹 제외 및 보정 항목")
        _render_bullets(exclusions)

    outputs = overview.get("outputs", [])
    if outputs:
        st.markdown("##### 🔹 최종 산출물")
        _render_bullets(outputs)

    key_points = overview.get("key_points", [])
    if key_points:
        st.markdown("##### ⭐ 핵심 포인트")
        _render_bullets(key_points)


def render_batch_process_block(result: Any) -> None:
    data = getattr(result, "structured_data", None) or {}
    batch_process = data.get("batch_process", {}) if "batch_process" in data else data
    title = batch_process.get("title", "배치 프로세스")
    st.markdown(f"#### 📌 {title}")

    steps = batch_process.get("steps", [])
    key_jobs: List[str] = []
    for step in steps:
        key_jobs.extend(step.get("key_jobs", []))

    if key_jobs:
        st.markdown("##### ⭐ 핵심 배치")
        _render_bullets([f"`{job}`" for job in key_jobs])

    if result.answer:
        st.markdown("##### 🔹 핵심 흐름")
        answer_text = result.answer.replace("핵심 흐름은 ", "").replace(" 순입니다.", "").strip()
        flow_items = [item.strip() for item in answer_text.split("→") if item.strip()]
        if flow_items:
            _render_bullets(flow_items)
        else:
            st.markdown(f"- {result.answer}")

    st.markdown("---")

    for step in steps:
        execution = step.get("execution", "")
        execution_kr = "병렬" if execution == "parallel" else "순차" if execution == "sequential" else execution

        st.markdown(f"##### STEP {step.get('step')}. {step.get('name')} ({execution_kr})")

        description = (step.get("description") or "").strip()
        if description:
            st.markdown(f"👉 {description}")

        for job in step.get("jobs", []):
            st.markdown(f"- `{job.get('job_id', '')}`: {job.get('description', '')}")

        st.markdown("")


def render_graph_summary(result: Any) -> None:
    if result.answer:
        st.write(result.answer)


def render_chart_summary(result: Any) -> None:
    if result.answer:
        st.write(result.answer)


def render_table_summary(result: Any) -> None:
    if result.answer:
        st.write(result.answer)


def render_chart(query_meta: Dict[str, Any], realtime_payload: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None) -> None:
    st.subheader(query_meta.get("title", "차트"))
    payload = realtime_payload or {}
    if payload.get("error"):
        st.warning(f"DB 조회 실패: {payload['error']}")
        st.info("DATABASE_URL과 실제 테이블/컬럼 구성을 확인하세요.")
        return

    empty_message = payload.get("empty_message")
    if empty_message:
        st.info(empty_message)
        return

    df = payload_to_dataframe(payload.get("dataframe"))
    if df.empty:
        st.info("조회 결과가 없습니다.")
        return

    x_field = query_meta.get("x_field", "billing_month")
    y_field = query_meta.get("y_field", "amount")
    title = query_meta.get("title", "차트")

    chart_df = df.copy()
    chart_df[x_field] = (
        chart_df[x_field]
        .astype(str)
        .str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
    )

    fig = px.bar(chart_df, x=x_field, y=y_field, title=title)
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=",")
    fig.update_layout(
        xaxis_title="년월",
        yaxis_title="금액",
    )
    chart_key = f"chart_{message_id}" if message_id else None
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    summary = payload.get("summary")
    if summary:
        st.markdown("##### 📈 데이터 요약")
        st.write(summary)
    elif payload.get("summary_error"):
        st.warning(f"LLM 차트 요약 생성 실패: {payload['summary_error']}")

    with st.expander("조회 데이터 보기"):
        st.dataframe(df, use_container_width=True)


def render_table(query_meta: Dict[str, Any], realtime_mode: str | None = None, realtime_payload: Optional[Dict[str, Any]] = None) -> None:
    st.subheader(query_meta.get("title", "테이블"))
    payload = realtime_payload or {}
    if payload.get("error"):
        st.warning(f"DB 조회 실패: {payload['error']}")
        st.info("DATABASE_URL과 실제 테이블/컬럼 구성을 확인하세요.")
        return

    empty_message = payload.get("empty_message")
    if empty_message:
        st.info(empty_message)
        return

    df = payload_to_dataframe(payload.get("dataframe"))
    if df.empty:
        if query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID:
            st.info("오늘 장애는 없습니다.")
        else:
            st.info("조회 결과가 없습니다.")
        return

    summary = payload.get("summary")
    if summary:
        st.write(summary)
    elif payload.get("summary_error") and (realtime_mode == "incident_table_with_summary" or query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID):
        st.warning(f"LLM 요약 생성 실패: {payload['summary_error']}")

    st.dataframe(df, use_container_width=True)




def _extract_where_filter(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "where_filter =" in log:
            return log.split("where_filter =", 1)[1].strip()
    return "(where 조건 없음 또는 default 검색)"


def _extract_search_query(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "search_query =" in log:
            return log.split("search_query =", 1)[1].strip()
    return ""


def build_used_json_view(result: Any) -> Dict[str, Any]:
    intent = getattr(result, "intent", None)
    structured_data = getattr(result, "structured_data", None) or {}
    graph_data = getattr(result, "graph_data", None) or {}
    query_meta = getattr(result, "query_meta", None) or {}

    if intent == "overview":
        overview = structured_data.get("overview", structured_data)
        return {
            "json_path": "domains[].systems[].overview",
            "used_fields": {
                "title": overview.get("title"),
                "summary": overview.get("summary"),
                "content": overview.get("content"),
                "input_data": overview.get("input_data", []),
                "target_transactions": overview.get("target_transactions", []),
                "exclusions": overview.get("exclusions", []),
                "outputs": overview.get("outputs", []),
                "key_points": overview.get("key_points", []),
            },
        }

    if intent == "batch_process":
        batch_process = structured_data.get("batch_process", structured_data)
        return {
            "json_path": "domains[].systems[].batch_process",
            "used_fields": {
                "title": batch_process.get("title"),
                "steps": batch_process.get("steps", []),
            },
        }

    if intent == "batch_flow":
        return {
            "json_path": "domains[].systems[].batch_flow",
            "used_fields": {
                "title": graph_data.get("title"),
                "summary": graph_data.get("summary"),
                "start_nodes": graph_data.get("start_nodes", []),
                "highlight_nodes": graph_data.get("highlight_nodes", []),
                "end_nodes": graph_data.get("end_nodes", []),
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
            },
        }

    if intent == "table_lineage":
        return {
            "json_path": "domains[].systems[].table_lineage",
            "used_fields": {
                "title": graph_data.get("title"),
                "summary": graph_data.get("summary"),
                "highlight_tables": graph_data.get("highlight_tables", []),
                "source_tables": graph_data.get("source_tables", []),
                "result_tables": graph_data.get("result_tables", []),
                "tables": graph_data.get("tables", []),
                "edges": graph_data.get("edges", []),
            },
        }

    if query_meta:
        return {"json_path": "realtime_queries[]", "used_fields": query_meta}

    return {"json_path": "(확인 불가)", "used_fields": {}}


def build_evaluation_checks(result: Any) -> List[str]:
    """
    평가용 문구 생성
    - 시스템별 RAG 질문과 realtime_query 질문을 분리해서 평가한다.
    - realtime_query는 특정 system_id 기준 검색이 아니므로 "다른 시스템 source" 검사를 하지 않는다.
    """
    system_id = getattr(result, "system_id", None)
    intent = getattr(result, "intent", None)
    sources = getattr(result, "sources", []) or []
    checks: List[str] = []

    realtime_intents = {"billing_monthly_amount", "today_incidents"}
    system_intents = {"overview", "batch_process", "batch_flow", "table_lineage"}

    if intent in realtime_intents:
        wrong_realtime_sources = [
            s for s in sources
            if s.get("section") and s.get("section") != "realtime_query"
        ]
        checks.append(
            "✅ realtime_query 기준으로 조회 정의가 선택됨"
            if not wrong_realtime_sources
            else "⚠️ realtime_query가 아닌 source가 포함됨"
        )

        query_meta = getattr(result, "query_meta", None) or {}
        query_id = query_meta.get("query_id")
        wrong_query_sources = [
            s for s in sources
            if s.get("query_id") and query_id and s.get("query_id") != query_id
        ]
        checks.append(
            f"✅ query_id 기준으로 {query_id or '대상 조회'}가 정확히 매칭됨"
            if not wrong_query_sources
            else "⚠️ 다른 query_id source가 섞였는지 확인 필요"
        )

    elif intent in system_intents:
        if system_id:
            mixed_sources = [
                s for s in sources
                if s.get("system_id") and s.get("system_id") != system_id
            ]
            checks.append(
                "✅ system_id 기준으로 대상 시스템이 분리됨"
                if not mixed_sources
                else "⚠️ 다른 시스템 source가 섞였는지 확인 필요"
            )
        else:
            checks.append("⚠️ 시스템별 질문인데 system_id가 확인되지 않음")

        wrong_section = [
            s for s in sources
            if s.get("section") and s.get("section") != intent
        ]
        checks.append(
            "✅ intent에 맞는 section을 사용함"
            if not wrong_section
            else "⚠️ 의도와 다른 section source가 포함됨"
        )

    else:
        checks.append("ℹ️ 기본 검색 질문으로 처리됨")

    if getattr(result, "structured_data", None) or getattr(result, "graph_data", None) or getattr(result, "query_meta", None):
        checks.append("✅ 답변 근거가 원본 JSON 구조와 연결됨")

    checks.append(
        f"✅ retrieval 근거 {len(sources)}건 확인 가능"
        if sources
        else "ℹ️ 구조화 JSON 직접 렌더링 중심이라 retrieval source가 없을 수 있음"
    )
    return checks

def render_evaluation_panel(result: Any) -> None:
    if not st.session_state.evaluation_mode:
        return

    used_json = build_used_json_view(result)
    sources = getattr(result, "sources", []) or []
    debug_logs = getattr(result, "debug_logs", []) or []

    with st.expander("📊 평가용 근거 확인", expanded=True):
        st.markdown("##### 1) 질문 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
        })

        st.markdown("##### 2) Builder 검색 조건")
        st.code(_extract_where_filter(debug_logs), language="python")

        search_query = _extract_search_query(debug_logs)
        if search_query:
            st.markdown("##### 3) 실제 검색 질문")
            st.code(search_query, language="text")

        st.markdown("##### 4) Vector DB에서 가져온 근거 chunk")
        if sources:
            st.dataframe(pd.DataFrame(sources), use_container_width=True)
        else:
            st.info("검색 chunk가 없거나, 구조화 JSON을 직접 사용한 응답입니다.")

        st.markdown("##### 5) 답변에 사용된 원본 JSON 데이터")
        st.caption(f"JSON 위치: {used_json.get('json_path')}")
        st.json(used_json.get("used_fields", {}))

        st.markdown("##### 6) 간단 평가")
        for check in build_evaluation_checks(result):
            st.markdown(f"- {check}")

def render_debug_blocks(*, normalized_question: str = "", rewritten_question: str = "", system_id: str | None = None, intent: str | None = None, render_type: str | None = None, sources: List[Dict[str, Any]] | None = None, debug_logs: List[str] | None = None,) -> None:
    if not st.session_state.debug_mode:
        return
    with st.expander("디버그 정보"):
        st.json({"normalized_question": normalized_question, "rewritten_question": rewritten_question, "system_id": system_id, "intent": intent, "render_type": render_type, "sources": sources or [],})
    with st.expander("질문 해석 로그 보기"):
        logs = debug_logs or []
        if logs:
            for log in logs:
                st.code(log)
        else:
            st.info("debug_logs가 없습니다. llm.py가 debug 통합 버전인지 확인하세요.")


def render_agent_result(result: Any) -> None:
    with st.chat_message("assistant"):
        if result.intent == "overview" and getattr(result, "structured_data", None):
            render_overview_block(result)
        elif result.intent == "batch_process" and getattr(result, "structured_data", None):
            render_batch_process_block(result)
        elif result.render_type == "graph" and result.graph_data:
            render_graph_summary(result)
            if result.intent == "table_lineage":
                draw_graphviz_graph(result.graph_data, graph_kind="lineage")
            else:
                draw_graphviz_graph(result.graph_data, graph_kind="flow")
        elif result.render_type == "chart" and result.query_meta:
            render_chart_summary(result)
            render_chart(result.query_meta, getattr(result, "realtime_payload", None), getattr(result, "message_id", None))
        elif result.render_type == "table" and result.query_meta:
            render_table_summary(result)
            render_table(result.query_meta, getattr(result, "realtime_mode", None), getattr(result, "realtime_payload", None))
        else:
            st.write(result.answer)

        render_evaluation_panel(result)

        render_debug_blocks(
            normalized_question=getattr(result, "normalized_question", ""),
            rewritten_question=getattr(result, "rewritten_question", ""),
            system_id=getattr(result, "system_id", None),
            intent=getattr(result, "intent", None),
            render_type=getattr(result, "render_type", None),
            sources=getattr(result, "sources", []),
            debug_logs=getattr(result, "debug_logs", []),
        )


def build_history_result(message: Dict[str, Any]) -> Any:
    return SimpleNamespace(
        answer=message.get("content", ""),
        intent=message.get("intent"),
        render_type=message.get("render_type"),
        graph_data=message.get("graph_data"),
        query_meta=message.get("query_meta"),
        realtime_mode=message.get("realtime_mode"),
        structured_data=message.get("structured_data"),
        realtime_payload=message.get("realtime_payload"),
        normalized_question=message.get("normalized_question", ""),
        rewritten_question=message.get("rewritten_question", ""),
        system_id=message.get("system_id"),
        sources=message.get("sources", []),
        debug_logs=message.get("debug_logs", []),
        message_id=message.get("message_id"),
    )


def render_history_messages() -> None:
    for message in st.session_state.message_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            if role == "assistant":
                result = build_history_result(message)

                if result.intent == "overview" and result.structured_data:
                    render_overview_block(result)
                elif result.intent == "batch_process" and result.structured_data:
                    render_batch_process_block(result)
                elif result.render_type == "graph" and result.graph_data:
                    render_graph_summary(result)
                    if result.intent == "table_lineage":
                        draw_graphviz_graph(result.graph_data, graph_kind="lineage")
                    else:
                        draw_graphviz_graph(result.graph_data, graph_kind="flow")
                elif result.render_type == "chart" and result.query_meta:
                    render_chart_summary(result)
                    render_chart(result.query_meta, result.realtime_payload, result.message_id)
                elif result.render_type == "table" and result.query_meta:
                    render_table_summary(result)
                    render_table(result.query_meta, result.realtime_mode, result.realtime_payload)
                else:
                    st.write(content)

                render_debug_blocks(
                    normalized_question=result.normalized_question,
                    rewritten_question=result.rewritten_question,
                    system_id=result.system_id,
                    intent=result.intent,
                    render_type=result.render_type,
                    sources=result.sources,
                    debug_logs=result.debug_logs,
                )
            else:
                st.write(content)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()
    st.title("🤖 업무 인수인계 에이전트")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
        st.session_state.evaluation_mode = st.checkbox("평가용 근거 보기", value=st.session_state.evaluation_mode)
        st.session_state.debug_mode = st.checkbox("디버그 정보 보기", value=st.session_state.debug_mode)
        st.markdown("### 최근 질문")
        recent_questions = get_recent_questions(st.session_state.message_list, limit=10)
        if recent_questions:
            for idx, q in enumerate(recent_questions, start=1):
                label = shorten_text(q, 40)
                if st.button(label, key=f"recent_q_{idx}", use_container_width=True):
                    st.session_state.pending_question = q
        else:
            st.caption("아직 질문 내역이 없습니다.")

    render_history_messages()

    chat_input_question = st.chat_input(placeholder="예) KKK은행 소득공제 배치 프로세스를 설명해줘")
    user_question = st.session_state.pending_question or chat_input_question
    st.session_state.pending_question = None

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})
        agent = get_agent()
        chat_history = build_chat_history(st.session_state.message_list[:-1])
        with st.spinner("답변을 생성하는 중입니다..."):
            result = agent.answer_question(question=user_question, chat_history=chat_history)
            result = enrich_result_with_realtime_payload(result)
            current_message_id = str(uuid.uuid4())
            setattr(result, "message_id", current_message_id)
            render_agent_result(result)
        st.session_state.message_list.append({
            "message_id": current_message_id,
            "role": "assistant",
            "content": result.answer,
            "intent": result.intent,
            "render_type": result.render_type,
            "graph_data": result.graph_data,
            "query_meta": result.query_meta,
            "realtime_mode": getattr(result, "realtime_mode", None),
            "realtime_payload": getattr(result, "realtime_payload", None),
            "sources": result.sources,
            "structured_data": getattr(result, "structured_data", None),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "debug_logs": getattr(result, "debug_logs", []),
        })


if __name__ == "__main__":
    main()
