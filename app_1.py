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

from llm import ChatConfig, HandoverAgent, ollama_generate, apply_dictionary_rewrite, detect_intent, get_llm_engine_name, get_langchain_feature_flags
from realtime_query_service import RealtimeQueryService
from batch_dev import BatchDevAgent

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

def _get_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _safe_cell(row: pd.Series, column: Optional[str], default: str = "") -> str:
    if not column:
        return default
    value = row.get(column)
    if pd.isna(value) or value is None:
        return default
    text = str(value).strip()
    return text if text else default

def generate_incident_summary(df: pd.DataFrame) -> Optional[str]:
    """
    장애현황은 정형 데이터이므로 LLM 호출 없이 코드로 요약한다.
    - Ollama timeout 방지
    - 조회 결과가 있을 때 '오늘 장애는 없습니다'가 섞여 출력되는 문제 방지
    - SQL alias가 영문/한글이어도 동작하도록 주요 컬럼 후보를 함께 처리
    """
    if df.empty:
        return None

    batch_col = _get_first_existing_column(df, ["batch_name", "배치명"])
    error_code_col = _get_first_existing_column(df, ["error_code", "오류코드"])
    error_msg_col = _get_first_existing_column(df, ["error_message", "오류메시지", "오류내용"])
    action_detail_col = _get_first_existing_column(df, ["action_detail", "조치방법", "조치내용"])
    action_owner_col = _get_first_existing_column(df, ["action_owner", "담당자", "조치담당자"])

    total_count = len(df)
    batch_names = []
    error_messages = []
    action_lines = []
    owners = []

    for _, row in df.iterrows():
        batch_name = _safe_cell(row, batch_col, "배치명 없음")
        error_code = _safe_cell(row, error_code_col, "오류코드 없음")
        error_message = _safe_cell(row, error_msg_col, "오류 메시지 없음")
        action_detail = _safe_cell(row, action_detail_col, "등록된 조치 방법 없음")
        action_owner = _safe_cell(row, action_owner_col, "담당자 미지정")

        batch_names.append(batch_name)
        error_messages.append(error_message)
        action_lines.append(f"- {batch_name} ({error_code}): {action_detail}")
        owners.append(action_owner)

    unique_error_messages = list(dict.fromkeys(error_messages))
    unique_owners = [owner for owner in dict.fromkeys(owners) if owner != "담당자 미지정"]

    lines = [
        "장애 현황 조회 결과",
        "",
        f"전체 장애 건수: {total_count}건",
        "",
        "장애 배치명 목록:",
        *[f"- {name}" for name in batch_names],
        "",
        "주요 오류 원인 요약:",
        f"- {', '.join(unique_error_messages)}",
        "",
        "조치 방법 요약:",
        *action_lines,
        "",
        f"담당자: {', '.join(unique_owners) if unique_owners else '담당자 미지정'}",
        "",
        "확인 필요사항: 조치 후 배치 재실행 여부와 후속 배치 영향도를 확인하세요.",
    ]

    return "\n".join(lines)

def format_krw(amount: Any) -> str:
    """DB 금액은 원 단위 정수로 관리하고, 화면/요약에서만 사람이 읽기 좋은 형태로 변환한다."""
    amount_int = int(round(float(amount or 0)))
    if amount_int != 0 and amount_int % 10000 == 0:
        return f"{amount_int // 10000:,}만 원"
    return f"{amount_int:,}원"

def _billing_pattern_text(summary: Dict[str, Any]) -> str:
    """최근 증감률 기준으로 간단한 패턴 문구를 만든다."""
    change_rate_pct = summary.get("change_rate_pct")
    if change_rate_pct is None:
        return "데이터 패턴은 단일 구간이므로 증감 판단은 생략합니다."
    if change_rate_pct > 0:
        return "데이터 패턴은 최근 구간에서 증가하는 흐름입니다."
    if change_rate_pct < 0:
        return "데이터 패턴은 최근 구간에서 감소하는 흐름입니다."
    return "데이터 패턴은 최근 구간에서 전월과 동일한 흐름입니다."

def generate_billing_summary(query_meta: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    """
    청구 월별 금액 요약은 LLM이 숫자 단위를 오해하지 않도록 코드로 생성한다.
    - 계산은 summarize_billing_dataframe에서 수행
    - 금액 표시는 원 단위 기준으로 format_krw에서 수행
    """
    if df.empty:
        return None

    summary = summarize_billing_dataframe(
        df,
        x_field=query_meta.get("x_field", "billing_month"),
        y_field=query_meta.get("y_field", "amount"),
    )

    lines = [
        (
            f"전체 흐름 요약: 총 {summary['row_count']}개월치의 청구 이용내역서 월별 금액 조회 결과, "
            f"총액은 {format_krw(summary['total_amount'])}으로 확인됩니다."
        ),
        (
            f"최고 금액 구간: 최고 금액은 {format_krw(summary['max_amount'])}으로, "
            f"{summary['max_period']}에 발생했습니다."
        ),
        (
            f"최저 금액 구간: 최저 금액은 {format_krw(summary['min_amount'])}으로, "
            f"{summary['min_period']}에 발생했습니다."
        ),
    ]

    if summary.get("change_rate_pct") is not None:
        lines.append(
            f"최근 구간의 증감 포인트: 최근 구간인 {summary['latest_period']}에는 "
            f"{format_krw(summary['latest_amount'])}으로, 전월 대비 {summary['change_rate_pct']}% 변동했습니다."
        )

    lines.append(_billing_pattern_text(summary))
    return "\n\n".join(lines)

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

def render_batch_development_evaluation_panel(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    batch_spec = payload.get("batch_spec", {}) or {}

    with st.expander("📊 배치개발 평가용 근거 확인", expanded=True):
        st.markdown("##### 1) 요청 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
            "success": payload.get("success"),
        })

        st.markdown("##### 2) 생성된 배치 명세")
        st.json({
            "batch_id": batch_spec.get("batch_id"),
            "batch_name": batch_spec.get("batch_name"),
            "batch_type": batch_spec.get("batch_type"),
            "source": batch_spec.get("source"),
            "target": batch_spec.get("target"),
        })

        st.markdown("##### 3) 사용한 ERWIN 메타")
        st.json(batch_spec.get("meta_source", {}))

        st.markdown("##### 4) 사용한 Rule / SQL Template")
        st.json(batch_spec.get("rule_source", {}))

        st.markdown("##### 5) 생성 SQL")
        st.code(batch_spec.get("sql", ""), language="sql")

        st.markdown("##### 6) 생성 파일")
        for file_path in payload.get("created_files", []):
            st.code(file_path, language="text")

        st.markdown("##### 7) 검증 결과")
        st.json({
            "warnings": payload.get("warnings", []),
            "errors": payload.get("errors", []),
        })

def render_evaluation_panel(result: Any) -> None:
    if not st.session_state.evaluation_mode:
        return

    if getattr(result, "intent", None) == "batch_development":
        render_batch_development_evaluation_panel(result)
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

def run_batch_development(user_question: str) -> Any:
    """
    배치 개발 요청은 기존 RAG 흐름과 분리해서 처리한다.
    기존 HandoverAgent/Chroma 검색 품질에 영향을 주지 않기 위한 별도 진입점이다.
    """
    dev_result = BatchDevAgent().run(user_question)
    answer = dev_result.message
    if dev_result.errors:
        answer = "배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요."
    return SimpleNamespace(
        answer=answer,
        intent="batch_development",
        render_type="batch_dev",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=apply_dictionary_rewrite(user_question),
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            "[BATCH_DEV 1] intent=batch_development",
            f"[BATCH_DEV 2] success={dev_result.success}",
            f"[BATCH_DEV 3] created_files={len(dev_result.created_files)}",
        ],
        batch_dev_result={
            "batch_spec": dev_result.batch_spec,
            "created_files": dev_result.created_files,
            "warnings": dev_result.warnings,
            "errors": dev_result.errors,
            "message": dev_result.message,
            "success": dev_result.success,
        },
    )

def render_batch_development_result(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    if not payload:
        st.warning("배치 개발 결과가 없습니다.")
        return

    success = payload.get("success")
    st.markdown("#### 🛠️ 배치 개발 결과")
    if success:
        st.success(payload.get("message", "배치 소스가 생성되었습니다."))
    else:
        st.error(payload.get("message", "배치 소스 생성에 실패했습니다."))

    errors = payload.get("errors") or []
    warnings = payload.get("warnings") or []
    if errors:
        st.markdown("##### ❌ 오류")
        for item in errors:
            st.markdown(f"- {item}")
    if warnings:
        st.markdown("##### ⚠️ 검토 필요")
        for item in warnings:
            st.markdown(f"- {item}")

    st.markdown("##### 생성된 batch_spec")
    st.json(payload.get("batch_spec", {}))

    created_files = payload.get("created_files") or []
    if created_files:
        st.markdown("##### 생성 파일")
        for file_path in created_files:
            st.code(file_path, language="text")

    st.info("운영 반영 전 query.sql, 컬럼, 인덱스, 검증조건, 파일 포맷을 개발자가 반드시 검토하세요.")

def render_agent_result(result: Any) -> None:
    with st.chat_message("assistant"):
        if result.intent == "batch_development":
            render_batch_development_result(result)
        elif result.intent == "overview" and getattr(result, "structured_data", None):
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
        batch_dev_result=message.get("batch_dev_result"),
    )

def render_history_messages() -> None:
    for message in st.session_state.message_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            if role == "assistant":
                result = build_history_result(message)

                if result.intent == "batch_development":
                    render_batch_development_result(result)
                elif result.intent == "overview" and result.structured_data:
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
            else:
                st.write(content)

def read_uploaded_text_file(uploaded_file: Any) -> str:
    """Streamlit 업로드 TXT 요청서를 문자열로 읽는다."""
    raw = uploaded_file.read()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").strip()

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()
    st.title("🤖 업무 인수인계 에이전트")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
        st.session_state.evaluation_mode = st.checkbox("평가용 근거 보기", value=st.session_state.evaluation_mode)

        st.markdown("### 배치 요청서 업로드")
        uploaded_batch_request = st.file_uploader(
            "TXT 요청서",
            type=["txt"],
            key="batch_request_txt",
            help="배치명/기준 테이블/파일명/기준일자/조건 등이 적힌 현업 요청서 TXT를 업로드합니다.",
        )
        if uploaded_batch_request is not None and st.button("요청서로 배치 생성", use_container_width=True):
            request_text = read_uploaded_text_file(uploaded_batch_request)
            if request_text:
                st.session_state.pending_question = request_text
            else:
                st.warning("요청서 파일 내용이 비어 있습니다.")

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
        normalized_for_intent = apply_dictionary_rewrite(user_question)
        initial_intent = detect_intent(normalized_for_intent)
        chat_history = build_chat_history(st.session_state.message_list[:-1])
        with st.spinner("답변을 생성하는 중입니다..."):
            if initial_intent == "batch_development":
                result = run_batch_development(user_question)
            else:
                agent = get_agent()
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
            "batch_dev_result": getattr(result, "batch_dev_result", None),
        })

if __name__ == "__main__":
    main()
