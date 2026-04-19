from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph

load_dotenv()

from llm import ChatConfig, HandoverAgent, ollama_generate
from realtime_query_service import RealtimeQueryService

PAGE_TITLE = "금융 업무 챗봇"
PAGE_ICON = "🤖"

JSON_PATH = os.getenv("JSON_PATH", "./handover_improved.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "handover_agent")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
BILLING_QUERY_ID = "billing_monthly_amount"


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


def render_billing_llm_summary(query_meta: Dict[str, Any], df: pd.DataFrame) -> None:
    x_field = query_meta.get("x_field", "billing_month")
    y_field = query_meta.get("y_field", "amount")

    try:
        summary_payload = summarize_billing_dataframe(df, x_field=x_field, y_field=y_field)
        summary_prompt = build_billing_summary_prompt(query_meta, summary_payload)
        summary = ollama_generate(
            prompt=summary_prompt,
            system_prompt="반드시 한국어로만, 요약 데이터 기반으로 간단히 해석해라.",
            config=ChatConfig(),
        )
        if summary:
            st.markdown("##### 📈 데이터 요약")
            st.write(summary)
    except Exception as e:
        st.warning(f"LLM 차트 요약 생성 실패: {e}")


def _render_bullets(items: List[str]) -> None:
    for item in items or []:
        st.markdown(f"- {item}")


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


def render_chart(query_meta: Dict[str, Any]) -> None:
    st.subheader(query_meta.get("title", "차트"))
    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as e:
        st.warning(f"DB 조회 실패: {e}")
        st.info("DATABASE_URL과 실제 테이블/컬럼 구성을 확인하세요.")
        return
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
    st.plotly_chart(fig, use_container_width=True)

    if query_meta.get("query_id") == BILLING_QUERY_ID:
        render_billing_llm_summary(query_meta, df)

    with st.expander("조회 데이터 보기"):
        st.dataframe(df, use_container_width=True)


def render_table(query_meta: Dict[str, Any], realtime_mode: str | None = None) -> None:
    st.subheader(query_meta.get("title", "테이블"))
    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as e:
        st.warning(f"DB 조회 실패: {e}")
        st.info("DATABASE_URL과 실제 테이블/컬럼 구성을 확인하세요.")
        return
    if df.empty:
        if query_meta.get("query_id") == "today_incidents":
            st.info("오늘 장애는 없습니다.")
        else:
            st.info("조회 결과가 없습니다.")
        return
    if realtime_mode == "incident_table_with_summary" or query_meta.get("query_id") == "today_incidents":
        try:
            rows_json = df.to_json(orient="records", force_ascii=False)
            summary_prompt = build_incident_summary_prompt(rows_json)
            summary = ollama_generate(
                prompt=summary_prompt,
                system_prompt="반드시 한국어로만, 데이터 기반으로 요약해라.",
                config=ChatConfig(),
            )
            st.write(summary)
        except Exception as e:
            st.warning(f"LLM 요약 생성 실패: {e}")
    st.dataframe(df, use_container_width=True)


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
            render_chart(result.query_meta)
        elif result.render_type == "table" and result.query_meta:
            render_table_summary(result)
            render_table(result.query_meta, getattr(result, "realtime_mode", None))
        else:
            st.write(result.answer)

        render_debug_blocks(
            normalized_question=getattr(result, "normalized_question", ""),
            rewritten_question=getattr(result, "rewritten_question", ""),
            system_id=getattr(result, "system_id", None),
            intent=getattr(result, "intent", None),
            render_type=getattr(result, "render_type", None),
            sources=getattr(result, "sources", []),
            debug_logs=getattr(result, "debug_logs", []),
        )


def render_history_messages() -> None:
    for message in st.session_state.message_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            render_type = message.get("render_type")
            intent = message.get("intent")
            graph_data = message.get("graph_data")
            query_meta = message.get("query_meta")
            realtime_mode = message.get("realtime_mode")
            structured_data = message.get("structured_data")

            if role == "assistant":
                class HistoryResult:
                    pass

                result = HistoryResult()
                result.answer = content
                result.intent = intent
                result.render_type = render_type
                result.graph_data = graph_data
                result.query_meta = query_meta
                result.realtime_mode = realtime_mode
                result.structured_data = structured_data

                if intent == "overview" and structured_data:
                    render_overview_block(result)
                elif intent == "batch_process" and structured_data:
                    render_batch_process_block(result)
                elif render_type == "graph" and graph_data:
                    render_graph_summary(result)
                    if intent == "table_lineage":
                        draw_graphviz_graph(graph_data, graph_kind="lineage")
                    else:
                        draw_graphviz_graph(graph_data, graph_kind="flow")
                elif render_type == "chart" and query_meta:
                    render_chart_summary(result)
                    render_chart(query_meta)
                elif render_type == "table" and query_meta:
                    render_table_summary(result)
                    render_table(query_meta, realtime_mode)
                else:
                    st.write(content)

                render_debug_blocks(
                    normalized_question=message.get("normalized_question", ""),
                    rewritten_question=message.get("rewritten_question", ""),
                    system_id=message.get("system_id"),
                    intent=message.get("intent"),
                    render_type=message.get("render_type"),
                    sources=message.get("sources", []),
                    debug_logs=message.get("debug_logs", []),
                )
            else:
                st.write(content)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()
    st.title("🤖 금융 업무 챗봇")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
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
            render_agent_result(result)
        st.session_state.message_list.append({
            "role": "assistant",
            "content": result.answer,
            "intent": result.intent,
            "render_type": result.render_type,
            "graph_data": result.graph_data,
            "query_meta": result.query_meta,
            "realtime_mode": getattr(result, "realtime_mode", None),
            "sources": result.sources,
            "structured_data": getattr(result, "structured_data", None),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "debug_logs": getattr(result, "debug_logs", []),
        })


if __name__ == "__main__":
    main()
