"""
streamlit.py

설명
- 기존 chat UI 스타일을 유지하면서 HandoverAgent와 연결
- 질문 유형에 따라 text / graph / chart / table 로 분기 렌더링
- 흐름도 / 리니지: graphviz 사용
- 월별 금액: DB 조회 후 plotly bar chart
- 장애 현황: DB 조회 후 dataframe
- debug_logs 지원: 질문 해석 로그 보기

필수 환경 변수 예시 (.env)
JSON_PATH=./handover.json
CHROMA_PERSIST_DIR=./chroma
CHROMA_COLLECTION=handover_agent
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=llama3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

선택 환경 변수 (DB 연결)
DATABASE_URL=mysql+pymysql://user:password@127.0.0.1:3306/mydb

실행
streamlit run streamlit.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph

from core.llm import HandoverAgent


load_dotenv()


# ---------------------------
# 설정
# ---------------------------

PAGE_TITLE = "금융 업무 챗봇"
PAGE_ICON = "🤖"

JSON_PATH = os.getenv("JSON_PATH", "./handover.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "handover_agent")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


# ---------------------------
# 캐시 리소스
# ---------------------------

@st.cache_resource
def get_agent() -> HandoverAgent:
    return HandoverAgent(
        json_path=JSON_PATH,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )


# ---------------------------
# 세션 초기화
# ---------------------------

def init_session_state() -> None:
    if "message_list" not in st.session_state:
        st.session_state.message_list = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False


# ---------------------------
# 히스토리 변환
# ---------------------------

def build_chat_history(message_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in message_list:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


# ---------------------------
# 그래프 렌더링
# ---------------------------

def draw_graphviz_graph(graph_data: Dict[str, Any], graph_kind: str = "flow") -> None:
    dot = Digraph()

    title = graph_data.get("title", "")
    if title:
        st.subheader(title)

    if graph_kind == "lineage":
        for table in graph_data.get("tables", []):
            node_id = str(table.get("id", ""))
            label = node_id
            dot.node(node_id, label)

        for edge in graph_data.get("edges", []):
            src = str(edge.get("from", ""))
            dst = str(edge.get("to", ""))
            if src and dst:
                dot.edge(src, dst)
    else:
        for node in graph_data.get("nodes", []):
            node_id = str(node.get("id", ""))
            label = str(node.get("label", node_id))
            dot.node(node_id, label)

        for edge in graph_data.get("edges", []):
            src = str(edge.get("from", ""))
            dst = str(edge.get("to", ""))
            if src and dst:
                dot.edge(src, dst)

    st.graphviz_chart(dot, use_container_width=True)


# ---------------------------
# DB 조회
# ---------------------------

def get_db_engine():
    if not DATABASE_URL:
        return None

    try:
        from sqlalchemy import create_engine
    except ImportError as e:
        raise RuntimeError(
            "sqlalchemy가 설치되어 있지 않습니다. pip install sqlalchemy pymysql 로 설치하세요."
        ) from e

    return create_engine(DATABASE_URL)


def query_dataframe(sql: str) -> pd.DataFrame:
    engine = get_db_engine()
    if engine is None:
        raise RuntimeError(
            "DATABASE_URL이 설정되지 않았습니다. .env에 DATABASE_URL을 설정하세요."
        )
    return pd.read_sql(sql, engine)


def fetch_realtime_dataframe(query_meta: Dict[str, Any]) -> pd.DataFrame:
    data_source = query_meta.get("data_source", {}) or {}
    table_name = data_source.get("table", "")
    render_type = query_meta.get("render_type", "")

    if not table_name:
        raise RuntimeError("query_meta에 data_source.table 정보가 없습니다.")

    if render_type == "chart":
        x_field = query_meta.get("x_field", "billing_month")
        y_field = query_meta.get("y_field", "amount")

        sql = f"""
        SELECT {x_field}, {y_field}
        FROM {table_name}
        ORDER BY {x_field}
        """
        return query_dataframe(sql)

    if render_type == "table":
        columns = query_meta.get("columns", [])
        if not columns:
            sql = f"SELECT * FROM {table_name} LIMIT 100"
        else:
            col_text = ", ".join(columns)
            sql = f"""
            SELECT {col_text}
            FROM {table_name}
            ORDER BY start_time DESC
            LIMIT 100
            """
        return query_dataframe(sql)

    raise RuntimeError(f"지원하지 않는 render_type 입니다: {render_type}")


# ---------------------------
# 차트/테이블 렌더링
# ---------------------------

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

    fig = px.bar(df, x=x_field, y=y_field, title=title)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("조회 데이터 보기"):
        st.dataframe(df, use_container_width=True)


def render_table(query_meta: Dict[str, Any]) -> None:
    st.subheader(query_meta.get("title", "테이블"))

    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as e:
        st.warning(f"DB 조회 실패: {e}")
        st.info("DATABASE_URL과 실제 테이블/컬럼 구성을 확인하세요.")
        return

    if df.empty:
        st.info("조회 결과가 없습니다.")
        return

    st.dataframe(df, use_container_width=True)


# ---------------------------
# 디버그 렌더링
# ---------------------------

def render_debug_blocks(
    *,
    normalized_question: str = "",
    rewritten_question: str = "",
    system_id: str | None = None,
    intent: str | None = None,
    render_type: str | None = None,
    sources: List[Dict[str, Any]] | None = None,
    debug_logs: List[str] | None = None,
) -> None:
    if not st.session_state.debug_mode:
        return

    with st.expander("디버그 정보"):
        st.json(
            {
                "normalized_question": normalized_question,
                "rewritten_question": rewritten_question,
                "system_id": system_id,
                "intent": intent,
                "render_type": render_type,
                "sources": sources or [],
            }
        )

    with st.expander("질문 해석 로그 보기"):
        logs = debug_logs or []
        if logs:
            for log in logs:
                st.code(log)
        else:
            st.info("debug_logs가 없습니다. llm.py가 debug 통합 버전인지 확인하세요.")


# ---------------------------
# 결과 렌더링
# ---------------------------

def render_agent_result(result: Any) -> None:
    """
    result는 HandoverAgent.answer_question()의 AgentResult
    """
    with st.chat_message("assistant"):
        st.write(result.answer)

        if result.render_type == "graph" and result.graph_data:
            if result.intent == "table_lineage":
                draw_graphviz_graph(result.graph_data, graph_kind="lineage")
            else:
                draw_graphviz_graph(result.graph_data, graph_kind="flow")

        elif result.render_type == "chart" and result.query_meta:
            render_chart(result.query_meta)

        elif result.render_type == "table" and result.query_meta:
            render_table(result.query_meta)

        render_debug_blocks(
            normalized_question=getattr(result, "normalized_question", ""),
            rewritten_question=getattr(result, "rewritten_question", ""),
            system_id=getattr(result, "system_id", None),
            intent=getattr(result, "intent", None),
            render_type=getattr(result, "render_type", None),
            sources=getattr(result, "sources", []),
            debug_logs=getattr(result, "debug_logs", []),
        )


# ---------------------------
# 과거 메시지 다시 렌더링
# ---------------------------

def render_history_messages() -> None:
    for message in st.session_state.message_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.write(content)

            render_type = message.get("render_type")
            intent = message.get("intent")
            graph_data = message.get("graph_data")
            query_meta = message.get("query_meta")

            if role == "assistant":
                if render_type == "graph" and graph_data:
                    if intent == "table_lineage":
                        draw_graphviz_graph(graph_data, graph_kind="lineage")
                    else:
                        draw_graphviz_graph(graph_data, graph_kind="flow")
                elif render_type == "chart" and query_meta:
                    render_chart(query_meta)
                elif render_type == "table" and query_meta:
                    render_table(query_meta)

                render_debug_blocks(
                    normalized_question=message.get("normalized_question", ""),
                    rewritten_question=message.get("rewritten_question", ""),
                    system_id=message.get("system_id"),
                    intent=message.get("intent"),
                    render_type=message.get("render_type"),
                    sources=message.get("sources", []),
                    debug_logs=message.get("debug_logs", []),
                )


# ---------------------------
# 메인
# ---------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()

    st.title("🤖 금융 업무 챗봇")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
        st.session_state.debug_mode = st.checkbox("디버그 정보 보기", value=st.session_state.debug_mode)
        st.markdown("### 현재 연결 정보")
        st.write(f"- JSON_PATH: `{JSON_PATH}`")
        st.write(f"- CHROMA_PERSIST_DIR: `{CHROMA_PERSIST_DIR}`")
        st.write(f"- CHROMA_COLLECTION: `{CHROMA_COLLECTION}`")
        st.write(f"- DATABASE_URL 설정 여부: `{'Y' if DATABASE_URL else 'N'}`")

    render_history_messages()

    user_question = st.chat_input(
        placeholder="예) KKK은행 소득공제 배치 프로세스를 설명해줘"
    )

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.message_list.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        agent = get_agent()
        chat_history = build_chat_history(st.session_state.message_list[:-1])

        with st.spinner("답변을 생성하는 중입니다..."):
            result = agent.answer_question(
                question=user_question,
                chat_history=chat_history,
            )

        render_agent_result(result)

        st.session_state.message_list.append(
            {
                "role": "assistant",
                "content": result.answer,
                "intent": result.intent,
                "render_type": result.render_type,
                "graph_data": result.graph_data,
                "query_meta": result.query_meta,
                "sources": result.sources,
                "normalized_question": getattr(result, "normalized_question", ""),
                "rewritten_question": getattr(result, "rewritten_question", ""),
                "system_id": getattr(result, "system_id", None),
                "debug_logs": getattr(result, "debug_logs", []),
            }
        )


if __name__ == "__main__":
    main()
