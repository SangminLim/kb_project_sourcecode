from __future__ import annotations
# 답변마다 고유 id 생성
# UUID.randomUUID().toString()
import uuid

from .context import *
from .state import *

# 핵심서비스 호출 사용자가 질문하면 실제 AI처리, RAG 검색, DB 조회, 그래프 생성 같은 핵심 로직은 여기서 수행됨
from .workflow.handover_graph import run_handover_graph

from .history.chat_history import (
    render_agent_result,
    render_history_messages,
    read_uploaded_text_file,
)

from .sql.sql_analysis_ui import parse_sql_analysis_request

# 이 함수는 AI 답변 결과를 세션에 저장하는 함수야
def _append_assistant_message(result: Any, message_id: str) -> None:
    """assistant 응답을 session_state.message_list에 저장한다.
    리팩토링 후 result 타입이 AgentResult/SimpleNamespace 등으로 달라질 수 있으므로
    모든 부가 필드는 getattr 기반으로 안전하게 저장한다.
    """
    st.session_state.message_list.append({
        "message_id": message_id,
        "role": "assistant",
        "content": getattr(result, "answer", ""),
        "intent": getattr(result, "intent", None),
        "render_type": getattr(result, "render_type", "text"),
        "graph_data": getattr(result, "graph_data", None),
        "query_meta": getattr(result, "query_meta", None),
        "realtime_mode": getattr(result, "realtime_mode", None),
        "realtime_payload": getattr(result, "realtime_payload", None),
        "sources": getattr(result, "sources", []),
        "structured_data": getattr(result, "structured_data", None),
        "normalized_question": getattr(result, "normalized_question", ""),
        "rewritten_question": getattr(result, "rewritten_question", ""),
        "system_id": getattr(result, "system_id", None),
        "debug_logs": getattr(result, "debug_logs", []),
        "batch_dev_result": getattr(result, "batch_dev_result", None),
        "sql_analysis_result": getattr(result, "sql_analysis_result", None),
    })


#하네스 테스트 실행 화면이야
def _render_harness_test_panel() -> None:
    """하네스 테스트를 Streamlit 화면에서 실행하고 결과를 표시한다."""
    if not st.session_state.get("run_harness_test", False):
        return

    try:
        import argparse
        import pandas as pd
        from harnessTest.handover_harness import run_tests
    except Exception as exc:
        st.error(f"Harness 모듈을 불러오지 못했습니다: {type(exc).__name__}: {exc}")
        st.info("handover_harness.py가 프로젝트 루트에 있는지 확인하세요.")
        st.session_state.run_harness_test = False
        return

    st.subheader("🧪 Harness Test Result")

    args = argparse.Namespace(
        llm_path="agents.handover.agent",
        json_path="ingest/handover.json",
        persist_dir="./chroma",
        collection="handover_agent",
        cases="harnessTest/query_routing_test_cases.json",
        output_csv="harnessTest/query_routing_test_results.csv",
        output_json="harnessTest/query_routing_test_results.json",
        top_k=4,
        pass_score=80.0,
        fail_under=None,
    )

    try:
        with st.spinner("Harness 테스트 실행 중입니다..."):
            payload = run_tests(args)
    except Exception as exc:
        st.error(f"Harness 테스트 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        st.session_state.run_harness_test = False
        return

    summary = payload.get("summary", {}) or {}
    results = payload.get("results", []) or []

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", summary.get("total", 0))
    c2.metric("Passed", summary.get("passed", 0))
    c3.metric("Failed", summary.get("failed", 0))
    c4.metric("Pass Rate", f"{summary.get('pass_rate', 0.0)}%")
    c5.metric("Avg Score", summary.get("avg_score", 0.0))

    category_summary = summary.get("category_summary", {}) or {}
    if category_summary:
        st.markdown("#### 카테고리별 결과")
        category_df = pd.DataFrame.from_dict(category_summary, orient="index").reset_index()
        category_df = category_df.rename(columns={"index": "category"})
        st.dataframe(category_df, use_container_width=True, hide_index=True)

    if results:
        df = pd.DataFrame(results)
        base_columns = [
            "case_id", "category", "question", "score", "pass_yn",
            "actual_system_id", "actual_intent", "actual_render_type",
            "rewritten_question", "error",
        ]
        visible_columns = [col for col in base_columns if col in df.columns]

        st.markdown("#### 상세 결과")
        st.dataframe(df[visible_columns], use_container_width=True, hide_index=True)

        failed_df = df[df.get("pass_yn") == "N"] if "pass_yn" in df.columns else pd.DataFrame()
        if not failed_df.empty:
            st.markdown("#### 실패 케이스")
            failed_columns = [
                "case_id", "category", "question", "score",
                "actual_system_id", "actual_intent", "actual_render_type", "error",
            ]
            failed_columns = [col for col in failed_columns if col in failed_df.columns]
            st.dataframe(failed_df[failed_columns], use_container_width=True, hide_index=True)

            with st.expander("실패 케이스 검증 항목 보기"):
                for row in failed_df.to_dict("records"):
                    st.markdown(f"**{row.get('case_id')} | {row.get('category')} | score={row.get('score')}**")
                    st.write(row.get("question", ""))
                    st.json(row.get("checks", {}))
        else:
            st.success("실패 케이스 없음")

    st.download_button(
        "CSV 결과 다운로드",
        data=open(args.output_csv, "rb").read(),
        file_name="query_routing_test_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "JSON 결과 다운로드",
        data=open(args.output_json, "rb").read(),
        file_name="query_routing_test_results.json",
        mime="application/json",
        use_container_width=True,
    )

    st.session_state.run_harness_test = False

# 화면 시작점
def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()
    st.title("🤖 업무 인수인계 에이전트")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
        st.session_state.evaluation_mode = st.checkbox(
            "평가용 근거 보기",
            value=st.session_state.evaluation_mode,
        )

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

        st.markdown("### SQL 분석 요청서 업로드")
        uploaded_sql_request = st.file_uploader(
            "SQL 요청서",
            type=["txt", "sql"],
            key="sql_analysis_request_txt",
            help="[SQL]과 선택 항목인 [수정내용]이 적힌 SQL 분석 요청서를 업로드합니다.",
        )
        if uploaded_sql_request is not None and st.button("요청서로 SQL 분석", use_container_width=True):
            request_text = read_uploaded_text_file(uploaded_sql_request)
            if request_text:
                st.session_state.pending_sql_analysis = request_text
            else:
                st.warning("SQL 요청서 파일 내용이 비어 있습니다.")

        st.markdown("### 최근 질문")
        recent_questions = get_recent_questions(st.session_state.message_list, limit=10)
        if recent_questions:
            for idx, question in enumerate(recent_questions, start=1):
                label = shorten_text(question, 40)
                if st.button(label, key=f"recent_q_{idx}", use_container_width=True):
                    st.session_state.pending_question = question
        else:
            st.caption("아직 질문 내역이 없습니다.")

        st.markdown("---")
        st.markdown("### 품질 검증")
        if st.button("🧪 Harness 테스트 실행", use_container_width=True):
            st.session_state.run_harness_test = True

    render_history_messages()
    _render_harness_test_panel()

    chat_input_question = st.chat_input(
        placeholder="예) KKK은행 소득공제 배치 프로세스를 설명해줘"
    )
    pending_sql_analysis = st.session_state.pending_sql_analysis
    user_question = pending_sql_analysis or st.session_state.pending_question or chat_input_question
    force_sql_analysis = pending_sql_analysis is not None

    st.session_state.pending_sql_analysis = None
    st.session_state.pending_question = None

    if not user_question:
        return

    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.message_list.append({
        "role": "user",
        "content": user_question,
    })

    chat_history = build_chat_history(st.session_state.message_list[:-1])

    with st.spinner("답변을 생성하는 중입니다..."):
        result = run_handover_graph(
            user_question=user_question,
            chat_history=chat_history,
            force_sql_analysis=force_sql_analysis,
        )
        current_message_id = str(uuid.uuid4())
        setattr(result, "message_id", current_message_id)
        render_agent_result(result)

    _append_assistant_message(result, current_message_id)
