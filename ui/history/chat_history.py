from __future__ import annotations

from ..context import *
from ..state import *
from ..renderers.streamlit_renderers import *
from ..evaluation.evaluation_panel import render_evaluation_panel
from ..sql.sql_analysis_ui import render_sql_analysis_result
from ..batch.batch_ui import render_batch_development_result, render_sql_improvement_report
from ..services.realtime_payload import enrich_result_with_realtime_payload

def render_agent_result(result: Any) -> None:
    with st.chat_message("assistant"):
        if result.intent == "batch_development":
            render_batch_development_result(result)
        elif result.intent == "sql_analysis":
            render_sql_analysis_result(result)
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
        sql_analysis_result=message.get("sql_analysis_result"),
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
                elif result.intent == "sql_analysis":
                    render_sql_analysis_result(result)
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
