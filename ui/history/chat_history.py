from __future__ import annotations

from ..context import *
from ..state import *
from ..renderers.streamlit_renderers import *
from ..evaluation.evaluation_panel import render_evaluation_panel
from ..sql.sql_analysis_ui import render_sql_analysis_result
from ..batch.batch_ui import render_batch_development_result, render_sql_improvement_report
from ..services.realtime_payload import enrich_result_with_realtime_payload


def _as_list(value: Any) -> List[Any]:
    """값을 화면 렌더링 가능한 list로 안전하게 변환한다."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_incident_payload(result: Any) -> Dict[str, Any]:
    """장애현황 reasoning 결과를 realtime_payload/structured_data 양쪽에서 안전하게 가져온다.

    실무 기준:
    - reasoning 계산은 renderer에서 하지 않는다.
    - renderer는 graph/service 계층이 넣어준 결과만 표시한다.
    - payload 구조가 조금 바뀌어도 화면이 깨지지 않도록 defensive하게 접근한다.
    """
    realtime_payload = getattr(result, "realtime_payload", None) or {}
    structured_data = getattr(result, "structured_data", None) or {}

    if isinstance(realtime_payload, dict) and realtime_payload.get("reasoning_results"):
        return realtime_payload

    if isinstance(structured_data, dict) and structured_data.get("reasoning_results"):
        return structured_data

    # 일부 구현에서는 incident_reasoning 하위 key에 넣을 수 있으므로 호환 처리한다.
    if isinstance(structured_data, dict):
        incident_reasoning = structured_data.get("incident_reasoning") or {}
        if isinstance(incident_reasoning, dict) and incident_reasoning.get("reasoning_results"):
            return incident_reasoning

    return {}


def render_incident_reasoning_panel(result: Any) -> None:
    """오늘 장애현황 multi reasoning step 결과를 화면에 표시한다.

    표시 조건:
    - today_incidents intent이거나
    - realtime_mode가 incident 계열이거나
    - payload에 reasoning_results가 존재하는 경우
    """
    intent = str(getattr(result, "intent", "") or "")
    realtime_mode = str(getattr(result, "realtime_mode", "") or "")
    payload = _get_incident_payload(result)
    reasoning_results = _as_list(payload.get("reasoning_results"))

    is_incident_result = (
        intent == "today_incidents"
        or "incident" in realtime_mode
        or bool(reasoning_results)
    )
    if not is_incident_result:
        return

    if reasoning_results:
        st.markdown("##### 🧠 Multi Reasoning Step")

        for idx, item in enumerate(reasoning_results, start=1):
            if not isinstance(item, dict):
                continue

            batch_name = (
                item.get("배치명")
                or item.get("batch_name")
                or item.get("job_name")
                or item.get("name")
                or f"장애 {idx}"
            )
            status_reason = item.get("상태판단") or item.get("status_reason") or item.get("status_judgement") or "-"
            impact_reason = item.get("영향도판단") or item.get("impact_level") or item.get("impact_judgement") or "-"
            priority = item.get("우선순위") or item.get("priority") or "-"
            needs_action = item.get("조치필요여부")
            if needs_action is None:
                needs_action = item.get("needs_action", "-")
            long_running = item.get("장기미처리여부")
            if long_running is None:
                long_running = item.get("is_long_running", "-")

            reasons = (
                item.get("판단근거")
                or item.get("reasons")
                or item.get("reasoning_basis")
                or item.get("basis")
                or []
            )
            reasons = [str(reason) for reason in _as_list(reasons) if str(reason).strip()]

            with st.container(border=True):
                st.markdown(f"**[{idx}] {batch_name}**")
                st.markdown(f"- 상태판단: {status_reason}")
                st.markdown(f"- 영향도판단: {impact_reason}")
                st.markdown(f"- 우선순위: {priority}")
                st.markdown(f"- 장기미처리여부: {long_running}")
                st.markdown(f"- 조치필요여부: {needs_action}")

                if reasons:
                    st.markdown("- 판단근거")
                    for reason in reasons:
                        st.markdown(f"  - {reason}")
    elif intent == "today_incidents" or "incident" in realtime_mode:
        st.caption("Multi reasoning 결과가 아직 payload에 없습니다. incident_reasoning.py에서 reasoning_results를 realtime_payload 또는 structured_data에 넣어야 표시됩니다.")

    # debug_logs는 평가용 근거 모드에서만 보여준다.
    if getattr(st.session_state, "evaluation_mode", False):
        debug_logs = _as_list(payload.get("debug_logs")) + _as_list(getattr(result, "debug_logs", []))
        incident_logs = [str(log) for log in debug_logs if str(log).startswith("[INCIDENT")]
        if incident_logs:
            with st.expander("🛠 Incident Reasoning Logs"):
                for log in incident_logs:
                    st.text(log)


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
            render_incident_reasoning_panel(result)
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
                    render_incident_reasoning_panel(result)
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
