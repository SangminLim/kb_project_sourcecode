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

def _get_reasoning_panel_title(domain: str, default: str = "📊 데이터 상세 분석") -> str:
    """도메인별 상세 분석 제목을 반환한다.

    화면 문구를 각 renderer 함수에 흩뿌리지 않기 위한 작은 헬퍼다.
    향후 SQL/정산/고객 등 도메인이 늘어나면 이 매핑만 확장하면 된다.
    """
    title_map = {
        "incident": "📌 장애 데이터 상세 분석",
        "billing": "📊 청구 데이터 상세 분석",
    }
    return title_map.get(str(domain or "").strip().lower(), default)


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




def _first_non_empty(item: Dict[str, Any], keys: List[str], default: Any = "-") -> Any:
    """여러 payload key 후보 중 첫 번째 유효값을 반환한다."""
    for key in keys:
        if key in item and item.get(key) not in (None, ""):
            return item.get(key)
    return default


def _get_by_alias(item: Dict[str, Any], aliases: List[Any], default: Any = "-") -> Any:
    """정책 alias 기준으로 row 값을 찾는다. 대소문자 차이도 흡수한다."""
    if not isinstance(item, dict):
        return default
    for key in aliases or []:
        text_key = str(key)
        if text_key in item and item.get(text_key) not in (None, ""):
            return item.get(text_key)
    lowered = {str(key).lower(): key for key in item.keys()}
    for key in aliases or []:
        found = lowered.get(str(key).lower())
        if found is not None and item.get(found) not in (None, ""):
            return item.get(found)
    return default


def _get_incident_execution_plan(result: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """result/query_meta/payload 어디에 있든 incident execution_plan을 가져온다."""
    query_meta = getattr(result, "query_meta", None) or {}
    if isinstance(query_meta, dict) and isinstance(query_meta.get("execution_plan"), dict):
        return query_meta.get("execution_plan") or {}
    if isinstance(payload, dict) and isinstance(payload.get("execution_plan"), dict):
        return payload.get("execution_plan") or {}
    if isinstance(payload, dict) and isinstance(payload.get("query_meta"), dict):
        plan = payload.get("query_meta", {}).get("execution_plan")
        if isinstance(plan, dict):
            return plan
    return {}


def _get_incident_policy(result: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """reasoning_policy를 가져와 화면 출력 시 컬럼명을 하드코딩하지 않도록 한다."""
    query_meta = getattr(result, "query_meta", None) or {}
    if isinstance(query_meta, dict) and isinstance(query_meta.get("reasoning_policy"), dict):
        return query_meta.get("reasoning_policy") or {}
    if isinstance(payload, dict) and isinstance(payload.get("reasoning_policy"), dict):
        return payload.get("reasoning_policy") or {}
    if isinstance(payload, dict) and isinstance(payload.get("query_meta"), dict):
        policy = payload.get("query_meta", {}).get("reasoning_policy")
        if isinstance(policy, dict):
            return policy
    return {}


def _get_incident_selected_steps(result: Any, payload: Dict[str, Any]) -> List[str]:
    """planner가 고른 steps를 안전하게 가져온다."""
    plan = _get_incident_execution_plan(result, payload)
    steps = plan.get("steps") if isinstance(plan, dict) else []
    return [str(step).strip() for step in _as_list(steps) if str(step).strip()]


def _render_action_guide_section(item: Dict[str, Any], policy: Dict[str, Any]) -> None:
    """action_guide step이 선택된 경우에만 조치 관련 판단을 별도 섹션으로 보여준다."""
    aliases = policy.get("column_aliases", {}) if isinstance(policy, dict) else {}
    output_columns = policy.get("output_columns", {}) if isinstance(policy, dict) else {}

    action_detail = _get_by_alias(item, _as_list(aliases.get("action_detail")), "-")
    action_owner = _get_by_alias(item, _as_list(aliases.get("action_owner")), "-")
    recommended_action = _first_non_empty(
        item,
        [str(output_columns.get("recommended_action", "권장조치")), "권장조치", "recommended_action"],
        "-",
    )
    has_action_record = _first_non_empty(
        item,
        [str(output_columns.get("has_action_record", "운영조치존재여부")), "운영조치존재여부", "has_action_record"],
        "-",
    )
    needs_additional_action = _first_non_empty(
        item,
        [str(output_columns.get("needs_additional_action", "추가조치필요여부")), "추가조치필요여부", "needs_additional_action", "needs_action"],
        "-",
    )

    st.markdown("**🛠 조치 가이드**")
    st.markdown(f"- 등록 조치내용: {action_detail}")
    st.markdown(f"- 담당자: {action_owner}")
    st.markdown(f"- 운영조치존재여부: {has_action_record}")
    st.markdown(f"- 추가조치필요여부: {needs_additional_action}")
    st.markdown(f"- 권장조치: {recommended_action}")


def _render_impact_analysis_section(item: Dict[str, Any], policy: Dict[str, Any]) -> None:
    """impact_analysis step이 선택된 경우에만 영향도 관련 판단을 별도 섹션으로 보여준다."""
    aliases = policy.get("column_aliases", {}) if isinstance(policy, dict) else {}
    output_columns = policy.get("output_columns", {}) if isinstance(policy, dict) else {}

    impact_raw = _get_by_alias(item, _as_list(aliases.get("impact_yn")), "-")
    impact_level = _first_non_empty(
        item,
        [str(output_columns.get("impact_level", "영향도")), "영향도판단", "영향도", "impact_level", "impact_judgement"],
        "-",
    )
    priority = _first_non_empty(
        item,
        [str(output_columns.get("priority", "우선순위")), "우선순위", "priority"],
        "-",
    )
    long_running = _first_non_empty(
        item,
        [str(output_columns.get("long_running", "장기미처리여부")), "장기미처리여부", "is_long_running"],
        "-",
    )
    downstream_impact = _first_non_empty(
        item,
        ["후속영향확인", "후속배치영향", "downstream_impact", "downstream_jobs", "후속배치"],
        "-",
    )
    recommended_action = _first_non_empty(
        item,
        [str(output_columns.get("recommended_action", "권장조치")), "권장조치", "recommended_action"],
        "-",
    )

    st.markdown("**📌 영향도 분석**")
    st.markdown(f"- 원천 영향여부: {impact_raw}")
    st.markdown(f"- 영향도판단: {impact_level}")
    st.markdown(f"- 후속영향확인: {downstream_impact}")
    st.markdown(f"- 우선순위: {priority}")
    st.markdown(f"- 장기미처리여부: {long_running}")
    st.markdown(f"- 권장조치: {recommended_action}")

def render_incident_reasoning_panel(result: Any) -> None:
    """오늘 장애현황 multi reasoning step 결과를 화면에 표시한다.

    selected_steps 기준으로 기본 상태판단, 조치 가이드, 영향도 분석 섹션을 분리한다.
    운영 화면에서는 질문에 필요한 reasoning만 보이고, 평가 모드에서는 planner log를 함께 보여준다.
    """
    intent = str(getattr(result, "intent", "") or "")
    realtime_mode = str(getattr(result, "realtime_mode", "") or "")
    payload = _get_incident_payload(result)
    reasoning_results = _as_list(payload.get("reasoning_results"))
    selected_steps = _get_incident_selected_steps(result, payload)
    selected_step_set = set(selected_steps)
    policy = _get_incident_policy(result, payload)

    is_incident_result = (
        intent == "today_incidents"
        or "incident" in realtime_mode
        or bool(reasoning_results)
    )
    if not is_incident_result:
        return

    if reasoning_results:
        st.markdown(f"##### {_get_reasoning_panel_title('incident')}")

        if selected_steps:
            step_labels = (_get_incident_execution_plan(result, payload).get("step_labels") or {})
            visible_steps = [
                str(step_labels.get(step) or step)
                for step in selected_steps
                if step not in {"prepare_realtime_query", "summarize_table"}
            ]
            if visible_steps:
                st.caption("선택된 reasoning step: " + " → ".join(visible_steps))

        for idx, item in enumerate(reasoning_results, start=1):
            if not isinstance(item, dict):
                continue

            output_columns = policy.get("output_columns", {}) if isinstance(policy, dict) else {}
            batch_name = (
                item.get("배치명")
                or item.get("batch_name")
                or item.get("job_name")
                or item.get("name")
                or f"장애 {idx}"
            )
            status_reason = _first_non_empty(
                item,
                [str(output_columns.get("status_reason", "상태판단")), "상태판단", "status_reason", "status_judgement"],
                "-",
            )
            impact_reason = _first_non_empty(
                item,
                [str(output_columns.get("impact_level", "영향도")), "영향도판단", "영향도", "impact_level", "impact_judgement"],
                "-",
            )
            priority = _first_non_empty(
                item,
                [str(output_columns.get("priority", "우선순위")), "우선순위", "priority"],
                "-",
            )
            long_running = _first_non_empty(
                item,
                [str(output_columns.get("long_running", "장기미처리여부")), "장기미처리여부", "is_long_running"],
                "-",
            )
            has_action_record = _first_non_empty(
                item,
                [str(output_columns.get("has_action_record", "운영조치존재여부")), "운영조치존재여부", "has_action_record"],
                None,
            )
            needs_additional_action = _first_non_empty(
                item,
                [str(output_columns.get("needs_additional_action", "추가조치필요여부")), "추가조치필요여부", "needs_additional_action"],
                None,
            )

            legacy_needs_action = item.get("조치필요여부")
            if needs_additional_action is None and legacy_needs_action is not None:
                needs_additional_action = legacy_needs_action
            if has_action_record is None and legacy_needs_action is not None:
                has_action_record = not bool(legacy_needs_action)
            if has_action_record is None:
                has_action_record = "-"
            if needs_additional_action is None:
                needs_additional_action = item.get("needs_action", "-")

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

                if not selected_steps or "status_reasoning" in selected_step_set:
                    st.markdown("**🔎 상태/우선순위 판단**")
                    st.markdown(f"- 상태판단: {status_reason}")
                    st.markdown(f"- 영향도판단: {impact_reason}")
                    st.markdown(f"- 우선순위: {priority}")
                    st.markdown(f"- 장기미처리여부: {long_running}")
                    st.markdown(f"- 운영조치존재여부: {has_action_record}")
                    st.markdown(f"- 추가조치필요여부: {needs_additional_action}")

                if "action_guide" in selected_step_set:
                    st.markdown("")
                    _render_action_guide_section(item, policy)

                if "impact_analysis" in selected_step_set:
                    st.markdown("")
                    _render_impact_analysis_section(item, policy)

                if reasons:
                    st.markdown("")
                    st.markdown("- 판단근거")
                    for reason in reasons:
                        st.markdown(f"  - {reason}")
    elif intent == "today_incidents" or "incident" in realtime_mode:
        st.caption("Multi reasoning 결과가 아직 payload에 없습니다. incident_reasoning.py에서 reasoning_results를 realtime_payload 또는 structured_data에 넣어야 표시됩니다.")

    # debug_logs는 평가용 근거 모드에서만 보여준다.
    if getattr(st.session_state, "evaluation_mode", False):
        debug_logs = _as_list(payload.get("debug_logs")) + _as_list(getattr(result, "debug_logs", []))
        planner_logs = [
            str(log)
            for log in debug_logs
            if str(log).startswith("[PLAN")
            or str(log).startswith("[INCIDENT")
            or str(log).startswith("[BILLING PLAN")
            or str(log).startswith("[BILLING")
        ]
        if planner_logs:
            with st.expander("🛠 Planner / Reasoning Logs"):
                for log in planner_logs:
                    st.text(log)

def _get_billing_payload(result: Any) -> Dict[str, Any]:
    """청구 그래프/요약 reasoning 결과를 realtime_payload/structured_data에서 안전하게 가져온다."""
    realtime_payload = getattr(result, "realtime_payload", None) or {}
    structured_data = getattr(result, "structured_data", None) or {}

    if isinstance(realtime_payload, dict) and realtime_payload.get("billing_reasoning_results"):
        return realtime_payload

    if isinstance(structured_data, dict) and structured_data.get("billing_reasoning_results"):
        return structured_data

    if isinstance(structured_data, dict):
        billing_reasoning = structured_data.get("billing_reasoning") or {}
        if isinstance(billing_reasoning, dict) and billing_reasoning.get("billing_reasoning_results"):
            return billing_reasoning

    return {}


def render_billing_reasoning_panel(result: Any) -> None:
    """청구 그래프/요약 multi reasoning step 결과를 화면에 표시한다.

    표시 조건은 query_id가 아니라 post_process, realtime_mode, payload 결과를 기준으로 한다.
    """
    query_meta = getattr(result, "query_meta", None) or {}
    realtime_mode = str(getattr(result, "realtime_mode", "") or "")
    post_process = str(query_meta.get("post_process") or "")
    payload = _get_billing_payload(result)
    reasoning_results = _as_list(payload.get("billing_reasoning_results"))

    is_billing_result = (
        post_process == "billing_graph_reasoning"
        or "billing" in realtime_mode.lower()
        or bool(reasoning_results)
    )
    if not is_billing_result:
        return

    if reasoning_results:
        st.markdown(f"##### {_get_reasoning_panel_title('billing')}")

        for idx, item in enumerate(reasoning_results, start=1):
            if not isinstance(item, dict):
                continue

            title = (
                item.get("title")
                or item.get("step_label")
                or item.get("step")
                or f"청구 reasoning {idx}"
            )
            status = item.get("status") or "-"
            details = [str(detail) for detail in _as_list(item.get("details")) if str(detail).strip()]

            with st.container(border=True):
                st.markdown(f"**[{idx}] {title}**")
                st.markdown(f"- 상태: {status}")
                if details:
                    st.markdown("- 판단근거")
                    for detail in details:
                        st.markdown(f"  - {detail}")
    elif post_process == "billing_graph_reasoning" or "billing" in realtime_mode.lower():
        st.caption("Billing multi reasoning 결과가 아직 payload에 없습니다. realtime_payload.py에서 billing_reasoning_results를 넣어야 표시됩니다.")

    if getattr(st.session_state, "evaluation_mode", False):
        debug_logs = _as_list(payload.get("debug_logs")) + _as_list(getattr(result, "debug_logs", []))
        billing_logs = [
            str(log)
            for log in debug_logs
            if str(log).startswith("[BILLING")
            or (str(log).startswith("[PLAN") and "billing" in str(log).lower())
        ]
        if billing_logs:
            with st.expander("🛠 Billing Planner / Reasoning Logs"):
                for log in billing_logs:
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
            render_billing_reasoning_panel(result)
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
                    render_billing_reasoning_panel(result)
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
