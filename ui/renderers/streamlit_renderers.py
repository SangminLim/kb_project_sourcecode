from __future__ import annotations

from ..context import *
from ..services.realtime_payload import payload_to_dataframe, get_realtime_policy

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

def _render_bullets(items: List[str]) -> None:
    for item in items or []:
        st.markdown(f"- {item}")


def unique_preserve_order(items: List[Any]) -> List[str]:
    """순서를 유지하면서 중복 값을 제거한다.

    화면 렌더링 단계에서 동일한 key_job이 여러 step에 들어오더라도
    한 번만 보여주기 위한 공통 유틸이다.
    """
    result: List[str] = []
    seen = set()
    for item in items or []:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def format_execution_label(execution: Any) -> str:
    """JSON의 execution 값을 화면용 라벨로 변환한다.

    parallel/sequential 외 값이 들어와도 원문을 보존해서 확장 가능하게 처리한다.
    """
    value = str(execution or "").strip()
    labels = {
        "parallel": "병렬",
        "sequential": "순차",
    }
    return labels.get(value.lower(), value)


def build_step_flow_text(steps: List[Dict[str, Any]]) -> str:
    """steps 메타데이터 기반으로 한 줄 흐름을 만든다."""
    step_names = [str(step.get("name", "")).strip() for step in steps or []]
    step_names = [name for name in step_names if name]
    return " → ".join(step_names)

def render_overview_block(result: Any) -> None:
    """overview 구조화 JSON을 카드 1개 안에 통합해서 렌더링한다.

    실무/확장성 원칙:
    - 시스템명/업무명은 코드에 하드코딩하지 않고 JSON의 title을 사용한다.
    - overview JSON에 있는 필드만 동적으로 표시한다.
    - HTML/CSS를 직접 출력하지 않고 Streamlit 기본 컴포넌트를 사용한다.
    - 핵심요약과 하위 항목을 여러 카드로 쪼개지 않고 하나의 카드 안에 정리한다.
    """
    data = getattr(result, "structured_data", None) or {}
    overview = data.get("overview", {}) if "overview" in data else data

    if not overview:
        st.info("업무 개요 정보가 없습니다.")
        return

    def as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            parts = [p.strip() for p in text.split(" / ") if p.strip()]
            return parts if len(parts) > 1 else [text]
        text = str(value).strip()
        return [text] if text else []

    def render_inline_items(items: List[str]) -> str:
        return " · ".join(items)

    title = str(overview.get("title") or "업무 개요").strip()
    summary_items = as_list(overview.get("summary") or overview.get("content"))

    st.markdown(f"#### 📌 {title}")

    section_specs = APP_RENDER_SCHEMA.get("overview_sections", []) or DEFAULT_APP_RENDER_SCHEMA.get("overview_sections", [])

    rendered_keys = {"title", "summary", "content"}
    visible_sections: List[tuple[str, str, List[str]]] = []

    for spec in section_specs:
        key = str(spec.get("key") or spec.get("field") or "").strip()
        if not key:
            continue
        label = str(spec.get("label") or key)
        icon = str(spec.get("icon") or "ℹ️")
        rendered_keys.add(key)
        items = as_list(overview.get(key))
        if items:
            visible_sections.append((label, icon, items))

    # 향후 overview JSON 필드가 추가되어도 설정 label이 있으면 사용하고, 없으면 원 key를 표시한다.
    label_map = deep_merge_config(
        DEFAULT_APP_RENDER_SCHEMA.get("overview_extra_labels", {}),
        APP_RENDER_SCHEMA.get("overview_extra_labels", {}) or {},
    )
    for key, value in overview.items():
        if key in rendered_keys:
            continue
        items = as_list(value)
        if not items:
            continue
        visible_sections.append((label_map.get(str(key), str(key)), "ℹ️", items))

    with st.container(border=True):
        st.markdown("##### 🔹 핵심 요약")
        if summary_items:
            for item in summary_items:
                st.markdown(f"- {item}")
        else:
            st.caption("등록된 요약 정보가 없습니다.")

        if visible_sections:
            st.markdown("")
            left_col, right_col = st.columns(2)

            for idx, (label, icon, items) in enumerate(visible_sections):
                target_col = left_col if idx % 2 == 0 else right_col
                with target_col:
                    st.markdown(f"**{icon} {label}**")
                    st.caption(render_inline_items(items))
                    st.markdown("")


def format_list_inline(values: Any) -> str:
    """list/string 값을 화면용 한 줄 문자열로 변환한다."""
    if values is None:
        return ""
    if isinstance(values, list):
        return ", ".join([str(v).strip() for v in values if str(v).strip()])
    return str(values).strip()


def format_duration_sec(value: Any) -> str:
    """초 단위 평균 수행시간을 보기 좋게 표시한다."""
    if value is None or str(value).strip() == "":
        return ""
    try:
        seconds = int(float(value))
    except Exception:
        return str(value).strip()

    if seconds < 60:
        return f"{seconds}초"

    minutes = seconds // 60
    remain = seconds % 60
    if remain:
        return f"{minutes}분 {remain}초"
    return f"{minutes}분"


def format_render_field_value(value: Any, formatter: str | None = None) -> str:
    """렌더링 스키마의 formatter 설정에 따라 표시값을 변환한다."""
    if formatter == "duration_sec":
        return format_duration_sec(value)
    return str(value or "").strip()


def _html_escape(value: Any) -> str:
    """Streamlit HTML 카드에 표시할 문자열을 안전하게 이스케이프한다."""
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_batch_job_card(job: Dict[str, Any]) -> None:
    """배치 1건을 하나의 심플한 네모 카드로 출력한다.

    내부에 항목별 작은 네모 박스를 만들지 않고,
    하나의 큰 카드 안에 배치명/설명/운영정보를 라벨-값 형태로 간단히 보여준다.
    """
    job_id = str(job.get("job_id", "")).strip()
    job_name = str(job.get("job_name", "")).strip()
    job_desc = str(job.get("description", "")).strip()

    title = job_id or job_name or "배치명 없음"
    subtitle = job_name if job_name and job_name != job_id else ""

    info_rows = []
    for field_spec in (APP_RENDER_SCHEMA.get("batch_job_info_fields", []) or DEFAULT_APP_RENDER_SCHEMA.get("batch_job_info_fields", [])):
        key = str(field_spec.get("key", "")).strip()
        if not key:
            continue
        value = format_render_field_value(job.get(key), field_spec.get("formatter"))
        if value:
            info_rows.append((str(field_spec.get("label") or key), value))

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size:14px; color:#374151; font-weight:700; margin-top:4px;">{_html_escape(subtitle)}</div>'

    desc_html = ""
    if job_desc:
        desc_html = f'<div style="font-size:14px; color:#4B5563; margin-top:8px; line-height:1.55;">{_html_escape(job_desc)}</div>'

    info_html = ""
    if info_rows:
        info_text = " &nbsp; | &nbsp; ".join(
            [
                f'<span style="white-space:nowrap;"><b style="color:#6B7280;">{_html_escape(label)}</b> <span style="color:#111827; font-weight:600;">{_html_escape(value)}</span></span>'
                for label, value in info_rows
            ]
        )
        info_html = f'''
        <div style="margin-top:8px; font-size:14px; line-height:1.7; color:#374151;">
            {info_text}
        </div>
        '''

    st.markdown(
        f'''
        <div style="border:1px solid #D1D5DB; border-radius:14px; padding:14px 18px; margin:10px 0 12px 0; background:#FFFFFF; box-shadow:0 1px 5px rgba(17,24,39,0.06);">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="display:inline-block; padding:3px 8px; border-radius:999px; background:#EEF2FF; color:#3730A3; font-size:12px; font-weight:800;">BATCH</span>
                <span style="font-size:17px; color:#111827; font-weight:800;">{_html_escape(title)}</span>
            </div>
            {subtitle_html}
            {desc_html}
            {info_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_job_operation_metadata(job: Dict[str, Any]) -> None:
    """이전 함수명과의 호환을 위해 카드 렌더링 함수로 위임한다."""
    render_batch_job_card(job)

def render_batch_process_block(result: Any) -> None:
    """배치 프로세스 화면 렌더링.

    핵심 원칙:
    - 구조화 JSON(steps/jobs/key_jobs)을 단일 source of truth로 사용한다.
    - result.answer는 fallback 문장일 수 있으므로 structured_data가 있으면 반복 출력하지 않는다.
    - 제목/핵심 배치/STEP 상세를 화면에서 한 번만 그린다.
    - 배치명이나 단계명을 코드에 박지 않고 JSON 메타데이터 기반으로 확장 가능하게 처리한다.
    """
    data = getattr(result, "structured_data", None) or {}
    batch_process = data.get("batch_process", {}) if "batch_process" in data else data

    title = batch_process.get("title", "배치 프로세스")
    steps = batch_process.get("steps", []) or []

    if not steps:
        if getattr(result, "answer", None):
            st.write(result.answer)
        else:
            st.info("배치 프로세스 정보가 없습니다.")
        return

    st.markdown(f"#### 📌 {title}")

    key_jobs: List[str] = []
    for step in steps:
        key_jobs.extend(step.get("key_jobs", []) or [])
    key_jobs = unique_preserve_order(key_jobs)

    if key_jobs:
        st.markdown("##### ⭐ 핵심 배치")
        _render_bullets([f"`{job}`" for job in key_jobs])

    flow_text = build_step_flow_text(steps)
    if flow_text:
        st.markdown("##### 🔹 한 줄 흐름")
        st.markdown(f"`{flow_text}`")

    st.markdown("##### 🔹 단계별 배치 프로세스")

    for step in steps:
        step_no = step.get("step", "")
        step_name = str(step.get("name", "")).strip()
        execution_label = format_execution_label(step.get("execution", ""))

        header = f"STEP {step_no}. {step_name}" if step_no != "" else step_name or "STEP"
        if execution_label:
            header = f"{header} ({execution_label})"
        
        st.markdown(
            f"""
            <div style="
                margin-top:18px;
                margin-bottom:10px;
                padding:10px 14px;
                border-radius:12px;
                background:linear-gradient(90deg, #EEF4FF 0%, #F8FAFC 100%);
                color:#1E3A8A;
                font-size:20px;
                font-weight:800;
                border-left:6px solid #2563EB;
                box-shadow:0 1px 4px rgba(37,99,235,0.08);
            ">
                {header}
            </div>
            """,
            unsafe_allow_html=True,
        )


        description = str(step.get("description") or "").strip()
        if description:
            st.markdown(f"👉 {description}")

        jobs = step.get("jobs", []) or []
        for job in jobs:
            render_batch_job_card(job)

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
        st.info(".env DB 설정과 실제 테이블/컬럼 구성을 확인하세요.")
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
        st.info(".env DB 설정과 실제 테이블/컬럼 구성을 확인하세요.")
        return

    empty_message = payload.get("empty_message")
    if empty_message:
        st.info(empty_message)
        return

    df = payload_to_dataframe(payload.get("dataframe"))
    if df.empty:
        policy = get_realtime_policy(query_meta)
        st.info(str(policy.get("empty_message") or "조회 결과가 없습니다."))
        return

    summary = payload.get("summary")
    if summary:
        st.write(summary)
    elif payload.get("summary_error"):
        st.warning(f"요약 생성 실패: {payload['summary_error']}")

    st.dataframe(df, use_container_width=True)
