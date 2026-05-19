"""
structured_renderers.py

Streamlit 구조화 응답 렌더러.
- 화면 함수는 result payload만 받아서 그림.
- intent별 분기는 renderer_registry.py에서 담당.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _render_metric_row(items: List[tuple[str, str]]) -> None:
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _render_chip_list(title: str, items: List[str]) -> None:
    clean_items = [str(item).strip() for item in items or [] if str(item).strip()]
    if not clean_items:
        return
    st.markdown(f"**{title}**")
    st.markdown("  ".join([f"`{item}`" for item in clean_items]))


def render_overview_block(result: Any) -> None:
    data: Dict[str, Any] = getattr(result, "structured_data", {}) or {}
    overview = data.get("overview", data)

    title = overview.get("title") or "업무 개요"
    summary = overview.get("summary") or overview.get("content") or getattr(result, "answer", "")
    st.subheader(title)
    if summary:
        st.info(summary)

    input_data = _as_list(overview.get("input_data"))
    target_transactions = _as_list(overview.get("target_transactions"))
    exclusions = _as_list(overview.get("exclusions"))
    outputs = _as_list(overview.get("outputs"))

    if any([input_data, target_transactions, exclusions, outputs]):
        _render_metric_row([
            ("입력 데이터", str(len(input_data))),
            ("대상 거래 유형", str(len(target_transactions))),
            ("제외/보정 항목", str(len(exclusions))),
            ("최종 산출물", str(len(outputs))),
        ])

    left, right = st.columns(2)
    with left:
        for title_text, items in [("주요 입력 데이터", input_data), ("주요 대상 거래", target_transactions)]:
            if items:
                st.markdown(f"**{title_text}**")
                for item in items:
                    st.markdown(f"- {item}")
    with right:
        for title_text, items in [("제외/보정 항목", exclusions), ("최종 산출물", outputs)]:
            if items:
                st.markdown(f"**{title_text}**")
                for item in items:
                    st.markdown(f"- {item}")

    answer = getattr(result, "answer", "")
    if answer and answer != summary:
        with st.expander("상세 설명"):
            st.write(answer)


def render_batch_process_block(result: Any) -> None:
    data: Dict[str, Any] = getattr(result, "structured_data", {}) or {}
    batch_process = data.get("batch_process", data)
    title = batch_process.get("title") or "배치 프로세스"
    steps = batch_process.get("steps", []) or []

    st.subheader(title)

    key_jobs: List[str] = []
    for step in steps:
        key_jobs.extend(step.get("key_jobs", []) or [])
    unique_key_jobs = sorted(set(str(job) for job in key_jobs if str(job).strip()))

    if steps:
        _render_metric_row([
            ("단계 수", str(len(steps))),
            ("핵심 배치", str(len(unique_key_jobs))),
            ("병렬 단계", str(sum(1 for s in steps if s.get("execution") == "parallel"))),
            ("순차 단계", str(sum(1 for s in steps if s.get("execution") == "sequential"))),
        ])

    if getattr(result, "answer", ""):
        st.info(result.answer)

    for step in steps:
        execution = str(step.get("execution", "")).strip()
        execution_label = {"parallel": "병렬", "sequential": "순차"}.get(execution, execution)
        with st.container(border=True):
            st.markdown(f"**[STEP {step.get('step', '')}] {step.get('name', '')} ({execution_label})**")
            if step.get("description"):
                st.caption(step.get("description"))
            if step.get("key_jobs"):
                _render_chip_list("핵심 배치", step.get("key_jobs", []))
            for job in step.get("jobs", []) or []:
                st.markdown(f"- `{job.get('job_id', '')}` : {job.get('description', '')}")


def render_graph_summary(result: Any, *, highlight_field: str = "highlight_nodes", highlight_title: str = "핵심 항목") -> None:
    data: Dict[str, Any] = getattr(result, "graph_data", None) or {}
    summary = getattr(result, "answer", "") or data.get("summary")
    if summary:
        st.info(summary)
    _render_chip_list(highlight_title, data.get(highlight_field, []))


def render_chart_summary(result: Any) -> None:
    if getattr(result, "answer", ""):
        st.info(result.answer)


def render_table_summary(result: Any) -> None:
    if getattr(result, "answer", ""):
        st.info(result.answer)
