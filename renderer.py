from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _render_metric_row(items: List[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _render_chip_list(title: str, items: List[str]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    st.markdown("  ".join([f"`{item}`" for item in items]))


def render_overview_block(result: Any) -> None:
    data: Dict[str, Any] = getattr(result, "structured_data", {}) or {}
    title = data.get("title") or "업무 개요"
    st.subheader(title)
    summary = data.get("summary") or result.answer
    st.info(summary)

    input_data = data.get("input_data", [])
    exclusions = data.get("exclusions", [])
    outputs = data.get("outputs", [])
    target_transactions = data.get("target_transactions", [])

    if any([input_data, exclusions, outputs]):
        _render_metric_row([
            ("입력 데이터", str(len(input_data))),
            ("대상 거래 유형", str(len(target_transactions))),
            ("제외/보정 항목", str(len(exclusions))),
            ("최종 산출물", str(len(outputs))),
        ])

    left, right = st.columns(2)
    with left:
        if input_data:
            st.markdown("**주요 입력 데이터**")
            for item in input_data:
                st.markdown(f"- {item}")
        if target_transactions:
            st.markdown("**주요 대상 거래**")
            for item in target_transactions:
                st.markdown(f"- {item}")
    with right:
        if exclusions:
            st.markdown("**제외/보정 항목**")
            for item in exclusions:
                st.markdown(f"- {item}")
        if outputs:
            st.markdown("**최종 산출물**")
            for item in outputs:
                st.markdown(f"- {item}")

    if result.answer and result.answer != summary:
        with st.expander("상세 설명"):
            st.write(result.answer)


def render_batch_process_block(result: Any) -> None:
    data: Dict[str, Any] = getattr(result, "structured_data", {}) or {}
    title = data.get("title") or "배치 프로세스"
    steps = data.get("steps", [])
    st.subheader(title)

    key_jobs = []
    for step in steps:
        key_jobs.extend(step.get("key_jobs", []))

    if steps:
        _render_metric_row([
            ("단계 수", str(len(steps))),
            ("핵심 배치", str(len(set(key_jobs)))),
            ("병렬 단계", str(sum(1 for s in steps if s.get('execution') == 'parallel'))),
            ("순차 단계", str(sum(1 for s in steps if s.get('execution') == 'sequential'))),
        ])

    if result.answer:
        st.info(result.answer)

    for step in steps:
        execution = step.get("execution", "")
        execution_kr = "병렬" if execution == "parallel" else "순차" if execution == "sequential" else execution
        with st.container(border=True):
            st.markdown(f"**[STEP {step.get('step')}] {step.get('name', '')} ({execution_kr})**")
            if step.get("description"):
                st.caption(step.get("description"))
            if step.get("key_jobs"):
                _render_chip_list("핵심 배치", step.get("key_jobs", []))
            for job in step.get("jobs", []):
                st.markdown(f"- `{job.get('job_id', '')}` : {job.get('description', '')}")


def render_graph_summary(result: Any) -> None:
    data: Dict[str, Any] = result.graph_data or {}
    summary = result.answer or data.get("summary")
    if summary:
        st.info(summary)

    if result.intent == "batch_flow":
        _render_chip_list("핵심 배치", data.get("highlight_nodes", []))
    elif result.intent == "table_lineage":
        _render_chip_list("핵심 테이블", data.get("highlight_tables", []))


def render_chart_summary(result: Any) -> None:
    if result.answer:
        st.info(result.answer)


def render_table_summary(result: Any) -> None:
    if result.answer:
        st.info(result.answer)
