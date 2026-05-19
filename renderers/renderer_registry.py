"""intent별 renderer 선택 registry."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .structured_renderers import (
    render_batch_process_block,
    render_chart_summary,
    render_graph_summary,
    render_overview_block,
    render_table_summary,
)


def _render_batch_flow(result: Any) -> None:
    render_graph_summary(result, highlight_field="highlight_nodes", highlight_title="핵심 배치")


def _render_table_lineage(result: Any) -> None:
    render_graph_summary(result, highlight_field="highlight_tables", highlight_title="핵심 테이블")


RENDERER_REGISTRY: Dict[str, Callable[[Any], None]] = {
    "overview": render_overview_block,
    "batch_process": render_batch_process_block,
    "batch_flow": _render_batch_flow,
    "table_lineage": _render_table_lineage,
    "billing_monthly_amount": render_chart_summary,
    "today_incidents": render_table_summary,
}


def render_by_intent(result: Any) -> None:
    renderer = RENDERER_REGISTRY.get(getattr(result, "intent", ""))
    if renderer:
        renderer(result)
    elif getattr(result, "answer", ""):
        # fallback renderer
        import streamlit as st
        st.write(result.answer)
