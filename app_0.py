from __future__ import annotations

import json
import os
import uuid
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph

load_dotenv()

from llm import ChatConfig, HandoverAgent, ollama_generate, apply_dictionary_rewrite, detect_intent, get_llm_engine_name, get_langchain_feature_flags
from realtime_query_service import RealtimeQueryService
from batch_dev import BatchDevAgent

try:
    from batch_dev.llm_batch_validator import validate_batch_generation
except Exception:
    validate_batch_generation = None

try:
    from batch_dev.sql_improvement_advisor import analyze_sql_improvement
except Exception:
    analyze_sql_improvement = None

PAGE_TITLE = "업무 인수인계 에이전트"
PAGE_ICON = "🤖"

JSON_PATH = os.getenv("JSON_PATH", "ingest/handover_improved.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "handover_agent")
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "3306").strip()
DB_SERVICE = os.getenv("DB_SERVICE", "").strip()

DATABASE_URL = ""
if all([DB_USER, DB_PASSWORD, DB_HOST, DB_SERVICE]):
    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
    )

BILLING_QUERY_ID = "billing_monthly_amount"
TODAY_INCIDENTS_QUERY_ID = "today_incidents"
BATCH_VALIDATION_USE_LLM = os.getenv("BATCH_VALIDATION_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}
BATCH_VALIDATION_LLM_MODEL = os.getenv("BATCH_VALIDATION_LLM_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")).strip()
BATCH_VALIDATION_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()

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
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "evaluation_mode" not in st.session_state:
        st.session_state.evaluation_mode = True

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
        raise RuntimeError("DB 접속 정보가 설정되지 않았습니다. .env에 DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SERVICE를 설정하세요.")
    return service.fetch_dataframe(query_meta)

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

def _get_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _safe_cell(row: pd.Series, column: Optional[str], default: str = "") -> str:
    if not column:
        return default
    value = row.get(column)
    if pd.isna(value) or value is None:
        return default
    text = str(value).strip()
    return text if text else default

def generate_incident_summary(df: pd.DataFrame) -> Optional[str]:
    """
    장애현황은 정형 데이터이므로 LLM 호출 없이 코드로 요약한다.
    - Ollama timeout 방지
    - 조회 결과가 있을 때 '오늘 장애는 없습니다'가 섞여 출력되는 문제 방지
    - SQL alias가 영문/한글이어도 동작하도록 주요 컬럼 후보를 함께 처리
    """
    if df.empty:
        return None

    batch_col = _get_first_existing_column(df, ["batch_name", "배치명"])
    error_code_col = _get_first_existing_column(df, ["error_code", "오류코드"])
    error_msg_col = _get_first_existing_column(df, ["error_message", "오류메시지", "오류내용"])
    action_detail_col = _get_first_existing_column(df, ["action_detail", "조치방법", "조치내용"])
    action_owner_col = _get_first_existing_column(df, ["action_owner", "담당자", "조치담당자"])

    total_count = len(df)
    batch_names = []
    error_messages = []
    action_lines = []
    owners = []

    for _, row in df.iterrows():
        batch_name = _safe_cell(row, batch_col, "배치명 없음")
        error_code = _safe_cell(row, error_code_col, "오류코드 없음")
        error_message = _safe_cell(row, error_msg_col, "오류 메시지 없음")
        action_detail = _safe_cell(row, action_detail_col, "등록된 조치 방법 없음")
        action_owner = _safe_cell(row, action_owner_col, "담당자 미지정")

        batch_names.append(batch_name)
        error_messages.append(error_message)
        action_lines.append(f"- {batch_name} ({error_code}): {action_detail}")
        owners.append(action_owner)

    unique_error_messages = list(dict.fromkeys(error_messages))
    unique_owners = [owner for owner in dict.fromkeys(owners) if owner != "담당자 미지정"]

    lines = [
        "장애 현황 조회 결과",
        "",
        f"전체 장애 건수: {total_count}건",
        "",
        "장애 배치명 목록:",
        *[f"- {name}" for name in batch_names],
        "",
        "주요 오류 원인 요약:",
        f"- {', '.join(unique_error_messages)}",
        "",
        "조치 방법 요약:",
        *action_lines,
        "",
        f"담당자: {', '.join(unique_owners) if unique_owners else '담당자 미지정'}",
        "",
        "확인 필요사항: 조치 후 배치 재실행 여부와 후속 배치 영향도를 확인하세요.",
    ]

    return "\n".join(lines)

def format_krw(amount: Any) -> str:
    """DB 금액은 원 단위 정수로 관리하고, 화면/요약에서만 사람이 읽기 좋은 형태로 변환한다."""
    amount_int = int(round(float(amount or 0)))
    if amount_int != 0 and amount_int % 10000 == 0:
        return f"{amount_int // 10000:,}만 원"
    return f"{amount_int:,}원"

def _billing_pattern_text(summary: Dict[str, Any]) -> str:
    """최근 증감률 기준으로 간단한 패턴 문구를 만든다."""
    change_rate_pct = summary.get("change_rate_pct")
    if change_rate_pct is None:
        return "데이터 패턴은 단일 구간이므로 증감 판단은 생략합니다."
    if change_rate_pct > 0:
        return "데이터 패턴은 최근 구간에서 증가하는 흐름입니다."
    if change_rate_pct < 0:
        return "데이터 패턴은 최근 구간에서 감소하는 흐름입니다."
    return "데이터 패턴은 최근 구간에서 전월과 동일한 흐름입니다."


def format_billing_month(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{6}", text):
        return f"{text[:4]}년 {int(text[4:6])}월"
    return text

def generate_billing_summary(query_meta: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    """
    청구 월별 금액 요약은 LLM이 숫자 단위를 오해하지 않도록 코드로 생성한다.
    - 계산은 summarize_billing_dataframe에서 수행
    - 금액 표시는 원 단위 기준으로 format_krw에서 수행
    """
    if df.empty:
        return None

    summary = summarize_billing_dataframe(
        df,
        x_field=query_meta.get("x_field", "billing_month"),
        y_field=query_meta.get("y_field", "amount"),
    )

    lines = [
        (
            f"전체 흐름 요약: 총 {summary['row_count']}개월치의 청구 이용내역서 월별 금액 조회 결과, "
            f"총액은 {format_krw(summary['total_amount'])}으로 확인됩니다."
        ),
        (
            f"최고 금액 구간: 최고 금액은 {format_krw(summary['max_amount'])}으로, "
            f"{format_billing_month(summary['max_period'])}에 발생했습니다."
        ),
        (
            f"최저 금액 구간: 최저 금액은 {format_krw(summary['min_amount'])}으로, "
            f"{format_billing_month(summary['min_period'])}에 발생했습니다."
        ),
    ]

    if summary.get("change_rate_pct") is not None:
        lines.append(
            f"최근 구간의 증감 포인트: 최근 구간인 {format_billing_month(summary['latest_period'])}에는 "
            f"{format_krw(summary['latest_amount'])}으로, 전월 대비 {summary['change_rate_pct']}% 변동했습니다."
        )

    lines.append(_billing_pattern_text(summary))
    return "\n\n".join(lines)

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

def dataframe_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    safe_df = df.where(pd.notnull(df), None)
    return {
        "columns": safe_df.columns.tolist(),
        "rows": safe_df.to_dict(orient="records"),
    }

def payload_to_dataframe(payload: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("rows", [])
    columns = payload.get("columns", [])
    return pd.DataFrame(rows, columns=columns)

def build_realtime_payload(query_meta: Dict[str, Any], render_type: str, realtime_mode: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query_id": query_meta.get("query_id"),
        "render_type": render_type,
        "summary": None,
        "dataframe": None,
        "empty_message": None,
        "error": None,
    }

    try:
        df = fetch_realtime_dataframe(query_meta)
    except Exception as e:
        payload["error"] = str(e)
        return payload

    payload["dataframe"] = dataframe_to_payload(df)

    if df.empty:
        if query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID:
            payload["empty_message"] = "오늘 장애는 없습니다."
        else:
            payload["empty_message"] = "조회 결과가 없습니다."
        return payload

    try:
        query_id = query_meta.get("query_id")
        if render_type == "table" and (realtime_mode == "incident_table_with_summary" or query_id == TODAY_INCIDENTS_QUERY_ID):
            payload["summary"] = generate_incident_summary(df)
        elif render_type == "chart" and query_id == BILLING_QUERY_ID:
            payload["summary"] = generate_billing_summary(query_meta, df)
    except Exception as e:
        payload["summary_error"] = str(e)

    return payload

def enrich_result_with_realtime_payload(result: Any) -> Any:
    if getattr(result, "render_type", None) not in {"table", "chart"} or not getattr(result, "query_meta", None):
        return result
    realtime_payload = build_realtime_payload(
        query_meta=result.query_meta,
        render_type=result.render_type,
        realtime_mode=getattr(result, "realtime_mode", None),
    )
    setattr(result, "realtime_payload", realtime_payload)
    return result




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

    section_specs = [
        ("input_data", "주요 입력 데이터", "📥"),
        ("target_transactions", "주요 대상 거래", "🎯"),
        ("exclusions", "제외 및 보정 항목", "🚫"),
        ("outputs", "최종 산출물", "📤"),
        ("key_points", "핵심 포인트", "⭐"),
    ]

    rendered_keys = {"title", "summary", "content"}
    visible_sections: List[tuple[str, str, List[str]]] = []

    for key, label, icon in section_specs:
        rendered_keys.add(key)
        items = as_list(overview.get(key))
        if items:
            visible_sections.append((label, icon, items))

    # 향후 overview JSON 필드가 추가되어도 기본 섹션으로 표시한다.
    label_map = {
        "owner": "담당자",
        "owner_team": "담당팀",
        "cycle": "처리 주기",
        "notes": "참고사항",
    }
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

    schedule_type = str(job.get("schedule_type", "")).strip()
    execution_time = str(job.get("execution_time", "")).strip()
    avg_duration = format_duration_sec(job.get("avg_duration_sec"))
    batch_file = str(job.get("batch_file", "")).strip()
    owner_team = str(job.get("owner_team", "")).strip()

    title = job_id or job_name or "배치명 없음"
    subtitle = job_name if job_name and job_name != job_id else ""

    info_rows = []
    if schedule_type:
        info_rows.append(("배치주기", schedule_type))
    if execution_time:
        info_rows.append(("실행시간", execution_time))
    if avg_duration:
        info_rows.append(("평균수행시간", avg_duration))
    if batch_file:
        info_rows.append(("실행파일", batch_file))
    if owner_team:
        info_rows.append(("담당자", owner_team))

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
        if query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID:
            st.info("오늘 장애는 없습니다.")
        else:
            st.info("조회 결과가 없습니다.")
        return

    summary = payload.get("summary")
    if summary:
        st.write(summary)
    elif payload.get("summary_error") and (realtime_mode == "incident_table_with_summary" or query_meta.get("query_id") == TODAY_INCIDENTS_QUERY_ID):
        st.warning(f"LLM 요약 생성 실패: {payload['summary_error']}")

    st.dataframe(df, use_container_width=True)

def _extract_where_filter(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "where_filter =" in log:
            return log.split("where_filter =", 1)[1].strip()
    return "(where 조건 없음 또는 default 검색)"

def _extract_search_query(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "search_query =" in log:
            return log.split("search_query =", 1)[1].strip()
    return ""

def build_used_json_view(result: Any) -> Dict[str, Any]:
    intent = getattr(result, "intent", None)
    structured_data = getattr(result, "structured_data", None) or {}
    graph_data = getattr(result, "graph_data", None) or {}
    query_meta = getattr(result, "query_meta", None) or {}

    if intent == "overview":
        overview = structured_data.get("overview", structured_data)
        return {
            "json_path": "domains[].systems[].overview",
            "used_fields": {
                "title": overview.get("title"),
                "summary": overview.get("summary"),
                "content": overview.get("content"),
                "input_data": overview.get("input_data", []),
                "target_transactions": overview.get("target_transactions", []),
                "exclusions": overview.get("exclusions", []),
                "outputs": overview.get("outputs", []),
                "key_points": overview.get("key_points", []),
            },
        }

    if intent == "batch_process":
        batch_process = structured_data.get("batch_process", structured_data)
        return {
            "json_path": "domains[].systems[].batch_process",
            "used_fields": {
                "title": batch_process.get("title"),
                "steps": batch_process.get("steps", []),
            },
        }

    if intent == "batch_flow":
        return {
            "json_path": "domains[].systems[].batch_flow",
            "used_fields": {
                "title": graph_data.get("title"),
                "summary": graph_data.get("summary"),
                "start_nodes": graph_data.get("start_nodes", []),
                "highlight_nodes": graph_data.get("highlight_nodes", []),
                "end_nodes": graph_data.get("end_nodes", []),
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
            },
        }

    if intent == "table_lineage":
        return {
            "json_path": "domains[].systems[].table_lineage",
            "used_fields": {
                "title": graph_data.get("title"),
                "summary": graph_data.get("summary"),
                "highlight_tables": graph_data.get("highlight_tables", []),
                "source_tables": graph_data.get("source_tables", []),
                "result_tables": graph_data.get("result_tables", []),
                "tables": graph_data.get("tables", []),
                "edges": graph_data.get("edges", []),
            },
        }

    if query_meta:
        return {"json_path": "realtime_queries[]", "used_fields": query_meta}

    return {"json_path": "(확인 불가)", "used_fields": {}}

def build_evaluation_checks(result: Any) -> List[str]:
    """
    평가용 문구 생성
    - 시스템별 RAG 질문과 realtime_query 질문을 분리해서 평가한다.
    - realtime_query는 특정 system_id 기준 검색이 아니므로 "다른 시스템 source" 검사를 하지 않는다.
    """
    system_id = getattr(result, "system_id", None)
    intent = getattr(result, "intent", None)
    sources = getattr(result, "sources", []) or []
    checks: List[str] = []

    realtime_intents = {"billing_monthly_amount", "today_incidents"}
    system_intents = {"overview", "batch_process", "batch_flow", "table_lineage"}

    if intent in realtime_intents:
        wrong_realtime_sources = [
            s for s in sources
            if s.get("section") and s.get("section") != "realtime_query"
        ]
        checks.append(
            "✅ realtime_query 기준으로 조회 정의가 선택됨"
            if not wrong_realtime_sources
            else "⚠️ realtime_query가 아닌 source가 포함됨"
        )

        query_meta = getattr(result, "query_meta", None) or {}
        query_id = query_meta.get("query_id")
        wrong_query_sources = [
            s for s in sources
            if s.get("query_id") and query_id and s.get("query_id") != query_id
        ]
        checks.append(
            f"✅ query_id 기준으로 {query_id or '대상 조회'}가 정확히 매칭됨"
            if not wrong_query_sources
            else "⚠️ 다른 query_id source가 섞였는지 확인 필요"
        )

    elif intent in system_intents:
        if system_id:
            mixed_sources = [
                s for s in sources
                if s.get("system_id") and s.get("system_id") != system_id
            ]
            checks.append(
                "✅ system_id 기준으로 대상 시스템이 분리됨"
                if not mixed_sources
                else "⚠️ 다른 시스템 source가 섞였는지 확인 필요"
            )
        else:
            checks.append("⚠️ 시스템별 질문인데 system_id가 확인되지 않음")

        wrong_section = [
            s for s in sources
            if s.get("section") and s.get("section") != intent
        ]
        checks.append(
            "✅ intent에 맞는 section을 사용함"
            if not wrong_section
            else "⚠️ 의도와 다른 section source가 포함됨"
        )

    else:
        checks.append("ℹ️ 기본 검색 질문으로 처리됨")

    if getattr(result, "structured_data", None) or getattr(result, "graph_data", None) or getattr(result, "query_meta", None):
        checks.append("✅ 답변 근거가 원본 JSON 구조와 연결됨")

    checks.append(
        f"✅ retrieval 근거 {len(sources)}건 확인 가능"
        if sources
        else "ℹ️ 구조화 JSON 직접 렌더링 중심이라 retrieval source가 없을 수 있음"
    )
    return checks

def render_batch_development_evaluation_panel(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    batch_spec = payload.get("batch_spec", {}) or {}

    with st.expander("📊 배치개발 평가용 근거 확인", expanded=False):
        st.markdown("##### 1) 요청 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
            "success": payload.get("success"),
        })

        st.markdown("##### 2) 생성된 배치 명세")
        st.json({
            "batch_id": batch_spec.get("batch_id"),
            "batch_name": batch_spec.get("batch_name"),
            "batch_type": batch_spec.get("batch_type"),
            "source": batch_spec.get("source"),
            "target": batch_spec.get("target"),
        })

        st.markdown("##### 3) 사용한 ERWIN 메타")
        st.json(batch_spec.get("meta_source", {}))

        st.markdown("##### 4) 사용한 Rule / SQL Template")
        st.json(batch_spec.get("rule_source", {}))

        st.markdown("##### 5) 생성 SQL")
        st.code(batch_spec.get("sql", ""), language="sql")

        st.markdown("##### 6) 생성 파일")
        for file_path in payload.get("created_files", []):
            st.code(file_path, language="text")

        st.markdown("##### 7) 기본 검증 결과")
        st.json({
            "warnings": payload.get("warnings", []),
            "errors": payload.get("errors", []),
        })

        validation_report = payload.get("validation_report")
        if validation_report:
            st.markdown("##### 8) LLM 해석/검증 요약")
            st.json({
                "valid": validation_report.get("valid"),
                "score": validation_report.get("score"),
                "summary": validation_report.get("summary"),
                "interpretation": validation_report.get("interpretation"),
                "warnings": validation_report.get("warnings", []),
                "recommendations": validation_report.get("recommendations", []),
            })

        sql_improvement = payload.get("sql_improvement")
        if sql_improvement:
            st.markdown("##### 9) SQL 자동 개선 제안")
            st.json({
                "enabled": sql_improvement.get("enabled"),
                "risk_level": sql_improvement.get("risk_level"),
                "generated_by": sql_improvement.get("generated_by"),
                "summary": sql_improvement.get("summary"),
                "suggestions": sql_improvement.get("suggestions", []),
                "warnings": sql_improvement.get("warnings", []),
            })

def render_evaluation_panel(result: Any) -> None:
    if not st.session_state.evaluation_mode:
        return

    if getattr(result, "intent", None) == "batch_development":
        render_batch_development_evaluation_panel(result)
        return

    used_json = build_used_json_view(result)
    sources = getattr(result, "sources", []) or []
    debug_logs = getattr(result, "debug_logs", []) or []

    with st.expander("📊 평가용 근거 확인", expanded=False):
        st.markdown("##### 1) 질문 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
        })

        st.markdown("##### 2) Builder 검색 조건")
        st.code(_extract_where_filter(debug_logs), language="python")

        search_query = _extract_search_query(debug_logs)
        if search_query:
            st.markdown("##### 3) 실제 검색 질문")
            st.code(search_query, language="text")

        st.markdown("##### 4) Vector DB에서 가져온 근거 chunk")
        if sources:
            st.dataframe(pd.DataFrame(sources), use_container_width=True)
        else:
            st.info("검색 chunk가 없거나, 구조화 JSON을 직접 사용한 응답입니다.")

        st.markdown("##### 5) 답변에 사용된 원본 JSON 데이터")
        st.caption(f"JSON 위치: {used_json.get('json_path')}")
        st.json(used_json.get("used_fields", {}))

        st.markdown("##### 6) 간단 평가")
        for check in build_evaluation_checks(result):
            st.markdown(f"- {check}")


def _read_generated_file(file_path: Any) -> tuple[str, str] | None:
    """생성된 파일 경로를 읽어서 validator 입력 형식으로 변환한다.

    BatchDevAgent가 created_files에 파일 경로만 넘기므로,
    app.py에서는 실제 파일 내용을 읽어 validator에 전달한다.
    파일이 없거나 읽을 수 없으면 해당 파일은 건너뛰어 화면 장애를 막는다.
    """
    try:
        path = Path(str(file_path))
        if not path.exists() or not path.is_file():
            return None
        return path.name, path.read_text(encoding="utf-8")
    except Exception:
        return None


def _build_generated_files_for_validation(batch_spec: Dict[str, Any], created_files: List[Any]) -> Dict[str, str]:
    """LLM 검증 모듈에 넘길 생성 파일 묶음을 만든다.

    특정 파일명을 하드코딩해서 판단하지 않고, BatchDevAgent가 돌려준
    created_files 목록을 기준으로 실제 파일 내용을 수집한다.
    단, query.sql이 created_files에 없더라도 batch_spec.sql이 있으면
    검증 정확도를 위해 query.sql 항목으로 보강한다.
    """
    generated_files: Dict[str, str] = {}

    for file_path in created_files or []:
        item = _read_generated_file(file_path)
        if item is None:
            continue
        file_name, content = item
        generated_files[file_name] = content

    if "query.sql" not in generated_files and batch_spec.get("sql"):
        generated_files["query.sql"] = str(batch_spec.get("sql") or "")

    generated_files.setdefault(
        "batch_spec.json",
        json.dumps(batch_spec or {}, ensure_ascii=False, indent=2),
    )
    return generated_files


def _resolve_validation_output_dir(created_files: List[Any]) -> Path | None:
    """validation_report.json/md를 저장할 폴더를 찾는다."""
    for file_path in created_files or []:
        try:
            path = Path(str(file_path))
            if path.exists():
                return path.parent if path.is_file() else path
        except Exception:
            continue
    return None



def run_batch_sql_improvement(dev_result: Any) -> Dict[str, Any] | None:
    """생성된 배치 SQL에 대해 자동 개선 제안을 생성한다.

    실무 적용 원칙:
    - query.sql 생성 이후에만 실행한다.
    - SQL을 직접 수정하지 않고 개선 후보만 제안한다.
    - LLM 모듈이 없거나 실패해도 룰 기반 분석 결과를 반환한다.
    - 테이블/컬럼은 batch_spec.sql, meta_source, created_files에서 읽어 하드코딩을 줄인다.
    """
    if analyze_sql_improvement is None:
        return {
            "enabled": False,
            "risk_level": "UNKNOWN",
            "summary": "sql_improvement_advisor 모듈을 찾지 못했습니다.",
            "suggestions": [],
            "warnings": ["batch_dev/sql_improvement_advisor.py 파일 위치를 확인하세요."],
            "generated_by": "none",
        }

    batch_spec = getattr(dev_result, "batch_spec", {}) or {}
    created_files = getattr(dev_result, "created_files", []) or []
    generated_files = _build_generated_files_for_validation(batch_spec, created_files)
    output_dir = _resolve_validation_output_dir(created_files)

    try:
        return analyze_sql_improvement(
            batch_spec=batch_spec,
            generated_files=generated_files,
            llm_generate_fn=ollama_generate,
            model=BATCH_VALIDATION_LLM_MODEL,
            use_llm=BATCH_VALIDATION_USE_LLM,
            output_dir=output_dir,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "risk_level": "UNKNOWN",
            "summary": "SQL 자동 개선 제안 생성에 실패했습니다.",
            "suggestions": [],
            "warnings": [str(exc)],
            "generated_by": "error",
        }


def render_sql_improvement_report(sql_improvement: Dict[str, Any] | None, *, max_items: int | None = None) -> None:
    """SQL 자동 개선 제안을 Streamlit 화면에 표시한다.

    출력 결과 화면이 길어지지 않도록 기본은 접힘 상태로 보여준다.
    """
    if not sql_improvement:
        return

    with st.expander("🚀 SQL 자동 개선 제안", expanded=False):
        generated_by = sql_improvement.get("generated_by", "-")
        risk_level = sql_improvement.get("risk_level", "-")
        summary = sql_improvement.get("summary", "")

        if risk_level == "HIGH":
            st.error(f"위험도: {risk_level} / 생성방식: {generated_by}")
        elif risk_level in {"MEDIUM", "UNKNOWN"}:
            st.warning(f"위험도: {risk_level} / 생성방식: {generated_by}")
        else:
            st.success(f"위험도: {risk_level} / 생성방식: {generated_by}")

        if summary:
            st.write(summary)

        warnings = sql_improvement.get("warnings") or []
        for warning in warnings:
            st.caption(f"⚠️ {warning}")

        suggestions = sql_improvement.get("suggestions") or []
        if max_items is not None:
            suggestions = suggestions[:max_items]

        for idx, item in enumerate(suggestions, start=1):
            title = str(item.get("type") or "RECOMMENDATION").strip()
            target = str(item.get("target") or "").strip()
            reason = str(item.get("reason") or "").strip()
            recommendation = str(item.get("recommendation") or "").strip()
            sql = str(item.get("sql") or "").strip()

            with st.container(border=True):
                st.markdown(f"**{idx}. {title}**")
                if target:
                    st.markdown(f"**대상:** `{target}`")
                if reason:
                    st.markdown(f"**이유:** {reason}")
                if recommendation:
                    st.markdown(f"**개선안:** {recommendation}")
                if sql:
                    language = "sql" if any(token in sql.upper() for token in ["SELECT", "CREATE", "INDEX", "WHERE", "JOIN"]) else "text"
                    st.code(sql, language=language)

def run_batch_llm_validation(user_question: str, dev_result: Any) -> Dict[str, Any] | None:
    """배치 생성 결과에 대해 룰 검증 + 선택적 LLM 검증을 수행한다.

    - validator 모듈이 없으면 앱 전체가 죽지 않도록 경고 payload만 반환한다.
    - Ollama/LLM 호출이 실패해도 룰 기반 검증으로 한 번 더 검증한다.
    - 결과는 dict로 저장해서 Streamlit session_state/history에 그대로 보관한다.
    """
    if validate_batch_generation is None:
        return {
            "valid": False,
            "score": 0.0,
            "summary": "llm_batch_validator 모듈을 찾지 못했습니다.",
            "interpretation": "batch_dev 폴더에 llm_batch_validator.py가 있는지 확인하세요.",
            "checks": [],
            "issues": ["검증 모듈 import 실패"],
            "warnings": [],
            "recommendations": ["from batch_dev.llm_batch_validator import validate_batch_generation 경로를 확인하세요."],
        }

    batch_spec = getattr(dev_result, "batch_spec", {}) or {}
    created_files = getattr(dev_result, "created_files", []) or []
    generated_files = _build_generated_files_for_validation(batch_spec, created_files)
    output_dir = _resolve_validation_output_dir(created_files)

    # llm_batch_validator.py 내부에서 프로젝트 공통 llm.py의 ollama_generate를 재사용한다.
    # app.py에서는 별도 Ollama client를 만들지 않는다.
    try:
        report = validate_batch_generation(
            request_text=user_question,
            batch_spec=batch_spec,
            generated_files=generated_files,
            llm_client=None,
            output_dir=output_dir,
        )
        return report.to_dict()
    except Exception as llm_error:
        # LLM 호출/응답 파싱 오류가 나도 검증 화면 자체는 유지한다.
        try:
            report = validate_batch_generation(
                request_text=user_question,
                batch_spec=batch_spec,
                generated_files=generated_files,
                llm_client=None,
                output_dir=output_dir,
            )
            payload = report.to_dict()
            payload.setdefault("warnings", [])
            payload["warnings"].append(f"LLM 검증 실패로 룰 기반 검증만 수행했습니다: {llm_error}")
            return payload
        except Exception as rule_error:
            return {
                "valid": False,
                "score": 0.0,
                "summary": "배치 검증 리포트 생성에 실패했습니다.",
                "interpretation": "생성 파일 경로, batch_spec 구조, validator 모듈을 확인하세요.",
                "checks": [],
                "issues": [str(rule_error)],
                "warnings": [f"LLM 검증 오류: {llm_error}"],
                "recommendations": ["created_files 경로가 실제 파일로 존재하는지 확인하세요."],
            }

def run_batch_development(user_question: str) -> Any:
    """
    배치 개발 요청은 기존 RAG 흐름과 분리해서 처리한다.
    기존 HandoverAgent/Chroma 검색 품질에 영향을 주지 않기 위한 별도 진입점이다.
    """
    dev_result = BatchDevAgent().run(user_question)
    sql_improvement = run_batch_sql_improvement(dev_result) if dev_result.success else None
    validation_report = run_batch_llm_validation(user_question, dev_result) if dev_result.success else None
    answer = dev_result.message
    if dev_result.errors:
        answer = "배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요."
    return SimpleNamespace(
        answer=answer,
        intent="batch_development",
        render_type="batch_dev",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=apply_dictionary_rewrite(user_question),
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            "[BATCH_DEV 1] intent=batch_development",
            f"[BATCH_DEV 2] success={dev_result.success}",
            f"[BATCH_DEV 3] created_files={len(dev_result.created_files)}",
        ],
        batch_dev_result={
            "batch_spec": dev_result.batch_spec,
            "created_files": dev_result.created_files,
            "warnings": dev_result.warnings,
            "errors": dev_result.errors,
            "message": dev_result.message,
            "success": dev_result.success,
            "validation_report": validation_report,
            "sql_improvement": sql_improvement,
        },
    )

def render_batch_development_result(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    if not payload:
        st.warning("배치 개발 결과가 없습니다.")
        return

    success = payload.get("success")
    st.markdown("#### 🛠️ 배치 개발 결과")
    if success:
        st.success(payload.get("message", "배치 소스가 생성되었습니다."))
    else:
        st.error(payload.get("message", "배치 소스 생성에 실패했습니다."))

    errors = payload.get("errors") or []
    warnings = payload.get("warnings") or []
    if errors:
        st.markdown("##### ❌ 오류")
        for item in errors:
            st.markdown(f"- {item}")
    if warnings:
        st.markdown("##### ⚠️ 검토 필요")
        for item in warnings:
            st.markdown(f"- {item}")

    sql_improvement = payload.get("sql_improvement")
    if sql_improvement:
        render_sql_improvement_report(sql_improvement, max_items=5)

    # 상세 batch_spec / 생성 파일 목록은 화면에서 숨긴다.
    # 필요 시 generated 폴더의 batch_spec.json, validation_report.json 파일로 확인한다.
    validation_report = payload.get("validation_report")
    if validation_report:
        with st.expander("🔍 LLM 해석/검증 결과", expanded=False):
            is_valid = bool(validation_report.get("valid"))
            score = validation_report.get("score", 0)
            summary = validation_report.get("summary", "")
            interpretation = validation_report.get("interpretation", "")

            if is_valid:
                st.success(f"검증 통과 - score={score}")
            else:
                st.warning(f"확인 필요 - score={score}")

            if summary:
                st.markdown("**요약**")
                st.write(summary)
            if interpretation:
                st.markdown("**배치 해석**")
                st.write(interpretation)

            recommendations = validation_report.get("recommendations") or []
            if recommendations:
                st.markdown("**권장사항**")
                for item in recommendations[:5]:
                    st.markdown(f"- {item}")

    st.info("운영 반영 전 query.sql, 컬럼, 인덱스, 검증조건, 파일 포맷을 개발자가 반드시 검토하세요.")

def render_agent_result(result: Any) -> None:
    with st.chat_message("assistant"):
        if result.intent == "batch_development":
            render_batch_development_result(result)
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

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    init_session_state()
    st.title("🤖 업무 인수인계 에이전트")
    st.caption("인수인계 문서 검색, 흐름도/리니지 시각화, DB 조회형 질문을 처리합니다.")

    with st.sidebar:
        st.markdown("### 설정")
        st.session_state.evaluation_mode = st.checkbox("평가용 근거 보기", value=st.session_state.evaluation_mode)

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
        normalized_for_intent = apply_dictionary_rewrite(user_question)
        initial_intent = detect_intent(normalized_for_intent)
        chat_history = build_chat_history(st.session_state.message_list[:-1])
        with st.spinner("답변을 생성하는 중입니다..."):
            if initial_intent == "batch_development":
                result = run_batch_development(user_question)
            else:
                agent = get_agent()
                result = agent.answer_question(question=user_question, chat_history=chat_history)
                result = enrich_result_with_realtime_payload(result)
            current_message_id = str(uuid.uuid4())
            setattr(result, "message_id", current_message_id)
            render_agent_result(result)
        st.session_state.message_list.append({
            "message_id": current_message_id,
            "role": "assistant",
            "content": result.answer,
            "intent": result.intent,
            "render_type": result.render_type,
            "graph_data": result.graph_data,
            "query_meta": result.query_meta,
            "realtime_mode": getattr(result, "realtime_mode", None),
            "realtime_payload": getattr(result, "realtime_payload", None),
            "sources": result.sources,
            "structured_data": getattr(result, "structured_data", None),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "debug_logs": getattr(result, "debug_logs", []),
            "batch_dev_result": getattr(result, "batch_dev_result", None),
        })

if __name__ == "__main__":
    main()
