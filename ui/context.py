from __future__ import annotations
import json
import os
import uuid
import re
import logging
from urllib.parse import quote_plus
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph

try:
    from langgraph.graph import StateGraph, START, END
except Exception:
    # LangGraph 미설치 환경에서도 기존 앱은 동작하도록 optional import 처리한다.
    StateGraph = None
    START = "__start__"
    END = "__end__"

load_dotenv()

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "conf"))

SQL_ANALYSIS_SECTION_ALIASES_PATH = (
    CONFIG_DIR / "sql_analysis_section_aliases.json"
)

REALTIME_QUERY_POLICY_PATH = (
    CONFIG_DIR / "realtime_query_policy.json"
)

APP_RENDER_SCHEMA_PATH = (
    CONFIG_DIR / "app_render_schema.json"
)

SQL_REVIEW_POLICY_PATH = (
    CONFIG_DIR / "sql_review_policy.json"
)

AGENT_GRAPH_POLICY_PATH = (
    CONFIG_DIR / "agent_graph_policy.json"
)

AGENT_INTENT_REGISTRY_PATH = (
    CONFIG_DIR / "agent_intent_registry.json"
)

from llm import ChatConfig, HandoverAgent, ollama_generate, apply_dictionary_rewrite, detect_intent, get_llm_engine_name, get_langchain_feature_flags
from realtime_query_service import RealtimeQueryService
from batch_dev import BatchDevAgent
try:
    from batch_dev.config import BATCH_SQL_IMPROVEMENT_ENABLED
except Exception:
    # config.py 반영 전에도 앱이 깨지지 않도록 환경변수 fallback을 둔다.
    # 기본값은 false: 배치 개발 요청 화면에서 SQL 자동 개선 제안을 실행/표시하지 않는다.
    BATCH_SQL_IMPROVEMENT_ENABLED = os.getenv(
        "BATCH_SQL_IMPROVEMENT_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes", "y"}
from batch_dev.llm_batch_validator import validate_batch_generation
from batch_dev.sql_improvement_advisor import analyze_sql_improvement

PAGE_TITLE = "업무 인수인계 에이전트"
PAGE_ICON = "🤖"

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("handover_app")

REALTIME_MAX_ROWS = int(os.getenv("REALTIME_MAX_ROWS", "1000"))

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
    # DB 계정/비밀번호에 @, #, %, / 같은 특수문자가 있어도 안전하게 접속 URL을 만든다.
    safe_user = quote_plus(DB_USER)
    safe_password = quote_plus(DB_PASSWORD)
    DATABASE_URL = (
        f"mysql+pymysql://{safe_user}:{safe_password}"
        f"@{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
    )

BATCH_VALIDATION_USE_LLM = os.getenv("BATCH_VALIDATION_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}
BATCH_VALIDATION_LLM_MODEL = os.getenv("BATCH_VALIDATION_LLM_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")).strip()
BATCH_VALIDATION_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()

SQL_ANALYSIS_USE_LLM = os.getenv("SQL_ANALYSIS_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}
SQL_ANALYSIS_LLM_MODEL = os.getenv("SQL_ANALYSIS_LLM_MODEL", BATCH_VALIDATION_LLM_MODEL).strip()
SQL_ANALYSIS_LLM_TIMEOUT = int(os.getenv("SQL_ANALYSIS_LLM_TIMEOUT", os.getenv("BATCH_VALIDATION_LLM_TIMEOUT", os.getenv("UPSTAGE_CHAT_TIMEOUT", "60"))))
SQL_ANALYSIS_TEMPERATURE = float(os.getenv("SQL_ANALYSIS_TEMPERATURE", os.getenv("UPSTAGE_TEMPERATURE", "0.1")))
SQL_ANALYSIS_MAX_TOKENS = int(os.getenv("SQL_ANALYSIS_MAX_TOKENS", os.getenv("UPSTAGE_MAX_TOKENS", "2048")))

# 업무 범위 밖 질문 처리 정책
# - false: 기존처럼 지원 범위 안내만 표시
# - true: 일반 질문은 LLM fallback으로 답변 시도
GENERAL_FALLBACK_USE_LLM = os.getenv("GENERAL_FALLBACK_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}
GENERAL_FALLBACK_MODEL = os.getenv("GENERAL_FALLBACK_MODEL", os.getenv("UPSTAGE_CHAT_MODEL", BATCH_VALIDATION_LLM_MODEL)).strip()
GENERAL_FALLBACK_TIMEOUT = int(os.getenv("GENERAL_FALLBACK_TIMEOUT", os.getenv("UPSTAGE_CHAT_TIMEOUT", "60")))
GENERAL_FALLBACK_TEMPERATURE = float(os.getenv("GENERAL_FALLBACK_TEMPERATURE", "0.2"))
GENERAL_FALLBACK_MAX_TOKENS = int(os.getenv("GENERAL_FALLBACK_MAX_TOKENS", os.getenv("UPSTAGE_MAX_TOKENS", "2048")))

DEFAULT_AGENT_INTENT_REGISTRY: Dict[str, Dict[str, Any]] = {}


# LangGraph route 정책은 intent registry를 기준으로 생성한다.
# 새 intent 추가 시 agent_intent_registry.json만 확장하면 기본 라우팅도 함께 확장된다.
SUPPORTED_AGENT_INTENTS = set(DEFAULT_AGENT_INTENT_REGISTRY.keys())

LANGGRAPH_ENABLED = os.getenv("LANGGRAPH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}

DEFAULT_AGENT_GRAPH_POLICY = {
    "routes": [
        {
            "name": "sql_analysis",
            "intents": [intent for intent, spec in DEFAULT_AGENT_INTENT_REGISTRY.items() if spec.get("node") == "sql_analysis"],
            "node": "sql_analysis",
        },
        {
            "name": "batch_development",
            "intents": [intent for intent, spec in DEFAULT_AGENT_INTENT_REGISTRY.items() if spec.get("node") == "batch_development"],
            "node": "batch_development",
        },
        {
            "name": "handover_agent",
            "intents": [intent for intent, spec in DEFAULT_AGENT_INTENT_REGISTRY.items() if spec.get("node") == "handover_agent"],
            "node": "handover_agent",
        },
    ],
    "fallback_node": "general_fallback",
    "force_sql_analysis_node": "sql_analysis",
    # LangGraph 후처리 정책
    # - app.py에 업무별 분기 로직을 더 늘리지 않고, 정책 파일에서 노드 사용 여부를 제어한다.
    # - batch_development는 생성 이후 검증/개선 노드를 통과한다.
    "post_nodes": {
        "default": ["finalize"],
        "batch_development": ["batch_validation", "sql_improvement", "finalize"],
    },
    "node_options": {
        "batch_validation": {"enabled": True},
        "sql_improvement": {"enabled_env": "BATCH_SQL_IMPROVEMENT_ENABLED"},
        "finalize": {"enabled": True},
    },
}


DEFAULT_SQL_ANALYSIS_SECTION_ALIASES = {
    "request_type": ["요청유형", "요청 유형", "REQUEST_TYPE", "TYPE"],
    "business_name": ["업무명", "업무 명", "BUSINESS_NAME", "업무"],
    "system_name": ["시스템", "시스템명", "SYSTEM", "SYSTEM_NAME"],
    "sql_id": ["SQL_ID", "SQLID", "쿼리ID", "QUERY_ID"],
    "sql_description": ["SQL설명", "SQL 설명", "쿼리설명", "QUERY_DESCRIPTION"],
    "sql": ["SQL", "쿼리", "QUERY", "원본SQL", "원본 SQL"],
    "change_request": ["수정내용", "수정 내용", "변경요청", "변경 요청", "개선요청", "개선 요청", "요청사항", "요청 사항"],
    "operation_info": ["운영정보", "운영 정보", "실행정보", "실행 정보", "OPERATION_INFO"],
    "review_points": ["검토포인트", "검토 포인트", "점검항목", "점검 항목", "REVIEW_POINTS"],
    "output_request": ["출력요청", "출력 요청", "OUTPUT_REQUEST"],
    "context": ["업무내용", "업무 내용", "배경", "목적", "비고", "참고사항", "참고 사항"],
}

DEFAULT_REALTIME_QUERY_POLICY = {
    "default": {
        "empty_message": "조회 결과가 없습니다.",
        "summary_handler": None,
        "summary_type": None,  # 기존 설정 호환용 alias
    },
}

DEFAULT_APP_RENDER_SCHEMA = {
    "overview_sections": [
        {"key": "input_data", "label": "주요 입력 데이터", "icon": "📥"},
        {"key": "target_transactions", "label": "주요 대상 거래", "icon": "🎯"},
        {"key": "exclusions", "label": "제외 및 보정 항목", "icon": "🚫"},
        {"key": "outputs", "label": "최종 산출물", "icon": "📤"},
        {"key": "key_points", "label": "핵심 포인트", "icon": "⭐"},
    ],
    "overview_extra_labels": {
        "owner": "담당자",
        "owner_team": "담당팀",
        "cycle": "처리 주기",
        "notes": "참고사항",
    },
    "batch_job_info_fields": [
        {"key": "schedule_type", "label": "배치주기"},
        {"key": "execution_time", "label": "실행시간"},
        {"key": "avg_duration_sec", "label": "평균수행시간", "formatter": "duration_sec"},
        {"key": "batch_file", "label": "실행파일"},
        {"key": "owner_team", "label": "담당자"},
    ],
}


DEFAULT_SQL_REVIEW_POLICY = {
    "large_table_threshold_rows": 10000000,
    "allow_realtime_with_without_limit": True
}


def deep_merge_config(default: Any, override: Any) -> Any:
    """기본 설정과 외부 설정을 안전하게 병합한다.

    외부 JSON이 일부 항목만 가지고 있어도 기본 설정을 잃지 않도록 한다.
    - dict: key 단위 병합
    - list: 외부 list가 비어 있으면 기본 list 유지, 값이 있으면 외부 list 사용
    - scalar: 외부 값이 None/빈 문자열이 아니면 외부 값 사용
    """
    if isinstance(default, dict) and isinstance(override, dict):
        merged = dict(default)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge_config(merged[key], value)
            elif value not in (None, "", [], {}):
                merged[key] = value
        return merged

    if isinstance(default, list):
        return override if isinstance(override, list) and len(override) > 0 else default

    return override if override not in (None, "") else default


def load_optional_json_config(path: Path, default: Any, config_name: str) -> Any:
    """설정 파일이 있으면 읽고, 없으면 기본값을 사용한다.

    app.py 안에 업무별 값을 직접 박지 않기 위한 공통 설정 로더다.
    기본값은 로컬 실행/경진대회 실행 안정성을 위한 fallback이다.
    외부 설정은 기본 설정과 병합하므로 일부 설정 누락으로 화면/정책이 깨지지 않는다.
    """
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return deep_merge_config(default, data)
    except Exception as exc:
        logger.exception("%s 설정 파일 로드 실패: %s", config_name, path)
        st.warning(f"{config_name} 설정 파일을 읽지 못해 기본값을 사용합니다: {exc}")
        return default


SQL_ANALYSIS_SECTION_ALIASES = load_optional_json_config(
    SQL_ANALYSIS_SECTION_ALIASES_PATH,
    DEFAULT_SQL_ANALYSIS_SECTION_ALIASES,
    "sql_analysis_section_aliases",
)
REALTIME_QUERY_POLICY = load_optional_json_config(
    REALTIME_QUERY_POLICY_PATH,
    DEFAULT_REALTIME_QUERY_POLICY,
    "realtime_query_policy",
)
APP_RENDER_SCHEMA = load_optional_json_config(
    APP_RENDER_SCHEMA_PATH,
    DEFAULT_APP_RENDER_SCHEMA,
    "app_render_schema",
)
SQL_REVIEW_POLICY = load_optional_json_config(
    SQL_REVIEW_POLICY_PATH,
    DEFAULT_SQL_REVIEW_POLICY,
    "sql_review_policy",
)
AGENT_GRAPH_POLICY = load_optional_json_config(
    AGENT_GRAPH_POLICY_PATH,
    DEFAULT_AGENT_GRAPH_POLICY,
    "agent_graph_policy",
)
AGENT_INTENT_REGISTRY = load_optional_json_config(
    AGENT_INTENT_REGISTRY_PATH,
    DEFAULT_AGENT_INTENT_REGISTRY,
    "agent_intent_registry",
)
SUPPORTED_AGENT_INTENTS = set(AGENT_INTENT_REGISTRY.keys())


def get_intent_spec(intent: str) -> Dict[str, Any]:
    return dict(AGENT_INTENT_REGISTRY.get(str(intent or ""), {}) or {})


def get_intents_by_category(category: str) -> set[str]:
    return {
        intent
        for intent, spec in AGENT_INTENT_REGISTRY.items()
        if str((spec or {}).get("category") or "") == category
    }


def is_supported_intent(intent: str) -> bool:
    return str(intent or "") in AGENT_INTENT_REGISTRY


def get_realtime_policy(query_meta: Dict[str, Any]) -> Dict[str, Any]:
    query_id = str((query_meta or {}).get("query_id") or "").strip()
    default_policy = dict(REALTIME_QUERY_POLICY.get("default", {}))

    # registry의 realtime intent 설정을 query_id 기준으로 병합한다.
    # 따라서 새 realtime 조회는 agent_intent_registry.json에 query_id/summary_handler만 추가하면 된다.
    registry_policy: Dict[str, Any] = {}
    for _intent, spec in AGENT_INTENT_REGISTRY.items():
        if str((spec or {}).get("query_id") or "").strip() == query_id:
            registry_policy = dict(spec or {})
            break

    query_policy = REALTIME_QUERY_POLICY.get(query_id, {}) if query_id else {}
    if isinstance(registry_policy, dict):
        default_policy.update(registry_policy)
    if isinstance(query_policy, dict):
        default_policy.update(query_policy)

    if not default_policy.get("summary_handler") and default_policy.get("summary_type"):
        default_policy["summary_handler"] = default_policy.get("summary_type")
    if not default_policy.get("summary_type") and default_policy.get("summary_handler"):
        default_policy["summary_type"] = default_policy.get("summary_handler")
    return default_policy

def get_agent() -> HandoverAgent:
    return HandoverAgent(
        json_path=JSON_PATH,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )
