from __future__ import annotations

"""
LLM 기반 batch_spec 초안 생성기.

역할 분리 원칙
- LLM: 배치 개발 요청서의 의미를 읽고 표준 draft JSON을 만든다.
- spec_builder: draft를 ERWin 메타/rule/template 기준으로 검증·보정하여 최종 batch_spec을 만든다.
- code_generator: 최종 batch_spec만 받아 파일을 생성한다.

이 파일은 특정 업무명/테이블명을 하드코딩하지 않는다.
ERWin 메타와 request_schema를 LLM 입력 컨텍스트로 제공해 확장 가능하게 동작한다.
"""

import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from ..config import (
    BATCH_SPEC_LLM_MAX_TOKENS,
    BATCH_SPEC_LLM_MODEL,
    BATCH_SPEC_LLM_TEMPERATURE,
    BATCH_SPEC_LLM_TIMEOUT,
)

ALLOWED_BATCH_TYPES = {
    "db_to_file",
    "file_to_db",
    "db_to_db",
    "aggregation_to_table",
}
ALLOWED_OUTPUT_FORMATS = {"csv", "txt", "xlsx"}


class BatchSpecDraftError(RuntimeError):
    """LLM draft 생성/파싱 실패."""


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _json_object_from_text(text: str) -> Dict[str, Any]:
    """LLM 응답에서 JSON 객체 하나를 안전하게 추출한다."""
    raw = (text or "").strip()
    if not raw:
        raise BatchSpecDraftError("LLM 응답이 비어 있습니다.")

    # Markdown code fence 제거
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise BatchSpecDraftError(f"LLM 응답에서 JSON 객체를 찾지 못했습니다: {raw[:300]}")
        payload = json.loads(raw[start : end + 1])

    if not isinstance(payload, dict):
        raise BatchSpecDraftError("LLM 응답 JSON은 object여야 합니다.")
    return payload


def _compact_erwin_context(erwin_meta: Optional[Mapping[str, Any]], max_tables: int = 30, max_columns: int = 80) -> Dict[str, Any]:
    """LLM에 넘길 ERWin 컨텍스트를 작고 안정적인 형태로 압축한다."""
    if not erwin_meta:
        return {"tables": [], "relations": []}

    tables: List[Dict[str, Any]] = []
    for table in list(erwin_meta.get("tables", []) or [])[:max_tables]:
        columns = []
        for col in list(table.get("columns", []) or [])[:max_columns]:
            columns.append(
                {
                    "column_name": col.get("column_name"),
                    "role": col.get("role"),
                    "data_type": col.get("data_type"),
                }
            )
        tables.append(
            {
                "table_name": table.get("table_name"),
                "table_kor_name": table.get("table_kor_name"),
                "table_role": table.get("table_role"),
                "aliases": table.get("aliases", []),
                "columns": columns,
            }
        )

    relations: List[Dict[str, Any]] = []
    for rel in list(erwin_meta.get("relations", []) or [])[:100]:
        relations.append(
            {
                "left_table": rel.get("left_table"),
                "right_table": rel.get("right_table"),
                "join_columns": rel.get("join_columns", []),
                "effective_date": rel.get("effective_date", {}),
            }
        )

    return {"tables": tables, "relations": relations}


def _compact_request_schema(schema: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not schema:
        return {}
    fields = schema.get("fields") or {}
    return {
        "request_type": schema.get("request_type"),
        "fields": {
            key: {
                "required": value.get("required", False),
                "aliases": value.get("aliases", []),
            }
            for key, value in fields.items()
            if isinstance(value, Mapping)
        },
    }


def _build_prompt(
    request_text: str,
    erwin_meta: Optional[Mapping[str, Any]],
    request_schema: Optional[Mapping[str, Any]],
) -> tuple[str, str]:
    schema_context = _compact_request_schema(request_schema)
    erwin_context = _compact_erwin_context(erwin_meta)

    system_prompt = """
너는 금융권 배치 개발 요청서를 표준 batch_spec draft JSON으로 변환하는 수석 배치 설계자다.
반드시 JSON 객체 하나만 반환한다. Markdown, 설명, 주석은 금지한다.
입력 요청서와 제공된 ERWin 메타데이터/request_schema에 근거해서만 작성한다.
테이블명/컬럼명/조건을 특정 업무에 맞춰 하드코딩하지 말고, 요청서와 메타데이터에서 추출한다.
확신할 수 없는 값은 빈 문자열 또는 빈 배열로 둔다.
SQL 전체를 직접 만들지 말고, SQL 생성을 위한 구조화된 spec 초안만 만든다.
조건식에는 배치 기준일 바인드 변수를 :base_date 또는 :base_ym 형태로 표현한다.
""".strip()

    user_prompt = f"""
[출력 JSON 스키마]
{{
  "batch_name": "string",
  "batch_type": "db_to_file|file_to_db|db_to_db|aggregation_to_table",
  "schedule_type": "daily|monthly|manual|string",
  "source_table": "string",
  "target_table": "string",
  "output_format": "csv|txt|xlsx|string",
  "output_file": "string",
  "output_file_prefix": "string",
  "base_date_column": "string",
  "base_parameter": "base_date|base_ym|string",
  "output_columns": ["COLUMN_NAME"],
  "conditions": ["safe SQL WHERE condition fragment"],
  "joins": [
    {{
      "join_type": "INNER|LEFT|string",
      "table": "string",
      "on": ["safe SQL ON condition fragment"]
    }}
  ],
  "validation_rules": {{
    "min_rows": 0,
    "not_null_columns": ["COLUMN_NAME"]
  }},
  "llm_notes": ["string"]
}}

[request_schema]
{json.dumps(schema_context, ensure_ascii=False, indent=2)}

[ERWin metadata compact]
{json.dumps(erwin_context, ensure_ascii=False, indent=2)}

[배치 개발 요청서]
{request_text}
""".strip()
    return system_prompt, user_prompt


def _call_project_llm(prompt: str, system_prompt: str) -> str:
    """프로젝트의 llm.py를 통해 LLM을 호출한다."""
    try:
        from agents.handover import ChatConfig, ollama_generate
    except Exception as exc:
        raise BatchSpecDraftError(f"llm.py import 실패: {type(exc).__name__}: {exc}") from exc

    config = ChatConfig(
        model=BATCH_SPEC_LLM_MODEL,
        timeout=BATCH_SPEC_LLM_TIMEOUT,
        temperature=BATCH_SPEC_LLM_TEMPERATURE,
        max_tokens=BATCH_SPEC_LLM_MAX_TOKENS,
    )
    return ollama_generate(prompt=prompt, system_prompt=system_prompt, config=config)


def _upper_identifier(value: Any) -> str:
    text = _normalize_text(value).strip("`[]\"'")
    return text.upper() if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", text) else ""


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,\n]+", value) if item.strip()]
    return []


def normalize_llm_batch_spec_draft(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """LLM draft를 spec_builder가 쓰기 쉬운 표준 형태로 정리한다."""
    draft: Dict[str, Any] = {}

    batch_type = _normalize_text(payload.get("batch_type")).lower()
    if batch_type in {"file_create", "file_export", "export", "파일 생성 배치"}:
        batch_type = "db_to_file"
    if batch_type not in ALLOWED_BATCH_TYPES:
        batch_type = ""

    output_format = _normalize_text(payload.get("output_format")).lower().lstrip(".")
    if output_format in {"excel", "xls"}:
        output_format = "xlsx"
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        output_format = ""

    schedule_type = _normalize_text(payload.get("schedule_type")).lower()
    if schedule_type in {"일", "일배치", "daily batch", "day"}:
        schedule_type = "daily"
    elif schedule_type in {"월", "월배치", "monthly batch", "month"}:
        schedule_type = "monthly"

    draft["batch_name"] = _normalize_text(payload.get("batch_name"))
    draft["batch_type"] = batch_type
    draft["schedule_type"] = schedule_type
    draft["source_table"] = _upper_identifier(payload.get("source_table"))
    draft["target_table"] = _upper_identifier(payload.get("target_table"))
    draft["output_format"] = output_format
    draft["output_file"] = _normalize_text(payload.get("output_file"))
    draft["output_file_prefix"] = _normalize_text(payload.get("output_file_prefix"))
    draft["base_date_column"] = _upper_identifier(payload.get("base_date_column"))
    draft["base_parameter"] = _normalize_text(payload.get("base_parameter")) or "base_date"

    columns = [_upper_identifier(item) for item in _normalize_list(payload.get("output_columns") or payload.get("columns"))]
    draft["output_columns"] = [item for item in columns if item]

    conditions = []
    for item in _normalize_list(payload.get("conditions")):
        text = _normalize_text(item)
        if text:
            conditions.append(text)
    draft["conditions"] = conditions

    joins = []
    for item in _normalize_list(payload.get("joins")):
        if isinstance(item, Mapping):
            joins.append(dict(item))
    draft["joins"] = joins

    validation_rules = payload.get("validation_rules") if isinstance(payload.get("validation_rules"), Mapping) else {}
    draft["validation_rules"] = dict(validation_rules or {})

    notes = [_normalize_text(item) for item in _normalize_list(payload.get("llm_notes"))]
    draft["llm_notes"] = [item for item in notes if item]

    return {key: value for key, value in draft.items() if value not in (None, "", [], {})}


def build_batch_spec_draft_with_llm(
    request_text: str,
    *,
    erwin_meta: Optional[Mapping[str, Any]] = None,
    request_schema: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """배치 개발 요청서를 LLM batch_spec draft로 변환한다."""
    if not request_text or not request_text.strip():
        return {}

    system_prompt, prompt = _build_prompt(
        request_text=request_text,
        erwin_meta=erwin_meta,
        request_schema=request_schema,
    )
    response = _call_project_llm(prompt=prompt, system_prompt=system_prompt)
    payload = _json_object_from_text(response)
    return normalize_llm_batch_spec_draft(payload)


def is_llm_batch_spec_enabled() -> bool:
    return os.getenv("BATCH_SPEC_USE_LLM", "false").strip().lower() in {"1", "true", "yes", "y"}
