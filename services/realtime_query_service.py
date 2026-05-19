"""
realtime_service.py

설명
- DB 연결과 실시간 조회형 SQL을 담당하는 서비스 레이어
- query_id 기준으로 SQL 실행 책임을 분리
- SQL 본문과 query registry는 코드가 아니라 conf/sql 파일에서 관리
- Streamlit UI는 렌더링에 집중하고, llm.py는 질문 해석에 집중하도록 역할 분리
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# services/realtime_query_service.py로 이동해도 프로젝트 루트의 conf/, sql/을 보도록 한다.
# 기본 위치:
#   PROJECT_ROOT/conf/realtime_query_registry.json
#   PROJECT_ROOT/sql/*.sql
PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)

CONFIG_DIR = Path(
    os.getenv(
        "CONFIG_DIR",
        str(PROJECT_ROOT / "conf"),
    )
)

SQL_DIR = Path(
    os.getenv(
        "REALTIME_SQL_DIR",
        str(PROJECT_ROOT / "sql"),
    )
)

REALTIME_QUERY_REGISTRY_PATH = Path(
    os.getenv(
        "REALTIME_QUERY_REGISTRY_PATH",
        str(CONFIG_DIR / "realtime_query_registry.json"),
    )
)


class RealtimeQueryConfigError(ValueError):
    """Realtime query 설정 오류."""


class RealtimeQuerySecurityError(RuntimeError):
    """Realtime SQL 안전성 검증 오류."""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RealtimeQueryConfigError(f"Realtime query registry 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RealtimeQueryConfigError("Realtime query registry는 JSON object 형식이어야 합니다.")
    return data


def _read_sql_file(sql_file: str) -> str:
    path = SQL_DIR / str(sql_file)
    if not path.exists() or not path.is_file():
        raise RealtimeQueryConfigError(f"SQL 파일이 없습니다: {path}")
    sql = path.read_text(encoding="utf-8").strip()
    if not sql:
        raise RealtimeQueryConfigError(f"SQL 파일이 비어 있습니다: {path}")
    return sql


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql or "", flags=re.DOTALL)
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    return sql.strip()


def validate_readonly_sql(sql: str) -> None:
    """실시간 조회는 SELECT/WITH만 허용한다.

    실제 운영에서는 read-only DB 계정, DB timeout, network ACL도 함께 적용해야 한다.
    이 검증은 애플리케이션 레벨의 2차 방어선이다.
    """
    cleaned = _strip_sql_comments(sql)
    upper_sql = cleaned.upper()

    if not re.match(r"^(SELECT|WITH)\b", upper_sql):
        raise RealtimeQuerySecurityError("Realtime SQL은 SELECT/WITH 문만 허용됩니다.")

    blocked_tokens = [
        "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "CALL", "EXEC",
    ]
    for token in blocked_tokens:
        if re.search(rf"\b{token}\b", upper_sql):
            raise RealtimeQuerySecurityError(f"Realtime SQL에 허용되지 않는 키워드가 포함되어 있습니다: {token}")


def _apply_sql_variables(sql: str, variables: Optional[Mapping[str, Any]] = None) -> str:
    """SQL 템플릿 변수를 안전하게 치환한다.

    - 값은 registry variables 또는 환경변수에서 주입한다.
    - 현재는 식별자/스키마명 치환만 허용하기 위해 보수적인 문자만 통과시킨다.
    - 예: {schema}.TB_TABLE
    """
    variables = dict(variables or {})

    # registry에서 env:NAME 형태를 쓰면 환경변수 값을 사용한다.
    resolved: Dict[str, str] = {}
    for key, value in variables.items():
        text = str(value or "").strip()
        if text.startswith("env:"):
            env_name = text.split(":", 1)[1].strip()
            text = os.getenv(env_name, "").strip()
        if text and not re.fullmatch(r"[A-Za-z0-9_.$]+", text):
            raise RealtimeQuerySecurityError(f"SQL 변수 {key} 값에 허용되지 않는 문자가 포함되어 있습니다.")
        resolved[str(key)] = text

    # 기본 schema 변수는 REALTIME_DB_SCHEMA에서 가져온다.
    resolved.setdefault("schema", os.getenv("REALTIME_DB_SCHEMA", "").strip())

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = resolved.get(key, "")
        if key == "schema" and value:
            return value
        if key == "schema" and not value:
            # schema가 비어 있으면 '{schema}.' 패턴이 남지 않도록 호출 SQL에서 {{schema_prefix}} 사용을 권장한다.
            return ""
        return value

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, sql)


def _validate_param_type(name: str, value: Any, spec: Dict[str, Any]) -> Any:
    expected_type = str(spec.get("type", "string")).lower()

    if expected_type == "integer":
        try:
            int_value = int(value)
        except Exception as exc:
            raise RealtimeQueryConfigError(f"파라미터 {name}는 integer여야 합니다.") from exc
        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is not None and int_value < int(min_value):
            raise RealtimeQueryConfigError(f"파라미터 {name}는 {min_value} 이상이어야 합니다.")
        if max_value is not None and int_value > int(max_value):
            raise RealtimeQueryConfigError(f"파라미터 {name}는 {max_value} 이하여야 합니다.")
        return int_value

    if expected_type == "number":
        try:
            return float(value)
        except Exception as exc:
            raise RealtimeQueryConfigError(f"파라미터 {name}는 number여야 합니다.") from exc

    # 기본은 string
    text = str(value)
    allowed_values = spec.get("allowed_values")
    if allowed_values and text not in allowed_values:
        raise RealtimeQueryConfigError(f"파라미터 {name} 값이 허용 목록에 없습니다: {text}")
    max_length = spec.get("max_length")
    if max_length is not None and len(text) > int(max_length):
        raise RealtimeQueryConfigError(f"파라미터 {name} 길이가 너무 깁니다.")
    return text


def _build_params(query_config: Dict[str, Any], runtime_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = dict(query_config.get("params") or {})
    runtime_params = dict(runtime_params or {})
    params = {**defaults, **runtime_params}

    schema = query_config.get("param_schema") or {}
    if not schema:
        return params

    validated: Dict[str, Any] = {}
    for name, spec in schema.items():
        spec = dict(spec or {})
        if name not in params:
            if spec.get("required", False):
                raise RealtimeQueryConfigError(f"필수 파라미터가 없습니다: {name}")
            continue
        validated[name] = _validate_param_type(name, params[name], spec)

    # schema에 없는 기본 파라미터도 backward compatibility를 위해 유지한다.
    for name, value in params.items():
        validated.setdefault(name, value)
    return validated


class RealtimeQueryService:
    def __init__(
        self,
        database_url: str,
        registry_path: Optional[str | Path] = None,
        sql_dir: Optional[str | Path] = None,
    ) -> None:
        if not database_url or not database_url.strip():
            raise ValueError("database_url이 비어 있습니다.")

        self.engine: Engine = create_engine(database_url)
        self.registry_path = Path(registry_path) if registry_path else REALTIME_QUERY_REGISTRY_PATH
        self.sql_dir = Path(sql_dir) if sql_dir else SQL_DIR
        self.registry = _load_json(self.registry_path)

    def _get_query_config(self, query_id: str) -> Dict[str, Any]:
        queries = self.registry.get("queries", self.registry)
        if not isinstance(queries, dict):
            raise RealtimeQueryConfigError("registry의 queries 항목은 object여야 합니다.")
        query_config = queries.get(query_id)
        if not isinstance(query_config, dict):
            raise RealtimeQueryConfigError(f"지원하지 않는 query_id 입니다: {query_id}")
        return query_config

    def _read_query_sql(self, query_config: Dict[str, Any]) -> str:
        sql_file = query_config.get("sql_file")
        if sql_file:
            path = self.sql_dir / str(sql_file)
            if not path.exists() or not path.is_file():
                raise RealtimeQueryConfigError(f"SQL 파일이 없습니다: {path}")
            sql = path.read_text(encoding="utf-8").strip()
        else:
            # 이전 설정 호환용. 신규 설정에서는 sql_file 사용을 권장한다.
            sql = str(query_config.get("sql") or "").strip()

        if not sql:
            raise RealtimeQueryConfigError("SQL이 등록되어 있지 않습니다.")

        sql = _apply_sql_variables(sql, query_config.get("variables"))
        validate_readonly_sql(sql)
        return sql

    def fetch_dataframe(self, query_meta: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        query_id = str((query_meta or {}).get("query_id", "")).strip()
        if not query_id:
            raise ValueError("query_meta에 query_id가 없습니다.")

        query_config = self._get_query_config(query_id)
        sql = self._read_query_sql(query_config)
        query_params = _build_params(query_config, params)

        return pd.read_sql(
            text(sql),
            self.engine,
            params=query_params,
        )
