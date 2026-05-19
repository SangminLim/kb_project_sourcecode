from __future__ import annotations

import os
from pathlib import Path


# 프로젝트 루트 기준. 기본값은 현재 실행 위치(kb_project)를 사용한다.
BASE_DIR = Path(os.getenv("BATCH_DEV_BASE_DIR", ".")).resolve()

BATCH_DEV_DIR = Path(
    os.getenv(
        "BATCH_DEV_DIR",
        str(BASE_DIR / "batch_dev"),
    )
).resolve()

TEMPLATE_DIR = Path(
    os.getenv(
        "BATCH_DEV_TEMPLATE_DIR",
        str(BATCH_DEV_DIR / "templates"),
    )
).resolve()

GENERATED_DIR = Path(
    os.getenv(
        "BATCH_DEV_GENERATED_DIR",
        str(BATCH_DEV_DIR / "generated"),
    )
).resolve()

OUTPUT_DIR = Path(
    os.getenv(
        "BATCH_DEV_OUTPUT_DIR",
        str(BATCH_DEV_DIR / "output"),
    )
).resolve()

METADATA_DIR = Path(
    os.getenv(
        "BATCH_DEV_METADATA_DIR",
        str(BATCH_DEV_DIR / "metadata"),
    )
).resolve()

ERWIN_METADATA_PATH = Path(
    os.getenv(
        "BATCH_DEV_ERWIN_METADATA_PATH",
        str(METADATA_DIR / "erwin_meta.json"),
    )
).resolve()

REQUEST_SCHEMA_PATH = Path(
    os.getenv(
        "BATCH_DEV_REQUEST_SCHEMA_PATH",
        str(BATCH_DEV_DIR / "request_schema.json"),
    )
).resolve()

BUSINESS_RULE_DIR = Path(
    os.getenv(
        "BATCH_DEV_BUSINESS_RULE_DIR",
        str(BATCH_DEV_DIR / "business_rules"),
    )
).resolve()

SQL_TEMPLATE_DIR = Path(
    os.getenv(
        "BATCH_DEV_SQL_TEMPLATE_DIR",
        str(BATCH_DEV_DIR / "sql_templates"),
    )
).resolve()

DEFAULT_BATCH_TYPE = os.getenv(
    "BATCH_DEV_DEFAULT_TYPE",
    "db_to_file",
)

DEFAULT_OUTPUT_ENCODING = os.getenv(
    "BATCH_DEV_OUTPUT_ENCODING",
    "utf-8-sig",
)

DB_DIALECT = os.getenv(
    "BATCH_DEV_DB_DIALECT",
    "mariadb",
)

# 배치 개발 결과에서 SQL 자동 개선 제안 실행/표시 여부
# - false: 배치 요청서로 소스 생성 시 SQL 개선 제안을 실행하지 않고 화면에도 표시하지 않는다.
# - true : 기존처럼 SQL 개선 제안/리포트를 생성한다.
BATCH_SQL_IMPROVEMENT_ENABLED = os.getenv(
    "BATCH_SQL_IMPROVEMENT_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "y"}

# LLM batch_spec draft generation
# - true: 요청서를 LLM이 먼저 batch_spec draft JSON으로 변환하고, spec_builder가 ERWin/rule로 검증·보정한다.
# - false: 기존 rule/parser 기반 spec_builder 흐름만 사용한다.
BATCH_SPEC_USE_LLM = os.getenv(
    "BATCH_SPEC_USE_LLM",
    "false",
).strip().lower() in {"1", "true", "yes", "y"}

BATCH_SPEC_LLM_MODEL = os.getenv(
    "BATCH_SPEC_LLM_MODEL",
    os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro3"),
)

BATCH_SPEC_LLM_TIMEOUT = int(os.getenv(
    "BATCH_SPEC_LLM_TIMEOUT",
    os.getenv("UPSTAGE_CHAT_TIMEOUT", "60"),
))

BATCH_SPEC_LLM_TEMPERATURE = float(os.getenv(
    "BATCH_SPEC_LLM_TEMPERATURE",
    os.getenv("UPSTAGE_TEMPERATURE", "0.1"),
))

BATCH_SPEC_LLM_MAX_TOKENS = int(os.getenv(
    "BATCH_SPEC_LLM_MAX_TOKENS",
    os.getenv("UPSTAGE_MAX_TOKENS", "2048"),
))
