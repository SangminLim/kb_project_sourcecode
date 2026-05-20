from __future__ import annotations

import os
from pathlib import Path


# batch_dev 패키지 기준 경로.
#
# batch_dev를 프로젝트 루트에서 agents/batch_dev로 이동했기 때문에
# 현재 실행 위치(kb_project)를 기준으로 BASE_DIR / "batch_dev"를 만들면
# C:/Users/KBCARD/kb_project/batch_dev 를 바라보게 되어 템플릿을 찾지 못한다.
#
# 기본값은 이 config.py가 들어있는 디렉터리(agents/batch_dev)로 잡고,
# 필요한 경우에만 환경변수 BATCH_DEV_DIR로 override한다.
PACKAGE_DIR = Path(__file__).resolve().parent

# 일부 기존 코드/테스트가 BASE_DIR을 import할 수 있으므로 호환용으로 유지한다.
# 의미상 프로젝트 루트가 아니라 batch_dev 패키지 디렉터리다.
BASE_DIR = Path(os.getenv("BATCH_DEV_BASE_DIR", str(PACKAGE_DIR))).resolve()

BATCH_DEV_DIR = Path(
    os.getenv(
        "BATCH_DEV_DIR",
        str(PACKAGE_DIR),
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

def _env_csv_set(name: str, default: str) -> set[str]:
    """쉼표 구분 환경변수를 set[str]로 변환한다.

    예)
    BATCH_DEV_SUPPORTED_TYPES=db_to_file,file_to_db,db_to_db,aggregation_to_table
    """
    raw = os.getenv(name, default)
    return {
        item.strip()
        for item in str(raw).split(",")
        if item.strip()
    }


SUPPORTED_BATCH_TYPES = _env_csv_set(
    "BATCH_DEV_SUPPORTED_TYPES",
    "db_to_file,file_to_db,db_to_db,aggregation_to_table",
)

SUPPORTED_OUTPUT_FORMATS = _env_csv_set(
    "BATCH_DEV_SUPPORTED_OUTPUT_FORMATS",
    "csv,txt,xlsx",
)

# 요청서 분류에 사용할 기본 신호.
# request_schema.json aliases를 우선 사용하고, 이 값은 보조 신호로만 사용한다.
REQUEST_CLASSIFIER_EXTRA_SIGNALS = _env_csv_set(
    "BATCH_DEV_REQUEST_CLASSIFIER_EXTRA_SIGNALS",
    "[배치 개발 요청서],배치 개발 요청서",
)

# 배치 개발 생성 플로우를 LangGraph workflow로 실행할지 여부.
# - true: spec 생성 → 검증 → 템플릿 선택 → 코드 생성 → 생성물 검증 → SQL 개선 → finalize 순서로 노드화
# - false: 기존 BatchDevAgent 선형 처리 사용
# LangGraph 미설치/실패 시 agent.py에서 기존 선형 처리로 fallback한다.
BATCH_DEV_LANGGRAPH_ENABLED = os.getenv(
    "BATCH_DEV_LANGGRAPH_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "y"}

