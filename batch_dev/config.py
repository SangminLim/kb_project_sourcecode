from __future__ import annotations

import os
from pathlib import Path


# 프로젝트 루트 기준. 기본값은 현재 실행 위치(kb_project)를 사용한다.
BASE_DIR = Path(os.getenv("BATCH_DEV_BASE_DIR", ".")).resolve()
BATCH_DEV_DIR = Path(os.getenv("BATCH_DEV_DIR", str(BASE_DIR / "batch_dev"))).resolve()
TEMPLATE_DIR = Path(os.getenv("BATCH_DEV_TEMPLATE_DIR", str(BASE_DIR / "templates"))).resolve()
GENERATED_DIR = Path(os.getenv("BATCH_DEV_GENERATED_DIR", str(BASE_DIR / "generated"))).resolve()

METADATA_DIR = Path(os.getenv("BATCH_DEV_METADATA_DIR", str(BATCH_DEV_DIR / "metadata"))).resolve()
ERWIN_METADATA_PATH = Path(os.getenv("BATCH_DEV_ERWIN_METADATA_PATH", str(METADATA_DIR / "erwin_meta.json"))).resolve()

REQUEST_SCHEMA_PATH = Path(os.getenv("BATCH_DEV_REQUEST_SCHEMA_PATH", str(BATCH_DEV_DIR / "request_schema.json"))).resolve()
BUSINESS_RULE_DIR = Path(os.getenv("BATCH_DEV_BUSINESS_RULE_DIR", str(BATCH_DEV_DIR / "business_rules"))).resolve()
SQL_TEMPLATE_DIR = Path(os.getenv("BATCH_DEV_SQL_TEMPLATE_DIR", str(BATCH_DEV_DIR / "sql_templates"))).resolve()

DEFAULT_BATCH_TYPE = os.getenv("BATCH_DEV_DEFAULT_TYPE", "db_to_file")
DEFAULT_OUTPUT_ENCODING = os.getenv("BATCH_DEV_OUTPUT_ENCODING", "utf-8-sig")
DB_DIALECT = os.getenv("BATCH_DEV_DB_DIALECT", "mariadb")
