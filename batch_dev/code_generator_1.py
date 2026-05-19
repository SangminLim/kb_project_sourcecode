from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import GENERATED_DIR, TEMPLATE_DIR


def _get_source_table_for_template(spec: Dict[str, Any]) -> str:
    source = spec.get("source") or {}
    meta_source = spec.get("meta_source") or {}
    resolved_tables = meta_source.get("resolved_tables") or {}

    return (
        source.get("table")
        or resolved_tables.get("base")
        or ""
    )


def _render_template(template_text: str, spec: Dict[str, Any]) -> str:
    target = spec.get("target") or {}
    source = spec.get("source") or {}
    source_table = _get_source_table_for_template(spec)

    flat_values = {
        "batch_id": spec.get("batch_id", ""),
        "batch_name": spec.get("batch_name", ""),
        "batch_type": spec.get("batch_type", ""),
        "description": spec.get("description", ""),
        "schedule_type": spec.get("schedule_type", ""),
        "sql": spec.get("sql", ""),
        "delete_sql": target.get("delete_sql", ""),

        "source_table": source_table,
        "table_name": source_table,

        # SQL clause 기반 최종 템플릿용
        "select_clause": spec.get("select_clause", "*"),
        "from_clause": spec.get("from_clause", source_table),
        "join_clause": spec.get("join_clause", ""),
        "where_clause": spec.get("where_clause", "1 = 1"),

        "source_table_role": source.get("table_role", ""),
        "base_date_column": source.get("base_date_column", "BASE_DATE"),

        "output_format": target.get("output_format", "csv"),
        "output_file_prefix": target.get("output_file_prefix", "batch_output"),
        "output_file_pattern": target.get("output_file_pattern", "batch_output_{base_date}.csv"),
        "output_dir": target.get("output_dir", "./output"),
        "encoding": target.get("encoding", "utf-8-sig"),
    }

    rendered = template_text
    for key, value in flat_values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))

    return rendered


def generate_code(spec: Dict[str, Any], template_type: str) -> List[str]:
    batch_id = spec["batch_id"]
    output_dir = GENERATED_DIR / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: List[str] = []

    spec_path = output_dir / "batch_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    created_files.append(str(spec_path))

    template_dir = TEMPLATE_DIR / template_type
    if not template_dir.exists():
        raise FileNotFoundError(f"템플릿 디렉터리가 없습니다: {template_dir}")

    mapping = {
        "job.py.j2": "job.py",
        "query.sql.j2": "query.sql",
        "README.md.j2": "README.md",
        "test_job.py.j2": "test_job.py",
    }

    for template_name, output_name in mapping.items():
        template_path = template_dir / template_name
        if not template_path.exists():
            continue

        rendered = _render_template(template_path.read_text(encoding="utf-8"), spec)
        output_path = output_dir / output_name
        output_path.write_text(rendered, encoding="utf-8")
        created_files.append(str(output_path))

    return created_files
