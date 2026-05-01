from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import GENERATED_DIR, TEMPLATE_DIR


def _render_template(template_text: str, spec: Dict[str, Any]) -> str:
    # 외부 의존성 최소화를 위해 Jinja2 대신 단순 치환을 사용한다.
    rendered = template_text
    flat_values = {
        "batch_id": spec.get("batch_id", ""),
        "batch_name": spec.get("batch_name", ""),
        "batch_type": spec.get("batch_type", ""),
        "description": spec.get("description", ""),
        "schedule_type": spec.get("schedule_type", ""),
        "sql": spec.get("sql", ""),
        "source_table": (spec.get("source") or {}).get("table", ""),
        "base_date_column": (spec.get("source") or {}).get("base_date_column", "BASE_DATE"),
        "output_format": (spec.get("target") or {}).get("output_format", "csv"),
        "output_file_prefix": (spec.get("target") or {}).get("output_file_prefix", "batch_output"),
        "output_file_pattern": (spec.get("target") or {}).get("output_file_pattern", "batch_output_{base_date}.csv"),
        "output_dir": (spec.get("target") or {}).get("output_dir", "./output"),
        "encoding": (spec.get("target") or {}).get("encoding", "utf-8-sig"),
    }
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
