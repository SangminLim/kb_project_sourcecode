from __future__ import annotations

import json
import re
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


def _normalize_identifier(value: Any) -> str:
    """SQL 식별자 안전성 검증용 정규화.

    실무 원칙:
    - 테이블명/컬럼명은 바인드 파라미터로 처리할 수 없으므로 코드에서 최소 검증한다.
    - 영문/숫자/언더스코어/점(schema.table)만 허용한다.
    - 업무별 테이블명을 코드에 박지 않고 spec 메타에서 읽는다.
    """
    text = str(value or "").strip()
    if not text:
        return ""

    if not re.fullmatch(r"[A-Za-z0-9_.$]+", text):
        raise ValueError(f"허용되지 않는 SQL 식별자입니다: {text}")

    return text


def _get_nested_value(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    current: Any = data

    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def _resolve_execution_strategy(spec: Dict[str, Any]) -> Dict[str, Any]:
    """batch_spec에서 실행 전략 메타를 찾는다.

    지원 위치:
    1. spec["execution_strategy"]
    2. spec["target"]["execution_strategy"]
    3. spec["rule_source"]["execution_strategy"]

    이렇게 여러 위치를 허용하면 rule/spec_builder 어느 쪽에서 메타를 만들어도
    code_generator는 동일하게 처리할 수 있다.
    """
    target = spec.get("target") or {}
    rule_source = spec.get("rule_source") or {}

    strategy = (
        spec.get("execution_strategy")
        or target.get("execution_strategy")
        or rule_source.get("execution_strategy")
        or {}
    )

    return strategy if isinstance(strategy, dict) else {}


def _build_pre_sql(spec: Dict[str, Any]) -> str:
    """실행 전략 메타 기반으로 SQL 앞에 붙을 pre_sql을 생성한다.

    지원 패턴:
    - none / append_only: pre_sql 없음
    - replace_partition: 특정 파티션/기준월/기준일자 데이터 삭제 후 INSERT
    - delete_then_insert: replace_partition과 동일한 의미로 지원

    필요한 메타 예:
    {
      "execution_strategy": {
        "type": "replace_partition",
        "target_table": "TB_DEDUCTION_MONTHLY_SUMMARY",
        "partition_column": "BASE_YM",
        "partition_param": "base_ym"
      }
    }

    target_table은 없으면 spec.target.table을 fallback으로 사용한다.
    """
    strategy = _resolve_execution_strategy(spec)
    strategy_type = str(strategy.get("type") or strategy.get("pattern") or "").strip().lower()

    if not strategy_type:
        strategy_type = str(spec.get("execution_pattern") or "").strip().lower()

    if strategy_type in {"", "none", "append", "append_only"}:
        return ""

    if strategy_type not in {"replace_partition", "delete_then_insert"}:
        return ""

    target = spec.get("target") or {}

    target_table = _normalize_identifier(
        strategy.get("target_table")
        or target.get("table")
        or spec.get("target_table")
    )
    partition_column = _normalize_identifier(
        strategy.get("partition_column")
        or strategy.get("base_column")
        or target.get("partition_column")
        or spec.get("partition_column")
    )
    partition_param = _normalize_identifier(
        strategy.get("partition_param")
        or strategy.get("base_param")
        or target.get("partition_param")
        or spec.get("partition_param")
    )

    if not target_table or not partition_column or not partition_param:
        raise ValueError(
            "replace_partition 실행 전략에는 target_table, partition_column, partition_param이 필요합니다."
        )

    return (
        f"DELETE FROM {target_table}\n"
        f"WHERE {partition_column} = :{partition_param};"
    )


def _build_template_values(spec: Dict[str, Any]) -> Dict[str, Any]:
    target = spec.get("target") or {}
    source = spec.get("source") or {}
    source_table = _get_source_table_for_template(spec)
    pre_sql = _build_pre_sql(spec)

    return {
        "batch_id": spec.get("batch_id", ""),
        "batch_name": spec.get("batch_name", ""),
        "batch_type": spec.get("batch_type", ""),
        "description": spec.get("description", ""),
        "schedule_type": spec.get("schedule_type", ""),

        # query.sql.j2에서 사용
        "pre_sql": pre_sql,
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


def _render_template(template_text: str, spec: Dict[str, Any]) -> str:
    """단순 placeholder 렌더링.

    기존 구조를 유지하되, {{ pre_sql }} 같은 확장 메타를 추가했다.
    복잡한 Jinja 조건문은 query.sql에 그대로 남을 위험이 있으므로 쓰지 않는다.
    """
    rendered = template_text
    flat_values = _build_template_values(spec)

    for key, value in flat_values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))

    # pre_sql이 비어 있을 때 불필요한 빈 줄을 조금 정리한다.
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip() + "\n"

    if "{{" in rendered or "{%" in rendered:
        raise ValueError(
            "렌더링되지 않은 템플릿 문법이 남아 있습니다. "
            "query.sql.j2에는 {{ pre_sql }}, {{ sql }} 같은 단순 placeholder만 사용하세요."
        )

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
