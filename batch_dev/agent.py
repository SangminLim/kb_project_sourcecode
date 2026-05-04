
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_META_PATH = Path(__file__).resolve().parent / "metadata" / "erwin_meta.json"
DEFAULT_RULE_CATALOG_PATH = Path(__file__).resolve().parent / "business_rules" / "rule_catalog.json"
DEFAULT_OUTPUT_ROOT = Path("generated")


@dataclass
class BatchDevResult:
    success: bool
    message: str
    batch_spec: Dict[str, Any] = field(default_factory=dict)
    created_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sanitize_identifier(text: str, default: str = "BATCH_GENERATED") -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", text or "").strip("_")
    return value or default


def _to_batch_id(rule_id: str) -> str:
    base = rule_id.upper()
    if not base.startswith("BATCH_"):
        base = f"BATCH_{base}"
    return _sanitize_identifier(base)


def _extract_section(text: str, section_name: str) -> str:
    pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=\n\s*[가-힣A-Za-z ]+\s*:|$)"
    m = re.search(pattern, text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "")


def _extract_batch_name(text: str) -> str:
    m = re.search(r"배치명\s*:\s*(.+)", text)
    if not m:
        return "배치 개발 요청"
    value = m.group(1).strip()
    value = re.split(r"\s+(기준|대상\s*테이블|처리\s*내용|출력|조건)\s*:", value)[0].strip()
    return value or "배치 개발 요청"


def _table_map(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["table_name"]: t for t in meta.get("tables", [])}


def _column_names(table_meta: Dict[str, Any]) -> List[str]:
    return [c["column_name"] for c in table_meta.get("columns", [])]


def _extract_tables(text: str, known_tables: List[str]) -> List[str]:
    found: List[str] = []
    upper_text = text.upper()
    for table in known_tables:
        if table.upper() in upper_text and table not in found:
            found.append(table)
    return found


def _extract_output_fields(text: str) -> List[str]:
    output = _extract_section(text, "출력")
    if not output:
        return []
    parts = re.split(r"[\n,]+", output)
    return [p.strip("- ").strip() for p in parts if p.strip("- ").strip()]


def _extract_conditions(text: str) -> List[str]:
    cond = _extract_section(text, "조건")
    if not cond:
        return []
    result = []
    for line in cond.splitlines():
        item = line.strip().strip("-").strip()
        if item:
            result.append(item)
    return result


def _rule_score(rule: Dict[str, Any], request_text: str, requested_tables: List[str], output_fields: List[str]) -> int:
    match = rule.get("match", {}) or {}
    score = int(rule.get("priority", 0))

    required_all = match.get("required_all", []) or []
    if any(str(item) not in request_text for item in required_all):
        return -1

    required_any = match.get("required_any", []) or []
    if required_any and not any(str(item) in request_text for item in required_any):
        return -1
    score += sum(5 for item in required_any if str(item) in request_text)

    table_any = match.get("table_any", []) or []
    if table_any and any(t in requested_tables for t in table_any):
        score += 20

    output_any = match.get("output_any", []) or []
    output_text = " ".join(output_fields)
    score += sum(3 for item in output_any if str(item) in output_text or str(item) in request_text)

    return score


def _select_rule(rule_catalog: Dict[str, Any], request_text: str, requested_tables: List[str], output_fields: List[str]) -> Dict[str, Any]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for rule in rule_catalog.get("rules", []):
        score = _rule_score(rule, request_text, requested_tables, output_fields)
        if score >= 0:
            candidates.append((score, rule))
    if not candidates:
        raise ValueError("요청서에 맞는 batch rule을 찾지 못했습니다. rule_catalog.json의 match 조건을 확인하세요.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _relations_for_base(meta: Dict[str, Any], base_table: str, join_tables: List[str]) -> List[Dict[str, Any]]:
    join_set = set(join_tables)
    return [
        rel for rel in meta.get("relations", [])
        if rel.get("left_table") == base_table and rel.get("right_table") in join_set
    ]


def _make_aliases(join_tables: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    used = set()
    preferred = {
        "TB_BOOK_PERF_MERCHANT": "B",
        "TB_TRAD_MARKET_MERCHANT": "T",
        "TB_GENERAL_DEDUCT_MERCHANT": "G",
    }
    for idx, table in enumerate(join_tables, start=1):
        alias = preferred.get(table) or f"J{idx}"
        if alias in used:
            alias = f"J{idx}"
        aliases[table] = alias
        used.add(alias)
    return aliases


def _build_effective_join_condition(base_alias: str, join_alias: str, rel: Dict[str, Any]) -> str:
    clauses = []
    for item in rel.get("join_columns", []) or []:
        clauses.append(f"{base_alias}.{item['left_column']} = {join_alias}.{item['right_column']}")
    eff = rel.get("effective_date") or {}
    if eff:
        trx_col = eff.get("transaction_date_column")
        start_col = eff.get("start_column")
        end_col = eff.get("end_column")
        if trx_col and start_col and end_col:
            clauses.append(
                f"{base_alias}.{trx_col} BETWEEN {join_alias}.{start_col} AND IFNULL({join_alias}.{end_col}, '99991231')"
            )
    return " AND ".join(clauses)


def _build_classification_case(classification: Dict[str, Any], aliases: Dict[str, str]) -> str:
    lines = ["CASE"]
    for table, merchant_type in (classification.get("join_table_roles") or {}).items():
        alias = aliases.get(table)
        if alias:
            lines.append(f"    WHEN {alias}.MERCHANT_ID IS NOT NULL THEN '{merchant_type}'")
    lines.append("    ELSE 'UNKNOWN'")
    lines.append("END")
    return "\n".join(lines)


def _apply_join_table_filters(rule: Dict[str, Any], join_alias: str, table_meta: Dict[str, Any]) -> List[str]:
    columns = set(_column_names(table_meta))
    clauses = []
    for filt in rule.get("filters", []) or []:
        if filt.get("scope") != "joined_tables":
            continue
        col = str(filt.get("column", "")).strip()
        if col and col in columns:
            op = filt.get("operator", "=")
            val = str(filt.get("value", "")).replace("'", "''")
            clauses.append(f"{join_alias}.{col} {op} '{val}'")
    return clauses


def _build_aggregation_sql(meta: Dict[str, Any], rule: Dict[str, Any]) -> str:
    table_by_name = _table_map(meta)
    source = rule.get("source", {}) or {}
    base_table = source.get("base_table")
    base_alias = source.get("base_alias", "A")
    if base_table not in table_by_name:
        raise ValueError(f"ERWin 메타에 기준 테이블이 없습니다: {base_table}")

    classification = rule.get("classification", {}) or {}
    join_tables = list((classification.get("join_table_roles") or {}).keys())
    missing = [t for t in join_tables if t not in table_by_name]
    if missing:
        raise ValueError(f"ERWin 메타에 조인 테이블이 없습니다: {missing}")

    aliases = _make_aliases(join_tables)
    relations = _relations_for_base(meta, base_table, join_tables)
    rel_by_right = {rel["right_table"]: rel for rel in relations}
    classification_case = _build_classification_case(classification, aliases)

    join_sql_lines = []
    matched_aliases = []
    for join_table in join_tables:
        alias = aliases[join_table]
        rel = rel_by_right.get(join_table)
        if not rel:
            raise ValueError(f"ERWin relations에 조인 관계가 없습니다: {base_table} -> {join_table}")
        join_condition = _build_effective_join_condition(base_alias, alias, rel)
        join_filters = _apply_join_table_filters(rule, alias, table_by_name[join_table])
        all_conditions = [join_condition] + join_filters
        join_sql_lines.append(f"LEFT JOIN {join_table} {alias}\n  ON " + "\n AND ".join(all_conditions))
        matched_aliases.append(alias)

    select_parts = []
    for item in rule.get("select", []) or []:
        expr = str(item.get("expr", "")).strip()
        alias = str(item.get("alias", "")).strip()
        if expr == "__CLASSIFICATION_CASE__":
            expr = classification_case
        select_parts.append(f"{expr} AS {alias}" if alias else expr)

    base_filters = []
    for filt in rule.get("filters", []) or []:
        if filt.get("scope") == "base" and filt.get("condition"):
            base_filters.append(str(filt["condition"]))

    if matched_aliases:
        base_filters.append("(" + " OR ".join([f"{a}.MERCHANT_ID IS NOT NULL" for a in matched_aliases]) + ")")

    group_by_parts = []
    for expr in rule.get("group_by", []) or []:
        expr = str(expr)
        if expr == "__CLASSIFICATION_CASE__":
            expr = classification_case
        group_by_parts.append(expr)

    return (
        "SELECT\n    "
        + ",\n    ".join(select_parts)
        + f"\nFROM {base_table} {base_alias}\n"
        + "\n".join(join_sql_lines)
        + "\nWHERE "
        + "\n  AND ".join(base_filters)
        + "\nGROUP BY\n    "
        + ",\n    ".join(group_by_parts)
    )


def _target_insert_sql(select_sql: str, rule: Dict[str, Any]) -> str:
    target = rule.get("target", {}) or {}
    table = target.get("table")
    columns = target.get("columns", [])
    if not table or not columns:
        return select_sql
    select_columns = [c for c in columns if c != "REG_DTM"]
    return (
        f"INSERT INTO {table} (\n    " + ",\n    ".join(columns) + "\n)\n"
        "SELECT\n    "
        + ",\n    ".join([f"S.{c}" for c in select_columns])
        + ",\n    NOW() AS REG_DTM\n"
        f"FROM (\n{select_sql.rstrip().rstrip(';')}\n) S"
    )


def _build_generic_export_sql(meta: Dict[str, Any], requested_tables: List[str]) -> Tuple[str, Dict[str, Any]]:
    table_by_name = _table_map(meta)
    if not requested_tables:
        raise ValueError("요청서에서 대상 테이블을 찾지 못했습니다.")
    table_name = requested_tables[0]
    table_meta = table_by_name.get(table_name)
    if not table_meta:
        raise ValueError(f"ERWin 메타에 테이블이 없습니다: {table_name}")
    columns = _column_names(table_meta)
    where = "1=1"
    if "BASE_YM" in columns:
        where = "BASE_YM = :base_ym"
    elif "APPLY_START_DT" in columns and "APPLY_END_DT" in columns:
        where = ":base_date BETWEEN APPLY_START_DT AND IFNULL(APPLY_END_DT, '99991231')"
    sql = "SELECT\n    " + ",\n    ".join(columns) + f"\nFROM {table_name}\nWHERE {where}"
    return sql, {"table": table_name, "columns": columns}


def _build_job_py(batch_id: str) -> str:
    return f"""from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


BATCH_ID = "{batch_id}"


def load_spec() -> dict:
    spec_path = Path(__file__).resolve().parent / "batch_spec.json"
    with spec_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(database_url: str, base_ym: str | None = None, base_date: str | None = None) -> None:
    spec = load_spec()
    params = {{"base_ym": base_ym, "base_date": base_date}}
    params = {{k: v for k, v in params.items() if v is not None}}

    engine = create_engine(database_url)
    print(f"[START] {{BATCH_ID}} batch_type={{spec.get('batch_type')}} params={{params}}")

    with engine.begin() as conn:
        target = spec.get("target", {{}})
        if spec.get("batch_type") == "aggregation_to_table" and target.get("load_strategy") == "delete_insert":
            delete_sql = target.get("delete_sql")
            if delete_sql:
                result = conn.execute(text(delete_sql), params)
                print(f"[DELETE] target={{target.get('table')}} rows={{result.rowcount}}")

            result = conn.execute(text(spec["sql"]), params)
            print(f"[INSERT] target={{target.get('table')}} rows={{result.rowcount}}")

        elif spec.get("batch_type") == "db_to_file":
            df = pd.read_sql(text(spec["sql"]), conn, params=params)
            output = spec.get("target", {{}})
            output_dir = Path(output.get("output_dir", "./output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            file_pattern = output.get("output_file_pattern", f"{{BATCH_ID}}_{{base_ym or base_date or 'result'}}.csv")
            file_name = file_pattern.format(base_ym=base_ym, base_date=base_date)
            file_path = output_dir / file_name
            df.to_csv(file_path, index=False, encoding=output.get("encoding", "utf-8-sig"))
            print(f"[FILE] rows={{len(df)}} file={{file_path}}")
        else:
            raise ValueError(f"지원하지 않는 batch_type입니다: {{spec.get('batch_type')}}")

    print(f"[END] {{BATCH_ID}} success")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--base-ym", required=False)
    parser.add_argument("--base-date", required=False)
    args = parser.parse_args()
    run(args.database_url, args.base_ym, args.base_date)


if __name__ == "__main__":
    main()
"""


def _build_readme(batch_spec: Dict[str, Any]) -> str:
    batch_id = batch_spec.get("batch_id", "BATCH")
    return f"""# {batch_id}

## 목적
{batch_spec.get("description", "")}

## 실행 예시

```bash
python generated/{batch_id}/job.py \\
  --database-url "mysql+pymysql://user:pass@localhost:3306/testDB?charset=utf8mb4" \\
  --base-ym 202604
```

## 생성 SQL
```sql
{batch_spec.get("sql", "")}
```

## 운영 반영 전 검토
- ERWin/실제 DB 컬럼 일치 여부
- 조인 결과 중복 여부
- 기준년월 재수행 시 delete_insert 범위
- 집계 금액 검증
"""


def _build_test_job_py() -> str:
    return """from pathlib import Path
import json


def test_batch_spec_exists():
    spec_path = Path(__file__).resolve().parent / "batch_spec.json"
    assert spec_path.exists()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["batch_id"]
    assert spec["sql"]
"""


class BatchDevAgent:
    """요청서 + ERWin 메타 + Rule Catalog 기반 배치 생성기."""

    def __init__(
        self,
        meta_path: str | Path = DEFAULT_META_PATH,
        rule_catalog_path: str | Path = DEFAULT_RULE_CATALOG_PATH,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self.meta_path = Path(meta_path)
        self.rule_catalog_path = Path(rule_catalog_path)
        self.output_root = Path(output_root)

    def run(self, request_text: str) -> BatchDevResult:
        warnings: List[str] = []
        errors: List[str] = []
        created_files: List[str] = []

        try:
            meta = _read_json(self.meta_path)
            rule_catalog = _read_json(self.rule_catalog_path)
            known_tables = list(_table_map(meta).keys())
            requested_tables = _extract_tables(request_text, known_tables)
            output_fields = _extract_output_fields(request_text)
            conditions = _extract_conditions(request_text)
            batch_name = _extract_batch_name(request_text)
            rule = _select_rule(rule_catalog, request_text, requested_tables, output_fields)
            rule_id = rule["rule_id"]
            batch_id = _to_batch_id(rule_id)

            if rule.get("batch_type") == "aggregation_to_table":
                select_sql = _build_aggregation_sql(meta, rule)
                sql = _target_insert_sql(select_sql, rule)
                target = dict(rule.get("target", {}) or {})
                if target.get("delete_condition") and target.get("table"):
                    target["delete_sql"] = f"DELETE FROM {target['table']} WHERE {target['delete_condition']}"
                source = rule.get("source", {})
                parameters = [
                    {"name": source.get("base_period_param", "base_ym"), "required": True, "description": "기준년월(YYYYMM)"}
                ]
            else:
                sql, source = _build_generic_export_sql(meta, requested_tables)
                table = source["table"]
                batch_id = f"BATCH_{table}_EXPORT"
                target = {
                    "output_format": "csv",
                    "output_file_prefix": table.lower(),
                    "output_file_pattern": f"{table.lower()}_{{base_ym}}.csv",
                    "output_dir": "./output",
                    "encoding": "utf-8-sig",
                }
                parameters = [{"name": "base_ym", "required": False, "description": "기준년월(YYYYMM)"}]

            batch_spec: Dict[str, Any] = {
                "version": "1.0",
                "batch_id": batch_id,
                "batch_name": batch_name,
                "batch_type": rule.get("batch_type"),
                "description": _normalize_text(request_text),
                "parameters": parameters,
                "source": source,
                "target": target,
                "requested_tables": requested_tables,
                "requested_output_fields": output_fields,
                "requested_conditions": conditions,
                "sql": sql,
                "validation_rules": {
                    "min_rows": 0,
                    "not_null_columns": ["CUSTOMER_ID"] if rule.get("batch_type") == "aggregation_to_table" else [],
                },
                "meta_source": {
                    "type": "erwin_meta",
                    "path": str(self.meta_path),
                    "tables": requested_tables,
                },
                "rule_source": {
                    "rule_id": rule_id,
                    "path": str(self.rule_catalog_path),
                    "sql_template": rule.get("sql_template"),
                },
            }

            out_dir = self.output_root / batch_id
            out_dir.mkdir(parents=True, exist_ok=True)
            files = {
                "batch_spec.json": json.dumps(batch_spec, ensure_ascii=False, indent=2),
                "query.sql": sql + "\n",
                "job.py": _build_job_py(batch_id),
                "README.md": _build_readme(batch_spec),
                "test_job.py": _build_test_job_py(),
            }
            for name, content in files.items():
                path = out_dir / name
                path.write_text(content, encoding="utf-8")
                created_files.append(str(path))

            if rule_id == "generic_export" and any(word in request_text for word in ["집계", "조인", "분류"]):
                warnings.append("요청서는 복잡 배치로 보이나 generic_export가 선택되었습니다. rule_catalog.json match 조건을 확인하세요.")

            return BatchDevResult(
                success=True,
                message=f"배치 소스가 {out_dir} 폴더에 생성되었습니다. 운영 반영 전 SQL/컬럼/검증조건을 반드시 검토하세요.",
                batch_spec=batch_spec,
                created_files=created_files,
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            return BatchDevResult(
                success=False,
                message="배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요.",
                created_files=created_files,
                warnings=warnings,
                errors=errors,
            )
