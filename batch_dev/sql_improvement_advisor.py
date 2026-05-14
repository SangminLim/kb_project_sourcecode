from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from llm import ChatConfig


SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AND", "OR",
    "CASE", "WHEN", "THEN", "ELSE", "END", "AS", "IS", "NOT", "NULL", "BETWEEN", "IN",
}


@dataclass
class SqlImprovementSuggestion:
    type: str
    target: str
    reason: str
    recommendation: str
    sql: str = ""


@dataclass
class SqlImprovementReport:
    enabled: bool
    risk_level: str
    summary: str
    suggestions: List[SqlImprovementSuggestion]
    generated_by: str = "rule"
    warnings: List[str] | None = None
    raw_llm_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["suggestions"] = [asdict(item) for item in self.suggestions]
        payload["warnings"] = self.warnings or []
        return payload


def _compact(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        value = _compact(item)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _safe_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", _compact(value).upper()).strip("_")
    return text or "UNKNOWN"


def _extract_base_table(sql: str) -> str:
    match = re.search(r"\bFROM\s+([A-Za-z0-9_.$]+)\s+(?:AS\s+)?([A-Za-z0-9_]+)?", sql, re.IGNORECASE)
    return _compact(match.group(1)).upper() if match else ""


def _extract_join_tables(sql: str) -> List[str]:
    return _dedupe([m.group(1).upper() for m in re.finditer(r"\bJOIN\s+([A-Za-z0-9_.$]+)", sql, re.IGNORECASE)])


def _extract_column_refs(sql: str, alias: str) -> List[str]:
    pattern = rf"\b{re.escape(alias)}\.([A-Za-z0-9_]+)\b"
    return _dedupe([m.group(1).upper() for m in re.finditer(pattern, sql, re.IGNORECASE)])


def _extract_alias_for_table(sql: str, table: str) -> str:
    table_pattern = re.escape(table)
    patterns = [
        rf"\bFROM\s+{table_pattern}\s+(?:AS\s+)?([A-Za-z0-9_]+)",
        rf"\bJOIN\s+{table_pattern}\s+(?:AS\s+)?([A-Za-z0-9_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            alias = _compact(match.group(1))
            if alias and alias.upper() not in SQL_KEYWORDS:
                return alias
    return ""


def _extract_where_columns(sql: str, alias: str) -> List[str]:
    where = ""
    match = re.search(r"\bWHERE\b(.+)$", sql, re.IGNORECASE | re.DOTALL)
    if match:
        where = match.group(1)
    return _extract_column_refs(where, alias)


def _extract_on_columns_for_alias(sql: str, alias: str) -> List[str]:
    cols: List[str] = []
    for match in re.finditer(r"\bON\b(.+?)(?=\bLEFT\b|\bRIGHT\b|\bINNER\b|\bOUTER\b|\bJOIN\b|\bWHERE\b|$)", sql, re.IGNORECASE | re.DOTALL):
        cols.extend(_extract_column_refs(match.group(1), alias))
    return _dedupe(cols)


def _choose_index_columns(sql: str, table: str, prefer_base: bool) -> List[str]:
    alias = _extract_alias_for_table(sql, table)
    if not alias:
        return []

    where_cols = _extract_where_columns(sql, alias)
    on_cols = _extract_on_columns_for_alias(sql, alias)

    priority_names = ["BASE_YM", "BASE_YYMM", "SALES_DT", "TR_DT", "MERCHANT_ID", "CUST_ID", "CUSTOMER_ID", "USE_YN"]
    if prefer_base:
        candidates = where_cols + on_cols
    else:
        candidates = on_cols + where_cols

    ordered: List[str] = []
    for col in priority_names:
        if col in candidates:
            ordered.append(col)
    for col in candidates:
        if col not in ordered:
            ordered.append(col)

    return ordered[:4]


def _index_name(table: str, idx_no: int = 1) -> str:
    base = _safe_identifier(table)
    if base.startswith("TB_"):
        base = base[3:]
    return f"IX_{base}_{idx_no:02d}"


def _has_left_join_filter_pattern(sql: str) -> bool:
    upper = sql.upper()
    return "LEFT JOIN" in upper and re.search(r"\bWHERE\b.+IS\s+NOT\s+NULL", upper, re.DOTALL) is not None


def _build_validation_sql(batch_spec: Dict[str, Any], base_table: str) -> str:
    sql = _compact(batch_spec.get("sql"))
    base_month_col = "BASE_YM"
    cancel_col = "CANCEL_YN"
    if sql:
        base_match = re.search(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]*BASE[A-Za-z0-9_]*(?:YM|YYMM|MONTH)?)\s*=\s*:base_ym", sql, re.IGNORECASE)
        if base_match:
            base_month_col = base_match.group(2).upper()
        cancel_match = re.search(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]*CANCEL[A-Za-z0-9_]*|CANCEL_YN)\s*=\s*'N'", sql, re.IGNORECASE)
        if cancel_match:
            cancel_col = cancel_match.group(2).upper()

    table = base_table or "기준테이블"
    return (
        f"SELECT\n"
        f"    COUNT(*) AS ROW_COUNT,\n"
        f"    COUNT(DISTINCT MERCHANT_ID) AS MERCHANT_COUNT,\n"
        f"    SUM(SALES_AMT) AS SALES_AMT_SUM\n"
        f"FROM {table}\n"
        f"WHERE {base_month_col} = :base_ym\n"
        f"  AND {cancel_col} = 'N';"
    )


def build_rule_based_sql_improvement(batch_spec: Dict[str, Any], generated_files: Optional[Dict[str, str]] = None) -> SqlImprovementReport:
    generated_files = generated_files or {}
    query_sql = _compact(batch_spec.get("sql") or generated_files.get("query.sql"))
    if not query_sql:
        return SqlImprovementReport(
            enabled=False,
            risk_level="UNKNOWN",
            summary="개선 제안을 만들 SQL을 찾지 못했습니다.",
            suggestions=[],
            warnings=["batch_spec.sql 또는 generated_files['query.sql']을 확인하세요."],
        )

    base_table = _extract_base_table(query_sql)
    join_tables = _extract_join_tables(query_sql)
    suggestions: List[SqlImprovementSuggestion] = []
    warnings: List[str] = []

    if base_table:
        cols = _choose_index_columns(query_sql, base_table, prefer_base=True)
        if cols:
            idx_sql = f"CREATE INDEX {_index_name(base_table, 1)} ON {base_table}({', '.join(cols)});"
            suggestions.append(SqlImprovementSuggestion(
                type="INDEX",
                target=base_table,
                reason="WHERE 조건과 JOIN 조건에 반복 사용되는 기준 테이블 컬럼을 기준으로 복합 인덱스를 추천합니다.",
                recommendation=f"{base_table} 기준 조회/조인 성능 개선을 위해 복합 인덱스 생성을 검토하세요.",
                sql=idx_sql,
            ))

    for idx, table in enumerate(join_tables, start=1):
        cols = _choose_index_columns(query_sql, table, prefer_base=False)
        if not cols:
            continue
        idx_sql = f"CREATE INDEX {_index_name(table, idx)} ON {table}({', '.join(cols)});"
        suggestions.append(SqlImprovementSuggestion(
            type="INDEX",
            target=table,
            reason="가맹점/참조 테이블은 조인키, 사용여부, 적용기간 조건을 함께 타므로 조건 컬럼 조합 인덱스 검토가 필요합니다.",
            recommendation=f"{table} 조인 성능 개선을 위해 조건 컬럼 복합 인덱스를 검토하세요.",
            sql=idx_sql,
        ))

    if _has_left_join_filter_pattern(query_sql):
        suggestions.append(SqlImprovementSuggestion(
            type="JOIN",
            target="LEFT JOIN + WHERE IS NOT NULL",
            reason="LEFT JOIN 후 WHERE에서 참조 테이블 존재 여부를 필터링하면 사실상 INNER JOIN/SEMI JOIN 성격이 됩니다.",
            recommendation="누락 거래를 의도적으로 제외하는 배치라면 INNER JOIN 또는 EXISTS 방식과 실행계획을 비교하세요. 여러 분류 테이블 중 하나라도 매칭되는 구조라면 UNION ALL + 우선순위 분류도 검토할 수 있습니다.",
            sql="-- 실행계획 비교 대상\n-- 1) 현재 LEFT JOIN 유지\n-- 2) EXISTS 조건으로 대상 거래 선별\n-- 3) 분류별 UNION ALL 후 우선순위 적용",
        ))

    if "BASE_YM" in query_sql.upper() or "SALES_DT" in query_sql.upper():
        partition_target = base_table or "대량 기준 테이블"
        suggestions.append(SqlImprovementSuggestion(
            type="PARTITION",
            target=partition_target,
            reason="월 배치/거래 원장성 테이블은 기준년월 또는 거래일자 조건으로 대량 데이터를 반복 조회할 가능성이 높습니다.",
            recommendation="운영 데이터 건수가 큰 경우 BASE_YM 또는 SALES_DT 기준 월 파티션 적용 여부를 DBA와 검토하세요.",
            sql="-- 예시: 월 단위 RANGE/LIST 파티션 정책 검토\n-- 기준 컬럼 후보: BASE_YM, SALES_DT",
        ))

    suggestions.append(SqlImprovementSuggestion(
        type="VALIDATION",
        target=base_table or "생성 SQL",
        reason="운영 반영 전 월별 대상 건수, 가맹점 수, 금액 합계를 비교해야 재처리/누락/중복 위험을 줄일 수 있습니다.",
        recommendation="배치 실행 전후 검증 SQL을 test_job.py 또는 별도 검증 스크립트에 추가하세요.",
        sql=_build_validation_sql(batch_spec, base_table),
    ))

    if not suggestions:
        warnings.append("SQL 구조에서 자동 추천할 조건 컬럼을 충분히 찾지 못했습니다.")

    risk_level = "MEDIUM" if join_tables else "LOW"
    if len(join_tables) >= 3:
        risk_level = "HIGH"

    summary = (
        f"기준 테이블 {base_table or '미확인'}와 참조 테이블 {len(join_tables)}개를 분석했습니다. "
        "운영 반영 전 인덱스, 조인 방식, 파티션, 데이터 검증 SQL을 확인하세요."
    )

    return SqlImprovementReport(
        enabled=True,
        risk_level=risk_level,
        summary=summary,
        suggestions=suggestions,
        generated_by="rule",
        warnings=warnings,
    )


def build_llm_prompt(batch_spec: Dict[str, Any], generated_files: Dict[str, str], rule_report: SqlImprovementReport) -> str:
    query_sql = _compact(batch_spec.get("sql") or generated_files.get("query.sql"))
    return f"""
너는 금융권 배치 SQL 성능/운영 리뷰어다.
아래 룰 기반 분석 결과를 참고해서 SQL 개선 제안을 더 구체화해라.

중요 원칙:
- 운영 SQL을 자동 변경하지 말고 개선 후보만 제안한다.
- 테이블명/컬럼명은 입력 SQL과 메타에 있는 값만 사용한다.
- 근거 없는 컬럼명은 만들지 않는다.
- 반드시 JSON만 응답한다.

응답 형식:
{{
  "risk_level": "LOW|MEDIUM|HIGH",
  "summary": "요약",
  "suggestions": [
    {{
      "type": "INDEX|JOIN|EXISTS|PARTITION|VALIDATION",
      "target": "대상",
      "reason": "이유",
      "recommendation": "개선 제안",
      "sql": "SQL 예시가 있으면 작성"
    }}
  ]
}}

생성 SQL:
{query_sql}

배치 명세:
{json.dumps(batch_spec, ensure_ascii=False, indent=2)}

룰 기반 1차 분석:
{json.dumps(rule_report.to_dict(), ensure_ascii=False, indent=2)}
""".strip()


def _call_llm(
    llm_generate_fn: Callable[..., Any],
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Upstage/Ollama 공통 LLM 호출 래퍼.

    - system_prompt/config 자동 주입
    - JSON 응답 안정화
    - provider 변경 시 app.py 수정 최소화
    """

    config = ChatConfig()

    system_prompt = """
너는 금융권 배치 SQL 성능 리뷰 전문가다.

반드시 유효한 JSON 객체 하나만 응답한다.
markdown code block, 설명 문장, 머리말, 꼬리말은 출력하지 않는다.
SQL 예시가 길거나 따옴표/줄바꿈이 포함되면 sql 필드는 생략한다.
문자열 안 줄바꿈은 반드시 \\n 으로 이스케이프한다.
""".strip()

    try:
        response = llm_generate_fn(
            prompt=prompt,
            system_prompt=system_prompt,
            config=config,
        )

        if hasattr(response, "content"):
            return str(response.content)

        return str(response)

    except Exception:
        # fallback 호환
        response = llm_generate_fn(prompt)

        if hasattr(response, "content"):
            return str(response.content)

        return str(response)



def _strip_markdown_fence(text: str) -> str:
    raw = _compact(text)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw.strip()).strip()
    return raw


def _extract_first_json_object(text: str) -> str:
    """응답 문자열에서 첫 번째 JSON object 후보를 안전하게 추출한다.

    단순 rfind 방식은 JSON 뒤에 설명 문장이 붙거나 문자열 내부 brace가 있을 때 취약하다.
    따옴표/escape 상태를 추적해서 균형 잡힌 첫 JSON 객체만 반환한다.
    """
    raw = _strip_markdown_fence(text)
    start = raw.find("{")
    if start < 0:
        raise ValueError("LLM 응답에서 JSON 시작 '{'를 찾지 못했습니다.")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(raw)):
        ch = raw[idx]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    raise ValueError("LLM 응답에서 완결된 JSON 객체를 찾지 못했습니다.")


def _coerce_llm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """LLM payload를 화면/저장용 표준 구조로 정규화한다.

    LLM이 일부 필드를 누락하거나 다른 타입으로 반환해도 앱이 깨지지 않도록 보정한다.
    하드코딩된 업무값을 만들지 않고, 들어온 값의 타입/필드만 정리한다.
    """
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload가 object 형식이 아닙니다.")

    suggestions = payload.get("suggestions", [])
    if isinstance(suggestions, dict):
        suggestions = [suggestions]
    if not isinstance(suggestions, list):
        suggestions = []

    normalized_suggestions: List[Dict[str, Any]] = []
    for item in suggestions:
        if not isinstance(item, dict):
            continue
        normalized_suggestions.append({
            "type": _compact(item.get("type")) or "RECOMMENDATION",
            "target": _compact(item.get("target")),
            "reason": _compact(item.get("reason")),
            "recommendation": _compact(item.get("recommendation")),
            "sql": _compact(item.get("sql")),
        })

    return {
        "risk_level": _compact(payload.get("risk_level")),
        "summary": _compact(payload.get("summary")),
        "suggestions": normalized_suggestions,
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = _strip_markdown_fence(text)
    if not raw:
        raise ValueError("empty llm response")

    # 1차: 응답 전체가 정상 JSON인 경우
    try:
        return _coerce_llm_payload(json.loads(raw))
    except json.JSONDecodeError:
        pass

    # 2차: 앞뒤 설명/코드블록이 섞인 경우 첫 JSON object만 추출
    json_candidate = _extract_first_json_object(raw)
    return _coerce_llm_payload(json.loads(json_candidate))


def _merge_llm_suggestions_with_rule_sql(
    llm_items: List[Dict[str, Any]],
    fallback: SqlImprovementReport,
) -> List[SqlImprovementSuggestion]:
    """LLM 제안에는 설명을, 룰 제안에는 안전한 SQL 예시를 맡긴다.

    LLM이 JSON 안에 긴 SQL을 넣다가 파싱 오류를 내는 문제를 줄이기 위해
    동일 type/target의 룰 SQL을 자동 보강한다.
    """
    rule_sql_by_key: Dict[tuple[str, str], str] = {}
    for rule_item in fallback.suggestions or []:
        key = (_compact(rule_item.type).upper(), _compact(rule_item.target).upper())
        if rule_item.sql:
            rule_sql_by_key[key] = rule_item.sql

    suggestions: List[SqlImprovementSuggestion] = []
    for item in llm_items or []:
        item_type = _compact(item.get("type")) or "RECOMMENDATION"
        target = _compact(item.get("target"))
        key = (item_type.upper(), target.upper())
        sql = _compact(item.get("sql")) or rule_sql_by_key.get(key, "")

        suggestions.append(SqlImprovementSuggestion(
            type=item_type,
            target=target,
            reason=_compact(item.get("reason")),
            recommendation=_compact(item.get("recommendation")),
            sql=sql,
        ))

    return suggestions


def _report_from_payload(payload: Dict[str, Any], fallback: SqlImprovementReport, raw: str) -> SqlImprovementReport:
    llm_items = payload.get("suggestions", []) or []
    suggestions = _merge_llm_suggestions_with_rule_sql(llm_items, fallback)

    return SqlImprovementReport(
        enabled=True,
        risk_level=_compact(payload.get("risk_level")) or fallback.risk_level,
        summary=_compact(payload.get("summary")) or fallback.summary,
        suggestions=suggestions or fallback.suggestions,
        generated_by="llm",
        warnings=fallback.warnings or [],
        raw_llm_response=raw,
    )


def analyze_sql_improvement(
    batch_spec: Dict[str, Any],
    generated_files: Optional[Dict[str, str]] = None,
    llm_generate_fn: Optional[Callable[..., Any]] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """배치 SQL 개선 제안을 생성한다.

    기본은 룰 기반 분석이며, llm_generate_fn이 전달되면 LLM으로 2차 보강한다.
    LLM 실패 시에도 룰 기반 결과를 반환해서 Streamlit 화면 장애를 막는다.
    """
    generated_files = generated_files or {}
    rule_report = build_rule_based_sql_improvement(batch_spec=batch_spec, generated_files=generated_files)
    final_report = rule_report

    if use_llm and llm_generate_fn is not None and rule_report.enabled:
        prompt = build_llm_prompt(batch_spec=batch_spec, generated_files=generated_files, rule_report=rule_report)
        try:
            raw = _call_llm(llm_generate_fn=llm_generate_fn, prompt=prompt, model=model)
            payload = _parse_json_object(raw)
            final_report = _report_from_payload(payload=payload, fallback=rule_report, raw=raw)
        except Exception as exc:
            warnings = list(rule_report.warnings or [])
            warnings.append(f"LLM SQL 개선 제안 실패로 룰 기반 결과를 사용했습니다: {exc}")
            rule_report.warnings = warnings
            final_report = rule_report

    report_dict = final_report.to_dict()
    if output_dir:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sql_improvement_report.json").write_text(
                json.dumps(report_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            report_dict.setdefault("warnings", []).append(f"sql_improvement_report.json 저장 실패: {exc}")
    return report_dict
