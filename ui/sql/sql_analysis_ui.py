from __future__ import annotations

from ..context import *
from ..renderers.streamlit_renderers import unique_preserve_order

def _normalize_section_name(name: str) -> str:
    """요청서 섹션명을 canonical key로 변환한다."""
    text = re.sub(r"[\s_\-:\[\]【】()（）]+", "", str(name or "")).upper()
    for canonical, aliases in SQL_ANALYSIS_SECTION_ALIASES.items():
        for alias in aliases:
            alias_text = re.sub(r"[\s_\-:\[\]【】()（）]+", "", alias).upper()
            if text == alias_text:
                return canonical
    return "context"


def _empty_sql_analysis_sections() -> Dict[str, str]:
    return {key: "" for key in SQL_ANALYSIS_SECTION_ALIASES.keys()}


def parse_sql_analysis_request(request_text: str) -> Dict[str, str]:
    """SQL 분석 요청서를 섹션 기반으로 파싱한다.

    실무 적용 원칙:
    - [SQL] SELECT ... 처럼 섹션명과 본문이 같은 줄에 있어도 인식한다.
    - [검토포인트], [출력요청], [운영정보] 같은 추가 섹션도 구조화해서 보존한다.
    - 섹션 alias는 SQL_ANALYSIS_SECTION_ALIASES만 확장하면 되도록 분리한다.
    - 섹션이 없어도 SQL 키워드 위치를 기준으로 최대한 복구한다.
    """
    text = str(request_text or "").strip()
    result = _empty_sql_analysis_sections()
    if not text:
        return result

    # [섹션명] 또는 【섹션명】 토큰을 문장 중간에서도 찾는다.
    # 예: "[SQL] SELECT ..." / "[운영정보] DBMS=ORACLE ..."
    bracket_pattern = re.compile(r"(?:\[([^\]]+)\]|【([^】]+)】)")
    matches = list(bracket_pattern.finditer(text))

    if matches:
        for idx, match in enumerate(matches):
            raw_name = next((g for g in match.groups() if g), "")
            key = _normalize_section_name(raw_name)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                result[key] = (result.get(key, "") + "\n\n" + body).strip()
    else:
        # 보조 포맷: "SQL: ..." / "수정내용: ..." 형태 지원
        line_header_pattern = re.compile(r"^\s*([^\n:：]{1,40})\s*[:：]\s*(.*)$", re.MULTILINE)
        line_matches = list(line_header_pattern.finditer(text))
        if line_matches:
            for idx, match in enumerate(line_matches):
                raw_name = match.group(1)
                key = _normalize_section_name(raw_name)
                inline_body = match.group(2).strip()
                start = match.end()
                end = line_matches[idx + 1].start() if idx + 1 < len(line_matches) else len(text)
                body = (inline_body + "\n" + text[start:end].strip()).strip()
                if body:
                    result[key] = (result.get(key, "") + "\n\n" + body).strip()

    if not result.get("sql"):
        sql_match = re.search(r"\b(WITH|SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP)\b", text, re.IGNORECASE)
        if sql_match:
            result["sql"] = text[sql_match.start():].strip()
            prefix = text[:sql_match.start()].strip()
            if prefix and not result.get("context"):
                result["context"] = prefix
        else:
            result["context"] = text

    return result


def _extract_sql_identifiers(sql: str) -> Dict[str, List[str]]:
    """SQL에서 주요 객체를 가볍게 추출한다. DB dialect 의존 파서는 쓰지 않고 보수적으로 처리한다."""
    cleaned = re.sub(r"/\*.*?\*/", " ", sql or "", flags=re.DOTALL)
    cleaned = re.sub(r"--.*?$", " ", cleaned, flags=re.MULTILINE)
    table_tokens = re.findall(r"\b(?:FROM|JOIN|INTO|UPDATE|MERGE\s+INTO|DELETE\s+FROM)\s+([A-Za-z0-9_.$]+)", cleaned, flags=re.IGNORECASE)
    aliases = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z0-9_.$]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?", cleaned, flags=re.IGNORECASE)
    return {
        "tables": unique_preserve_order([token.strip() for token in table_tokens if token.strip()]),
        "aliases": unique_preserve_order([f"{table} {alias}".strip() for table, alias in aliases if table and alias and alias.upper() not in {"ON", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "GROUP", "ORDER"}]),
    }


def _detect_sql_statement_type(sql: str) -> str:
    match = re.search(r"\b(WITH|SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP)\b", sql or "", re.IGNORECASE)
    if not match:
        return "UNKNOWN"
    token = match.group(1).upper()
    if token == "WITH":
        return "SELECT/CTE"
    return token



def _parse_key_value_text(text: str) -> Dict[str, str]:
    """요청서의 운영정보/메타정보를 key-value 사전으로 변환한다.

    지원 예:
    - DBMS=ORACLE 배치주기=MONTHLY 예상데이터건수=50000000
    - DBMS=ORACLE
      PARTITION_COLUMN=BASE_YM
    - 파티션컬럼: BASE_YM
    """
    raw = str(text or "").strip()
    meta: Dict[str, str] = {}
    if not raw:
        return meta

    # key=value 형태는 같은 줄에 여러 개 있어도 추출한다.
    for key, value in re.findall(r"([A-Za-z가-힣_][A-Za-z0-9가-힣_ ]{0,40})\s*=\s*([^\s,;]+)", raw):
        norm_key = re.sub(r"\s+", "", key).strip().upper()
        meta[norm_key] = str(value).strip()

    # key: value 형태도 보조 지원한다.
    for line in raw.splitlines():
        if "=" in line:
            continue
        match = re.match(r"\s*([^:：]{1,40})\s*[:：]\s*(.+?)\s*$", line)
        if not match:
            continue
        norm_key = re.sub(r"\s+", "", match.group(1)).strip().upper()
        meta[norm_key] = match.group(2).strip()

    alias_map = {
        "PARTITION_COLUMN": ["PARTITION_COLUMN", "PARTITIONCOL", "파티션컬럼", "파티션키", "PARTITIONKEY"],
        "INDEX_COLUMN": ["INDEX_COLUMN", "INDEXCOL", "인덱스컬럼", "인덱스키", "INDEXKEY"],
        "EXPECTED_ROWS": ["EXPECTED_ROWS", "예상데이터건수", "예상건수", "데이터건수", "ROWS", "ROWCOUNT"],
        "DBMS": ["DBMS", "DATABASE", "DB"],
        "BATCH_CYCLE": ["배치주기", "BATCH_CYCLE", "CYCLE", "SCHEDULE"],
    }

    normalized: Dict[str, str] = dict(meta)
    for canonical, aliases in alias_map.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            alias_key = re.sub(r"\s+", "", alias).strip().upper()
            if alias_key in meta:
                normalized[canonical] = meta[alias_key]
                break

    return normalized


def _safe_int(value: Any, default: int = 0) -> int:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.$]", "", str(value or "")).upper()


def _extract_where_clause(sql: str) -> str:
    """SQL에서 WHERE 절을 보수적으로 추출한다."""
    text = str(sql or "")
    match = re.search(
        r"\bWHERE\b([\s\S]*?)(\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bUNION\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _append_finding(
    findings: List[Dict[str, str]],
    *,
    item: str,
    level: str,
    detail: str,
    evidence: str = "",
    recommendation: str = "",
) -> None:
    """동일 item 중복 추가를 방지하면서 표준 finding을 추가한다."""
    if any(existing.get("item") == item for existing in findings):
        return
    payload = {
        "item": item,
        "level": level,
        "detail": detail,
    }
    if evidence:
        payload["evidence"] = evidence
    if recommendation:
        payload["recommendation"] = recommendation
    findings.append(payload)


def _review_points_contain(review_points: str, *keywords: str) -> bool:
    upper = str(review_points or "").upper()
    return any(str(keyword).upper() in upper for keyword in keywords)


def build_rule_based_sql_analysis(
    sql: str,
    change_request: str = "",
    operation_info: str = "",
    review_points: str = "",
) -> Dict[str, Any]:
    """LLM 없이도 동작하는 보수적 SQL 분석 결과를 만든다.

    실무 적용 원칙:
    - 특정 업무명/테이블명/컬럼명을 코드에 박지 않는다.
    - 요청서의 운영정보(PARTITION_COLUMN, INDEX_COLUMN, 예상데이터건수)를 룰 판단에 사용한다.
    - LLM은 이 결과를 설명/보강만 하고, deterministic finding은 평가용 근거로 남긴다.
    """
    sql_text = str(sql or "").strip()
    upper_sql = sql_text.upper()
    identifiers = _extract_sql_identifiers(sql_text)
    statement_type = _detect_sql_statement_type(sql_text)
    operation_meta = _parse_key_value_text(operation_info)
    where_clause = _extract_where_clause(sql_text)
    where_upper = where_clause.upper()

    findings: List[Dict[str, str]] = []
    warnings: List[str] = []
    guide: List[str] = []

    if not sql_text:
        return {
            "success": False,
            "summary": "분석할 SQL이 없습니다.",
            "statement_type": "UNKNOWN",
            "tables": [],
            "aliases": [],
            "operation_meta": operation_meta,
            "where_clause": "",
            "findings": [],
            "warnings": ["요청서에 [SQL] 섹션 또는 SQL 문장을 포함하세요."],
            "change_guide": [],
            "generated_by": "rule",
        }

    checks = [
        (
            "WHERE 조건 확인",
            statement_type in {"UPDATE", "DELETE"} and " WHERE " not in f" {upper_sql} ",
            "UPDATE/DELETE SQL에 WHERE 조건이 없으면 대량 변경 위험이 있습니다.",
            "HIGH",
            "UPDATE/DELETE 문에서 WHERE 절 미검출",
            "대량 변경 방지를 위해 WHERE 조건 또는 승인된 전체 처리 근거를 확인하세요.",
        ),
        (
            "SELECT 전체 컬럼 확인",
            bool(re.search(r"SELECT\s+\*", upper_sql)),
            "SELECT * 는 컬럼 변경 영향과 불필요한 I/O가 커질 수 있어 필요한 컬럼 명시를 권장합니다.",
            "WARN",
            "SELECT * 패턴 검출",
            "필요 컬럼만 명시하고 결과 컬럼 변경 영향을 줄이세요.",
        ),
        (
            "JOIN 조건 확인",
            " JOIN " in f" {upper_sql} " and " ON " not in f" {upper_sql} " and " USING " not in f" {upper_sql} ",
            "JOIN이 있지만 ON/USING 조건이 확인되지 않습니다. 카티션 조인 위험을 확인하세요.",
            "HIGH",
            "JOIN 키워드는 있으나 ON/USING 조건 미검출",
            "조인 조건 누락 여부와 카티션 조인 가능성을 확인하세요.",
        ),
        (
            "정렬 비용 확인",
            " ORDER BY " in f" {upper_sql} ",
            "ORDER BY는 대량 데이터에서 정렬 비용이 큽니다. 페이징/인덱스/정렬 필요성을 확인하세요.",
            "WARN",
            "ORDER BY 절 검출",
            "정렬 컬럼 인덱스, 페이징 조건, 정렬 필요성을 확인하세요.",
        ),
        (
            "그룹 집계 확인",
            " GROUP BY " in f" {upper_sql} ",
            "GROUP BY 집계 SQL입니다. 집계 기준 컬럼과 중복 발생 여부를 확인하세요.",
            "INFO",
            "GROUP BY 절 검출",
            "집계 기준 컬럼이 업무 key와 일치하는지, 조인 후 중복 집계가 없는지 확인하세요.",
        ),
        (
            "서브쿼리 확인",
            bool(re.search(r"\(\s*SELECT\b", upper_sql)),
            "서브쿼리가 있습니다. 실행계획에서 반복 수행 여부와 조인 전환 가능성을 확인하세요.",
            "WARN",
            "서브쿼리 패턴 검출",
            "상관 서브쿼리 반복 수행 여부와 JOIN/CTE 전환 가능성을 검토하세요.",
        ),
        (
            "함수 조건 확인",
            bool(re.search(r"WHERE[\s\S]*(DATE_FORMAT|SUBSTR|SUBSTRING|TO_CHAR|NVL|COALESCE|IFNULL)\s*\(", upper_sql)),
            "WHERE 조건의 컬럼 함수 사용은 인덱스 사용을 방해할 수 있습니다.",
            "WARN",
            "WHERE 절 내 함수 사용 패턴 검출",
            "컬럼에 함수 적용 대신 범위 조건 또는 함수 기반 인덱스 사용 여부를 확인하세요.",
        ),
        (
            "UNION 중복 제거 확인",
            " UNION " in f" {upper_sql} " and " UNION ALL " not in f" {upper_sql} ",
            "UNION은 중복 제거 비용이 있습니다. 중복 제거가 불필요하면 UNION ALL 검토가 필요합니다.",
            "WARN",
            "UNION 사용 및 UNION ALL 미사용",
            "중복 제거가 필요한 요구사항인지 확인하고 불필요하면 UNION ALL을 검토하세요.",
        ),
    ]

    for title, condition, message, level, evidence, recommendation in checks:
        if condition:
            _append_finding(
                findings,
                item=title,
                level=level,
                detail=message,
                evidence=evidence,
                recommendation=recommendation,
            )

    partition_column = (
        operation_meta.get("PARTITION_COLUMN")
        or operation_meta.get("PARTITIONCOL")
        or operation_meta.get("파티션컬럼")
        or ""
    )
    index_columns_raw = (
        operation_meta.get("INDEX_COLUMN")
        or operation_meta.get("INDEXCOL")
        or operation_meta.get("인덱스컬럼")
        or ""
    )
    expected_rows = _safe_int(operation_meta.get("EXPECTED_ROWS") or operation_meta.get("예상데이터건수"))
    large_table_threshold_rows = _safe_int(
        SQL_REVIEW_POLICY.get("large_table_threshold_rows"),
        10_000_000,
    )

    if partition_column:
        partition_norm = _normalize_identifier(partition_column)
        where_norm = _normalize_identifier(where_clause)
        if where_clause and partition_norm and partition_norm not in where_norm:
            _append_finding(
                findings,
                item="파티션 조건 누락",
                level="HIGH" if expected_rows >= large_table_threshold_rows else "MEDIUM",
                detail=f"운영정보의 PARTITION_COLUMN={partition_column} 이지만 WHERE 절에서 해당 조건이 확인되지 않습니다.",
                evidence=f"PARTITION_COLUMN={partition_column}, WHERE={where_clause}",
                recommendation=f"대상 기간/기준월이 명확하다면 WHERE 절에 {partition_column} 조건 추가를 검토하세요.",
            )
        elif not where_clause:
            _append_finding(
                findings,
                item="WHERE 절 없음",
                level="HIGH" if expected_rows >= large_table_threshold_rows else "MEDIUM",
                detail=f"PARTITION_COLUMN={partition_column} 이 제공됐지만 WHERE 절이 없어 파티션 프루닝 여부를 판단할 수 없습니다.",
                evidence=f"PARTITION_COLUMN={partition_column}, WHERE 절 미검출",
                recommendation=f"대량 데이터 SQL이면 {partition_column} 기준 필터 조건을 명확히 지정하세요.",
            )

    if expected_rows >= large_table_threshold_rows:
        if not where_clause:
            _append_finding(
                findings,
                item="대량 데이터 무조건 조회 위험",
                level="HIGH",
                detail=f"예상 데이터 건수 {expected_rows:,}건인데 WHERE 절이 확인되지 않습니다.",
                evidence=f"EXPECTED_ROWS={expected_rows}",
                recommendation="업무 기준일자, 파티션 키, 상태값 등 최소 필터 조건을 확인하세요.",
            )
        elif not partition_column:
            _append_finding(
                findings,
                item="대량 데이터 파티션 기준정보 부족",
                level="MEDIUM",
                detail=f"예상 데이터 건수 {expected_rows:,}건이지만 운영정보에 PARTITION_COLUMN이 없습니다.",
                evidence=f"EXPECTED_ROWS={expected_rows}, PARTITION_COLUMN 미제공",
                recommendation="대량 테이블이면 파티션 키 또는 기준일자 컬럼 정보를 요청서에 포함하세요.",
            )

    if index_columns_raw:
        index_columns = [
            item.strip()
            for item in re.split(r"[,/| ]+", index_columns_raw)
            if item.strip()
        ]
        where_and_join = f"{where_clause}\n{sql_text}"
        normalized_sql_area = _normalize_identifier(where_and_join)
        missing_index_columns = [
            col for col in index_columns
            if _normalize_identifier(col) and _normalize_identifier(col) not in normalized_sql_area
        ]
        if missing_index_columns and _review_points_contain(review_points, "INDEX", "인덱스"):
            _append_finding(
                findings,
                item="요청 인덱스 컬럼 활용 확인 필요",
                level="MEDIUM",
                detail=f"운영정보의 INDEX_COLUMN 중 SQL 조건/조인 영역에서 명확히 확인되지 않는 컬럼이 있습니다: {', '.join(missing_index_columns)}",
                evidence=f"INDEX_COLUMN={index_columns_raw}",
                recommendation="실제 실행계획에서 인덱스 스캔 여부와 선행 컬럼 조건 사용 여부를 확인하세요.",
            )

    if _review_points_contain(review_points, "FULL TABLE SCAN", "FULL SCAN", "풀스캔") and expected_rows >= large_table_threshold_rows:
        _append_finding(
            findings,
            item="FULL SCAN 중점 검토",
            level="HIGH",
            detail="검토포인트에 FULL SCAN 확인이 포함되어 있고 예상 데이터 건수가 큽니다.",
            evidence=f"review_points={review_points.strip()}, EXPECTED_ROWS={expected_rows}",
            recommendation="DB 실행계획에서 TABLE ACCESS FULL, PARTITION RANGE, INDEX RANGE SCAN 여부를 확인하세요.",
        )

    if _review_points_contain(review_points, "중복", "DUPLICATE") and " JOIN " in f" {upper_sql} " and " GROUP BY " in f" {upper_sql} ":
        _append_finding(
            findings,
            item="조인 후 중복 집계 확인",
            level="MEDIUM",
            detail="JOIN 이후 GROUP BY 집계가 수행됩니다. 마스터/상세 테이블의 조인 cardinality에 따라 중복 집계가 발생할 수 있습니다.",
            evidence="JOIN + GROUP BY + 중복 검토포인트 검출",
            recommendation="조인 대상 키의 유일성, 조인 전/후 row count, 집계 금액 합계를 비교하세요.",
        )

    if not identifiers["tables"]:
        warnings.append("테이블명을 추출하지 못했습니다. SQL 문법 또는 동적 SQL 여부를 확인하세요.")

    if change_request.strip():
        guide.extend([
            "수정내용을 기준으로 SQL을 바로 변경하기 전에 영향 범위, 대상 건수, 기존 결과와의 차이를 먼저 확인하세요.",
            "변경 전/후 SQL을 같은 기준일자와 샘플 파라미터로 실행해 row count, 금액 합계, key 중복 여부를 비교하세요.",
            "운영 반영이 필요한 SQL이면 실행계획, 인덱스, 트랜잭션 범위, 롤백 방법을 함께 검토하세요.",
        ])
    else:
        guide.append("수정내용이 없으므로 SQL 구조/위험요소 중심으로 분석했습니다.")

    summary_parts = [f"{statement_type} SQL로 판단됩니다."]
    if identifiers["tables"]:
        summary_parts.append(f"주요 테이블은 {', '.join(identifiers['tables'])} 입니다.")
    if operation_meta:
        summary_parts.append("운영정보를 룰 분석에 반영했습니다.")
    summary_parts.append(f"검토 포인트 {len(findings)}건을 확인했습니다.")

    return {
        "success": True,
        "summary": " ".join(summary_parts),
        "statement_type": statement_type,
        "tables": identifiers["tables"],
        "aliases": identifiers["aliases"],
        "operation_meta": operation_meta,
        "where_clause": where_clause,
        "findings": findings,
        "warnings": warnings,
        "change_guide": guide,
        "generated_by": "rule",
    }



def _build_sql_analysis_chat_config() -> Any:
    """프로젝트 공통 ChatConfig를 우선 사용하되, 생성자 차이가 있어도 앱이 죽지 않도록 보수적으로 처리한다."""
    candidates = [
        {
            "model": SQL_ANALYSIS_LLM_MODEL,
            "temperature": SQL_ANALYSIS_TEMPERATURE,
            "timeout": SQL_ANALYSIS_LLM_TIMEOUT,
            "max_tokens": SQL_ANALYSIS_MAX_TOKENS,
        },
        {
            "model": SQL_ANALYSIS_LLM_MODEL,
            "temperature": SQL_ANALYSIS_TEMPERATURE,
            "timeout": SQL_ANALYSIS_LLM_TIMEOUT,
        },
        {
            "model": SQL_ANALYSIS_LLM_MODEL,
            "timeout": SQL_ANALYSIS_LLM_TIMEOUT,
        },
        {
            "model": SQL_ANALYSIS_LLM_MODEL,
        },
    ]

    for kwargs in candidates:
        try:
            return ChatConfig(**kwargs)
        except TypeError:
            continue

    # ChatConfig 생성자 형태가 달라도 ollama_generate가 속성 접근 방식이면 동작하도록 fallback을 둔다.
    return SimpleNamespace(
        model=SQL_ANALYSIS_LLM_MODEL,
        temperature=SQL_ANALYSIS_TEMPERATURE,
        timeout=SQL_ANALYSIS_LLM_TIMEOUT,
        max_tokens=SQL_ANALYSIS_MAX_TOKENS,
    )


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    """LLM 응답에서 JSON 객체만 보수적으로 추출한다."""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except Exception:
        return None


def run_sql_analysis_llm(sql: str, change_request: str, rule_report: Dict[str, Any], parsed: Optional[Dict[str, str]] = None) -> Dict[str, Any] | None:
    """공통 LLM 호출 함수를 재사용해 SQL 해석/수정가이드를 보강한다.

    기존 프로젝트가 LLM_PROVIDER=upstage로 정상 동작한다면 llm.py의 ollama_generate가
    provider 설정을 보고 Upstage Solar로 라우팅한다. 여기서는 provider를 하드코딩하지 않고
    공통 ChatConfig + system_prompt + user prompt 규격만 맞춘다.
    """
    if not SQL_ANALYSIS_USE_LLM:
        return None

    parsed = parsed or {}
    system_prompt = """
너는 금융권 배치/SQL 코드리뷰 담당자다.
SQL을 보수적으로 분석하고, 운영 반영 가능 여부를 단정하지 않는다.
반드시 한국어로 답한다.
결과는 JSON 객체만 반환한다.
룰 기반 1차 분석 결과(findings, level, evidence, recommendation)는 deterministic 판단 근거다.
LLM은 룰 결과를 삭제하거나 위험도를 낮추지 말고, 사람이 이해하기 쉽게 설명/보강만 한다.
추가로 추정한 내용은 반드시 '추가 확인 필요' 관점으로 표현한다.
""".strip()

    user_prompt = f"""
아래 SQL 분석 요청서를 검토하라.
룰 기반 findings가 있으면 risks/change_guide/review_checklist에 반드시 반영하라.

출력 JSON 형식:
{{
  "summary": "한 문단 요약",
  "interpretation": ["SQL 처리 흐름 설명"],
  "table_roles": [{{"table":"테이블명", "role":"역할 설명"}}],
  "join_analysis": ["JOIN 구조 설명"],
  "risks": [{{"level":"LOW|MEDIUM|HIGH", "item":"검토항목", "detail":"설명"}}],
  "change_guide": ["수정내용이 있을 때 수정 가이드"],
  "improved_sql_example": "수정 예시 SQL. 확실하지 않으면 빈 문자열",
  "performance_points": ["성능 개선 포인트"],
  "review_checklist": ["운영 반영 전 확인사항"]
}}

요청 메타:
{json.dumps({k: v for k, v in parsed.items() if k != "sql"}, ensure_ascii=False, indent=2)}

수정내용:
{change_request or "(없음)"}

룰 기반 1차 분석:
{json.dumps(rule_report, ensure_ascii=False, indent=2)}

SQL:
{sql}
""".strip()

    try:
        config = _build_sql_analysis_chat_config()
        raw = ollama_generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            config=config,
        )

        if isinstance(raw, dict):
            return raw

        parsed_json = _extract_json_object(str(raw or ""))
        if parsed_json is not None:
            return parsed_json

        return {
            "summary": str(raw or "").strip(),
            "interpretation": [],
            "table_roles": [],
            "join_analysis": [],
            "risks": [],
            "change_guide": [],
            "improved_sql_example": "",
            "performance_points": [],
            "review_checklist": [],
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_sql_analysis_request(request_text: str) -> Any:
    parsed = parse_sql_analysis_request(request_text)
    sql = parsed.get("sql", "")
    change_request = parsed.get("change_request", "")
    rule_report = build_rule_based_sql_analysis(
        sql=sql,
        change_request=change_request,
        operation_info=parsed.get("operation_info", ""),
        review_points=parsed.get("review_points", ""),
    )
    llm_report = run_sql_analysis_llm(sql, change_request, rule_report, parsed) if rule_report.get("success") else None

    summary = rule_report.get("summary", "SQL 분석 요청을 처리했습니다.")
    if isinstance(llm_report, dict) and llm_report.get("summary"):
        summary = str(llm_report.get("summary"))

    return SimpleNamespace(
        answer=summary,
        intent="sql_analysis",
        render_type="sql_analysis",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=apply_dictionary_rewrite(request_text),
        rewritten_question=request_text,
        system_id=None,
        sources=[],
        debug_logs=[
            "[SQL_ANALYSIS 1] intent=sql_analysis",
            f"[SQL_ANALYSIS 2] statement_type={rule_report.get('statement_type')}",
            f"[SQL_ANALYSIS 3] has_change_request={bool(change_request.strip())}",
            f"[SQL_ANALYSIS 4] generated_by={'llm+rule' if llm_report and not llm_report.get('error') else 'rule'}",
        ],
        sql_analysis_result={
            "request_text": request_text,
            "parsed": parsed,
            "rule_report": rule_report,
            "llm_report": llm_report,
            "success": bool(rule_report.get("success")),
        },
    )


def render_sql_analysis_result(result: Any) -> None:
    payload = getattr(result, "sql_analysis_result", None) or {}
    parsed = payload.get("parsed", {}) or {}
    rule_report = payload.get("rule_report", {}) or {}
    llm_report = payload.get("llm_report") or {}

    st.markdown("#### 🔎 SQL 분석 결과")
    if payload.get("success"):
        st.success(rule_report.get("summary", "SQL 분석이 완료되었습니다."))
    else:
        st.error(rule_report.get("summary", "SQL 분석에 실패했습니다."))

    sql = parsed.get("sql", "")
    change_request = parsed.get("change_request", "")

    if sql:
        with st.expander("원본 SQL", expanded=False):
            st.code(sql, language="sql")

    if change_request:
        st.markdown("##### 📝 수정내용")
        st.write(change_request)

    if rule_report.get("tables"):
        st.markdown("##### 📌 주요 객체")
        st.markdown(", ".join([f"`{item}`" for item in rule_report.get("tables", [])]))

    if isinstance(llm_report, dict) and llm_report.get("summary") and not llm_report.get("error"):
        st.markdown("##### 🤖 LLM 해석")
        st.write(llm_report.get("summary"))
        for title, key in [
            ("처리 흐름", "interpretation"),
            ("테이블 역할", "table_roles"),
            ("JOIN 구조", "join_analysis"),
            ("수정 가이드", "change_guide"),
            ("성능 개선 포인트", "performance_points"),
            ("운영 반영 전 체크리스트", "review_checklist"),
        ]:
            items = llm_report.get(key) or []
            if items:
                st.markdown(f"**{title}**")
                for item in items[:8]:
                    if isinstance(item, dict):
                        st.markdown(f"- {json.dumps(item, ensure_ascii=False)}")
                    else:
                        st.markdown(f"- {item}")

        improved_sql = str(llm_report.get("improved_sql_example") or "").strip()
        if improved_sql:
            st.markdown("**개선 SQL 예시**")
            st.code(improved_sql, language="sql")

        risks = llm_report.get("risks") or []
        if risks:
            st.markdown("**위험요소**")
            for risk in risks[:8]:
                level = str(risk.get("level", "INFO"))
                item = str(risk.get("item", "검토항목"))
                detail = str(risk.get("detail", ""))
                st.markdown(f"- `{level}` **{item}**: {detail}")
    elif isinstance(llm_report, dict) and llm_report.get("error"):
        st.warning(f"LLM 보강 분석 실패로 룰 기반 분석만 표시합니다: {llm_report.get('error')}")

    findings = rule_report.get("findings") or []
    if findings:
        st.markdown("##### ⚠️ 룰 기반 검토 포인트")
        for item in findings:
            st.markdown(f"- `{item.get('level', 'INFO')}` **{item.get('item', '')}**: {item.get('detail', '')}")
            if item.get("evidence"):
                st.caption(f"근거: {item.get('evidence')}")
            if item.get("recommendation"):
                st.caption(f"권고: {item.get('recommendation')}")

    guides = rule_report.get("change_guide") or []
    if guides:
        st.markdown("##### ✅ 기본 수정/검증 가이드")
        for item in guides:
            st.markdown(f"- {item}")

    warnings = rule_report.get("warnings") or []
    for warning in warnings:
        st.caption(f"⚠️ {warning}")

    st.info("SQL 분석 결과는 코드리뷰 보조 자료입니다. 운영 반영 전 실제 DB 실행계획, 인덱스, 대상 건수, 권한, 트랜잭션 범위를 반드시 확인하세요.")


def render_sql_analysis_evaluation_panel(result: Any) -> None:
    payload = getattr(result, "sql_analysis_result", None) or {}
    with st.expander("📊 SQL 분석 평가용 근거 확인", expanded=False):
        st.markdown("##### 1) 요청 파싱 결과")
        st.json(payload.get("parsed", {}))
        st.markdown("##### 2) 룰 기반 분석 결과")
        st.json(payload.get("rule_report", {}))
        if payload.get("llm_report"):
            st.markdown("##### 3) LLM 보강 분석 결과")
            st.json(payload.get("llm_report", {}))
