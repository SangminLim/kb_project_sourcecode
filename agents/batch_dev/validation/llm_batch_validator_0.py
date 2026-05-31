from __future__ import annotations

"""
LLM 기반 배치 생성 결과 해석/검증 모듈.

핵심 원칙
- 배치 생성은 기존 spec_builder / rule_engine / template generator가 담당한다.
- 이 모듈은 생성된 결과물을 해석하고 검증 리포트를 만든다.
- 하드코딩된 특정 배치명/특정 테이블 기준으로 판단하지 않는다.
- 룰 기반 검증과 LLM 기반 의미 검증을 함께 사용한다.

사용 예시
    from llm_batch_validator import validate_batch_generation

    report = validate_batch_generation(
        request_text=request_text,
        batch_spec=batch_spec,
        generated_files={
            "query.sql": query_sql,
            "job.py": job_py,
            "test_job.py": test_job_py,
        },
        llm_client=my_llm_client,
        output_dir=Path("generated/my_batch"),
    )
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple


class LLMClient(Protocol):
    """
    프로젝트에서 사용하는 LLM 호출 객체의 최소 인터페이스.

    Upstage, OpenAI, Ollama, LangChain, 사내 LLM Gateway 등 무엇을 쓰든
    아래 invoke(prompt: str) -> str 형태만 맞춰주면 이 검증 모듈은 변경하지 않아도 된다.
    """

    def invoke(self, prompt: str) -> str:
        ...


@dataclass
class ValidationCheck:
    """개별 검증 항목 결과."""

    item: str
    result: str  # PASS / WARN / FAIL
    detail: str


@dataclass
class ValidationReport:
    """배치 생성 결과 검증/해석 리포트 표준 구조."""

    valid: bool
    score: float
    summary: str
    interpretation: str
    detected_batch_type: Optional[str] = None
    checks: List[ValidationCheck] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None
    score_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleValidationConfig:
    """
    룰 검증 설정.

    특정 업무명이나 특정 테이블을 하드코딩하지 않고,
    생성 결과 자체에서 일반적으로 확인 가능한 품질 기준만 둔다.
    """

    required_generated_files: Tuple[str, ...] = ("batch_spec.json", "query.sql", "job.py")
    dangerous_sql_patterns: Tuple[str, ...] = (
        r"\bDROP\s+TABLE\b",
        r"\bTRUNCATE\s+TABLE\b",
        r"\bDELETE\s+FROM\b(?![\s\S]*\bWHERE\b)",
        r"\bUPDATE\b[\s\S]{0,500}\bSET\b(?![\s\S]*\bWHERE\b)",
    )
    minimum_score_when_rule_fail: float = 0.3
    minimum_score_when_rule_warn: float = 0.7


VALIDATION_POLICY_VERSION = "practical-scoring-v3-nonblocking-warn"


DEFAULT_LLM_JSON_SCHEMA = {
    "valid": "boolean",
    "score": "number between 0 and 1",
    "summary": "생성 결과를 1문장으로 요약",
    "interpretation": "배치가 무엇을 하는지 핵심 처리 2~3개로 짧게 해석",
    "detected_batch_type": "string or null",
    "checks": [
        {
            "item": "검증 항목명",
            "result": "PASS|WARN|FAIL",
            "detail": "왜 그렇게 판단했는지 구체적 근거",
        }
    ],
    "issues": ["반드시 수정해야 하는 문제"],
    "warnings": ["운영 반영 전 확인해야 하는 위험"],
    "recommendations": ["구체적인 개선 권장사항"],
}


def validate_batch_generation(
    request_text: str,
    batch_spec: Mapping[str, Any],
    generated_files: Mapping[str, str],
    llm_client: Optional[LLMClient] = None,
    output_dir: Optional[Path] = None,
    config: Optional[RuleValidationConfig] = None,
) -> ValidationReport:
    """
    배치 생성 결과를 검증하고 리포트를 생성한다.

    처리 순서
    1. 룰 기반 검증을 먼저 수행한다.
       - 파일 존재 여부
       - SQL 위험 패턴
       - spec과 SQL 간 기본 일관성
    2. LLM Client가 있으면 의미 기반 검증을 수행한다.
       - 요청서 의도와 생성 결과 일치 여부
       - 누락 조건 여부
       - 사람이 이해하기 쉬운 해석 생성
    3. 두 결과를 병합한다.
    4. output_dir가 있으면 validation_report.json / validation_report.md를 저장한다.

    LLM이 없어도 룰 기반 리포트는 생성된다.
    """

    config = config or RuleValidationConfig()
    normalized_files = _normalize_generated_files(batch_spec, generated_files)

    rule_report = _run_rule_validation(
        request_text=request_text,
        batch_spec=batch_spec,
        generated_files=normalized_files,
        config=config,
    )

    effective_llm_client = llm_client or _build_default_llm_client_if_enabled()

    if effective_llm_client is None:
        final_report = rule_report
    else:
        try:
            llm_report = _run_llm_validation(
                request_text=request_text,
                batch_spec=batch_spec,
                generated_files=normalized_files,
                llm_client=effective_llm_client,
            )
            final_report = _merge_reports(rule_report, llm_report)
        except Exception as e:
            # LLM timeout/JSON parsing 오류가 나도 배치 생성 자체를 실패로 만들지 않는다.
            # 실무에서는 LLM 검증은 보조 검증이고, 최소한의 룰 검증 결과는 반드시 남긴다.
            final_report = _append_llm_failure_warning(rule_report, e)

    if output_dir is not None:
        write_validation_reports(final_report, output_dir)

    return final_report


class ProjectLLMClient:
    """배치 검증용 LLM Client.

    LLM_PROVIDER=upstage 인 경우 llm.py를 거치지 않고 Upstage Chat API를 직접 호출한다.
    이렇게 하면 llm.py의 Ollama 호환 함수/ChatConfig 차이 때문에 검증 LLM이 실패하는 문제를 줄일 수 있다.

    그 외 provider는 기존 프로젝트 llm.py의 generate_text / ollama_generate를 fallback으로 사용한다.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "upstage").strip().lower()
        self.model = os.getenv(
            "BATCH_VALIDATION_LLM_MODEL",
            os.getenv("UPSTAGE_CHAT_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "solar-pro3")),
        ).strip()
        self.timeout = int(
            os.getenv(
                "BATCH_VALIDATION_LLM_TIMEOUT",
                os.getenv("UPSTAGE_CHAT_TIMEOUT", "120"),
            )
        )
        self.temperature = float(os.getenv("UPSTAGE_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("UPSTAGE_MAX_TOKENS", "2048"))

        self.system_prompt = (
            "너는 금융권 계정계/정보계 배치 시스템을 검증하는 수석 배치 아키텍트다. "
            "생성된 batch_spec, SQL, job.py 실행 단서를 기준으로 배치 목적 적합성, 데이터 정합성, "
            "운영 안정성, 성능 위험, 재처리 위험, 파일 생성 위험을 검토한다. "
            "반드시 한국어 JSON 객체 하나만 반환한다. JSON 밖의 설명이나 Markdown은 금지한다. "
            "입력에 없는 테이블/컬럼/업무 규칙은 추측하지 않는다. "
            "단순 칭찬보다 실제 운영 반영 전 확인해야 할 위험과 개선점을 우선 제시한다."
        )

    def invoke(self, prompt: str) -> str:
        """프로젝트 공통 LLM client를 통해 검증 LLM을 호출한다.

        이전 구현은 provider=upstage일 때 requests.post로 Upstage API를 직접 호출했다.
        그 경로는 LangSmith의 upstage_chat_generate trace와 token usage 기록을 우회하므로,
        배치 검증 결과(raw_llm_response)는 생성되지만 LangSmith 토큰 사용량은 보이지 않았다.

        현재 구현은 업무인수인계와 동일하게 agents.handover.ollama_generate를 사용한다.
        따라서 llm_client.py의 _run_with_langsmith_trace / _attach_langsmith_usage_metadata 경로를 함께 타며,
        validate_generated_files 하위에 upstage_chat_generate와 token_usage가 남는다.
        """
        return self._invoke_project_llm(prompt)

    def _invoke_upstage(self, prompt: str) -> str:
        """하위 호환용 메서드. 직접 requests.post를 하지 않고 공통 LLM 경로로 위임한다."""
        return self._invoke_project_llm(prompt)

    def _build_project_chat_config(self) -> Any:
        """프로젝트 공통 ChatConfig를 버전 차이에 안전하게 생성한다."""
        try:
            from agents.handover import ChatConfig
        except Exception:
            try:
                from llm import ChatConfig  # legacy fallback
            except Exception:
                return None

        candidates = [
            {
                "model": self.model,
                "timeout": self.timeout,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            {
                "model": self.model,
                "timeout": self.timeout,
                "temperature": self.temperature,
            },
            {
                "model": self.model,
                "timeout": self.timeout,
            },
            {"model": self.model},
            {},
        ]

        for kwargs in candidates:
            try:
                return ChatConfig(**kwargs)
            except TypeError:
                continue
        return None

    def _invoke_project_llm(self, prompt: str) -> str:
        """프로젝트 공통 LLM 호출 함수를 사용한다.

        우선 agents.handover의 ollama_generate를 사용한다. 이 함수는 이름은 ollama_generate지만
        현재 프로젝트에서는 Upstage Chat API 호출과 LangSmith usage 기록을 담당한다.
        """
        config = self._build_project_chat_config()

        try:
            from agents.handover import ollama_generate

            return ollama_generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                config=config,
            )
        except Exception:
            # legacy fallback: 예전 구조에서 agents.handover export가 없을 때만 사용한다.
            try:
                from llm import generate_text

                return generate_text(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    config=config,
                )
            except ImportError:
                from llm import ollama_generate

                return ollama_generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    config=config,
                )


def _build_default_llm_client_if_enabled() -> Optional[LLMClient]:
    """환경변수 기준으로 기본 LLM Client를 만든다.

    BATCH_VALIDATION_USE_LLM=false 로 끄면 룰 검증만 수행한다.
    기본값은 true다.
    """
    use_llm = os.getenv("BATCH_VALIDATION_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "y"}
    if not use_llm:
        return None
    try:
        return ProjectLLMClient()
    except Exception as e:
        # Streamlit 화면은 죽이지 않되, 콘솔에는 원인을 남긴다.
        print(f"[WARN] 기본 LLM Client 생성 실패: {type(e).__name__}: {e}")
        return None

def write_validation_reports(report: ValidationReport, output_dir: Path) -> None:
    """
    검증 리포트를 JSON과 Markdown 파일로 저장한다.

    - validation_report.json: 시스템 연계/자동화용
    - validation_report.md: 개발자/운영자 확인용
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "validation_report.json"
    md_path = output_dir / "validation_report.md"

    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")


def _normalize_generated_files(
    batch_spec: Mapping[str, Any], generated_files: Mapping[str, str]
) -> Dict[str, str]:
    """
    생성 파일 목록을 표준화한다.

    호출부에서 batch_spec.json을 따로 넘기지 않아도 항상 검증 대상에 포함되도록 한다.
    """

    result = dict(generated_files)
    result.setdefault(
        "batch_spec.json",
        json.dumps(dict(batch_spec), ensure_ascii=False, indent=2),
    )
    return result



def _infer_batch_validation_profile(
    batch_spec: Mapping[str, Any],
    generated_files: Mapping[str, str],
) -> Dict[str, Any]:
    """생성 결과에서 검토항목 압축에 필요한 일반 특성을 추론한다.

    특정 배치명/테이블명을 보지 않고 batch_type, target, SQL 구조, 컬럼 role 단서만 사용한다.
    """
    compact = _build_compact_spec(batch_spec)
    query_sql = _extract_query_sql(batch_spec, generated_files)
    upper_sql = query_sql.upper()

    source = compact.get("source", {}) if isinstance(compact.get("source", {}), Mapping) else {}
    target = compact.get("target", {}) if isinstance(compact.get("target", {}), Mapping) else {}

    batch_type = str(compact.get("batch_type") or "").lower()
    output_format = str(target.get("output_format") or "").lower()
    source_table = str(source.get("table_name") or "").strip()
    output_pattern = str(target.get("output_file_pattern") or "").strip()

    has_aggregate = bool(re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", upper_sql))
    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", upper_sql))
    has_amount = bool(re.search(r"\b[A-Z0-9_]*(AMT|AMOUNT|PRICE|FEE|TOTAL)[A-Z0-9_]*\b", upper_sql))
    has_use_yn = bool(re.search(r"\bUSE_YN\b", upper_sql))
    has_effective_date = bool(re.search(r"APPLY_START|START_DT|VALID_START", upper_sql)) and bool(
        re.search(r"APPLY_END|END_DT|VALID_END", upper_sql)
    )
    has_base_date_param = bool(re.search(r":base_date\b", query_sql, flags=re.IGNORECASE))
    has_file_output = "file" in batch_type or output_format in {"csv", "txt", "xlsx"}
    has_join = bool(re.search(r"\bJOIN\b", upper_sql))
    has_insert_or_delete = bool(re.search(r"\b(INSERT\s+INTO|DELETE\s+FROM)\b", upper_sql))
    is_simple_export = bool(
        has_file_output
        and not has_aggregate
        and not has_group_by
        and not has_insert_or_delete
        and len(re.findall(r"\bJOIN\b", upper_sql)) <= 1
    )

    return {
        "batch_type": batch_type,
        "output_format": output_format,
        "source_table": source_table,
        "output_pattern": output_pattern,
        "query_sql": query_sql,
        "has_file_output": has_file_output,
        "has_aggregate": has_aggregate,
        "has_group_by": has_group_by,
        "has_amount": has_amount,
        "has_use_yn": has_use_yn,
        "has_effective_date": has_effective_date,
        "has_base_date_param": has_base_date_param,
        "has_join": has_join,
        "has_insert_or_delete": has_insert_or_delete,
        "is_simple_export": is_simple_export,
    }


def _profile_review_items(profile: Mapping[str, Any]) -> List[str]:
    """배치 특성에 맞는 운영 검토항목을 만든다.

    일반 규칙:
    - 파일 export면 파일명/포맷/파라미터 검증을 우선한다.
    - 유효기간/사용여부 조건이 있으면 해당 조건 인덱스를 검토한다.
    - 금액/집계 검증은 SQL에 금액/집계 단서가 있을 때만 제안한다.
    """
    items: List[str] = []

    if profile.get("has_use_yn") or profile.get("has_effective_date"):
        if profile.get("has_effective_date"):
            items.append("USE_YN / APPLY_START_DT 조건 인덱스 확인" if profile.get("has_use_yn") else "APPLY_START_DT / APPLY_END_DT 조건 인덱스 확인")
        elif profile.get("has_use_yn"):
            items.append("USE_YN 조건 인덱스 확인")

    if profile.get("has_file_output"):
        items.append("CSV 파일 중복 생성 방지 확인" if profile.get("output_format") == "csv" else "파일 중복 생성 방지 확인")
        items.append("출력 헤더/구분자 확인" if profile.get("output_format") == "csv" else "출력 파일 포맷 확인")

    if profile.get("has_base_date_param"):
        items.append("기준일자(base_date) 파라미터 검증 확인")

    if profile.get("has_aggregate") or profile.get("has_group_by"):
        items.append("집계 기준별 row count 및 중복 검증 확인")

    if profile.get("has_amount"):
        items.append("금액 합계 검증 확인")

    if not items:
        items.append("운영 반영 전 실행계획 및 기본 데이터 검증 확인")

    return _dedupe(items)[:4]


def _profile_recommendations(profile: Mapping[str, Any]) -> List[str]:
    """화면 권장사항용 짧은 문구를 만든다."""
    items = _profile_review_items(profile)
    # 파일 export에서 '검토 필요'와 중복되더라도 validation_report 안에서는 동일한 실무 포인트로 보이게 유지한다.
    return items[:3]


def _compact_validation_text(text: str) -> str:
    """LLM/룰 warning 문장을 화면용 짧은 검토문구로 정규화한다."""
    raw = str(text or "").strip()
    upper = raw.upper()

    if not raw:
        return ""

    if "금액" in raw and "합계" in raw:
        # 금액 단서가 없는 배치에서는 별도 필터에서 제거된다.
        return "금액 합계 검증 확인"

    if "중복" in raw and ("파일" in raw or "덮어쓰기" in raw or "멱등" in raw):
        return "CSV 파일 중복 생성 방지 확인"

    if "USE_YN" in upper or "APPLY_START" in upper or "APPLY_END" in upper or "인덱스" in raw:
        if "USE_YN" in upper and ("APPLY_START" in upper or "APPLY_END" in upper):
            return "USE_YN / APPLY_START_DT 조건 인덱스 확인"
        if "APPLY_START" in upper or "APPLY_END" in upper:
            return "APPLY_START_DT / APPLY_END_DT 조건 인덱스 확인"
        return "조건 컬럼 인덱스 확인"

    if "BASE_DATE" in upper or "기준일자" in raw:
        return "기준일자(base_date) 파라미터 검증 확인"

    if "OUTPUT_FORMAT" in upper or "헤더" in raw or "구분자" in raw or "ENCODING" in upper:
        return "출력 헤더/구분자 확인"

    if "ROW COUNT" in upper or "건수" in raw or "중복" in raw:
        return "row count 및 중복 검증 확인"

    if "테스트" in raw or "TEST" in upper:
        return "SQL/파일포맷/파라미터 테스트 보강"

    # 너무 긴 LLM 문장은 첫 문장만 사용한다.
    first_sentence = re.split(r"(?<=[.!?。])\s+|\n", raw)[0].strip()
    return first_sentence[:80]


def _filter_profile_items(items: List[str], profile: Mapping[str, Any]) -> List[str]:
    """배치 특성과 맞지 않는 과한 일반론을 제거한다."""
    filtered: List[str] = []
    for item in items or []:
        text = _compact_validation_text(item)
        if not text:
            continue

        # 금액/집계 없는 단순 마스터 export에 금액 합계 검증을 띄우지 않는다.
        if "금액" in text and not profile.get("has_amount"):
            continue
        if "집계" in text and not (profile.get("has_aggregate") or profile.get("has_group_by")):
            continue

        # 단순 export에서는 일반적인 '테스트 범위 확대'보다 파일/파라미터/인덱스 검토를 우선한다.
        if profile.get("is_simple_export") and "테스트" in text:
            continue

        filtered.append(text)

    base_items = _profile_review_items(profile)
    return _dedupe(base_items + filtered)[:4]


def _make_concise_summary(batch_spec: Mapping[str, Any], generated_files: Mapping[str, str], fallback_summary: str = "") -> str:
    """생성 배치 요약을 한 문장으로 압축한다."""
    profile = _infer_batch_validation_profile(batch_spec, generated_files)
    source_table = profile.get("source_table") or "소스 테이블"
    output_format = str(profile.get("output_format") or "파일").upper()
    query_sql = str(profile.get("query_sql") or "")
    upper_sql = query_sql.upper()

    conditions: List[str] = []
    if profile.get("has_use_yn"):
        conditions.append("사용가능(USE_YN='Y')")
    if profile.get("has_effective_date"):
        conditions.append("적용기간 유효")
    if profile.get("has_base_date_param"):
        conditions.append("기준일자 기준")

    if profile.get("has_file_output"):
        condition_text = " + ".join(conditions) if conditions else "조건에 맞는"
        return f"{source_table}에서 {condition_text} 데이터를 조회하여 {output_format} 파일로 생성하는 배치입니다."

    if profile.get("has_aggregate") or profile.get("has_group_by"):
        return f"{source_table} 기준 데이터를 집계하여 결과를 생성하는 배치입니다."

    if fallback_summary:
        return str(fallback_summary).strip()
    return f"{source_table} 기준 데이터를 처리하는 배치입니다."


def _make_concise_interpretation(batch_spec: Mapping[str, Any], generated_files: Mapping[str, str], fallback: str = "") -> str:
    """배치 해석을 화면에 바로 보일 짧은 bullet 텍스트로 만든다."""
    profile = _infer_batch_validation_profile(batch_spec, generated_files)
    source_table = profile.get("source_table") or "소스 테이블"
    lines: List[str] = []

    lines.append(f"- {source_table} 조회")

    if profile.get("has_base_date_param"):
        if profile.get("has_effective_date"):
            lines.append("- 기준일자(base_date) 기준 유효 데이터 필터링")
        else:
            lines.append("- 기준일자(base_date) 파라미터 기준 필터링")
    elif profile.get("has_use_yn") or profile.get("has_effective_date"):
        lines.append("- 사용여부/적용기간 조건 필터링")

    if profile.get("has_aggregate") or profile.get("has_group_by"):
        lines.append("- 집계 결과 생성")
    elif profile.get("has_file_output"):
        fmt = str(profile.get("output_format") or "파일").upper()
        lines.append(f"- {fmt} 파일 생성")

    if not lines and fallback:
        return str(fallback).strip()

    return "\n".join(_dedupe(lines)[:4])

def _run_rule_validation(
    request_text: str,
    batch_spec: Mapping[str, Any],
    generated_files: Mapping[str, str],
    config: RuleValidationConfig,
) -> ValidationReport:
    """
    LLM 없이도 수행 가능한 결정적 검증.

    이 함수는 업무별 하드코딩을 피하고, 공통적인 생성 품질만 검증한다.
    """

    checks: List[ValidationCheck] = []
    issues: List[str] = []
    warnings: List[str] = []

    for file_name in config.required_generated_files:
        if file_name in generated_files and str(generated_files[file_name]).strip():
            checks.append(ValidationCheck(file_name, "PASS", "생성 파일이 존재합니다."))
        else:
            checks.append(ValidationCheck(file_name, "FAIL", "필수 생성 파일이 없습니다."))
            issues.append(f"필수 생성 파일 누락: {file_name}")

    query_sql = generated_files.get("query.sql", "")
    if query_sql.strip():
        danger_found = False
        for pattern in config.dangerous_sql_patterns:
            if re.search(pattern, query_sql, flags=re.IGNORECASE | re.MULTILINE):
                danger_found = True
                checks.append(
                    ValidationCheck(
                        "SQL 위험 패턴",
                        "FAIL",
                        f"위험하거나 운영 반영 전 확인이 필요한 SQL 패턴 감지: {pattern}",
                    )
                )
                issues.append("SQL에 위험 패턴이 포함되어 있습니다.")
        if not danger_found:
            checks.append(ValidationCheck("SQL 위험 패턴", "PASS", "명백한 위험 SQL 패턴은 발견되지 않았습니다."))
    else:
        checks.append(ValidationCheck("SQL 존재 여부", "FAIL", "query.sql 내용이 비어 있습니다."))
        issues.append("query.sql 내용이 비어 있습니다.")


    sql_quality_checks, sql_quality_warnings = _run_sql_quality_validation(
        query_sql=query_sql,
        batch_spec=batch_spec,
    )
    checks.extend(sql_quality_checks)
    warnings.extend(sql_quality_warnings)

    semantic_checks, semantic_issues, semantic_warnings = _run_semantic_requirement_validation(
        request_text=request_text,
        query_sql=query_sql,
        batch_spec=batch_spec,
    )
    checks.extend(semantic_checks)
    issues.extend(semantic_issues)
    warnings.extend(semantic_warnings)

    output_checks, output_warnings = _run_output_validation(batch_spec)
    checks.extend(output_checks)
    warnings.extend(output_warnings)

    table_candidates = _extract_table_candidates(batch_spec)
    if table_candidates and query_sql.strip():
        missing_tables = [table for table in table_candidates if table.upper() not in query_sql.upper()]
        if missing_tables:
            checks.append(
                ValidationCheck(
                    "spec 테이블과 SQL 일치성",
                    "WARN",
                    f"batch_spec의 테이블 후보가 SQL에서 확인되지 않습니다: {', '.join(missing_tables)}",
                )
            )
            warnings.append(f"batch_spec 테이블 후보가 SQL에 없을 수 있습니다: {', '.join(missing_tables)}")
        else:
            checks.append(ValidationCheck("spec 테이블과 SQL 일치성", "PASS", "batch_spec의 테이블 후보가 SQL에서 확인됩니다."))

    if not request_text.strip():
        checks.append(ValidationCheck("요청서 존재 여부", "WARN", "원본 요청서가 비어 있습니다."))
        warnings.append("원본 요청서가 없어 요청 대비 검증 정확도가 낮습니다.")

    valid = not issues
    score = _calculate_rule_score(checks, config)
    profile = _infer_batch_validation_profile(batch_spec, generated_files)

    static_interpretation = _build_static_interpretation(batch_spec, generated_files)
    rule_warnings = _filter_profile_items(warnings, profile)

    return ValidationReport(
        valid=valid,
        score=score,
        summary="기본 룰 검증을 완료했습니다." if valid else "기본 룰 검증에서 오류가 발견되었습니다.",
        interpretation=static_interpretation,
        detected_batch_type=_safe_get_str(batch_spec, "batch_type") or _safe_get_str(batch_spec, "type"),
        checks=checks,
        issues=issues,
        warnings=rule_warnings,
        recommendations=_profile_recommendations(profile),
        score_breakdown={
            "query_sql": query_sql,
            "batch_type": _safe_get_str(batch_spec, "batch_type") or _safe_get_str(batch_spec, "type"),
            "target": dict(batch_spec.get("target", {})) if isinstance(batch_spec.get("target", {}), Mapping) else {},
            "validation_rules": dict(batch_spec.get("validation_rules", {})) if isinstance(batch_spec.get("validation_rules", {}), Mapping) else {},
            "validation_profile": dict(profile),
        },
    )



def _run_sql_quality_validation(
    query_sql: str,
    batch_spec: Mapping[str, Any],
) -> Tuple[List[ValidationCheck], List[str]]:
    """SQL 품질을 일반 규칙으로 점검한다.

    특정 업무/테이블에 종속되지 않고 모든 생성 SQL에 공통 적용 가능한 항목만 본다.
    """
    checks: List[ValidationCheck] = []
    warnings: List[str] = []
    sql = str(query_sql or "").strip()
    upper_sql = sql.upper()

    if not sql:
        return checks, warnings

    if re.search(r"\bSELECT\s+\*", sql, flags=re.IGNORECASE):
        checks.append(
            ValidationCheck(
                "SELECT * 사용",
                "WARN",
                "운영 배치에서는 필요한 컬럼을 명시하는 것이 안전합니다.",
            )
        )
        warnings.append("SELECT * 사용으로 컬럼 변경 시 파일 포맷 또는 후속 처리 영향 가능성이 있습니다.")

    if re.search(r"\bFROM\b", upper_sql) and not re.search(r"\bWHERE\b", upper_sql):
        checks.append(
            ValidationCheck(
                "WHERE 조건",
                "WARN",
                "조회 SQL에 WHERE 조건이 없어 대량 데이터 Full Scan 가능성이 있습니다.",
            )
        )
        warnings.append("WHERE 조건 부재로 대량 데이터 조회 위험이 있습니다.")

    parameter_names = sorted(set(re.findall(r":([A-Za-z_][A-Za-z0-9_]*)", sql)))
    spec_parameters = {
        str(item.get("name", "")).strip()
        for item in batch_spec.get("parameters", []) or []
        if isinstance(item, Mapping)
    }
    missing_in_spec = [name for name in parameter_names if name and name not in spec_parameters]
    if missing_in_spec:
        checks.append(
            ValidationCheck(
                "SQL 파라미터와 batch_spec 일치성",
                "WARN",
                f"SQL 파라미터가 batch_spec.parameters에 명확히 정의되어 있지 않습니다: {', '.join(missing_in_spec)}",
            )
        )
        warnings.append(f"SQL 파라미터 정의 확인 필요: {', '.join(missing_in_spec)}")
    elif parameter_names:
        checks.append(
            ValidationCheck(
                "SQL 파라미터와 batch_spec 일치성",
                "PASS",
                f"SQL 파라미터가 batch_spec.parameters와 연결됩니다: {', '.join(parameter_names)}",
            )
        )

    where_part = ""
    where_match = re.search(r"\bWHERE\b([\s\S]+)", sql, flags=re.IGNORECASE)
    if where_match:
        where_part = where_match.group(1)

    if where_part and not re.search(r"\b(Y|N)\b|USE_YN|DEL_YN|CANCEL_YN|STATUS|APPLY|BASE_|DT|DATE", where_part, flags=re.IGNORECASE):
        checks.append(
            ValidationCheck(
                "업무 필터 조건",
                "WARN",
                "WHERE 절은 있으나 상태값/사용여부/기준일자/기간 조건으로 보이는 필터가 약합니다.",
            )
        )
        warnings.append("업무 필터 조건이 충분한지 확인이 필요합니다.")

    if re.search(r"\bJOIN\b", upper_sql) and not re.search(r"\bON\b|\bUSING\b", upper_sql):
        checks.append(
            ValidationCheck(
                "JOIN 조건",
                "FAIL",
                "JOIN 문이 있으나 ON/USING 조건을 찾지 못했습니다.",
            )
        )

    return checks, warnings



def _infer_request_capabilities_for_validation(request_text: str) -> set[str]:
    """요청서에서 일반 처리 능력을 추론한다.

    특정 배치명/테이블명이 아니라 출력 컬럼, 처리 동사, 기준 파라미터 같은
    구조적 단서만 사용한다.
    """
    text = str(request_text or "")
    upper = text.upper()
    compact = re.sub(r"\s+", "", text).lower()

    capabilities: set[str] = set()

    if any(token in upper for token in ["TOTAL_", "SUM(", "COUNT(", "TXN_COUNT"]) or any(token in compact for token in ["집계", "합계", "건수"]):
        capabilities.add("aggregation")

    if any(token in upper for token in ["GROUP BY", "CUSTOMER_ID", "BASE_YM", "MERCHANT_TYPE"]) or any(token in compact for token in ["고객별", "월별", "유형별", "기준으로"]):
        capabilities.add("group_by")

    if "MERCHANT_TYPE" in upper or any(token in compact for token in ["가맹점유형", "분류마스터", "참조"]):
        capabilities.add("classification_join")

    if any(token in upper for token in ["APPLY_START_DT", "APPLY_END_DT", "BETWEEN"]) or any(token in compact for token in ["유효기간", "적용기간", "적용시작", "적용종료"]):
        capabilities.add("effective_date_matching")

    if "CANCEL_YN" in upper or any(token in compact for token in ["취소여부", "취소거래", "취소"]):
        capabilities.add("exclude_cancelled")

    if "USE_YN" in upper or "사용여부" in compact:
        capabilities.add("use_flag_filter")

    if "BASE_YM" in upper or "기준년월" in compact:
        capabilities.add("base_month_parameter")

    if any(token in compact for token in ["deleteinsert", "delete-insert", "delete후insert", "삭제후삽입"]):
        capabilities.add("replace_partition")

    return capabilities


def _run_semantic_requirement_validation(
    request_text: str,
    query_sql: str,
    batch_spec: Mapping[str, Any],
) -> Tuple[List[ValidationCheck], List[str], List[str]]:
    """요청서가 요구한 처리 능력과 생성 결과의 의미 일치성을 결정적으로 검증한다."""
    checks: List[ValidationCheck] = []
    issues: List[str] = []
    warnings: List[str] = []

    capabilities = _infer_request_capabilities_for_validation(request_text)
    sql = str(query_sql or "")
    upper_sql = sql.upper()
    batch_type = str(batch_spec.get("batch_type") or batch_spec.get("type") or "").lower()

    if "aggregation" in capabilities:
        has_aggregate = bool(re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", upper_sql))
        has_group_by = bool(re.search(r"\bGROUP\s+BY\b", upper_sql))
        if batch_type != "aggregation_to_table":
            checks.append(
                ValidationCheck(
                    "요청 목적 적합성",
                    "FAIL",
                    "요청서는 집계 처리를 요구하지만 batch_type이 aggregation_to_table이 아닙니다.",
                )
            )
            issues.append("집계 요청이 단순 추출 배치로 생성되었습니다.")
        if not has_aggregate or not has_group_by:
            checks.append(
                ValidationCheck(
                    "집계 SQL 의미 일치성",
                    "FAIL",
                    "요청서는 집계 결과를 요구하지만 SQL에 집계 함수 또는 GROUP BY가 부족합니다.",
                )
            )
            issues.append("집계 요청에 필요한 SUM/COUNT/GROUP BY SQL이 누락되었습니다.")
        if has_aggregate and has_group_by and batch_type == "aggregation_to_table":
            checks.append(
                ValidationCheck(
                    "집계 SQL 의미 일치성",
                    "PASS",
                    "집계 함수와 GROUP BY가 SQL에 포함되어 있습니다.",
                )
            )

    if "classification_join" in capabilities:
        has_join = bool(re.search(r"\bJOIN\b", upper_sql))
        if not has_join:
            checks.append(
                ValidationCheck(
                    "분류 마스터 조인",
                    "FAIL",
                    "요청서는 분류/유형 판단을 요구하지만 SQL에 JOIN이 없습니다.",
                )
            )
            issues.append("가맹점 유형 판단을 위한 분류 마스터 JOIN이 누락되었습니다.")

    if "effective_date_matching" in capabilities:
        has_effective = bool(
            re.search(r"APPLY_START|START_DT|VALID_START", upper_sql)
            and re.search(r"APPLY_END|END_DT|VALID_END", upper_sql)
            and (re.search(r"\bBETWEEN\b", upper_sql) or re.search(r"<=|>=", upper_sql))
        )
        if not has_effective:
            checks.append(
                ValidationCheck(
                    "유효기간 매칭",
                    "FAIL",
                    "요청서는 시작/종료일 기준 유효기간 매칭을 요구하지만 SQL에서 기간 조건을 확인하지 못했습니다.",
                )
            )
            issues.append("유효기간 매칭 조건이 SQL에 반영되지 않았습니다.")

    if "exclude_cancelled" in capabilities:
        has_cancel_filter = bool(re.search(r"CANCEL_YN\s*=\s*['\"]?N['\"]?|CNCL_YN\s*=\s*['\"]?N['\"]?", upper_sql))
        if not has_cancel_filter:
            checks.append(
                ValidationCheck(
                    "취소 거래 제외",
                    "FAIL",
                    "요청서는 취소 거래 제외를 요구하지만 SQL에서 취소여부=N 조건을 확인하지 못했습니다.",
                )
            )
            issues.append("취소 거래 제외 조건이 SQL에 반영되지 않았습니다.")

    if "use_flag_filter" in capabilities:
        has_use_filter = bool(re.search(r"USE_YN\s*=\s*['\"]?Y['\"]?|VALID_YN\s*=\s*['\"]?Y['\"]?", upper_sql))
        if not has_use_filter:
            checks.append(
                ValidationCheck(
                    "사용여부 필터",
                    "WARN",
                    "요청서는 사용여부=Y 조건을 요구하지만 SQL에서 명확한 사용여부 필터를 확인하지 못했습니다.",
                )
            )
            warnings.append("사용여부=Y 조건 반영 여부 확인 필요")

    if "replace_partition" in capabilities:
        execution_strategy = batch_spec.get("execution_strategy") or {}
        target = batch_spec.get("target") or {}
        has_strategy = bool(execution_strategy) or bool(isinstance(target, Mapping) and target.get("execution_strategy"))
        if not has_strategy:
            checks.append(
                ValidationCheck(
                    "재처리 전략",
                    "WARN",
                    "요청서는 Delete Insert 계열 재처리를 요구하지만 batch_spec에서 execution_strategy가 명확하지 않습니다.",
                )
            )
            warnings.append("Delete Insert 재처리 전략 확인 필요")

    return checks, issues, warnings



def _run_output_validation(batch_spec: Mapping[str, Any]) -> Tuple[List[ValidationCheck], List[str]]:
    """파일/테이블 출력 설정을 일반 규칙으로 점검한다."""
    checks: List[ValidationCheck] = []
    warnings: List[str] = []

    target = batch_spec.get("target", {})
    if not isinstance(target, Mapping):
        return checks, warnings

    batch_type = str(batch_spec.get("batch_type") or batch_spec.get("type") or "").lower()
    output_format = str(target.get("output_format") or "").strip().lower()
    output_pattern = str(target.get("output_file_pattern") or "").strip()
    output_dir = str(target.get("output_dir") or "").strip()
    encoding = str(target.get("encoding") or "").strip()

    if "file" in batch_type or output_format:
        if not output_format:
            checks.append(ValidationCheck("출력 형식", "WARN", "파일 생성 배치로 보이나 output_format이 없습니다."))
            warnings.append("output_format 설정 확인 필요")
        else:
            checks.append(ValidationCheck("출력 형식", "PASS", f"출력 형식이 정의되어 있습니다: {output_format}"))

        if not output_pattern:
            checks.append(ValidationCheck("출력 파일명 패턴", "WARN", "output_file_pattern이 없어 파일명 관리 기준을 확인해야 합니다."))
            warnings.append("output_file_pattern 설정 확인 필요")
        elif not re.search(r"\{[^}]+\}", output_pattern):
            checks.append(
                ValidationCheck(
                    "출력 파일명 패턴",
                    "WARN",
                    "파일명 패턴에 기준일자 등 치환자가 없어 재실행 시 덮어쓰기 위험이 있습니다.",
                )
            )
            warnings.append("파일명 패턴에 기준일자 치환자 추가 검토 필요")
        else:
            checks.append(ValidationCheck("출력 파일명 패턴", "PASS", f"파일명 패턴이 정의되어 있습니다: {output_pattern}"))

        if not output_dir:
            checks.append(ValidationCheck("출력 경로", "WARN", "output_dir이 없어 운영 파일 생성 경로 확인이 필요합니다."))
            warnings.append("output_dir 설정 확인 필요")

        if output_format in {"csv", "txt"} and not encoding:
            checks.append(ValidationCheck("파일 인코딩", "WARN", "텍스트 파일 출력인데 encoding이 정의되어 있지 않습니다."))
            warnings.append("파일 encoding 설정 확인 필요")

    return checks, warnings


def _build_static_interpretation(batch_spec: Mapping[str, Any], generated_files: Mapping[str, str]) -> str:
    """LLM이 없거나 실패해도 의미 있는 기본 해석을 제공한다."""
    compact = _build_compact_spec(batch_spec)
    batch_name = compact.get("batch_name") or compact.get("batch_id") or "생성 배치"
    batch_type = compact.get("batch_type") or "확인 필요"
    source_table = (
        compact.get("resolved_tables", {}).get("base")
        if isinstance(compact.get("resolved_tables"), Mapping)
        else None
    ) or compact.get("source", {}).get("table_name")
    output_format = compact.get("target", {}).get("output_format")
    output_pattern = compact.get("target", {}).get("output_file_pattern")
    params = [
        str(item.get("name"))
        for item in compact.get("parameters", []) or []
        if isinstance(item, Mapping) and item.get("name")
    ]
    query_sql = _extract_query_sql(batch_spec, generated_files)
    sql_params = sorted(set(re.findall(r":([A-Za-z_][A-Za-z0-9_]*)", query_sql or "")))

    sentences = [
        f"{batch_name}은 batch_type={batch_type} 유형으로 생성되었습니다.",
    ]
    if source_table:
        sentences.append(f"주요 기준 테이블은 {source_table}로 해석됩니다.")
    if output_format or output_pattern:
        sentences.append(f"출력은 {output_format or '형식 미정'} 파일이며 파일명 패턴은 {output_pattern or '미정'}입니다.")
    if params or sql_params:
        sentences.append(f"배치 파라미터는 {', '.join(params or sql_params)} 기준으로 사용됩니다.")
    if "WHERE" in (query_sql or "").upper():
        sentences.append("SQL에는 조건절이 포함되어 있어 무조건 전체 추출보다는 기준 조건 기반 처리로 보입니다.")
    sentences.append("운영 반영 전에는 실제 컬럼 존재 여부, 인덱스, 파일 경로 권한, 재실행 시 중복/덮어쓰기 정책을 확인해야 합니다.")
    return " ".join(sentences)


def _run_llm_validation(
    request_text: str,
    batch_spec: Mapping[str, Any],
    generated_files: Mapping[str, str],
    llm_client: LLMClient,
) -> ValidationReport:
    """LLM을 이용해 요청서와 생성 결과의 의미 일치성을 검증한다."""

    prompt = _build_validation_prompt(request_text, batch_spec, generated_files)
    raw = llm_client.invoke(prompt)
    parsed = _parse_llm_json(raw)

    checks = [
        ValidationCheck(
            item=str(item.get("item", "LLM 검증")),
            result=str(item.get("result", "WARN")).upper(),
            detail=str(item.get("detail", "")),
        )
        for item in parsed.get("checks", [])
        if isinstance(item, dict)
    ]

    profile = _infer_batch_validation_profile(batch_spec, generated_files)
    llm_warnings = _filter_profile_items(_safe_str_list(parsed.get("warnings")), profile)
    llm_recommendations = _filter_profile_items(_safe_str_list(parsed.get("recommendations")), profile)

    return ValidationReport(
        valid=bool(parsed.get("valid", False)),
        score=_safe_score(parsed.get("score", 0.0)),
        summary=_make_concise_summary(batch_spec, generated_files, str(parsed.get("summary", ""))),
        interpretation=_make_concise_interpretation(batch_spec, generated_files, str(parsed.get("interpretation", ""))),
        detected_batch_type=parsed.get("detected_batch_type"),
        checks=checks,
        issues=_safe_str_list(parsed.get("issues")),
        warnings=llm_warnings,
        recommendations=llm_recommendations,
        raw_llm_response=raw,
    )


def _build_validation_prompt(
    request_text: str,
    batch_spec: Mapping[str, Any],
    generated_files: Mapping[str, str],
) -> str:
    """
    LLM 검증 프롬프트를 생성한다.

    특정 업무/테이블을 하드코딩하지 않고, batch_spec/SQL/job.py 단서만으로
    생성 배치의 목적 적합성, SQL 의미, 운영 위험, 보완점을 평가하게 한다.
    """

    compact_spec = _build_compact_spec(batch_spec)
    query_sql = _extract_query_sql(batch_spec, generated_files)
    job_py = generated_files.get("job.py", "")
    readme = generated_files.get("README.md", "")
    test_job = generated_files.get("test_job.py", "")

    schema = json.dumps(DEFAULT_LLM_JSON_SCHEMA, ensure_ascii=False, indent=2)

    return f"""
너는 금융권 배치 개발 산출물을 검증하는 수석 배치 아키텍트다.
아래 입력만 기준으로 생성 결과를 해석하고 검증해라.

절대 규칙:
- 반드시 JSON 객체 하나만 출력한다.
- JSON 밖에 설명, Markdown, 코드블록을 쓰지 않는다.
- 입력에 없는 테이블/컬럼/업무 규칙을 추측하지 않는다.
- 특정 업무명/테이블명에 하드코딩된 판단을 하지 않는다.
- PASS만 남발하지 말고, 운영 반영 전 실제로 확인해야 할 위험을 WARN으로 분리한다.
- FAIL은 필수 파일 누락, query.sql 없음, 위험 SQL, JOIN ON/USING 누락, SQL 구문 오류, 필수 파라미터 불일치처럼 실행 실패 가능성이 직접적인 경우에만 사용한다.
- 테스트 부족, 인덱스 확인 필요, 데이터 품질 검증 부족, 집계 검증 부족, 운영 재처리 검토, 중복 가능성 검토는 FAIL이 아니라 WARN으로 분류한다.
- summary는 1문장으로 짧게 쓴다.
- interpretation은 최대 3개 핵심 처리만 담는다. 장문 설명은 금지한다.
- warnings/recommendations는 최대 3개만 작성한다.
- 금액/집계 컬럼이 없는 단순 파일 export에는 금액 합계 검증을 쓰지 않는다.
- 단순 파일 export에서는 인덱스, 파일 중복 생성, 기준 파라미터, 파일 포맷 확인 위주로 작성한다.

출력 JSON 스키마:
{schema}

원본 요청:
{_clip_text(request_text, 1500)}

batch_spec 핵심 정보:
{json.dumps(compact_spec, ensure_ascii=False, indent=2)}

생성 SQL:
{_clip_text(query_sql, 2500)}

job.py 실행 단서:
{_summarize_job_py(job_py)}

README 단서:
{_clip_text(readme, 700)}

test_job.py 단서:
{_clip_text(test_job, 700)}

검증 관점:
1. 원본 요청의 목적과 batch_type이 일치하는가?
2. batch_spec의 source/target/parameters와 SQL이 의미적으로 연결되는가?
3. SQL의 FROM/JOIN/WHERE/파라미터 조건이 배치 목적에 맞는가?
4. 기준일자/기간 조건이 있는 경우 경계 조건이 자연스러운가?
5. 파일 생성 배치라면 output_format, output_file_pattern, output_dir, encoding이 운영 관점에서 적절한가?
6. job.py가 DB 조회, 파라미터 처리, 파일 출력 또는 테이블 적재를 수행할 단서를 갖는가?
7. 테스트 파일이 생성 산출물 존재 여부만 보는지, SQL/파일포맷/파라미터까지 검증하는지 판단하라.
8. 성능 위험: Full Scan, 인덱스 필요 컬럼, 대량 데이터 조회 가능성을 검토하라.
9. 데이터 품질 위험: row count, 중복, 기준일자 검증 필요 여부를 검토하라. 금액 컬럼이 없으면 금액 합계 검증은 제외하라.
10. 재처리 위험: 파일 덮어쓰기, 중복 적재, 삭제 후 적재 여부, 멱등성 여부를 검토하라.

checks 작성 가이드:
- 4~6개만 작성한다.
- 항목 예시: 요청 목적 적합성, SQL 의미 일치성, 파라미터 일치성, 파일 출력 설정, 운영 재처리 위험, 성능 위험
- detail에는 입력에서 확인한 근거를 짧게 포함한다.

score 기준:
- 0.90 이상: 단순하고 위험이 경미하며 테스트/재처리/성능 검토사항이 거의 없음
- 0.80~0.89: 기본적으로 사용 가능하지만 운영 확인/WARN 필요
- 0.70~0.79: 실행 가능하지만 JOIN/GROUP BY/INSERT/DELETE 등으로 운영 검증 부담이 큼
- 0.50~0.69: 실행 가능성은 있으나 운영 반영 전 구조적 보완이 필요함
- 0.50 미만: 생성 결과를 그대로 쓰기 어려움

실무 해석:
- valid=true이고 FAIL이 없으면 일반적으로 0.80 이상을 우선 고려한다.
- 0.70대는 실행은 가능하지만 보완 부담이 큰 경우에만 사용한다.

중요:
- 단순 파일 export와 다중 JOIN/집계/적재 배치에 같은 점수를 주지 마라.
- 단, WARN은 실패가 아니다. SQL 구조가 정상이고 FAIL이 없다면 0.70 미만으로 과도하게 낮추지 마라.
- JOIN, GROUP BY, SUM/COUNT, CASE, INSERT, DELETE/INSERT 재처리, 테스트 부족, 데이터 품질 검증 부족이 있으면 점수를 합리적으로 낮춰라.
""".strip()


def _build_compact_spec(batch_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """LLM에 전달할 batch_spec 핵심 필드만 구성한다."""
    source = batch_spec.get("source", {}) if isinstance(batch_spec.get("source", {}), Mapping) else {}
    target = batch_spec.get("target", {}) if isinstance(batch_spec.get("target", {}), Mapping) else {}
    meta_source = batch_spec.get("meta_source", {}) if isinstance(batch_spec.get("meta_source", {}), Mapping) else {}
    rule_source = batch_spec.get("rule_source", {}) if isinstance(batch_spec.get("rule_source", {}), Mapping) else {}

    return {
        "batch_id": batch_spec.get("batch_id"),
        "batch_name": batch_spec.get("batch_name"),
        "batch_type": batch_spec.get("batch_type") or batch_spec.get("type"),
        "description": batch_spec.get("description"),
        "schedule_type": batch_spec.get("schedule_type"),
        "parameters": batch_spec.get("parameters", []),
        "source": {
            "table_name": source.get("table_name") or source.get("table") or source.get("base_table"),
            "table_role": source.get("table_role"),
            "base_date_column_role": source.get("base_date_column_role"),
            "dynamic_inference": source.get("dynamic_inference"),
        },
        "target": {
            "table_name": target.get("table_name") or target.get("table") or target.get("target_table"),
            "output_format": target.get("output_format"),
            "output_file_pattern": target.get("output_file_pattern"),
            "output_dir": target.get("output_dir"),
            "encoding": target.get("encoding"),
        },
        "resolved_tables": meta_source.get("resolved_tables", {}),
        "resolved_columns": meta_source.get("resolved_columns", {}),
        "rule_id": rule_source.get("rule_id"),
        "sql_template": rule_source.get("sql_template"),
        "template_type": rule_source.get("template_type"),
        "validation_rules": batch_spec.get("validation_rules", {}),
        "sql_present": bool(str(batch_spec.get("sql", "") or "").strip()),
    }


def _extract_query_sql(batch_spec: Mapping[str, Any], generated_files: Mapping[str, str]) -> str:
    """query.sql 파일 내용을 우선 사용하고, 없으면 batch_spec.sql을 사용한다."""
    query_sql = str(generated_files.get("query.sql", "") or "").strip()
    if query_sql:
        return query_sql
    return str(batch_spec.get("sql", "") or "").strip()


def _summarize_job_py(job_py: str) -> str:
    """job.py 전체 대신 검증에 필요한 단서만 짧게 요약한다."""
    text = str(job_py or "")
    if not text.strip():
        return "job.py 내용 없음"

    signals: List[str] = []
    patterns = {
        "CSV 출력 로직": r"\.to_csv\(|csv\.",
        "파일 open/write 로직": r"open\(|\.write\(",
        "DB 접속/SQL 실행 로직": r"execute\(|read_sql|create_engine|connect\(",
        "파라미터 처리": r"argparse|base_date|params?\s*=",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            signals.append(label)

    if not signals:
        return "주요 실행 단서 없음"
    return ", ".join(signals)


def _append_llm_failure_warning(rule_report: ValidationReport, error: Exception) -> ValidationReport:
    """LLM 검증 실패 시 룰 검증 결과를 유지하면서 경고만 추가한다."""
    error_message = f"LLM 검증 실패: {type(error).__name__}: {error}"
    return ValidationReport(
        valid=rule_report.valid,
        score=rule_report.score,
        summary=rule_report.summary,
        interpretation=(
            (rule_report.interpretation or "")
            + " LLM 의미 검증은 호출 오류로 생략되었으므로, 위 해석은 룰 기반 정적 해석입니다."
        ).strip(),
        detected_batch_type=rule_report.detected_batch_type,
        checks=rule_report.checks + [
            ValidationCheck("LLM 의미 검증", "WARN", error_message)
        ],
        issues=rule_report.issues,
        warnings=_dedupe(rule_report.warnings + [error_message]),
        recommendations=_dedupe(
            rule_report.recommendations
            + [
                "LLM 검증이 반복해서 느리면 BATCH_VALIDATION_USE_LLM=false로 끄고 룰 검증만 사용하세요.",
                "BATCH_VALIDATION_LLM_TIMEOUT, UPSTAGE_CHAT_MODEL, UPSTAGE_BASE_URL, UPSTAGE_MAX_TOKENS 설정을 확인하세요.",
            ]
        ),
        raw_llm_response=None,
    )


def _normalize_text_for_policy(value: Any) -> str:
    """정책 분류용 텍스트 정규화."""
    return str(value or "").upper()


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(keyword.upper() in text for keyword in keywords)


def _classify_check_category(check: ValidationCheck) -> str:
    """검증 항목을 실무 카테고리로 분류한다.

    특정 배치명/테이블명에 의존하지 않고, 항목명과 상세 메시지의 일반 품질 용어로 분류한다.
    """

    text = _normalize_text_for_policy(f"{check.item} {check.detail}")

    if _has_any(text, ["필수 생성 파일", "파일이 없습니다", "QUERY.SQL 내용이 비어", "SQL 존재 여부"]):
        return "artifact_missing"

    if _has_any(text, ["위험 SQL", "DROP TABLE", "TRUNCATE TABLE", "DELETE FROM(?!", "UPDATE WITHOUT WHERE"]):
        return "dangerous_sql"

    if _has_any(text, ["JOIN 문이 있으나 ON/USING", "JOIN 조건 누락", "ON/USING 조건을 찾지 못"]):
        return "join_condition_missing"

    if _has_any(text, ["파라미터 불일치", "PARAMETER MISMATCH", "필수 파라미터 누락"]):
        return "parameter_blocker"

    if _has_any(text, ["구문 오류", "SYNTAX", "실행 불가", "컴파일 오류"]):
        return "execution_blocker"

    if _has_any(text, ["요청 목적", "목적 적합", "SQL 의미", "의미 일치"]):
        return "semantic"

    if _has_any(text, ["테스트", "TEST", "검증 범위"]):
        return "test_coverage"

    if _has_any(text, ["성능", "FULL SCAN", "인덱스", "실행계획"]):
        return "performance_review"

    if _has_any(text, ["데이터 품질", "품질 검증", "NOT NULL", "ROW COUNT", "중복", "집계", "금액", "건수", "정합성"]):
        return "data_quality_review"

    if _has_any(text, ["재처리", "멱등", "덮어쓰기", "중복 적재", "DELETE 후 INSERT"]):
        return "reprocess_review"

    if _has_any(text, ["파일 출력", "출력 형식", "출력 파일", "OUTPUT", "ENCODING", "OUTPUT_DIR"]):
        return "output_review"

    return "general_review"


def _is_blocking_fail_check(check: ValidationCheck) -> bool:
    """FAIL이 최종 실패를 만들 정도의 blocking 오류인지 판단한다.

    blocking FAIL:
    - 필수 산출물 누락
    - query.sql 없음
    - 위험 SQL
    - JOIN ON/USING 누락
    - 필수 파라미터 불일치
    - SQL 구문/실행 불가

    non-blocking WARN:
    - 테스트 보강 필요
    - 성능/인덱스 확인 필요
    - 데이터 품질 검증 추가 필요
    - 재처리/멱등성 검토 필요
    - 중복 가능성 검토
    """

    if check.result.upper() != "FAIL":
        return False

    category = _classify_check_category(check)
    return category in {
        "artifact_missing",
        "dangerous_sql",
        "join_condition_missing",
        "parameter_blocker",
        "execution_blocker",
        "semantic",
    }


def _normalize_checks_for_operation(checks: List[ValidationCheck]) -> Tuple[List[ValidationCheck], List[str]]:
    """보완성 FAIL을 WARN으로 낮춘다."""

    normalized: List[ValidationCheck] = []
    downgraded: List[str] = []

    for check in checks:
        if check.result.upper() == "FAIL" and not _is_blocking_fail_check(check):
            normalized.append(
                ValidationCheck(
                    item=check.item,
                    result="WARN",
                    detail=f"{check.detail} (운영 보완 항목이므로 WARN 처리)",
                )
            )
            downgraded.append(f"{check.item}: {check.detail}")
        else:
            normalized.append(check)

    return normalized, downgraded


def _issue_is_blocking(issue: str) -> bool:
    """LLM issues 항목이 blocking인지 일반 규칙으로 판단한다."""

    pseudo = ValidationCheck("LLM issue", "FAIL", issue)
    return _is_blocking_fail_check(pseudo)


def _calculate_quality_score(
    *,
    rule_report: ValidationReport,
    llm_report: ValidationReport,
    checks: List[ValidationCheck],
    warnings: List[str],
    has_blocking_fail: bool,
    risk_result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """실무형 품질 점수를 계산한다.

    하드코딩된 특정 배치 기준이 아니라 다음 일반 기준을 사용한다.
    - 산출물/위험 SQL/파라미터/JOIN 조건 같은 실행 차단 오류는 큰 감점
    - 테스트/성능/품질/재처리 검토는 WARN으로 보고 작은 감점
    - blocking 오류가 없으면 PASS_WITH_WARNINGS 영역을 유지
    """

    pass_count = sum(1 for c in checks if c.result.upper() == "PASS")
    warn_count = sum(1 for c in checks if c.result.upper() == "WARN")
    fail_count = sum(1 for c in checks if c.result.upper() == "FAIL")

    # 룰 검증은 생성 산출물 안정성, LLM 검증은 의미 검증으로 본다.
    # 룰이 WARN 때문에 0.7이어도 blocking 이슈가 없으면 생성 안정성을 0.86 이상으로 본다.
    effective_rule_score = rule_report.score
    if not rule_report.issues:
        effective_rule_score = max(effective_rule_score, 0.86)

    effective_llm_score = max(0.0, min(1.0, llm_report.score))

    base_score = (effective_rule_score * 0.45) + (effective_llm_score * 0.55)

    # PASS가 충분하면 생성 자체는 성공한 것으로 보정한다.
    if pass_count >= 6 and not has_blocking_fail:
        base_score = max(base_score, 0.82)

    # WARN은 실패가 아니므로 작은 감점만 적용한다.
    warn_penalty = min(0.035, (warn_count + len(warnings)) * 0.002)

    risk_penalty = float(risk_result.get("total_penalty", 0.0) or 0.0)
    score = base_score - warn_penalty - risk_penalty

    if has_blocking_fail:
        score = min(score, 0.69)
    else:
        # blocking 실패가 없으면 0.6대로 내려가지 않는다.
        if effective_llm_score >= 0.80:
            score = max(score, 0.84)
        elif effective_llm_score >= 0.75:
            score = max(score, 0.82)
        elif effective_llm_score >= 0.70:
            score = max(score, 0.80)
        else:
            score = max(score, 0.74)

    score = round(max(0.0, min(1.0, score)), 3)

    return score, {
        "policy_version": VALIDATION_POLICY_VERSION,
        "effective_rule_score": round(effective_rule_score, 3),
        "effective_llm_score": round(effective_llm_score, 3),
        "base_score_before_penalty": round(base_score, 3),
        "warn_penalty": round(warn_penalty, 3),
        "risk_penalty": round(risk_penalty, 3),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count_after_normalization": fail_count,
        "has_blocking_fail": has_blocking_fail,
        "score_policy": "실행 차단 오류만 FAIL. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS 영역 유지",
    }


def _merge_reports(rule_report: ValidationReport, llm_report: ValidationReport) -> ValidationReport:
    """룰 검증 결과와 LLM 검증 결과를 병합한다."""

    raw_checks = rule_report.checks + llm_report.checks
    all_checks, downgraded_fail_messages = _normalize_checks_for_operation(raw_checks)

    blocking_fail_checks = [
        check for check in all_checks
        if check.result.upper() == "FAIL" and _is_blocking_fail_check(check)
    ]

    blocking_issues = list(rule_report.issues)
    llm_non_blocking_issues: List[str] = []

    for issue in llm_report.issues:
        if _issue_is_blocking(str(issue)):
            blocking_issues.append(str(issue))
        else:
            llm_non_blocking_issues.append(str(issue))

    all_issues = _dedupe(blocking_issues)
    raw_warnings = _dedupe(
        rule_report.warnings
        + llm_report.warnings
        + llm_non_blocking_issues
        + downgraded_fail_messages
    )
    profile = {}
    if isinstance(rule_report.score_breakdown, Mapping):
        maybe_profile = rule_report.score_breakdown.get("validation_profile", {})
        if isinstance(maybe_profile, Mapping):
            profile = dict(maybe_profile)

    all_warnings = _filter_profile_items(raw_warnings, profile) if profile else raw_warnings[:4]
    all_recommendations = (
        _filter_profile_items(rule_report.recommendations + llm_report.recommendations, profile)
        if profile
        else _dedupe(rule_report.recommendations + llm_report.recommendations)[:4]
    )

    has_blocking_fail = bool(blocking_fail_checks) or bool(all_issues)

    # LLM valid=false라도 원인이 non-blocking 보완사항이면 PASS_WITH_WARNINGS로 본다.
    effective_llm_valid = llm_report.valid or (not has_blocking_fail and llm_report.score >= 0.60)
    valid = (not has_blocking_fail) and rule_report.valid and effective_llm_valid

    score_context = _build_score_context_from_report(
        rule_report=rule_report,
        llm_report=llm_report,
        all_checks=all_checks,
        all_warnings=all_warnings,
    )
    risk_result = _calculate_operational_risk_penalty(score_context)

    score, score_policy = _calculate_quality_score(
        rule_report=rule_report,
        llm_report=llm_report,
        checks=all_checks,
        warnings=all_warnings,
        has_blocking_fail=has_blocking_fail,
        risk_result=risk_result,
    )

    summary = llm_report.summary or rule_report.summary
    # 검토항목은 별도 섹션에서 보여주므로 요약 문장에는 중복 문구를 붙이지 않는다.

    return ValidationReport(
        valid=valid,
        score=score,
        summary=summary,
        interpretation=llm_report.interpretation or rule_report.interpretation,
        detected_batch_type=llm_report.detected_batch_type or rule_report.detected_batch_type,
        checks=all_checks,
        issues=all_issues,
        warnings=all_warnings,
        recommendations=all_recommendations,
        raw_llm_response=llm_report.raw_llm_response,
        score_breakdown={
            "policy_version": VALIDATION_POLICY_VERSION,
            "final_score": score,
            "rule_score": rule_report.score,
            "llm_score": llm_report.score,
            "valid_policy": "실행 차단 FAIL만 blocking. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS",
            "blocking_fail_checks": [f"{c.item}: {c.detail}" for c in blocking_fail_checks],
            "downgraded_fail_checks": downgraded_fail_messages,
            "score_policy": score_policy,
            "risk_penalty": risk_result,
        },
    )


def _build_score_context_from_report(
    rule_report: ValidationReport,
    llm_report: ValidationReport,
    all_checks: List[ValidationCheck],
    all_warnings: List[str],
) -> Dict[str, Any]:
    """점수 산정용 컨텍스트를 만든다.

    핵심 원칙:
    - 복잡도 산정은 실제 생성 SQL 원문 기준으로만 한다.
    - LLM 해석문, warnings, raw_llm_response는 같은 단어가 반복되어 JOIN/INSERT 등이 과다 집계될 수 있으므로
      복잡도 카운트에는 사용하지 않는다.
    - warnings/checks는 테스트 부족, 데이터 품질, 멱등성 같은 운영 검토 신호에만 사용한다.
    """

    query_sql = ""
    batch_type = ""
    if isinstance(rule_report.score_breakdown, Mapping):
        query_sql = str(rule_report.score_breakdown.get("query_sql", "") or "")
        batch_type = str(rule_report.score_breakdown.get("batch_type", "") or "")

    review_texts: List[str] = []
    review_texts.append(rule_report.interpretation or "")
    review_texts.append(llm_report.interpretation or "")
    review_texts.extend([check.item + " " + check.detail for check in all_checks])
    review_texts.extend(all_warnings)
    review_texts.extend(llm_report.recommendations)

    review_text = "\n".join([str(t) for t in review_texts if str(t).strip()])

    return {
        "sql": query_sql,
        "sql_upper": query_sql.upper(),
        "review_text": review_text,
        "review_upper": review_text.upper(),
        "batch_type": batch_type,
        "warning_count": len(all_warnings),
        "fail_count": sum(1 for c in all_checks if c.result.upper() == "FAIL"),
        "warn_check_count": sum(1 for c in all_checks if c.result.upper() == "WARN"),
        "pass_check_count": sum(1 for c in all_checks if c.result.upper() == "PASS"),
    }


def _calculate_operational_risk_penalty(context: Dict[str, Any]) -> Dict[str, Any]:
    """SQL/운영 복잡도 기반 감점.

    구조 복잡도는 실제 query.sql 기준으로만 계산한다.
    review_text는 비구조적 검토 신호의 존재 여부만 본다.
    """

    sql_upper = context.get("sql_upper", "")
    review_text = context.get("review_text", "")
    review_upper = context.get("review_upper", "")
    batch_type = str(context.get("batch_type", "") or "").lower()

    penalties: Dict[str, float] = {}

    join_count = len(re.findall(r"\bJOIN\b", sql_upper))
    if join_count:
        penalties["join_complexity"] = min(0.024, join_count * 0.006)

    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", sql_upper))
    if has_group_by:
        penalties["aggregation_group_by"] = 0.010

    aggregate_count = len(re.findall(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", sql_upper))
    if aggregate_count:
        penalties["aggregate_functions"] = min(0.016, aggregate_count * 0.006)

    has_case = bool(re.search(r"\bCASE\b", sql_upper))
    if has_case:
        penalties["case_classification"] = 0.006

    has_insert = bool(re.search(r"\bINSERT\s+INTO\b", sql_upper))
    if has_insert:
        penalties["table_load"] = 0.006

    has_delete = bool(re.search(r"\bDELETE\s+FROM\b|DELETE_INSERT", sql_upper)) or "delete_insert" in batch_type
    if has_delete:
        penalties["delete_insert_reprocess_review"] = 0.004

    if re.search(r"\bOR\b|IFNULL\s*\(|NVL\s*\(|COALESCE\s*\(", sql_upper):
        penalties["condition_complexity"] = 0.006

    if "FULL SCAN" in review_upper or "인덱스" in review_text:
        penalties["performance_review"] = 0.008

    if "테스트" in review_text and ("부족" in review_text or "누락" in review_text or "존재 여부만" in review_text):
        penalties["test_coverage_gap"] = 0.008

    if "중복" in review_text or "멱등" in review_text or "덮어쓰기" in review_text:
        penalties["reprocess_or_duplication_review"] = 0.008

    if "데이터 품질" in review_text or "ROW COUNT" in review_upper or "NOT NULL" in review_upper:
        penalties["data_quality_review"] = 0.008

    warning_count = int(context.get("warning_count", 0) or 0)
    if warning_count:
        penalties["warning_volume"] = min(0.012, warning_count * 0.0015)

    total_raw = round(sum(penalties.values()), 3)

    if not join_count and not has_group_by and not aggregate_count and not has_insert:
        total_penalty = min(total_raw, 0.025)
    else:
        total_penalty = min(total_raw, 0.055)

    return {
        "total_penalty": total_penalty,
        "penalties": penalties,
        "signals": {
            "join_count": join_count,
            "aggregate_count": aggregate_count,
            "warning_count": warning_count,
            "has_group_by": has_group_by,
            "has_case": has_case,
            "has_insert": has_insert,
            "has_delete_or_delete_insert": has_delete,
            "scoring_source": "query.sql for structural risk; checks/warnings only for review signals",
        },
    }


def _render_markdown_report(report: ValidationReport) -> str:
    """ValidationReport를 운영/발표용 Markdown 형태로 렌더링한다."""

    if report.valid and report.warnings:
        status = "PASS WITH WARNINGS"
    elif report.valid:
        status = "PASS"
    else:
        status = "CHECK REQUIRED"

    review_items = []
    for warning in report.warnings[:4]:
        review_items.append(f"- {warning}")

    if not review_items and report.issues:
        for issue in report.issues[:4]:
            review_items.append(f"- {issue}")

    if not review_items:
        review_items.append("- 운영 반영 전 실행계획 및 데이터 검증 확인")

    lines = [
        "# 🔍 생성 결과 검증",
        "",
        f"- 상태: **{status}**",
        f"- 점수: **{report.score:.2f}**",
        f"- 배치유형: **{report.detected_batch_type or '확인 필요'}**",
        "",
        "## 요약",
        report.summary or "생성 결과 요약 없음",
        "",
        "## 주요 처리",
        report.interpretation or "배치 해석 정보 없음",
        "",
        "## 검토 필요",
    ]

    lines.extend(review_items)

    return "\n".join(lines) + "\n"


def _extract_table_candidates(batch_spec: Mapping[str, Any]) -> List[str]:
    """batch_spec에서 실제 테이블명 후보만 추출한다.

    이전 방식처럼 batch_id/db_to_file/manual/csv 같은 모든 대문자 문자열을
    테이블로 오인하지 않도록, 명확한 테이블 필드만 본다.
    """
    result: List[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        # 테이블명은 보통 TB_*, VW_* 또는 스키마.테이블 형태다.
        upper = text.upper()
        if re.match(r"^[A-Z][A-Z0-9_]*\.[A-Z][A-Z0-9_]+$", upper):
            result.append(upper)
        elif re.match(r"^(TB|VW|TBL|MST|TR)_[A-Z0-9_]+$", upper):
            result.append(upper)

    # ERWin 동적 추론 결과
    resolved_tables = (
        batch_spec.get("meta_source", {})
        .get("resolved_tables", {})
        if isinstance(batch_spec.get("meta_source", {}), Mapping)
        else {}
    )
    if isinstance(resolved_tables, Mapping):
        for table_name in resolved_tables.values():
            add(table_name)

    # 일반 source/target 테이블 필드
    source = batch_spec.get("source", {})
    if isinstance(source, Mapping):
        for key in ("table_name", "table", "base_table", "source_table"):
            add(source.get(key))

    target = batch_spec.get("target", {})
    if isinstance(target, Mapping):
        for key in ("table_name", "table", "target_table"):
            add(target.get(key))

    # 최후 보강: SQL FROM/JOIN 절에서 실제 테이블명 추출
    sql = str(batch_spec.get("sql", "") or "")
    for table_name in re.findall(r"\b(?:FROM|JOIN)\s+([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)?)", sql, flags=re.IGNORECASE):
        add(table_name)

    return _dedupe(result)

def _calculate_rule_score(checks: List[ValidationCheck], config: RuleValidationConfig) -> float:
    if not checks:
        return 0.0
    fail_count = sum(1 for c in checks if c.result.upper() == "FAIL")
    warn_count = sum(1 for c in checks if c.result.upper() == "WARN")
    if fail_count:
        return config.minimum_score_when_rule_fail
    if warn_count:
        return config.minimum_score_when_rule_warn
    return 1.0


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """LLM 응답에서 JSON 객체를 안전하게 추출한다."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```\s*$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"LLM 응답에서 JSON을 파싱하지 못했습니다: {raw[:500]}")


def _clip_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...생략..."


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _safe_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _safe_get_str(mapping: Mapping[str, Any], key: str) -> Optional[str]:
    value = mapping.get(key)
    return str(value) if value is not None and str(value).strip() else None


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
