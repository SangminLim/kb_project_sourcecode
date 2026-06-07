from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import *
from ..services.realtime_payload import *
from ..renderers.streamlit_renderers import *

def _read_generated_file(file_path: Any) -> tuple[str, str] | None:
    """생성된 파일 경로를 읽어서 validator 입력 형식으로 변환한다.

    BatchDevAgent가 created_files에 파일 경로만 넘기므로,
    app.py에서는 실제 파일 내용을 읽어 validator에 전달한다.
    파일이 없거나 읽을 수 없으면 해당 파일은 건너뛰어 화면 장애를 막는다.
    """
    try:
        path = Path(str(file_path))
        if not path.exists() or not path.is_file():
            return None
        return path.name, path.read_text(encoding="utf-8")
    except Exception:
        return None


def _build_generated_files_for_validation(batch_spec: Dict[str, Any], created_files: List[Any]) -> Dict[str, str]:
    """LLM 검증 모듈에 넘길 생성 파일 묶음을 만든다.

    특정 파일명을 하드코딩해서 판단하지 않고, BatchDevAgent가 돌려준
    created_files 목록을 기준으로 실제 파일 내용을 수집한다.
    단, query.sql이 created_files에 없더라도 batch_spec.sql이 있으면
    검증 정확도를 위해 query.sql 항목으로 보강한다.
    """
    generated_files: Dict[str, str] = {}

    for file_path in created_files or []:
        item = _read_generated_file(file_path)
        if item is None:
            continue
        file_name, content = item
        generated_files[file_name] = content

    if "query.sql" not in generated_files and batch_spec.get("sql"):
        generated_files["query.sql"] = str(batch_spec.get("sql") or "")

    generated_files.setdefault(
        "batch_spec.json",
        json.dumps(batch_spec or {}, ensure_ascii=False, indent=2),
    )
    return generated_files


def _resolve_validation_output_dir(created_files: List[Any]) -> Path | None:
    """validation_report.json/md를 저장할 폴더를 찾는다."""
    for file_path in created_files or []:
        try:
            path = Path(str(file_path))
            if path.exists():
                return path.parent if path.is_file() else path
        except Exception:
            continue
    return None




def _compact_batch_warning(text: Any) -> str:
    """배치 개발 화면의 검토 문구를 실무형 짧은 문장으로 정규화한다.

    특정 배치명/테이블명을 보지 않고 경고 문구의 일반 품질 신호만 축약한다.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    raw = raw.replace("생성물 검증 경고:", "").strip()
    upper = raw.upper()

    # 내부 추론 방식 안내는 평가 근거로는 유용하지만 사용자 상단 검토항목에서는 과하게 보일 수 있어 숨긴다.
    if "ERWIN" in upper and "추론" in raw:
        return ""

    if "USE_YN" in upper or "APPLY_START" in upper or "APPLY_END" in upper or "인덱스" in raw:
        if "USE_YN" in upper and ("APPLY_START" in upper or "APPLY_END" in upper):
            return "USE_YN / APPLY_START_DT 조건 인덱스 확인"
        if "APPLY_START" in upper or "APPLY_END" in upper:
            return "APPLY_START_DT / APPLY_END_DT 인덱스 확인"
        return "조건 컬럼 인덱스 확인"

    if "파일" in raw and ("덮어쓰기" in raw or "멱등" in raw or "중복" in raw):
        return "CSV 파일 중복 생성 방지 확인"

    if "BASE_DATE" in upper or "기준일자" in raw or "파라미터" in raw:
        return "기준일자(base_date) 파라미터 검증 확인"

    if "헤더" in raw or "구분자" in raw or "CSV" in upper or "OUTPUT_FORMAT" in upper:
        return "출력 헤더/구분자 확인"

    if "금액" in raw and "합계" in raw:
        return "금액 합계 검증 확인"

    if "ROW COUNT" in upper or "건수" in raw or "중복" in raw:
        return "row count 및 중복 검증 확인"

    if "테스트" in raw:
        return "SQL/파일포맷/파라미터 테스트 보강"

    return raw[:80]


def _dedupe_compact_items(items: List[Any], *, max_items: int = 3) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items or []:
        compact = _compact_batch_warning(item)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        result.append(compact)
        if len(result) >= max_items:
            break
    return result


def _split_interpretation_lines(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []

    lines: List[str] = []
    for line in text.splitlines():
        clean = line.strip().lstrip("-•").strip()
        if clean:
            lines.append(clean)

    if len(lines) <= 1:
        # LLM이 한 문단으로 준 경우 문장 단위로 최대 3개만 보여준다.
        lines = [
            item.strip()
            for item in re.split(r"(?<=[.!?。])\s+", text)
            if item.strip()
        ]

    return lines[:3]


def _build_validation_review_items(validation_report: Dict[str, Any]) -> List[str]:
    warnings = validation_report.get("warnings") or []
    recommendations = validation_report.get("recommendations") or []
    issues = validation_report.get("issues") or []
    return _dedupe_compact_items(list(warnings) + list(recommendations) + list(issues), max_items=3)

def run_batch_sql_improvement(dev_result: Any) -> Dict[str, Any] | None:
    """생성된 배치 SQL에 대해 자동 개선 제안을 생성한다.

    실무 적용 원칙:
    - query.sql 생성 이후에만 실행한다.
    - SQL을 직접 수정하지 않고 개선 후보만 제안한다.
    - LLM 모듈이 없거나 실패해도 룰 기반 분석 결과를 반환한다.
    - 테이블/컬럼은 batch_spec.sql, meta_source, created_files에서 읽어 하드코딩을 줄인다.
    - BATCH_SQL_IMPROVEMENT_ENABLED=false이면 아예 실행하지 않아 화면/리포트에 표시하지 않는다.
    """
    if not BATCH_SQL_IMPROVEMENT_ENABLED:
        return None

    if analyze_sql_improvement is None:
        return {
            "enabled": False,
            "risk_level": "UNKNOWN",
            "summary": "sql_improvement_advisor 모듈을 찾지 못했습니다.",
            "suggestions": [],
            "warnings": ["batch_dev/sql_improvement_advisor.py 파일 위치를 확인하세요."],
            "generated_by": "none",
        }

    batch_spec = getattr(dev_result, "batch_spec", {}) or {}
    created_files = getattr(dev_result, "created_files", []) or []
    generated_files = _build_generated_files_for_validation(batch_spec, created_files)
    output_dir = _resolve_validation_output_dir(created_files)

    try:
        return analyze_sql_improvement(
            batch_spec=batch_spec,
            generated_files=generated_files,
            llm_generate_fn=ollama_generate,
            model=BATCH_VALIDATION_LLM_MODEL,
            use_llm=BATCH_VALIDATION_USE_LLM,
            output_dir=output_dir,
        )
    except Exception as exc:
        logger.exception("SQL improvement generation failed")
        return {
            "enabled": False,
            "risk_level": "UNKNOWN",
            "summary": "SQL 자동 개선 제안 생성에 실패했습니다.",
            "suggestions": [],
            "warnings": [str(exc)],
            "generated_by": "error",
        }


def render_sql_improvement_report(sql_improvement: Dict[str, Any] | None, *, max_items: int | None = None) -> None:
    """SQL 자동 개선 제안을 Streamlit 화면에 표시한다.

    출력 결과 화면이 길어지지 않도록 기본은 접힘 상태로 보여준다.
    """
    if not sql_improvement:
        return

    with st.expander("🚀 SQL 자동 개선 제안", expanded=False):
        generated_by = sql_improvement.get("generated_by", "-")
        risk_level = sql_improvement.get("risk_level", "-")
        summary = sql_improvement.get("summary", "")

        if risk_level == "HIGH":
            st.error(f"위험도: {risk_level} / 생성방식: {generated_by}")
        elif risk_level in {"MEDIUM", "UNKNOWN"}:
            st.warning(f"위험도: {risk_level} / 생성방식: {generated_by}")
        else:
            st.success(f"위험도: {risk_level} / 생성방식: {generated_by}")

        if summary:
            st.write(summary)

        warnings = sql_improvement.get("warnings") or []
        for warning in warnings:
            st.caption(f"⚠️ {warning}")

        suggestions = sql_improvement.get("suggestions") or []
        if max_items is not None:
            suggestions = suggestions[:max_items]

        for idx, item in enumerate(suggestions, start=1):
            title = str(item.get("type") or "RECOMMENDATION").strip()
            target = str(item.get("target") or "").strip()
            reason = str(item.get("reason") or "").strip()
            recommendation = str(item.get("recommendation") or "").strip()
            sql = str(item.get("sql") or "").strip()

            with st.container(border=True):
                st.markdown(f"**{idx}. {title}**")
                if target:
                    st.markdown(f"**대상:** `{target}`")
                if reason:
                    st.markdown(f"**이유:** {reason}")
                if recommendation:
                    st.markdown(f"**개선안:** {recommendation}")
                if sql:
                    language = "sql" if any(token in sql.upper() for token in ["SELECT", "CREATE", "INDEX", "WHERE", "JOIN"]) else "text"
                    st.code(sql, language=language)

def run_batch_llm_validation(user_question: str, dev_result: Any) -> Dict[str, Any] | None:
    """배치 생성 결과에 대해 룰 검증 + 선택적 LLM 검증을 수행한다.

    - validator 모듈이 없으면 앱 전체가 죽지 않도록 경고 payload만 반환한다.
    - Ollama/LLM 호출이 실패해도 룰 기반 검증으로 한 번 더 검증한다.
    - 결과는 dict로 저장해서 Streamlit session_state/history에 그대로 보관한다.
    """
    if validate_batch_generation is None:
        return {
            "valid": False,
            "score": 0.0,
            "summary": "llm_batch_validator 모듈을 찾지 못했습니다.",
            "interpretation": "batch_dev 폴더에 llm_batch_validator.py가 있는지 확인하세요.",
            "checks": [],
            "issues": ["검증 모듈 import 실패"],
            "warnings": [],
            "recommendations": ["from batch_dev.llm_batch_validator import validate_batch_generation 경로를 확인하세요."],
        }

    batch_spec = getattr(dev_result, "batch_spec", {}) or {}
    created_files = getattr(dev_result, "created_files", []) or []
    generated_files = _build_generated_files_for_validation(batch_spec, created_files)
    output_dir = _resolve_validation_output_dir(created_files)

    # llm_batch_validator.py 내부에서 프로젝트 공통 llm.py의 ollama_generate를 재사용한다.
    # app.py에서는 별도 Ollama client를 만들지 않는다.
    try:
        report = validate_batch_generation(
            request_text=user_question,
            batch_spec=batch_spec,
            generated_files=generated_files,
            llm_client=None,
            output_dir=output_dir,
        )
        return report.to_dict()
    except Exception as llm_error:
        # LLM 호출/응답 파싱 오류가 나도 검증 화면 자체는 유지한다.
        try:
            report = validate_batch_generation(
                request_text=user_question,
                batch_spec=batch_spec,
                generated_files=generated_files,
                llm_client=None,
                output_dir=output_dir,
            )
            payload = report.to_dict()
            payload.setdefault("warnings", [])
            payload["warnings"].append(f"LLM 검증 실패로 룰 기반 검증만 수행했습니다: {llm_error}")
            return payload
        except Exception as rule_error:
            return {
                "valid": False,
                "score": 0.0,
                "summary": "배치 검증 리포트 생성에 실패했습니다.",
                "interpretation": "생성 파일 경로, batch_spec 구조, validator 모듈을 확인하세요.",
                "checks": [],
                "issues": [str(rule_error)],
                "warnings": [f"LLM 검증 오류: {llm_error}"],
                "recommendations": ["created_files 경로가 실제 파일로 존재하는지 확인하세요."],
            }

def run_batch_development(user_question: str) -> Any:
    """
    배치 개발 요청은 기존 RAG 흐름과 분리해서 처리한다.
    기존 HandoverAgent/Chroma 검색 품질에 영향을 주지 않기 위한 별도 진입점이다.
    """
    dev_result = BatchDevAgent().run(user_question)
    sql_improvement = run_batch_sql_improvement(dev_result) if dev_result.success and BATCH_SQL_IMPROVEMENT_ENABLED else None
    validation_report = run_batch_llm_validation(user_question, dev_result) if dev_result.success else None
    answer = dev_result.message
    if dev_result.errors:
        answer = "배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요."
    return SimpleNamespace(
        answer=answer,
        intent="batch_development",
        render_type="batch_dev",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=apply_dictionary_rewrite(user_question),
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            "[BATCH_DEV 1] intent=batch_development",
            f"[BATCH_DEV 2] success={dev_result.success}",
            f"[BATCH_DEV 3] created_files={len(dev_result.created_files)}",
        ],
        batch_dev_result={
            "batch_spec": dev_result.batch_spec,
            "created_files": dev_result.created_files,
            "warnings": dev_result.warnings,
            "errors": dev_result.errors,
            "message": dev_result.message,
            "success": dev_result.success,
            "validation_report": validation_report,
            "sql_improvement": sql_improvement,
        },
    )


def _build_general_fallback_chat_config() -> Any:
    """일반 질문 fallback용 ChatConfig를 만든다.

    app.py가 특정 LLM provider에 묶이지 않도록 프로젝트 공통 ChatConfig를 우선 사용하고,
    생성자 차이가 있으면 SimpleNamespace로 안전하게 대체한다.
    """
    candidates = [
        {
            "model": GENERAL_FALLBACK_MODEL,
            "temperature": GENERAL_FALLBACK_TEMPERATURE,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
            "max_tokens": GENERAL_FALLBACK_MAX_TOKENS,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
            "temperature": GENERAL_FALLBACK_TEMPERATURE,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
            "timeout": GENERAL_FALLBACK_TIMEOUT,
        },
        {
            "model": GENERAL_FALLBACK_MODEL,
        },
    ]
    for kwargs in candidates:
        try:
            return ChatConfig(**kwargs)
        except TypeError:
            continue
    return SimpleNamespace(
        model=GENERAL_FALLBACK_MODEL,
        temperature=GENERAL_FALLBACK_TEMPERATURE,
        timeout=GENERAL_FALLBACK_TIMEOUT,
        max_tokens=GENERAL_FALLBACK_MAX_TOKENS,
    )


def build_out_of_scope_message(question: str = "") -> str:
    """업무 범위 밖 질문 안내 문구를 한 곳에서 관리한다."""
    return """
현재 질문은 업무 인수인계 지원 범위를 벗어난 일반 질문으로 판단되었습니다.

이 에이전트는 현재 카드업무 인수인계 및 운영 지원 기능에 특화되어 있습니다.

지원 기능 예시:
- 업무 개요 조회
- 배치 프로세스 조회
- 배치 흐름도 조회
- 테이블 리니지 조회
- 청구 이용내역서 월별 금액 조회
- 오늘 장애현황 조회
- 배치 개발 요청
- SQL 분석 및 개선 검토

예시 질문:
- "소득공제 업무 개요 알려줘"
- "청구 배치 흐름도 보여줘"
- "오늘 장애현황 조회"
- "가맹점 월정산 배치 개발 요청"
- "이 SQL 문제점 분석해줘"

일반 질문 응답을 사용하려면 GENERAL_FALLBACK_USE_LLM=true 로 설정하세요.
""".strip()


def _looks_like_out_of_scope_answer(answer: str) -> bool:
    """기존 Agent가 반환한 범위 밖 안내 문구를 감지한다.

    llm.py 내부 문구가 일부 바뀌어도 핵심 표현 기준으로 보수적으로 감지한다.
    """
    text = str(answer or "")
    patterns = [
        "업무 인수인계 범위",
        "지원 범위",
        "현재 업무 인수인계",
        "질문만 지원",
        "지원하지 않는 질문",
    ]
    return any(pattern in text for pattern in patterns)


def run_general_fallback(user_question: str, chat_history: Optional[List[Dict[str, str]]] = None, reason: str = "out_of_scope") -> Any:
    """업무 범위 밖 일반 질문을 처리한다.

    - 설정으로 LLM fallback을 끌 수 있다.
    - LLM 호출 실패 시에도 사용자에게 딱딱한 차단 문구가 아니라 자연스러운 안내를 보여준다.
    - 업무 전용 RAG/DB/배치 로직과 분리해 운영 영향도를 낮춘다.
    """
    normalized_question = apply_dictionary_rewrite(user_question)
    answer = build_out_of_scope_message(user_question)
    generated_by = "guide"
    error_message = None

    if GENERAL_FALLBACK_USE_LLM:
        system_prompt = """
너는 업무 인수인계 에이전트의 일반 질문 fallback assistant다.
사용자가 업무 범위 밖의 일반 질문을 하면 한국어로 간단하고 실용적으로 답한다.
단, 회사 내부 데이터/DB/문서가 필요한 척 추측하지 않는다.
최신 정보, 날씨, 주가, 법률/의료/금융 투자판단처럼 실시간성 또는 전문성이 필요한 내용은 확인 필요성을 명확히 말한다.
""".strip()
        history_text = ""
        for item in (chat_history or [])[-6:]:
            role = item.get("role", "")
            content = str(item.get("content", "")).strip()
            if role and content:
                history_text += f"{role}: {content}\n"
        user_prompt = f"""
최근 대화:
{history_text or "(없음)"}

사용자 질문:
{user_question}

답변 조건:
- 업무 인수인계 범위 밖 질문임을 길게 반복하지 말 것
- 가능한 경우 바로 답변할 것
- 실시간 정보가 필요한 질문이면 현재 시스템에서는 실시간 조회가 필요하다고 말하고, 확인 방법을 짧게 안내할 것
""".strip()
        try:
            raw = ollama_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                config=_build_general_fallback_chat_config(),
            )
            if isinstance(raw, dict):
                answer = str(raw.get("answer") or raw.get("content") or raw)
            else:
                answer = str(raw or "").strip() or answer
            generated_by = "llm_fallback"
        except Exception as exc:
            logger.exception("General fallback generation failed")
            error_message = str(exc)
            answer = build_out_of_scope_message(user_question)

    return SimpleNamespace(
        answer=answer,
        intent="general_fallback",
        render_type="text",
        graph_data=None,
        query_meta=None,
        realtime_mode=None,
        structured_data=None,
        realtime_payload=None,
        normalized_question=normalized_question,
        rewritten_question=user_question,
        system_id=None,
        sources=[],
        debug_logs=[
            f"[GENERAL_FALLBACK 1] reason={reason}",
            f"[GENERAL_FALLBACK 2] enabled={GENERAL_FALLBACK_USE_LLM}",
            f"[GENERAL_FALLBACK 3] generated_by={generated_by}",
            f"[GENERAL_FALLBACK 4] error={error_message}" if error_message else "[GENERAL_FALLBACK 4] error=None",
        ],
    )

def render_batch_development_result(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    if not payload:
        st.warning("배치 개발 결과가 없습니다.")
        return

    success = payload.get("success")
    validation_report = payload.get("validation_report") or {}

    st.markdown("#### 🛠️ 배치 개발 결과")
    if success:
        st.success(payload.get("message", "배치 소스가 생성되었습니다."))
    else:
        st.error(payload.get("message", "배치 소스 생성에 실패했습니다."))

    errors = payload.get("errors") or []
    raw_warnings = list(payload.get("warnings") or [])
    if isinstance(validation_report, dict):
        raw_warnings.extend(validation_report.get("warnings") or [])
        raw_warnings.extend(validation_report.get("recommendations") or [])

    if errors:
        st.markdown("##### ❌ 오류")
        for item in errors:
            st.markdown(f"- {item}")

    display_warnings = _dedupe_compact_items(raw_warnings, max_items=3)
    if display_warnings:
        st.markdown("##### ⚠️ 검토 필요")
        for item in display_warnings:
            st.markdown(f"- {item}")

    sql_improvement = payload.get("sql_improvement")
    if sql_improvement:
        render_sql_improvement_report(sql_improvement, max_items=3)

    # 상세 batch_spec / 생성 파일 목록은 화면에서 숨긴다.
    # 필요 시 generated 폴더의 batch_spec.json, validation_report.json 파일로 확인한다.
    if validation_report:
        with st.expander("🔍 생성 결과 요약", expanded=False):
            is_valid = bool(validation_report.get("valid"))
            score = validation_report.get("score", 0)
            summary = str(validation_report.get("summary") or "").strip()
            interpretation = validation_report.get("interpretation", "")

            if is_valid:
                st.success(f"검증 통과 - score={score}")
            else:
                st.warning(f"확인 필요 - score={score}")

            if summary:
                st.markdown("**요약**")
                st.write(summary)

            flow_items = _split_interpretation_lines(interpretation)
            if flow_items:
                st.markdown("**주요 처리**")
                for item in flow_items:
                    st.markdown(f"- {item}")

            review_items = _build_validation_review_items(validation_report)
            if review_items:
                st.markdown("**검토 포인트**")
                for item in review_items:
                    st.markdown(f"- {item}")

    st.info("운영 반영 전 query.sql, 주요 조건, 인덱스, 파일 포맷을 확인하세요.")

def render_batch_development_evaluation_panel(result: Any) -> None:
    """배치개발 평가용 근거 패널.

    ui/evaluation/evaluation_panel.py에서 import하는 공개 함수다.
    분리 리팩토링 후에도 기존 화면의 평가 근거 표시가 깨지지 않도록
    batch 개발 결과 payload를 기준으로 렌더링한다.
    """
    payload: Dict[str, Any] = getattr(result, "batch_dev_result", None) or {}
    batch_spec: Dict[str, Any] = payload.get("batch_spec", {}) or {}

    with st.expander("📊 배치개발 평가용 근거 확인", expanded=False):
        st.markdown("##### 1) 요청 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
            "success": payload.get("success"),
        })

        st.markdown("##### 2) 생성된 배치 명세")
        st.json({
            "batch_id": batch_spec.get("batch_id"),
            "batch_name": batch_spec.get("batch_name"),
            "batch_type": batch_spec.get("batch_type"),
            "source": batch_spec.get("source"),
            "target": batch_spec.get("target"),
        })

        st.markdown("##### 3) 사용한 ERWIN 메타")
        st.json(batch_spec.get("meta_source", {}))

        st.markdown("##### 4) 사용한 Rule / SQL Template")
        st.json(batch_spec.get("rule_source", {}))

        st.markdown("##### 5) 생성 SQL")
        st.code(batch_spec.get("sql", ""), language="sql")

        st.markdown("##### 6) 생성 파일")
        for file_path in payload.get("created_files", []) or []:
            st.code(str(file_path), language="text")

        st.markdown("##### 7) 기본 검증 결과")
        st.json({
            "warnings": payload.get("warnings", []),
            "errors": payload.get("errors", []),
        })

        validation_report = payload.get("validation_report")
        if validation_report:
            st.markdown("##### 8) LLM 해석/검증 요약")
            st.json({
                "valid": validation_report.get("valid"),
                "score": validation_report.get("score"),
                "summary": validation_report.get("summary"),
                "interpretation": validation_report.get("interpretation"),
                "warnings": validation_report.get("warnings", []),
                "recommendations": validation_report.get("recommendations", []),
            })

        sql_improvement = payload.get("sql_improvement")
        if sql_improvement:
            st.markdown("##### 9) SQL 자동 개선 제안")
            st.json({
                "enabled": sql_improvement.get("enabled"),
                "risk_level": sql_improvement.get("risk_level"),
                "generated_by": sql_improvement.get("generated_by"),
                "summary": sql_improvement.get("summary"),
                "suggestions": sql_improvement.get("suggestions", []),
                "warnings": sql_improvement.get("warnings", []),
            })
