from __future__ import annotations

from ..context import *
from ..sql.sql_analysis_ui import render_sql_analysis_evaluation_panel
from ..batch.batch_ui import render_batch_development_evaluation_panel

def _extract_where_filter(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "where_filter =" in log:
            return log.split("where_filter =", 1)[1].strip()
    return "(where 조건 없음 또는 default 검색)"

def _extract_search_query(debug_logs: List[str] | None) -> str:
    for log in debug_logs or []:
        if "search_query =" in log:
            return log.split("search_query =", 1)[1].strip()
    return ""

def _filter_fields(data: Dict[str, Any], field_names: List[str]) -> Dict[str, Any]:
    """registry의 used_fields 기준으로 사용 JSON 필드를 추출한다."""
    if not field_names:
        return data
    return {field: data.get(field, [] if isinstance(data.get(field), list) else None) for field in field_names}


def build_used_json_view(result: Any) -> Dict[str, Any]:
    intent = getattr(result, "intent", None)
    structured_data = getattr(result, "structured_data", None) or {}
    graph_data = getattr(result, "graph_data", None) or {}
    query_meta = getattr(result, "query_meta", None) or {}
    spec = get_intent_spec(str(intent or ""))

    if query_meta:
        return {"json_path": spec.get("json_path", "realtime_queries[]"), "used_fields": query_meta}

    source_data = graph_data or structured_data
    if isinstance(source_data, dict) and intent in source_data and isinstance(source_data.get(intent), dict):
        source_data = source_data.get(intent) or {}

    if source_data:
        return {
            "json_path": spec.get("json_path", "domains[].systems[]"),
            "used_fields": _filter_fields(source_data, list(spec.get("used_fields", []) or [])),
        }

    return {"json_path": spec.get("json_path", "(확인 불가)"), "used_fields": {}}


def build_evaluation_checks(result: Any) -> List[str]:
    """
    평가용 문구 생성.
    intent 분류는 코드 상수 대신 agent_intent_registry.json의 category/section/query_id를 사용한다.
    """
    system_id = getattr(result, "system_id", None)
    intent = str(getattr(result, "intent", None) or "")
    sources = getattr(result, "sources", []) or []
    spec = get_intent_spec(intent)
    category = str(spec.get("category") or "")
    checks: List[str] = []

    if category == "realtime":
        expected_section = str(spec.get("source_section") or "realtime_query")
        wrong_realtime_sources = [
            s for s in sources
            if s.get("section") and s.get("section") != expected_section
        ]
        checks.append(
            f"✅ {expected_section} 기준으로 조회 정의가 선택됨"
            if not wrong_realtime_sources
            else f"⚠️ {expected_section}가 아닌 source가 포함됨"
        )

        query_meta = getattr(result, "query_meta", None) or {}
        query_id = query_meta.get("query_id") or spec.get("query_id")
        wrong_query_sources = [
            s for s in sources
            if s.get("query_id") and query_id and s.get("query_id") != query_id
        ]
        checks.append(
            f"✅ query_id 기준으로 {query_id or '대상 조회'}가 정확히 매칭됨"
            if not wrong_query_sources
            else "⚠️ 다른 query_id source가 섞였는지 확인 필요"
        )

    elif category == "system":
        if system_id:
            mixed_sources = [
                s for s in sources
                if s.get("system_id") and s.get("system_id") != system_id
            ]
            checks.append(
                "✅ system_id 기준으로 대상 시스템이 분리됨"
                if not mixed_sources
                else "⚠️ 다른 시스템 source가 섞였는지 확인 필요"
            )
        else:
            checks.append("⚠️ 시스템별 질문인데 system_id가 확인되지 않음")

        expected_section = str(spec.get("section") or intent)
        wrong_section = [
            s for s in sources
            if s.get("section") and s.get("section") != expected_section
        ]
        checks.append(
            "✅ intent에 맞는 section을 사용함"
            if not wrong_section
            else "⚠️ 의도와 다른 section source가 포함됨"
        )

    elif category == "tool":
        checks.append(f"✅ registry 기준 tool intent로 처리됨: {intent}")
    else:
        checks.append("ℹ️ 기본 검색 질문으로 처리됨")

    if getattr(result, "structured_data", None) or getattr(result, "graph_data", None) or getattr(result, "query_meta", None):
        checks.append("✅ 답변 근거가 원본 JSON 구조와 연결됨")

    checks.append(
        f"✅ retrieval 근거 {len(sources)}건 확인 가능"
        if sources
        else "ℹ️ 구조화 JSON 직접 렌더링 중심이라 retrieval source가 없을 수 있음"
    )
    return checks

def render_batch_development_evaluation_panel(result: Any) -> None:
    payload = getattr(result, "batch_dev_result", None) or {}
    batch_spec = payload.get("batch_spec", {}) or {}

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
        for file_path in payload.get("created_files", []):
            st.code(file_path, language="text")

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

def render_evaluation_panel(result: Any) -> None:
    if not st.session_state.evaluation_mode:
        return

    if getattr(result, "intent", None) == "batch_development":
        render_batch_development_evaluation_panel(result)
        return

    if getattr(result, "intent", None) == "sql_analysis":
        render_sql_analysis_evaluation_panel(result)
        return

    used_json = build_used_json_view(result)
    sources = getattr(result, "sources", []) or []
    debug_logs = getattr(result, "debug_logs", []) or []

    with st.expander("📊 평가용 근거 확인", expanded=False):
        st.markdown("##### 1) 질문 해석 결과")
        st.json({
            "original_question": getattr(result, "original_question", ""),
            "normalized_question": getattr(result, "normalized_question", ""),
            "rewritten_question": getattr(result, "rewritten_question", ""),
            "system_id": getattr(result, "system_id", None),
            "intent": getattr(result, "intent", None),
            "render_type": getattr(result, "render_type", None),
        })

        st.markdown("##### 2) Builder 검색 조건")
        st.code(_extract_where_filter(debug_logs), language="python")

        search_query = _extract_search_query(debug_logs)
        if search_query:
            st.markdown("##### 3) 실제 검색 질문")
            st.code(search_query, language="text")

        st.markdown("##### 4) Vector DB에서 가져온 근거 chunk")
        if sources:
            st.dataframe(pd.DataFrame(sources), use_container_width=True)
        else:
            st.info("검색 chunk가 없거나, 구조화 JSON을 직접 사용한 응답입니다.")

        st.markdown("##### 5) 답변에 사용된 원본 JSON 데이터")
        st.caption(f"JSON 위치: {used_json.get('json_path')}")
        st.json(used_json.get("used_fields", {}))

        st.markdown("##### 6) 간단 평가")
        for check in build_evaluation_checks(result):
            st.markdown(f"- {check}")
