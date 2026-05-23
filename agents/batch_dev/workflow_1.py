from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    StateGraph = None
    START = "__start__"
    END = "__end__"

from .generator.code_generator import generate_code
from .config import BATCH_SQL_IMPROVEMENT_ENABLED, ERWIN_METADATA_PATH
from .validation.llm_batch_validator import validate_batch_generation
from .models import BatchDevResult
from .spec.spec_builder import build_batch_spec
from .generator.template_selector import select_template
from .validation.validator import validate_batch_spec

try:
    from .advisor.sql_improvement_advisor import analyze_sql_improvement
except Exception:
    analyze_sql_improvement = None


class BatchDevWorkflowState(TypedDict, total=False):
    request_text: str
    batch_spec: Dict[str, Any]
    template_type: str
    created_files: List[str]
    generated_files: Dict[str, str]
    validation_report: Any
    sql_improvement_report: Any
    warnings: List[str]
    errors: List[str]
    debug_logs: List[str]
    result: BatchDevResult


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_created_files(created_files: List[str]) -> Dict[str, str]:
    """생성된 파일 내용을 검증 노드에서 사용할 수 있게 표준 dict로 변환한다."""
    result: Dict[str, str] = {}

    for file_name in created_files or []:
        path = Path(file_name)
        if not path.exists() or not path.is_file():
            continue
        try:
            result[path.name] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result[path.name] = path.read_text(encoding="utf-8-sig")
        except Exception:
            # 생성 파일 하나를 못 읽어도 전체 workflow는 계속 진행한다.
            continue

    return result


def _resolve_template_type(batch_spec: Dict[str, Any]) -> str:
    """실제 사용할 템플릿 타입을 결정한다.

    특정 배치명/업무명을 보지 않고, rule/spec/template selector 메타만 사용한다.
    """
    rule_source = batch_spec.get("rule_source") or {}
    return str(
        rule_source.get("template_type")
        or batch_spec.get("template_type")
        or select_template(batch_spec)
    )


def _append_warning(state: BatchDevWorkflowState, message: str) -> List[str]:
    warnings = list(state.get("warnings", []))
    if message and message not in warnings:
        warnings.append(message)
    return warnings


def _prepare_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 1] node = prepare")
    return {
        **state,
        "request_text": str(state.get("request_text") or ""),
        "warnings": list(state.get("warnings", [])),
        "errors": list(state.get("errors", [])),
        "debug_logs": debug_logs,
    }


def _build_spec_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 2] node = build_spec")

    batch_spec = build_batch_spec(state.get("request_text", ""))
    debug_logs.append(f"[BATCH_LG 2-1] batch_id = {batch_spec.get('batch_id')}")
    debug_logs.append(f"[BATCH_LG 2-2] batch_type = {batch_spec.get('batch_type')}")

    return {**state, "batch_spec": batch_spec, "debug_logs": debug_logs}


def _validate_spec_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 3] node = validate_spec")

    batch_spec = state.get("batch_spec", {}) or {}
    erwin_meta = _read_json(ERWIN_METADATA_PATH, {"tables": [], "relations": []})
    spec_errors, spec_warnings = validate_batch_spec(batch_spec, erwin_meta)

    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    errors.extend([f"spec 검증 오류: {item}" for item in spec_errors])
    warnings.extend([f"spec 검증 경고: {item}" for item in spec_warnings])

    source = batch_spec.get("source") or {}
    if source.get("dynamic_inference"):
        warnings.append("ERWin 메타의 table_role/column role/relations 기반으로 테이블과 조인을 추론했습니다. 운영 반영 전 SQL 정합성 검토가 필요합니다.")

    target = batch_spec.get("target") or {}
    if target.get("table") == "TODO_TARGET_TABLE":
        warnings.append("집계 결과 target 테이블을 요청서나 rule에 지정하지 않아 TODO_TARGET_TABLE로 생성했습니다.")

    debug_logs.append(f"[BATCH_LG 3-1] spec_error_count = {len(spec_errors)}")
    debug_logs.append(f"[BATCH_LG 3-2] spec_warning_count = {len(spec_warnings)}")

    return {
        **state,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "debug_logs": debug_logs,
    }


def _select_template_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 4] node = select_template")

    template_type = _resolve_template_type(state.get("batch_spec", {}) or {})
    debug_logs.append(f"[BATCH_LG 4-1] template_type = {template_type}")
    return {**state, "template_type": template_type, "debug_logs": debug_logs}


def _generate_code_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 5] node = generate_code")

    created_files = generate_code(
        state.get("batch_spec", {}) or {},
        state.get("template_type", "") or select_template(state.get("batch_spec", {}) or {}),
    )
    generated_files = _read_created_files(created_files)

    debug_logs.append(f"[BATCH_LG 5-1] created_file_count = {len(created_files)}")
    return {
        **state,
        "created_files": created_files,
        "generated_files": generated_files,
        "debug_logs": debug_logs,
    }


def _validate_generated_files_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 6] node = validate_generated_files")

    batch_spec = state.get("batch_spec", {}) or {}
    created_files = state.get("created_files", []) or []
    generated_files = state.get("generated_files", {}) or {}

    output_dir: Optional[Path] = None
    if created_files:
        output_dir = Path(created_files[0]).parent

    report = validate_batch_generation(
        request_text=state.get("request_text", ""),
        batch_spec=batch_spec,
        generated_files=generated_files,
        output_dir=output_dir,
    )

    warnings = list(state.get("warnings", []))
    if getattr(report, "warnings", None):
        warnings.extend([f"생성물 검증 경고: {item}" for item in report.warnings])

    errors = list(state.get("errors", []))
    if getattr(report, "issues", None) and not getattr(report, "valid", True):
        errors.extend([f"생성물 검증 오류: {item}" for item in report.issues])

    debug_logs.append(f"[BATCH_LG 6-1] validation_valid = {getattr(report, 'valid', None)}")
    debug_logs.append(f"[BATCH_LG 6-2] validation_score = {getattr(report, 'score', None)}")

    return {
        **state,
        "validation_report": report,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors)),
        "debug_logs": debug_logs,
    }


def _sql_improvement_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 7] node = sql_improvement")

    if not BATCH_SQL_IMPROVEMENT_ENABLED:
        debug_logs.append("[BATCH_LG 7-1] skipped = disabled")
        return {**state, "debug_logs": debug_logs}

    if analyze_sql_improvement is None:
        warnings = _append_warning(state, "SQL 개선 제안 모듈을 불러오지 못해 해당 단계를 건너뜁니다.")
        debug_logs.append("[BATCH_LG 7-1] skipped = module_unavailable")
        return {**state, "warnings": warnings, "debug_logs": debug_logs}

    try:
        report = analyze_sql_improvement(
            batch_spec=state.get("batch_spec", {}) or {},
            generated_files=state.get("generated_files", {}) or {},
        )
        debug_logs.append("[BATCH_LG 7-1] sql_improvement = success")
        return {**state, "sql_improvement_report": report, "debug_logs": debug_logs}
    except Exception as exc:
        warnings = _append_warning(
            state,
            f"SQL 개선 제안 생성에 실패했습니다: {type(exc).__name__}: {exc}",
        )
        debug_logs.append(f"[BATCH_LG 7-1] sql_improvement = failed ({type(exc).__name__})")
        return {**state, "warnings": warnings, "debug_logs": debug_logs}


def _finalize_node(state: BatchDevWorkflowState) -> BatchDevWorkflowState:
    debug_logs = list(state.get("debug_logs", []))
    debug_logs.append("[BATCH_LG 8] node = finalize")

    created_files = state.get("created_files", []) or []
    warnings = list(dict.fromkeys(state.get("warnings", []) or []))
    errors = list(dict.fromkeys(state.get("errors", []) or []))

    if not created_files and not errors:
        errors.append("생성된 파일이 없습니다.")

    result = BatchDevResult(
        batch_spec=state.get("batch_spec", {}) or {},
        created_files=created_files,
        warnings=warnings,
        errors=errors,
        message=(
            f"배치 소스가 생성되었습니다: {Path(created_files[0]).parent}"
            if created_files and not errors
            else "배치 개발 요청을 처리했지만 확인이 필요한 항목이 있습니다."
        ),
    )

    # 기존 BatchDevResult 구조를 깨지 않으면서 상세 리포트는 동적 속성으로 보존한다.
    setattr(result, "validation_report", state.get("validation_report"))
    setattr(result, "sql_improvement_report", state.get("sql_improvement_report"))
    setattr(result, "debug_logs", debug_logs)

    return {**state, "result": result, "debug_logs": debug_logs}


def run_batch_dev_graph(request_text: str) -> BatchDevResult:
    """LangGraph 기반 배치 개발 workflow 실행.

    기존 build/spec/generate/validate 함수를 노드로 감싸므로
    업무별 하드코딩 없이 기존 rule, schema, ERWin meta, template 설정을 그대로 따른다.
    """
    if StateGraph is None:
        raise RuntimeError("LangGraph가 설치되어 있지 않습니다. pip install langgraph 후 사용하세요.")

    workflow = StateGraph(BatchDevWorkflowState)
    workflow.add_node("prepare", _prepare_node)
    workflow.add_node("build_spec", _build_spec_node)
    workflow.add_node("validate_spec", _validate_spec_node)
    workflow.add_node("select_template", _select_template_node)
    workflow.add_node("generate_code", _generate_code_node)
    workflow.add_node("validate_generated_files", _validate_generated_files_node)
    workflow.add_node("sql_improvement", _sql_improvement_node)
    workflow.add_node("finalize", _finalize_node)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "build_spec")
    workflow.add_edge("build_spec", "validate_spec")
    workflow.add_edge("validate_spec", "select_template")
    workflow.add_edge("select_template", "generate_code")
    workflow.add_edge("generate_code", "validate_generated_files")
    workflow.add_edge("validate_generated_files", "sql_improvement")
    workflow.add_edge("sql_improvement", "finalize")
    workflow.add_edge("finalize", END)

    compiled = workflow.compile()
    final_state = compiled.invoke(
        {
            "request_text": request_text,
            "warnings": [],
            "errors": [],
            "debug_logs": ["[BATCH_LG 0] workflow = batch_dev"],
        }
    )

    result = final_state.get("result")
    if not result:
        raise RuntimeError("BatchDev LangGraph workflow did not return BatchDevResult.")
    return result
