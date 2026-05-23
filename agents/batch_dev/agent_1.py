from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .generator.code_generator import generate_code
from .config import BATCH_DEV_LANGGRAPH_ENABLED
from .models import BatchDevResult
from .spec.spec_builder import build_batch_spec
from .generator.template_selector import select_template


class BatchDevAgent:
    """ERWin 메타 기반 동적 배치 생성 에이전트.

    기본 생성 로직은 _run_linear에 유지한다.
    BATCH_DEV_LANGGRAPH_ENABLED=true 인 경우에만 LangGraph workflow를 사용하고,
    LangGraph 미설치/실행 실패 시 기존 선형 처리로 자동 fallback한다.
    """

    def run(self, request_text: str) -> BatchDevResult:
        if BATCH_DEV_LANGGRAPH_ENABLED:
            try:
                from .workflow import run_batch_dev_graph

                return run_batch_dev_graph(request_text)
            except Exception as exc:
                fallback_result = self._run_linear(request_text)
                fallback_result.warnings.append(
                    f"LangGraph 배치 workflow 실행 실패로 기존 선형 처리로 전환했습니다: {type(exc).__name__}: {exc}"
                )
                return fallback_result

        return self._run_linear(request_text)

    def _run_linear(self, request_text: str) -> BatchDevResult:
        created_files: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []
        batch_spec: Dict[str, Any] = {}

        try:
            batch_spec = build_batch_spec(request_text)

            # 중요:
            # batch_type은 실행/처리 유형(db_to_file, aggregation_to_table 등)을 의미한다.
            # template_type은 실제 사용할 템플릿 폴더명을 의미한다.
            #
            # 예)
            # - batch_type    = db_to_file
            # - template_type = ledger_extract_with_classification
            #
            # 따라서 rule_source.template_type이 있으면 그것을 우선 사용해야 한다.
            # 그렇지 않으면 db_to_file 템플릿을 타서 query.sql/job.py가 잘못 생성될 수 있다.
            rule_source = batch_spec.get("rule_source") or {}
            template_type = (
                rule_source.get("template_type")
                or batch_spec.get("template_type")
                or select_template(batch_spec)
            )

            created_files = generate_code(batch_spec, template_type)

            target = batch_spec.get("target") or {}
            if target.get("table") == "TODO_TARGET_TABLE":
                warnings.append("집계 결과 target 테이블을 요청서나 rule에 지정하지 않아 TODO_TARGET_TABLE로 생성했습니다.")

            if batch_spec.get("source", {}).get("dynamic_inference"):
                warnings.append("ERWin 메타의 table_role/column role/relations 기반으로 테이블과 조인을 추론했습니다. 운영 반영 전 SQL 정합성 검토가 필요합니다.")

            return BatchDevResult(
                batch_spec=batch_spec,
                created_files=created_files,
                warnings=warnings,
                errors=errors,
                message=f"배치 소스가 생성되었습니다: {Path(created_files[0]).parent if created_files else ''}",
            )

        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            return BatchDevResult(
                batch_spec=batch_spec,
                created_files=created_files,
                warnings=warnings,
                errors=errors,
                message="배치 개발 요청을 처리하지 못했습니다. 오류를 확인하세요.",
            )
