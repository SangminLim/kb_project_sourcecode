from __future__ import annotations

from typing import Any, Dict, Optional

from .code_generator import generate_code
from .models import BatchDevResult
from .spec_builder import build_batch_spec
from .template_selector import select_template
from .validator import validate_batch_spec


class BatchDevAgent:
    """
    배치 개발 전용 에이전트.
    기존 RAG/실시간 조회 흐름과 분리되어 동작한다.
    """

    def run(self, user_request: str, preset_spec: Optional[Dict[str, Any]] = None) -> BatchDevResult:
        spec = preset_spec or build_batch_spec(user_request)
        errors, warnings = validate_batch_spec(spec)

        if errors:
            return BatchDevResult(
                batch_spec=spec,
                created_files=[],
                warnings=warnings,
                errors=errors,
                message="배치 명세 검증에 실패했습니다. 오류를 수정한 뒤 다시 생성하세요.",
            )

        template_type = select_template(spec)
        try:
            created_files = generate_code(spec, template_type)
        except Exception as exc:
            return BatchDevResult(
                batch_spec=spec,
                created_files=[],
                warnings=warnings,
                errors=[str(exc)],
                message="배치 소스 생성 중 오류가 발생했습니다.",
            )

        return BatchDevResult(
            batch_spec=spec,
            created_files=created_files,
            warnings=warnings,
            errors=[],
            message="배치 소스 초안이 generated 폴더에 생성되었습니다. 운영 반영 전 SQL/컬럼/검증조건을 반드시 검토하세요.",
        )
