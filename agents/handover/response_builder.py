from __future__ import annotations

import re
from typing import Any, Dict, List

def build_graph_answer(graph_data: Dict[str, Any], intent: str) -> str:
    summary = graph_data.get("summary")
    if summary:
        return summary
    if intent == "batch_flow":
        return "배치 흐름의 시작, 핵심 처리, 종료 단계를 아래 흐름도에서 확인할 수 있습니다."
    return "원천 테이블부터 결과 테이블까지의 데이터 흐름을 아래 리니지에서 확인할 수 있습니다."


def build_chart_answer(query_meta: Dict[str, Any]) -> str:
    return f"{query_meta.get('title', '조회 결과')}를 시각화했습니다. 월별 추이와 분포를 바로 확인할 수 있습니다."


def build_table_answer(query_meta: Dict[str, Any]) -> str:
    return f"{query_meta.get('title', '조회 결과')}을 표 형태로 정리했습니다. 아래에서 세부 항목을 확인할 수 있습니다."


def build_overview_fallback(overview: Dict[str, Any]) -> str:
    """overview는 화면에서 구조화 렌더링하므로 answer에는 짧은 대표 문장만 담는다."""
    return str(
        overview.get("summary")
        or overview.get("content")
        or overview.get("title")
        or "업무 개요 정보가 없습니다."
    ).strip()


def _format_execution_label(execution: str) -> str:
    """배치 실행 방식을 사용자에게 보여줄 한글 라벨로 변환한다.

    실행 방식 값은 JSON 메타에서 오므로 새 값이 추가되어도 원문을 보존한다.
    """
    labels = {
        "parallel": "병렬",
        "sequential": "순차",
    }
    return labels.get(str(execution or "").strip().lower(), str(execution or "").strip())


def _format_list_inline(values: Any) -> str:
    """list/string 값을 화면용 한 줄 문자열로 변환한다."""
    if values is None:
        return ""
    if isinstance(values, list):
        return ", ".join([str(v).strip() for v in values if str(v).strip()])
    return str(values).strip()


def _format_duration_sec(value: Any) -> str:
    """초 단위 평균 수행시간을 보기 좋게 표시한다."""
    if value is None or str(value).strip() == "":
        return ""
    try:
        seconds = int(float(value))
    except Exception:
        return str(value).strip()

    if seconds < 60:
        return f"{seconds}초"

    minutes = seconds // 60
    remain = seconds % 60
    if remain:
        return f"{minutes}분 {remain}초"
    return f"{minutes}분"


def build_batch_process_fallback(batch_process: Dict[str, Any]) -> str:
    """배치 프로세스를 중복 없이 구조화해서 생성한다.

    원칙:
    - 하드코딩된 배치명/단계명 없이 JSON steps/jobs/key_jobs 기반으로 생성한다.
    - 같은 내용을 문장형 설명과 STEP 상세로 두 번 반복하지 않는다.
    - 한 줄 흐름 -> STEP 상세 -> 핵심 배치 순서로만 출력한다.
    - batch_process.jobs[]에 운영 메타가 있으면 함께 출력한다.
    - LangChain 실패 시 fallback으로 사용해도 그대로 사용자에게 보여줄 수 있는 품질을 유지한다.
    """
    title = batch_process.get("title", "배치 프로세스")
    steps = batch_process.get("steps", [])
    if not steps:
        return str(title or "배치 프로세스 정보가 없습니다.")

    lines: List[str] = [f"📌 {title}", ""]

    step_names = [str(step.get("name", "")).strip() for step in steps if str(step.get("name", "")).strip()]
    if step_names:
        lines.append("🔹 한 줄 흐름")
        lines.append(" → ".join(step_names))
        lines.append("")

    lines.append("🔹 단계별 배치 프로세스")
    for step in steps:
        step_no = step.get("step", "")
        step_name = str(step.get("name", "")).strip()
        execution_label = _format_execution_label(str(step.get("execution", "")))
        description = str(step.get("description", "")).strip()

        header_parts = [f"STEP {step_no}" if step_no != "" else "STEP", step_name]
        header = ". ".join([part for part in header_parts if part])
        if execution_label:
            header = f"{header} ({execution_label})"
        lines.append(f"{header}")

        if description:
            lines.append(f"👉 {description}")

        for job in step.get("jobs", []) or []:
            job_id = str(job.get("job_id", "")).strip()
            job_desc = str(job.get("description", "")).strip()
            job_name = str(job.get("job_name", "")).strip()

            display_name = job_name if job_name and job_name != job_id else ""
            if job_id and job_desc:
                if display_name:
                    lines.append(f"- {job_id} ({display_name}): {job_desc}")
                else:
                    lines.append(f"- {job_id}: {job_desc}")
            elif job_id:
                lines.append(f"- {job_id}" + (f" ({display_name})" if display_name else ""))
            elif job_desc:
                lines.append(f"- {job_desc}")

            operation_lines: List[str] = []
            schedule_type = str(job.get("schedule_type", "")).strip()
            execution_time = str(job.get("execution_time", "")).strip()
            avg_duration = _format_duration_sec(job.get("avg_duration_sec"))
            batch_file = str(job.get("batch_file", "")).strip()
            program_name = str(job.get("program_name", "")).strip()
            owner_team = str(job.get("owner_team", "")).strip()
            retry_count = str(job.get("retry_count", "")).strip()
            upstream_jobs = _format_list_inline(job.get("upstream_jobs"))
            downstream_jobs = _format_list_inline(job.get("downstream_jobs"))
            failure_action = str(job.get("failure_action", "")).strip()

            if schedule_type:
                operation_lines.append(f"  - 실행주기: {schedule_type}")
            if execution_time:
                operation_lines.append(f"  - 실행시간: {execution_time}")
            if avg_duration:
                operation_lines.append(f"  - 평균수행시간: {avg_duration}")
            if batch_file:
                operation_lines.append(f"  - 실행배치파일: {batch_file}")
            if program_name:
                operation_lines.append(f"  - 프로그램명: {program_name}")
            if owner_team:
                operation_lines.append(f"  - 담당팀: {owner_team}")
            if retry_count:
                operation_lines.append(f"  - 재시도횟수: {retry_count}")
            if upstream_jobs:
                operation_lines.append(f"  - 선행배치: {upstream_jobs}")
            if downstream_jobs:
                operation_lines.append(f"  - 후행배치: {downstream_jobs}")
            if failure_action:
                operation_lines.append(f"  - 장애조치방법: {failure_action}")

            lines.extend(operation_lines)

    key_jobs: List[str] = []
    for step in steps:
        for job_id in step.get("key_jobs", []) or []:
            job_id_str = str(job_id).strip()
            if job_id_str and job_id_str not in key_jobs:
                key_jobs.append(job_id_str)

    if key_jobs:
        lines.append("⭐ 핵심 배치")
        for job_id in key_jobs:
            lines.append(f"- {job_id}")

    return "".join(lines).strip()


def remove_repeated_step_sections(answer: str) -> str:
    """LLM이 문장형 단계 설명 뒤에 STEP 상세를 반복 출력한 경우 앞부분을 제거한다.

    하드코딩된 업무/배치명 기준이 아니라 STEP 패턴 반복 여부만 본다.
    이미 깔끔한 답변이면 원문을 그대로 반환한다.
    """
    text = str(answer or "").strip()
    if not text:
        return text

    # 'STEP 1.' 형태 상세 구간이 있으면 그 앞의 장황한 '1단계에서는...' 문장형 반복을 제거한다.
    step_match = re.search(r"(?im)^\s*STEP\s*1\s*[\.).]", text)
    if not step_match:
        return text

    prefix = text[: step_match.start()].strip()
    suffix = text[step_match.start() :].strip()

    # 앞부분에 단계형 문장과 배치명이 이미 있고, 뒤에 STEP 상세가 다시 있으면 중복으로 판단한다.
    prefix_has_stage_text = bool(re.search(r"[123]\s*단계|Step\s*[123]", prefix, flags=re.IGNORECASE))
    prefix_has_batch_id = bool(re.search(r"BATCH_\d+_", prefix))
    suffix_has_batch_id = bool(re.search(r"BATCH_\d+_", suffix))

    if prefix_has_stage_text and prefix_has_batch_id and suffix_has_batch_id:
        # 제목/핵심 배치/핵심 흐름 같은 짧은 헤더는 유지하고, 긴 문장형 단계 설명만 제거한다.
        header_lines: List[str] = []
        for line in prefix.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"[123]\s*단계|Step\s*[123]", stripped, flags=re.IGNORECASE):
                break
            header_lines.append(stripped)
        if header_lines:
            return "\n".join(header_lines + ["", suffix]).strip()
        return suffix

    return text
