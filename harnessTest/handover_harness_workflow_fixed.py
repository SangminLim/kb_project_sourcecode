from __future__ import annotations

import argparse
import ast
import csv
import importlib
import importlib.util
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str

    expected_system_id: Any
    actual_system_id: Any
    expected_intent: Any
    actual_intent: Any
    expected_render_type: Any
    actual_render_type: Any
    expected_answer_contains: Any

    # 확장 평가 항목: 값이 테스트케이스에 있을 때만 점수에 반영한다.
    expected_workflow_steps: Any
    actual_workflow_steps: Any
    expected_tool_calls: Any
    actual_tool_calls: Any
    expected_reasoning_contains: Any

    rewritten_question: str
    score: float
    pass_yn: str
    error: str


def _ensure_parent_dir(path: str) -> None:
    """출력 파일의 부모 디렉터리를 자동 생성한다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_module_name(module_path: str) -> str | None:
    """
    agents/handover/agent.py 또는 agents\handover\agent.py 형태를
    agents.handover.agent 모듈명으로 변환한다.

    상대 import(from .config import ...)가 있는 패키지 파일은
    파일 직접 로딩(spec_from_file_location)으로 실행하면 실패하므로,
    가능하면 importlib.import_module 방식으로 로드한다.
    """
    value = str(module_path or "").strip().strip('"').strip("'")
    if not value:
        return None

    # 이미 agents.handover.agent 같은 모듈명으로 들어온 경우
    if value.endswith(".py") is False and "/" not in value and "\\" not in value:
        return value

    normalized = value.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]

    path = Path(normalized)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return None

    parts = [part for part in path.parts if part not in (".", "")]
    if not parts:
        return None
    return ".".join(parts)


def _load_module(module_path: str):
    module_name = _to_module_name(module_path)

    # 1순위: 패키지 모듈 방식 로드
    # 예: agents.handover.agent
    # 이 방식이어야 agent.py 내부의 from .config import ... 상대 import가 정상 동작한다.
    if module_name:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            module = None
        except Exception as exc:
            raise ImportError(f"모듈을 로드할 수 없습니다: {module_name}") from exc
        else:
            if not hasattr(module, "HandoverAgent"):
                raise AttributeError(f"{module_name} 모듈에 HandoverAgent 클래스가 없습니다.")
            return module

    # 2순위: 기존 방식 유지.
    # llm.py처럼 상대 import가 없는 단일 파일 테스트를 위해 fallback으로 남겨둔다.
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"llm 모듈 파일이 없습니다: {path}")

    spec = importlib.util.spec_from_file_location("llm_under_test", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"llm 모듈을 로드할 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_under_test"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "HandoverAgent"):
        raise AttributeError(f"{path} 파일에 HandoverAgent 클래스가 없습니다.")
    return module


def _load_workflow_runner(workflow_path: str, workflow_func: str):
    """Streamlit 화면에서 쓰는 handover workflow 실행 함수를 로드한다.

    기본값은 ui.workflow.handover_workflow.run_handover_graph 이다.
    프로젝트에서 파일명이 다르면 --workflow-path / --workflow-func 로 바꿀 수 있다.
    """
    module_name = _to_module_name(workflow_path) or str(workflow_path or "").strip()
    if not module_name:
        raise ImportError("workflow_path가 비어 있습니다.")

    module = importlib.import_module(module_name)
    runner = getattr(module, workflow_func, None)
    if runner is None or not callable(runner):
        raise AttributeError(f"{module_name} 모듈에 callable {workflow_func} 함수가 없습니다.")
    return runner


def _build_direct_agent(args: argparse.Namespace):
    """기존 HandoverAgent 직접 호출용 agent를 생성한다."""
    llm_module = _load_module(args.llm_path)
    return llm_module.HandoverAgent(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )


def _answer_with_direct_agent(agent: Any, question: str, chat_history: List[Dict[str, str]], args: argparse.Namespace) -> Any:
    return agent.answer_question(question=question, chat_history=chat_history, top_k=args.top_k)


def _answer_with_workflow(case: Dict[str, Any], args: argparse.Namespace) -> Any:
    """Streamlit과 동일한 workflow 진입점으로 테스트 케이스를 실행한다."""
    runner = _load_workflow_runner(args.workflow_path, args.workflow_func)
    question = str(case.get("question", ""))
    chat_history = case.get("chat_history") or []
    force_sql_analysis = bool(case.get("force_sql_analysis", False))
    return runner(
        user_question=question,
        chat_history=chat_history,
        force_sql_analysis=force_sql_analysis,
    )


def _answer_case(case: Dict[str, Any], args: argparse.Namespace, direct_agent_holder: Dict[str, Any]) -> Any:
    """테스트 케이스를 실행한다.

    - runner=auto 또는 workflow: 실제 화면과 같은 run_handover_graph를 우선 사용한다.
    - workflow 로드/실행 실패 시 auto는 기존 direct 방식으로 fallback한다.
    - runner=workflow는 실패를 그대로 노출해서 workflow 연결 문제를 빨리 찾는다.
    - runner=direct는 기존 하네스와 동일하게 HandoverAgent.answer_question을 직접 호출한다.
    """
    runner_mode = str(getattr(args, "runner", "auto") or "auto").strip().lower()
    question = str(case.get("question", ""))
    chat_history = case.get("chat_history") or []

    if runner_mode in {"auto", "workflow"}:
        try:
            return _answer_with_workflow(case, args)
        except Exception:
            if runner_mode == "workflow":
                raise
            # auto 모드에서는 Streamlit workflow import 문제가 있어도 기존 direct 테스트를 유지한다.

    if direct_agent_holder.get("agent") is None:
        direct_agent_holder["agent"] = _build_direct_agent(args)
    return _answer_with_direct_agent(direct_agent_holder["agent"], question, chat_history, args)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return [None]
    if isinstance(value, list):
        return value
    return [value]


def _matches(expected: Any, actual: Any) -> bool:
    return actual in _as_list(expected)


def _contains_all(expected_values: Any, actual_values: Any) -> bool:
    """
    expected_values가 모두 actual_values 안에 존재하는지 확인한다.
    actual_values가 문자열이면 부분 문자열 포함 여부로 판단하고,
    리스트/튜플/셋이면 항목 포함 여부로 판단한다.
    """
    expected = expected_values or []
    if isinstance(expected, str):
        expected = [expected]
    if not expected:
        return True

    if actual_values is None:
        return False

    if isinstance(actual_values, str):
        actual_text = actual_values
        return all(str(item) in actual_text for item in expected)

    if not isinstance(actual_values, (list, tuple, set)):
        actual_values = [actual_values]

    actual_text_values = [str(item) for item in actual_values]
    return all(str(item) in actual_text_values for item in expected)


def _answer_contains_all(expected_terms: Any, answer: str) -> bool:
    return _contains_all(expected_terms, str(answer or ""))


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_debug_values(result: Any, keys: Sequence[str]) -> List[str]:
    """
    result.debug_logs 안에 workflow/tool/retrieval 정보가 dict 또는 문자열로 들어있는 경우
    하네스 평가에서 사용할 수 있도록 평탄화한다.

    LangGraph/Planner 로그 예시도 함께 해석한다.
    - [GRAPH 1] classify intent=...
    - [GRAPH 2] route=...
    - [PLAN 2] selected_steps = ['prepare_realtime_query', ...]
    """
    debug_logs = _get_attr_or_key(result, "debug_logs", []) or []
    values: List[str] = []

    def add_value(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_value(item)
            return
        text = str(value).strip()
        if text:
            values.append(text)

    for log in debug_logs:
        if isinstance(log, dict):
            for key in keys:
                add_value(log.get(key))
            continue

        text = str(log)

        # Planner selected_steps 리스트를 실제 step 목록으로 분해한다.
        if "selected_steps" in text:
            raw = text.split("=", 1)[1].strip() if "=" in text else ""
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple, set)):
                    for item in parsed:
                        add_value(item)
            except Exception:
                add_value(text)

        # LangGraph trace를 관측 가능한 workflow step으로 변환한다.
        if text.startswith("[GRAPH"):
            if "classify" in text:
                add_value("classify")
                add_value("classify_intent")
            if "route=" in text:
                add_value("route")
            if "executed=handover_agent" in text:
                add_value("handover_agent")
            if "executed=batch_development" in text:
                add_value("batch_development")
            if "executed=sql_analysis" in text:
                add_value("sql_analysis")
            if "executed=general_fallback" in text:
                add_value("general_fallback")
            if "batch_validation" in text:
                add_value("batch_validation")
            if "sql_improvement" in text:
                add_value("sql_improvement")
            if "finalized" in text:
                add_value("finalize")

        for key in keys:
            if key in text:
                add_value(text)

    # 순서 유지 중복 제거
    result_values: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result_values.append(value)
    return result_values


def _derive_workflow_steps(result: Any) -> List[str]:
    """
    AgentResult에 workflow_steps가 있으면 그대로 사용하고,
    없으면 실제 결과(system_id/intent/render_type)를 기준으로 관측 가능한 처리 단계를 보수적으로 추정한다.
    - 이 값은 내부 Chain-of-Thought가 아니라 테스트 관측용 처리 단계명이다.
    """
    explicit = _get_attr_or_key(result, "workflow_steps", None)
    if explicit:
        return list(explicit) if isinstance(explicit, (list, tuple, set)) else [str(explicit)]

    debug_steps = _extract_debug_values(result, ["workflow_step", "workflow_steps", "node", "step"])
    if debug_steps:
        return debug_steps

    intent = _get_attr_or_key(result, "intent", None)
    system_id = _get_attr_or_key(result, "system_id", None)
    render_type = _get_attr_or_key(result, "render_type", None)

    steps = ["classify_intent"]
    if system_id is not None:
        steps.append("resolve_system")

    if intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
        steps.append("retrieve_context")
    elif intent == "billing_monthly_amount":
        steps.append("query_billing_data")
        steps.append("analyze_billing_trend")
    elif intent == "today_incidents":
        steps.append("query_incidents")
        steps.append("analyze_incident_action")
    elif intent == "unknown_system":
        steps.append("handle_unknown_system")
    elif intent in {"out_of_scope", "default"}:
        steps.append("handle_out_of_scope")

    if render_type:
        steps.append(f"render_{render_type}")

    return steps


def _derive_tool_calls(result: Any) -> List[str]:
    """
    AgentResult에 tool_calls가 있으면 그대로 사용하고,
    없으면 intent/render_type 기준으로 외부 조회/렌더링 도구 호출을 관측값으로 추정한다.
    """
    explicit = _get_attr_or_key(result, "tool_calls", None)
    if explicit:
        return list(explicit) if isinstance(explicit, (list, tuple, set)) else [str(explicit)]

    debug_tools = _extract_debug_values(result, ["tool_call", "tool_calls", "tool", "retriever"])
    if debug_tools:
        return debug_tools

    intent = _get_attr_or_key(result, "intent", None)
    render_type = _get_attr_or_key(result, "render_type", None)

    tools: List[str] = []
    if intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
        tools.append("vector_retriever")
    if intent == "billing_monthly_amount":
        tools.append("billing_query")
    if intent == "today_incidents":
        tools.append("incident_query")
    if render_type == "graph":
        tools.append("graph_renderer")
    elif render_type == "chart":
        tools.append("chart_renderer")
    elif render_type == "table":
        tools.append("table_renderer")

    return tools


def _score_case(case: Dict[str, Any], result: Any) -> tuple[float, Dict[str, bool], Dict[str, Any]]:
    actual_workflow_steps = _derive_workflow_steps(result)
    actual_tool_calls = _derive_tool_calls(result)
    answer = str(_get_attr_or_key(result, "answer", "") or "")

    checks: Dict[str, bool] = {}
    checks["system_id"] = _matches(case.get("expected_system_id"), _get_attr_or_key(result, "system_id", None))
    checks["intent"] = _matches(case.get("expected_intent"), _get_attr_or_key(result, "intent", None))
    checks["render_type"] = _matches(case.get("expected_render_type"), _get_attr_or_key(result, "render_type", None))
    checks["answer_contains"] = _answer_contains_all(case.get("expected_answer_contains"), answer)

    # 확장 검증 항목
    checks["workflow_steps"] = _contains_all(case.get("expected_workflow_steps"), actual_workflow_steps)
    checks["tool_calls"] = _contains_all(case.get("expected_tool_calls"), actual_tool_calls)
    checks["reasoning_contains"] = _answer_contains_all(case.get("expected_reasoning_contains"), answer)

    # 기본 평가는 기존 비중을 유지하고, 확장 항목은 보조 검증으로 반영한다.
    weights = {
        "system_id": 25,
        "intent": 25,
        "render_type": 15,
        "answer_contains": 15,
        "workflow_steps": 10,
        "tool_calls": 5,
        "reasoning_contains": 5,
    }

    expected_key_map = {
        "system_id": "expected_system_id",
        "intent": "expected_intent",
        "render_type": "expected_render_type",
        "answer_contains": "expected_answer_contains",
        "workflow_steps": "expected_workflow_steps",
        "tool_calls": "expected_tool_calls",
        "reasoning_contains": "expected_reasoning_contains",
    }

    # expected 값이 아예 없는 항목은 평가에서 제외하고 가중치를 재분배한다.
    active_keys = [key for key in weights if expected_key_map[key] in case]

    active_weight_sum = sum(weights[key] for key in active_keys) or 1
    score = sum(weights[key] for key in active_keys if checks[key]) / active_weight_sum * 100

    observed = {
        "actual_workflow_steps": actual_workflow_steps,
        "actual_tool_calls": actual_tool_calls,
    }
    return round(score, 2), checks, observed


def _load_cases(cases_path: str) -> List[Dict[str, Any]]:
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("테스트 케이스 JSON은 list 형식이어야 합니다.")
    return data


def _write_csv(path: str, rows: Sequence[CaseResult]) -> None:
    _ensure_parent_dir(path)

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "case_id", "category", "question",
        "expected_system_id", "actual_system_id",
        "expected_intent", "actual_intent",
        "expected_render_type", "actual_render_type",
        "expected_answer_contains",
        "expected_workflow_steps", "actual_workflow_steps",
        "expected_tool_calls", "actual_tool_calls",
        "expected_reasoning_contains",
        "rewritten_question", "score", "pass_yn", "error",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _format_failed_checks(checks: Any) -> str:
    """실패한 검증 항목만 보기 좋게 문자열로 변환한다."""
    if not isinstance(checks, dict):
        return "-"
    failed = [name for name, ok in checks.items() if ok is False]
    return " / ".join(failed) if failed else "-"


def print_summary_report(payload: Dict[str, Any], output_csv: str, output_json: str) -> None:
    """
    콘솔에는 사람이 읽기 쉬운 요약만 출력한다.
    상세 debug_logs는 JSON 파일에만 저장한다.
    """
    summary = payload.get("summary", {}) or {}
    results = payload.get("results", []) or []

    failed_cases = [row for row in results if row.get("pass_yn") == "N"]
    low_score_cases = [
        row for row in results
        if row.get("pass_yn") == "Y" and float(row.get("score") or 0) < 85
    ]

    print("\n" + "=" * 40)
    print(" Handover Harness Test Result")
    print("=" * 40)
    print(f"Total      : {summary.get('total', 0)}")
    print(f"Passed     : {summary.get('passed', 0)}")
    print(f"Failed     : {summary.get('failed', 0)}")
    print(f"Pass Rate  : {summary.get('pass_rate', 0.0)}%")
    print(f"Avg Score  : {summary.get('avg_score', 0.0)}")
    print(f"Pass Score : {summary.get('pass_score', 0.0)}")

    if failed_cases:
        print("\n[FAILED CASES]")
        for row in failed_cases:
            print(f"{row.get('case_id')} | {row.get('category')} | {row.get('score')}")
            print(f"- question : {row.get('question')}")
            print(f"- reason   : {_format_failed_checks(row.get('checks'))} 불일치")
            if row.get("error"):
                print(f"- error    : {row.get('error')}")
    else:
        print("\n[FAILED CASES]")
        print("- 없음")

    if low_score_cases:
        print("\n[LOW SCORE CASES]")
        for row in low_score_cases:
            print(f"{row.get('case_id')} | {row.get('category')} | {row.get('score')}")
    else:
        print("\n[LOW SCORE CASES]")
        print("- 없음")

    print("\n[OUTPUT FILES]")
    print(f"- CSV  : {output_csv}")
    print(f"- JSON : {output_json}")
    print("=" * 40)


def run_tests(args: argparse.Namespace) -> Dict[str, Any]:
    # 기본은 Streamlit 화면과 동일한 workflow(run_handover_graph)를 타도록 한다.
    # direct fallback이 필요할 때만 HandoverAgent를 지연 생성한다.
    direct_agent_holder: Dict[str, Any] = {"agent": None}

    cases = _load_cases(args.cases)
    rows: List[CaseResult] = []
    detail_results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = str(case.get("case_id", ""))
        category = str(case.get("category", ""))
        question = str(case.get("question", ""))
        chat_history = case.get("chat_history") or []
        try:
            result = _answer_case(case, args, direct_agent_holder)
            score, checks, observed = _score_case(case, result)
            pass_yn = "Y" if score >= args.pass_score else "N"
            error = ""
            row = CaseResult(
                case_id=case_id,
                category=category,
                question=question,
                expected_system_id=case.get("expected_system_id"),
                actual_system_id=_get_attr_or_key(result, "system_id", None),
                expected_intent=case.get("expected_intent"),
                actual_intent=_get_attr_or_key(result, "intent", None),
                expected_render_type=case.get("expected_render_type"),
                actual_render_type=_get_attr_or_key(result, "render_type", None),
                expected_answer_contains=case.get("expected_answer_contains"),
                expected_workflow_steps=case.get("expected_workflow_steps"),
                actual_workflow_steps=observed["actual_workflow_steps"],
                expected_tool_calls=case.get("expected_tool_calls"),
                actual_tool_calls=observed["actual_tool_calls"],
                expected_reasoning_contains=case.get("expected_reasoning_contains"),
                rewritten_question=_get_attr_or_key(result, "rewritten_question", ""),
                score=score,
                pass_yn=pass_yn,
                error=error,
            )
            detail_results.append({
                **asdict(row),
                "checks": checks,
                "answer_preview": str(_get_attr_or_key(result, "answer", ""))[:500],
                "debug_logs": _get_attr_or_key(result, "debug_logs", []),
            })
        except Exception as exc:
            row = CaseResult(
                case_id=case_id,
                category=category,
                question=question,
                expected_system_id=case.get("expected_system_id"),
                actual_system_id=None,
                expected_intent=case.get("expected_intent"),
                actual_intent=None,
                expected_render_type=case.get("expected_render_type"),
                actual_render_type=None,
                expected_answer_contains=case.get("expected_answer_contains"),
                expected_workflow_steps=case.get("expected_workflow_steps"),
                actual_workflow_steps=[],
                expected_tool_calls=case.get("expected_tool_calls"),
                actual_tool_calls=[],
                expected_reasoning_contains=case.get("expected_reasoning_contains"),
                rewritten_question="",
                score=0.0,
                pass_yn="N",
                error=f"{type(exc).__name__}: {exc}",
            )
            detail_results.append({
                **asdict(row),
                "traceback": traceback.format_exc(),
            })
        rows.append(row)

    total = len(rows)
    passed = sum(1 for row in rows if row.pass_yn == "Y")
    avg_score = round(sum(row.score for row in rows) / total, 2) if total else 0.0

    category_summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = category_summary.setdefault(row.category, {"total": 0, "passed": 0, "avg_score": 0.0, "score_sum": 0.0})
        bucket["total"] += 1
        bucket["passed"] += 1 if row.pass_yn == "Y" else 0
        bucket["score_sum"] += row.score
    for bucket in category_summary.values():
        bucket["avg_score"] = round(bucket["score_sum"] / bucket["total"], 2) if bucket["total"] else 0.0
        del bucket["score_sum"]

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 2) if total else 0.0,
        "avg_score": avg_score,
        "pass_score": args.pass_score,
        "category_summary": category_summary,
    }

    payload = {"summary": summary, "results": detail_results}
    _write_csv(args.output_csv, rows)
    _write_json(args.output_json, payload)

    print_summary_report(payload, args.output_csv, args.output_json)

    if args.fail_under is not None and avg_score < args.fail_under:
        raise SystemExit(f"평균 점수 {avg_score}가 기준 {args.fail_under} 미만입니다.")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="업무 인수인계 에이전트 질의/워크플로우 테스트 하네스")
    parser.add_argument("--llm-path", default="agents.handover.agent", help="테스트할 Agent 모듈 경로 또는 모듈명 예: agents.handover.agent")
    parser.add_argument("--json-path", default="ingest/handover.json", help="업무 JSON 경로")
    parser.add_argument("--persist-dir", default="./chroma", help="Chroma persist dir")
    parser.add_argument("--collection", default="handover_agent", help="Chroma collection name")
    parser.add_argument("--cases", default="query_routing_test_cases.json", help="테스트 케이스 JSON")
    parser.add_argument("--output-csv", default="harnessTest/query_routing_test_results.csv", help="CSV 결과 파일")
    parser.add_argument("--output-json", default="harnessTest/query_routing_test_results.json", help="상세 JSON 결과 파일")
    parser.add_argument("--runner", choices=["auto", "workflow", "direct"], default="auto", help="auto/workflow는 run_handover_graph를 우선 사용하고, direct는 기존 Agent 직접 호출을 사용")
    parser.add_argument("--workflow-path", default="ui.workflow.handover_workflow", help="workflow 모듈 경로 예: ui.workflow.handover_workflow")
    parser.add_argument("--workflow-func", default="run_handover_graph", help="workflow 실행 함수명")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pass-score", type=float, default=80.0, help="케이스별 통과 점수")
    parser.add_argument("--fail-under", type=float, default=None, help="평균 점수가 이 값 미만이면 종료코드 실패")
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
