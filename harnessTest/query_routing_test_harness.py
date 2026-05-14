from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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
    rewritten_question: str
    score: float
    pass_yn: str
    error: str


def _load_module(module_path: str):
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"llm 모듈 파일이 없습니다: {path}")
    spec = importlib.util.spec_from_file_location("llm_under_test", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"llm 모듈을 로드할 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return [None]
    if isinstance(value, list):
        return value
    return [value]


def _matches(expected: Any, actual: Any) -> bool:
    return actual in _as_list(expected)


def _answer_contains_all(expected_terms: Any, answer: str) -> bool:
    terms = expected_terms or []
    if isinstance(terms, str):
        terms = [terms]
    if not terms:
        return True
    answer_text = str(answer or "")
    return all(str(term) in answer_text for term in terms)


def _score_case(case: Dict[str, Any], result: Any) -> tuple[float, Dict[str, bool]]:
    checks: Dict[str, bool] = {}

    checks["system_id"] = _matches(case.get("expected_system_id"), getattr(result, "system_id", None))
    checks["intent"] = _matches(case.get("expected_intent"), getattr(result, "intent", None))
    checks["render_type"] = _matches(case.get("expected_render_type"), getattr(result, "render_type", None))
    checks["answer_contains"] = _answer_contains_all(case.get("expected_answer_contains"), getattr(result, "answer", ""))

    weights = {
        "system_id": 35,
        "intent": 35,
        "render_type": 15,
        "answer_contains": 15,
    }

    # expected 값이 아예 없는 항목은 평가에서 제외하고 가중치를 재분배한다.
    active_keys = []
    for key in weights:
        expected_key = {
            "system_id": "expected_system_id",
            "intent": "expected_intent",
            "render_type": "expected_render_type",
            "answer_contains": "expected_answer_contains",
        }[key]
        if expected_key in case:
            active_keys.append(key)

    active_weight_sum = sum(weights[key] for key in active_keys) or 1
    score = sum(weights[key] for key in active_keys if checks[key]) / active_weight_sum * 100
    return round(score, 2), checks


def _load_cases(cases_path: str) -> List[Dict[str, Any]]:
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("테스트 케이스 JSON은 list 형식이어야 합니다.")
    return data


def _write_csv(path: str, rows: Sequence[CaseResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "case_id", "category", "question", "expected_system_id", "actual_system_id",
        "expected_intent", "actual_intent", "expected_render_type", "actual_render_type",
        "expected_answer_contains", "rewritten_question", "score", "pass_yn", "error",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_tests(args: argparse.Namespace) -> Dict[str, Any]:
    llm_module = _load_module(args.llm_path)
    agent = llm_module.HandoverAgent(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    cases = _load_cases(args.cases)
    rows: List[CaseResult] = []
    detail_results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = str(case.get("case_id", ""))
        category = str(case.get("category", ""))
        question = str(case.get("question", ""))
        chat_history = case.get("chat_history") or []
        try:
            result = agent.answer_question(question=question, chat_history=chat_history, top_k=args.top_k)
            score, checks = _score_case(case, result)
            pass_yn = "Y" if score >= args.pass_score else "N"
            error = ""
            row = CaseResult(
                case_id=case_id,
                category=category,
                question=question,
                expected_system_id=case.get("expected_system_id"),
                actual_system_id=getattr(result, "system_id", None),
                expected_intent=case.get("expected_intent"),
                actual_intent=getattr(result, "intent", None),
                expected_render_type=case.get("expected_render_type"),
                actual_render_type=getattr(result, "render_type", None),
                expected_answer_contains=case.get("expected_answer_contains"),
                rewritten_question=getattr(result, "rewritten_question", ""),
                score=score,
                pass_yn=pass_yn,
                error=error,
            )
            detail_results.append({
                **asdict(row),
                "checks": checks,
                "answer_preview": str(getattr(result, "answer", ""))[:500],
                "debug_logs": getattr(result, "debug_logs", []),
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

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] CSV 결과: {args.output_csv}")
    print(f"[INFO] JSON 결과: {args.output_json}")

    if args.fail_under is not None and avg_score < args.fail_under:
        raise SystemExit(f"평균 점수 {avg_score}가 기준 {args.fail_under} 미만입니다.")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="업무 인수인계 에이전트 질의 라우팅 테스트 하네스")
    parser.add_argument("--llm-path", default="llm.py", help="테스트할 llm.py 경로")
    parser.add_argument("--json-path", default="ingest/handover_improved.json", help="업무 JSON 경로")
    parser.add_argument("--persist-dir", default="./chroma", help="Chroma persist dir")
    parser.add_argument("--collection", default="handover_agent", help="Chroma collection name")
    parser.add_argument("--cases", default="query_routing_test_cases.json", help="테스트 케이스 JSON")
    parser.add_argument("--output-csv", default="harnessTest/query_routing_test_results.csv", help="CSV 결과 파일")
    parser.add_argument("--output-json", default="harnessTest/query_routing_test_results.json", help="상세 JSON 결과 파일")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pass-score", type=float, default=80.0, help="케이스별 통과 점수")
    parser.add_argument("--fail-under", type=float, default=None, help="평균 점수가 이 값 미만이면 종료코드 실패")
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
