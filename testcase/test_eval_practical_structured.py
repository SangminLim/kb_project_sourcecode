from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def contains_all(text: str, keywords: List[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def contains_none(text: str, keywords: List[str]) -> bool:
    return all(keyword not in text for keyword in keywords)


def group_match(text: str, groups: List[List[str]]) -> Dict[str, Any]:
    matched_groups = 0
    details = []
    for group in groups:
        ok = any(candidate in text for candidate in group)
        matched_groups += int(ok)
        details.append({"group": group, "matched": ok})
    return {
        "matched_groups": matched_groups,
        "total_groups": len(groups),
        "ok": matched_groups == len(groups),
        "details": details,
    }


def rewritten_match(result_rewritten: str, tc: Dict[str, Any]) -> Dict[str, Any]:
    expected_exact = tc.get("expected_rewritten_question")
    expected_contains = tc.get("expected_rewritten_contains", [])

    exact_ok = True if not expected_exact else (result_rewritten == expected_exact)
    contains_ok = True if not expected_contains else all(token in result_rewritten for token in expected_contains)

    return {
        "exact_ok": exact_ok,
        "contains_ok": contains_ok,
        "ok": exact_ok and contains_ok,
    }


def structured_to_text(result: Any) -> str:
    parts: List[str] = []
    structured_data = getattr(result, "structured_data", None) or {}
    intent = getattr(result, "intent", "")

    if intent == "overview" and structured_data:
        overview = structured_data.get("overview", structured_data)
        for key in ["title", "summary", "content"]:
            value = overview.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        for key in ["input_data", "target_transactions", "exclusions", "outputs", "key_points"]:
            values = overview.get(key, [])
            if isinstance(values, list):
                parts.extend([str(v) for v in values if str(v).strip()])

    elif intent == "batch_process" and structured_data:
        batch_process = structured_data.get("batch_process", structured_data)
        title = batch_process.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(title)
        steps = batch_process.get("steps", [])
        for step in steps:
            step_num = step.get("step")
            step_name = step.get("name", "")
            execution = step.get("execution", "")
            execution_kr = "병렬" if execution == "parallel" else "순차" if execution == "sequential" else execution
            if step_num:
                parts.append(f"{step_num}단계 {step_name}({execution_kr})")
            description = step.get("description")
            if isinstance(description, str) and description.strip():
                parts.append(description)
            for key_job in step.get("key_jobs", []):
                parts.append(str(key_job))
            for job in step.get("jobs", []):
                job_id = job.get("job_id")
                job_desc = job.get("description")
                if job_id:
                    parts.append(str(job_id))
                if job_desc:
                    parts.append(str(job_desc))

    answer = getattr(result, "answer", "") or ""
    if answer.strip():
        parts.append(answer)

    return "\n".join(parts)


def evaluate_answer_text(text_for_eval: str, tc: Dict[str, Any]) -> Dict[str, Any]:
    required_all = tc.get("required_all", [])
    required_any_groups = tc.get("required_any_groups", [])
    forbidden_keywords = tc.get("forbidden_keywords", [])

    required_all_ok = contains_all(text_for_eval, required_all)
    group_result = group_match(text_for_eval, required_any_groups)
    forbidden_ok = contains_none(text_for_eval, forbidden_keywords)

    return {
        "required_all_ok": required_all_ok,
        "group_ok": group_result["ok"],
        "group_result": group_result,
        "forbidden_ok": forbidden_ok,
        "ok": required_all_ok and group_result["ok"] and forbidden_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="실무형 HandoverAgent 평가 스크립트 (structured_data 반영)")
    parser.add_argument("--dataset", default="./eval_dataset_practical_structured.json")
    parser.add_argument("--json_path", default="./handover_improved.json")
    parser.add_argument("--persist_dir", default="./chroma")
    parser.add_argument("--collection", default="handover_agent")
    parser.add_argument("--output", default="./eval_result_practical_structured.json")
    parser.add_argument("--agent_module", default="llm", help="HandoverAgent가 들어있는 모듈명")
    args = parser.parse_args()

    module = __import__(args.agent_module, fromlist=["HandoverAgent"])
    HandoverAgent = getattr(module, "HandoverAgent")

    dataset = load_dataset(args.dataset)

    agent = HandoverAgent(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    results: List[Dict[str, Any]] = []
    counters = {
        "system": 0,
        "intent": 0,
        "render": 0,
        "rewritten": 0,
        "answer": 0,
        "execution": 0,
        "final": 0,
    }

    for tc in dataset:
        result = agent.answer_question(question=tc["question"], chat_history=[])

        system_ok = result.system_id == tc.get("expected_system_id")
        intent_ok = result.intent == tc.get("expected_intent")
        render_ok = result.render_type == tc.get("expected_render_type")

        rewritten_result = rewritten_match(result.rewritten_question or "", tc)

        text_for_eval = structured_to_text(result)
        answer_eval = evaluate_answer_text(text_for_eval, tc)

        execution_checks = []
        if tc.get("require_graph_data"):
            execution_checks.append(bool(getattr(result, "graph_data", None)))
        if tc.get("require_query_meta"):
            execution_checks.append(bool(getattr(result, "query_meta", None)))
        execution_ok = all(execution_checks) if execution_checks else True

        final_ok = all([
            system_ok,
            intent_ok,
            render_ok,
            rewritten_result["ok"],
            answer_eval["ok"],
            execution_ok,
        ])

        counters["system"] += int(system_ok)
        counters["intent"] += int(intent_ok)
        counters["render"] += int(render_ok)
        counters["rewritten"] += int(rewritten_result["ok"])
        counters["answer"] += int(answer_eval["ok"])
        counters["execution"] += int(execution_ok)
        counters["final"] += int(final_ok)

        results.append({
            "question": tc["question"],
            "result": {
                "system_id": result.system_id,
                "intent": result.intent,
                "render_type": result.render_type,
                "rewritten_question": result.rewritten_question,
                "answer": result.answer,
                "text_for_eval": text_for_eval,
                "has_graph_data": bool(getattr(result, "graph_data", None)),
                "has_query_meta": bool(getattr(result, "query_meta", None)),
            },
            "evaluation": {
                "system_ok": system_ok,
                "intent_ok": intent_ok,
                "render_ok": render_ok,
                "rewritten": rewritten_result,
                "answer_eval": answer_eval,
                "execution_ok": execution_ok,
                "final_ok": final_ok,
            },
            "expected": tc,
        })

    total = len(results)
    summary = {
        "total": total,
        "final_pass": counters["final"],
        "final_accuracy": round((counters["final"] / total) * 100, 2) if total else 0.0,
        "system_accuracy": round((counters["system"] / total) * 100, 2) if total else 0.0,
        "intent_accuracy": round((counters["intent"] / total) * 100, 2) if total else 0.0,
        "render_accuracy": round((counters["render"] / total) * 100, 2) if total else 0.0,
        "rewritten_accuracy": round((counters["rewritten"] / total) * 100, 2) if total else 0.0,
        "answer_accuracy": round((counters["answer"] / total) * 100, 2) if total else 0.0,
        "execution_accuracy": round((counters["execution"] / total) * 100, 2) if total else 0.0,
    }

    fail_cases = [r for r in results if not r["evaluation"]["final_ok"]]
    report = {"summary": summary, "fail_cases": fail_cases, "results": results}

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== PRACTICAL STRUCTURED SUMMARY =====")
    print(f"TOTAL: {summary['total']}")
    print(f"FINAL PASS: {summary['final_pass']}")
    print(f"FINAL ACCURACY: {summary['final_accuracy']}%")
    print(f"SYSTEM ACCURACY: {summary['system_accuracy']}%")
    print(f"INTENT ACCURACY: {summary['intent_accuracy']}%")
    print(f"RENDER ACCURACY: {summary['render_accuracy']}%")
    print(f"REWRITTEN ACCURACY: {summary['rewritten_accuracy']}%")
    print(f"ANSWER ACCURACY: {summary['answer_accuracy']}%")
    print(f"EXECUTION ACCURACY: {summary['execution_accuracy']}%")

    print("\n===== FAIL CASES =====")
    if not fail_cases:
        print("없음")
    else:
        for item in fail_cases:
            q = item["question"]
            ev = item["evaluation"]
            print(f"Q: {q}")
            print(
                f"  system_ok={ev['system_ok']}, intent_ok={ev['intent_ok']}, render_ok={ev['render_ok']}, "
                f"rewritten_ok={ev['rewritten']['ok']}, answer_ok={ev['answer_eval']['ok']}, execution_ok={ev['execution_ok']}"
            )
            print()

    print(f"\n상세 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
