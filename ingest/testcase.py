"""
testcase.py

목적
- 질문 1건을 빠르게 점검하는 CLI
- chat history 포함
- debug 출력 가능
- 경진대회 시연용으로 모델링 포인트를 보기 쉽게 출력

실행 예시
python testcase.py --json_path ./handover.json --question "BBBK증권 흐름도 보여줘" --debug
python testcase.py --json_path ./handover.json --question "월별 금액 그래프 보여줘" --history_file ./chat_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from llm import HandoverAgent


def load_history(history_file: str) -> List[Dict[str, str]]:
    path = Path(history_file)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history_file: str, chat_history: List[Dict[str, str]]) -> None:
    path = Path(history_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)


def print_sources(sources: List[Dict[str, str]]) -> None:
    print("[DEBUG] sources")
    for idx, item in enumerate(sources, start=1):
        print(
            f"doc[{idx}] system={item.get('system_name')} "
            f"section={item.get('section')} title={item.get('title')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="handover JSON 경로")
    parser.add_argument("--persist_dir", default="./chroma")
    parser.add_argument("--collection", default="handover_agent")
    parser.add_argument("--question", required=True)
    parser.add_argument("--history_file", default="./chat_history.json")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chat_history = load_history(args.history_file)
    agent = HandoverAgent(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    result = agent.answer_question(
        question=args.question,
        chat_history=chat_history,
    )

    if args.debug:
        print("[DEBUG] query_info")
        print(
            json.dumps(
                {
                    "original_question": result.original_question,
                    "normalized_question": result.normalized_question,
                    "rewritten_question": result.rewritten_question,
                    "system_id": result.system_id,
                    "intent": result.intent,
                    "render_type": result.render_type,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print_sources(result.sources)

        if result.graph_data:
            print("[DEBUG] graph_data")
            print(json.dumps(result.graph_data, ensure_ascii=False, indent=2))

        if result.query_meta:
            print("[DEBUG] query_meta")
            print(json.dumps(result.query_meta, ensure_ascii=False, indent=2))

    print("\n[답변]")
    print(result.answer)

    chat_history.append({"role": "user", "content": args.question})
    chat_history.append({"role": "assistant", "content": result.answer})
    save_history(args.history_file, chat_history)


if __name__ == "__main__":
    main()
