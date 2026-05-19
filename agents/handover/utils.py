from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_input_typos(text: str) -> str:
    """Deprecated compatibility wrapper.

    오타 치환을 코드/JSON으로 관리하지 않는다.
    여기서는 최소 정리인 공백 정리만 수행하고, 실제 오타/구어체 보정은 LLM rewrite에서 처리한다.
    """
    return normalize_whitespace(text)


def history_to_text(chat_history: List[Dict[str, str]], max_turns: int = 4) -> str:
    recent = chat_history[-max_turns:]
    return "\n".join([f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent]).strip()


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """LLM 출력이 JSON이어야 하는 영역에서 사용할 수 있는 안전 parser.

    현재 일반 답변에는 적용하지 않는다. 배치 개발 요청 파싱처럼 JSON이 필요한 기능을
    추가할 때 재사용하도록 분리했다.
    """
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
