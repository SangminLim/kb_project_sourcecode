from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

MODULE_DIR = Path(__file__).resolve().parent

def _find_project_root(start_dir: Path) -> Path:
    """현재 파일 위치가 바뀌어도 프로젝트 루트를 안정적으로 찾는다.

    리팩토링 후 agents/handover/llm.py처럼 하위 패키지로 이동해도
    conf/system_registry.json은 프로젝트 루트의 conf/에서 읽어야 한다.
    PROJECT_ROOT 환경변수가 있으면 그 값을 우선 사용한다.
    """
    env_root = os.getenv("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root)

    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "conf" / "system_registry.json").exists():
            return candidate
        if (candidate / "conf").exists() and (candidate / "app.py").exists():
            return candidate

    # agents/handover/llm.py 기준으로는 parents[2]가 프로젝트 루트다.
    # 그래도 구조가 달라도 깨지지 않게 가능한 경우에만 사용한다.
    parents = list(start_dir.parents)
    return parents[1] if len(parents) >= 2 else start_dir


def _load_required_json_file(path: Path, config_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{config_name} 설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json_file(path: Path, config_name: str, default: Any) -> Any:
    """선택 설정 파일 로더.

    업무별 문장 템플릿/실시간 조회 intent처럼 운영 중 바뀔 수 있는 값은
    코드에 직접 박지 않고 conf JSON에서 읽는다. 파일이 없거나 형식이 틀려도
    서비스가 바로 중단되지 않도록 default를 반환한다.
    """
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_system_specs() -> List[Dict[str, Any]]:
    data = _load_required_json_file(SYSTEM_REGISTRY_PATH, "system_registry")
    if not isinstance(data, list) or not data:
        raise ValueError("system_registry.json 형식이 올바르지 않습니다. 비어 있지 않은 list여야 합니다.")
    return data


def load_question_replacements() -> Dict[str, str]:
    data = _load_required_json_file(QUESTION_DICTIONARY_PATH, "question_dictionary")
    if not isinstance(data, dict):
        raise ValueError("question_dictionary.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return {str(k): str(v) for k, v in data.items()}


def load_typo_normalization() -> Dict[str, str]:
    """Deprecated: typo_normalization.json은 더 이상 필수 설정이 아니다.

    실무 기준으로 오타/띄어쓰기/구어체 보정은 LLM rewrite가 담당한다.
    이 함수는 기존 환경 호환을 위해 파일이 있으면 읽되, 없으면 빈 dict를 반환한다.
    """
    if not TYPO_NORMALIZATION_PATH.exists():
        return {}
    data = _load_required_json_file(TYPO_NORMALIZATION_PATH, "typo_normalization")
    if not isinstance(data, dict):
        raise ValueError("typo_normalization.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return {str(k): str(v) for k, v in data.items()}


def load_intent_patterns() -> Dict[str, List[str]]:
    """conf/intent_registry.json에서 intent별 매칭 패턴을 읽는다.

    지원 형식:
    1) {"overview": {"patterns": ["개요"]}}
    2) {"overview": ["개요"]}
    """
    data = _load_required_json_file(INTENT_REGISTRY_PATH, "intent_registry")
    if not isinstance(data, dict) or not data:
        raise ValueError("intent_registry.json 형식이 올바르지 않습니다. 비어 있지 않은 dict여야 합니다.")

    loaded: Dict[str, List[str]] = {}
    for intent, value in data.items():
        if isinstance(value, dict):
            patterns = value.get("patterns", [])
        elif isinstance(value, list):
            patterns = value
        else:
            patterns = []
        loaded[str(intent)] = [str(item) for item in patterns if str(item).strip()]

    if not any(loaded.values()):
        raise ValueError("intent_registry.json에 사용할 수 있는 patterns가 없습니다.")
    return loaded


def load_prompt_templates() -> Dict[str, Any]:
    """conf/prompt_templates.json에서 intent별 prompt 설정을 읽는다.

    하위 호환:
    - 기존 형식: {"overview": "system prompt 문자열"}
    - 확장 형식: {"overview": {"system_prompt": "...", "answer_rules": [...]}}
    - 공통 답변 규칙: "_common_answer_rules": [...]
    """
    data = _load_required_json_file(PROMPT_TEMPLATE_PATH, "prompt_templates")
    if not isinstance(data, dict) or not data:
        raise ValueError("prompt_templates.json 형식이 올바르지 않습니다. 비어 있지 않은 dict여야 합니다.")

    loaded: Dict[str, Any] = {}
    for key, value in data.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, dict):
            loaded[key_text] = value
        elif isinstance(value, list):
            loaded[key_text] = [str(v) for v in value if str(v).strip()]
        elif str(value).strip():
            loaded[key_text] = str(value)

    if "default" not in loaded:
        raise ValueError("prompt_templates.json에는 반드시 default 프롬프트가 있어야 합니다.")
    return loaded


def load_few_shot_examples() -> List[Dict[str, str]]:
    """conf/few_shot_examples.json에서 질문 재작성 few-shot 예시를 읽는다."""
    data = _load_required_json_file(FEW_SHOT_PATH, "few_shot_examples")
    if not isinstance(data, list):
        raise ValueError("few_shot_examples.json 형식이 올바르지 않습니다. list여야 합니다.")

    loaded: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user and assistant:
            loaded.append({"user": user, "assistant": assistant})
    return loaded


def load_conversation_policy() -> Dict[str, Any]:
    """conf/conversation_policy.json에서 대화 정책을 읽는다.

    운영 설정으로 관리할 항목:
    - followup_signals
    - intent_conflict_keywords
    - system_required_intents
    - out_of_scope_message
    - missing_system_message
    """
    data = _load_required_json_file(CONVERSATION_POLICY_PATH, "conversation_policy")
    if not isinstance(data, dict):
        raise ValueError("conversation_policy.json 형식이 올바르지 않습니다. dict여야 합니다.")
    return data


def load_canonical_question_templates() -> Dict[str, str]:
    """intent별 표준 질문 템플릿을 conf/canonical_question_templates.json에서 읽는다."""
    data = _load_optional_json_file(
        CANONICAL_QUESTION_TEMPLATE_PATH,
        "canonical_question_templates",
        {},
    )
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}


def load_realtime_intent_registry() -> Dict[str, Dict[str, Any]]:
    """실시간 조회 intent와 query_id/render_type 매핑을 설정 파일에서 읽는다."""
    data = _load_optional_json_file(
        REALTIME_INTENT_REGISTRY_PATH,
        "realtime_intent_registry",
        {},
    )
    if not isinstance(data, dict):
        return {}

    registry: Dict[str, Dict[str, Any]] = {}
    for intent, value in data.items():
        if not isinstance(value, dict):
            continue
        intent_name = str(intent).strip()
        query_id = str(value.get("query_id") or intent_name).strip()
        render_type = str(value.get("render_type") or "table").strip()
        if not intent_name or not query_id:
            continue

        registry[intent_name] = {
            **value,
            "query_id": query_id,
            "render_type": render_type,
            "realtime_mode": value.get("realtime_mode"),
            "canonical_question": value.get("canonical_question"),
        }
    return registry


PROJECT_ROOT = _find_project_root(MODULE_DIR)
CONF_DIR = Path(os.getenv("CONF_DIR", str(PROJECT_ROOT / "conf")))

SYSTEM_REGISTRY_PATH = Path(os.getenv("SYSTEM_REGISTRY_PATH", str(CONF_DIR / "system_registry.json")))
QUESTION_DICTIONARY_PATH = Path(os.getenv("QUESTION_DICTIONARY_PATH", str(CONF_DIR / "question_dictionary.json")))
TYPO_NORMALIZATION_PATH = Path(os.getenv("TYPO_NORMALIZATION_PATH", str(CONF_DIR / "typo_normalization.json")))
INTENT_REGISTRY_PATH = Path(os.getenv("INTENT_REGISTRY_PATH", str(CONF_DIR / "intent_registry.json")))
PROMPT_TEMPLATE_PATH = Path(os.getenv("PROMPT_TEMPLATE_PATH", str(CONF_DIR / "prompt_templates.json")))
FEW_SHOT_PATH = Path(os.getenv("FEW_SHOT_PATH", str(CONF_DIR / "few_shot_examples.json")))
CONVERSATION_POLICY_PATH = Path(os.getenv("CONVERSATION_POLICY_PATH", str(CONF_DIR / "conversation_policy.json")))
CANONICAL_QUESTION_TEMPLATE_PATH = Path(os.getenv("CANONICAL_QUESTION_TEMPLATE_PATH", str(CONF_DIR / "canonical_question_templates.json")))
REALTIME_INTENT_REGISTRY_PATH = Path(os.getenv("REALTIME_INTENT_REGISTRY_PATH", str(CONF_DIR / "realtime_intent_registry.json")))
LLM_LANGGRAPH_ENABLED = os.getenv("LLM_LANGGRAPH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class ChatConfig:
    api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    model: str = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro3")
    timeout: int = int(os.getenv("UPSTAGE_CHAT_TIMEOUT", "120"))
    temperature: float = float(os.getenv("UPSTAGE_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("UPSTAGE_MAX_TOKENS", "2048"))


@dataclass
class EmbedConfig:
    api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    model: str = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large-query")
    timeout: int = int(os.getenv("UPSTAGE_EMBED_TIMEOUT", "60"))


SYSTEM_SPECS = load_system_specs()
QUESTION_REPLACEMENTS = load_question_replacements()
TYPO_NORMALIZATION = load_typo_normalization()
INTENT_PATTERNS = load_intent_patterns()
SYSTEM_PROMPT_BY_INTENT = load_prompt_templates()
FEW_SHOT_EXAMPLES = load_few_shot_examples()
CONVERSATION_POLICY = load_conversation_policy()
CANONICAL_QUESTION_TEMPLATES = load_canonical_question_templates()
REALTIME_INTENT_REGISTRY = load_realtime_intent_registry()
SYSTEM_NAME_BY_ID: Dict[str, str] = {spec["system_id"]: spec["canonical_name"] for spec in SYSTEM_SPECS}
