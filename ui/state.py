from __future__ import annotations

from .context import *

def init_session_state() -> None:
    if "message_list" not in st.session_state:
        st.session_state.message_list = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "pending_sql_analysis" not in st.session_state:
        st.session_state.pending_sql_analysis = None
    if "evaluation_mode" not in st.session_state:
        st.session_state.evaluation_mode = True

def build_chat_history(message_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in message_list:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history

def get_recent_questions(message_list: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    questions: List[str] = []
    seen = set()
    for item in reversed(message_list):
        if item.get("role") != "user":
            continue
        content = (item.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        questions.append(content)
        if len(questions) >= limit:
            break
    return questions

def shorten_text(text: str, max_len: int = 40) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."
