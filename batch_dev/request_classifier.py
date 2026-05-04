from __future__ import annotations

def detect_structured_request_type(text: str) -> str | None:
    """배치 요청서 형태의 입력이면 batch_development intent로 분류한다."""
    q = (text or "").strip()
    if not q:
        return None
    signals = ["[배치 개발 요청서]", "배치명:", "대상 테이블:", "처리 내용:", "출력:"]
    if any(signal in q for signal in signals):
        return "batch_development"
    return None
