from __future__ import annotations

from typing import Any, Dict, Optional

from .config import REALTIME_INTENT_REGISTRY

def get_system_by_id(payload: Dict[str, Any], system_id: str) -> Optional[Dict[str, Any]]:
    for domain in payload.get("domains", []):
        for system in domain.get("systems", []):
            if system.get("system_id") == system_id:
                return system
    return None


def get_realtime_query(payload: Dict[str, Any], query_id: str) -> Optional[Dict[str, Any]]:
    for item in payload.get("realtime_queries", []):
        if item.get("query_id") == query_id:
            return item
    return None


def get_realtime_intent_spec(intent: str) -> Optional[Dict[str, Any]]:
    spec = REALTIME_INTENT_REGISTRY.get(intent)
    return dict(spec) if isinstance(spec, dict) else None
