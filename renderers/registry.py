from __future__ import annotations

from typing import Any, Callable, Dict

Renderer = Callable[[Any], None]

_RENDERERS: Dict[str, Renderer] = {}

def register_renderer(name: str, renderer: Renderer) -> None:
    _RENDERERS[name] = renderer

def get_renderer(name: str) -> Renderer | None:
    return _RENDERERS.get(name)
