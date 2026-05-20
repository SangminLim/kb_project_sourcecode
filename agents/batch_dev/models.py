from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BatchDevResult:
    batch_spec: Dict[str, Any]
    created_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def success(self) -> bool:
        return not self.errors
