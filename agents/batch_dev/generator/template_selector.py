from __future__ import annotations

from typing import Any, Dict


def select_template(spec: Dict[str, Any]) -> str:
    batch_type = spec.get("batch_type", "db_to_file")

    if batch_type == "aggregation_to_table":
        return "aggregation_to_table"

    if batch_type in {"db_to_file", "file_to_db", "db_to_db"}:
        return batch_type

    return "db_to_file"
