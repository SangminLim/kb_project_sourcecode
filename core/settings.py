from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", str(BASE_DIR / "conf")))
SQL_DIR = Path(os.getenv("REALTIME_SQL_DIR", str(BASE_DIR / "sql")))
