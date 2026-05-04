from pathlib import Path
import json


def test_batch_spec_exists():
    spec_path = Path(__file__).resolve().parent / "batch_spec.json"
    assert spec_path.exists()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["batch_id"]
    assert spec["sql"]
