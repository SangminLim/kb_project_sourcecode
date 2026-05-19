from pathlib import Path


def test_generated_files_exist():
    base_dir = Path(__file__).resolve().parent
    assert (base_dir / "batch_spec.json").exists()
    assert (base_dir / "query.sql").exists()
    assert (base_dir / "job.py").exists()
