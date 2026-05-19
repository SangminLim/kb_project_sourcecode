from pathlib import Path


def test_generated_files_exist():
    base_dir = Path(__file__).resolve().parent
    assert (base_dir / "batch_spec.json").exists()
    assert (base_dir / "query.sql").exists()
    assert (base_dir / "job.py").exists()


def test_query_is_insert_sql():
    sql = (Path(__file__).resolve().parent / "query.sql").read_text(encoding="utf-8").upper()
    assert "INSERT INTO" in sql
    assert ":BASE_YM" in sql


def test_job_has_delete_sql_replaced():
    job = (Path(__file__).resolve().parent / "job.py").read_text(encoding="utf-8")

    assert "DELETE FROM TB_DEDUCTION_MONTHLY_SUMMARY WHERE BASE_YM = :base_ym" in job