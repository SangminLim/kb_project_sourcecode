from pathlib import Path


def test_generated_files_exist():
    base_dir = Path(__file__).resolve().parent

    assert (base_dir / "batch_spec.json").exists()
    assert (base_dir / "query.sql").exists()
    assert (base_dir / "job.py").exists()


def test_query_has_ledger_and_classification_join():
    sql = (Path(__file__).resolve().parent / "query.sql").read_text(encoding="utf-8").upper()

    assert "FROM" in sql
    assert "TB_CARD_SALES_LEDGER" in sql
    assert "LEFT JOIN" in sql
    assert "MERCHANT_TYPE" in sql
    assert "CANCEL_YN = 'N'" in sql
    assert ":BASE_YM" in sql


def test_job_uses_base_ym_argument():
    job = (Path(__file__).resolve().parent / "job.py").read_text(encoding="utf-8")

    assert "--base-ym" in job
    assert "base_ym" in job
