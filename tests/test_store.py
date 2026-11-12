import duckdb

from edgewise.store import Store


def test_old_database_gains_new_columns(tmp_path):
    # A database written before `swap_used_gb` and `mean_logprob` existed, with a row in it.
    path = tmp_path / "old.duckdb"
    with duckdb.connect(str(path)) as db:
        db.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, model_id VARCHAR)")
        db.execute("CREATE TABLE quality (run_id VARCHAR, score DOUBLE)")
        db.execute("INSERT INTO runs VALUES ('r1', 'm')")

    with Store(path) as store:
        store.add_run({"run_id": "r2", "model_id": "m", "swap_used_gb": 0.5})
        store.add_quality([{"run_id": "r2", "score": 1.0, "mean_logprob": -0.2}])
        rows = store.query("SELECT run_id, swap_used_gb FROM runs ORDER BY run_id")
    assert rows == [
        {"run_id": "r1", "swap_used_gb": None},
        {"run_id": "r2", "swap_used_gb": 0.5},
    ]


def test_reopening_a_current_database_changes_nothing(tmp_path):
    path = tmp_path / "new.duckdb"
    with Store(path) as store:
        store.add_run({"run_id": "r1", "model_id": "m"})
    with Store(path) as store:
        assert store.count("runs", "r1") == 1
