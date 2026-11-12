"""Results store: every benchmark run, measurement and score, in one DuckDB file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_DB = Path("results/edgewise.duckdb")

# `decoding` is "standard", "speculative" or "prompt_lookup"; `k_policy` and `draft_model_id`
# qualify the accelerated ones. Together with model_id they identify a configuration.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    git_sha VARCHAR,
    edgewise_version VARCHAR,
    model_id VARCHAR,
    backend VARCHAR,
    decoding VARCHAR,
    draft_model_id VARCHAR,
    k_policy VARCHAR,
    num_draft_tokens INTEGER,
    hardware JSON,
    backend_versions JSON,
    on_ac_power BOOLEAN,
    swap_used_gb DOUBLE,
    config JSON,
    est_memory_gb DOUBLE,
    load_s DOUBLE,
    memory_after_load_mb DOUBLE,
    notes VARCHAR
);
CREATE TABLE IF NOT EXISTS perf (
    run_id VARCHAR,
    model_id VARCHAR,
    prompt_len INTEGER,
    repeat INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms DOUBLE,
    prefill_tps DOUBLE,
    decode_tps DOUBLE,
    itl_p50_ms DOUBLE,
    itl_p95_ms DOUBLE,
    peak_memory_mb DOUBLE
);
CREATE TABLE IF NOT EXISTS quality (
    run_id VARCHAR,
    model_id VARCHAR,
    task_id VARCHAR,
    item_id VARCHAR,
    score DOUBLE,
    valid BOOLEAN,
    detail VARCHAR,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms DOUBLE,
    decode_ms DOUBLE,
    latency_ms DOUBLE,
    peak_memory_mb DOUBLE,
    mean_logprob DOUBLE,
    proposed_draft_tokens INTEGER,
    accepted_draft_tokens INTEGER,
    output_text VARCHAR
);
-- One row per routed request. Prompts and answers are deliberately not stored.
CREATE TABLE IF NOT EXISTS route_log (
    request_id VARCHAR,
    ts TIMESTAMP,
    policy VARCHAR,
    route VARCHAR,
    reason VARCHAR,
    escalated BOOLEAN,
    model_id VARCHAR,
    confidence DOUBLE,
    private BOOLEAN,
    pii VARCHAR,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd DOUBLE,
    latency_ms DOUBLE
);
"""


class Store:
    def __init__(self, path: Path = DEFAULT_DB, read_only: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            self.db.execute(SCHEMA)
            self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Upgrade a database written by an older version: add columns SCHEMA has gained."""
        columns = "SELECT table_name, column_name, data_type FROM information_schema.columns"
        with duckdb.connect(":memory:") as reference:
            reference.execute(SCHEMA)
            wanted = reference.execute(columns).fetchall()
        have = {(table, column) for table, column, _ in self.db.execute(columns).fetchall()}
        for table, column, data_type in wanted:
            if (table, column) not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {data_type}")

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_run(self, row: dict[str, Any]) -> None:
        self._insert("runs", [{k: _plain(v) for k, v in row.items()}])

    def add_perf(self, rows: list[dict[str, Any]]) -> None:
        self._insert("perf", rows)

    def add_quality(self, rows: list[dict[str, Any]]) -> None:
        self._insert("quality", rows)

    def add_route(self, row: dict[str, Any]) -> None:
        self._insert("route_log", [row])

    def _insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        columns = list(rows[0])
        placeholders = ", ".join("?" for _ in columns)
        self.db.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [[row[c] for c in columns] for row in rows],
        )

    def count(self, table: str, run_id: str) -> int:
        return self.db.execute(
            f"SELECT count(*) FROM {table} WHERE run_id = ?", [run_id]
        ).fetchone()[0]

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self.db.execute(sql, params or [])
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _plain(value: Any) -> Any:
    """Nested values are stored as JSON text."""
    return json.dumps(value, default=str) if isinstance(value, dict | list) else value
