"""Results store: every benchmark run, measurement and score, in one DuckDB file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from edgewise.schema import RunManifest

DEFAULT_DB = Path("results/edgewise.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    git_sha VARCHAR,
    edgewise_version VARCHAR,
    model_id VARCHAR,
    backend VARCHAR,
    hardware JSON,
    backend_versions JSON,
    on_ac_power BOOLEAN,
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
    latency_ms DOUBLE,
    completion_tokens INTEGER,
    output_text VARCHAR
);
"""


class Store:
    def __init__(self, path: Path = DEFAULT_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = duckdb.connect(str(path))
        self.db.execute(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_run(
        self,
        manifest: RunManifest,
        model_id: str,
        backend: str,
        config: dict[str, Any],
        est_memory_gb: float,
        load_s: float,
        memory_after_load_mb: float,
    ) -> None:
        self.db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                manifest.run_id,
                manifest.started_at,
                manifest.git_sha,
                manifest.edgewise_version,
                model_id,
                backend,
                manifest.hardware.model_dump_json(),
                _json(manifest.backend_versions),
                manifest.on_ac_power,
                _json(config),
                est_memory_gb,
                load_s,
                memory_after_load_mb,
                manifest.notes,
            ],
        )

    def add_perf(self, rows: list[dict[str, Any]]) -> None:
        self._insert("perf", rows)

    def add_quality(self, rows: list[dict[str, Any]]) -> None:
        self._insert("quality", rows)

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


def _json(value: Any) -> str:
    return json.dumps(value, default=str)
