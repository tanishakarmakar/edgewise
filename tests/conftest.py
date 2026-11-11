"""Shared test helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from edgewise.store import Store

_clock = datetime(2026, 1, 1)


def record_run(
    store: Store,
    model_id: str,
    task_id: str = "tool_calling",
    scores: list[float] = (1.0,),
    tps: float = 50.0,
    peak_mb: float = 1000.0,
    decoding: str = "standard",
    draft_model_id: str | None = None,
    k_policy: str = "fixed",
    logprobs: list[float] | None = None,
    outputs: list[str] | None = None,
    on_ac_power: bool = True,
    tokens: int = 51,
    prompt_tokens: int = 200,
) -> str:
    """Write a synthetic benchmark run: one quality row per score, decoding at `tps` tokens/s."""
    global _clock
    _clock += timedelta(minutes=1)
    run_id = f"run-{_clock:%H%M}-{model_id}-{decoding}"
    store.add_run(
        {
            "run_id": run_id,
            "started_at": _clock,
            "git_sha": "test",
            "edgewise_version": "0",
            "model_id": model_id,
            "backend": "fake",
            "decoding": decoding,
            "draft_model_id": draft_model_id,
            "k_policy": k_policy,
            "num_draft_tokens": 4,
            "hardware": {},
            "backend_versions": {},
            "on_ac_power": on_ac_power,
            "config": {},
            "est_memory_gb": peak_mb / 1000,
            "load_s": 1.0,
            "memory_after_load_mb": peak_mb,
            "notes": "",
        }
    )
    decode_ms = (tokens - 1) / tps * 1000
    store.add_quality(
        [
            {
                "run_id": run_id,
                "model_id": model_id,
                "task_id": task_id,
                "item_id": f"item-{i}",
                "score": score,
                "valid": True,
                "detail": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "ttft_ms": 100.0,
                "decode_ms": decode_ms,
                "latency_ms": 100.0 + decode_ms,
                "peak_memory_mb": peak_mb,
                "mean_logprob": logprobs[i] if logprobs else -0.2,
                "proposed_draft_tokens": 0,
                "accepted_draft_tokens": 0,
                "output_text": outputs[i] if outputs else f"answer {i}",
            }
            for i, score in enumerate(scores)
        ]
    )
    return run_id


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "results.duckdb") as s:
        yield s
