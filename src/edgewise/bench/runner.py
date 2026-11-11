"""The bench runner: performance + quality for one model on this device (see docs/SPEC.md §6.5)."""

from __future__ import annotations

import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from edgewise import __version__
from edgewise.backends.base import Backend, LoadedModel
from edgewise.bench.metrics import Spread, collect, percentile
from edgewise.catalog import estimate_memory
from edgewise.evals.tasks import DEFAULT_DATA_DIR, get_task
from edgewise.hardware import detect
from edgewise.schema import GenerationResult, GenerationSettings, Message, ModelVariant, RunManifest
from edgewise.store import Store

Log = Callable[[str], None]

# Neutral filler used to build performance prompts of a given token length.
FILLER = (
    "Modern vehicles contain dozens of electronic control units that exchange messages over "
    "in-vehicle networks. Engineers test each software change against requirements, run "
    "simulations, and record measurements so that results can be traced and reproduced later. "
)
PERF_INSTRUCTION = "\n\nContinue writing about the topic above in plain prose."


class BenchError(RuntimeError):
    pass


class BenchConfig(BaseModel):
    tasks: list[str] = Field(default_factory=list)
    prompt_lengths: list[int] = Field(default_factory=lambda: [128, 1024, 4096])
    repeats: int = 5
    warmup: int = 3
    perf_max_tokens: int = 128
    cooldown_s: float = 10.0
    budget_gb: float | None = None  # None = this device's model budget
    force: bool = False  # run even if the memory estimate says it won't fit
    data_dir: Path = DEFAULT_DATA_DIR


class PerfSummary(BaseModel):
    prompt_len: int
    prompt_tokens: int
    ttft_ms: Spread
    prefill_tps: Spread
    decode_tps: Spread
    itl_p95_ms: Spread
    peak_memory_mb: float


class TaskSummary(BaseModel):
    task_id: str
    n: int
    mean_score: float
    valid_rate: float
    mean_latency_ms: float


class BenchReport(BaseModel):
    run_id: str
    model_id: str
    est_memory_gb: float
    load_s: float
    memory_after_load_mb: float
    perf: list[PerfSummary]
    quality: list[TaskSummary]
    warnings: list[str]


def run_bench(
    variant: ModelVariant,
    backend: Backend,
    config: BenchConfig,
    store: Store,
    log: Log = lambda msg: None,
) -> BenchReport:
    tasks = [get_task(t) for t in config.tasks]
    manifest = _manifest(backend)
    warnings: list[str] = []
    if manifest.on_ac_power is False:
        warnings.append("running on battery: results may be slower and noisier")

    # 1. Check memory before loading anything.
    context = max(config.prompt_lengths, default=0) + config.perf_max_tokens
    estimate = estimate_memory(variant, context_tokens=context)
    budget = config.budget_gb if config.budget_gb is not None else manifest.hardware.model_budget_gb
    fit = estimate.fit(budget)
    if fit == "too big" and not config.force:
        raise BenchError(
            f"{variant.id} needs ~{estimate.total_gb:.2f} GB at {context} tokens of context, "
            f"but the budget is {budget:.1f} GB (use --force to try anyway)"
        )
    if fit != "fits":
        warnings.append(f"memory is {fit}: ~{estimate.total_gb:.2f} GB of {budget:.1f} GB budget")
    for w in warnings:
        log(f"[yellow]warning:[/yellow] {w}")

    # 2. Load.
    log(f"loading {variant.id} …")
    backend.reset_peak_memory()
    t0 = time.perf_counter()
    model = backend.load(variant)
    load_s = time.perf_counter() - t0
    memory_after_load_mb = backend.peak_memory_mb()

    try:
        # 3. Warm-up.
        for _ in range(config.warmup):
            _generate(backend, model, [Message(role="user", content="Say hello.")], 16)

        # 4. Performance pass.
        perf_rows = []
        for prompt_len in config.prompt_lengths:
            messages = [Message(role="user", content=_prompt_of_length(backend, model, prompt_len))]
            for repeat in range(config.repeats):
                log(f"perf: {prompt_len} tokens, repeat {repeat + 1}/{config.repeats}")
                result = _generate(backend, model, messages, config.perf_max_tokens)
                perf_rows.append(_perf_row(manifest.run_id, variant.id, prompt_len, repeat, result))

        # 5. Quality pass.
        quality_rows = []
        for task in tasks:
            items = task.load_items(config.data_dir)
            for i, item in enumerate(items, 1):
                log(f"quality: {task.id} {i}/{len(items)}")
                start = backend.now_ns()
                result = _generate(backend, model, task.build(item), task.max_tokens)
                latency_ms = (backend.now_ns() - start) / 1e6
                score = task.score(result.text, item)
                quality_rows.append(
                    {
                        "run_id": manifest.run_id,
                        "model_id": variant.id,
                        "task_id": task.id,
                        "item_id": item.id,
                        "score": score.value,
                        "valid": score.valid,
                        "detail": score.detail,
                        "latency_ms": latency_ms,
                        "completion_tokens": result.completion_tokens,
                        "output_text": result.text,
                    }
                )
    finally:
        # 6. Unload and let the chip cool down before the next model.
        backend.unload(model)
        if config.cooldown_s > 0:
            log(f"cooling down for {config.cooldown_s:.0f}s")
            time.sleep(config.cooldown_s)

    # 7. Persist.
    store.add_run(
        manifest,
        model_id=variant.id,
        backend=backend.name,
        config=config.model_dump(mode="json"),
        est_memory_gb=estimate.total_gb,
        load_s=load_s,
        memory_after_load_mb=memory_after_load_mb,
    )
    store.add_perf(perf_rows)
    store.add_quality(quality_rows)

    return BenchReport(
        run_id=manifest.run_id,
        model_id=variant.id,
        est_memory_gb=estimate.total_gb,
        load_s=load_s,
        memory_after_load_mb=memory_after_load_mb,
        perf=_summarize_perf(perf_rows),
        quality=_summarize_quality(quality_rows),
        warnings=warnings,
    )


def _generate(
    backend: Backend, model: LoadedModel, messages: list[Message], max_tokens: int
) -> GenerationResult:
    backend.reset_peak_memory()
    start = backend.now_ns()
    stream = backend.generate(model, messages, GenerationSettings(max_tokens=max_tokens))
    result = collect(stream, start)
    result.peak_memory_mb = backend.peak_memory_mb()
    return result


def _prompt_of_length(backend: Backend, model: LoadedModel, n_tokens: int) -> str:
    """Filler text trimmed to roughly `n_tokens` (chat template tokens are added on top)."""
    instruction_tokens = len(backend.tokenize(model, PERF_INSTRUCTION))
    target = max(n_tokens - instruction_tokens, 1)
    tokens: list[int] = []
    text = FILLER
    while len(tokens) < target:
        tokens = backend.tokenize(model, text)
        text += FILLER
    return backend.detokenize(model, tokens[:target]) + PERF_INSTRUCTION


def _perf_row(run_id: str, model_id: str, prompt_len: int, repeat: int, r: GenerationResult):
    return {
        "run_id": run_id,
        "model_id": model_id,
        "prompt_len": prompt_len,
        "repeat": repeat,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "ttft_ms": r.ttft_ms,
        "prefill_tps": r.prefill_tps,
        "decode_tps": r.decode_tps,
        "itl_p50_ms": percentile(r.itl_ms, 50),
        "itl_p95_ms": percentile(r.itl_ms, 95),
        "peak_memory_mb": r.peak_memory_mb,
    }


def _summarize_perf(rows: list[dict]) -> list[PerfSummary]:
    summaries = []
    for prompt_len in dict.fromkeys(r["prompt_len"] for r in rows):
        group = [r for r in rows if r["prompt_len"] == prompt_len]
        summaries.append(
            PerfSummary(
                prompt_len=prompt_len,
                prompt_tokens=group[0]["prompt_tokens"],
                ttft_ms=Spread.of(r["ttft_ms"] for r in group),
                prefill_tps=Spread.of(r["prefill_tps"] for r in group),
                decode_tps=Spread.of(r["decode_tps"] for r in group),
                itl_p95_ms=Spread.of(r["itl_p95_ms"] for r in group),
                peak_memory_mb=max(r["peak_memory_mb"] for r in group),
            )
        )
    return summaries


def _summarize_quality(rows: list[dict]) -> list[TaskSummary]:
    summaries = []
    for task_id in dict.fromkeys(r["task_id"] for r in rows):
        group = [r for r in rows if r["task_id"] == task_id]
        summaries.append(
            TaskSummary(
                task_id=task_id,
                n=len(group),
                mean_score=statistics.fmean(r["score"] for r in group),
                valid_rate=statistics.fmean(float(r["valid"]) for r in group),
                mean_latency_ms=statistics.fmean(r["latency_ms"] for r in group),
            )
        )
    return summaries


def _manifest(backend: Backend) -> RunManifest:
    started = datetime.now()
    return RunManifest(
        run_id=f"{started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        started_at=started,
        git_sha=_git_sha(),
        edgewise_version=__version__,
        hardware=detect(),
        backend_versions=backend.versions(),
        on_ac_power=_on_ac_power(),
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() + ("-dirty" if dirty else "")


def _on_ac_power() -> bool | None:
    """macOS only; None when unknown."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return "AC Power" in out.stdout
