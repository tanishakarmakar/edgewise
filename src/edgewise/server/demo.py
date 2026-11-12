"""Data behind the demo page (`GET /`): what the page shows beyond the OpenAI-style API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edgewise import __version__
from edgewise.advisor import advise
from edgewise.advisor.recommend import Preference
from edgewise.catalog import Catalog
from edgewise.evals import TASKS
from edgewise.hardware import detect
from edgewise.report.svg import short_name
from edgewise.router import Router, Tier
from edgewise.schema import GenerationSettings, Message

Emit = Callable[[dict[str, Any]], None]

# The comparison runs the same prompt twice on the local model: once normally, once with this.
COMPARE_DECODING = "prompt_lookup"


def overview(router: Router, catalog: Catalog) -> dict[str, Any]:
    hardware = detect()

    def tier(t: Tier | None) -> dict[str, Any] | None:
        if t is None:
            return None
        return {
            "id": t.model_id,
            "label": short_name(t.model_id),
            "on_device": not t.needs_network,
        }

    benchmarked = 0
    if router.store is not None:
        benchmarked = router.store.query(
            "SELECT count(DISTINCT q.model_id) AS n FROM quality q JOIN runs r USING (run_id) "
            "WHERE r.decoding = 'standard'"
        )[0]["n"]
    return {
        "version": __version__,
        "device": {
            "chip": hardware.chip,
            "ram_gb": hardware.ram_gb,
            "budget_gb": hardware.model_budget_gb,
        },
        "router": {
            "local": tier(router.local),
            "remote": tier(router.remote),
            "policy": router.policy,
            "min_logprob": router.min_logprob,
        },
        "tasks": [{"id": t.id, "description": t.description} for t in TASKS.values()],
        "catalog_size": len(catalog),
        "benchmarked": benchmarked,
        "stats": {**router.stats.model_dump(), "local_share": router.stats.local_share},
    }


def advice(
    router: Router,
    catalog: Catalog,
    task: str,
    memory_gb: float,
    min_score: float,
    prefer: Preference,
) -> dict[str, Any]:
    """The Advisor's answer, plus display names for the chart."""
    if router.store is None:
        raise LookupError("this server was started without a results database")
    rec = advise(catalog, router.store, task, memory_gb, min_score=min_score, prefer=prefer)
    data = rec.model_dump()
    for row in data["rows"]:
        row["label"] = short_name(row["model_id"])
    for key in ("choice", "runner_up"):
        if data[key]:
            data[key]["label"] = short_name(data[key]["model_id"])
    return data


def run_comparison(tier: Tier, messages: list[Message], max_tokens: int, emit: Emit) -> None:
    """Generate the same answer with standard and accelerated decoding, reporting as it goes.

    One model can run at a time on a small device, so the two lanes run one after the other.
    """
    _warm_up(tier, messages)
    standard_text, standard_tps = _run_lane(tier, messages, max_tokens, "standard", emit)
    fast_text, fast_tps = _run_lane(tier, messages, max_tokens, COMPARE_DECODING, emit)
    emit(
        {
            "type": "result",
            "identical": standard_text == fast_text,
            "speedup": round(fast_tps / standard_tps, 2) if standard_tps else None,
        }
    )


def _warm_up(tier: Tier, messages: list[Message]) -> None:
    """Once per loaded model, pay the one-off costs before anything is timed: the first
    generation is slow, and accelerated decoding measures its pass-cost curve on first use.
    Without this the first comparison charges those costs to one lane or the other."""
    if tier.model.meta.get("compare_ready"):
        return
    for _ in tier.backend.generate(tier.model, messages, GenerationSettings(max_tokens=4)).events:
        pass
    calibrate = getattr(tier.backend, "pass_costs", None)
    if calibrate is not None:
        calibrate(tier.model)
    tier.model.meta["compare_ready"] = True


def _run_lane(
    tier: Tier, messages: list[Message], max_tokens: int, lane: str, emit: Emit
) -> tuple[str, float]:
    """Returns the generated text and its decode speed in tokens per second.

    Each "tokens" event is one forward pass: tokens that arrive together were verified
    together, so all but the last of them are guesses the model accepted.
    """
    settings = GenerationSettings(max_tokens=max_tokens, decoding=lane, k_policy="cost_aware")
    start = tier.backend.now_ns()
    stream = tier.backend.generate(tier.model, messages, settings)
    emit({"type": "start", "lane": lane, "prompt_tokens": stream.prompt_tokens})

    passes: list[tuple[int, list[str]]] = []  # (timestamp, tokens produced by that pass)
    for event in stream.events:
        if passes and passes[-1][0] == event.t_ns:
            passes[-1][1].append(event.text)
            continue
        if passes:
            _emit_pass(emit, lane, passes[-1], start)
        passes.append((event.t_ns, [event.text]))
    if passes:
        _emit_pass(emit, lane, passes[-1], start)

    count = sum(len(tokens) for _, tokens in passes)
    first_ns, last_ns = (passes[0][0], passes[-1][0]) if passes else (start, start)
    decode_s = (last_ns - first_ns) / 1e9
    # Tokens after the first pass, over the time they took: comparable across decodings.
    after_first = count - len(passes[0][1]) if passes else 0
    tps = after_first / decode_s if decode_s > 0 else 0.0
    emit(
        {
            "type": "done",
            "lane": lane,
            "tokens": count,
            "ttft_ms": round((first_ns - start) / 1e6, 1),
            "total_ms": round((last_ns - start) / 1e6, 1),
            "decode_tps": round(tps, 1),
            "proposed": stream.stats.get("proposed", 0),
            "accepted": stream.stats.get("accepted", 0),
        }
    )
    return "".join(text for _, tokens in passes for text in tokens), tps


def _emit_pass(emit: Emit, lane: str, one_pass: tuple[int, list[str]], start: int) -> None:
    t_ns, tokens = one_pass
    emit({"type": "tokens", "lane": lane, "tokens": tokens, "t_ms": round((t_ns - start) / 1e6, 2)})
