"""The Advisor: pick the best model variant for a task and memory budget (docs/SPEC.md §7).

Only measured variants are ranked. A variant with no benchmark for the task is listed as
"not benchmarked" together with the command that would benchmark it; it is never ranked on a
guess.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from edgewise.catalog import Catalog, estimate_memory
from edgewise.schema import ModelVariant
from edgewise.store import Store

Preference = Literal["balanced", "speed", "quality"]
# "balanced" treats scores this close to the best as equally good and then picks the fastest.
# A gap of a single eval item also counts as equal, however few items there are: with 16
# items one answer is worth 0.06, and that is noise, not a reason to run a 5x slower model.
BALANCED_TOLERANCE = 0.05
Status = Literal[
    "recommended", "runner-up", "candidate", "dominated", "below quality floor", "too big",
    "not benchmarked",
]  # fmt: skip

# Latest standard-decoding run per model that covers the task.
EVIDENCE_SQL = """
WITH latest AS (
    SELECT q.model_id, max(r.started_at) AS started_at
    FROM quality q JOIN runs r USING (run_id)
    WHERE q.task_id = ? AND r.decoding = 'standard'
    GROUP BY q.model_id
)
SELECT
    q.model_id,
    r.run_id,
    r.on_ac_power,
    r.swap_used_gb,
    count(*) AS n_items,
    avg(q.score) AS score,
    avg(CAST(q.valid AS DOUBLE)) AS valid_rate,
    sum(greatest(q.completion_tokens - 1, 0)) / nullif(sum(q.decode_ms) / 1000, 0) AS decode_tps,
    median(q.ttft_ms) AS ttft_ms,
    max(q.peak_memory_mb) AS peak_memory_mb
FROM quality q
JOIN runs r USING (run_id)
JOIN latest l ON l.model_id = q.model_id AND l.started_at = r.started_at
WHERE q.task_id = ? AND r.decoding = 'standard'
GROUP BY q.model_id, r.run_id, r.on_ac_power, r.swap_used_gb
"""


class Evidence(BaseModel):
    """What the results store knows about one model on one task."""

    model_id: str
    run_id: str
    on_ac_power: bool | None
    swap_used_gb: float | None = None
    n_items: int
    score: float
    valid_rate: float
    decode_tps: float
    ttft_ms: float
    peak_memory_mb: float


class Row(BaseModel):
    """One line of the Advisor's answer."""

    model_id: str
    status: Status
    memory_gb: float  # measured peak if benchmarked, otherwise the estimate
    memory_measured: bool
    evidence: Evidence | None = None
    vs_full_precision: float | None = (
        None  # score ÷ score of the same model at its highest precision
    )
    on_frontier: bool = False


class Recommendation(BaseModel):
    task_id: str
    budget_gb: float
    min_score: float
    prefer: Preference
    choice: Row | None
    runner_up: Row | None
    reasons: list[str]
    rows: list[Row]


def load_evidence(store: Store, task_id: str) -> dict[str, Evidence]:
    rows = store.query(EVIDENCE_SQL, [task_id, task_id])
    return {r["model_id"]: Evidence(**{**r, "decode_tps": r["decode_tps"] or 0.0}) for r in rows}


def pareto_frontier(evidence: list[Evidence]) -> set[str]:
    """Models that no other model beats on both score and speed."""

    def dominated(e: Evidence) -> bool:
        return any(
            o.score >= e.score
            and o.decode_tps >= e.decode_tps
            and (o.score > e.score or o.decode_tps > e.decode_tps)
            for o in evidence
        )

    return {e.model_id for e in evidence if not dominated(e)}


def advise(
    catalog: Catalog,
    store: Store,
    task_id: str,
    budget_gb: float,
    min_score: float = 0.0,
    prefer: Preference = "balanced",
    context_tokens: int = 4096,
) -> Recommendation:
    evidence = load_evidence(store, task_id)
    variants = list(catalog)

    # 1. Memory: the measured peak when we have one, the estimate otherwise.
    rows: dict[str, Row] = {}
    for v in variants:
        e = evidence.get(v.id)
        memory_gb = (
            e.peak_memory_mb / 1000
            if e
            else estimate_memory(v, context_tokens=context_tokens).total_gb
        )
        status: Status = "candidate"
        if memory_gb > budget_gb:
            status = "too big"
        elif e is None:
            status = "not benchmarked"
        elif e.score < min_score:
            status = "below quality floor"
        rows[v.id] = Row(
            model_id=v.id,
            status=status,
            memory_gb=memory_gb,
            memory_measured=e is not None,
            evidence=e,
            vs_full_precision=_vs_full_precision(v, variants, evidence),
        )

    # 2. Pareto frontier among the candidates that passed every filter.
    candidates = [r.evidence for r in rows.values() if r.status == "candidate"]
    frontier = pareto_frontier(candidates)
    for row in rows.values():
        if row.status == "candidate":
            row.on_frontier = row.model_id in frontier
            if not row.on_frontier:
                row.status = "dominated"

    # 3. Pick from the frontier.
    ranked = sorted(
        (rows[m] for m in frontier), key=lambda r: _preference_key(r.evidence, candidates, prefer)
    )
    ranked.reverse()
    choice = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    if choice:
        choice.status = "recommended"
    if runner_up:
        runner_up.status = "runner-up"

    order = ["recommended", "runner-up", "candidate", "dominated", "below quality floor",
             "too big", "not benchmarked"]  # fmt: skip
    return Recommendation(
        task_id=task_id,
        budget_gb=budget_gb,
        min_score=min_score,
        prefer=prefer,
        choice=choice,
        runner_up=runner_up,
        reasons=_reasons(choice, runner_up, rows, task_id, prefer),
        rows=sorted(
            rows.values(),
            key=lambda r: (order.index(r.status), -(r.evidence.score if r.evidence else 0)),
        ),
    )


def _preference_key(e: Evidence, candidates: list[Evidence], prefer: Preference):
    if prefer == "speed":
        return (e.decode_tps, e.score)
    if prefer == "quality":
        return (e.score, e.decode_tps)
    # Balanced: don't give up quality that matters, then take speed. Models within
    # BALANCED_TOLERANCE of the best score rank by speed; the rest rank below them by score.
    best = max(c.score for c in candidates)
    one_item = 1 / min(c.n_items for c in candidates if c.n_items) if candidates else 0
    if e.score >= best - max(BALANCED_TOLERANCE, one_item) - 1e-9:
        return (1, e.decode_tps, e.score)
    return (0, e.score, e.decode_tps)


def _vs_full_precision(
    variant: ModelVariant, variants: list[ModelVariant], evidence: dict[str, Evidence]
) -> float | None:
    """Score relative to the highest-precision benchmarked variant of the same model."""
    if variant.id not in evidence:
        return None
    siblings = [
        v
        for v in variants
        if v.family == variant.family and v.params_b == variant.params_b and v.id in evidence
    ]
    reference = max(siblings, key=lambda v: v.bits_per_weight)
    if reference.id == variant.id or evidence[reference.id].score == 0:
        return None
    return evidence[variant.id].score / evidence[reference.id].score


def _reasons(
    choice: Row | None,
    runner_up: Row | None,
    rows: dict[str, Row],
    task_id: str,
    prefer: Preference,
) -> list[str]:
    reasons: list[str] = []
    if choice is None:
        unbenchmarked = [r.model_id for r in rows.values() if r.status == "not benchmarked"]
        if unbenchmarked:
            reasons.append(
                f"No model that fits has been benchmarked on {task_id}. "
                f"Try: edgewise bench -m {unbenchmarked[0]} -t {task_id}"
            )
        elif any(r.status == "below quality floor" for r in rows.values()):
            reasons.append("Every benchmarked model that fits scores below the quality floor.")
        else:
            reasons.append("No catalog model fits in this memory budget.")
        return reasons

    e = choice.evidence
    reasons.append(
        f"{choice.model_id} scores {e.score:.2f} on {task_id} at {e.decode_tps:.0f} tokens/s, "
        f"using {choice.memory_gb:.1f} GB (measured peak)."
    )
    if choice.vs_full_precision is not None:
        reasons.append(
            f"It keeps {choice.vs_full_precision:.0%} of the score of the same model "
            "at higher precision."
        )
    if runner_up:
        r = runner_up.evidence
        speed, quality = e.decode_tps / r.decode_tps if r.decode_tps else 0, e.score - r.score
        faster = f"{speed:.1f}x faster" if speed >= 1 else f"{1 / speed:.1f}x slower"
        if abs(quality) < 0.005:
            better = f"scores the same ({e.score:.2f})"
        else:
            direction = "higher" if quality > 0 else "lower"
            better = f"scores {abs(quality):.2f} {direction} ({e.score:.2f} vs {r.score:.2f})"
        reasons.append(
            f"Compared with {runner_up.model_id}, it is {faster} and {better}; "
            f"preference: {prefer}."
        )
    if e.n_items < 30:
        reasons.append(
            f"Based on only {e.n_items} test items: a single item moves the score by "
            f"{1 / e.n_items:.2f}, so small differences are not meaningful."
        )
    if e.on_ac_power is False:
        reasons.append("Measured on battery power: speed may be understated.")
    if (e.swap_used_gb or 0) > 1.0:
        reasons.append(
            f"Measured while the system was using {e.swap_used_gb:.1f} GB of swap: "
            "speed is unreliable; re-run with other apps closed."
        )
    return reasons
