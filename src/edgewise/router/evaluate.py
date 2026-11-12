"""Evaluate the cascade offline, from benchmark results already in the store.

For every eval item we know how the local model did (score, confidence) and how the stronger
model did. That is enough to replay the cascade at any confidence threshold without running
either model again: an item whose local confidence falls below the threshold takes the strong
model's score and cost; the rest keep the local score for free.

A threshold picked on the same items it is scored on looks better than it is. So when there
are enough items, the threshold is chosen on one half and reported on the other.
"""

from __future__ import annotations

import zlib

from pydantic import BaseModel

from edgewise.store import Store

# Latest standard-decoding run per (model, task).
ITEMS_SQL = """
WITH latest AS (
    SELECT q.task_id, max(r.started_at) AS started_at
    FROM quality q JOIN runs r USING (run_id)
    WHERE q.model_id = ? AND r.decoding = 'standard'
    GROUP BY q.task_id
)
SELECT q.task_id, q.item_id, q.score, q.valid, q.mean_logprob,
       q.prompt_tokens, q.completion_tokens, q.latency_ms
FROM quality q
JOIN runs r USING (run_id)
JOIN latest l ON l.task_id = q.task_id AND l.started_at = r.started_at
WHERE q.model_id = ? AND r.decoding = 'standard'
"""


class CascadePoint(BaseModel):
    threshold: float  # escalate when local mean log-prob is below this
    local_share: float  # fraction of requests answered on the device
    score: float  # mean score of the cascade
    cost_per_1k_usd: float  # escalation cost per 1,000 requests
    mean_latency_ms: float  # escalated requests wait for both models


class HeldOut(BaseModel):
    """The recommended threshold applied to items that were not used to choose it."""

    n_items: int
    point: CascadePoint
    local_only_score: float
    strong_only_score: float


class CascadeEvaluation(BaseModel):
    local_model: str
    strong_model: str
    n_items: int
    tasks: list[str]
    local_only_score: float
    strong_only_score: float
    strong_only_cost_per_1k_usd: float
    points: list[CascadePoint]  # the whole curve, over all items
    # Cheapest threshold that keeps at least `target` of the strong model's score, chosen on
    # the fitting half of the items (on all of them when there are too few to split).
    target: float
    recommended: CascadePoint | None
    # The same threshold applied to every item: the point on `points` it corresponds to.
    recommended_overall: CascadePoint | None = None
    # How that threshold does on the other half. None when there were too few items to split,
    # in which case `recommended` is an optimistic estimate.
    heldout: HeldOut | None = None


# Each half needs at least this many items for a held-out estimate to mean anything.
MIN_HALF = 10


def _items(store: Store, model_id: str) -> dict[tuple[str, str], dict]:
    rows = store.query(ITEMS_SQL, [model_id, model_id])
    return {(r["task_id"], r["item_id"]): r for r in rows}


def _in_fit_half(key: tuple[str, str]) -> bool:
    """Stable split that does not depend on run order or Python's hash seed."""
    return zlib.crc32("/".join(key).encode()) % 2 == 0


def evaluate_cascade(
    store: Store,
    local_model: str,
    strong_model: str,
    input_price: float = 0.0,
    output_price: float = 0.0,
    target: float = 0.95,
    tasks: list[str] | None = None,
) -> CascadeEvaluation:
    """Prices are USD per million tokens of the strong model (0 if it also runs on-device)."""
    local, strong = _items(store, local_model), _items(store, strong_model)
    keys = sorted(k for k in local.keys() & strong.keys() if tasks is None or k[0] in tasks)
    if not keys:
        raise ValueError(
            f"no eval items benchmarked for both {local_model} and {strong_model}: "
            "run `edgewise bench` on each with the same tasks first"
        )

    def strong_cost(key) -> float:
        s = strong[key]
        return (s["prompt_tokens"] * input_price + s["completion_tokens"] * output_price) / 1e6

    def mean(values) -> float:
        values = list(values)
        return sum(values) / len(values)

    def replay(threshold: float, subset: list) -> CascadePoint:
        score = cost = latency = 0.0
        kept = 0
        for key in subset:
            item = local[key]
            confidence = item["mean_logprob"]
            escalate = not item["valid"] or (confidence is not None and confidence < threshold)
            if escalate:
                score += strong[key]["score"]
                cost += strong_cost(key)
                latency += item["latency_ms"] + strong[key]["latency_ms"]
            else:
                score += item["score"]
                latency += item["latency_ms"]
                kept += 1
        n = len(subset)
        return CascadePoint(
            threshold=threshold,
            local_share=kept / n,
            score=score / n,
            cost_per_1k_usd=cost / n * 1000,
            mean_latency_ms=latency / n,
        )

    def curve(subset: list) -> list[CascadePoint]:
        # Thresholds that matter: just above each observed confidence, plus "never escalate".
        confidences = sorted(
            {local[k]["mean_logprob"] for k in subset if local[k]["mean_logprob"] is not None}
        )
        return [replay(t, subset) for t in [float("-inf")] + [c + 1e-9 for c in confidences]]

    def choose(subset: list) -> CascadePoint | None:
        floor = target * mean(strong[k]["score"] for k in subset)
        good_enough = [p for p in curve(subset) if p.score >= floor]
        return max(good_enough, key=lambda p: p.local_share) if good_enough else None

    fit = [k for k in keys if _in_fit_half(k)]
    held = [k for k in keys if not _in_fit_half(k)]
    can_split = min(len(fit), len(held)) >= MIN_HALF
    recommended = choose(fit if can_split else keys)
    heldout = None
    if can_split and recommended is not None:
        heldout = HeldOut(
            n_items=len(held),
            point=replay(recommended.threshold, held),
            local_only_score=mean(local[k]["score"] for k in held),
            strong_only_score=mean(strong[k]["score"] for k in held),
        )

    return CascadeEvaluation(
        local_model=local_model,
        strong_model=strong_model,
        n_items=len(keys),
        tasks=sorted({k[0] for k in keys}),
        local_only_score=mean(local[k]["score"] for k in keys),
        strong_only_score=mean(strong[k]["score"] for k in keys),
        strong_only_cost_per_1k_usd=mean(strong_cost(k) for k in keys) * 1000,
        points=curve(keys),
        target=target,
        recommended=recommended,
        recommended_overall=replay(recommended.threshold, keys) if recommended else None,
        heldout=heldout,
    )
