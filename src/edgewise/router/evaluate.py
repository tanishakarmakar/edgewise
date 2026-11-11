"""Evaluate the cascade offline, from benchmark results already in the store.

For every eval item we know how the local model did (score, confidence) and how the stronger
model did. That is enough to replay the cascade at any confidence threshold without running
either model again: an item whose local confidence falls below the threshold takes the strong
model's score and cost; the rest keep the local score for free.
"""

from __future__ import annotations

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


class CascadeEvaluation(BaseModel):
    local_model: str
    strong_model: str
    n_items: int
    tasks: list[str]
    local_only_score: float
    strong_only_score: float
    strong_only_cost_per_1k_usd: float
    points: list[CascadePoint]
    # Cheapest threshold that keeps at least `target` of the strong model's score.
    target: float
    recommended: CascadePoint | None


def _items(store: Store, model_id: str) -> dict[tuple[str, str], dict]:
    rows = store.query(ITEMS_SQL, [model_id, model_id])
    return {(r["task_id"], r["item_id"]): r for r in rows}


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

    def replay(threshold: float) -> CascadePoint:
        score = cost = latency = 0.0
        kept = 0
        for key in keys:
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
        n = len(keys)
        return CascadePoint(
            threshold=threshold,
            local_share=kept / n,
            score=score / n,
            cost_per_1k_usd=cost / n * 1000,
            mean_latency_ms=latency / n,
        )

    # Thresholds that matter: just above each observed confidence, plus "never escalate".
    confidences = sorted(
        {local[k]["mean_logprob"] for k in keys if local[k]["mean_logprob"] is not None}
    )
    thresholds = [float("-inf")] + [c + 1e-9 for c in confidences]
    points = [replay(t) for t in thresholds]

    strong_only = sum(strong[k]["score"] for k in keys) / len(keys)
    good_enough = [p for p in points if p.score >= target * strong_only]
    return CascadeEvaluation(
        local_model=local_model,
        strong_model=strong_model,
        n_items=len(keys),
        tasks=sorted({k[0] for k in keys}),
        local_only_score=sum(local[k]["score"] for k in keys) / len(keys),
        strong_only_score=strong_only,
        strong_only_cost_per_1k_usd=sum(strong_cost(k) for k in keys) / len(keys) * 1000,
        points=points,
        target=target,
        recommended=max(good_enough, key=lambda p: p.local_share) if good_enough else None,
    )
