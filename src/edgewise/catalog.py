"""Model catalog and memory model (see docs/SPEC.md §6.2–6.3)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from edgewise.schema import ModelVariant

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "catalog" / "models.toml"

# Working memory beyond weights and KV cache: a little at rest, more while a prompt is being
# processed, and flat once the prompt is longer than one processing chunk. The three numbers
# are fitted to the measured peaks of eight variants at three prompt lengths on an 8 GB
# Apple A18 Pro (docs/SPEC.md §17). They err on the high side for small models.
BASE_OVERHEAD_GB = 0.1
PREFILL_WORKING_GB = 0.6
PREFILL_SATURATION_TOKENS = 1024
# Variants that use more than this share of the budget are flagged "tight".
TIGHT_FRACTION = 0.9
GB = 1e9


class MemoryEstimate(BaseModel):
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    context_tokens: int

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.overhead_gb

    def fit(self, budget_gb: float) -> str:
        """Classify against a memory budget: "fits", "tight" or "too big"."""
        if self.total_gb > budget_gb:
            return "too big"
        if self.total_gb > TIGHT_FRACTION * budget_gb:
            return "tight"
        return "fits"


def estimate_memory(
    variant: ModelVariant, context_tokens: int = 4096, kv_bytes: int = 2
) -> MemoryEstimate:
    """Estimate peak memory for running `variant` with `context_tokens` of context.

    KV cache = 2 (keys and values) × layers × KV heads × head dim × tokens × bytes per value.
    Overhead = base + prefill working memory, which grows with the prompt up to one chunk.
    """
    weights = variant.params_b * 1e9 * variant.bits_per_weight / 8
    kv = 2 * variant.n_layers * variant.n_kv_heads * variant.head_dim * context_tokens * kv_bytes
    prefill_share = min(context_tokens, PREFILL_SATURATION_TOKENS) / PREFILL_SATURATION_TOKENS
    return MemoryEstimate(
        weights_gb=weights / GB,
        kv_cache_gb=kv / GB,
        overhead_gb=BASE_OVERHEAD_GB + PREFILL_WORKING_GB * prefill_share,
        context_tokens=context_tokens,
    )


class Catalog:
    def __init__(self, variants: list[ModelVariant]):
        ids = [v.id for v in variants]
        if duplicates := {i for i in ids if ids.count(i) > 1}:
            raise ValueError(f"duplicate model ids in catalog: {sorted(duplicates)}")
        self._by_id = {v.id: v for v in variants}

    @classmethod
    def load(cls, path: Path = DEFAULT_CATALOG) -> Catalog:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls([ModelVariant(**entry) for entry in data.get("model", [])])

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, model_id: str) -> ModelVariant:
        try:
            return self._by_id[model_id]
        except KeyError:
            known = ", ".join(sorted(self._by_id))
            raise KeyError(f"unknown model {model_id!r}. Known models: {known}") from None
