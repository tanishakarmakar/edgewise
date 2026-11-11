"""Adapter that lets the speculative decoding engine drive an MLX model."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

# Long prompts are processed in chunks so prefill memory stays bounded (same size mlx-lm uses).
PREFILL_CHUNK = 2048


class MLXStepModel:
    def __init__(self, model):
        self.model = model
        self.cache = make_prompt_cache(model)

    def forward(self, tokens: list[int], last_only: bool = False) -> tuple[list[int], list[float]]:
        if last_only:
            for start in range(0, len(tokens), PREFILL_CHUNK):
                chunk = mx.array(tokens[start : start + PREFILL_CHUNK])[None]
                logits = self.model(chunk, cache=self.cache)
                if start + PREFILL_CHUNK < len(tokens):
                    mx.eval([c.state for c in self.cache])
                    mx.clear_cache()
            rows = logits[0, -1:]
        else:
            rows = self.model(mx.array(tokens)[None], cache=self.cache)[0]

        chosen = mx.argmax(rows, axis=-1)
        logprobs = mx.max(rows, axis=-1) - mx.logsumexp(rows, axis=-1)
        mx.eval(chosen, logprobs)
        return chosen.tolist(), logprobs.tolist()

    def trim(self, n: int) -> None:
        if n > 0:
            trim_prompt_cache(self.cache, n)
