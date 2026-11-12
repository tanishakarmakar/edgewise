"""Adapter that lets the speculative decoding engine drive an MLX model."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

# Long prompts are processed in chunks so prefill memory stays bounded (same size mlx-lm uses).
PREFILL_CHUNK = 2048
# For a handful of tokens one pass is cheaper than two; the wasted output-layer work is small.
SPLIT_PREFILL_ABOVE = 16


class MLXStepModel:
    def __init__(self, model):
        self.model = model
        self.cache = make_prompt_cache(model)

    def forward(self, tokens: list[int], last_only: bool = False) -> tuple[list[int], list[float]]:
        if last_only and len(tokens) > SPLIT_PREFILL_ABOVE:
            # Feed everything but the last token through for its cache entries only: the
            # output layer (hidden size x vocabulary, per position) is never evaluated for
            # them. Then run the last token alone. This is how mlx-lm prefills too.
            head = tokens[:-1]
            for start in range(0, len(head), PREFILL_CHUNK):
                self.model(mx.array(head[start : start + PREFILL_CHUNK])[None], cache=self.cache)
                mx.eval([c.state for c in self.cache])
                mx.clear_cache()
            rows = self.model(mx.array(tokens[-1:])[None], cache=self.cache)[0]
        else:
            rows = self.model(mx.array(tokens)[None], cache=self.cache)[0]
            if last_only:
                rows = rows[-1:]

        chosen = mx.argmax(rows, axis=-1)
        logprobs = mx.max(rows, axis=-1) - mx.logsumexp(rows, axis=-1)
        mx.eval(chosen, logprobs)
        return chosen.tolist(), logprobs.tolist()

    def trim(self, n: int) -> None:
        if n > 0:
            trim_prompt_cache(self.cache, n)
