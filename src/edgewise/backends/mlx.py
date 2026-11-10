"""MLX backend: runs models in-process on Apple Silicon via mlx-lm."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from importlib.metadata import version

from edgewise.backends.base import LoadedModel, TokenStream
from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
except ImportError as e:  # pragma: no cover - depends on the optional extra
    raise ImportError("MLX backend needs the mlx extra: `uv sync --extra mlx`") from e


def _mx_memory(name: str):
    """mx.get_peak_memory() etc. moved out of mx.metal in newer MLX; support both."""
    return getattr(mx, name, None) or getattr(mx.metal, name)


class MLXBackend:
    name = "mlx"

    def load(self, variant: ModelVariant) -> LoadedModel:
        model, tokenizer = mlx_load(variant.source)
        return LoadedModel(variant=variant, handle=(model, tokenizer))

    def unload(self, model: LoadedModel) -> None:
        model.handle = None
        gc.collect()
        _mx_memory("clear_cache")()

    def tokenize(self, model: LoadedModel, text: str) -> list[int]:
        _, tokenizer = model.handle
        return tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, model: LoadedModel, tokens: list[int]) -> str:
        _, tokenizer = model.handle
        return tokenizer.decode(tokens)

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def reset_peak_memory(self) -> None:
        _mx_memory("reset_peak_memory")()

    def peak_memory_mb(self) -> float:
        return _mx_memory("get_peak_memory")() / 1e6

    def versions(self) -> dict[str, str]:
        return {"mlx": version("mlx"), "mlx-lm": version("mlx-lm")}

    def generate(
        self, model: LoadedModel, messages: list[Message], settings: GenerationSettings
    ) -> TokenStream:
        if settings.decoding != "standard":
            raise NotImplementedError("accelerated decoding arrives in milestone 3")
        mlx_model, tokenizer = model.handle
        # Render the chat template to text, then encode without adding special tokens again
        # (the template already contains BOS where the model expects it).
        text = tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], add_generation_prompt=True, tokenize=False
        )
        prompt = tokenizer.encode(text, add_special_tokens=False)
        mx.random.seed(settings.seed)
        stream = stream_generate(
            mlx_model,
            tokenizer,
            prompt,
            max_tokens=settings.max_tokens,
            sampler=make_sampler(temp=settings.temperature),
        )
        return TokenStream(prompt_tokens=len(prompt), events=self._events(stream))

    def _events(self, stream) -> Iterator[TokenEvent]:
        # Every response is one forward pass, including the final end-of-sequence token.
        for response in stream:
            yield TokenEvent(
                token_id=int(response.token),
                text=response.text,
                t_ns=time.perf_counter_ns(),
                logprob=response.logprobs[response.token].item(),
            )
