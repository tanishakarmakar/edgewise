"""MLX backend: runs models in-process on Apple Silicon via mlx-lm."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from importlib.metadata import version

from edgewise.accelerator import engine
from edgewise.backends.base import LoadedModel, TokenStream
from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate
    from mlx_lm.generate import wired_limit
    from mlx_lm.sample_utils import make_sampler

    from edgewise.accelerator.mlx_model import MLXStepModel
except ImportError as e:  # pragma: no cover - depends on the optional extra
    raise ImportError("MLX backend needs the mlx extra: `uv sync --extra mlx`") from e

# Prompt lookup guesses are free, so it can afford to guess further ahead than a draft model.
LOOKUP_K_MAX = 16
DRAFT_K_MAX = 8
CALIBRATION_WIDTHS = (1, 2, 3, 5, 9, 17)
CALIBRATION_REPEATS = 3


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
        self,
        model: LoadedModel,
        messages: list[Message],
        settings: GenerationSettings,
        draft: LoadedModel | None = None,
    ) -> TokenStream:
        mlx_model, tokenizer = model.handle
        # Render the chat template to text, then encode without adding special tokens again
        # (the template already contains BOS where the model expects it).
        text = tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], add_generation_prompt=True, tokenize=False
        )
        prompt = tokenizer.encode(text, add_special_tokens=False)
        mx.random.seed(settings.seed)
        stream = TokenStream(prompt_tokens=len(prompt), events=iter(()))

        if settings.decoding == "standard":
            responses = stream_generate(
                mlx_model,
                tokenizer,
                prompt,
                max_tokens=settings.max_tokens,
                sampler=make_sampler(temp=settings.temperature),
            )
            stream.events = self._standard_events(responses)
            return stream

        if settings.temperature != 0:
            raise ValueError("accelerated decoding verifies greedily: temperature must be 0")
        if settings.decoding == "speculative":
            if draft is None:
                raise ValueError("speculative decoding needs a draft model")
            if draft.variant.tokenizer_family != model.variant.tokenizer_family:
                raise ValueError(
                    f"draft {draft.variant.id} and target {model.variant.id} "
                    "use different tokenizers"
                )
            proposer = engine.DraftProposer(MLXStepModel(draft.handle[0]), prompt)
            k_max = DRAFT_K_MAX
        else:
            proposer = engine.LookupProposer(prompt)
            k_max = LOOKUP_K_MAX
        if settings.k_policy == "cost_aware":
            guess_ms = self.pass_costs(draft)[1] if settings.decoding == "speculative" else 0.0
            controller = engine.CostAwareK(self.pass_costs(model), guess_ms, k_max=k_max)
        elif settings.k_policy == "heuristic":
            controller = engine.HeuristicK(start=settings.num_draft_tokens, k_max=k_max)
        else:
            controller = engine.FixedK(settings.num_draft_tokens)
        steps = engine.generate(
            MLXStepModel(mlx_model),
            proposer,
            prompt,
            settings.max_tokens,
            set(tokenizer.eos_token_ids),
            controller,
        )
        stream.events = self._accelerated_events(
            steps, mlx_model, tokenizer, settings.max_tokens, stream
        )
        return stream

    def pass_costs(self, model: LoadedModel) -> dict[int, float]:
        """Milliseconds for one forward pass over 1, 2, 3, 5, 9 and 17 new tokens.

        Measured once per loaded model (about a second) and cached. This is the cost curve
        CostAwareK uses to decide how many guesses are worth verifying.
        """
        if "pass_costs" not in model.meta:
            filler = self.tokenize(model, "The quick brown fox jumps over the lazy dog. " * 12)
            costs = {}
            with wired_limit(model.handle[0]):
                step_model = MLXStepModel(model.handle[0])
                step_model.forward(filler, last_only=True)
                for width in CALIBRATION_WIDTHS:
                    samples = []
                    for i in range(CALIBRATION_REPEATS + 1):
                        start = time.perf_counter()
                        step_model.forward(filler[:width])
                        if i > 0:  # the first pass at each width warms up
                            samples.append((time.perf_counter() - start) * 1000)
                        step_model.trim(width)
                    costs[width] = sorted(samples)[len(samples) // 2]
            del step_model
            _mx_memory("clear_cache")()
            model.meta["pass_costs"] = costs
        return model.meta["pass_costs"]

    def _standard_events(self, responses) -> Iterator[TokenEvent]:
        # Every response is one forward pass, including the final end-of-sequence token.
        for response in responses:
            yield TokenEvent(
                token_id=int(response.token),
                text=response.text,
                t_ns=time.perf_counter_ns(),
                logprob=response.logprobs[response.token].item(),
            )

    def _accelerated_events(
        self,
        steps: Iterator[engine.Step],
        mlx_model,
        tokenizer,
        max_tokens: int,
        stream: TokenStream,
    ) -> Iterator[TokenEvent]:
        eos = set(tokenizer.eos_token_ids)
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        stream.stats.update(proposed=0, accepted=0)
        count = 0
        # Pin model memory like mlx-lm's own generate loop does; without this macOS pages
        # weights out under memory pressure and single passes stall for seconds.
        with wired_limit(mlx_model):
            for step in steps:
                now = time.perf_counter_ns()  # tokens accepted in one pass arrive together
                stream.stats["proposed"] += step.proposed
                stream.stats["accepted"] += step.accepted
                for token, logprob in zip(step.tokens, step.logprobs, strict=True):
                    count += 1
                    if token not in eos:
                        detokenizer.add_token(token)
                    if token in eos or count == max_tokens:
                        detokenizer.finalize()
                    yield TokenEvent(
                        token_id=token, text=detokenizer.last_segment, t_ns=now, logprob=logprob
                    )
