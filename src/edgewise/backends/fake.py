"""A deterministic backend with a virtual clock, so bench logic is testable without real models."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from edgewise.backends.base import LoadedModel, TokenStream
from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent

Responder = Callable[[list[Message]], str]


class FakeBackend:
    """Tokens are whitespace-separated words; time only advances when tokens are "computed".

    prefill_ns_per_token: virtual time to process each prompt token
    decode_ns_per_token: virtual time to generate each output token
    """

    name = "fake"

    def __init__(
        self,
        responder: Responder | None = None,
        prefill_ns_per_token: int = 100_000,
        decode_ns_per_token: int = 10_000_000,
        memory_mb: float = 500.0,
    ):
        self.responder = responder or (lambda messages: "ok")
        self.prefill_ns_per_token = prefill_ns_per_token
        self.decode_ns_per_token = decode_ns_per_token
        self.memory_mb = memory_mb
        self.clock_ns = 0
        self.loaded: list[str] = []
        self.generations = 0

    def load(self, variant: ModelVariant) -> LoadedModel:
        self.loaded.append(variant.id)
        return LoadedModel(variant=variant, handle=None)

    def unload(self, model: LoadedModel) -> None:
        self.loaded.remove(model.variant.id)

    def tokenize(self, model: LoadedModel, text: str) -> list[int]:
        return [len(word) for word in text.split()]

    def detokenize(self, model: LoadedModel, tokens: list[int]) -> str:
        return " ".join("x" * n for n in tokens)

    def now_ns(self) -> int:
        return self.clock_ns

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_mb(self) -> float:
        return self.memory_mb

    def versions(self) -> dict[str, str]:
        return {"fake": "0"}

    def generate(
        self,
        model: LoadedModel,
        messages: list[Message],
        settings: GenerationSettings,
        draft: LoadedModel | None = None,
    ) -> TokenStream:
        self.generations += 1
        self.last_settings = settings
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        words = self.responder(messages).split()[: settings.max_tokens]
        return TokenStream(prompt_tokens=prompt_tokens, events=self._events(prompt_tokens, words))

    def _events(self, prompt_tokens: int, words: list[str]) -> Iterator[TokenEvent]:
        self.clock_ns += prompt_tokens * self.prefill_ns_per_token
        for i, word in enumerate(words):
            self.clock_ns += self.decode_ns_per_token
            yield TokenEvent(
                token_id=i, text=word if i == 0 else " " + word, t_ns=self.clock_ns, logprob=-0.1
            )
