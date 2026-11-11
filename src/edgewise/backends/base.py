"""The interface every inference backend implements (see docs/SPEC.md §6.1)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent


@dataclass
class LoadedModel:
    variant: ModelVariant
    handle: Any  # backend-specific: (model, tokenizer) for MLX, a server URL for llama.cpp, ...
    meta: dict[str, Any] = field(default_factory=dict)  # backend-specific cached measurements


@dataclass
class TokenStream:
    """A generation in progress. `prompt_tokens` is known before the first token arrives.

    `stats` is filled in while `events` is consumed (e.g. proposed/accepted draft tokens).
    """

    prompt_tokens: int
    events: Iterator[TokenEvent]
    stats: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def load(self, variant: ModelVariant) -> LoadedModel: ...

    def unload(self, model: LoadedModel) -> None: ...

    def generate(
        self,
        model: LoadedModel,
        messages: list[Message],
        settings: GenerationSettings,
        draft: LoadedModel | None = None,
    ) -> TokenStream:
        """`draft` is the small model used when settings.decoding == "speculative"."""
        ...

    def tokenize(self, model: LoadedModel, text: str) -> list[int]: ...

    def detokenize(self, model: LoadedModel, tokens: list[int]) -> str: ...

    def now_ns(self) -> int:
        """The clock that TokenEvent.t_ns timestamps come from."""
        ...

    def reset_peak_memory(self) -> None: ...

    def peak_memory_mb(self) -> float: ...

    def versions(self) -> dict[str, str]: ...
