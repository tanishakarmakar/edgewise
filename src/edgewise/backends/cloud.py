"""Cloud backend: the Claude API, used as the Router's escalation model.

Never used for local benchmarks. Credentials come from the environment (ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile); nothing is read from config files.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from importlib.metadata import version

from edgewise.backends.base import LoadedModel, TokenStream
from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent

try:
    import anthropic
except ImportError as e:  # pragma: no cover - depends on the optional extra
    raise ImportError("Cloud backend needs the cloud extra: `uv sync --extra cloud`") from e

DEFAULT_MODEL = "claude-opus-5-5"
# Thinking is always on for this model and its tokens count against max_tokens, so a small
# local-style limit (e.g. 128) could be spent before any answer is written.
MIN_MAX_TOKENS = 16_000
# If a safety classifier declines a request, the API retries it on Anthropic's recommended
# fallback model within the same call instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class CloudUnavailable(RuntimeError):
    """The cloud model could not answer (network, credentials, rate limit, outage, refusal)."""


def cloud_variant(model: str = DEFAULT_MODEL) -> ModelVariant:
    """A catalog-style entry for a cloud model (no local weights, so sizes are zero)."""
    return ModelVariant(
        id=f"cloud:{model}",
        family="claude",
        params_b=0,
        quant="none",
        bits_per_weight=0,
        backend="cloud",
        source=model,
        tokenizer_family="claude",
        n_layers=0,
        n_kv_heads=0,
        head_dim=0,
        context_max=1_000_000,
    )


class CloudBackend:
    name = "cloud"

    def __init__(self, client: anthropic.Anthropic | None = None, effort: str = "medium"):
        self._client = client
        self.effort = effort

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def load(self, variant: ModelVariant) -> LoadedModel:
        return LoadedModel(variant=variant, handle=variant.source)

    def unload(self, model: LoadedModel) -> None:
        pass

    def tokenize(self, model: LoadedModel, text: str) -> list[int]:
        raise NotImplementedError("the cloud backend does not expose token ids")

    def detokenize(self, model: LoadedModel, tokens: list[int]) -> str:
        raise NotImplementedError("the cloud backend does not expose token ids")

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_mb(self) -> float:
        return 0.0

    def versions(self) -> dict[str, str]:
        return {"anthropic": version("anthropic")}

    def generate(
        self,
        model: LoadedModel,
        messages: list[Message],
        settings: GenerationSettings,
        draft: LoadedModel | None = None,
    ) -> TokenStream:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        request = {
            "model": model.handle,
            "max_tokens": max(settings.max_tokens, MIN_MAX_TOKENS),
            "messages": turns,
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        # Token counts are only known once the response finishes; they arrive in `stats`.
        stream = TokenStream(prompt_tokens=0, events=iter(()))
        stream.events = self._events(request, stream)
        return stream

    def _events(self, request: dict, stream: TokenStream) -> Iterator[TokenEvent]:
        """Yields text chunks (not single tokens) as they arrive."""
        try:
            with self.client.beta.messages.stream(**request) as response:
                for i, text in enumerate(response.text_stream):
                    yield TokenEvent(token_id=i, text=text, t_ns=time.perf_counter_ns())
                final = response.get_final_message()
        except anthropic.AuthenticationError as e:
            raise CloudUnavailable("no valid Claude API credentials") from e
        except anthropic.PermissionDeniedError as e:
            raise CloudUnavailable("credentials lack permission for this model") from e
        except anthropic.RateLimitError as e:
            raise CloudUnavailable("rate limited") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise CloudUnavailable(f"server error {e.status_code}") from e
            raise
        except anthropic.APIConnectionError as e:
            raise CloudUnavailable("network error") from e

        stream.stats.update(
            input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens
        )
        stream.stats["served_by"] = final.model  # differs from the request if a fallback ran
        if final.stop_reason == "refusal":
            raise CloudUnavailable("the request was declined by the model's safety system")
