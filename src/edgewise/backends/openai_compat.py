"""Remote backend for any server that speaks the OpenAI chat-completions format.

Used as the Router's escalation model: OpenRouter by default, and equally a self-hosted
`llama-server`, vLLM or any other compatible endpoint. Never used for local benchmarks.
The API key is read from an environment variable named in the config; it is never stored.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from importlib.metadata import version

import httpx

from edgewise.backends.base import LoadedModel, TokenStream
from edgewise.schema import GenerationSettings, Message, ModelVariant, TokenEvent

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
# Generous, because a remote model may think before it writes its first token.
TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


class RemoteUnavailable(RuntimeError):
    """The remote model could not answer (network, credentials, rate limit, outage)."""


def remote_variant(model: str) -> ModelVariant:
    """A catalog-style entry for a remote model (no local weights, so sizes are zero)."""
    return ModelVariant(
        id=f"remote:{model}",
        family=model.split("/")[0],
        params_b=0,
        quant="none",
        bits_per_weight=0,
        backend="openai",
        source=model,
        tokenizer_family="unknown",
        n_layers=0,
        n_kv_heads=0,
        head_dim=0,
        context_max=0,
    )


class OpenAICompatBackend:
    name = "openai"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_KEY_ENV,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=TIMEOUT)
        return self._client

    def load(self, variant: ModelVariant) -> LoadedModel:
        return LoadedModel(variant=variant, handle=variant.source)

    def unload(self, model: LoadedModel) -> None:
        pass

    def tokenize(self, model: LoadedModel, text: str) -> list[int]:
        raise NotImplementedError("a remote backend does not expose token ids")

    def detokenize(self, model: LoadedModel, tokens: list[int]) -> str:
        raise NotImplementedError("a remote backend does not expose token ids")

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_mb(self) -> float:
        return 0.0

    def versions(self) -> dict[str, str]:
        return {"httpx": version("httpx")}

    def generate(
        self,
        model: LoadedModel,
        messages: list[Message],
        settings: GenerationSettings,
        draft: LoadedModel | None = None,
    ) -> TokenStream:
        body = {
            "model": model.handle,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # token counts arrive in a last chunk
        }
        stream = TokenStream(prompt_tokens=0, events=iter(()))
        stream.events = self._events(body, stream)
        return stream

    def _events(self, body: dict, stream: TokenStream) -> Iterator[TokenEvent]:
        """Yields text chunks (not single tokens) as they arrive."""
        key = os.environ.get(self.api_key_env, "")
        # A local server needs no key; a hosted one will answer 401 and say so below.
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        count = 0
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=body, headers=headers
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise RemoteUnavailable(_explain(response, self.api_key_env, bool(key)))
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators and ": keep-alive" comments
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):  # some providers report failures mid-stream
                        raise RemoteUnavailable(str(chunk["error"].get("message", chunk["error"])))
                    if chunk.get("model"):
                        stream.stats["served_by"] = chunk["model"]
                    if usage := chunk.get("usage"):
                        stream.stats["input_tokens"] = usage.get("prompt_tokens", 0)
                        stream.stats["output_tokens"] = usage.get("completion_tokens", 0)
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield TokenEvent(token_id=count, text=text, t_ns=time.perf_counter_ns())
                            count += 1
        except httpx.TimeoutException as e:
            raise RemoteUnavailable("the remote model timed out") from e
        except httpx.HTTPError as e:
            raise RemoteUnavailable("network error") from e
        except json.JSONDecodeError as e:
            raise RemoteUnavailable("the remote server sent a malformed response") from e


def _explain(response: httpx.Response, key_env: str, has_key: bool) -> str:
    status = response.status_code
    if status in (401, 403):
        return f"credentials rejected ({status})" if has_key else f"no API key: set {key_env}"
    if status == 402:
        return "the account has no credit for this model"
    if status == 404:
        return "model not found on the remote server"
    if status == 429:
        return "rate limited"
    if status >= 500:
        return f"server error {status}"
    try:
        detail = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        detail = ""
    return f"request rejected ({status})" + (f": {detail}" if detail else "")
