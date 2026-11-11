"""Cloud backend tests with a stand-in client: no network, no API key."""

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from edgewise.backends.cloud import (  # noqa: E402
    FALLBACK_BETA,
    MIN_MAX_TOKENS,
    CloudBackend,
    CloudUnavailable,
    cloud_variant,
)
from edgewise.bench.metrics import collect  # noqa: E402
from edgewise.router import Router, RouteRequest, Tier  # noqa: E402
from edgewise.schema import GenerationSettings, Message  # noqa: E402


class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn", model="claude-opus-5-5", error=None):
        self.chunks, self.stop_reason, self.model, self.error = chunks, stop_reason, model, error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        if self.error:
            raise self.error
        yield from self.chunks

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            model=self.model,
            usage=SimpleNamespace(input_tokens=120, output_tokens=45),
        )


class FakeClient:
    def __init__(self, stream):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self._result = stream

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return self._result


MESSAGES = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="Capital of France?"),
]


def run(stream, **settings):
    client = FakeClient(stream)
    backend = CloudBackend(client=client)
    model = backend.load(cloud_variant())
    token_stream = backend.generate(model, MESSAGES, GenerationSettings(**settings))
    return client, token_stream


def test_request_shape():
    client, stream = run(FakeStream(["Par", "is."]), max_tokens=64)
    result = collect(stream, start_ns=0)
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == "Be brief."
    assert request["messages"] == [{"role": "user", "content": "Capital of France?"}]
    assert request["max_tokens"] == MIN_MAX_TOKENS  # room for thinking
    assert request["output_config"] == {"effort": "medium"}
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert "thinking" not in request and "temperature" not in request
    assert result.text == "Paris."
    assert stream.stats["input_tokens"] == 120 and stream.stats["output_tokens"] == 45


def test_large_max_tokens_is_passed_through():
    client, stream = run(FakeStream(["ok"]), max_tokens=40_000)
    collect(stream, 0)
    assert client.requests[0]["max_tokens"] == 40_000


def test_refusal_is_reported_as_unavailable():
    _, stream = run(FakeStream([], stop_reason="refusal"))
    with pytest.raises(CloudUnavailable, match="declined"):
        collect(stream, 0)


def test_network_error_is_reported_as_unavailable():
    error = anthropic.APIConnectionError(request=SimpleNamespace())
    _, stream = run(FakeStream([], error=error))
    with pytest.raises(CloudUnavailable, match="network"):
        collect(stream, 0)


def test_served_by_reports_the_fallback_model():
    _, stream = run(FakeStream(["ok"], model="claude-opus-4-8"))
    collect(stream, 0)
    assert stream.stats["served_by"] == "claude-opus-4-8"


def test_router_prices_cloud_answers_from_reported_usage():
    from edgewise.backends.fake import FakeBackend
    from edgewise.catalog import Catalog

    local_backend = FakeBackend(logprob=-3.0)
    local = Tier(
        local_backend, local_backend.load(Catalog.load().get("qwen2.5-0.5b-instruct@mlx-4bit"))
    )
    cloud = CloudBackend(client=FakeClient(FakeStream(["Paris."])))
    remote = Tier(cloud, cloud.load(cloud_variant()), input_price=4.0, output_price=20.0,
                  needs_network=True)  # fmt: skip
    d = Router(local, remote).route(RouteRequest(messages=MESSAGES))
    assert d.route == "remote" and d.text == "Paris." and d.model_id == "claude-opus-5-5"
    assert (d.prompt_tokens, d.completion_tokens) == (120, 45)
    assert d.cost_usd == pytest.approx((120 * 4 + 45 * 20) / 1e6)


def test_router_falls_back_to_local_when_cloud_declines():
    from edgewise.backends.fake import FakeBackend
    from edgewise.catalog import Catalog

    local_backend = FakeBackend(responder=lambda m: "local answer", logprob=-3.0)
    local = Tier(
        local_backend, local_backend.load(Catalog.load().get("qwen2.5-0.5b-instruct@mlx-4bit"))
    )
    cloud = CloudBackend(client=FakeClient(FakeStream([], stop_reason="refusal")))
    d = Router(local, Tier(cloud, cloud.load(cloud_variant()))).route(
        RouteRequest(messages=MESSAGES)
    )
    assert d.route == "local" and d.text == "local answer" and "declined" in d.reason
