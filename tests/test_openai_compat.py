"""OpenAI-compatible backend tests against an in-process fake server: no network, no key."""

import json

import httpx
import pytest

from edgewise.backends.fake import FakeBackend
from edgewise.backends.openai_compat import OpenAICompatBackend, RemoteUnavailable, remote_variant
from edgewise.bench.metrics import collect
from edgewise.catalog import Catalog
from edgewise.config import Config
from edgewise.router import Router, RouteRequest, Tier
from edgewise.router.build import build_router
from edgewise.schema import GenerationSettings, Message

MESSAGES = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="Capital of France?"),
]
SMALL = "qwen2.5-0.5b-instruct@mlx-4bit"


def sse(*chunks, done=True):
    lines = [": keep-alive", ""]
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines)


def delta(text, model="vendor/model-x"):
    return {"model": model, "choices": [{"index": 0, "delta": {"content": text}}]}


USAGE = {
    "model": "vendor/model-x",
    "choices": [],
    "usage": {"prompt_tokens": 21, "completion_tokens": 3},
}


def backend_answering(status=200, body="", seen=None, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, text=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatBackend(client=client, **kwargs)


def generate(backend, max_tokens=64):
    model = backend.load(remote_variant("vendor/model-x"))
    stream = backend.generate(model, MESSAGES, GenerationSettings(max_tokens=max_tokens))
    return stream, collect(stream, backend.now_ns())


def test_streams_text_and_reports_usage_and_serving_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = []
    backend = backend_answering(body=sse(delta("Par"), delta("is"), delta("."), USAGE), seen=seen)
    stream, result = generate(backend)
    assert result.text == "Paris."
    assert stream.stats == {"served_by": "vendor/model-x", "input_tokens": 21, "output_tokens": 3}

    request = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    sent = json.loads(request.content)
    assert sent["model"] == "vendor/model-x" and sent["stream"] is True
    assert sent["max_tokens"] == 64 and sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][0] == {"role": "system", "content": "Be brief."}


def test_custom_endpoint_and_key_variable_and_no_key_for_a_local_server(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    seen = []
    backend = backend_answering(
        body=sse(delta("ok")), seen=seen, base_url="http://127.0.0.1:8081/v1/", api_key_env="MY_KEY"
    )
    generate(backend)
    assert str(seen[0].url) == "http://127.0.0.1:8081/v1/chat/completions"
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    ("status", "has_key", "message"),
    [
        (401, False, "no API key: set OPENROUTER_API_KEY"),
        (401, True, "credentials rejected"),
        (402, True, "no credit"),
        (404, True, "model not found"),
        (429, True, "rate limited"),
        (503, True, "server error 503"),
    ],
)
def test_http_errors_become_remote_unavailable(monkeypatch, status, has_key, message):
    if has_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    else:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RemoteUnavailable, match=message):
        generate(backend_answering(status=status, body="{}"))


def test_error_reported_mid_stream_and_malformed_data(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    failing = sse(delta("Par"), {"error": {"message": "provider overloaded"}})
    with pytest.raises(RemoteUnavailable, match="provider overloaded"):
        generate(backend_answering(body=failing))
    with pytest.raises(RemoteUnavailable, match="malformed"):
        generate(backend_answering(body="data: {not json\n\n"))


def test_network_failure_becomes_remote_unavailable():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    backend = OpenAICompatBackend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(RemoteUnavailable, match="network error"):
        generate(backend)


def test_router_escalates_to_the_endpoint_and_tracks_cost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    local = FakeBackend(responder=lambda m: "not sure", logprob=-2.0)
    remote = backend_answering(body=sse(delta("Paris."), USAGE))
    router = Router(
        Tier(local, local.load(Catalog.load().get(SMALL))),
        Tier(
            remote, remote.load(remote_variant("vendor/model-x")), input_price=1.0, output_price=2.0
        ),
    )
    decision = router.route(RouteRequest(messages=MESSAGES))
    assert decision.route == "remote" and decision.escalated and decision.text == "Paris."
    assert decision.model_id == "vendor/model-x"
    assert decision.cost_usd == pytest.approx((21 * 1.0 + 3 * 2.0) / 1e6)


def test_router_keeps_the_local_answer_when_the_endpoint_fails(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    local = FakeBackend(responder=lambda m: "not sure", logprob=-2.0)
    remote = backend_answering(status=401, body="{}")
    router = Router(
        Tier(local, local.load(Catalog.load().get(SMALL))),
        Tier(remote, remote.load(remote_variant("vendor/model-x"))),
    )
    decision = router.route(RouteRequest(messages=MESSAGES))
    assert decision.route == "local" and decision.text == "not sure"
    assert "no API key: set OPENROUTER_API_KEY" in decision.reason


def test_build_router_with_openrouter_and_with_a_local_server():
    config = Config.model_validate(
        {
            "router": {"local_model": SMALL, "remote_model": "openrouter"},
            "remote": {"model": "vendor/model-x", "input_price_per_mtok": 0.5},
        }
    )
    router = build_router(
        config, Catalog.load(), budget_gb=4.5, backend_for=lambda n: FakeBackend()
    )
    assert router.remote.model_id == "remote:vendor/model-x"
    assert router.remote.needs_network and router.remote.input_price == 0.5

    config.remote.base_url = "http://localhost:8081/v1"
    config.router.remote_model = "openai-compatible"
    router = build_router(
        config, Catalog.load(), budget_gb=4.5, backend_for=lambda n: FakeBackend()
    )
    assert router.remote.needs_network is False and router.is_online()
