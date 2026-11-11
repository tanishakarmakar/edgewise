"""Config, router assembly, the local HTTP API, and the route command."""

import json

import pytest
from typer.testing import CliRunner

from edgewise.backends.fake import FakeBackend
from edgewise.catalog import Catalog
from edgewise.cli import app as cli
from edgewise.config import Config, load_config
from edgewise.router import Router, Tier
from edgewise.router.build import build_router

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from edgewise.server import create_app  # noqa: E402

SMALL, BIG = "qwen2.5-0.5b-instruct@mlx-4bit", "qwen2.5-3b-instruct@mlx-4bit"


def fake_tier(answer, logprob=-0.1, model_id=SMALL, **kwargs):
    backend = FakeBackend(responder=lambda messages: answer, logprob=logprob)
    return Tier(backend, backend.load(Catalog.load().get(model_id)), **kwargs)


# ------------------------------------------------------------------ config


def test_config_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config()
    assert config.router.policy == "cascade" and config.cloud.model == "claude-opus-5-5"


def test_config_file_overrides_defaults(tmp_path):
    path = tmp_path / "edgewise.toml"
    path.write_text(
        '[router]\nlocal_model = "x"\nremote_model = "none"\nmin_logprob = -0.5\n'
        "[cloud]\ninput_price_per_mtok = 1.5\n"
    )
    config = load_config(path)
    assert config.router.local_model == "x" and config.router.min_logprob == -0.5
    assert config.cloud.input_price_per_mtok == 1.5 and config.cloud.output_price_per_mtok == 20.0


def test_config_rejects_bad_values_and_missing_file(tmp_path):
    path = tmp_path / "edgewise.toml"
    path.write_text('[router]\npolicy = "always-cloud"\n')
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


# ------------------------------------------------------------------ router assembly


def config_with(**router):
    return Config.model_validate({"router": router})


def test_build_local_only_router():
    backends = []

    def backend_for(name):
        backends.append(FakeBackend())
        return backends[-1]

    router = build_router(
        config_with(local_model=SMALL, remote_model="none"), Catalog.load(), budget_gb=4.5,
        backend_for=backend_for,
    )  # fmt: skip
    assert router.remote is None and backends[0].loaded == [SMALL]


def test_build_with_larger_local_model_as_escalation():
    backend = FakeBackend()
    router = build_router(
        config_with(local_model=SMALL, remote_model=BIG, min_logprob=-0.3), Catalog.load(),
        budget_gb=4.5, backend_for=lambda name: backend,
    )  # fmt: skip
    assert router.remote.model_id == BIG and not router.remote.needs_network
    assert router.remote.input_price == 0 and router.min_logprob == -0.3
    assert backend.loaded == [SMALL, BIG]


def test_build_refuses_models_that_do_not_fit_before_loading():
    backend = FakeBackend()
    with pytest.raises(ValueError, match="memory budget"):
        build_router(
            config_with(local_model=SMALL, remote_model=BIG), Catalog.load(), budget_gb=1.0,
            backend_for=lambda name: backend,
        )  # fmt: skip
    assert backend.loaded == []


def test_build_with_cloud_escalation_uses_configured_prices():
    pytest.importorskip("anthropic")
    config = Config.model_validate(
        {"router": {"local_model": SMALL}, "cloud": {"input_price_per_mtok": 2.5}}
    )
    router = build_router(
        config, Catalog.load(), budget_gb=4.5, backend_for=lambda n: FakeBackend()
    )
    assert router.remote.model_id == "cloud:claude-opus-5-5"
    assert router.remote.needs_network and router.remote.input_price == 2.5


# ------------------------------------------------------------------ HTTP API


@pytest.fixture
def client():
    router = Router(
        fake_tier("local answer", logprob=-0.2),
        fake_tier("remote answer", model_id=BIG, input_price=4.0, output_price=20.0),
        min_logprob=-0.8,
    )
    return TestClient(create_app(router)), router


def post(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "Capital of France?"}])
    return client.post("/v1/chat/completions", json=body)


def test_chat_completion_shape(client):
    http, _ = client
    data = post(http).json()
    assert data["object"] == "chat.completion" and data["model"] == SMALL
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "local answer"}
    assert data["usage"]["total_tokens"] == 3 + 2
    assert data["edgewise"]["route"] == "local" and data["edgewise"]["cost_usd"] == 0


def test_low_confidence_request_is_escalated(client):
    http, router = client
    router.min_logprob = 0.0
    data = post(http).json()
    assert data["model"] == BIG and data["choices"][0]["message"]["content"] == "remote answer"
    assert data["edgewise"]["escalated"] is True and data["edgewise"]["cost_usd"] > 0


def test_private_flag_and_pii_stay_local(client):
    http, router = client
    router.min_logprob = 0.0
    assert post(http, edgewise={"private": True}).json()["edgewise"]["route"] == "local"
    pii = post(http, messages=[{"role": "user", "content": "my email is a@b.co"}]).json()
    assert pii["edgewise"]["route"] == "local" and pii["edgewise"]["pii"] == ["email address"]


def test_json_response_format_escalates_invalid_json(client):
    http, _ = client
    data = post(http, response_format={"type": "json_object"}).json()
    assert data["edgewise"]["route"] == "remote" and "JSON" in data["edgewise"]["reason"]


def test_content_parts_are_accepted(client):
    http, _ = client
    parts = [{"type": "text", "text": "Capital "}, {"type": "text", "text": "of France?"}]
    assert post(http, messages=[{"role": "user", "content": parts}]).status_code == 200


def test_streaming_sends_openai_style_events(client):
    http, _ = client
    lines = [x for x in post(http, stream=True).text.split("\n\n") if x]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(x.removeprefix("data: ")) for x in lines[:-1]]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "local answer"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["edgewise"]["route"] == "local"


def test_validation_and_support_endpoints(client):
    http, _ = client
    assert http.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert http.get("/healthz").json() == {"status": "ok"}
    assert [m["id"] for m in http.get("/v1/models").json()["data"]] == [SMALL, BIG]
    post(http)
    metrics = http.get("/metrics").json()
    assert metrics["requests"] == 1 and metrics["local_share"] == 1.0


def test_router_factory_runs_at_startup():
    built = []

    def factory():
        built.append(1)
        return Router(fake_tier("hello"))

    with TestClient(create_app(factory)) as http:
        assert post(http).json()["choices"][0]["message"]["content"] == "hello"
    assert built == [1]


# ------------------------------------------------------------------ CLI


def test_cli_route(tmp_path, monkeypatch):
    import edgewise.router.build as build

    seen = {}

    def fake_build(config, catalog, store=None, **kwargs):
        seen["config"] = config
        return Router(fake_tier("Paris"), store=store)

    monkeypatch.setattr(build, "build_router", fake_build)
    runner = CliRunner()
    args = ["route", "Capital of France?", "--db", str(tmp_path / "r.duckdb"), "--remote", "none"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0 and "Paris" in result.stdout and "on device" in result.stdout
    assert seen["config"].router.remote_model == "none"

    as_json = runner.invoke(cli, [*args, "--json", "--private"])
    assert json.loads(as_json.stdout)["route"] == "local"

    bad = runner.invoke(cli, [*args, "--policy", "sometimes"])
    assert bad.exit_code == 1
